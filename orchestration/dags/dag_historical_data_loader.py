import time
from datetime import datetime, timedelta
from io import BytesIO
import os
import pandas as pd
from airflow import DAG
from airflow.hooks.S3_hook import S3Hook
from airflow.hooks.postgres_hook import PostgresHook
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago
from binance.client import Client
from psycopg2.extras import execute_values

bucket_name = os.getenv('MINIO_BUCKET_NAME', 'binance')


def fetch_and_stream_to_minio(execution_date, **kwargs):
    client = Client()

    if isinstance(execution_date, str):
        execution_date = datetime.fromisoformat(execution_date)

    earliest_dt = datetime(2017, 8, 17)  # Binance 1m data starts here
    now_dt = datetime.now()

    # Define chunk size to avoid rate limits (e.g., 1 day per request)
    chunk_size = timedelta(days=1)
    current_dt = earliest_dt

    s3_hook = S3Hook(aws_conn_id='minio_conn')

    while current_dt < now_dt:
        current_dt_str = current_dt.strftime('%Y%m%d')
        object_key = f"klines/klines_{current_dt_str}.csv"
        processed_key = f"klines/klines_{current_dt_str}_PROCESSED.csv"

        if (s3_hook.check_for_key(key=object_key, bucket_name=bucket_name)
                or s3_hook.check_for_key(key=processed_key, bucket_name=bucket_name)):
            print(f"File already exists for {current_dt_str}. Skipping.")
            current_dt += chunk_size
            continue

        start_ts = int(current_dt.timestamp() * 1000)
        end_ts = int((current_dt + chunk_size).timestamp() * 1000)

        try:
            klines = client.get_klines(
                symbol='BTCUSDT',
                interval='1m',
                startTime=start_ts,
                endTime=end_ts
            )

            if not klines:
                print(f"No data returned for {current_dt_str}")
                return

            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'num_trades',
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            df['symbol'] = 'BTCUSDT'
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

            save_pd_to_minio(s3_hook, df, object_key)

            print(f"Saved data for {current_dt_str}")

        except Exception as e:
            print(
                f"Error fetching data for {current_dt.strftime('%Y-%m-%d')}: {e}")
            time.sleep(60)  # Backoff to avoid rate limit

        current_dt += chunk_size
        time.sleep(0.5)  # Light delay between requests to avoid throttling
    kwargs['ti'].xcom_push(key='s3_key', value=object_key)


def save_pd_to_minio(s3_hook, df: pd.DataFrame, object_key: str):
    with BytesIO() as buffer:
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        s3_hook.load_bytes(
            bytes_data=buffer.read(),
            key=object_key,
            bucket_name=bucket_name,  # Make sure this is defined or passed in
            replace=True
        )

def upsert_csv_from_minio():
    s3_hook = S3Hook(aws_conn_id='minio_conn')

    # List all objects in the bucket
    keys = s3_hook.list_keys(bucket_name=bucket_name)
    if not keys:
        raise ValueError(f"No objects found in bucket '{bucket_name}'")

    # Filter out already processed files
    unprocessed_keys = [key for key in keys if not key.endswith('_PROCESSED.csv')]

    # Sort keys by timestamp embedded in filename
    def extract_timestamp(key):
        try:
            date_str = key.split('_')[-1].replace('.csv', '')
            return pd.to_datetime(date_str)
        except Exception:
            return pd.Timestamp.max

    sorted_keys = sorted(unprocessed_keys, key=extract_timestamp)

    for object_key in sorted_keys:
        print(f"Processing file: {object_key}")
        obj = s3_hook.get_key(key=object_key, bucket_name=bucket_name)

        with BytesIO(obj.get()['Body'].read()) as buffer:
            df = pd.read_csv(buffer)
            df['open_time'] = pd.to_datetime(df['open_time'])

            rows = [
                (
                    row['open_time'], row['symbol'], row['open'], row['high'],
                    row['low'], row['close'], row['volume'],
                    row['taker_buy_base_volume'], row['taker_buy_quote_volume']
                )
                for _, row in df.iterrows()
            ]

            upsert_to_postgres(rows=rows)

        # Rename the file to mark it as processed
        processed_key = object_key.replace('.csv', '_PROCESSED.csv')
        s3_hook.copy_object(
            source_bucket_key=object_key,
            dest_bucket_key=processed_key,
            source_bucket_name=bucket_name,
            dest_bucket_name=bucket_name
        )
        s3_hook.delete_objects(bucket_name=bucket_name, keys=[object_key])
        print(f"Marked as processed: {processed_key}")


def upsert_to_postgres(rows):
    hook = PostgresHook(postgres_conn_id="postgres_default")
    with hook.get_conn() as conn:
        with conn.cursor() as cursor:
            execute_values(cursor, """
                INSERT INTO binance_klines (open_time, symbol, open, high, low, close, volume,
                                            taker_buy_base_volume, taker_buy_quote_volume)
                VALUES %s ON CONFLICT (open_time, symbol) DO
                UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    taker_buy_base_volume = EXCLUDED.taker_buy_base_volume,
                    taker_buy_quote_volume = EXCLUDED.taker_buy_quote_volume;
                """, rows)
            conn.commit()


# DAG config
default_args = {
    'owner': 'nghia',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='dag_binance_1m_to_postgres',
    default_args=default_args,
    schedule_interval='@hourly',
    start_date=days_ago(1),
    catchup=True,
    max_active_runs=1,
    tags=['binance', 'postgres', 'minio'],

) as dag:
    fetch_task = PythonOperator(
        task_id='fetch_and_stream_to_minio',
        python_callable=fetch_and_stream_to_minio,
        op_kwargs={'execution_date': '{{ execution_date }}'},
    )

    upsert_task = PythonOperator(
        task_id='upsert_from_minio_task',
        python_callable=upsert_csv_from_minio,
    )

    fetch_task >> upsert_task
