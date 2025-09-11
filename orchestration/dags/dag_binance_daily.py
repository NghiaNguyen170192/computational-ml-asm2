from datetime import datetime, timedelta
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from binance_plugin import fetch_binance_klines, save_df_to_minio, upsert_to_postgres, BUCKET_NAME
import io

def dag_binance_daily_task(**kwargs):
    s3_hook = S3Hook(aws_conn_id='minio_conn')

    # Define the current day range
    now = datetime.now().replace(second=0, microsecond=0)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)

    # Daily file name (same pattern as backfill)
    object_key = f"klines/klines_{start_of_day.strftime('%Y%m%d')}.csv"

    # Fetch only the latest chunk (e.g., last minute) but append to today's file
    one_min_ago = now - timedelta(minutes=1)
    klines = fetch_binance_klines(one_min_ago, now)
    if not klines:
        print("No data returned.")
        return

    df_new = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df_new['symbol'] = 'BTCUSDT'
    df_new['open_time'] = pd.to_datetime(df_new['open_time'], unit='ms')

    if s3_hook.check_for_key(key=object_key, bucket_name=BUCKET_NAME):
        print(f"{object_key} exists — appending new data.")
        existing_obj = s3_hook.read_key(
            key=object_key, bucket_name=BUCKET_NAME)
        df_existing = pd.read_csv(io.StringIO(existing_obj))
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset=['open_time'], inplace=True)
    else:
        print(f"{object_key} does not exist — creating new file.")
        df_combined = df_new

    save_df_to_minio(df_combined, object_key, s3_hook)
    print(f"Saved/updated {object_key}")

    rows = [
        (
            row['open_time'], row['symbol'], row['open'], row['high'],
            row['low'], row['close'], row['volume'],
            row['taker_buy_base_volume'], row['taker_buy_quote_volume']
        )
        for _, row in df_new.iterrows()
    ]
    upsert_to_postgres(rows)
    print(f"Inserted {len(rows)} new rows into Postgres")


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(seconds=30),
}

with DAG(
    dag_id='dag_binance_daily',
    default_args=default_args,
    schedule_interval='* * * * *',  # every minute
    start_date=datetime.now() - timedelta(minutes=1),
    catchup=False,
    max_active_runs=1,
    tags=['binance', 'realtime'],
) as dag:
    realtime = PythonOperator(
        task_id='dag_binance_daily_task',
        python_callable=dag_binance_daily_task,
    )
