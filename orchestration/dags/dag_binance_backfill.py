from datetime import datetime, timedelta
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from binance_plugin import fetch_binance_klines, save_df_to_minio, BUCKET_NAME, upsert_to_postgres

def dag_binance_backfill_task(logical_date, **kwargs):
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    chunk_size = timedelta(days=1)

    current_dt = pd.to_datetime(logical_date)
    next_dt = current_dt + chunk_size
    object_key = f"klines/klines_{current_dt.strftime('%Y%m%d')}.csv"

    if s3_hook.check_for_key(key=object_key, bucket_name=BUCKET_NAME):
        print(f"{object_key} already exists. Skipping.")
        return

    klines = fetch_binance_klines(current_dt, next_dt)
    if not klines:
        print(f"No data for {current_dt}")
        return

    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['symbol'] = 'BTCUSDT'
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

    save_df_to_minio(df, object_key, s3_hook)
    print(f"Saved {object_key}")

    rows = [
        (
            row['open_time'], row['symbol'], row['open'], row['high'],
            row['low'], row['close'], row['volume'],
            row['taker_buy_base_volume'], row['taker_buy_quote_volume']
        )
        for _, row in df.iterrows()
    ]

    upsert_to_postgres(rows)
    print(f"Saved {object_key} and inserted into Postgres")

default_args = {
    'owner': 'airflow',
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='dag_binance_backfill',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=datetime(2017, 8, 17),
    end_date=datetime.now() - timedelta(days=1),
    catchup=True,
    max_active_runs=1,
    tags=['binance', 'backfill'],
) as dag:
    backfill = PythonOperator(
        task_id='dag_binance_backfill_task',
        python_callable=dag_binance_backfill_task,
    )