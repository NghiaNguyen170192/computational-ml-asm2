# orchestration/plugins/binance_plugin.py
import os
import pandas as pd
from io import BytesIO
from datetime import datetime
from binance.client import Client
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.hooks.postgres_hook import PostgresHook
from psycopg2.extras import execute_values

BUCKET_NAME = os.getenv('MINIO_BUCKET_NAME', 'binance')

def fetch_binance_klines(start_dt: datetime, end_dt: datetime, symbol='BTCUSDT', interval='1m'):
    client = Client()
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    return client.get_klines(symbol=symbol, interval=interval, startTime=start_ts, endTime=end_ts)

def save_df_to_minio(df: pd.DataFrame, object_key: str, s3_hook: S3Hook):
    with BytesIO() as buffer:
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        s3_hook.load_bytes(
            bytes_data=buffer.read(),
            key=object_key,
            bucket_name=BUCKET_NAME,
            replace=True
        )

def upsert_to_postgres(rows):
    hook = PostgresHook(postgres_conn_id="postgres_default")
    with hook.get_conn() as conn:
        with conn.cursor() as cursor:
            execute_values(cursor, """
                INSERT INTO binance_klines (open_time, symbol, open, high, low, close, volume,
                                            taker_buy_base_volume, taker_buy_quote_volume)
                VALUES %s ON CONFLICT (open_time, symbol) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    taker_buy_base_volume = EXCLUDED.taker_buy_base_volume,
                    taker_buy_quote_volume = EXCLUDED.taker_buy_quote_volume;
            """, rows)
        conn.commit()