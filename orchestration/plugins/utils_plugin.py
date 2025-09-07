# plugins/binance_plugin.py
import time
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from psycopg2.extras import execute_values
from binance.client import Client

def fetch_to_minio(start_dt, end_dt, chunk_size, symbol='BTCUSDT'):
    client = Client()
    s3_hook = Utils.get_s3_hook()
    bucket_name = Utils.get_minio_bucket()

    current_dt = start_dt
    while current_dt < end_dt:
        current_dt_str = current_dt.strftime('%Y%m%d')
        object_key = f"klines/klines_{current_dt_str}.csv"

        if s3_hook.check_for_key(key=object_key, bucket_name=bucket_name):
            print(f"File already exists for {current_dt_str}. Skipping.")
            current_dt += chunk_size
            continue

        start_ts = int(current_dt.timestamp() * 1000)
        end_ts = int((current_dt + chunk_size).timestamp() * 1000)

        try:
            klines = client.get_klines(
                symbol=symbol,
                interval='1m',
                startTime=start_ts,
                endTime=end_ts
            )
            if not klines:
                print(f"No data returned for {current_dt_str}")
                current_dt += chunk_size
                continue

            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'num_trades',
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            df['symbol'] = symbol
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

            save_pd_to_minio(s3_hook, df, object_key)
            print(f"Saved data for {current_dt_str}")

        except Exception as e:
            print(f"Error fetching {current_dt_str}: {e}")
            time.sleep(60)

        current_dt += chunk_size
        time.sleep(0.5)


def save_pd_to_minio(s3_hook, df: pd.DataFrame, object_key: str):
    bucket_name = Utils.get_minio_bucket()
    with BytesIO() as buffer:
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        s3_hook.load_bytes(
            bytes_data=buffer.read(),
            key=object_key,
            bucket_name=bucket_name,
            replace=True
        )


def upsert_csv_from_minio():
    s3_hook = Utils.get_s3_hook()
    bucket_name = Utils.get_minio_bucket()
    keys = s3_hook.list_keys(bucket_name=bucket_name)
    if not keys:
        raise ValueError(f"No objects found in bucket '{bucket_name}'")

    unprocessed_keys = [k for k in keys if not k.endswith('_PROCESSED.csv')]

    def extract_timestamp(key):
        try:
            date_str = key.split('_')[-1].replace('.csv', '')
            return pd.to_datetime(date_str)
        except Exception:
            return pd.Timestamp.max

    sorted_keys = sorted(unprocessed_keys, key=extract_timestamp)

    for object_key in sorted_keys:
        print(f"Processing {object_key}")
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
            upsert_to_postgres(rows)

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
    hook = Utils.get_postgres_hook()
    with hook.get_conn() as conn:
        with conn.cursor() as cursor:
            execute_values(cursor, """
                INSERT INTO binance_klines (open_time, symbol, open, high, low, close, volume,
                                            taker_buy_base_volume, taker_buy_quote_volume)
                VALUES %s
                ON CONFLICT (open_time, symbol) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    taker_buy_base_volume = EXCLUDED.taker_buy_base_volume,
                    taker_buy_quote_volume = EXCLUDED.taker_buy_quote_volume;
            """, rows)
            conn.commit()
