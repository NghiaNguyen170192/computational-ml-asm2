from airflow import DAG
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.operators.python import PythonOperator
from datetime import datetime
from collections import defaultdict
import pandas as pd
import orjson
from tqdm import tqdm
import tempfile
import os


def read_jsonl_chunks(path, chunk_size):
    with open(path, "rb") as f:
        chunk = []
        for i, line in enumerate(f):
            obj = orjson.loads(line)
            chunk.append(obj)
            if (i + 1) % chunk_size == 0:
                yield pd.DataFrame(chunk)
                chunk = []
        if chunk:
            yield pd.DataFrame(chunk)


def stream_grouped_sessions(input_path, chunk_size, time_window_days, output_path):
    events_by_session = defaultdict(list)
    max_ts = 0

    for chunk in tqdm(read_jsonl_chunks(input_path, chunk_size), desc="Processing chunks"):
        for _, row in chunk.iterrows():
            session_id = row["session"]
            for event in row["events"]:
                ts = event["ts"]
                max_ts = max(max_ts, ts)
                events_by_session[session_id].append(event)

    cutoff_ts = max_ts - (time_window_days * 24 * 60 * 60 * 1000)

    with open(output_path, "wb") as out_file:
        for session_id, events in events_by_session.items():
            filtered = [e for e in events if e["ts"] >= cutoff_ts]
            if not filtered:
                continue

            deduped = {}
            for e in filtered:
                key = (e["aid"], e["type"])
                if key not in deduped or e["ts"] > deduped[key]["ts"]:
                    deduped[key] = e

            final_events = list(deduped.values())
            if not final_events or (len(final_events) == 1 and final_events[0]["type"] == "clicks"):
                continue

            out_obj = {
                "session": session_id,
                "events": sorted(final_events, key=lambda x: x["ts"])
            }
            out_file.write(orjson.dumps(out_obj) + b"\n")


def process_train_jsonl_from_minio():
    hook = S3Hook(aws_conn_id='minio_conn')
    bucket = 'airflow'
    input_key = 'test.jsonl'
    output_key = 'processed_sessions.jsonl'

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, 'test.jsonl')
        output_path = os.path.join(tmpdir, 'processed_sessions.jsonl')

        # Download from MinIO
        hook.get_key(key=input_key, bucket_name=bucket).download_file(
            input_path)

        # Process the file
        stream_grouped_sessions(
            input_path=input_path,
            chunk_size=10000,
            time_window_days=7,
            output_path=output_path
        )

        # Upload result back to MinIO
        hook.load_file(
            filename=output_path,
            key=output_key,
            bucket_name=bucket,
            replace=True
        )


with DAG(
    dag_id='process_train_jsonl_minio',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    process_task = PythonOperator(
        task_id='process_train_jsonl',
        python_callable=process_train_jsonl_from_minio
    )
