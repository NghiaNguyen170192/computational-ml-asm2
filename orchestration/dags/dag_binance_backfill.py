from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
from plugins.binance_plugin import fetch_to_minio, upsert_csv_from_minio

default_args = {
    'owner': 'nghia',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='dag_binance_backfill',
    default_args=default_args,
    schedule_interval='@once',
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['binance', 'historical'],
) as dag:
    fetch_task = PythonOperator(
        task_id='fetch_historical',
        python_callable=lambda: fetch_to_minio(
            start_dt=datetime(2017, 8, 17),
            end_dt=datetime.now() - timedelta(days=1),
            chunk_size=timedelta(days=1)
        ),
    )

    upsert_task = PythonOperator(
        task_id='upsert_from_minio',
        python_callable=upsert_csv_from_minio,
    )

    fetch_task >> upsert_task
