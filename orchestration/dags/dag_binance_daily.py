with DAG(
    dag_id='dag_binance_daily',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['binance', 'daily'],
) as dag:
    fetch_task = PythonOperator(
        task_id='fetch_daily',
        python_callable=lambda: fetch_to_minio(
            start_dt=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
            end_dt=datetime.now(),
            chunk_size=timedelta(days=1)
        ),
    )

    upsert_task = PythonOperator(
        task_id='upsert_from_minio',
        python_callable=upsert_csv_from_minio,
    )

    fetch_task >> upsert_task
