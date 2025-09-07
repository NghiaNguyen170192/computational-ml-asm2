# Prerequisites

Make sure the following tools are installed on your system:

- [Docker](https://docs.docker.com/get-docker/) `^28.1.1`
- [Docker Compose](https://docs.docker.com/compose/install/) `^2.35.1`
- [Python](https://www.python.org/downloads/) `^3.12`

# Configure Localhost DNS Aliases for Airflow, pgAdmin, and MinIO

By default, services running in Docker or locally are accessed via `http://localhost:<port>`.  
To make them easier to remember, we can add custom hostnames like:

- `http://airflow.local`
- `http://pgadmin.local`
- `http://minio.local`

This is done by editing the **hosts file** in your operating system to map names to `127.0.0.1`.

---

## 🖥️ Unix / Linux / macOS

1. Open a terminal and edit the hosts file: `sudo nano /etc/hosts`
2. Add these entries at the bottom
   ```bash
   127.0.0.1   airflow.local
   127.0.0.1   pgadmin.local
   127.0.0.1   minio.local

## 🖥️ Windows
1. Run Notepad as **Administrator**.
2. Open the hosts file located at: `C:\Windows\System32\drivers\etc\hosts`
3. Add these entries at the bottom:
   ```bash
   127.0.0.1   airflow.local
   127.0.0.1   pgadmin.local
   127.0.0.1   minio.local

# Bootstrapping Airflow

1. `cd .\orchestration`
2. Create `.env` file in the root of orchestration directory
3. Fill in these variables in `.env`

   ```env
   # Airflow
   AIRFLOW_UID=change_me
   AIRFLOW_GID=change_me
   AIRFLOW_PROJ_DIR=change_me
   
   # Postgres
   POSTGRES_USER=change_me
   POSTGRES_PASSWORD=change_me
   POSTGRES_DB=change_me
   
   # pgAdmin
   PGADMIN_DEFAULT_EMAIL=change_me
   PGADMIN_DEFAULT_PASSWORD=change_me
   
   # MinIO
   MINIO_ROOT_USER=change_me
   MINIO_ROOT_PASSWORD=change_me
   MINIO_BUCKET_NAME=change_me
    ```

4. Run `docker build . --no-cache`
5. Run `docker compose up -d`

6. **Access Airflow UI**  
   Open [http://airflow.local](http://airflow.local) in your browser.  
   Default credentials:
   - Username: `airflow`
   - Password: `airflow`

7. **Access pgAdmin**  
   Open [http://pgadmin.local](http://pgadmin.local) in your browser.  
   Credentials from `.env`:
   - Email: `PGADMIN_DEFAULT_EMAIL`
   - Password: `PGADMIN_DEFAULT_PASSWORD`

8. **Access MinIO**  
   Open [http://minio.local](http://minio.local) in your browser.  
   Default credentials (or set in `.env`):
   - Username: `MINIO_ROOT_USER`
   - Password: `MINIO_ROOT_PASSWORD`


# Troubleshooting

- **Restart services:**

  ```sh
  docker compose restart
  ```

- **Remove all containers:**

  ```sh
  docker compose down
  ```

# Folder structure
```
├── config/ # Configuration files for Airflow or services
├── dags/ # Airflow DAGs (workflows)
├── migrations/ # Database migration scripts
│ └── 000_init.sql # Initialization script for Postgres (used by docker postgres-init service)
├── nginx/
│ └── conf.d/ # Nginx configuration files
├── plugins/ # Custom Airflow plugins
├── .env.example # Environment variable template
├── Dockerfile # Custom Airflow Docker image
├── README.md # Project documentation
├── docker-compose.yaml # Docker Compose setup for Airflow, Postgres, pgAdmin, MinIO
└── requirements.txt # Python dependencies for Airflow
```