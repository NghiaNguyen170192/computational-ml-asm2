# Project Structure

```
computational-ml-asm2/
├── README.md                    # Main project documentation
├── start.sh                     # One-command startup script
├── orchestration/               # Data collection system
│   ├── docker-compose.yaml     # Orchestration services
│   ├── dags/                   # Airflow DAGs for data collection
│   ├── migrations/             # Database schema
│   └── plugins/                # Custom Airflow plugins
└── app/                        # Bitcoin Price Predictor application
    ├── Dockerfile              # Container definition
    ├── docker-compose.yml      # App services
    ├── requirements.txt        # Python dependencies
    ├── app.py                  # Main Flask application
    ├── bitcoin_data_fetcher.py # Database connection & data fetching
    ├── bitcoin_predictor.py    # ML models (Prophet + fallback)
    ├── templates/              # Web UI templates
    │   ├── base.html
    │   ├── index.html
    │   └── dashboard.html
    ├── docker-start.sh         # App startup script
    ├── deploy-to-aws.sh        # AWS deployment script
    ├── aws-deploy.yml          # ECS task definition
    ├── nginx.conf              # Reverse proxy config
    └── README_DOCKER.md        # Docker deployment guide
```

## 🚀 **Quick Start**

```bash
# One command to start everything
./start.sh
```

## 🐳 **Docker Services**

### **Orchestration System** (`orchestration/`)
- **PostgreSQL**: Database for Bitcoin price data
- **Airflow**: Data collection and scheduling
- **MinIO**: Object storage for data files

### **Bitcoin Predictor App** (`app/`)
- **bitcoin-predictor**: Main Flask application
- **postgres**: Optional included database
- **nginx**: Optional reverse proxy

## 📁 **Key Files**

- `start.sh` - Main startup script
- `app/app.py` - Flask web application
- `app/bitcoin_predictor.py` - ML models
- `app/bitcoin_data_fetcher.py` - Database operations
- `orchestration/docker-compose.yaml` - Data collection services

## 🔧 **Configuration**

All configuration is done through:
- Environment variables in `docker-compose.yml`
- Database connection settings in `app.py`
- Model parameters in `bitcoin_predictor.py`

## ☁️ **Deployment**

- **Local Development**: Use `./start.sh`
- **AWS Deployment**: Use `app/deploy-to-aws.sh`
- **Docker Only**: Use `app/docker-start.sh`
