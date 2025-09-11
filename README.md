# Bitcoin Price Predictor - RMIT ML Course

A comprehensive Bitcoin price prediction system that integrates real-time data from PostgreSQL, news sentiment analysis, and advanced time series forecasting using Prophet.

## 🚀 **Quick Start**

### **One-Command Startup**

```bash
# From project root directory
./start.sh
```

This will:
1. ✅ Start the orchestration system (PostgreSQL, Airflow)
2. ✅ Build and start the Bitcoin prediction app
3. ✅ Make everything available at http://localhost:5500

### **Manual Docker Commands**

```bash
# 1. Start orchestration system
cd orchestration
docker-compose up -d

# 2. Start Bitcoin predictor app
cd ../app
docker-compose up --build -d
```

### **App Management Script**

For easier app management, use the provided script:

```bash
# Go to app directory
cd app

# Start the app
./manage-app.sh start

# Restart the app
./manage-app.sh restart

# Check status
./manage-app.sh status

# View logs
./manage-app.sh logs

# Stop the app
./manage-app.sh stop

# Show help
./manage-app.sh help
```

## 🌐 **Access the Application**

- **Web Interface**: http://localhost:5500
- **Health Check**: http://localhost:5500/health

### **Login Credentials**
- Username: `student` | Password: `ml2025`
- Username: `demo` | Password: `password123`
- Username: `admin` | Password: `rmit2025`

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Orchestration │    │   PostgreSQL     │    │   Web App       │
│   (Airflow)     │───▶│   Database       │◀───│   (Docker)      │
│                 │    │   -binance_klines │   │                 │
└─────────────────┘    │   -crypto_news   │    └─────────────────┘
                       └──────────────────┘
                                ▲
                                │
                       ┌──────────────────┐
                       │   News Data      │
                       │   (CSV File)     │
                       └──────────────────┘
```

## 📋 **Prerequisites**

- **Docker** and **Docker Compose**
- **Git** (to clone the repository)

## 🛠️ **Features**

- **Real-time Data**: Connects to PostgreSQL database populated by orchestration system
- **News Sentiment Analysis**: Integrates crypto news sentiment for enhanced predictions
- **Advanced ML Models**: Uses Facebook Prophet for time series forecasting with fallback models
- **Interactive UI**: Modern web interface with real-time charts and confidence intervals
- **Model Evaluation**: Comprehensive metrics including RMSE, MAE, and MAPE
- **Dockerized**: Consistent environment across all machines

## 🔧 **Configuration**

### **Environment Variables**

The app supports these environment variables:

```bash
DB_HOST=localhost          # Database host
DB_PORT=5432              # Database port
DB_NAME=airflow           # Database name
DB_USER=airflow           # Database user
DB_PASSWORD=airflow       # Database password
FLASK_ENV=production      # Flask environment
```

### **Docker Services**

1. **orchestration**: PostgreSQL, Airflow, MinIO (data collection)
2. **bitcoin-predictor**: Main application (Flask) - connects to orchestration PostgreSQL

## 📊 **Data Sources**

### **1. Bitcoin Price Data (binance_klines table)**
- Real-time Bitcoin price data from Binance API
- Collected via Airflow DAGs
- Fields: open_time, symbol, open, high, low, close, volume, etc.

### **2. News Data (cryptonews-2022-2023.csv)**
- Crypto news articles with sentiment analysis
- Fields: date, title, text, sentiment, source, etc.
- Integrated as external regressors in Prophet model

## 🎯 **Usage**

### **Making Predictions**

1. **Login** using demo credentials
2. **Test Connection**: Click "Test Connection & Load Data" to verify database connectivity
3. **Make Prediction**:
   - Select prediction date
   - Choose number of days ahead (1-30)
   - Click "Predict Bitcoin Price"
4. **View Results**: See predictions with confidence intervals and model information

### **Model Management**

- **Retrain Model**: Click "Retrain Model" to update with latest data
- **View Logs**: Access prediction logs and model information
- **Clear Data**: Reset models and logs

## 📈 **Model Performance**

The system provides comprehensive evaluation metrics:

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **MAPE**: Mean Absolute Percentage Error
- **Confidence Intervals**: Upper and lower bounds for predictions

## 🐳 **Docker Commands**

### **App Management (Recommended)**

Use the management script for easier control:

```bash
# Go to app directory
cd app

# Start app
./manage-app.sh start

# Restart app
./manage-app.sh restart

# View logs
./manage-app.sh logs

# Check status
./manage-app.sh status

# Stop app
./manage-app.sh stop
```

### **Manual Docker Commands**

```bash
# View logs
cd app && docker-compose logs -f bitcoin-predictor

# Stop app
cd app && docker-compose down

# Restart app
cd app && docker-compose restart bitcoin-predictor

# Rebuild app
cd app && docker-compose up --build -d
```

### **Orchestration Management**

```bash
# View orchestration logs
cd orchestration && docker-compose logs -f

# Stop orchestration
cd orchestration && docker-compose down

# Restart orchestration
cd orchestration && docker-compose restart
```

## 🔍 **Troubleshooting**

### **App Won't Start**

```bash
# Check if orchestration is running
cd orchestration && docker-compose ps

# Check app logs
cd app && docker-compose logs bitcoin-predictor

# Rebuild app
cd app && docker-compose up --build -d
```

### **Database Connection Issues**

```bash
# Check database status
cd orchestration && docker-compose ps postgres

# Check database logs
cd orchestration && docker-compose logs postgres

# Restart database
cd orchestration && docker-compose restart postgres
```

### **No Data Available**

1. Wait for Airflow to collect data (check Airflow UI at http://localhost:8080)
2. Manually trigger data collection DAGs
3. Check if orchestration system is running properly


## 🎓 **Assignment Requirements**

This system fulfills the ML assignment requirements:

- ✅ **Real Machine Learning**: Prophet time series forecasting with news sentiment
- ✅ **Complete System**: Web application with database integration
- ✅ **User Interface**: Accessible web interface for normal users
- ✅ **Data Processing**: Real-time data collection and processing
- ✅ **Model Evaluation**: Comprehensive performance metrics
- ✅ **Documentation**: Complete setup and usage instructions
- ✅ **Dockerized**: Consistent deployment across machines

## 📚 **Technical Details**

### **Prophet Model Configuration**

```python
Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0
)
```

### **News Sentiment Integration**

- **Polarity**: Sentiment strength (-1 to 1)
- **Subjectivity**: Opinion vs fact (0 to 1)
- **Positive Ratio**: Percentage of positive news
- **External Regressors**: Added to Prophet model for enhanced predictions

## 🤝 **Support**

For issues or questions:

1. Check the troubleshooting section
2. Check Docker logs: `docker-compose logs`
3. Verify orchestration system is running: `docker-compose ps`
4. Check system health: http://localhost:5000/health

## 📄 **License**

This project is part of the RMIT Machine Learning course assignment.

---

**🚀 Ready to predict Bitcoin prices? Run `./start.sh` and visit http://localhost:5500!**