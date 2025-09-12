# Configuration Guide - Bitcoin Price Predictor

## 📋 **Overview**

This guide explains how to configure the Bitcoin Price Predictor application using environment variables. The application uses `.env` files for easy configuration management across different environments.

## 🔧 **Configuration Files**

### **`.env.sample`**
- Template file with all available configuration options
- Contains detailed comments and examples
- Safe to commit to version control
- Copy to `.env` and customize for your environment

### **`.env`**
- Your actual configuration file
- Contains sensitive information (passwords, API keys)
- **NEVER commit to version control**
- Automatically loaded by the application

## 🚀 **Quick Setup**

### **1. Copy the Sample File**
```bash
cp .env.sample .env
```

### **2. Edit Configuration**
```bash
nano .env  # or use your preferred editor
```

### **3. Update Key Settings**
- Database connection details
- Secret key for security
- Any other environment-specific settings

## 📊 **Configuration Categories**

### **🗄️ Database Configuration**
```env
# PostgreSQL connection settings
DB_HOST=postgres                    # Database host (Docker container name)
DB_PORT=5432                        # PostgreSQL port
DB_NAME=airflow                     # Database name
DB_USER=airflow                     # Database username
DB_PASSWORD=airflow                 # Database password
```

### **🌐 Flask Application**
```env
# Flask application settings
FLASK_APP=app.py                    # Main application file
FLASK_ENV=production                # Environment (development/production)
FLASK_DEBUG=False                   # Debug mode
SECRET_KEY=your-secret-key-here     # Session secret key
```

### **🤖 Machine Learning**
```env
# Model training settings
TRAINING_DATA_DAYS=365              # Days of data for training
PREDICTION_DAYS=7                   # Days to predict ahead
DRIFT_THRESHOLD=0.1                 # Data drift detection threshold
MIN_SAMPLES_FOR_DRIFT=100           # Minimum samples for drift detection
RETRAIN_FREQUENCY_DAYS=7            # Days between retraining
```

### **📰 News Sentiment**
```env
# News analysis settings
NEWS_LOOKBACK_DAYS=30               # Days of news to analyze
SENTIMENT_WEIGHT=0.3                # Weight of sentiment in predictions
IMPACT_WEIGHT=0.2                   # Weight of impact score
BITCOIN_KEYWORDS=bitcoin,btc,crypto # Keywords for news filtering
```

### **📝 Logging**
```env
# Logging configuration
LOG_LEVEL=INFO                      # Log level (DEBUG/INFO/WARNING/ERROR)
ENABLE_FILE_LOGGING=True            # Enable file logging
ENABLE_CONSOLE_LOGGING=True         # Enable console logging
```

## 🔒 **Security Configuration**

### **Secret Key**
```env
# Change this in production!
SECRET_KEY=bitcoinpredictor-secret-key-2024-change-in-production
```

### **Session Management**
```env
# Security settings
SESSION_TIMEOUT=3600                # Session timeout in seconds
MAX_LOGIN_ATTEMPTS=5                # Maximum login attempts
LOGIN_LOCKOUT_TIME=300              # Lockout time in seconds
```

## 🐳 **Docker Configuration**

### **Container Settings**
```env
# Docker-specific settings
DOCKER_NETWORK=orchestration_nginx-network  # Docker network name
CONTAINER_NAME=bitcoinpredictor            # Container name
```

### **Port Mapping**
```env
# Application port (mapped to 5000 in docker-compose.yml)
APP_PORT=5000
```

## 🔧 **Environment-Specific Configurations**

### **Development Environment**
```env
FLASK_ENV=development
FLASK_DEBUG=True
LOG_LEVEL=DEBUG
ENABLE_DEBUG_TOOLBAR=True
ENABLE_PROFILING=True
```

### **Production Environment**
```env
FLASK_ENV=production
FLASK_DEBUG=False
LOG_LEVEL=INFO
ENABLE_DEBUG_TOOLBAR=False
ENABLE_PROFILING=False
```

### **Testing Environment**
```env
FLASK_ENV=testing
FLASK_DEBUG=True
LOG_LEVEL=DEBUG
ENABLE_TEST_DATA=True
```

## 📋 **Configuration Validation**

### **Required Variables**
- `DB_HOST` - Database host
- `DB_PORT` - Database port
- `DB_NAME` - Database name
- `DB_USER` - Database username
- `DB_PASSWORD` - Database password
- `SECRET_KEY` - Flask secret key

### **Optional Variables**
- All other variables have sensible defaults
- Can be overridden for specific environments

## 🔄 **Configuration Loading Order**

1. **Default values** (hardcoded in application)
2. **Environment variables** (from system)
3. **`.env` file** (from current directory)
4. **Docker environment** (from docker-compose.yml)

## 🚨 **Common Configuration Issues**

### **Database Connection Issues**
```bash
# Check if database is accessible
docker exec -it bitcoinpredictor python -c "
from bitcoin_data_fetcher import BitcoinDataFetcher
import os
db_config = {
    'host': os.getenv('DB_HOST', 'postgres'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'airflow'),
    'user': os.getenv('DB_USER', 'airflow'),
    'password': os.getenv('DB_PASSWORD', 'airflow')
}
fetcher = BitcoinDataFetcher(db_config)
print('Database connection successful!')
"
```

### **Environment Variable Not Loading**
```bash
# Check if .env file is loaded
docker exec -it bitcoinpredictor python -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('DB_HOST:', os.getenv('DB_HOST'))
print('DB_PORT:', os.getenv('DB_PORT'))
"
```

## 📚 **Advanced Configuration**

### **Custom Model Parameters**
```env
# Override default model parameters
RMSE_THRESHOLD=5000.0               # RMSE threshold for model evaluation
MAE_THRESHOLD=3000.0                # MAE threshold for model evaluation
MAPE_THRESHOLD=0.15                 # MAPE threshold for model evaluation
```

### **External API Integration**
```env
# External API keys (optional)
BINANCE_API_KEY=your-binance-api-key
NEWS_API_KEY=your-news-api-key
REDDIT_CLIENT_ID=your-reddit-client-id
REDDIT_CLIENT_SECRET=your-reddit-client-secret
```

### **Monitoring Configuration**
```env
# Health check settings
HEALTH_CHECK_INTERVAL=30            # Health check interval in seconds
HEALTH_CHECK_TIMEOUT=10             # Health check timeout in seconds
HEALTH_CHECK_RETRIES=3              # Health check retries
```

## 🔄 **Configuration Updates**

### **After Changing Configuration**
```bash
# Restart the application
docker-compose restart bitcoinpredictor

# Or rebuild if needed
docker-compose up --build -d bitcoinpredictor
```

### **View Current Configuration**
```bash
# Check environment variables in container
docker exec -it bitcoinpredictor env | grep -E "(DB_|FLASK_|LOG_)"
```

## 📖 **Best Practices**

### **Security**
- ✅ Use strong, unique secret keys
- ✅ Never commit `.env` files to version control
- ✅ Use different configurations for different environments
- ✅ Regularly rotate passwords and API keys

### **Performance**
- ✅ Adjust logging levels based on environment
- ✅ Configure appropriate timeouts
- ✅ Set reasonable retry limits

### **Maintenance**
- ✅ Document custom configurations
- ✅ Use `.env.sample` as a template
- ✅ Test configurations before deployment
- ✅ Keep configurations in sync across environments

## 🆘 **Troubleshooting**

### **Configuration Not Loading**
1. Check if `.env` file exists in the correct directory
2. Verify file permissions
3. Check for syntax errors in `.env` file
4. Restart the application

### **Database Connection Issues**
1. Verify database credentials
2. Check if database is running
3. Verify network connectivity
4. Check Docker network configuration

### **Application Startup Issues**
1. Check all required environment variables
2. Verify Python dependencies
3. Check application logs
4. Validate configuration syntax

## 📞 **Support**

For configuration issues:
1. Check the application logs: `docker-compose logs bitcoinpredictor`
2. Verify environment variables: `docker exec -it bitcoinpredictor env`
3. Test database connection using the validation commands above
4. Review this guide for common issues and solutions
