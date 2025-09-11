# Comprehensive Logging System Guide

## Overview

The Bitcoin Price Prediction system now includes a comprehensive logging system that tracks all operations, model training, predictions, and system activities. All logs are organized by date and operation type for easy analysis and debugging.

## Log Directory Structure

```
logs/
├── training/                    # Model training logs
│   └── training_YYYY-MM-DD.log
├── predictions/                 # Prediction request logs
│   └── predictions_YYYY-MM-DD.log
├── drift_detection/            # Data drift analysis logs
│   └── drift_YYYY-MM-DD.log
├── model_evaluation/           # Model performance logs
│   └── evaluation_YYYY-MM-DD.log
├── data_processing/            # Data processing logs
│   └── data_YYYY-MM-DD.log
├── system_operations/          # System operation logs
│   └── system_YYYY-MM-DD.log
├── errors/                     # Error logs
│   └── errors_YYYY-MM-DD.log
└── daily_summaries/            # Daily summary reports
    └── summary_YYYY-MM-DD.json
```

## What Gets Logged

### 1. Model Training Logs (`logs/training/`)

**When**: Every time a model is trained or retrained

**What's Logged**:
- Training start with data information
- Step-by-step training process
- Data preparation details
- Feature engineering steps
- Model configuration parameters
- Training performance metrics
- Model saving details

**Example Log Entry**:
```
2025-09-11 10:30:15 - training - INFO - ================================================================================
2025-09-11 10:30:15 - training - INFO - TRAINING STARTED - BITCOIN PRICE PREDICTION
2025-09-11 10:30:15 - training - INFO - Session ID: 20250911_103015
2025-09-11 10:30:15 - training - INFO - Timestamp: 2025-09-11T10:30:15.123456
2025-09-11 10:30:15 - training - INFO - ================================================================================
2025-09-11 10:30:15 - training - INFO - DATA INFORMATION:
2025-09-11 10:30:15 - training - INFO -   - Data points: 1004
2025-09-11 10:30:15 - training - INFO -   - Date range: ('2022-01-01', '2025-09-11')
2025-09-11 10:30:15 - training - INFO -   - Has news data: True
2025-09-11 10:30:15 - training - INFO -   - Data quality: Good
```

### 2. Prediction Logs (`logs/predictions/`)

**When**: Every prediction request is made

**What's Logged**:
- Prediction request details
- Input parameters
- Model used for prediction
- Prediction results with confidence intervals
- Model performance information
- Explanations for predictions

**Example Log Entry**:
```
2025-09-11 10:35:22 - predictions - INFO - ============================================================
2025-09-11 10:35:22 - predictions - INFO - PREDICTION REQUEST - BITCOIN PRICE PREDICTION
2025-09-11 10:35:22 - predictions - INFO - Timestamp: 2025-09-11T10:35:22.789012
2025-09-11 10:35:22 - predictions - INFO - ============================================================
2025-09-11 10:35:22 - predictions - INFO - INPUT DATA:
2025-09-11 10:35:22 - predictions - INFO -   prediction_date: 2025-09-11
2025-09-11 10:35:22 - predictions - INFO -   days_ahead: 7
2025-09-11 10:35:22 - predictions - INFO -   has_news_data: True
2025-09-11 10:35:22 - predictions - INFO -   model_trained: True
```

### 3. Drift Detection Logs (`logs/drift_detection/`)

**When**: Data drift analysis is performed

**What's Logged**:
- Drift detection analysis start
- Statistical tests performed (KS test, t-test, F-test)
- Current vs baseline statistics comparison
- Volatility and price level changes
- Recommendations for retraining
- Time-based retraining triggers

**Example Log Entry**:
```
2025-09-11 10:40:15 - drift - INFO - ======================================================================
2025-09-11 10:40:15 - drift - INFO - DRIFT DETECTION ANALYSIS
2025-09-11 10:40:15 - drift - INFO - Timestamp: 2025-09-11T10:40:15.456789
2025-09-11 10:40:15 - drift - INFO - ======================================================================
2025-09-11 10:40:15 - drift - INFO - Drift Detected: True
2025-09-11 10:40:15 - drift - INFO - Reason: Distribution change detected - retrain model
2025-09-11 10:40:15 - drift - INFO - Recommendations:
2025-09-11 10:40:15 - drift - INFO -   - Distribution change detected - retrain model
2025-09-11 10:40:15 - drift - INFO -   - Volatility changed by 15.3%
```

### 4. Model Evaluation Logs (`logs/model_evaluation/`)

**When**: Model performance is evaluated

**What's Logged**:
- Model performance metrics (RMSE, MAE, R²)
- Training vs validation performance
- Model comparison results
- Feature importance (for gradient boosting models)
- Model selection rationale

**Example Log Entry**:
```
2025-09-11 10:32:45 - evaluation - INFO - MODEL PERFORMANCE - XGBOOST
2025-09-11 10:32:45 - evaluation - INFO - ==================================================
2025-09-11 10:32:45 - evaluation - INFO - rmse: 1250.45
2025-09-11 10:32:45 - evaluation - INFO - mae: 980.23
2025-09-11 10:32:45 - evaluation - INFO - r2: 0.85
2025-09-11 10:32:45 - evaluation - INFO - ==================================================
```

### 5. Data Processing Logs (`logs/data_processing/`)

**When**: Data is processed, cleaned, or transformed

**What's Logged**:
- Data loading from database
- Data cleaning operations
- Feature engineering steps
- Data validation results
- News data processing
- Data quality metrics

**Example Log Entry**:
```
2025-09-11 10:31:10 - data - INFO - DATA PROCESSING: Data Cleaning
2025-09-11 10:31:10 - data - INFO - Timestamp: 2025-09-11T10:31:10.234567
2025-09-11 10:31:10 - data - INFO -   before_cleaning: 1004
2025-09-11 10:31:10 - data - INFO -   columns: ['ds', 'y', 'avg_polarity', 'avg_subjectivity', 'positive_ratio']
2025-09-11 10:31:10 - data - INFO -   data_types: {'ds': 'datetime64[ns]', 'y': 'float64', 'avg_polarity': 'float64'}
```

### 6. System Operations Logs (`logs/system_operations/`)

**When**: System operations occur

**What's Logged**:
- Application startup/shutdown
- Database connections
- Model loading/saving
- Configuration changes
- Health checks

**Example Log Entry**:
```
2025-09-11 10:30:00 - system - INFO - SYSTEM OPERATION: Application Startup
2025-09-11 10:30:00 - system - INFO - Timestamp: 2025-09-11T10:30:00.123456
2025-09-11 10:30:00 - system - INFO -   status: started
2025-09-11 10:30:00 - system - INFO -   models_available: ['XGBoost', 'LightGBM', 'Statistical Fallback']
2025-09-11 10:30:00 - system - INFO -   drift_detection_enabled: True
```

### 7. Error Logs (`logs/errors/`)

**When**: Errors occur during any operation

**What's Logged**:
- Error type and message
- Context information
- Stack traces
- Recovery attempts
- Error resolution

**Example Log Entry**:
```
2025-09-11 10:35:30 - errors - ERROR - ==================================================
2025-09-11 10:35:30 - errors - ERROR - ERROR: Prediction Error
2025-09-11 10:35:30 - errors - ERROR - Timestamp: 2025-09-11T10:35:30.789012
2025-09-11 10:35:30 - errors - ERROR - Message: No trained model found. Train the model first.
2025-09-11 10:35:30 - errors - ERROR - Context:
2025-09-11 10:35:30 - errors - ERROR -   prediction_date: 2025-09-11
2025-09-11 10:35:30 - errors - ERROR -   days_ahead: 7
2025-09-11 10:35:30 - errors - ERROR - ==================================================
```

### 8. Daily Summaries (`logs/daily_summaries/`)

**When**: End of each day

**What's Logged**:
- Session summary with operation counts
- Models trained during the day
- Predictions made
- Errors encountered
- Performance metrics
- System health status

**Example JSON Summary**:
```json
{
  "session_id": "20250911_103015",
  "start_time": "2025-09-11T10:30:15.123456",
  "end_time": "2025-09-11T18:45:30.789012",
  "duration_minutes": 495.25,
  "operations": [
    {
      "type": "training_start",
      "model_type": "Bitcoin Price Prediction",
      "timestamp": "2025-09-11T10:30:15.123456",
      "data_info": {
        "data_points": 1004,
        "date_range": ["2022-01-01", "2025-09-11"],
        "has_news_data": true,
        "data_quality": "Good"
      }
    }
  ],
  "models_trained": [
    {
      "model_type": "XGBoost",
      "performance": {
        "rmse": 1250.45,
        "mae": 980.23,
        "r2": 0.85
      },
      "timestamp": "2025-09-11T10:32:45.456789"
    }
  ],
  "predictions_made": 15,
  "errors_encountered": 2
}
```

## How to Use the Logs

### 1. Monitor Training Progress
```bash
# Watch training logs in real-time
tail -f logs/training/training_$(date +%Y-%m-%d).log

# Search for specific training steps
grep "TRAINING STEP" logs/training/training_$(date +%Y-%m-%d).log
```

### 2. Debug Prediction Issues
```bash
# Check prediction logs
tail -f logs/predictions/predictions_$(date +%Y-%m-%d).log

# Find prediction errors
grep "ERROR" logs/predictions/predictions_$(date +%Y-%m-%d).log
```

### 3. Monitor Data Drift
```bash
# Check drift detection results
grep "Drift Detected" logs/drift_detection/drift_$(date +%Y-%m-%d).log

# View statistical test results
grep "Statistical Tests" logs/drift_detection/drift_$(date +%Y-%m-%d).log
```

### 4. Analyze Model Performance
```bash
# Check model evaluation logs
grep "MODEL PERFORMANCE" logs/model_evaluation/evaluation_$(date +%Y-%m-%d).log

# Compare different models
grep -A 5 "XGBOOST\|LIGHTGBM\|STATISTICAL" logs/model_evaluation/evaluation_$(date +%Y-%m-%d).log
```

### 5. Track System Health
```bash
# Monitor system operations
tail -f logs/system_operations/system_$(date +%Y-%m-%d).log

# Check for errors
grep "ERROR" logs/errors/errors_$(date +%Y-%m-%d).log
```

### 6. View Daily Summary
```bash
# Check daily summary
cat logs/daily_summaries/summary_$(date +%Y-%m-%d).json | jq '.'

# Get operation counts
cat logs/daily_summaries/summary_$(date +%Y-%m-%d).json | jq '.operations | length'
```

## Log Analysis Tools

### 1. Real-time Monitoring
```bash
# Monitor all logs simultaneously
tail -f logs/*/$(date +%Y-%m-%d).log

# Monitor specific operations
tail -f logs/training/training_$(date +%Y-%m-%d).log logs/predictions/predictions_$(date +%Y-%m-%d).log
```

### 2. Search and Filter
```bash
# Find all errors across all logs
grep -r "ERROR" logs/

# Find specific model training
grep -r "XGBoost" logs/training/

# Find drift detection results
grep -r "Drift Detected: True" logs/drift_detection/
```

### 3. Performance Analysis
```bash
# Extract performance metrics
grep "rmse\|mae\|r2" logs/model_evaluation/evaluation_$(date +%Y-%m-%d).log

# Find best performing models
grep -A 3 "MODEL PERFORMANCE" logs/model_evaluation/evaluation_$(date +%Y-%m-%d).log | grep -E "rmse|mae|r2"
```

## Benefits for Your ML Assignment

### 1. **Academic Excellence**
- **Detailed Documentation**: Every step is logged with explanations
- **Reproducible Results**: Complete audit trail of all operations
- **Performance Tracking**: Comprehensive metrics for model comparison
- **Error Analysis**: Detailed error tracking and resolution

### 2. **Research Quality**
- **Statistical Rigor**: All statistical tests are logged with results
- **Model Comparison**: Side-by-side performance comparison
- **Data Quality**: Complete data processing audit trail
- **Drift Analysis**: Detailed drift detection and retraining logs

### 3. **Production Readiness**
- **Monitoring**: Real-time system health monitoring
- **Debugging**: Comprehensive error tracking and context
- **Performance**: Detailed performance metrics and trends
- **Maintenance**: Clear logs for system maintenance

### 4. **Lecturer Evidence**
- **Complete Audit Trail**: Every operation is documented
- **Model Performance**: Detailed performance metrics
- **Data Processing**: Step-by-step data handling
- **System Robustness**: Error handling and recovery logs

## Log Retention and Cleanup

The logging system automatically creates daily log files. For production use, consider implementing log rotation:

```bash
# Example log rotation (add to crontab)
0 0 * * * find logs/ -name "*.log" -mtime +30 -delete
```

This comprehensive logging system ensures that every aspect of your Bitcoin price prediction system is thoroughly documented, making it easy to understand, debug, and demonstrate the system's capabilities to your lecturer! 🎯
