"""
Comprehensive Logging System for Bitcoin Price Prediction

This module provides detailed logging for all operations including:
- Model training with step-by-step details
- Data processing and validation
- Model performance evaluation
- Drift detection analysis
- Prediction results and explanations
- System operations and errors

All logs are organized by date and operation type for easy analysis.
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

class ComprehensiveLogger:
    """
    Comprehensive logging system for Bitcoin price prediction operations
    
    Features:
    - Date-based log organization
    - Operation-specific logging
    - Detailed step-by-step tracking
    - Performance metrics logging
    - Error tracking and debugging
    - Model comparison and evaluation
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize comprehensive logger
        
        Args:
            log_dir: Directory to store all log files
        """
        self.log_dir = log_dir
        self.today = datetime.now().strftime('%Y-%m-%d')
        self.current_session = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create log directory structure
        self._create_log_structure()
        
        # Initialize loggers for different operations
        self._setup_loggers()
        
        # Session tracking
        self.session_data = {
            'session_id': self.current_session,
            'start_time': datetime.now().isoformat(),
            'operations': [],
            'models_trained': [],
            'predictions_made': [],
            'errors_encountered': []
        }
    
    def _create_log_structure(self):
        """Create organized log directory structure"""
        subdirs = [
            'training',
            'predictions', 
            'drift_detection',
            'model_evaluation',
            'data_processing',
            'system_operations',
            'errors',
            'daily_summaries'
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(self.log_dir, subdir), exist_ok=True)
    
    def _setup_loggers(self):
        """Setup specialized loggers for different operations"""
        self.loggers = {}
        
        # Training logger
        self.loggers['training'] = self._create_logger(
            'training', 
            os.path.join(self.log_dir, 'training', f'training_{self.today}.log')
        )
        
        # Predictions logger
        self.loggers['predictions'] = self._create_logger(
            'predictions',
            os.path.join(self.log_dir, 'predictions', f'predictions_{self.today}.log')
        )
        
        # Drift detection logger
        self.loggers['drift'] = self._create_logger(
            'drift',
            os.path.join(self.log_dir, 'drift_detection', f'drift_{self.today}.log')
        )
        
        # Model evaluation logger
        self.loggers['evaluation'] = self._create_logger(
            'evaluation',
            os.path.join(self.log_dir, 'model_evaluation', f'evaluation_{self.today}.log')
        )
        
        # Data processing logger
        self.loggers['data'] = self._create_logger(
            'data',
            os.path.join(self.log_dir, 'data_processing', f'data_{self.today}.log')
        )
        
        # System operations logger
        self.loggers['system'] = self._create_logger(
            'system',
            os.path.join(self.log_dir, 'system_operations', f'system_{self.today}.log')
        )
        
        # Errors logger
        self.loggers['errors'] = self._create_logger(
            'errors',
            os.path.join(self.log_dir, 'errors', f'errors_{self.today}.log')
        )
    
    def _create_logger(self, name: str, log_file: str) -> logging.Logger:
        """Create a logger with file and console handlers"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_training_start(self, model_type: str, data_info: Dict):
        """Log the start of model training"""
        self.loggers['training'].info("="*80)
        self.loggers['training'].info(f"TRAINING STARTED - {model_type.upper()}")
        self.loggers['training'].info(f"Session ID: {self.current_session}")
        self.loggers['training'].info(f"Timestamp: {datetime.now().isoformat()}")
        self.loggers['training'].info("="*80)
        
        # Log data information
        self.loggers['training'].info("DATA INFORMATION:")
        self.loggers['training'].info(f"  - Data points: {data_info.get('data_points', 'Unknown')}")
        self.loggers['training'].info(f"  - Date range: {data_info.get('date_range', 'Unknown')}")
        self.loggers['training'].info(f"  - Has news data: {data_info.get('has_news_data', False)}")
        self.loggers['training'].info(f"  - Data quality: {data_info.get('data_quality', 'Unknown')}")
        
        # Track in session
        self.session_data['operations'].append({
            'type': 'training_start',
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'data_info': data_info
        })
    
    def log_training_step(self, step: str, details: Dict):
        """Log a specific training step"""
        self.loggers['training'].info(f"TRAINING STEP: {step}")
        self.loggers['training'].info(f"  Details: {json.dumps(details, indent=2, default=str)}")
    
    def log_model_performance(self, model_type: str, performance: Dict):
        """Log model performance metrics"""
        self.loggers['evaluation'].info(f"MODEL PERFORMANCE - {model_type.upper()}")
        self.loggers['evaluation'].info("="*50)
        
        for metric, value in performance.items():
            if isinstance(value, dict):
                self.loggers['evaluation'].info(f"{metric}:")
                for sub_metric, sub_value in value.items():
                    self.loggers['evaluation'].info(f"  {sub_metric}: {sub_value}")
            else:
                self.loggers['evaluation'].info(f"{metric}: {value}")
        
        self.loggers['evaluation'].info("="*50)
        
        # Track in session
        self.session_data['models_trained'].append({
            'model_type': model_type,
            'performance': performance,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_prediction_request(self, prediction_type: str, input_data: Dict):
        """Log prediction request details"""
        self.loggers['predictions'].info("="*60)
        self.loggers['predictions'].info(f"PREDICTION REQUEST - {prediction_type.upper()}")
        self.loggers['predictions'].info(f"Timestamp: {datetime.now().isoformat()}")
        self.loggers['predictions'].info("="*60)
        
        self.loggers['predictions'].info("INPUT DATA:")
        for key, value in input_data.items():
            self.loggers['predictions'].info(f"  {key}: {value}")
    
    def log_prediction_result(self, prediction_result: Dict):
        """Log prediction results with detailed analysis"""
        self.loggers['predictions'].info("PREDICTION RESULTS:")
        self.loggers['predictions'].info("="*40)
        
        # Log basic prediction info
        if 'predictions' in prediction_result:
            preds = prediction_result['predictions']
            self.loggers['predictions'].info(f"Number of predictions: {len(preds)}")
            self.loggers['predictions'].info(f"Prediction horizon: {prediction_result.get('horizon', 'Unknown')}")
            
            # Log individual predictions
            for i, pred in enumerate(preds):
                self.loggers['predictions'].info(f"  Prediction {i+1}:")
                self.loggers['predictions'].info(f"    Date: {pred.get('date', 'Unknown')}")
                self.loggers['predictions'].info(f"    Price: ${pred.get('price', 0):.2f}")
                self.loggers['predictions'].info(f"    Confidence: {pred.get('confidence', 0):.2f}%")
                if 'explanation' in pred:
                    self.loggers['predictions'].info(f"    Explanation: {pred['explanation']}")
        
        # Log model information
        if 'model_info' in prediction_result:
            model_info = prediction_result['model_info']
            self.loggers['predictions'].info(f"Model used: {model_info.get('model_type', 'Unknown')}")
            self.loggers['predictions'].info(f"Model performance: {model_info.get('performance', 'Unknown')}")
        
        # Log confidence intervals
        if 'confidence_intervals' in prediction_result:
            ci = prediction_result['confidence_intervals']
            self.loggers['predictions'].info(f"Confidence intervals: {ci}")
        
        self.loggers['predictions'].info("="*40)
        
        # Track in session
        self.session_data['predictions_made'].append({
            'timestamp': datetime.now().isoformat(),
            'result': prediction_result
        })
    
    def log_drift_detection(self, drift_results: Dict):
        """Log drift detection analysis"""
        self.loggers['drift'].info("="*70)
        self.loggers['drift'].info("DRIFT DETECTION ANALYSIS")
        self.loggers['drift'].info(f"Timestamp: {datetime.now().isoformat()}")
        self.loggers['drift'].info("="*70)
        
        # Log drift status
        self.loggers['drift'].info(f"Drift Detected: {drift_results.get('drift_detected', False)}")
        self.loggers['drift'].info(f"Reason: {drift_results.get('reason', 'Unknown')}")
        
        # Log recommendations
        if 'recommendations' in drift_results:
            self.loggers['drift'].info("Recommendations:")
            for rec in drift_results['recommendations']:
                self.loggers['drift'].info(f"  - {rec}")
        
        # Log statistical tests
        if 'tests_performed' in drift_results:
            self.loggers['drift'].info("Statistical Tests:")
            for test in drift_results['tests_performed']:
                self.loggers['drift'].info(f"  {test['test']}:")
                self.loggers['drift'].info(f"    Statistic: {test.get('statistic', 'N/A')}")
                self.loggers['drift'].info(f"    P-value: {test.get('p_value', 'N/A')}")
                self.loggers['drift'].info(f"    Significant: {test.get('significant', False)}")
        
        # Log current vs baseline stats
        if 'current_stats' in drift_results and 'baseline_stats' in drift_results:
            self.loggers['drift'].info("Statistical Comparison:")
            current = drift_results['current_stats']
            baseline = drift_results['baseline_stats']
            
            stats_to_compare = ['mean', 'std', 'skewness', 'kurtosis', 'volatility']
            for stat in stats_to_compare:
                if stat in current and stat in baseline:
                    change = ((current[stat] - baseline[stat]) / baseline[stat]) * 100
                    self.loggers['drift'].info(f"  {stat}: {current[stat]:.4f} (baseline: {baseline[stat]:.4f}, change: {change:+.2f}%)")
    
    def log_data_processing(self, operation: str, data_info: Dict):
        """Log data processing operations"""
        self.loggers['data'].info(f"DATA PROCESSING: {operation}")
        self.loggers['data'].info(f"Timestamp: {datetime.now().isoformat()}")
        
        for key, value in data_info.items():
            if isinstance(value, (list, tuple)) and len(value) > 5:
                self.loggers['data'].info(f"  {key}: {len(value)} items (first 5: {value[:5]})")
            elif isinstance(value, pd.DataFrame):
                self.loggers['data'].info(f"  {key}: DataFrame with {len(value)} rows, {len(value.columns)} columns")
                self.loggers['data'].info(f"    Columns: {list(value.columns)}")
                self.loggers['data'].info(f"    Data types: {dict(value.dtypes)}")
            else:
                self.loggers['data'].info(f"  {key}: {value}")
    
    def log_error(self, error_type: str, error_message: str, context: Dict = None):
        """Log errors with context"""
        self.loggers['errors'].error("="*50)
        self.loggers['errors'].error(f"ERROR: {error_type}")
        self.loggers['errors'].error(f"Timestamp: {datetime.now().isoformat()}")
        self.loggers['errors'].error(f"Message: {error_message}")
        
        if context:
            self.loggers['errors'].error("Context:")
            for key, value in context.items():
                self.loggers['errors'].error(f"  {key}: {value}")
        
        self.loggers['errors'].error("="*50)
        
        # Track in session
        self.session_data['errors_encountered'].append({
            'type': error_type,
            'message': error_message,
            'context': context,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_system_operation(self, operation: str, details: Dict):
        """Log system operations"""
        self.loggers['system'].info(f"SYSTEM OPERATION: {operation}")
        self.loggers['system'].info(f"Timestamp: {datetime.now().isoformat()}")
        
        for key, value in details.items():
            self.loggers['system'].info(f"  {key}: {value}")
    
    def log_prophet_training_details(self, prophet_model, training_data: pd.DataFrame, news_data: pd.DataFrame = None):
        """Log detailed Prophet training information"""
        self.loggers['training'].info("PROPHET MODEL TRAINING DETAILS:")
        self.loggers['training'].info("="*50)
        
        # Log Prophet configuration
        if hasattr(prophet_model, 'params'):
            self.loggers['training'].info("Prophet Configuration:")
            for param, value in prophet_model.params.items():
                self.loggers['training'].info(f"  {param}: {value}")
        
        # Log training data details
        self.loggers['training'].info(f"Training data shape: {training_data.shape}")
        self.loggers['training'].info(f"Date range: {training_data['ds'].min()} to {training_data['ds'].max()}")
        self.loggers['training'].info(f"Price range: ${training_data['y'].min():.2f} to ${training_data['y'].max():.2f}")
        
        # Log external regressors
        if news_data is not None and not news_data.empty:
            self.loggers['training'].info("News data integration:")
            self.loggers['training'].info(f"  News articles: {len(news_data)}")
            self.loggers['training'].info(f"  Date range: {news_data['date'].min()} to {news_data['date'].max()}")
            
            # Log sentiment statistics
            if 'polarity' in news_data.columns:
                self.loggers['training'].info(f"  Average polarity: {news_data['polarity'].mean():.4f}")
                self.loggers['training'].info(f"  Polarity std: {news_data['polarity'].std():.4f}")
            
            if 'subjectivity' in news_data.columns:
                self.loggers['training'].info(f"  Average subjectivity: {news_data['subjectivity'].mean():.4f}")
                self.loggers['training'].info(f"  Subjectivity std: {news_data['subjectivity'].std():.4f}")
        
        # Log seasonality components
        self.loggers['training'].info("Seasonality components:")
        if hasattr(prophet_model, 'seasonalities'):
            for seasonality in prophet_model.seasonalities:
                self.loggers['training'].info(f"  {seasonality}")
        
        self.loggers['training'].info("="*50)
    
    def log_gradient_boosting_training(self, model_type: str, features_df: pd.DataFrame, target_series: pd.Series, 
                                     model_params: Dict, performance: Dict):
        """Log gradient boosting model training details"""
        self.loggers['training'].info(f"{model_type.upper()} TRAINING DETAILS:")
        self.loggers['training'].info("="*50)
        
        # Log feature information
        self.loggers['training'].info(f"Feature matrix shape: {features_df.shape}")
        self.loggers['training'].info(f"Target series length: {len(target_series)}")
        self.loggers['training'].info(f"Feature columns: {list(features_df.columns)}")
        
        # Log feature statistics
        self.loggers['training'].info("Feature statistics:")
        for col in features_df.columns:
            if features_df[col].dtype in ['float64', 'int64']:
                self.loggers['training'].info(f"  {col}: mean={features_df[col].mean():.4f}, std={features_df[col].std():.4f}")
        
        # Log model parameters
        self.loggers['training'].info(f"Model parameters: {json.dumps(model_params, indent=2)}")
        
        # Log performance
        self.loggers['training'].info("Training performance:")
        for metric, value in performance.items():
            self.loggers['training'].info(f"  {metric}: {value}")
        
        self.loggers['training'].info("="*50)
    
    def save_daily_summary(self):
        """Save daily summary of all operations"""
        summary_file = os.path.join(self.log_dir, 'daily_summaries', f'summary_{self.today}.json')
        
        # Add session end time
        self.session_data['end_time'] = datetime.now().isoformat()
        self.session_data['duration_minutes'] = (
            datetime.fromisoformat(self.session_data['end_time']) - 
            datetime.fromisoformat(self.session_data['start_time'])
        ).total_seconds() / 60
        
        # Save summary
        with open(summary_file, 'w') as f:
            json.dump(self.session_data, f, indent=2, default=str)
        
        self.loggers['system'].info(f"Daily summary saved to: {summary_file}")
    
    def get_session_summary(self) -> Dict:
        """Get current session summary"""
        return {
            'session_id': self.current_session,
            'start_time': self.session_data['start_time'],
            'operations_count': len(self.session_data['operations']),
            'models_trained_count': len(self.session_data['models_trained']),
            'predictions_made_count': len(self.session_data['predictions_made']),
            'errors_count': len(self.session_data['errors_encountered'])
        }
