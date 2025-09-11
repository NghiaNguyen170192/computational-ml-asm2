#!/usr/bin/env python3
"""
Bitcoin Price Predictor - Flask Web Application
===============================================

A Bitcoin price prediction system that uses multiple ML models and real-time data.

SYSTEM OVERVIEW:
- Multi-model ensemble approach (Prophet, XGBoost, LightGBM, Statistical)
- Real-time news sentiment analysis integration
- Interactive web interface with Plotly visualizations
- Automated model retraining with drift detection
- Logging and monitoring system
- Production-ready error handling and security

ARCHITECTURE:
- Frontend: HTML5/CSS3/JavaScript with Bootstrap and Plotly
- Backend: Flask web application with RESTful API
- ML Layer: Multiple ML models with ensemble prediction
- Data Layer: PostgreSQL database with real-time Bitcoin data
- News Layer: PostgreSQL-based news sentiment analysis (table `crypto_news`)

AUTHOR: RMIT ML Course Student
DATE: September 2025
VERSION: 2.0.0
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from datetime import datetime, timedelta
import json
import os
import threading
import numpy as np
from .bitcoin_data_fetcher import BitcoinDataFetcher
from .bitcoin_predictor import BitcoinPredictor
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.utils
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# FLASK APPLICATION INITIALIZATION
# =============================================================================

# Initialize Flask application with security configurations
# Set template and static folders relative to the app directory (not src/)
app = Flask(__name__, 
           template_folder='../templates',
           static_folder='../static')
app.secret_key = 'rmit_ml_course_demo_key_2025'  # In production, use environment variable

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Database configuration from environment variables
# This allows for flexible deployment across different environments
# The configuration connects to the orchestration PostgreSQL database
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),           # Database host (Docker container name)
    'port': int(os.getenv('DB_PORT', 5432)),             # PostgreSQL default port
    'database': os.getenv('DB_NAME', 'airflow'),         # Database name from orchestration
    'user': os.getenv('DB_USER', 'airflow'),             # Database user credentials
    'password': os.getenv('DB_PASSWORD', 'airflow')      # Database password
}

# =============================================================================
# CORE SYSTEM COMPONENTS INITIALIZATION
# =============================================================================

# Initialize core system components
# These components handle the main business logic of the application
data_fetcher = BitcoinDataFetcher(DB_CONFIG)  # Handles database operations and data fetching
predictor = BitcoinPredictor()                # Manages ML models and predictions

# =============================================================================
# AUTHENTICATION AND SECURITY
# =============================================================================

# Demo credentials for academic demonstration
# In production, these would be stored securely in a database with hashed passwords
DEMO_USERS = {
    'student': 'ml2025',      # Student account for academic use
    'demo': 'password123',    # Demo account for testing
    'admin': 'rmit2025'       # Admin account for system management
}

# =============================================================================
# GLOBAL STATE MANAGEMENT
# =============================================================================

# Global progress tracking for long-running operations
# This allows the UI to show real-time progress updates
progress_data = {'current': 0, 'status': 'idle', 'message': ''}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def update_progress(progress, message):
    """
    Update global progress tracking for long-running operations
    
    This function is used to provide real-time feedback to the user interface
    during operations that take time to complete, such as model training or
    data fetching.
    
    Args:
        progress (int): Progress percentage (0-100)
        message (str): Status message to display to user
    """
    global progress_data
    progress_data['current'] = progress
    progress_data['message'] = message

# =============================================================================
# ROUTE HANDLERS - AUTHENTICATION
# =============================================================================

@app.route('/')
def home():
    """
    Homepage route - displays login form and demo credentials
    
    This is the entry point of the application. It provides:
    - User authentication form
    - Demo credentials for testing
    - System information and instructions
    
    Returns:
        HTML template: index.html with demo user credentials
    """
    return render_template('index.html', demo_users=DEMO_USERS)

@app.route('/login', methods=['POST'])
def login():
    """
    Handle user authentication and session management
    
    This function processes login requests and validates user credentials.
    It implements basic authentication for the demo system.
    
    Security considerations:
    - In production, passwords should be hashed and stored securely
    - Session management should include timeout and CSRF protection
    - Input validation could be improved
    
    Returns:
        Redirect: To dashboard on success, back to home on failure
    """
    username = request.form['username']
    password = request.form['password']
    
    # Validate credentials against demo user database
    if username in DEMO_USERS and DEMO_USERS[username] == password:
        session['logged_in'] = True
        session['username'] = username
        flash(f'Welcome, {username}!', 'success')
        return redirect(url_for('dashboard'))
    else:
        flash('Invalid credentials', 'error')
        return redirect(url_for('home'))

@app.route('/logout')
def logout():
    """Handle logout"""
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    """Main dashboard"""
    if not session.get('logged_in'):
        return redirect(url_for('home'))
    
    # Test database connection
    db_connected = data_fetcher.test_connection()
    
    # Get available date range
    min_date, max_date = data_fetcher.get_available_date_range()
    
    # Get latest Bitcoin price
    latest_price = data_fetcher.get_latest_price()
    
    # Get price statistics
    price_stats = data_fetcher.get_price_statistics()
    
    return render_template('dashboard.html', 
                         db_connected=db_connected,
                         min_date=min_date,
                         max_date=max_date,
                         latest_price=latest_price,
                         price_stats=price_stats,
                         username=session['username'])

@app.route('/predict', methods=['POST'])
def predict():
    """Handle Bitcoin prediction request"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        prediction_date = request.json['prediction_date']
        days_ahead = int(request.json['days_ahead'])
        
        # Check if we have a trained model
        if not predictor.load_model():
            # Train model first - fetch Bitcoin data
            try:
                # Fetch Bitcoin price data
                price_data = data_fetcher.fetch_bitcoin_data()
                if price_data.empty:
                    return jsonify({'error': 'No Bitcoin data available. Please ensure the database is running and has data.'}), 400
                
                # Fetch news data
                news_data = data_fetcher.fetch_news_data()
                
                # Train model
                train_result = predictor.train(price_data, news_data)
                if train_result['status'] != 'success':
                    return jsonify({'error': 'Failed to train model'}), 500
                    
            except Exception as e:
                return jsonify({'error': f'Failed to fetch data: {str(e)}'}), 400
        
        # Fetch news data for prediction period
        news_data = data_fetcher.fetch_news_data()
        
        # Make prediction
        result = predictor.predict(prediction_date, days_ahead, news_data)
        
        # Check if we can calculate RMSE (if prediction date is in the past)
        pred_date = datetime.strptime(prediction_date, '%Y-%m-%d')
        end_date = pred_date + timedelta(days=days_ahead-1)
        today = datetime.now()
        
        rmse = None
        if end_date < today:
            # We can calculate RMSE
            price_data = data_fetcher.fetch_bitcoin_data()
            if not price_data.empty:
                eval_result = predictor.evaluate_model(
                    price_data, 
                    prediction_date, 
                    end_date.strftime('%Y-%m-%d')
                )
                if 'rmse' in eval_result:
                    rmse = eval_result['rmse']
        
        result['rmse'] = rmse
        return jsonify(result)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in predict: {str(e)}")
        print(f"FULL TRACEBACK:\n{error_details}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain Bitcoin model"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        # Fetch fresh Bitcoin data
        price_data = data_fetcher.fetch_bitcoin_data()
        if price_data.empty:
            return jsonify({'error': 'No Bitcoin data available'}), 400
        
        # Fetch news data
        news_data = data_fetcher.fetch_news_data()
        
        # Retrain model
        result = predictor.train(price_data, news_data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fetch_data', methods=['POST'])
def fetch_data():
    """Test database connection and fetch latest Bitcoin data"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401
    
    def fetch_in_background():
        global progress_data
        progress_data['status'] = 'fetching'
        progress_data['current'] = 0
        progress_data['message'] = 'Testing database connection...'
        
        try:
            # Test database connection
            if not data_fetcher.test_connection():
                progress_data['status'] = 'error'
                progress_data['message'] = 'Database connection failed. Please ensure PostgreSQL is running.'
                return
            
            progress_data['current'] = 50
            progress_data['message'] = 'Fetching Bitcoin data...'
            
            # Fetch latest data
            price_data = data_fetcher.fetch_bitcoin_data()
            if price_data.empty:
                progress_data['status'] = 'error'
                progress_data['message'] = 'No Bitcoin data found in database. Please run the orchestration system first.'
                return
            
            # Get the last record details with comprehensive information
            last_record = data_fetcher.get_last_record()
            if last_record:
                # Create detailed last record summary
                last_record_info = f"""
                Last Record Evidence:
                • Time: {last_record['open_time']} to {last_record['close_time']}
                • Price: ${last_record['open']:.2f} → ${last_record['close']:.2f} ({last_record['price_change_pct']:+.2f}%)
                • Volume: {last_record['volume']:,.2f} BTC
                • Taker Buy Base Volume: {last_record['taker_buy_base_volume']:,.2f} BTC
                • Taker Buy Quote Volume: {last_record['taker_buy_quote_volume']:,.2f} USDT
                • Trades: {last_record['number_of_trades']:,} transactions
                • Buy Ratio: {last_record['taker_buy_ratio']:.1f}%
                """
                progress_data['status'] = 'completed'
                progress_data['message'] = f'Data fetching completed. Found {len(price_data)} records.\n{last_record_info}'
            else:
                progress_data['status'] = 'completed'
                progress_data['message'] = f'Data fetching completed. Found {len(price_data)} records.'
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"ERROR in fetch_data: {str(e)}")
            print(f"FULL TRACEBACK:\n{error_details}")
            progress_data['status'] = 'error'
            progress_data['message'] = f'Error: {str(e)}'
    
    # Start background task
    thread = threading.Thread(target=fetch_in_background)
    thread.start()
    
    return jsonify({'status': 'started', 'message': 'Data fetching started'})

@app.route('/delete_data', methods=['POST'])
def delete_data():
    """Clear model and logs"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        # Clear prediction logs
        predictor.clear_prediction_logs()
        
        # Clear model files
        import shutil
        if os.path.exists('models'):
            shutil.rmtree('models')
        os.makedirs('models', exist_ok=True)
        
        return jsonify({'status': 'success', 'message': 'Model and logs cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/progress')
def get_progress():
    """Get current progress"""
    return jsonify(progress_data)

@app.route('/chart_data')
def chart_data():
    """
    Get Bitcoin price data for charts with proper time range handling
    
    This endpoint handles different time range requests:
    - All: Returns all available data
    - 6 months: Last 6 months of data
    - 3 months: Last 3 months of data
    - 1 day: Last 24 hours with higher resolution
    - 10 minutes: Last 10 minutes with minute-level data
    
    Query Parameters:
    - start_date: Start date filter (YYYY-MM-DD)
    - end_date: End date filter (YYYY-MM-DD)
    - limit: Maximum number of records to return
    
    Returns:
        JSON with dates, prices, and metadata for chart rendering
    """
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401
    
    # Get date range from query parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    limit = request.args.get('limit', type=int)
    
    try:
        # Fetch Bitcoin price data with specified filters
        data = data_fetcher.fetch_bitcoin_data(start_date, end_date, limit)
        
        if data.empty:
            return jsonify({'error': 'No Bitcoin data available'}), 400
        
        # Sort data by date to ensure proper time series order
        data = data.sort_values('ds')
        
        # Calculate additional metrics for chart display
        price_min = data['y'].min()
        price_max = data['y'].max()
        price_mean = data['y'].mean()
        price_std = data['y'].std()
        
        # Calculate price changes for trend analysis
        data['price_change'] = data['y'].pct_change().fillna(0)
        data['price_change_abs'] = data['y'].diff().fillna(0)
        
        # Clean data to remove NaN and infinite values
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Ensure all numeric values are finite
        data['y'] = data['y'].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
        data['price_change'] = data['price_change'].replace([np.inf, -np.inf], np.nan).fillna(0)
        data['price_change_abs'] = data['price_change_abs'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Recalculate statistics after cleaning
        price_min = data['y'].min()
        price_max = data['y'].max()
        price_mean = data['y'].mean()
        price_std = data['y'].std()
        
        # Handle any remaining NaN values in statistics
        if np.isnan(price_min) or np.isinf(price_min):
            price_min = 0.0
        if np.isnan(price_max) or np.isinf(price_max):
            price_max = 0.0
        if np.isnan(price_mean) or np.isinf(price_mean):
            price_mean = 0.0
        if np.isnan(price_std) or np.isinf(price_std):
            price_std = 0.0
        
        # Prepare data for chart with enhanced information
        chart_data = {
            'dates': data['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'prices': data['y'].fillna(0).tolist(),
            'price_changes': data['price_change'].fillna(0).tolist(),
            'price_changes_abs': data['price_change_abs'].fillna(0).tolist(),
            'symbol': 'BTCUSDT',
            'metadata': {
                'min_price': float(price_min),
                'max_price': float(price_max),
                'mean_price': float(price_mean),
                'std_price': float(price_std),
                'data_points': len(data),
                'date_range': {
                    'start': data['ds'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': data['ds'].max().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
        }
        
        return jsonify(chart_data)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in chart_data: {str(e)}")
        print(f"FULL TRACEBACK:\n{error_details}")
        return jsonify({'error': str(e)}), 500

@app.route('/drift_detection_status')
def drift_detection_status():
    """
    Get current drift detection status and configuration
    
    Returns:
        JSON with drift detection configuration and status
    """
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        status = predictor.get_drift_detection_status()
        return jsonify(status)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in drift_detection_status: {str(e)}")
        print(f"FULL TRACEBACK:\n{error_details}")
        return jsonify({'error': str(e)}), 500

@app.route('/check_drift', methods=['POST'])
def check_drift():
    """
    Check for data drift in recent Bitcoin data
    
    Returns:
        JSON with drift detection results and recommendations
    """
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        # Get recent data for drift detection (last 7 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Fetch recent data
        recent_data = data_fetcher.fetch_bitcoin_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if recent_data.empty:
            return jsonify({'error': 'No recent data available for drift detection'}), 400
        
        # Check for drift
        drift_results = predictor.detect_data_drift(recent_data)
        
        return jsonify(drift_results)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in check_drift: {str(e)}")
        print(f"FULL TRACEBACK:\n{error_details}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrain_with_drift_detection', methods=['POST'])
def retrain_with_drift_detection():
    """
    Retrain model with automatic drift detection
    
    Returns:
        JSON with retraining results and drift information
    """
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        # Fetch fresh data for retraining
        data = data_fetcher.fetch_bitcoin_data()
        news_data = data_fetcher.fetch_news_data()
        
        if data.empty:
            return jsonify({'error': 'No Bitcoin data available for retraining'}), 400
        
        # Retrain with drift detection
        retrain_results = predictor.retrain_with_drift_detection(data, news_data)
        
        return jsonify(retrain_results)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in retrain_with_drift_detection: {str(e)}")
        print(f"FULL TRACEBACK:\n{error_details}")
        return jsonify({'error': str(e)}), 500

@app.route('/last_record_details')
def last_record_details():
    """
    Get detailed information about the last Bitcoin record from the database
    
    This endpoint provides comprehensive evidence of database connectivity and data
    availability, including all trading metrics from the binance_klines table.
    Perfect for demonstrating that the system is properly connected
    to the PostgreSQL database and has access to real Bitcoin trading data.
    
    Returns:
        JSON with detailed last record information including:
        - Price data (open, high, low, close)
        - Volume metrics (base and quote volumes)
        - Trading activity (number of trades, taker buy ratios)
        - Calculated metrics (price changes, volume ratios)
    """
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        print("DEBUG: Starting last_record_details function")
        
        # Get comprehensive last record details
        print("DEBUG: Calling data_fetcher.get_last_record()")
        last_record = data_fetcher.get_last_record()
        print(f"DEBUG: get_last_record() returned: {type(last_record)} - {last_record is not None}")
        
        if not last_record:
            print("DEBUG: last_record is None or empty")
            return jsonify({'error': 'No Bitcoin data available'}), 400
        
        print("DEBUG: last_record retrieved successfully, building analysis")
        print(f"DEBUG: last_record keys: {list(last_record.keys()) if isinstance(last_record, dict) else 'Not a dict'}")
        
        # Add additional context for better understanding
        print("DEBUG: Building database_evidence section")
        record_analysis = {
            'database_evidence': {
                'connection_status': 'SUCCESS',
                'table_accessed': 'binance_klines',
                'symbol': 'BTCUSDT',
                'record_retrieved': True,
                'timestamp': last_record['record_timestamp']
            },
        }
        
        print("DEBUG: Building trading_data section")
        record_analysis['trading_data'] = {
            'time_period': {
                'open_time': last_record['open_time'],
                'close_time': last_record['close_time'],
                'duration_minutes': '1 minute (standard kline interval)'
            },
        }
        
        print("DEBUG: Building price_movement section")
        record_analysis['trading_data']['price_movement'] = {
            'open_price': last_record['open'],
            'high_price': last_record['high'],
            'low_price': last_record['low'],
            'close_price': last_record['close'],
            'price_change': last_record['price_change'],
            'price_change_percentage': last_record['price_change_pct']
        }
        
        print("DEBUG: Building volume_metrics section")
        record_analysis['trading_data']['volume_metrics'] = {
            'total_volume_btc': last_record['volume'],
            'quote_asset_volume_usdt': last_record['quote_asset_volume'],
            'taker_buy_base_volume_btc': last_record['taker_buy_base_volume'],
            'taker_buy_quote_volume_usdt': last_record['taker_buy_quote_volume']
        }
        
        print("DEBUG: Building trading_activity section")
        record_analysis['trading_data']['trading_activity'] = {
            'number_of_trades': last_record['number_of_trades'],
            'taker_buy_ratio_percentage': last_record['taker_buy_ratio'],
            'quote_volume_ratio_percentage': last_record['quote_volume_ratio']
        }
        
        print("DEBUG: Building analysis_metrics section")
        # Check each value before comparison
        taker_buy_ratio = last_record.get('taker_buy_ratio', 0)
        number_of_trades = last_record.get('number_of_trades', 0)
        price_change_pct = last_record.get('price_change_pct', 0)
        volume = last_record.get('volume', 0)
        
        print(f"DEBUG: taker_buy_ratio type: {type(taker_buy_ratio)}, value: {taker_buy_ratio}")
        print(f"DEBUG: number_of_trades type: {type(number_of_trades)}, value: {number_of_trades}")
        print(f"DEBUG: price_change_pct type: {type(price_change_pct)}, value: {price_change_pct}")
        print(f"DEBUG: volume type: {type(volume)}, value: {volume}")
        
        # Convert to appropriate types for comparison
        try:
            taker_buy_ratio_num = float(taker_buy_ratio) if taker_buy_ratio != 'N/A' else 0
        except (ValueError, TypeError):
            taker_buy_ratio_num = 0
            
        try:
            number_of_trades_num = int(number_of_trades) if number_of_trades != 'N/A' else 0
        except (ValueError, TypeError):
            number_of_trades_num = 0
            
        try:
            price_change_pct_num = float(price_change_pct) if price_change_pct != 'N/A' else 0
        except (ValueError, TypeError):
            price_change_pct_num = 0
            
        try:
            volume_num = float(volume) if volume != 'N/A' else 0
        except (ValueError, TypeError):
            volume_num = 0
        
        record_analysis['analysis_metrics'] = {
            'buying_pressure': 'High' if taker_buy_ratio_num > 50 else 'Low',
            'trading_intensity': 'High' if number_of_trades_num > 1000 else 'Moderate',
            'price_volatility': 'High' if abs(price_change_pct_num) > 2 else 'Low',
            'volume_significance': 'High' if volume_num > 100 else 'Moderate'
        }
        
        print("DEBUG: Building system_info section")
        record_analysis['system_info'] = {
            'data_source': 'Binance API via Airflow DAG',
            'database': 'PostgreSQL (orchestration system)',
            'table_structure': 'Standard Binance klines format',
            'update_frequency': 'Every minute (real-time)',
            'data_quality': 'High - includes all trading metrics'
        }
        
        print("DEBUG: Building final response")
        return jsonify({
            'status': 'success',
            'last_record': last_record,
            'analysis': record_analysis,
            'message': 'Last record details retrieved successfully - Database connection verified'
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in last_record_details: {str(e)}")
        print(f"ERROR TYPE: {type(e).__name__}")
        print(f"FULL TRACEBACK:\n{error_details}")
        
        # Enhanced error response with more details
        return jsonify({
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': error_details,
            'debug_info': {
                'function': 'last_record_details',
                'line_causing_error': 'See traceback for exact line',
                'last_record_type': str(type(last_record)) if 'last_record' in locals() else 'Not defined',
                'last_record_keys': list(last_record.keys()) if 'last_record' in locals() and isinstance(last_record, dict) else 'Not a dict or not defined'
            }
        }), 500

@app.route('/prediction_logs')
def prediction_logs():
    """Get prediction logs"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401
    
    logs = predictor.get_prediction_logs()
    return jsonify({'logs': logs})

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    """Clear prediction logs"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401
    
    predictor.clear_prediction_logs()
    return jsonify({'status': 'success', 'message': 'Logs cleared'})

@app.route('/health')
def health_check():
    """Health check endpoint for Docker"""
    try:
        # Test database connection
        db_connected = data_fetcher.test_connection()
        return jsonify({
            'status': 'healthy' if db_connected else 'unhealthy',
            'database': 'connected' if db_connected else 'disconnected',
            'timestamp': datetime.now().isoformat()
        }), 200 if db_connected else 503
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)