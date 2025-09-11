"""
Bitcoin Price Predictor - Source Code Package
============================================

This package contains the core application logic for the Bitcoin price prediction system.

Modules:
- app: Flask web application and API endpoints
- bitcoin_data_fetcher: Data fetching from PostgreSQL database
- bitcoin_predictor: ML models and prediction logic
- comprehensive_logger: Structured logging system
- temp_fetch_news: News data fetching utility
"""

from .app import app
from .bitcoin_data_fetcher import BitcoinDataFetcher
from .bitcoin_predictor import BitcoinPredictor
from .comprehensive_logger import ComprehensiveLogger

__all__ = ['app', 'BitcoinDataFetcher', 'BitcoinPredictor', 'ComprehensiveLogger']
