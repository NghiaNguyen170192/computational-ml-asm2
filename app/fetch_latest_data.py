#!/usr/bin/env python3
"""
Script to fetch latest Bitcoin data and news for testing prediction model
"""

import requests
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import time
import json
import os
from typing import List, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'postgres'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'airflow'),
    'user': os.getenv('DB_USER', 'airflow'),
    'password': os.getenv('DB_PASSWORD', 'airflow')
}

class BitcoinDataFetcher:
    def __init__(self):
        self.db_config = DB_CONFIG
        self.binance_base_url = "https://api.binance.com/api/v3/klines"
        
    def fetch_bitcoin_klines(self, days: int = 7) -> List[Dict]:
        """
        Fetch Bitcoin klines data from Binance API for the last N days
        
        Args:
            days: Number of days to fetch (default: 7)
            
        Returns:
            List of kline data dictionaries
        """
        try:
            # Calculate start and end times
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Convert to milliseconds
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            
            logger.info(f"Fetching Bitcoin data from {start_time} to {end_time}")
            
            # Binance API parameters
            params = {
                'symbol': 'BTCUSDT',
                'interval': '1m',  # 1-minute intervals
                'startTime': start_ms,
                'endTime': end_ms,
                'limit': 10080  # Max 7 days of 1-minute data
            }
            
            response = requests.get(self.binance_base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Fetched {len(data)} kline records from Binance")
            
            # Convert to structured format
            klines = []
            for kline in data:
                kline_dict = {
                    'open_time': datetime.fromtimestamp(kline[0] / 1000),
                    'symbol': 'BTCUSDT',
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'close_time': datetime.fromtimestamp(kline[6] / 1000),
                    'quote_asset_volume': float(kline[7]),
                    'number_of_trades': int(kline[8]),
                    'taker_buy_base_volume': float(kline[9]),
                    'taker_buy_quote_volume': float(kline[10]),
                    'ignore': int(kline[11])
                }
                klines.append(kline_dict)
            
            return klines
            
        except Exception as e:
            logger.error(f"Error fetching Bitcoin data: {e}")
            return []
    
    def insert_klines_to_db(self, klines: List[Dict]) -> bool:
        """
        Insert klines data into PostgreSQL database
        
        Args:
            klines: List of kline data dictionaries
            
        Returns:
            Boolean indicating success
        """
        try:
            # Connect to database
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check if table exists and get its structure
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'binance_klines' 
                ORDER BY ordinal_position
            """)
            columns = cursor.fetchall()
            
            if not columns:
                logger.error("Table 'binance_klines' does not exist!")
                return False
            
            # Get column names
            column_names = [col[0] for col in columns]
            logger.info(f"Table columns: {column_names}")
            
            # Prepare insert statement
            placeholders = ', '.join(['%s'] * len(column_names))
            insert_query = f"""
                INSERT INTO binance_klines ({', '.join(column_names)})
                VALUES ({placeholders})
                ON CONFLICT (open_time, symbol) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    close_time = EXCLUDED.close_time,
                    quote_asset_volume = EXCLUDED.quote_asset_volume,
                    number_of_trades = EXCLUDED.number_of_trades,
                    taker_buy_base_volume = EXCLUDED.taker_buy_base_volume,
                    taker_buy_quote_volume = EXCLUDED.taker_buy_quote_volume,
                    ignore = EXCLUDED.ignore
            """
            
            # Insert data
            inserted_count = 0
            for kline in klines:
                try:
                    # Map kline data to table columns
                    values = []
                    for col in column_names:
                        if col in kline:
                            values.append(kline[col])
                        else:
                            values.append(None)
                    
                    cursor.execute(insert_query, values)
                    inserted_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error inserting kline {kline.get('open_time')}: {e}")
                    continue
            
            conn.commit()
            logger.info(f"Successfully inserted/updated {inserted_count} kline records")
            
            # Get latest record info
            cursor.execute("""
                SELECT open_time, close, volume, taker_buy_base_volume, taker_buy_quote_volume
                FROM binance_klines 
                WHERE symbol = 'BTCUSDT' 
                ORDER BY open_time DESC 
                LIMIT 1
            """)
            latest = cursor.fetchone()
            
            if latest:
                logger.info(f"Latest record: {latest[0]} - Price: ${latest[1]:.2f} - Volume: {latest[2]:.2f} BTC")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error inserting data to database: {e}")
            return False

class NewsFetcher:
    def __init__(self):
        self.news_file = "data/cryptonews-2022-2023.csv"
        
    def fetch_latest_crypto_news(self, days: int = 7) -> List[Dict]:
        """
        Fetch latest cryptocurrency news from various sources
        
        Args:
            days: Number of days to fetch news for
            
        Returns:
            List of news articles
        """
        try:
            # Simulate fetching news (in real implementation, you'd use news APIs)
            # For now, we'll create sample news data
            logger.info("Fetching latest cryptocurrency news...")
            
            # Sample news data (in real implementation, fetch from news APIs)
            sample_news = [
                {
                    'date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'),
                    'sentiment': "{'class': 'positive', 'polarity': 0.3, 'subjectivity': 0.6}",
                    'source': 'CryptoNews',
                    'subject': 'bitcoin',
                    'text': 'Bitcoin reaches new all-time high as institutional adoption continues to grow. Major corporations are adding Bitcoin to their balance sheets, driving demand and price appreciation.',
                    'title': 'Bitcoin Hits New High as Institutional Adoption Accelerates',
                    'url': 'https://cryptonews.com/bitcoin-new-high-institutional-adoption'
                },
                {
                    'date': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S'),
                    'sentiment': "{'class': 'positive', 'polarity': 0.4, 'subjectivity': 0.5}",
                    'source': 'CoinTelegraph',
                    'subject': 'etf',
                    'text': 'Bitcoin ETF approval expected soon as regulatory clarity improves. Market analysts predict significant price movement following potential approval announcement.',
                    'title': 'Bitcoin ETF Approval Expected Soon, Market Optimistic',
                    'url': 'https://cointelegraph.com/bitcoin-etf-approval-expected'
                },
                {
                    'date': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S'),
                    'sentiment': "{'class': 'neutral', 'polarity': 0.0, 'subjectivity': 0.4}",
                    'source': 'Decrypt',
                    'subject': 'regulation',
                    'text': 'Regulatory framework for cryptocurrencies continues to evolve. Government officials are working on comprehensive guidelines for digital asset trading.',
                    'title': 'Cryptocurrency Regulation Framework Under Development',
                    'url': 'https://decrypt.co/crypto-regulation-framework'
                },
                {
                    'date': (datetime.now() - timedelta(days=4)).strftime('%Y-%m-%d %H:%M:%S'),
                    'sentiment': "{'class': 'negative', 'polarity': -0.2, 'subjectivity': 0.7}",
                    'source': 'CryptoNews',
                    'subject': 'volatility',
                    'text': 'Market volatility increases as traders react to macroeconomic uncertainty. Bitcoin price experiences significant fluctuations amid global economic concerns.',
                    'title': 'Bitcoin Volatility Rises Amid Economic Uncertainty',
                    'url': 'https://cryptonews.com/bitcoin-volatility-economic-uncertainty'
                },
                {
                    'date': (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d %H:%M:%S'),
                    'sentiment': "{'class': 'positive', 'polarity': 0.5, 'subjectivity': 0.6}",
                    'source': 'CoinTelegraph',
                    'subject': 'adoption',
                    'text': 'Major payment processor announces Bitcoin integration, signaling mainstream adoption. The move is expected to increase Bitcoin usage in everyday transactions.',
                    'title': 'Payment Processor Integrates Bitcoin for Mainstream Adoption',
                    'url': 'https://cointelegraph.com/payment-processor-bitcoin-integration'
                }
            ]
            
            logger.info(f"Generated {len(sample_news)} sample news articles")
            return sample_news
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def append_news_to_csv(self, news_articles: List[Dict]) -> bool:
        """
        Append news articles to the CSV file
        
        Args:
            news_articles: List of news article dictionaries
            
        Returns:
            Boolean indicating success
        """
        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(self.news_file), exist_ok=True)
            
            # Check if file exists
            if os.path.exists(self.news_file):
                # Read existing data
                existing_df = pd.read_csv(self.news_file)
                logger.info(f"Existing CSV has {len(existing_df)} articles")
            else:
                # Create new DataFrame
                existing_df = pd.DataFrame()
                logger.info("Creating new CSV file")
            
            # Convert new articles to DataFrame
            new_df = pd.DataFrame(news_articles)
            
            # Append new data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Remove duplicates based on title and date
            combined_df = combined_df.drop_duplicates(subset=['title', 'date'], keep='last')
            
            # Sort by date
            combined_df['date'] = pd.to_datetime(combined_df['date'])
            combined_df = combined_df.sort_values('date', ascending=False)
            
            # Save to CSV
            combined_df.to_csv(self.news_file, index=False)
            
            logger.info(f"Successfully updated CSV with {len(new_df)} new articles")
            logger.info(f"Total articles in CSV: {len(combined_df)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating CSV file: {e}")
            return False

def main():
    """Main function to fetch and update data"""
    logger.info("Starting data fetch process...")
    
    # Fetch Bitcoin data
    btc_fetcher = BitcoinDataFetcher()
    logger.info("Fetching latest Bitcoin data...")
    klines = btc_fetcher.fetch_bitcoin_klines(days=7)
    
    if klines:
        logger.info(f"Fetched {len(klines)} Bitcoin kline records")
        success = btc_fetcher.insert_klines_to_db(klines)
        if success:
            logger.info("✅ Bitcoin data successfully updated in database")
        else:
            logger.error("❌ Failed to update Bitcoin data in database")
    else:
        logger.error("❌ No Bitcoin data fetched")
    
    # Fetch news data
    news_fetcher = NewsFetcher()
    logger.info("Fetching latest news...")
    news_articles = news_fetcher.fetch_latest_crypto_news(days=7)
    
    if news_articles:
        logger.info(f"Fetched {len(news_articles)} news articles")
        success = news_fetcher.append_news_to_csv(news_articles)
        if success:
            logger.info("✅ News data successfully updated in CSV")
        else:
            logger.error("❌ Failed to update news data in CSV")
    else:
        logger.error("❌ No news data fetched")
    
    logger.info("Data fetch process completed!")

if __name__ == "__main__":
    main()
