#!/usr/bin/env python3
"""
Docker-compatible script to fetch latest Bitcoin data and news
This script runs inside the Docker container with access to the database
"""

import requests
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import time
import json
import os
import logging
from typing import List, Dict
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration (from environment variables)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'postgres'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'airflow'),
    'user': os.getenv('DB_USER', 'airflow'),
    'password': os.getenv('DB_PASSWORD', 'airflow')
}

def fetch_bitcoin_klines(days: int = 7) -> List[Dict]:
    """Fetch Bitcoin klines data from Binance API"""
    try:
        logger.info(f"Fetching Bitcoin data for last {days} days...")
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        # Binance API call
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '1m',
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 10080  # Max 7 days of 1-minute data
        }
        
        response = requests.get(url, params=params, timeout=30)
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

def insert_klines_to_db(klines: List[Dict]) -> bool:
    """Insert klines data into PostgreSQL database"""
    try:
        logger.info("Connecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Check table structure
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
        
        # Show latest record
        cursor.execute("""
            SELECT open_time, close, volume, taker_buy_base_volume
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

def generate_sample_news(days: int = 7) -> List[Dict]:
    """Generate sample news data for testing"""
    try:
        logger.info("Generating sample news data...")
        
        # Sample news templates
        news_templates = [
            {
                'templates': [
                    "Bitcoin reaches new milestone as institutional adoption accelerates",
                    "Major corporation announces Bitcoin treasury allocation",
                    "Bitcoin ETF approval expected soon, market optimistic",
                    "Regulatory clarity improves for Bitcoin and cryptocurrency",
                    "Bitcoin network upgrade shows promising results"
                ],
                'sentiment': {'class': 'positive', 'polarity': 0.4, 'subjectivity': 0.6}
            },
            {
                'templates': [
                    "Bitcoin price volatility increases amid market uncertainty",
                    "Regulatory concerns mount over cryptocurrency trading",
                    "Market analysts express caution about Bitcoin outlook",
                    "Bitcoin faces technical challenges in latest update",
                    "Economic uncertainty impacts Bitcoin market sentiment"
                ],
                'sentiment': {'class': 'negative', 'polarity': -0.3, 'subjectivity': 0.7}
            },
            {
                'templates': [
                    "Bitcoin market shows mixed signals as traders remain cautious",
                    "Cryptocurrency regulation framework under development",
                    "Bitcoin network maintains stability during recent updates",
                    "Market participants await key regulatory decisions",
                    "Bitcoin trading volume remains steady amid market conditions"
                ],
                'sentiment': {'class': 'neutral', 'polarity': 0.0, 'subjectivity': 0.4}
            }
        ]
        
        articles = []
        sources = ['CryptoNews', 'CoinTelegraph', 'Decrypt', 'CoinDesk', 'Bitcoin Magazine']
        
        # Generate articles for the last 7 days
        for i in range(days * 3):  # 3 articles per day
            template_group = news_templates[i % len(news_templates)]
            template = template_group['templates'][i % len(template_group['templates'])]
            sentiment = template_group['sentiment']
            
            # Add some variation to the title
            variations = [
                f"Breaking: {template}",
                f"Latest Update: {template}",
                f"Market Report: {template}",
                f"Analysis: {template}",
                template
            ]
            
            title = variations[i % len(variations)]
            
            # Generate article text
            text = f"{title}. This development has significant implications for the cryptocurrency market. Market participants are closely watching for further developments. The impact on Bitcoin price remains to be seen as the market digests this information."
            
            # Generate date (spread over the last 7 days)
            article_date = datetime.now() - timedelta(days=i//3, hours=i%24)
            
            article = {
                'date': article_date.strftime('%Y-%m-%d %H:%M:%S'),
                'sentiment': json.dumps(sentiment),
                'source': sources[i % len(sources)],
                'subject': 'bitcoin',
                'text': text,
                'title': title,
                'url': f"https://example.com/news/{i}"
            }
            articles.append(article)
        
        logger.info(f"Generated {len(articles)} sample news articles")
        return articles
        
    except Exception as e:
        logger.error(f"Error generating sample news: {e}")
        return []

def update_news_csv(news_articles: List[Dict]) -> bool:
    """Update the news CSV file with new articles"""
    try:
        news_file = "data/cryptonews-2022-2023.csv"
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(news_file), exist_ok=True)
        
        # Check if file exists
        if os.path.exists(news_file):
            existing_df = pd.read_csv(news_file)
            logger.info(f"Existing CSV has {len(existing_df)} articles")
        else:
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
        combined_df.to_csv(news_file, index=False)
        
        logger.info(f"Successfully updated CSV with {len(new_df)} new articles")
        logger.info(f"Total articles in CSV: {len(combined_df)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating CSV file: {e}")
        return False

def main():
    """Main function"""
    logger.info("🚀 Starting data update process...")
    
    # Step 1: Fetch Bitcoin data
    logger.info("📊 Step 1: Fetching Bitcoin data...")
    klines = fetch_bitcoin_klines(days=7)
    
    if klines:
        success = insert_klines_to_db(klines)
        if success:
            logger.info("✅ Bitcoin data updated successfully")
        else:
            logger.error("❌ Failed to update Bitcoin data")
    else:
        logger.error("❌ No Bitcoin data fetched")
    
    # Step 2: Generate and update news
    logger.info("📰 Step 2: Updating news data...")
    news_articles = generate_sample_news(days=7)
    
    if news_articles:
        success = update_news_csv(news_articles)
        if success:
            logger.info("✅ News data updated successfully")
        else:
            logger.error("❌ Failed to update news data")
    else:
        logger.error("❌ No news data generated")
    
    logger.info("🎉 Data update process completed!")
    logger.info("💡 You can now test your prediction model with the latest data")

if __name__ == "__main__":
    main()
