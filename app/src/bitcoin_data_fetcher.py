import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Optional
import logging
import json

class BitcoinDataFetcher:
    def __init__(self, db_config: Dict = None):
        """
        Initialize Bitcoin data fetcher with PostgreSQL connection
        
        Args:
            db_config: Database configuration dictionary
        """
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 5432,
            'database': 'airflow',
            'user': 'airflow',
            'password': 'airflow'
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # News data file path
        self.news_file = os.path.join('data', 'cryptonews-2022-2023.csv')
        
    def get_connection(self):
        """Get PostgreSQL connection"""
        try:
            conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            return conn
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            raise
    
    def fetch_bitcoin_data(self, start_date: str = None, end_date: str = None, 
                          limit: int = None) -> pd.DataFrame:
        """
        Fetch Bitcoin price data from binance_klines table
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of records to fetch
            
        Returns:
            DataFrame with Bitcoin price data
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Build query
            query = """
                SELECT 
                    open_time,
                    symbol,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    taker_buy_base_volume,
                    taker_buy_quote_volume
                FROM binance_klines 
                WHERE symbol = 'BTCUSDT'
            """
            
            params = []
            if start_date:
                query += " AND open_time >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND open_time <= %s"
                params.append(end_date)
            
            query += " ORDER BY open_time ASC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            self.logger.info(f"Executing query: {query}")
            self.logger.info(f"Query parameters: {params}")
            
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            
            conn.close()
            
            self.logger.info(f"Query returned {len(data)} results")
            
            if not data:
                self.logger.warning("No Bitcoin data found in database")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=columns)
            df['open_time'] = pd.to_datetime(df['open_time'])
            df = df.sort_values('open_time')
            
            # Convert decimal columns to float to avoid numpy log issues
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_volume', 'taker_buy_quote_volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            
            # Rename columns for Prophet compatibility
            df = df.rename(columns={
                'open_time': 'ds',
                'close': 'y'
            })
            
            self.logger.info(f"Fetched {len(df)} Bitcoin price records")
            return df
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"Failed to fetch Bitcoin data: {e}")
            self.logger.error(f"FULL TRACEBACK:\n{error_details}")
            print(f"ERROR in fetch_bitcoin_data: {str(e)}")
            print(f"FULL TRACEBACK:\n{error_details}")
            return pd.DataFrame()
    
    def fetch_news_data(self) -> pd.DataFrame:
        """
        Fetch and process crypto news data from PostgreSQL crypto_news table
        
        This function:
        1. Loads news data from PostgreSQL crypto_news table
        2. Parses sentiment information from JSONB fields
        3. Categorizes news by impact type (regulatory, market, technology, etc.)
        4. Filters for Bitcoin-related content
        5. Creates time series features for Prophet integration
        
        Returns:
            DataFrame with processed news data including categories and impact scores
        """
        try:
            # Connect to database
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check if crypto_news table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'crypto_news'
                );
            """)
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                self.logger.warning("crypto_news table not found in database")
                cursor.close()
                conn.close()
                return pd.DataFrame()
            
            # Load news data from PostgreSQL table
            self.logger.info("Loading news data from PostgreSQL crypto_news table...")
            cursor.execute("""
                SELECT id, date, sentiment, source, subject, text, title, url
                FROM crypto_news 
                ORDER BY date DESC
            """)
            
            # Fetch all results
            columns = ['id', 'date', 'sentiment', 'source', 'subject', 'text', 'title', 'url']
            rows = cursor.fetchall()
            
            if not rows:
                self.logger.warning("No news data found in crypto_news table")
                cursor.close()
                conn.close()
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=columns)
            df['date'] = pd.to_datetime(df['date'])
            
            self.logger.info(f"Loaded {len(df)} news articles from database")
            
            cursor.close()
            conn.close()
            
            # Parse sentiment JSONB data from PostgreSQL
            def parse_sentiment(sentiment_data):
                """
                Parse sentiment JSONB data from PostgreSQL and extract sentiment metrics
                
                Args:
                    sentiment_data: JSONB data from PostgreSQL (already parsed as dict)
                    
                Returns:
                    Dictionary with parsed sentiment data or None if parsing fails
                """
                try:
                    if pd.isna(sentiment_data) or sentiment_data is None:
                        return None
                    
                    # If it's already a dict (from JSONB), return as is
                    if isinstance(sentiment_data, dict):
                        return sentiment_data
                    
                    # If it's a string, try to parse it
                    if isinstance(sentiment_data, str):
                        # Handle both single and double quotes
                        sentiment_str = sentiment_data.replace("'", '"')
                        return json.loads(sentiment_str)
                    
                    return None
                except Exception as e:
                    self.logger.warning(f"Failed to parse sentiment: {e}")
                    return None
            
            # Apply sentiment parsing to all rows
            df['sentiment_parsed'] = df['sentiment'].apply(parse_sentiment)
            
            # Extract sentiment metrics
            df['sentiment_class'] = df['sentiment_parsed'].apply(
                lambda x: x.get('class', 'neutral') if x else 'neutral'
            )
            df['sentiment_polarity'] = df['sentiment_parsed'].apply(
                lambda x: x.get('polarity', 0.0) if x else 0.0
            )
            df['sentiment_subjectivity'] = df['sentiment_parsed'].apply(
                lambda x: x.get('subjectivity', 0.0) if x else 0.0
            )
            
            # Categorize news by impact type for better Prophet integration
            def categorize_news_impact(row):
                """
                Categorize news articles by their potential market impact
                
                Args:
                    row: DataFrame row containing title and text
                    
                Returns:
                    String indicating news category
                """
                title = str(row['title']).lower()
                text = str(row['text']).lower()
                subject = str(row['subject']).lower()
                
                # Regulatory news (high impact)
                regulatory_keywords = ['sec', 'regulation', 'regulatory', 'compliance', 'legal', 'court', 
                                     'settlement', 'fine', 'ban', 'approval', 'etf', 'government']
                if any(keyword in title or keyword in text for keyword in regulatory_keywords):
                    return 'regulatory'
                
                # Market news (medium-high impact)
                market_keywords = ['price', 'trading', 'volume', 'market', 'bull', 'bear', 'crash', 
                                 'rally', 'pump', 'dump', 'whale', 'institutional']
                if any(keyword in title or keyword in text for keyword in market_keywords):
                    return 'market'
                
                # Technology news (medium impact)
                tech_keywords = ['blockchain', 'mining', 'hash', 'network', 'upgrade', 'fork', 
                               'wallet', 'exchange', 'defi', 'nft', 'smart contract']
                if any(keyword in title or keyword in text for keyword in tech_keywords):
                    return 'technology'
                
                # Adoption news (medium impact)
                adoption_keywords = ['adoption', 'partnership', 'integration', 'merchant', 'payment', 
                                   'bank', 'corporate', 'institution', 'mainstream']
                if any(keyword in title or keyword in text for keyword in adoption_keywords):
                    return 'adoption'
                
                # Security news (high impact)
                security_keywords = ['hack', 'breach', 'security', 'vulnerability', 'exploit', 
                                   'attack', 'stolen', 'theft', 'scam']
                if any(keyword in title or keyword in text for keyword in security_keywords):
                    return 'security'
                
                # Default category
                return 'general'
            
            # Apply categorization
            df['news_category'] = df.apply(categorize_news_impact, axis=1)
            
            # Calculate impact score based on category and sentiment
            def calculate_impact_score(row):
                """
                Calculate impact score for news articles
                
                Args:
                    row: DataFrame row with category and sentiment data
                    
                Returns:
                    Float impact score (higher = more significant impact)
                """
                # Base impact by category
                category_weights = {
                    'regulatory': 0.9,    # High impact
                    'security': 0.8,      # High impact
                    'market': 0.7,        # Medium-high impact
                    'adoption': 0.6,      # Medium impact
                    'technology': 0.5,    # Medium impact
                    'general': 0.3        # Low impact
                }
                
                # Get base weight
                base_weight = category_weights.get(row['news_category'], 0.3)
                
                # Adjust by sentiment polarity (absolute value for impact)
                sentiment_multiplier = 1 + abs(row['sentiment_polarity'])
                
                # Adjust by subjectivity (more subjective = potentially more impactful)
                subjectivity_multiplier = 1 + (row['sentiment_subjectivity'] * 0.5)
                
                return base_weight * sentiment_multiplier * subjectivity_multiplier
            
            # Calculate impact scores
            df['impact_score'] = df.apply(calculate_impact_score, axis=1)
            
            # Filter for Bitcoin-related news using keywords
            bitcoin_keywords = [
                'bitcoin', 'btc', 'crypto', 'cryptocurrency', 'blockchain', 'satoshi',
                'digital currency', 'virtual currency', 'crypto market', 'crypto trading'
            ]
            
            # Check both title and text for Bitcoin relevance
            df['is_bitcoin_related'] = (
                df['title'].str.lower().str.contains('|'.join(bitcoin_keywords), na=False) |
                df['text'].str.lower().str.contains('|'.join(bitcoin_keywords), na=False)
            )
            
            # Filter for Bitcoin-related news only
            bitcoin_news = df[df['is_bitcoin_related']].copy()
            
            # Create daily aggregated features for Prophet
            bitcoin_news['date_only'] = bitcoin_news['date'].dt.date
            
            # Aggregate daily news metrics
            daily_news = bitcoin_news.groupby('date_only').agg({
                'sentiment_polarity': ['mean', 'std', 'count'],
                'sentiment_subjectivity': 'mean',
                'impact_score': ['mean', 'sum'],
                'news_category': lambda x: x.mode().iloc[0] if len(x) > 0 else 'general'
            }).reset_index()
            
            # Flatten column names
            daily_news.columns = [
                'date', 'sentiment_polarity_mean', 'sentiment_polarity_std', 'news_count',
                'sentiment_subjectivity_mean', 'impact_score_mean', 'impact_score_sum', 'dominant_category'
            ]
            
            # Fill missing values
            daily_news['sentiment_polarity_std'] = daily_news['sentiment_polarity_std'].fillna(0)
            
            # Create additional features for Prophet
            daily_news['positive_news_ratio'] = bitcoin_news.groupby('date_only')['sentiment_class'].apply(
                lambda x: (x == 'positive').sum() / len(x) if len(x) > 0 else 0
            ).reset_index(drop=True)
            
            daily_news['negative_news_ratio'] = bitcoin_news.groupby('date_only')['sentiment_class'].apply(
                lambda x: (x == 'negative').sum() / len(x) if len(x) > 0 else 0
            ).reset_index(drop=True)
            
            # Convert date back to datetime
            daily_news['date'] = pd.to_datetime(daily_news['date'])
            
            self.logger.info(f"Loaded {len(bitcoin_news)} Bitcoin-related news articles")
            self.logger.info(f"Created {len(daily_news)} daily news aggregations")
            self.logger.info(f"News categories: {bitcoin_news['news_category'].value_counts().to_dict()}")
            
            return daily_news
            
        except Exception as e:
            self.logger.error(f"Error fetching news data from database: {e}")
            return pd.DataFrame()
    
    def get_available_date_range(self) -> Tuple[str, str]:
        """
        Get the available date range for Bitcoin data
        
        Returns:
            Tuple of (start_date, end_date) in YYYY-MM-DD format
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    MIN(open_time) as min_date,
                    MAX(open_time) as max_date
                FROM binance_klines 
                WHERE symbol = 'BTCUSDT'
            """)
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] and result[1]:
                min_date = result[0].strftime('%Y-%m-%d')
                max_date = result[1].strftime('%Y-%m-%d')
                return min_date, max_date
            
            return None, None
            
        except Exception as e:
            self.logger.error(f"Failed to get date range: {e}")
            return None, None
    
    def get_latest_price(self) -> Optional[float]:
        """
        Get the latest Bitcoin price
        
        Returns:
            Latest Bitcoin price or None if not available
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT close 
                FROM binance_klines 
                WHERE symbol = 'BTCUSDT'
                ORDER BY open_time DESC 
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return float(result[0])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get latest price: {e}")
            return None
    
    def get_price_statistics(self) -> Dict:
        """
        Get basic price statistics
        
        Returns:
            Dictionary with price statistics
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(close) as min_price,
                    MAX(close) as max_price,
                    AVG(close) as avg_price,
                    STDDEV(close) as std_price
                FROM binance_klines 
                WHERE symbol = 'BTCUSDT'
            """)
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'total_records': result[0],
                    'min_price': float(result[1]) if result[1] else 0,
                    'max_price': float(result[2]) if result[2] else 0,
                    'avg_price': float(result[3]) if result[3] else 0,
                    'std_price': float(result[4]) if result[4] else 0
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Failed to get price statistics: {e}")
            return {}
    
    def get_last_record(self) -> Optional[Dict]:
        """
        Get the last Bitcoin record from the database with details
        
        This method fetches the most recent Bitcoin price record from the binance_klines table
        and includes all relevant trading metrics to provide evidence of successful database
        connectivity and data availability.
        
        Returns:
            Dictionary with detailed last record information or None if not available
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # First, check table structure and data
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'binance_klines'
                ORDER BY ordinal_position
            """)
            columns_info = cursor.fetchall()
            self.logger.info(f"Available columns in binance_klines: {[col[0] for col in columns_info]}")
            
            cursor.execute("""
                SELECT COUNT(*) as total_records, 
                       COUNT(CASE WHEN symbol = 'BTCUSDT' THEN 1 END) as btc_records
                FROM binance_klines
            """)
            count_result = cursor.fetchone()
            self.logger.info(f"Database check - Total records: {count_result[0]}, BTC records: {count_result[1]}")
            
            # Build query based on available columns
            available_columns = [col[0] for col in columns_info]
            
            # Define the columns we want to select (only if they exist)
            select_columns = []
            if 'open_time' in available_columns:
                select_columns.append('open_time')
            if 'symbol' in available_columns:
                select_columns.append('symbol')
            if 'open' in available_columns:
                select_columns.append('open')
            if 'high' in available_columns:
                select_columns.append('high')
            if 'low' in available_columns:
                select_columns.append('low')
            if 'close' in available_columns:
                select_columns.append('close')
            if 'volume' in available_columns:
                select_columns.append('volume')
            if 'quote_asset_volume' in available_columns:
                select_columns.append('quote_asset_volume')
            if 'number_of_trades' in available_columns:
                select_columns.append('number_of_trades')
            if 'taker_buy_base_volume' in available_columns:
                select_columns.append('taker_buy_base_volume')
            if 'taker_buy_quote_volume' in available_columns:
                select_columns.append('taker_buy_quote_volume')
            if 'ignore' in available_columns:
                select_columns.append('ignore')
            
            # Build the query
            query = f"""
                SELECT {', '.join(select_columns)}
                FROM binance_klines 
                WHERE symbol = 'BTCUSDT'
                ORDER BY open_time DESC 
                LIMIT 1
            """
            
            self.logger.info(f"Executing query: {query}")
            cursor.execute(query)
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                # Create a mapping of column names to values
                column_values = dict(zip(select_columns, result))
                
                # Extract data with proper column mapping and robust type conversion
                open_time = column_values.get('open_time', datetime.now())
                close_time = open_time + timedelta(minutes=1)  # For 1-minute klines
                symbol = str(column_values.get('symbol', 'BTCUSDT'))
                
                # Safe numeric conversions with error handling
                try:
                    open_price = float(column_values.get('open', 0.0))
                except (ValueError, TypeError):
                    open_price = 0.0
                
                try:
                    high_price = float(column_values.get('high', 0.0))
                except (ValueError, TypeError):
                    high_price = 0.0
                
                try:
                    low_price = float(column_values.get('low', 0.0))
                except (ValueError, TypeError):
                    low_price = 0.0
                
                try:
                    close_price = float(column_values.get('close', 0.0))
                except (ValueError, TypeError):
                    close_price = 0.0
                
                try:
                    volume = float(column_values.get('volume', 0.0))
                except (ValueError, TypeError):
                    volume = 0.0
                
                try:
                    quote_volume = float(column_values.get('quote_asset_volume', 0.0))
                except (ValueError, TypeError):
                    quote_volume = 0.0
                
                try:
                    trades = int(column_values.get('number_of_trades', 0))
                except (ValueError, TypeError):
                    trades = 0
                
                try:
                    taker_buy_base = float(column_values.get('taker_buy_base_volume', 0.0))
                except (ValueError, TypeError):
                    taker_buy_base = 0.0
                
                try:
                    taker_buy_quote = float(column_values.get('taker_buy_quote_volume', 0.0))
                except (ValueError, TypeError):
                    taker_buy_quote = 0.0
                
                try:
                    ignore = int(column_values.get('ignore', 0))
                except (ValueError, TypeError):
                    ignore = 0
                
                # Calculate additional metrics for better analysis with error handling
                try:
                    price_change = close_price - open_price
                    price_change_pct = (price_change / open_price) * 100 if open_price > 0 else 0
                except (TypeError, ZeroDivisionError):
                    price_change = 0.0
                    price_change_pct = 0.0
                
                # Calculate trading ratios (with safe division)
                try:
                    taker_buy_ratio = (taker_buy_base / volume) * 100 if volume > 0 else 0
                except (TypeError, ZeroDivisionError):
                    taker_buy_ratio = 0.0
                
                try:
                    quote_volume_ratio = (quote_volume / volume) * 100 if volume > 0 else 0
                except (TypeError, ZeroDivisionError):
                    quote_volume_ratio = 0.0
                
                # Ensure available_columns is defined and is a list
                if 'available_columns' not in locals():
                    available_columns = []
                
                # Convert to list if it's not already
                if not isinstance(available_columns, list):
                    available_columns = list(available_columns) if available_columns else []
                
                return {
                    # Basic record information
                    'open_time': open_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(open_time, datetime) else str(open_time),
                    'close_time': close_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(close_time, datetime) else str(close_time),
                    'symbol': str(symbol),
                    
                    # Price information (ensure numeric values)
                    'open': float(open_price),
                    'high': float(high_price),
                    'low': float(low_price),
                    'close': float(close_price),
                    'price_change': round(float(price_change), 2),
                    'price_change_pct': round(float(price_change_pct), 2),
                    
                    # Volume information (ensure numeric values)
                    'volume': float(volume),
                    'quote_asset_volume': float(quote_volume) if 'quote_asset_volume' in available_columns else 'N/A',
                    'number_of_trades': int(trades) if 'number_of_trades' in available_columns else 'N/A',
                    
                    # Taker buy volumes (key metrics for analysis)
                    'taker_buy_base_volume': float(taker_buy_base) if 'taker_buy_base_volume' in available_columns else 'N/A',
                    'taker_buy_quote_volume': float(taker_buy_quote) if 'taker_buy_quote_volume' in available_columns else 'N/A',
                    
                    # Calculated ratios (ensure numeric values)
                    'taker_buy_ratio': round(float(taker_buy_ratio), 2),
                    'quote_volume_ratio': round(float(quote_volume_ratio), 2),
                    
                    # Additional metadata
                    'ignore': ignore if 'ignore' in available_columns else 'N/A',
                    'record_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    
                    # Database evidence
                    'database_evidence': {
                        'table_name': 'binance_klines',
                        'available_columns': available_columns,
                        'query_executed': query,
                        'total_records': count_result[0] if count_result else 0,
                        'btc_records': count_result[1] if count_result and len(count_result) > 1 else 0
                    }
                }
            
            return None
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"Failed to get last record: {e}")
            self.logger.error(f"FULL TRACEBACK:\n{error_details}")
            print(f"ERROR in get_last_record: {str(e)}")
            print(f"FULL TRACEBACK:\n{error_details}")
            return None
    
    def test_connection(self) -> bool:
        """
        Test database connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
