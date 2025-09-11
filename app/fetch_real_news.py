#!/usr/bin/env python3
"""
Script to fetch real cryptocurrency news from various APIs
"""

import requests
import pandas as pd
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

class RealNewsFetcher:
    def __init__(self):
        self.news_file = "data/cryptonews-2022-2023.csv"
        
    def fetch_news_from_reddit(self, subreddit: str = "cryptocurrency", limit: int = 50) -> List[Dict]:
        """
        Fetch news from Reddit r/cryptocurrency
        
        Args:
            subreddit: Subreddit to fetch from
            limit: Number of posts to fetch
            
        Returns:
            List of news articles
        """
        try:
            logger.info(f"Fetching news from r/{subreddit}...")
            
            # Reddit API endpoint (no auth required for public data)
            url = f"https://www.reddit.com/r/{subreddit}/hot.json"
            headers = {
                'User-Agent': 'BitcoinPredictor/1.0 (Educational Purpose)'
            }
            
            params = {
                'limit': limit,
                't': 'week'  # Last week
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            posts = data['data']['children']
            
            articles = []
            for post in posts:
                post_data = post['data']
                
                # Extract sentiment from title and selftext
                title = post_data.get('title', '')
                selftext = post_data.get('selftext', '')
                full_text = f"{title} {selftext}"
                
                # Simple sentiment analysis
                sentiment = self._analyze_sentiment(full_text)
                
                article = {
                    'date': datetime.fromtimestamp(post_data.get('created_utc', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                    'sentiment': json.dumps(sentiment),
                    'source': 'Reddit',
                    'subject': 'cryptocurrency',
                    'text': full_text[:500],  # Limit text length
                    'title': title,
                    'url': f"https://reddit.com{post_data.get('permalink', '')}"
                }
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from Reddit")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching Reddit news: {e}")
            return []
    
    def fetch_news_from_cryptopanic(self, limit: int = 50) -> List[Dict]:
        """
        Fetch news from CryptoPanic API (free tier)
        
        Args:
            limit: Number of articles to fetch
            
        Returns:
            List of news articles
        """
        try:
            logger.info("Fetching news from CryptoPanic...")
            
            # CryptoPanic API (free tier)
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                'auth_token': 'free',  # Free tier
                'currencies': 'BTC',
                'public': 'true',
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            posts = data.get('results', [])
            
            articles = []
            for post in posts:
                title = post.get('title', '')
                domain = post.get('domain', 'CryptoPanic')
                url_link = post.get('url', '')
                created_at = post.get('created_at', '')
                
                # Extract sentiment from title
                sentiment = self._analyze_sentiment(title)
                
                article = {
                    'date': created_at,
                    'sentiment': json.dumps(sentiment),
                    'source': domain,
                    'subject': 'bitcoin',
                    'text': title,  # CryptoPanic doesn't provide full text
                    'title': title,
                    'url': url_link
                }
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from CryptoPanic")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching CryptoPanic news: {e}")
            return []
    
    def fetch_news_from_newsapi(self, api_key: str = None) -> List[Dict]:
        """
        Fetch news from NewsAPI (requires API key)
        
        Args:
            api_key: NewsAPI key (optional)
            
        Returns:
            List of news articles
        """
        if not api_key:
            logger.warning("No NewsAPI key provided, skipping NewsAPI")
            return []
        
        try:
            logger.info("Fetching news from NewsAPI...")
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'apiKey': api_key,
                'q': 'bitcoin OR cryptocurrency OR blockchain',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 50,
                'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            articles_data = data.get('articles', [])
            
            articles = []
            for article_data in articles_data:
                title = article_data.get('title', '')
                description = article_data.get('description', '')
                content = article_data.get('content', '')
                full_text = f"{title} {description} {content or ''}"
                
                # Extract sentiment
                sentiment = self._analyze_sentiment(full_text)
                
                article = {
                    'date': article_data.get('publishedAt', ''),
                    'sentiment': json.dumps(sentiment),
                    'source': article_data.get('source', {}).get('name', 'NewsAPI'),
                    'subject': 'bitcoin',
                    'text': full_text[:500],
                    'title': title,
                    'url': article_data.get('url', '')
                }
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from NewsAPI")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching NewsAPI news: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """
        Simple sentiment analysis based on keywords
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis
        """
        text_lower = text.lower()
        
        # Positive keywords
        positive_words = [
            'bullish', 'surge', 'rally', 'growth', 'adoption', 'institutional',
            'etf', 'approval', 'breakthrough', 'milestone', 'partnership',
            'investment', 'positive', 'mainstream', 'acceptance', 'innovation',
            'up', 'rise', 'gain', 'profit', 'success', 'win', 'breakthrough'
        ]
        
        # Negative keywords
        negative_words = [
            'bearish', 'crash', 'decline', 'drop', 'fall', 'loss', 'risk',
            'volatility', 'uncertainty', 'fear', 'concern', 'warning',
            'regulation', 'ban', 'restriction', 'hack', 'security', 'fraud',
            'scam', 'caution', 'down', 'sell', 'correction', 'crash'
        ]
        
        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate polarity (-1 to 1)
        total_words = positive_count + negative_count
        if total_words == 0:
            polarity = 0.0
        else:
            polarity = (positive_count - negative_count) / total_words
        
        # Determine class
        if polarity > 0.1:
            sentiment_class = 'positive'
        elif polarity < -0.1:
            sentiment_class = 'negative'
        else:
            sentiment_class = 'neutral'
        
        # Calculate subjectivity (0 to 1)
        subjectivity = min(1.0, total_words / 10.0)  # More keywords = more subjective
        
        return {
            'class': sentiment_class,
            'polarity': round(polarity, 3),
            'subjectivity': round(subjectivity, 3)
        }
    
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
    """Main function to fetch real news"""
    logger.info("Starting real news fetch process...")
    
    news_fetcher = RealNewsFetcher()
    all_articles = []
    
    # Fetch from Reddit
    logger.info("Fetching from Reddit...")
    reddit_articles = news_fetcher.fetch_news_from_reddit(limit=30)
    all_articles.extend(reddit_articles)
    
    # Fetch from CryptoPanic
    logger.info("Fetching from CryptoPanic...")
    cryptopanic_articles = news_fetcher.fetch_news_from_cryptopanic(limit=30)
    all_articles.extend(cryptopanic_articles)
    
    # Fetch from NewsAPI (if key provided)
    newsapi_key = os.getenv('NEWSAPI_KEY')
    if newsapi_key:
        logger.info("Fetching from NewsAPI...")
        newsapi_articles = news_fetcher.fetch_news_from_newsapi(newsapi_key)
        all_articles.extend(newsapi_articles)
    else:
        logger.info("No NewsAPI key provided, skipping NewsAPI")
    
    # Remove duplicates
    unique_articles = []
    seen_titles = set()
    for article in all_articles:
        if article['title'] not in seen_titles:
            unique_articles.append(article)
            seen_titles.add(article['title'])
    
    logger.info(f"Total unique articles: {len(unique_articles)}")
    
    # Append to CSV
    if unique_articles:
        success = news_fetcher.append_news_to_csv(unique_articles)
        if success:
            logger.info("✅ Real news data successfully updated in CSV")
        else:
            logger.error("❌ Failed to update news data in CSV")
    else:
        logger.error("❌ No news articles fetched")
    
    logger.info("Real news fetch process completed!")

if __name__ == "__main__":
    main()
