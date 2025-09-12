#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Bitcoin Price Prediction
===========================================================

This script generates comprehensive EDA visualizations for academic reporting:
1. Bitcoin Price Time Series Analysis
2. Volatility and Return Analysis
3. News Sentiment Distribution
4. Feature Correlation Analysis
5. Statistical Summary Visualizations

Author: RMIT ML Course Student
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for academic plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BitcoinEDA:
    def __init__(self):
        """Initialize EDA with synthetic data generation"""
        self.price_data = None
        self.news_data = None
        
    def fetch_data(self):
        """Generate synthetic data for EDA demonstration"""
        print("Generating synthetic Bitcoin data for EDA analysis...")
        self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate realistic synthetic data for EDA demonstration"""
        np.random.seed(42)
        
        # Generate 6 months of hourly Bitcoin data
        start_date = datetime.now() - timedelta(days=180)
        dates = pd.date_range(start=start_date, end=datetime.now(), freq='1H')
        n = len(dates)
        
        # Generate realistic Bitcoin price data with trends and volatility
        base_price = 45000
        trend = np.linspace(0, 0.4, n)  # 40% growth over 6 months
        volatility = np.random.normal(0, 0.015, n)  # 1.5% hourly volatility
        seasonal = 0.05 * np.sin(2 * np.pi * np.arange(n) / (24 * 7))  # Weekly seasonality
        noise = np.random.normal(0, 0.005, n)  # Random noise
        
        # Ensure all values are finite
        price_changes = trend + volatility + seasonal + noise
        price_changes = np.clip(price_changes, -0.1, 0.1)  # Limit extreme changes
        price = base_price * (1 + price_changes).cumprod()
        
        # Generate OHLCV data with realistic constraints
        price_noise = np.random.normal(0, 0.0005, n)
        high_noise = np.abs(np.random.normal(0, 0.008, n))
        low_noise = np.abs(np.random.normal(0, 0.008, n))
        
        # Ensure high >= close >= low
        high = price * (1 + high_noise)
        low = price * (1 - low_noise)
        open_price = price * (1 + price_noise)
        
        # Ensure proper OHLC relationships
        high = np.maximum(high, np.maximum(price, open_price))
        low = np.minimum(low, np.minimum(price, open_price))
        
        self.price_data = pd.DataFrame({
            'open_time': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': np.random.lognormal(12, 1.5, n),
            'price_range': high - low,
            'hourly_return': np.random.normal(0, 0.015, n),
            'price_change_pct': np.random.normal(0, 0.015, n)
        })
        
        # Generate synthetic news data
        sources = ['CoinDesk', 'CoinTelegraph', 'Bitcoin Magazine', 'CryptoNews', 'Decrypt', 'The Block']
        sentiments = ['positive', 'negative', 'neutral']
        
        news_dates = pd.date_range(start=start_date, end=datetime.now(), freq='1D')
        n_news = 200
        
        self.news_data = pd.DataFrame({
            'date': np.random.choice(news_dates, n_news),
            'sentiment': np.random.choice(sentiments, n_news, p=[0.4, 0.3, 0.3]),
            'source': np.random.choice(sources, n_news),
            'title': [f'Bitcoin Market Analysis: {i}' for i in range(n_news)],
            'text': [f'This is sample news text for article {i}' for i in range(n_news)]
        })
        
        print("Generated synthetic data for EDA demonstration")
    
    def plot_price_time_series(self):
        """Create comprehensive price time series visualization"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Bitcoin Price Over Time', 'Trading Volume', 'Hourly Returns'),
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(x=self.price_data['open_time'], y=self.price_data['close'],
                      mode='lines', name='BTC Price', line=dict(color='#f7931a', width=2)),
            row=1, col=1
        )
        
        # Volume chart
        fig.add_trace(
            go.Bar(x=self.price_data['open_time'], y=self.price_data['volume'],
                   name='Volume', marker_color='#1f77b4', opacity=0.7),
            row=2, col=1
        )
        
        # Returns chart
        fig.add_trace(
            go.Scatter(x=self.price_data['open_time'], y=self.price_data['hourly_return'],
                      mode='lines', name='Hourly Returns %', line=dict(color='#2ca02c', width=1)),
            row=3, col=1
        )
        
        fig.update_layout(
            title='Bitcoin Price Analysis: Time Series, Volume, and Returns',
            height=900,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Returns (%)", row=3, col=1)
        
        fig.write_html('app/asset-images/bitcoin_price_timeseries.html')
        fig.write_image('app/asset-images/bitcoin_price_timeseries.png', width=1200, height=900)
        print("Generated: Bitcoin Price Time Series Analysis")
    
    def plot_volatility_analysis(self):
        """Analyze Bitcoin volatility patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Bitcoin Volatility Analysis', fontsize=16, fontweight='bold')
        
        # Daily volatility calculation
        daily_data = self.price_data.set_index('open_time').resample('1D').agg({
            'close': 'last',
            'high': 'max',
            'low': 'min',
            'volume': 'sum',
            'hourly_return': 'std'
        }).dropna()
        
        daily_data['daily_return'] = daily_data['close'].pct_change() * 100
        daily_data['volatility'] = daily_data['hourly_return'] * np.sqrt(24)  # Annualized
        
        # 1. Volatility over time
        axes[0,0].plot(daily_data.index, daily_data['volatility'], color='#e74c3c', linewidth=2)
        axes[0,0].set_title('Daily Volatility Over Time', fontweight='bold')
        axes[0,0].set_ylabel('Volatility (Annualized)')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Returns distribution
        axes[0,1].hist(daily_data['daily_return'].dropna(), bins=50, alpha=0.7, color='#3498db', edgecolor='black')
        axes[0,1].axvline(daily_data['daily_return'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {daily_data["daily_return"].mean():.2f}%')
        axes[0,1].set_title('Daily Returns Distribution', fontweight='bold')
        axes[0,1].set_xlabel('Daily Returns (%)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Volume vs Volatility
        axes[1,0].scatter(daily_data['volume'], daily_data['volatility'], alpha=0.6, color='#9b59b6')
        axes[1,0].set_title('Volume vs Volatility', fontweight='bold')
        axes[1,0].set_xlabel('Daily Volume')
        axes[1,0].set_ylabel('Volatility')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Rolling volatility (30-day window)
        rolling_vol = daily_data['daily_return'].rolling(window=30).std() * np.sqrt(365)
        axes[1,1].plot(daily_data.index, rolling_vol, color='#f39c12', linewidth=2)
        axes[1,1].set_title('30-Day Rolling Volatility', fontweight='bold')
        axes[1,1].set_ylabel('Rolling Volatility (Annualized)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('app/asset-images/bitcoin_volatility_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Generated: Bitcoin Volatility Analysis")
    
    def plot_sentiment_analysis(self):
        """Analyze news sentiment patterns"""
        if self.news_data is None or len(self.news_data) == 0:
            print("No news data available for sentiment analysis")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Bitcoin News Sentiment Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sentiment distribution
        sentiment_counts = self.news_data['sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        axes[0,0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                     colors=colors, startangle=90)
        axes[0,0].set_title('Sentiment Distribution', fontweight='bold')
        
        # 2. Sentiment over time
        daily_sentiment = self.news_data.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        daily_sentiment.plot(kind='bar', stacked=True, ax=axes[0,1], color=colors)
        axes[0,1].set_title('Sentiment Over Time', fontweight='bold')
        axes[0,1].set_xlabel('Date')
        axes[0,1].set_ylabel('Number of Articles')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Source distribution
        source_counts = self.news_data['source'].value_counts().head(10)
        axes[1,0].barh(range(len(source_counts)), source_counts.values, color='#3498db')
        axes[1,0].set_yticks(range(len(source_counts)))
        axes[1,0].set_yticklabels(source_counts.index)
        axes[1,0].set_title('News Sources Distribution', fontweight='bold')
        axes[1,0].set_xlabel('Number of Articles')
        
        # 4. Sentiment by source
        sentiment_source = pd.crosstab(self.news_data['source'], self.news_data['sentiment'])
        sentiment_source.plot(kind='bar', ax=axes[1,1], color=colors)
        axes[1,1].set_title('Sentiment by Source', fontweight='bold')
        axes[1,1].set_xlabel('Source')
        axes[1,1].set_ylabel('Number of Articles')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('app/asset-images/bitcoin_sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Generated: Bitcoin Sentiment Analysis")
    
    def plot_correlation_analysis(self):
        """Analyze correlations between different features"""
        # Calculate additional features
        df = self.price_data.copy()
        df['hour'] = df['open_time'].dt.hour
        df['day_of_week'] = df['open_time'].dt.dayofweek
        df['price_ma_24h'] = df['close'].rolling(window=24).mean()
        df['volume_ma_24h'] = df['volume'].rolling(window=24).mean()
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # Select numeric columns for correlation
        numeric_cols = ['close', 'volume', 'price_range', 'hourly_return', 'hour', 'day_of_week', 
                       'price_ma_24h', 'volume_ma_24h', 'rsi']
        corr_data = df[numeric_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Bitcoin Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('app/asset-images/bitcoin_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Generated: Bitcoin Feature Correlation Analysis")
    
    def plot_statistical_summary(self):
        """Create comprehensive statistical summary"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Bitcoin Statistical Summary Analysis', fontsize=16, fontweight='bold')
        
        # 1. Price distribution
        axes[0,0].hist(self.price_data['close'], bins=50, alpha=0.7, color='#f7931a', edgecolor='black')
        axes[0,0].axvline(self.price_data['close'].mean(), color='red', linestyle='--', 
                         label=f'Mean: ${self.price_data["close"].mean():,.0f}')
        axes[0,0].set_title('Price Distribution', fontweight='bold')
        axes[0,0].set_xlabel('Price (USD)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Box plot of returns by hour
        hourly_returns = self.price_data.groupby(self.price_data['open_time'].dt.hour)['hourly_return'].apply(list)
        axes[0,1].boxplot([hourly_returns[i] for i in range(24)], labels=range(24))
        axes[0,1].set_title('Returns Distribution by Hour', fontweight='bold')
        axes[0,1].set_xlabel('Hour of Day')
        axes[0,1].set_ylabel('Hourly Returns (%)')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Volume distribution
        axes[0,2].hist(np.log(self.price_data['volume']), bins=50, alpha=0.7, color='#2ecc71', edgecolor='black')
        axes[0,2].set_title('Volume Distribution (Log Scale)', fontweight='bold')
        axes[0,2].set_xlabel('Log Volume')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Price vs Volume scatter
        axes[1,0].scatter(self.price_data['close'], self.price_data['volume'], alpha=0.5, color='#9b59b6')
        axes[1,0].set_title('Price vs Volume', fontweight='bold')
        axes[1,0].set_xlabel('Price (USD)')
        axes[1,0].set_ylabel('Volume')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Rolling statistics
        rolling_mean = self.price_data['close'].rolling(window=168).mean()  # 1 week
        rolling_std = self.price_data['close'].rolling(window=168).std()
        
        axes[1,1].plot(self.price_data['open_time'], self.price_data['close'], alpha=0.3, color='gray', label='Price')
        axes[1,1].plot(self.price_data['open_time'], rolling_mean, color='red', linewidth=2, label='7-Day MA')
        axes[1,1].fill_between(self.price_data['open_time'], 
                              rolling_mean - 2*rolling_std, 
                              rolling_mean + 2*rolling_std, 
                              alpha=0.2, color='red', label='±2σ')
        axes[1,1].set_title('Price with Rolling Statistics', fontweight='bold')
        axes[1,1].set_xlabel('Date')
        axes[1,1].set_ylabel('Price (USD)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Autocorrelation (simplified without statsmodels)
        returns = self.price_data['hourly_return'].dropna()
        autocorr = [returns.autocorr(lag=i) for i in range(1, 51)]
        
        axes[1,2].plot(autocorr, color='#e67e22', linewidth=2)
        axes[1,2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1,2].axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
        axes[1,2].axhline(y=-0.1, color='red', linestyle='--', alpha=0.5)
        axes[1,2].set_title('Returns Autocorrelation', fontweight='bold')
        axes[1,2].set_xlabel('Lag')
        axes[1,2].set_ylabel('Autocorrelation')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('app/asset-images/bitcoin_statistical_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Generated: Bitcoin Statistical Summary")
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_all_plots(self):
        """Generate all EDA visualizations"""
        print("Starting Bitcoin EDA Analysis...")
        self.fetch_data()
        
        # Create asset-images directory if it doesn't exist
        os.makedirs('app/asset-images', exist_ok=True)
        
        # Generate all plots
        self.plot_price_time_series()
        self.plot_volatility_analysis()
        self.plot_sentiment_analysis()
        self.plot_correlation_analysis()
        self.plot_statistical_summary()
        
        print("\n=== EDA Analysis Complete ===")
        print("Generated visualizations:")
        print("- Bitcoin Price Time Series Analysis")
        print("- Bitcoin Volatility Analysis")
        print("- Bitcoin Sentiment Analysis")
        print("- Bitcoin Feature Correlation Analysis")
        print("- Bitcoin Statistical Summary")
        print("\nAll plots saved to app/asset-images/")

if __name__ == "__main__":
    eda = BitcoinEDA()
    eda.generate_all_plots()
