#!/bin/bash

# Script to update latest Bitcoin data and news for testing prediction model

echo "🚀 Starting Bitcoin Data and News Update Process..."
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "fetch_latest_data.py" ]; then
    echo "❌ Error: Please run this script from the app directory"
    exit 1
fi

# Check if Docker is running
if ! docker ps > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if the orchestration system is running
if ! docker ps | grep -q "postgres"; then
    echo "⚠️  Warning: PostgreSQL container not found. Make sure orchestration is running."
    echo "   Run: cd ../orchestration && docker-compose up -d"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "📊 Step 1: Fetching latest Bitcoin data (last 7 days)..."
echo "--------------------------------------------------------"

# Run the Bitcoin data fetcher
python3 fetch_latest_data.py

if [ $? -eq 0 ]; then
    echo "✅ Bitcoin data fetch completed successfully"
else
    echo "❌ Bitcoin data fetch failed"
    exit 1
fi

echo ""
echo "📰 Step 2: Fetching latest cryptocurrency news..."
echo "------------------------------------------------"

# Run the real news fetcher
python3 fetch_real_news.py

if [ $? -eq 0 ]; then
    echo "✅ News data fetch completed successfully"
else
    echo "❌ News data fetch failed"
fi

echo ""
echo "🔄 Step 3: Restarting Bitcoin Predictor to load new data..."
echo "----------------------------------------------------------"

# Restart the Bitcoin predictor to load new data
docker-compose down bitcoin-predictor
docker-compose up -d bitcoin-predictor

if [ $? -eq 0 ]; then
    echo "✅ Bitcoin Predictor restarted successfully"
else
    echo "❌ Failed to restart Bitcoin Predictor"
    exit 1
fi

echo ""
echo "🎉 Data Update Process Completed!"
echo "================================="
echo ""
echo "📈 What was updated:"
echo "  • Last 7 days of Bitcoin price data (1-minute intervals)"
echo "  • Latest cryptocurrency news from multiple sources"
echo "  • Bitcoin Predictor restarted with fresh data"
echo ""
echo "🌐 Access your updated prediction system at:"
echo "  http://localhost:5500"
echo ""
echo "💡 Tips:"
echo "  • Click 'Test Connection & Load Data' to verify new data"
echo "  • Make predictions to see the latest sentiment analysis"
echo "  • Check 'Show Last Record Details' for the most recent data"
echo ""
echo "📊 Data Sources:"
echo "  • Bitcoin: Binance API (1-minute klines)"
echo "  • News: Reddit r/cryptocurrency, CryptoPanic, NewsAPI"
echo ""

# Show some stats
echo "📊 Quick Stats:"
echo "---------------"

# Check database connection and show latest record
python3 -c "
import psycopg2
import os
from datetime import datetime

try:
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'postgres'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'airflow'),
        user=os.getenv('DB_USER', 'airflow'),
        password=os.getenv('DB_PASSWORD', 'airflow')
    )
    cursor = conn.cursor()
    
    # Get latest Bitcoin record
    cursor.execute('''
        SELECT open_time, close, volume, taker_buy_base_volume
        FROM binance_klines 
        WHERE symbol = 'BTCUSDT' 
        ORDER BY open_time DESC 
        LIMIT 1
    ''')
    latest = cursor.fetchone()
    
    if latest:
        print(f'  • Latest Bitcoin Price: \$${latest[1]:.2f}')
        print(f'  • Latest Record Time: {latest[0]}')
        print(f'  • Volume: {latest[2]:.2f} BTC')
        print(f'  • Taker Buy Volume: {latest[3]:.2f} BTC')
    
    # Count total records
    cursor.execute('SELECT COUNT(*) FROM binance_klines WHERE symbol = \\'BTCUSDT\\'')
    count = cursor.fetchone()[0]
    print(f'  • Total Bitcoin Records: {count:,}')
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f'  • Database connection failed: {e}')
"

# Check news file
if [ -f "data/cryptonews-2022-2023.csv" ]; then
    news_count=$(wc -l < data/cryptonews-2022-2023.csv)
    echo "  • Total News Articles: $((news_count - 1))"  # Subtract header
else
    echo "  • News file not found"
fi

echo ""
echo "🎯 Ready to test your prediction model with the latest data!"
