#!/bin/bash

# Script to run data fetching inside the Docker container

echo "🚀 Fetching latest Bitcoin data and news..."
echo "=========================================="

# Check if Docker container is running
if ! docker ps | grep -q "bitcoin-predictor"; then
    echo "❌ Error: Bitcoin predictor container is not running"
    echo "   Please start the container first:"
    echo "   docker-compose up -d bitcoin-predictor"
    exit 1
fi

echo "📊 Fetching latest Bitcoin data (last 7 days)..."
echo "📰 Updating news data..."
echo ""

# Run the data fetcher inside the container
docker exec bitcoin-predictor python3 docker_fetch_data.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Data update completed successfully!"
    echo ""
    echo "🎯 What was updated:"
    echo "  • Last 7 days of Bitcoin price data (1-minute intervals)"
    echo "  • Latest cryptocurrency news for sentiment analysis"
    echo ""
    echo "🌐 Test your updated prediction system at:"
    echo "  http://localhost:5500"
    echo ""
    echo "💡 Next steps:"
    echo "  1. Go to http://localhost:5500"
    echo "  2. Click 'Test Connection & Load Data'"
    echo "  3. Make a prediction to see the latest sentiment analysis"
    echo "  4. Check 'Show Last Record Details' for the most recent data"
else
    echo ""
    echo "❌ Data update failed. Check the logs above for details."
    exit 1
fi
