#!/bin/bash

# Bitcoin Price Predictor - Docker Startup Script
# This script starts the complete Bitcoin prediction system using Docker

echo "🚀 Bitcoin Price Predictor - Docker Startup"
echo "============================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "orchestration/docker-compose.yaml" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   The script should be in the same directory as the 'orchestration' folder"
    exit 1
fi

# Function to start orchestration system
start_orchestration() {
    echo "📦 Starting orchestration system (PostgreSQL, Airflow, etc.)..."
    cd orchestration
    
    # Check if already running
    if docker-compose ps | grep -q "Up"; then
        echo "✅ Orchestration system is already running"
    else
        echo "🔄 Starting orchestration services..."
        docker-compose up -d
        
        # Wait for services to be ready
        echo "⏳ Waiting for services to start..."
        sleep 30
        
        # Check if services are running
        if docker-compose ps | grep -q "Up"; then
            echo "✅ Orchestration system started successfully"
        else
            echo "❌ Failed to start orchestration system"
            echo "💡 Try running: cd orchestration && docker-compose up -d"
            exit 1
        fi
    fi
    
    cd ..
}

# Function to start Bitcoin predictor app
start_app() {
    echo "🐳 Starting Bitcoin Price Predictor app..."
    cd app
    
    # Build and run the app
    docker-compose up --build -d bitcoin-predictor
    
    if [ $? -eq 0 ]; then
        echo "✅ Bitcoin Price Predictor started successfully!"
        echo "🌐 Access the app at: http://localhost:5500"
        echo "🔍 Health check: http://localhost:5500/health"
        echo ""
        echo "📝 Demo credentials:"
        echo "   - Username: student, Password: ml2025"
        echo "   - Username: demo, Password: password123"
        echo "   - Username: admin, Password: rmit2025"
        echo ""
        echo "📋 Useful commands:"
        echo "   View logs: cd app && docker-compose logs -f bitcoin-predictor"
        echo "   Stop app: cd app && docker-compose down"
        echo "   Restart: cd app && docker-compose restart bitcoin-predictor"
    else
        echo "❌ Failed to start Bitcoin Price Predictor"
        exit 1
    fi
    
    cd ..
}

# Main execution
main() {
    echo "🔍 Checking prerequisites..."
    
    echo ""
    echo "📋 Starting setup process..."
    echo ""
    
    # Step 1: Start orchestration
    start_orchestration
    echo ""
    
    # Step 2: Start app
    start_app
}

# Run main function
main
