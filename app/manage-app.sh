#!/bin/bash

# Bitcoin Price Predictor - App Management Script
# This script manages the Bitcoin prediction app Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# App configuration (with environment variable fallbacks)
APP_NAME="${CONTAINER_NAME:-bitcoinpredictor}"
APP_PORT="${APP_PORT:-5000}"
APP_URL="http://localhost:${APP_PORT}"
DOCKER_NETWORK="${DOCKER_NETWORK:-orchestration_nginx-network}"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Function to check if orchestration is running
check_orchestration() {
    print_status "Checking if orchestration system is running..."
    
    if docker network ls | grep -q "${DOCKER_NETWORK}"; then
        print_success "Orchestration network found"
    else
        print_warning "Orchestration network not found. Make sure orchestration is running."
        print_status "To start orchestration: cd ../orchestration && docker-compose up -d"
    fi
}

# Function to start the app
start_app() {
    print_status "Starting Bitcoin Price Predictor app..."
    
    check_docker
    check_orchestration
    
    # Build and start the container
    docker-compose up --build -d ${APP_NAME}
    
    if [ $? -eq 0 ]; then
        print_success "App started successfully!"
        print_status "App URL: ${APP_URL}"
        print_status "Health check: ${APP_URL}/health"
        print_status ""
        print_status "Demo credentials:"
        print_status "  - Username: student, Password: ml2025"
        print_status "  - Username: demo, Password: password123"
        print_status "  - Username: admin, Password: rmit2025"
        print_status ""
        print_status "Useful commands:"
        print_status "  View logs: $0 logs"
        print_status "  Stop app: $0 stop"
        print_status "  Restart: $0 restart"
    else
        print_error "Failed to start app"
        exit 1
    fi
}

# Function to stop the app
stop_app() {
    print_status "Stopping Bitcoin Price Predictor app..."
    
    docker-compose down
    
    if [ $? -eq 0 ]; then
        print_success "App stopped successfully"
    else
        print_error "Failed to stop app"
        exit 1
    fi
}

# Function to restart the app
restart_app() {
    print_status "Restarting Bitcoin Price Predictor app..."
    
    stop_app
    sleep 2
    start_app
}

# Function to show app status
show_status() {
    print_status "Bitcoin Price Predictor Status:"
    echo ""
    
    # Check if container is running
    if docker-compose ps | grep -q "Up"; then
        print_success "App is running"
        print_status "URL: ${APP_URL}"
        
        # Check health
        if curl -s -f "${APP_URL}/health" > /dev/null 2>&1; then
            print_success "Health check: OK"
        else
            print_warning "Health check: Failed"
        fi
    else
        print_warning "App is not running"
    fi
    
    echo ""
    print_status "Container status:"
    docker-compose ps
}

# Function to show logs
show_logs() {
    print_status "Showing app logs (Press Ctrl+C to exit):"
    echo ""
    docker-compose logs -f ${APP_NAME}
}

# Function to show help
show_help() {
    echo "Bitcoin Price Predictor - App Management Script"
    echo "=============================================="
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     Start the Bitcoin prediction app"
    echo "  stop      Stop the Bitcoin prediction app"
    echo "  restart   Restart the Bitcoin prediction app"
    echo "  status    Show app status and health"
    echo "  logs      Show app logs (follow mode)"
    echo "  build     Build the app without starting"
    echo "  clean     Clean up containers and images"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start          # Start the app"
    echo "  $0 restart        # Restart the app"
    echo "  $0 logs           # View logs"
    echo "  $0 status         # Check status"
    echo ""
    echo "Prerequisites:"
    echo "  - Docker and Docker Compose installed"
    echo "  - Orchestration system running (PostgreSQL, Airflow)"
    echo ""
    echo "For orchestration management:"
    echo "  cd ../orchestration && docker-compose up -d"
}

# Function to build the app
build_app() {
    print_status "Building Bitcoin Price Predictor app..."
    
    check_docker
    
    docker-compose build ${APP_NAME}
    
    if [ $? -eq 0 ]; then
        print_success "App built successfully"
    else
        print_error "Failed to build app"
        exit 1
    fi
}

# Function to clean up
clean_app() {
    print_status "Cleaning up Bitcoin Price Predictor app..."
    
    # Stop and remove containers
    docker-compose down
    
    # Remove images
    docker rmi $(docker images -q "${APP_NAME}" 2>/dev/null) 2>/dev/null || true
    
    # Remove unused volumes
    docker volume prune -f
    
    print_success "Cleanup completed"
}

# Main script logic
case "${1:-help}" in
    start)
        start_app
        ;;
    stop)
        stop_app
        ;;
    restart)
        restart_app
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    build)
        build_app
        ;;
    clean)
        clean_app
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
