#!/bin/bash

DOMAINS=("pgadmin.local" "minio.local" "airflow.local")
NGINX_SSL_DIR="./nginx/ssl"   # relative to orchestration/
KEY_FILE="$NGINX_SSL_DIR/dev.key"
CERT_FILE="$NGINX_SSL_DIR/dev.crt"
NGINX_CONTAINER="nginx"

mkdir -p "$NGINX_SSL_DIR"

echo "Generating self-signed certificate for: ${DOMAINS[*]}..."
openssl req -x509 -nodes -days 365 \
    -newkey rsa:2048 \
    -keyout "$KEY_FILE" \
    -out "$CERT_FILE" \
    -subj "/CN=${DOMAINS[0]}" \
    -addext "subjectAltName=DNS:${DOMAINS[0]},DNS:${DOMAINS[1]},DNS:${DOMAINS[2]}"

if [[ $? -ne 0 ]]; then
    echo "Failed to generate certificate"
    exit 1
fi

echo "Self-signed certificate created at $CERT_FILE"

echo "Building Docker images without cache..."
docker build . --no-cache
if [[ $? -ne 0 ]]; then
    echo "Docker build failed"
    exit 1
fi

echo "Starting all services with docker compose..."
docker compose up -d
if [[ $? -eq 0 ]]; then
    echo "All services started successfully"
else
    echo "Failed to start services"
fi
