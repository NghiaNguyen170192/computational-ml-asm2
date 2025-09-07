#!/bin/bash

# === CONFIGURATION ===
DOMAIN="yourdomain"            # Replace with your domain
EMAIL="you@example.com"            # Replace with your email
WEBROOT_PATH="/var/www/certbot"    # Must match the volume mount
NGINX_CONTAINER="nginx"
CERTBOT_CONTAINER="certbot"

# === STEP 1: Ensure Nginx is running ===
echo "🔄 Starting Nginx container..."
docker compose up -d $NGINX_CONTAINER

# === STEP 2: Request SSL certificate ===
echo "🔐 Requesting SSL certificate for $DOMAIN..."
docker exec $CERTBOT_CONTAINER certbot certonly \
    --webroot -w $WEBROOT_PATH \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    -d $DOMAIN

# === STEP 3: Reload Nginx to apply new certificate ===
echo "🔁 Reloading Nginx to apply certificate..."
docker exec $NGINX_CONTAINER nginx -s reload

echo "✅ SSL certificate generation complete!"