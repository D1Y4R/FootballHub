#!/bin/bash

echo "🚀 Starting Football Prediction App on Render..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Set production environment
export FLASK_ENV=production

# Get port from Render environment or use default
PORT=${PORT:-10000}

echo "🌐 Starting server on port $PORT..."

# Start with gunicorn
gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 main:app