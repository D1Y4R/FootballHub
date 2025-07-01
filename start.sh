#!/bin/bash

echo "🚀 Starting Football Prediction App..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🔧 Setting up environment..."
# Load environment variables
export $(cat .env | xargs)

echo "🌐 Starting server..."
# Start the application with gunicorn
gunicorn --bind 0.0.0.0:5000 main:app --timeout 120 --workers 1