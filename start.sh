#!/bin/bash

echo "🚀 Starting Football Prediction App for CodeSandbox..."

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found. Please make sure you're in the correct directory."
    exit 1
fi

# Try to install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt --user --break-system-packages 2>/dev/null || {
        echo "⚠️  Pip install failed, trying alternative method..."
        pip install --user flask gunicorn requests 2>/dev/null || {
            echo "⚠️  Using system packages..."
        }
    }
else
    echo "⚠️  requirements.txt not found, trying to install basic packages..."
    pip install --user flask gunicorn requests 2>/dev/null
fi

echo "🔧 Loading environment variables..."
# Load environment variables from .env if it exists
if [ -f ".env" ]; then
    export $(cat .env | xargs) 2>/dev/null
    echo "✅ Environment variables loaded from .env"
else
    echo "❌ .env file not found"
fi

echo "🌐 Starting server..."

# Try different methods to start the server
if command -v gunicorn >/dev/null 2>&1; then
    echo "Using gunicorn..."
    gunicorn --bind 0.0.0.0:5000 main:app --timeout 120 --workers 1 --preload
elif command -v python3 >/dev/null 2>&1; then
    echo "Using python3 directly..."
    python3 run.py
else
    echo "Using python directly..."
    python run.py
fi