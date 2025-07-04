#!/bin/bash

# Football Predictor - CodeSandbox Optimized Startup Script
# Handles CPU optimization and resource management

echo "🚀 Football Predictor - CodeSandbox Startup"
echo "============================================"

# Set environment variables for CodeSandbox
export PYTHON_ENV=codesandbox
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# CPU optimization settings
export GUNICORN_WORKERS=1
export GUNICORN_THREADS=2
export GUNICORN_TIMEOUT=30

# Memory optimization
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_MAX_=65536

echo "📦 Installing dependencies..."

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found!"
    exit 1
fi

# Install with minimal output and no cache to save space
pip install -r requirements.txt --no-cache-dir --quiet --disable-pip-version-check

if [ $? -ne 0 ]; then
    echo "❌ Package installation failed!"
    echo "💡 Trying with individual packages..."
    
    # Try installing core packages individually
    pip install Flask==3.0.0 --no-cache-dir --quiet
    pip install requests==2.31.0 --no-cache-dir --quiet  
    pip install pytz==2023.3 --no-cache-dir --quiet
    pip install gunicorn==21.2.0 --no-cache-dir --quiet
    pip install flask-caching==2.1.0 --no-cache-dir --quiet
    pip install mmh3==4.1.0 --no-cache-dir --quiet
    
    echo "✅ Core packages installed"
fi

echo "🔧 Starting application..."

# Check available startup options
if command -v gunicorn &> /dev/null; then
    echo "🏭 Starting with Gunicorn (recommended for CodeSandbox)"
    
    # Check if gunicorn.conf.py exists
    if [ -f "gunicorn.conf.py" ]; then
        echo "📋 Using gunicorn.conf.py configuration"
        exec gunicorn -c gunicorn.conf.py main:app
    else
        echo "⚙️ Using inline Gunicorn configuration"
        exec gunicorn \
            --bind 0.0.0.0:${PORT:-5000} \
            --workers 1 \
            --threads 2 \
            --timeout 30 \
            --keepalive 2 \
            --max-requests 100 \
            --max-requests-jitter 10 \
            --preload \
            --log-level info \
            main:app
    fi
else
    echo "🐍 Starting with Python (fallback mode)"
    exec python3 main.py
fi