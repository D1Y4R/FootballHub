#!/usr/bin/env python3
"""
Simple runner for CodeSandbox - handles environment loading and starts the app
"""

import os
import sys

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Environment variables loaded from .env")
    except FileNotFoundError:
        print("❌ .env file not found")
    except Exception as e:
        print(f"❌ Error loading .env: {e}")

# Load environment variables
load_env_file()

# Check if API keys are loaded
api_key = os.environ.get('APIFOOTBALL_API_KEY', '908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485')
print(f"✅ API Key loaded: {api_key[:20]}...")

# Now import and run the main app
if __name__ == '__main__':
    try:
        # Import the Flask app
        from main import app
        
        # Get port from environment or use 5000
        port = int(os.environ.get('PORT', 5000))
        
        print(f"🚀 Starting Football Prediction App on port {port}")
        print(f"📍 URL: http://localhost:{port}")
        
        # Run the app
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        sys.exit(1)