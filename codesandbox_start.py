#!/usr/bin/env python3
"""
CodeSandbox startup script with error handling and dependency management
"""

import os
import sys
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install dependencies with fallback options"""
    logger.info("🔧 Installing dependencies...")
    
    # Basic dependencies that should work in most environments
    basic_deps = [
        'flask>=3.0.0',
        'gunicorn>=20.0.0',
        'requests>=2.25.0',
        'python-dotenv>=0.19.0',
        'pytz>=2021.1'
    ]
    
    # Install basic dependencies first
    for dep in basic_deps:
        try:
            logger.info(f"Installing {dep}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', dep], 
                         check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            logger.warning(f"Failed to install {dep}, continuing...")
    
    # Try to install ML dependencies
    ml_deps = [
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0'
    ]
    
    for dep in ml_deps:
        try:
            logger.info(f"Installing {dep}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', dep], 
                         check=True, capture_output=True, text=True, timeout=120)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.warning(f"Failed to install {dep}, using fallback...")

def setup_environment():
    """Setup environment variables"""
    logger.info("🌐 Setting up environment...")
    
    # Load .env file
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    # Set default port for CodeSandbox
    if not os.environ.get('PORT'):
        os.environ['PORT'] = '5000'
    
    # Ensure required environment variables
    defaults = {
        'FLASK_ENV': 'production',
        'FLASK_DEBUG': 'False',
        'APIFOOTBALL_API_KEY': '908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485'
    }
    
    for key, default_value in defaults.items():
        if not os.environ.get(key):
            os.environ[key] = default_value

def start_application():
    """Start the Flask application"""
    logger.info("🚀 Starting application...")
    
    try:
        # Try to import the main app
        from main import app
        
        port = int(os.environ.get('PORT', 5000))
        host = os.environ.get('HOST', '0.0.0.0')
        
        logger.info(f"Starting server on {host}:{port}")
        
        # For CodeSandbox, use the development server
        app.run(host=host, port=port, debug=False, threaded=True)
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Attempting to install missing dependencies...")
        install_dependencies()
        
        # Try again
        try:
            from main import app
            port = int(os.environ.get('PORT', 5000))
            host = os.environ.get('HOST', '0.0.0.0')
            app.run(host=host, port=port, debug=False, threaded=True)
        except Exception as e2:
            logger.error(f"Failed to start application: {e2}")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Application startup error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    try:
        logger.info("🏗️  CodeSandbox Football Prediction App Startup")
        
        # Install dependencies
        install_dependencies()
        
        # Setup environment
        setup_environment()
        
        # Start application
        start_application()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)