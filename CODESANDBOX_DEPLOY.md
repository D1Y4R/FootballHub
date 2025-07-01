# CodeSandbox Deployment Instructions

## 🚀 Quick Start

### Method 1: Automatic Setup
1. Upload all files to CodeSandbox
2. Run: `./start.sh`

### Method 2: Manual Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   export $(cat .env | xargs)
   ```

3. Start the application:
   ```bash
   gunicorn --bind 0.0.0.0:5000 main:app
   ```

## 📝 Files Created for CodeSandbox:

- ✅ `requirements.txt` - Python dependencies
- ✅ `.env` - Environment variables with API key
- ✅ `start.sh` - Startup script
- ✅ `package.json` - Project metadata

## 🔧 Manual Commands:

If you prefer to run commands manually:

```bash
# Install dependencies
pip install gunicorn flask requests numpy pandas scikit-learn tensorflow pytz python-dotenv tabulate Flask-Caching

# Start with gunicorn
gunicorn --bind 0.0.0.0:5000 main:app --timeout 120 --workers 1
```

## 🌐 Access URLs:

After starting, the app will be available at:
- Main page: `http://localhost:5000/`
- API endpoints: `http://localhost:5000/api/`
- Predictions: `http://localhost:5000/predictions`

## 🔍 Troubleshooting:

### If API keys don't work:
1. Check `.env` file exists
2. Verify environment variables are loaded:
   ```bash
   python -c "import os; print('API Key:', os.environ.get('APIFOOTBALL_API_KEY', 'NOT SET'))"
   ```

### If dependencies fail:
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### If port is busy:
```bash
gunicorn --bind 0.0.0.0:8000 main:app  # Try different port
```

## 📊 API Key Status:

Your API key is configured as: `908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485`

The application will automatically load this from the `.env` file.

## 🎯 Features Available:

- ⚽ Match predictions
- 📈 Team statistics
- 🎲 Monte Carlo simulations
- 📊 Betting odds calculations
- 🔍 Advanced analytics