# API Key Setup Instructions

## Security Notice ⚠️
**NEVER commit API keys to version control!** Always use environment variables or secure configuration files.

## Quick Setup

### Method 1: Environment Variables (Recommended)

**Linux/Mac:**
```bash
export APIFOOTBALL_API_KEY="908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485"
export FOOTBALL_DATA_API_KEY="908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485"
export API_FOOTBALL_KEY="908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485"
export APIFOOTBALL_PREMIUM_KEY="908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485"
export SESSION_SECRET="your-secure-session-secret-here"
```

**Windows:**
```cmd
set APIFOOTBALL_API_KEY=908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485
set FOOTBALL_DATA_API_KEY=908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485
set API_FOOTBALL_KEY=908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485
set APIFOOTBALL_PREMIUM_KEY=908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485
set SESSION_SECRET=your-secure-session-secret-here
```

### Method 2: .env File

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file:
   ```bash
   # Football API Keys
   APIFOOTBALL_API_KEY=908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485
   FOOTBALL_DATA_API_KEY=908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485
   API_FOOTBALL_KEY=908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485
   APIFOOTBALL_PREMIUM_KEY=908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485
   
   # Session Security
   SESSION_SECRET=your-secure-session-secret-here
   ```

3. Load environment variables (if using python-dotenv):
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

## Verification

After setting up your API keys, you can verify they're working by running:

```bash
python3 -c "
import os
print('API Keys Status:')
print(f'APIFOOTBALL_API_KEY: {'✅ Set' if os.environ.get('APIFOOTBALL_API_KEY') else '❌ Missing'}')
print(f'FOOTBALL_DATA_API_KEY: {'✅ Set' if os.environ.get('FOOTBALL_DATA_API_KEY') else '❌ Missing'}')
print(f'API_FOOTBALL_KEY: {'✅ Set' if os.environ.get('API_FOOTBALL_KEY') else '❌ Missing'}')
print(f'SESSION_SECRET: {'✅ Set' if os.environ.get('SESSION_SECRET') else '❌ Missing'}')
"
```

## Production Deployment

For production environments, use your hosting platform's environment variable settings:

- **Heroku**: `heroku config:set APIFOOTBALL_API_KEY=your_key_here`
- **Vercel**: Add variables in the dashboard or `.env.local`
- **AWS**: Use Parameter Store or Secrets Manager
- **Docker**: Use `--env-file` or environment sections in docker-compose

## Security Best Practices

1. ✅ **Use environment variables** - Never hardcode keys
2. ✅ **Add .env to .gitignore** - Prevent accidental commits
3. ✅ **Rotate keys regularly** - Update periodically for security
4. ✅ **Use different keys for different environments** - Dev/staging/prod
5. ✅ **Monitor usage** - Watch for unexpected API usage

## Troubleshooting

If you see warnings like "API key not set", check:

1. Environment variables are properly exported
2. Variable names match exactly (case-sensitive)
3. No extra spaces or quotes in the values
4. Application restarted after setting variables