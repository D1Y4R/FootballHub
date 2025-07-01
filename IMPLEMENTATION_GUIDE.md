# Performance Optimization Implementation Guide

This guide provides step-by-step instructions for implementing the performance optimizations identified in the analysis.

## Quick Start - Immediate Improvements (< 30 minutes)

### 1. Optimize Cache File (Immediate - 95% cache size reduction)

```bash
# Run cache optimization immediately
python3 cache_optimizer.py optimize

# Check results
python3 cache_optimizer.py stats
```

**Integration into existing code:**
```python
# In match_prediction.py, replace the existing load_cache method:
from cache_optimizer import CacheOptimizer

class MatchPredictor:
    def __init__(self):
        # ... existing code ...
        self.cache_optimizer = CacheOptimizer()
        self.predictions_cache = self.cache_optimizer.load_cache_optimized()
        
    def save_cache(self):
        return self.cache_optimizer.save_cache_optimized(self.predictions_cache)
```

### 2. Optimize Static Assets (Immediate - 70% bundle size reduction)

```bash
# Run static asset optimization
python3 static_optimizer.py optimize static

# Results will be in static/optimized/ directory
```

**Update templates to use optimized assets:**
```html
<!-- In templates/base.html, replace static asset links with: -->

<!-- Optimized CSS Bundle -->
<link rel="stylesheet" href="{{ url_for('static', filename='optimized/css/app-styles.min.css') }}">

<!-- Optimized JavaScript Bundles -->
<script src="{{ url_for('static', filename='optimized/js/app-core.min.js') }}" defer></script>
<script src="{{ url_for('static', filename='optimized/js/app-stats.min.js') }}" defer></script>
```

### 3. Enable Compression (5 minutes)

```python
# Add to main.py or app initialization:
from flask_compress import Compress

app = Flask(__name__)
Compress(app)  # Enable gzip compression for all responses
```

**Install dependency:**
```bash
pip install flask-compress
```

## Phase 1 Implementation - Backend Optimizations (1-2 hours)

### 4. Integrate Performance Monitoring

```python
# In api_routes.py, add monitoring to endpoints:
from performance_monitor import monitor_api_endpoint

@api_v3_bp.route('/predict-match/<home_team_id>/<away_team_id>')
@monitor_api_endpoint('predict_match')
@api_cache(timeout=600)
def api_v3_predict_match(home_team_id, away_team_id):
    # ... existing code ...
```

### 5. Optimize Cache Management

```python
# In main.py, update cache configuration:
cache_config = {
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 600,  # Increased from 300
    "CACHE_THRESHOLD": 1000,       # Increased from 500
    "CACHE_OPTIONS": {
        "threshold": 1000,
        "default_timeout": 600
    }
}
cache = Cache(app, config=cache_config)
```

### 6. Add Cache Health Monitoring

```python
# Add new route in main.py:
@app.route('/api/cache/health')
def cache_health():
    from cache_optimizer import get_cache_health
    return jsonify(get_cache_health())

@app.route('/api/performance/summary')
def performance_summary():
    from performance_monitor import get_performance_summary
    return jsonify(get_performance_summary(24))
```

## Phase 2 Implementation - Advanced Optimizations (2-4 hours)

### 7. Lazy Loading for ML Models

```python
# In match_prediction.py, implement lazy loading:
class MatchPredictor:
    def __init__(self):
        # ... existing code ...
        self._model_home = None
        self._model_away = None
        
    @property
    def model_home(self):
        if self._model_home is None:
            logger.info("Lazy loading home model...")
            self._model_home = self.load_model('model_home.h5')
        return self._model_home
        
    @property
    def model_away(self):
        if self._model_away is None:
            logger.info("Lazy loading away model...")
            self._model_away = self.load_model('model_away.h5')
        return self._model_away
```

### 8. API Request Batching

```python
# In api_routes.py, add request batching:
@api_v3_bp.route('/predict-matches', methods=['POST'])
@monitor_api_endpoint('predict_matches_batch')
def predict_matches_batch():
    """Batch prediction endpoint for multiple matches"""
    try:
        match_requests = request.json.get('matches', [])
        if len(match_requests) > 10:  # Limit batch size
            return jsonify({"error": "Maximum 10 matches per batch"}), 400
            
        results = []
        for match_req in match_requests:
            # Process each match prediction
            result = process_single_prediction(
                match_req['home_team_id'], 
                match_req['away_team_id']
            )
            results.append(result)
            
        return jsonify({"predictions": results})
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500
```

### 9. Database Query Optimization

```python
# Add database connection pooling and indexing
# Create database_optimizer.py:

import sqlite3
import logging
from contextlib import contextmanager

class DatabaseOptimizer:
    def __init__(self, db_path='team_performance.db'):
        self.db_path = db_path
        self.setup_indexes()
    
    def setup_indexes(self):
        """Create indexes for frequently queried columns"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Add indexes for common queries
                conn.execute("CREATE INDEX IF NOT EXISTS idx_team_id ON team_performance(team_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_match_date ON team_performance(match_date)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_team_date ON team_performance(team_id, match_date)")
                conn.commit()
                logger.info("Database indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating database indexes: {e}")
```

## Phase 3 Implementation - Production Optimizations (4-6 hours)

### 10. Implement Response Caching

```python
# Create response_cache.py for sophisticated caching:
from functools import wraps
import json
import hashlib
from datetime import datetime, timedelta

class ResponseCache:
    def __init__(self):
        self.cache = {}
        self.max_size = 1000
        self.default_ttl = 300  # 5 minutes
        
    def cache_key(self, *args, **kwargs):
        """Generate cache key from function arguments"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def cached_response(self, ttl=None):
        """Decorator for caching function responses"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = f"{func.__name__}_{self.cache_key(*args, **kwargs)}"
                now = datetime.now()
                
                # Check if cached response exists and is valid
                if key in self.cache:
                    cached_data, expiry = self.cache[key]
                    if now < expiry:
                        return cached_data
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                cache_ttl = ttl or self.default_ttl
                expiry_time = now + timedelta(seconds=cache_ttl)
                
                # Store in cache (with size limit)
                if len(self.cache) >= self.max_size:
                    # Remove oldest entries
                    oldest_keys = sorted(self.cache.keys())[:100]
                    for old_key in oldest_keys:
                        del self.cache[old_key]
                
                self.cache[key] = (result, expiry_time)
                return result
                
            return wrapper
        return decorator
```

### 11. Add Health Check Endpoints

```python
# In main.py, add comprehensive health checks:
@app.route('/health')
def health_check():
    """Comprehensive health check"""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'checks': {}
    }
    
    # Cache health
    try:
        from cache_optimizer import get_cache_health
        health_status['checks']['cache'] = get_cache_health()
    except Exception as e:
        health_status['checks']['cache'] = {'status': 'error', 'error': str(e)}
        health_status['status'] = 'unhealthy'
    
    # Performance metrics
    try:
        from performance_monitor import get_performance_summary
        perf_summary = get_performance_summary(1)  # Last hour
        health_status['checks']['performance'] = {
            'avg_response_time': perf_summary.get('response_time', {}).get('avg_seconds', 0),
            'error_rate': perf_summary.get('error_rate_percent', 0),
            'status': 'healthy' if perf_summary.get('error_rate_percent', 0) < 5 else 'warning'
        }
    except Exception as e:
        health_status['checks']['performance'] = {'status': 'error', 'error': str(e)}
    
    # Database health
    try:
        import sqlite3
        with sqlite3.connect('team_performance.db') as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            health_status['checks']['database'] = {
                'status': 'healthy',
                'table_count': table_count
            }
    except Exception as e:
        health_status['checks']['database'] = {'status': 'error', 'error': str(e)}
        health_status['status'] = 'unhealthy'
    
    status_code = 200 if health_status['status'] == 'healthy' else 503
    return jsonify(health_status), status_code
```

## Configuration Updates

### 12. Update pyproject.toml

```toml
# Add new dependencies:
[project]
dependencies = [
    # ... existing dependencies ...
    "flask-compress>=1.13",
    "psutil>=5.9.0",
    "dataclasses>=0.8",  # For Python < 3.7 compatibility
]
```

### 13. Environment Variables

```bash
# Add to your environment configuration:
export CACHE_MAX_SIZE=104857600  # 100MB
export CACHE_MAX_AGE=604800      # 7 days
export PERFORMANCE_MONITOR=true
export ENABLE_COMPRESSION=true
```

## Monitoring and Maintenance

### 14. Daily Maintenance Script

```bash
#!/bin/bash
# daily_maintenance.sh

echo "Starting daily maintenance..."

# Cleanup old cache entries
python3 cache_optimizer.py optimize

# Cleanup old performance metrics
python3 performance_monitor.py cleanup 7

# Generate performance report
python3 performance_monitor.py summary 24 > daily_performance_report.txt

echo "Daily maintenance completed."
```

### 15. Performance Dashboard Route

```python
# Add to main.py for a simple performance dashboard:
@app.route('/admin/performance')
def performance_dashboard():
    """Simple performance dashboard"""
    from performance_monitor import get_performance_summary, get_slowest_endpoints
    from cache_optimizer import get_cache_health
    
    data = {
        'performance': get_performance_summary(24),
        'slowest_endpoints': get_slowest_endpoints(5),
        'cache_health': get_cache_health(),
        'timestamp': datetime.now().isoformat()
    }
    
    return render_template('performance_dashboard.html', data=data)
```

## Verification and Testing

### 16. Performance Testing

```python
# Create performance_test.py:
import time
import requests
import concurrent.futures
from statistics import mean, median

def test_endpoint_performance(url, num_requests=100, concurrent=10):
    """Test endpoint performance with concurrent requests"""
    
    def make_request():
        start_time = time.time()
        try:
            response = requests.get(url, timeout=10)
            response_time = time.time() - start_time
            return {
                'response_time': response_time,
                'status_code': response.status_code,
                'success': response.status_code == 200
            }
        except Exception as e:
            return {
                'response_time': time.time() - start_time,
                'status_code': 0,
                'success': False,
                'error': str(e)
            }
    
    # Run concurrent requests
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    # Calculate statistics
    response_times = [r['response_time'] for r in results if r['success']]
    success_rate = len([r for r in results if r['success']]) / len(results) * 100
    
    return {
        'total_requests': num_requests,
        'successful_requests': len(response_times),
        'success_rate_percent': round(success_rate, 2),
        'avg_response_time': round(mean(response_times), 3) if response_times else 0,
        'median_response_time': round(median(response_times), 3) if response_times else 0,
        'min_response_time': round(min(response_times), 3) if response_times else 0,
        'max_response_time': round(max(response_times), 3) if response_times else 0
    }

if __name__ == "__main__":
    # Test main endpoints
    base_url = "http://localhost:5000"
    
    endpoints = [
        "/",
        "/health",
        "/api/cache/health",
        "/api/performance/summary"
    ]
    
    for endpoint in endpoints:
        print(f"\nTesting {endpoint}...")
        results = test_endpoint_performance(f"{base_url}{endpoint}")
        print(f"Success rate: {results['success_rate_percent']}%")
        print(f"Avg response time: {results['avg_response_time']}s")
        print(f"Response time range: {results['min_response_time']}-{results['max_response_time']}s")
```

## Expected Results

After implementing all optimizations:

- **Cache file size**: 4.4MB → 200KB (95% reduction)
- **Frontend bundle size**: 82KB → 25KB (70% reduction)
- **Page load time**: 3-5s → 1-2s (60% improvement)
- **Memory usage**: ~200MB → ~100MB (50% reduction)
- **API response time**: 2-3s → 0.5-1s (70% improvement)

## Rollback Plan

If any optimization causes issues:

1. **Static assets**: Simply revert template changes to use original files
2. **Cache optimization**: Restore from backup (`predictions_cache_backup.json`)
3. **Performance monitoring**: Remove decorators from endpoints
4. **Compression**: Comment out `Compress(app)` line

## Next Steps

1. Implement optimizations in order of priority
2. Monitor performance metrics daily
3. Set up automated cache cleanup
4. Consider implementing Redis for production caching
5. Add more sophisticated performance monitoring
6. Implement database connection pooling for high-traffic scenarios

This guide provides a complete roadmap for optimizing the football prediction application's performance while maintaining stability and reliability.