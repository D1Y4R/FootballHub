"""
Performance Monitoring Module for Football Prediction System
Tracks API response times, memory usage, and optimization opportunities
"""
import time
import psutil
import logging
import json
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import os

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Data class for storing performance metrics"""
    timestamp: str
    endpoint: str
    method: str
    response_time: float
    memory_usage_mb: float
    cpu_percent: float
    cache_hit: bool = False
    error: bool = False
    status_code: int = 200

class PerformanceMonitor:
    """Monitors application performance and tracks metrics"""
    
    def __init__(self, metrics_file: str = 'performance_metrics.json'):
        self.metrics_file = metrics_file
        self.metrics: List[PerformanceMetric] = []
        self.start_time = time.time()
        self.load_metrics()
        
        # Performance thresholds
        self.slow_response_threshold = 2.0  # seconds
        self.high_memory_threshold = 200  # MB
        self.high_cpu_threshold = 80  # percent
        
    def load_metrics(self):
        """Load existing metrics from file"""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics = [PerformanceMetric(**item) for item in data]
                logger.info(f"Loaded {len(self.metrics)} performance metrics")
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            self.metrics = []
    
    def save_metrics(self):
        """Save metrics to file"""
        try:
            # Keep only last 1000 metrics to prevent file bloat
            recent_metrics = self.metrics[-1000:] if len(self.metrics) > 1000 else self.metrics
            
            with open(self.metrics_file, 'w') as f:
                json.dump([asdict(metric) for metric in recent_metrics], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def record_metric(self, endpoint: str, method: str, response_time: float, 
                     cache_hit: bool = False, error: bool = False, status_code: int = 200):
        """Record a performance metric"""
        try:
            # Get system metrics
            memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
            cpu_percent = psutil.cpu_percent()
            
            metric = PerformanceMetric(
                timestamp=datetime.now().isoformat(),
                endpoint=endpoint,
                method=method,
                response_time=response_time,
                memory_usage_mb=memory_usage,
                cpu_percent=cpu_percent,
                cache_hit=cache_hit,
                error=error,
                status_code=status_code
            )
            
            self.metrics.append(metric)
            
            # Log warnings for poor performance
            if response_time > self.slow_response_threshold:
                logger.warning(f"Slow response: {endpoint} took {response_time:.2f}s")
            
            if memory_usage > self.high_memory_threshold:
                logger.warning(f"High memory usage: {memory_usage:.1f}MB")
                
            if cpu_percent > self.high_cpu_threshold:
                logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                
        except Exception as e:
            logger.error(f"Error recording metric: {e}")
    
    def monitor_endpoint(self, endpoint_name: str = None, cache_check: callable = None):
        """Decorator to monitor endpoint performance"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                endpoint = endpoint_name or func.__name__
                method = 'GET'  # Default, could be enhanced to detect actual method
                error = False
                status_code = 200
                cache_hit = False
                
                try:
                    # Check cache hit if function provided
                    if cache_check:
                        cache_hit = cache_check(*args, **kwargs)
                    
                    result = func(*args, **kwargs)
                    
                    # Try to extract status code from Flask response
                    if hasattr(result, 'status_code'):
                        status_code = result.status_code
                    
                    return result
                    
                except Exception as e:
                    error = True
                    status_code = 500
                    logger.error(f"Error in monitored endpoint {endpoint}: {e}")
                    raise
                    
                finally:
                    response_time = time.time() - start_time
                    self.record_metric(endpoint, method, response_time, cache_hit, error, status_code)
                    
            return wrapper
        return decorator
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of metrics for the last N hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_metrics = [
                m for m in self.metrics 
                if datetime.fromisoformat(m.timestamp) > cutoff_time
            ]
            
            if not recent_metrics:
                return {
                    'period_hours': hours,
                    'total_requests': 0,
                    'message': 'No metrics available for this period'
                }
            
            # Calculate statistics
            response_times = [m.response_time for m in recent_metrics if not m.error]
            memory_usage = [m.memory_usage_mb for m in recent_metrics]
            cpu_usage = [m.cpu_percent for m in recent_metrics]
            
            # Count metrics by endpoint
            endpoint_counts = {}
            endpoint_avg_times = {}
            for metric in recent_metrics:
                endpoint = metric.endpoint
                endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
                
                if endpoint not in endpoint_avg_times:
                    endpoint_avg_times[endpoint] = []
                if not metric.error:
                    endpoint_avg_times[endpoint].append(metric.response_time)
            
            # Calculate averages for each endpoint
            for endpoint in endpoint_avg_times:
                times = endpoint_avg_times[endpoint]
                endpoint_avg_times[endpoint] = sum(times) / len(times) if times else 0
            
            # Performance issues
            slow_requests = len([m for m in recent_metrics if m.response_time > self.slow_response_threshold])
            error_requests = len([m for m in recent_metrics if m.error])
            cache_hits = len([m for m in recent_metrics if m.cache_hit])
            
            return {
                'period_hours': hours,
                'total_requests': len(recent_metrics),
                'successful_requests': len(recent_metrics) - error_requests,
                'error_requests': error_requests,
                'error_rate_percent': round((error_requests / len(recent_metrics)) * 100, 2),
                'cache_hits': cache_hits,
                'cache_hit_rate_percent': round((cache_hits / len(recent_metrics)) * 100, 2),
                'response_time': {
                    'avg_seconds': round(sum(response_times) / len(response_times), 3) if response_times else 0,
                    'min_seconds': round(min(response_times), 3) if response_times else 0,
                    'max_seconds': round(max(response_times), 3) if response_times else 0,
                    'slow_requests': slow_requests,
                    'slow_requests_percent': round((slow_requests / len(recent_metrics)) * 100, 2)
                },
                'memory_usage': {
                    'avg_mb': round(sum(memory_usage) / len(memory_usage), 1),
                    'min_mb': round(min(memory_usage), 1),
                    'max_mb': round(max(memory_usage), 1)
                },
                'cpu_usage': {
                    'avg_percent': round(sum(cpu_usage) / len(cpu_usage), 1),
                    'min_percent': round(min(cpu_usage), 1),
                    'max_percent': round(max(cpu_usage), 1)
                },
                'endpoints': {
                    'request_counts': endpoint_counts,
                    'avg_response_times': {k: round(v, 3) for k, v in endpoint_avg_times.items()}
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating metrics summary: {e}")
            return {'error': str(e)}
    
    def get_slowest_endpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the slowest endpoints by average response time"""
        try:
            endpoint_times = {}
            
            for metric in self.metrics:
                if metric.error:
                    continue
                    
                endpoint = metric.endpoint
                if endpoint not in endpoint_times:
                    endpoint_times[endpoint] = []
                endpoint_times[endpoint].append(metric.response_time)
            
            # Calculate averages and sort
            endpoint_averages = [
                {
                    'endpoint': endpoint,
                    'avg_response_time': sum(times) / len(times),
                    'request_count': len(times),
                    'max_response_time': max(times),
                    'min_response_time': min(times)
                }
                for endpoint, times in endpoint_times.items()
            ]
            
            return sorted(endpoint_averages, key=lambda x: x['avg_response_time'], reverse=True)[:limit]
            
        except Exception as e:
            logger.error(f"Error getting slowest endpoints: {e}")
            return []
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_metrics = [
                m for m in self.metrics 
                if datetime.fromisoformat(m.timestamp) > cutoff_time
            ]
            
            if len(recent_metrics) < 2:
                return {'message': 'Insufficient data for trend analysis'}
            
            # Split into first and second half for comparison
            mid_point = len(recent_metrics) // 2
            first_half = recent_metrics[:mid_point]
            second_half = recent_metrics[mid_point:]
            
            def calculate_averages(metrics_list):
                response_times = [m.response_time for m in metrics_list if not m.error]
                memory_usage = [m.memory_usage_mb for m in metrics_list]
                return {
                    'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
                    'avg_memory_usage': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                    'error_rate': len([m for m in metrics_list if m.error]) / len(metrics_list) * 100
                }
            
            first_avg = calculate_averages(first_half)
            second_avg = calculate_averages(second_half)
            
            # Calculate trends (positive = getting worse, negative = getting better)
            response_time_trend = second_avg['avg_response_time'] - first_avg['avg_response_time']
            memory_trend = second_avg['avg_memory_usage'] - first_avg['avg_memory_usage']
            error_rate_trend = second_avg['error_rate'] - first_avg['error_rate']
            
            return {
                'period_hours': hours,
                'first_half_avg': first_avg,
                'second_half_avg': second_avg,
                'trends': {
                    'response_time_change_seconds': round(response_time_trend, 3),
                    'memory_usage_change_mb': round(memory_trend, 1),
                    'error_rate_change_percent': round(error_rate_trend, 2),
                    'performance_direction': 'improving' if response_time_trend < 0 else 'degrading'
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {'error': str(e)}
    
    def cleanup_old_metrics(self, days: int = 7):
        """Remove metrics older than specified days"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            original_count = len(self.metrics)
            
            self.metrics = [
                m for m in self.metrics 
                if datetime.fromisoformat(m.timestamp) > cutoff_time
            ]
            
            removed_count = original_count - len(self.metrics)
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old metrics")
                self.save_metrics()
                
        except Exception as e:
            logger.error(f"Error cleaning up metrics: {e}")

# Global monitor instance
performance_monitor = PerformanceMonitor()

# Convenience functions
def monitor_api_endpoint(endpoint_name: str = None):
    """Decorator for monitoring API endpoints"""
    return performance_monitor.monitor_endpoint(endpoint_name)

def get_performance_summary(hours: int = 24) -> Dict[str, Any]:
    """Get performance summary for the last N hours"""
    return performance_monitor.get_metrics_summary(hours)

def get_slowest_endpoints(limit: int = 10) -> List[Dict[str, Any]]:
    """Get the slowest endpoints"""
    return performance_monitor.get_slowest_endpoints(limit)

def save_performance_metrics():
    """Save performance metrics to file"""
    performance_monitor.save_metrics()

if __name__ == "__main__":
    # CLI usage for performance analysis
    import sys
    import json
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'summary':
            hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
            summary = get_performance_summary(hours)
            print(f"Performance Summary (Last {hours} hours):")
            print(json.dumps(summary, indent=2))
            
        elif command == 'slowest':
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            slowest = get_slowest_endpoints(limit)
            print(f"Top {limit} Slowest Endpoints:")
            for i, endpoint in enumerate(slowest, 1):
                print(f"{i}. {endpoint['endpoint']}: {endpoint['avg_response_time']:.3f}s avg")
                
        elif command == 'trends':
            hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
            trends = performance_monitor.get_performance_trends(hours)
            print(f"Performance Trends (Last {hours} hours):")
            print(json.dumps(trends, indent=2))
            
        elif command == 'cleanup':
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
            performance_monitor.cleanup_old_metrics(days)
            print(f"Cleaned up metrics older than {days} days")
            
        else:
            print("Unknown command")
    else:
        print("Usage:")
        print("  python performance_monitor.py summary [hours]")
        print("  python performance_monitor.py slowest [limit]")
        print("  python performance_monitor.py trends [hours]")
        print("  python performance_monitor.py cleanup [days]")