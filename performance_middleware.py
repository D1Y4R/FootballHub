"""
Performance Middleware for FootballHub
Request monitoring, throttling, and performance optimization
"""

import time
import psutil
import threading
from collections import defaultdict, deque
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
from flask import request, jsonify, g, current_app
import logging

logger = logging.getLogger(__name__)

class PerformanceMiddleware:
    """
    Performance monitoring and throttling middleware
    """
    
    def __init__(self, app=None):
        self.app = app
        
        # Request tracking
        self.request_counts = defaultdict(int)
        self.request_times = defaultdict(deque)
        self.response_times = defaultdict(deque)
        self.error_counts = defaultdict(int)
        
        # System monitoring
        self.cpu_usage_history = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)
        self.last_system_check = 0
        
        # Performance thresholds
        self.cpu_threshold = 80.0  # CPU usage percentage
        self.memory_threshold = 80.0  # Memory usage percentage
        self.max_requests_per_minute = 60
        self.max_response_time = 30.0  # seconds
        
        # Rate limiting
        self.rate_limits = {}
        self.blocked_ips = set()
        
        # Performance stats
        self.stats = {
            'total_requests': 0,
            'total_errors': 0,
            'slow_requests': 0,
            'throttled_requests': 0,
            'avg_response_time': 0.0,
            'peak_cpu': 0.0,
            'peak_memory': 0.0
        }
        
        self.lock = threading.RLock()
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize middleware with Flask app"""
        self.app = app
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        app.teardown_request(self.teardown_request)
    
    def get_client_ip(self) -> str:
        """Get client IP address"""
        if request.headers.get('X-Forwarded-For'):
            return request.headers.get('X-Forwarded-For').split(',')[0].strip()
        elif request.headers.get('X-Real-IP'):
            return request.headers.get('X-Real-IP')
        return request.remote_addr or 'unknown'
    
    def check_system_resources(self):
        """Check current system resource usage"""
        current_time = time.time()
        
        # Check every 10 seconds
        if current_time - self.last_system_check < 10:
            return
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            with self.lock:
                self.cpu_usage_history.append(cpu_percent)
                self.memory_usage_history.append(memory_percent)
                
                # Update peak values
                self.stats['peak_cpu'] = max(self.stats['peak_cpu'], cpu_percent)
                self.stats['peak_memory'] = max(self.stats['peak_memory'], memory_percent)
                
                self.last_system_check = current_time
                
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
    
    def is_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate limited"""
        with self.lock:
            if client_ip in self.blocked_ips:
                return True
            
            current_time = time.time()
            minute_ago = current_time - 60
            
            # Clean old entries
            if client_ip in self.request_times:
                self.request_times[client_ip] = deque(
                    [t for t in self.request_times[client_ip] if t > minute_ago],
                    maxlen=self.max_requests_per_minute * 2
                )
            
            # Check rate limit
            if len(self.request_times[client_ip]) >= self.max_requests_per_minute:
                return True
            
            return False
    
    def should_throttle(self) -> bool:
        """Check if requests should be throttled due to high resource usage"""
        if not self.cpu_usage_history or not self.memory_usage_history:
            return False
        
        # Check recent CPU usage
        recent_cpu = list(self.cpu_usage_history)[-5:]  # Last 5 checks
        avg_cpu = sum(recent_cpu) / len(recent_cpu)
        
        # Check recent memory usage
        recent_memory = list(self.memory_usage_history)[-5:]
        avg_memory = sum(recent_memory) / len(recent_memory)
        
        return avg_cpu > self.cpu_threshold or avg_memory > self.memory_threshold
    
    def before_request(self):
        """Called before each request"""
        g.start_time = time.time()
        
        # Check system resources
        self.check_system_resources()
        
        client_ip = self.get_client_ip()
        
        # Rate limiting check
        if self.is_rate_limited(client_ip):
            with self.lock:
                self.stats['throttled_requests'] += 1
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': 'Too many requests. Please slow down.'
            }), 429
        
        # Resource throttling check
        if self.should_throttle():
            with self.lock:
                self.stats['throttled_requests'] += 1
            return jsonify({
                'error': 'Server overloaded',
                'message': 'Server is under heavy load. Please try again later.'
            }), 503
        
        # Track request
        with self.lock:
            self.request_times[client_ip].append(time.time())
            self.request_counts[client_ip] += 1
            self.stats['total_requests'] += 1
    
    def after_request(self, response):
        """Called after each request"""
        if hasattr(g, 'start_time'):
            response_time = time.time() - g.start_time
            client_ip = self.get_client_ip()
            
            with self.lock:
                # Track response time
                self.response_times[client_ip].append(response_time)
                
                # Update average response time
                all_times = []
                for times in self.response_times.values():
                    all_times.extend(times)
                
                if all_times:
                    self.stats['avg_response_time'] = sum(all_times) / len(all_times)
                
                # Track slow requests
                if response_time > self.max_response_time:
                    self.stats['slow_requests'] += 1
                    logger.warning(f"Slow request: {request.endpoint} took {response_time:.2f}s")
                
                # Track errors
                if response.status_code >= 400:
                    self.error_counts[client_ip] += 1
                    self.stats['total_errors'] += 1
        
        return response
    
    def teardown_request(self, exception=None):
        """Called after request teardown"""
        if exception:
            logger.error(f"Request exception: {exception}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        with self.lock:
            current_time = time.time()
            
            # Calculate active connections
            active_connections = sum(
                1 for times in self.request_times.values()
                if any(t > current_time - 300 for t in times)  # Active in last 5 minutes
            )
            
            # Calculate average CPU and memory usage
            avg_cpu = sum(self.cpu_usage_history) / len(self.cpu_usage_history) if self.cpu_usage_history else 0
            avg_memory = sum(self.memory_usage_history) / len(self.memory_usage_history) if self.memory_usage_history else 0
            
            return {
                'requests': {
                    'total': self.stats['total_requests'],
                    'errors': self.stats['total_errors'],
                    'slow': self.stats['slow_requests'],
                    'throttled': self.stats['throttled_requests'],
                    'error_rate': (self.stats['total_errors'] / self.stats['total_requests'] * 100) if self.stats['total_requests'] > 0 else 0,
                    'avg_response_time': round(self.stats['avg_response_time'], 3)
                },
                'system': {
                    'cpu_usage': {
                        'current': self.cpu_usage_history[-1] if self.cpu_usage_history else 0,
                        'average': round(avg_cpu, 2),
                        'peak': self.stats['peak_cpu']
                    },
                    'memory_usage': {
                        'current': self.memory_usage_history[-1] if self.memory_usage_history else 0,
                        'average': round(avg_memory, 2),
                        'peak': self.stats['peak_memory']
                    },
                    'active_connections': active_connections,
                    'blocked_ips': len(self.blocked_ips)
                },
                'thresholds': {
                    'cpu_threshold': self.cpu_threshold,
                    'memory_threshold': self.memory_threshold,
                    'max_requests_per_minute': self.max_requests_per_minute,
                    'max_response_time': self.max_response_time
                }
            }
    
    def get_client_stats(self, client_ip: str) -> Dict[str, Any]:
        """Get statistics for a specific client"""
        with self.lock:
            current_time = time.time()
            minute_ago = current_time - 60
            
            recent_requests = sum(1 for t in self.request_times.get(client_ip, []) if t > minute_ago)
            
            return {
                'ip': client_ip,
                'total_requests': self.request_counts.get(client_ip, 0),
                'recent_requests': recent_requests,
                'errors': self.error_counts.get(client_ip, 0),
                'is_blocked': client_ip in self.blocked_ips,
                'avg_response_time': round(
                    sum(self.response_times.get(client_ip, [])) / len(self.response_times.get(client_ip, [1])), 3
                )
            }
    
    def block_ip(self, client_ip: str):
        """Block an IP address"""
        with self.lock:
            self.blocked_ips.add(client_ip)
            logger.warning(f"Blocked IP: {client_ip}")
    
    def unblock_ip(self, client_ip: str):
        """Unblock an IP address"""
        with self.lock:
            self.blocked_ips.discard(client_ip)
            logger.info(f"Unblocked IP: {client_ip}")
    
    def reset_stats(self):
        """Reset all statistics"""
        with self.lock:
            self.request_counts.clear()
            self.request_times.clear()
            self.response_times.clear()
            self.error_counts.clear()
            self.cpu_usage_history.clear()
            self.memory_usage_history.clear()
            self.blocked_ips.clear()
            
            self.stats = {
                'total_requests': 0,
                'total_errors': 0,
                'slow_requests': 0,
                'throttled_requests': 0,
                'avg_response_time': 0.0,
                'peak_cpu': 0.0,
                'peak_memory': 0.0
            }

# Global middleware instance
performance_middleware = PerformanceMiddleware()

def monitor_performance(threshold_cpu: float = 80.0, threshold_memory: float = 80.0):
    """
    Decorator for monitoring endpoint performance
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log performance if thresholds exceeded
                execution_time = time.time() - start_time
                if execution_time > performance_middleware.max_response_time:
                    logger.warning(f"Slow endpoint {func.__name__}: {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                raise
            
        return wrapper
    return decorator

def get_performance_stats():
    """Get performance statistics"""
    return performance_middleware.get_performance_stats()

def reset_performance_stats():
    """Reset performance statistics"""
    performance_middleware.reset_stats()

def block_client_ip(ip: str):
    """Block a client IP"""
    performance_middleware.block_ip(ip)

def unblock_client_ip(ip: str):
    """Unblock a client IP"""
    performance_middleware.unblock_ip(ip)