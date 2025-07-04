"""
API Cache Manager for FootballHub
Multi-level API response caching system for improved performance
"""

import json
import time
import hashlib
import gzip
import threading
from typing import Dict, Any, Optional, Union, List
from functools import wraps
from flask import request, jsonify, current_app
import logging

logger = logging.getLogger(__name__)

class APICacheManager:
    """
    Multi-level API response caching system
    """
    
    def __init__(self, default_ttl: int = 3600, max_cache_size: int = 1000):
        self.default_ttl = default_ttl
        self.max_cache_size = max_cache_size
        
        # Level 1: In-memory cache (fastest)
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Level 2: Compressed cache (memory efficient)
        self.compressed_cache: Dict[str, bytes] = {}
        
        # Access tracking for LRU eviction
        self.access_times: Dict[str, float] = {}
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'compressed_hits': 0,
            'evictions': 0
        }
        
        self.lock = threading.RLock()
    
    def _generate_cache_key(self, endpoint: str, params: Dict[str, Any] = None, 
                          headers: Dict[str, str] = None) -> str:
        """Generate unique cache key for API request"""
        key_parts = [endpoint]
        
        if params:
            # Sort parameters for consistent key generation
            sorted_params = sorted(params.items())
            key_parts.append(json.dumps(sorted_params, sort_keys=True))
        
        if headers:
            # Include relevant headers (e.g., user-agent, accept-language)
            relevant_headers = {k: v for k, v in headers.items() 
                              if k.lower() in ['user-agent', 'accept-language']}
            if relevant_headers:
                key_parts.append(json.dumps(relevant_headers, sort_keys=True))
        
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _compress_response(self, response_data: Dict[str, Any]) -> bytes:
        """Compress response data for storage"""
        try:
            json_str = json.dumps(response_data, separators=(',', ':'))
            return gzip.compress(json_str.encode())
        except Exception as e:
            logger.error(f"Failed to compress response: {e}")
            return b''
    
    def _decompress_response(self, compressed_data: bytes) -> Dict[str, Any]:
        """Decompress response data"""
        try:
            json_str = gzip.decompress(compressed_data).decode()
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to decompress response: {e}")
            return {}
    
    def _is_expired(self, cached_data: Dict[str, Any]) -> bool:
        """Check if cached data is expired"""
        if 'expires_at' not in cached_data:
            return True
        return time.time() > cached_data['expires_at']
    
    def _evict_lru(self, cache_dict: Dict[str, Any], max_size: int):
        """Evict least recently used items"""
        if len(cache_dict) >= max_size:
            # Remove 20% of oldest items
            evict_count = max(1, len(cache_dict) // 5)
            sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
            
            for key, _ in sorted_items[:evict_count]:
                cache_dict.pop(key, None)
                self.access_times.pop(key, None)
                self.stats['evictions'] += 1
    
    def get_cached_response(self, endpoint: str, params: Dict[str, Any] = None,
                           headers: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
        """Get cached API response"""
        with self.lock:
            cache_key = self._generate_cache_key(endpoint, params, headers)
            current_time = time.time()
            
            # Level 1: Check memory cache
            if cache_key in self.memory_cache:
                cached_data = self.memory_cache[cache_key]
                if not self._is_expired(cached_data):
                    self.access_times[cache_key] = current_time
                    self.stats['hits'] += 1
                    self.stats['memory_hits'] += 1
                    return cached_data['data']
                else:
                    # Remove expired entry
                    self.memory_cache.pop(cache_key, None)
            
            # Level 2: Check compressed cache
            if cache_key in self.compressed_cache:
                try:
                    compressed_data = self.compressed_cache[cache_key]
                    decompressed = self._decompress_response(compressed_data)
                    
                    if not self._is_expired(decompressed):
                        # Move to memory cache for faster access
                        self.memory_cache[cache_key] = decompressed
                        self.access_times[cache_key] = current_time
                        self.stats['hits'] += 1
                        self.stats['compressed_hits'] += 1
                        return decompressed['data']
                    else:
                        # Remove expired entry
                        self.compressed_cache.pop(cache_key, None)
                        
                except Exception as e:
                    logger.error(f"Error accessing compressed cache: {e}")
                    self.compressed_cache.pop(cache_key, None)
            
            self.stats['misses'] += 1
            return None
    
    def cache_response(self, endpoint: str, response_data: Dict[str, Any],
                      params: Dict[str, Any] = None, headers: Dict[str, str] = None,
                      ttl: Optional[int] = None) -> bool:
        """Cache API response"""
        with self.lock:
            try:
                cache_key = self._generate_cache_key(endpoint, params, headers)
                current_time = time.time()
                expires_at = current_time + (ttl or self.default_ttl)
                
                cached_data = {
                    'data': response_data,
                    'cached_at': current_time,
                    'expires_at': expires_at,
                    'endpoint': endpoint
                }
                
                # Determine cache level based on response size
                response_size = len(json.dumps(response_data, separators=(',', ':')))
                
                if response_size < 50000:  # < 50KB: Store in memory cache
                    self._evict_lru(self.memory_cache, self.max_cache_size)
                    self.memory_cache[cache_key] = cached_data
                else:  # >= 50KB: Store in compressed cache
                    self._evict_lru(self.compressed_cache, self.max_cache_size)
                    compressed_data = self._compress_response(cached_data)
                    if compressed_data:
                        self.compressed_cache[cache_key] = compressed_data
                
                self.access_times[cache_key] = current_time
                return True
                
            except Exception as e:
                logger.error(f"Failed to cache response: {e}")
                return False
    
    def invalidate_cache(self, endpoint: str = None, pattern: str = None):
        """Invalidate cache entries"""
        with self.lock:
            if endpoint:
                # Invalidate specific endpoint
                keys_to_remove = []
                for key in list(self.memory_cache.keys()):
                    if self.memory_cache[key].get('endpoint') == endpoint:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    self.memory_cache.pop(key, None)
                    self.compressed_cache.pop(key, None)
                    self.access_times.pop(key, None)
            
            elif pattern:
                # Invalidate by pattern (simple substring match)
                keys_to_remove = []
                for key in list(self.memory_cache.keys()):
                    if pattern in self.memory_cache[key].get('endpoint', ''):
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    self.memory_cache.pop(key, None)
                    self.compressed_cache.pop(key, None)
                    self.access_times.pop(key, None)
    
    def clear_cache(self):
        """Clear all cache entries"""
        with self.lock:
            self.memory_cache.clear()
            self.compressed_cache.clear()
            self.access_times.clear()
            self.stats = {
                'hits': 0,
                'misses': 0,
                'memory_hits': 0,
                'compressed_hits': 0,
                'evictions': 0
            }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'memory_entries': len(self.memory_cache),
                'compressed_entries': len(self.compressed_cache),
                'total_entries': len(self.memory_cache) + len(self.compressed_cache),
                'hit_rate': round(hit_rate, 2),
                'memory_hit_rate': round(
                    (self.stats['memory_hits'] / self.stats['hits'] * 100) if self.stats['hits'] > 0 else 0, 2
                ),
                'compressed_hit_rate': round(
                    (self.stats['compressed_hits'] / self.stats['hits'] * 100) if self.stats['hits'] > 0 else 0, 2
                ),
                **self.stats
            }

# Global cache manager instance
cache_manager = APICacheManager()

def cached_api_response(ttl: int = 3600, cache_key_params: List[str] = None):
    """
    Decorator for caching API responses
    
    Args:
        ttl: Time to live in seconds
        cache_key_params: List of request parameters to include in cache key
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from request
            endpoint = request.endpoint or func.__name__
            
            # Extract parameters for cache key
            params = {}
            if cache_key_params:
                for param in cache_key_params:
                    if param in request.args:
                        params[param] = request.args[param]
                    elif param in request.json if request.json else {}:
                        params[param] = request.json[param]
            else:
                # Use all query parameters
                params = dict(request.args)
            
            # Check cache
            cached_response = cache_manager.get_cached_response(
                endpoint, params, dict(request.headers)
            )
            
            if cached_response:
                return jsonify(cached_response)
            
            # Execute function and cache result
            response = func(*args, **kwargs)
            
            # Cache the response
            if hasattr(response, 'get_json'):
                response_data = response.get_json()
                cache_manager.cache_response(
                    endpoint, response_data, params, dict(request.headers), ttl
                )
            
            return response
        
        return wrapper
    return decorator

def get_api_cache_stats():
    """Get API cache statistics"""
    return cache_manager.get_cache_stats()

def clear_api_cache():
    """Clear API cache"""
    cache_manager.clear_cache()

def invalidate_api_cache(endpoint: str = None, pattern: str = None):
    """Invalidate API cache entries"""
    cache_manager.invalidate_cache(endpoint, pattern)