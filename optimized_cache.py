"""
Optimized Cache Module for FootballHub
Performance-optimized caching system for resource-constrained environments
"""

import json
import time
import threading
from typing import Dict, Any, Optional, Tuple
from functools import wraps
import hashlib
import gzip
import pickle
import os

class OptimizedCache:
    """
    Memory-efficient cache with compression and TTL support
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        
    def _generate_key(self, key: str) -> str:
        """Generate hash key for cache storage"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _is_expired(self, timestamp: float, ttl: int) -> bool:
        """Check if cache entry is expired"""
        return time.time() - timestamp > ttl
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data using gzip"""
        try:
            serialized = pickle.dumps(data)
            return gzip.compress(serialized)
        except Exception:
            return pickle.dumps(data)
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data"""
        try:
            decompressed = gzip.decompress(compressed_data)
            return pickle.loads(decompressed)
        except Exception:
            return pickle.loads(compressed_data)
    
    def _evict_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if self._is_expired(timestamp, self.default_ttl)
        ]
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
    
    def _evict_lru(self):
        """Evict least recently used entries"""
        if len(self.cache) >= self.max_size:
            # Remove 25% of oldest entries
            sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
            evict_count = max(1, len(sorted_keys) // 4)
            
            for key, _ in sorted_keys[:evict_count]:
                self.cache.pop(key, None)
                self.access_times.pop(key, None)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        with self.lock:
            hashed_key = self._generate_key(key)
            
            if hashed_key in self.cache:
                compressed_data, timestamp = self.cache[hashed_key]
                
                if not self._is_expired(timestamp, self.default_ttl):
                    self.access_times[hashed_key] = time.time()
                    self.hits += 1
                    return self._decompress_data(compressed_data)
                else:
                    # Remove expired entry
                    self.cache.pop(hashed_key, None)
                    self.access_times.pop(hashed_key, None)
            
            self.misses += 1
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        with self.lock:
            try:
                # Clean up expired entries
                self._evict_expired()
                
                # Evict LRU if needed
                self._evict_lru()
                
                hashed_key = self._generate_key(key)
                compressed_data = self._compress_data(value)
                current_time = time.time()
                
                self.cache[hashed_key] = (compressed_data, current_time)
                self.access_times[hashed_key] = current_time
                
                return True
            except Exception:
                return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            hashed_key = self._generate_key(key)
            if hashed_key in self.cache:
                self.cache.pop(hashed_key, None)
                self.access_times.pop(hashed_key, None)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.hits = 0
            self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'entries': len(self.cache),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': round(hit_rate, 2),
                'max_size': self.max_size
            }

# Global cache instance
cache = OptimizedCache(max_size=500, default_ttl=1800)

def cached(ttl: int = 1800, key_prefix: str = ""):
    """
    Decorator for caching function results
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__, str(args), str(sorted(kwargs.items()))]
            cache_key = "|".join(key_parts)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator

def get_cache_stats():
    """Get cache statistics"""
    return cache.stats()

def clear_cache():
    """Clear all cache"""
    cache.clear()