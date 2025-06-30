#!/usr/bin/env python3
"""
Cache Optimization Module
Handles cache rotation, compression, and performance improvements for predictions cache.
"""

import json
import gzip
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import threading
import time

logger = logging.getLogger(__name__)

class OptimizedCacheManager:
    """
    Optimized cache manager with compression, rotation, and performance improvements.
    """
    
    def __init__(self, 
                 cache_file: str = 'predictions_cache.json',
                 compressed_cache_file: str = 'predictions_cache.json.gz',
                 max_size_mb: int = 2,
                 max_age_days: int = 30,
                 compression_enabled: bool = True,
                 auto_cleanup_interval: int = 3600):  # 1 hour
        
        self.cache_file = cache_file
        self.compressed_cache_file = compressed_cache_file
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_age_days = max_age_days
        self.compression_enabled = compression_enabled
        self.auto_cleanup_interval = auto_cleanup_interval
        
        self._cache_data = {}
        self._cache_modified = False
        self._last_cleanup = time.time()
        self._lock = threading.RLock()
        
        # Initialize cache
        self._load_cache()
        
        # Start background cleanup if enabled
        if auto_cleanup_interval > 0:
            self._start_background_cleanup()
    
    def _start_background_cleanup(self):
        """Start background thread for periodic cache cleanup."""
        def cleanup_worker():
            while True:
                time.sleep(self.auto_cleanup_interval)
                try:
                    self._auto_cleanup()
                except Exception as e:
                    logger.error(f"Background cache cleanup failed: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("Background cache cleanup started")
    
    def _load_cache(self):
        """Load cache from file with compression support."""
        with self._lock:
            try:
                # Try to load compressed cache first
                if self.compression_enabled and os.path.exists(self.compressed_cache_file):
                    logger.info("Loading compressed cache...")
                    with gzip.open(self.compressed_cache_file, 'rt', encoding='utf-8') as f:
                        self._cache_data = json.load(f)
                    logger.info(f"Compressed cache loaded: {len(self._cache_data)} entries")
                    return
                
                # Fallback to uncompressed cache
                if os.path.exists(self.cache_file):
                    logger.info("Loading uncompressed cache...")
                    with open(self.cache_file, 'r', encoding='utf-8') as f:
                        self._cache_data = json.load(f)
                    logger.info(f"Uncompressed cache loaded: {len(self._cache_data)} entries")
                    
                    # Migrate to compressed format if enabled
                    if self.compression_enabled:
                        self._save_compressed_cache()
                        logger.info("Cache migrated to compressed format")
                    return
                
                # No cache file exists
                self._cache_data = {}
                logger.info("No cache file found, starting with empty cache")
                
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                self._cache_data = {}
    
    def _save_compressed_cache(self):
        """Save cache in compressed format."""
        try:
            with gzip.open(self.compressed_cache_file, 'wt', encoding='utf-8') as f:
                json.dump(self._cache_data, f, ensure_ascii=False, separators=(',', ':'))
            logger.debug(f"Compressed cache saved: {len(self._cache_data)} entries")
        except Exception as e:
            logger.error(f"Error saving compressed cache: {e}")
            raise
    
    def _save_uncompressed_cache(self):
        """Save cache in uncompressed format (fallback)."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Uncompressed cache saved: {len(self._cache_data)} entries")
        except Exception as e:
            logger.error(f"Error saving uncompressed cache: {e}")
            raise
    
    def save_cache(self):
        """Save cache to file with automatic compression."""
        if not self._cache_modified:
            logger.debug("Cache not modified, skipping save")
            return
        
        with self._lock:
            try:
                # Clean old entries before saving
                self._cleanup_old_entries()
                
                if self.compression_enabled:
                    self._save_compressed_cache()
                else:
                    self._save_uncompressed_cache()
                
                self._cache_modified = False
                logger.info("Cache saved successfully")
                
            except Exception as e:
                logger.error(f"Error saving cache: {e}")
                # Try fallback to uncompressed format
                try:
                    self._save_uncompressed_cache()
                    self._cache_modified = False
                    logger.warning("Cache saved in uncompressed format (fallback)")
                except Exception as fallback_error:
                    logger.error(f"Fallback save also failed: {fallback_error}")
                    raise
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get item from cache."""
        with self._lock:
            entry = self._cache_data.get(key)
            if entry and self._is_entry_valid(entry):
                return entry
            elif entry:
                # Remove expired entry
                del self._cache_data[key]
                self._cache_modified = True
            return None
    
    def set(self, key: str, value: Dict[str, Any], ttl_hours: int = 24):
        """Set item in cache with TTL."""
        with self._lock:
            # Add timestamp and TTL to value
            enhanced_value = {
                **value,
                '_cache_timestamp': time.time(),
                '_cache_ttl_hours': ttl_hours
            }
            
            self._cache_data[key] = enhanced_value
            self._cache_modified = True
            
            # Check if cache needs cleanup
            if self._should_cleanup():
                self._auto_cleanup()
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache_data.clear()
            self._cache_modified = True
            logger.info("Cache cleared")
    
    def delete(self, key: str) -> bool:
        """Delete specific key from cache."""
        with self._lock:
            if key in self._cache_data:
                del self._cache_data[key]
                self._cache_modified = True
                return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            cache_size_bytes = 0
            if os.path.exists(self.compressed_cache_file):
                cache_size_bytes = os.path.getsize(self.compressed_cache_file)
            elif os.path.exists(self.cache_file):
                cache_size_bytes = os.path.getsize(self.cache_file)
            
            return {
                'total_entries': len(self._cache_data),
                'cache_size_bytes': cache_size_bytes,
                'cache_size_mb': round(cache_size_bytes / (1024 * 1024), 2),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'compression_enabled': self.compression_enabled,
                'last_cleanup': self._last_cleanup,
                'cache_modified': self._cache_modified
            }
    
    def _is_entry_valid(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        if '_cache_timestamp' not in entry:
            return True  # Old entries without timestamp are considered valid
        
        timestamp = entry['_cache_timestamp']
        ttl_hours = entry.get('_cache_ttl_hours', 24)
        age_seconds = time.time() - timestamp
        max_age_seconds = ttl_hours * 3600
        
        return age_seconds < max_age_seconds
    
    def _should_cleanup(self) -> bool:
        """Check if cache needs cleanup."""
        # Cleanup if cache size is too large
        if len(self._cache_data) > 1000:  # Arbitrary threshold
            return True
        
        # Cleanup if last cleanup was too long ago
        if time.time() - self._last_cleanup > self.auto_cleanup_interval:
            return True
        
        return False
    
    def _auto_cleanup(self):
        """Perform automatic cache cleanup."""
        with self._lock:
            logger.info("Starting automatic cache cleanup...")
            
            original_count = len(self._cache_data)
            
            # Remove expired entries
            self._cleanup_old_entries()
            
            # Remove oldest entries if cache is still too large
            self._cleanup_by_size()
            
            self._last_cleanup = time.time()
            
            cleaned_count = original_count - len(self._cache_data)
            if cleaned_count > 0:
                logger.info(f"Cache cleanup completed: removed {cleaned_count} entries")
                self._cache_modified = True
    
    def _cleanup_old_entries(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache_data.items():
            if not self._is_entry_valid(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache_data[key]
        
        if expired_keys:
            logger.debug(f"Removed {len(expired_keys)} expired entries")
    
    def _cleanup_by_size(self):
        """Remove oldest entries if cache exceeds size limit."""
        if len(self._cache_data) <= 500:  # Keep reasonable number of entries
            return
        
        # Sort entries by timestamp (oldest first)
        sorted_entries = sorted(
            self._cache_data.items(),
            key=lambda x: x[1].get('_cache_timestamp', 0)
        )
        
        # Keep only the newest 500 entries
        entries_to_keep = dict(sorted_entries[-500:])
        removed_count = len(self._cache_data) - len(entries_to_keep)
        
        if removed_count > 0:
            self._cache_data = entries_to_keep
            logger.debug(f"Removed {removed_count} old entries to maintain size limit")

class CacheMetrics:
    """Cache performance metrics collector."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.start_time = time.time()
    
    def record_hit(self):
        self.hits += 1
        self.total_requests += 1
    
    def record_miss(self):
        self.misses += 1
        self.total_requests += 1
    
    def get_hit_ratio(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    def get_stats(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': self.total_requests,
            'hit_ratio': self.get_hit_ratio(),
            'uptime_seconds': uptime
        }

# Global instance for application use
cache_manager = OptimizedCacheManager()
cache_metrics = CacheMetrics()