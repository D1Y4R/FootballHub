"""
Cache Optimization Module for Football Prediction System
Addresses the 4.4MB cache file performance bottleneck
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CacheOptimizer:
    """Optimizes prediction cache performance and memory usage"""
    
    def __init__(self, cache_file_path: str = 'predictions_cache.json'):
        self.cache_file_path = cache_file_path
        self.backup_file_path = cache_file_path.replace('.json', '_backup.json')
        
        # Cache management settings
        self.max_file_size = 100 * 1024 * 1024  # 100MB max
        self.max_entries = 10000  # Maximum cache entries
        self.max_age_days = 7  # Keep entries for 7 days
        self.cleanup_threshold = 50 * 1024 * 1024  # Cleanup when file > 50MB
        
    def get_cache_size(self) -> int:
        """Get current cache file size in bytes"""
        try:
            if os.path.exists(self.cache_file_path):
                return os.path.getsize(self.cache_file_path)
            return 0
        except OSError:
            return 0
    
    def should_cleanup(self) -> bool:
        """Check if cache cleanup is needed"""
        file_size = self.get_cache_size()
        return file_size > self.cleanup_threshold
    
    def load_cache_optimized(self) -> Dict[str, Any]:
        """Load cache with size checking and automatic cleanup"""
        try:
            if not os.path.exists(self.cache_file_path):
                logger.info("Cache file doesn't exist, starting with empty cache")
                return {}
            
            file_size = self.get_cache_size()
            logger.info(f"Loading cache file: {file_size / (1024*1024):.2f} MB")
            
            # If file is too large, cleanup before loading
            if file_size > self.max_file_size:
                logger.warning(f"Cache file too large ({file_size / (1024*1024):.2f} MB), performing cleanup...")
                return self.cleanup_and_load()
            
            # Load normally
            with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Clean old entries
            cleaned_cache = self.clean_old_entries(cache_data)
            
            # If entries were removed, save the cleaned cache
            if len(cleaned_cache) < len(cache_data):
                self.save_cache_optimized(cleaned_cache)
                logger.info(f"Cleaned cache: {len(cache_data)} -> {len(cleaned_cache)} entries")
            
            return cleaned_cache
            
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return {}
    
    def cleanup_and_load(self) -> Dict[str, Any]:
        """Perform aggressive cleanup and load cache"""
        try:
            # Create backup
            if os.path.exists(self.cache_file_path):
                import shutil
                shutil.copy2(self.cache_file_path, self.backup_file_path)
                logger.info("Created cache backup")
            
            # Load raw data
            with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                raw_cache = json.load(f)
            
            # Aggressive cleanup
            cleaned_cache = self.aggressive_cleanup(raw_cache)
            
            # Save cleaned cache
            self.save_cache_optimized(cleaned_cache)
            
            # Remove backup if successful
            if os.path.exists(self.backup_file_path):
                os.remove(self.backup_file_path)
            
            new_size = self.get_cache_size()
            logger.info(f"Cache cleanup complete: {new_size / (1024*1024):.2f} MB")
            
            return cleaned_cache
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            # Restore backup if available
            self.restore_backup()
            return {}
    
    def clean_old_entries(self, cache_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove entries older than max_age_days"""
        if not cache_data:
            return {}
        
        cutoff_time = datetime.now() - timedelta(days=self.max_age_days)
        cleaned_cache = {}
        
        for key, value in cache_data.items():
            if not isinstance(value, dict):
                continue
                
            entry_time = self.get_entry_timestamp(value)
            if entry_time and entry_time > cutoff_time:
                cleaned_cache[key] = value
        
        return cleaned_cache
    
    def aggressive_cleanup(self, cache_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform aggressive cleanup keeping only the most recent entries"""
        if not cache_data:
            return {}
        
        # First clean old entries
        cleaned_cache = self.clean_old_entries(cache_data)
        
        # If still too many entries, keep only the most recent ones
        if len(cleaned_cache) > self.max_entries:
            # Sort by timestamp and keep most recent
            entries_with_time = []
            for key, value in cleaned_cache.items():
                timestamp = self.get_entry_timestamp(value)
                if timestamp:
                    entries_with_time.append((key, value, timestamp))
            
            # Sort by timestamp (newest first) and take top entries
            entries_with_time.sort(key=lambda x: x[2], reverse=True)
            cleaned_cache = {
                key: value 
                for key, value, _ in entries_with_time[:self.max_entries]
            }
            
            logger.info(f"Reduced cache entries to {len(cleaned_cache)}")
        
        return cleaned_cache
    
    def get_entry_timestamp(self, entry: Dict[str, Any]) -> Optional[datetime]:
        """Extract timestamp from cache entry"""
        try:
            timestamp_str = entry.get('timestamp')
            if timestamp_str:
                # Handle different timestamp formats
                if isinstance(timestamp_str, str):
                    if 'T' in timestamp_str:
                        # ISO format
                        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        # Simple date format
                        return datetime.fromisoformat(timestamp_str)
            return None
        except (ValueError, TypeError):
            return None
    
    def save_cache_optimized(self, cache_data: Dict[str, Any]) -> bool:
        """Save cache with optimization checks"""
        try:
            # Add timestamps to entries that don't have them
            current_time = datetime.now().isoformat()
            for key, value in cache_data.items():
                if isinstance(value, dict) and 'timestamp' not in value:
                    value['timestamp'] = current_time
            
            # Check if cleanup is needed before saving
            if len(cache_data) > self.max_entries:
                cache_data = self.aggressive_cleanup(cache_data)
            
            # Save to file
            with open(self.cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            # Check final file size
            final_size = self.get_cache_size()
            logger.info(f"Cache saved: {len(cache_data)} entries, {final_size / (1024*1024):.2f} MB")
            
            # If still too large, perform more aggressive cleanup
            if final_size > self.max_file_size:
                logger.warning("Cache still too large after save, performing additional cleanup")
                reduced_cache = dict(list(cache_data.items())[:self.max_entries // 2])
                with open(self.cache_file_path, 'w', encoding='utf-8') as f:
                    json.dump(reduced_cache, f, ensure_ascii=False, indent=2)
                logger.info(f"Reduced cache to {len(reduced_cache)} entries")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving optimized cache: {e}")
            return False
    
    def restore_backup(self) -> bool:
        """Restore cache from backup"""
        try:
            if os.path.exists(self.backup_file_path):
                import shutil
                shutil.copy2(self.backup_file_path, self.cache_file_path)
                os.remove(self.backup_file_path)
                logger.info("Cache restored from backup")
                return True
            return False
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            file_size = self.get_cache_size()
            
            if file_size == 0:
                return {
                    'file_size_mb': 0,
                    'entry_count': 0,
                    'needs_cleanup': False,
                    'health': 'empty'
                }
            
            # Try to count entries without loading full cache
            entry_count = 0
            try:
                with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    entry_count = len(cache_data)
            except:
                entry_count = -1  # Error reading
            
            needs_cleanup = self.should_cleanup()
            
            # Determine health status
            if file_size > self.max_file_size:
                health = 'critical'
            elif needs_cleanup:
                health = 'warning'
            else:
                health = 'good'
            
            return {
                'file_size_mb': round(file_size / (1024*1024), 2),
                'entry_count': entry_count,
                'needs_cleanup': needs_cleanup,
                'health': health,
                'max_size_mb': round(self.max_file_size / (1024*1024), 2),
                'max_entries': self.max_entries
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                'file_size_mb': 0,
                'entry_count': -1,
                'needs_cleanup': True,
                'health': 'error'
            }

# Convenience functions for easy integration
def optimize_existing_cache(cache_file_path: str = 'predictions_cache.json') -> bool:
    """Optimize an existing cache file"""
    optimizer = CacheOptimizer(cache_file_path)
    
    if not optimizer.should_cleanup():
        logger.info("Cache is already optimized")
        return True
    
    logger.info("Starting cache optimization...")
    cache_data = optimizer.cleanup_and_load()
    return len(cache_data) > 0

def get_cache_health() -> Dict[str, Any]:
    """Get cache health status"""
    optimizer = CacheOptimizer()
    return optimizer.get_cache_stats()

if __name__ == "__main__":
    # CLI usage for manual cache optimization
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'optimize':
        cache_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions_cache.json'
        print(f"Optimizing cache: {cache_file}")
        
        if optimize_existing_cache(cache_file):
            print("Cache optimization completed successfully")
        else:
            print("Cache optimization failed")
            
    elif len(sys.argv) > 1 and sys.argv[1] == 'stats':
        stats = get_cache_health()
        print("Cache Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("Usage:")
        print("  python cache_optimizer.py optimize [cache_file]")
        print("  python cache_optimizer.py stats")