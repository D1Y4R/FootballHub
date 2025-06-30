#!/usr/bin/env python3
"""
Performance Monitoring Module
Tracks application performance metrics and provides optimization insights.
"""

import time
import psutil
import logging
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
from functools import wraps

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric data."""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceSnapshot:
    """Complete performance snapshot at a point in time."""
    timestamp: datetime
    cpu_percent: float
    memory_rss: int
    memory_vms: int
    memory_percent: float
    disk_io: Dict[str, int]
    network_io: Dict[str, int]
    response_times: List[float]
    cache_hit_ratio: float
    active_connections: int

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    """
    
    def __init__(self, 
                 collection_interval: int = 30,
                 max_history: int = 1000,
                 auto_start: bool = True):
        
        self.collection_interval = collection_interval
        self.max_history = max_history
        self._metrics_history: List[PerformanceSnapshot] = []
        self._response_times: List[float] = []
        self._cache_stats = {'hits': 0, 'misses': 0}
        self._active_connections = 0
        
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.RLock()
        
        # Performance thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'response_time_avg': 2.0,
            'cache_hit_ratio': 0.7,
            'disk_io_high': 100_000_000  # 100MB
        }
        
        if auto_start:
            self.start_monitoring()
    
    def start_monitoring(self):
        """Start background performance monitoring."""
        if self._monitoring:
            logger.warning("Performance monitoring already running")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Performance monitoring started (interval: {self.collection_interval}s)")
    
    def stop_monitoring(self):
        """Stop background performance monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitor_worker(self):
        """Background worker for collecting performance metrics."""
        while self._monitoring:
            try:
                snapshot = self._collect_snapshot()
                with self._lock:
                    self._metrics_history.append(snapshot)
                    
                    # Maintain history size
                    if len(self._metrics_history) > self.max_history:
                        self._metrics_history = self._metrics_history[-self.max_history:]
                
                # Check for performance issues
                self._check_thresholds(snapshot)
                
            except Exception as e:
                logger.error(f"Error collecting performance metrics: {e}")
            
            time.sleep(self.collection_interval)
    
    def _collect_snapshot(self) -> PerformanceSnapshot:
        """Collect current performance snapshot."""
        process = psutil.Process()
        
        # CPU and Memory
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # Disk I/O
        try:
            disk_io = process.io_counters()
            disk_io_dict = {
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes,
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count
            }
        except (AttributeError, psutil.AccessDenied):
            disk_io_dict = {'read_bytes': 0, 'write_bytes': 0, 'read_count': 0, 'write_count': 0}
        
        # Network I/O (system-wide)
        try:
            net_io = psutil.net_io_counters()
            network_io_dict = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
        except (AttributeError, psutil.AccessDenied):
            network_io_dict = {'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 'packets_recv': 0}
        
        # Application-specific metrics
        with self._lock:
            recent_response_times = self._response_times[-100:] if self._response_times else []
            cache_hit_ratio = self._calculate_cache_hit_ratio()
            active_connections = self._active_connections
        
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_rss=memory_info.rss,
            memory_vms=memory_info.vms,
            memory_percent=memory_percent,
            disk_io=disk_io_dict,
            network_io=network_io_dict,
            response_times=recent_response_times,
            cache_hit_ratio=cache_hit_ratio,
            active_connections=active_connections
        )
    
    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total = self._cache_stats['hits'] + self._cache_stats['misses']
        if total == 0:
            return 0.0
        return self._cache_stats['hits'] / total
    
    def _check_thresholds(self, snapshot: PerformanceSnapshot):
        """Check performance thresholds and log warnings."""
        warnings = []
        
        if snapshot.cpu_percent > self.thresholds['cpu_percent']:
            warnings.append(f"High CPU usage: {snapshot.cpu_percent:.1f}%")
        
        if snapshot.memory_percent > self.thresholds['memory_percent']:
            warnings.append(f"High memory usage: {snapshot.memory_percent:.1f}%")
        
        if snapshot.response_times:
            avg_response_time = sum(snapshot.response_times) / len(snapshot.response_times)
            if avg_response_time > self.thresholds['response_time_avg']:
                warnings.append(f"Slow response times: {avg_response_time:.2f}s avg")
        
        if snapshot.cache_hit_ratio < self.thresholds['cache_hit_ratio']:
            warnings.append(f"Low cache hit ratio: {snapshot.cache_hit_ratio:.2f}")
        
        disk_io_total = snapshot.disk_io['read_bytes'] + snapshot.disk_io['write_bytes']
        if disk_io_total > self.thresholds['disk_io_high']:
            warnings.append(f"High disk I/O: {disk_io_total / 1024 / 1024:.1f} MB")
        
        if warnings:
            logger.warning(f"Performance issues detected: {'; '.join(warnings)}")
    
    def record_response_time(self, duration: float):
        """Record an API response time."""
        with self._lock:
            self._response_times.append(duration)
            # Keep only recent measurements
            if len(self._response_times) > 1000:
                self._response_times = self._response_times[-500:]
    
    def record_cache_hit(self):
        """Record a cache hit."""
        with self._lock:
            self._cache_stats['hits'] += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        with self._lock:
            self._cache_stats['misses'] += 1
    
    def record_connection(self, active: bool):
        """Record connection status change."""
        with self._lock:
            if active:
                self._active_connections += 1
            else:
                self._active_connections = max(0, self._active_connections - 1)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        try:
            current_snapshot = self._collect_snapshot()
            
            with self._lock:
                recent_response_times = self._response_times[-100:] if self._response_times else []
            
            stats = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': current_snapshot.cpu_percent,
                'memory': {
                    'rss_mb': current_snapshot.memory_rss / 1024 / 1024,
                    'vms_mb': current_snapshot.memory_vms / 1024 / 1024,
                    'percent': current_snapshot.memory_percent
                },
                'disk_io': current_snapshot.disk_io,
                'network_io': current_snapshot.network_io,
                'response_times': {
                    'count': len(recent_response_times),
                    'avg': sum(recent_response_times) / len(recent_response_times) if recent_response_times else 0,
                    'min': min(recent_response_times) if recent_response_times else 0,
                    'max': max(recent_response_times) if recent_response_times else 0,
                    'p95': self._percentile(recent_response_times, 95) if recent_response_times else 0
                },
                'cache': {
                    'hit_ratio': current_snapshot.cache_hit_ratio,
                    'hits': self._cache_stats['hits'],
                    'misses': self._cache_stats['misses']
                },
                'connections': {
                    'active': current_snapshot.active_connections
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting current stats: {e}")
            return {'error': str(e)}
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of a list of values."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        index = min(index, len(sorted_data) - 1)
        return sorted_data[index]
    
    def get_performance_report(self, hours: int = 1) -> Dict[str, Any]:
        """Get comprehensive performance report for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_snapshots = [
                snapshot for snapshot in self._metrics_history
                if snapshot.timestamp >= cutoff_time
            ]
        
        if not recent_snapshots:
            return {'error': 'No performance data available for the specified period'}
        
        # Calculate statistics
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        memory_values = [s.memory_percent for s in recent_snapshots]
        cache_ratios = [s.cache_hit_ratio for s in recent_snapshots]
        
        all_response_times = []
        for snapshot in recent_snapshots:
            all_response_times.extend(snapshot.response_times)
        
        report = {
            'period_hours': hours,
            'snapshots_count': len(recent_snapshots),
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values),
                'p95': self._percentile(cpu_values, 95)
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values),
                'min': min(memory_values),
                'max': max(memory_values),
                'p95': self._percentile(memory_values, 95)
            },
            'response_times': {
                'count': len(all_response_times),
                'avg': sum(all_response_times) / len(all_response_times) if all_response_times else 0,
                'min': min(all_response_times) if all_response_times else 0,
                'max': max(all_response_times) if all_response_times else 0,
                'p95': self._percentile(all_response_times, 95)
            },
            'cache': {
                'avg_hit_ratio': sum(cache_ratios) / len(cache_ratios) if cache_ratios else 0,
                'min_hit_ratio': min(cache_ratios) if cache_ratios else 0,
                'max_hit_ratio': max(cache_ratios) if cache_ratios else 0
            },
            'recommendations': self._generate_recommendations(recent_snapshots)
        }
        
        return report
    
    def _generate_recommendations(self, snapshots: List[PerformanceSnapshot]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if not snapshots:
            return recommendations
        
        # CPU recommendations
        avg_cpu = sum(s.cpu_percent for s in snapshots) / len(snapshots)
        if avg_cpu > 70:
            recommendations.append("High CPU usage detected. Consider optimizing algorithms or adding more workers.")
        
        # Memory recommendations
        avg_memory = sum(s.memory_percent for s in snapshots) / len(snapshots)
        if avg_memory > 80:
            recommendations.append("High memory usage detected. Consider implementing memory optimization or increasing server memory.")
        
        # Response time recommendations
        all_response_times = []
        for snapshot in snapshots:
            all_response_times.extend(snapshot.response_times)
        
        if all_response_times:
            avg_response_time = sum(all_response_times) / len(all_response_times)
            if avg_response_time > 2.0:
                recommendations.append("Slow response times detected. Consider caching, database optimization, or async processing.")
        
        # Cache recommendations
        avg_cache_ratio = sum(s.cache_hit_ratio for s in snapshots) / len(snapshots)
        if avg_cache_ratio < 0.6:
            recommendations.append("Low cache hit ratio. Consider increasing cache size or improving cache strategy.")
        
        # Disk I/O recommendations
        total_disk_reads = sum(s.disk_io['read_bytes'] for s in snapshots)
        total_disk_writes = sum(s.disk_io['write_bytes'] for s in snapshots)
        if total_disk_reads + total_disk_writes > 1_000_000_000:  # 1GB
            recommendations.append("High disk I/O detected. Consider optimizing database queries or using SSD storage.")
        
        return recommendations
    
    def export_metrics(self, filename: str):
        """Export performance metrics to JSON file."""
        try:
            with self._lock:
                data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'metrics_count': len(self._metrics_history),
                    'collection_interval': self.collection_interval,
                    'snapshots': [
                        {
                            'timestamp': s.timestamp.isoformat(),
                            'cpu_percent': s.cpu_percent,
                            'memory_rss': s.memory_rss,
                            'memory_vms': s.memory_vms,
                            'memory_percent': s.memory_percent,
                            'disk_io': s.disk_io,
                            'network_io': s.network_io,
                            'response_times_count': len(s.response_times),
                            'response_times_avg': sum(s.response_times) / len(s.response_times) if s.response_times else 0,
                            'cache_hit_ratio': s.cache_hit_ratio,
                            'active_connections': s.active_connections
                        }
                        for s in self._metrics_history
                    ]
                }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Performance metrics exported to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

# Decorators for automatic performance monitoring

def monitor_response_time(monitor_instance: PerformanceMonitor):
    """Decorator to automatically monitor function response times."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                monitor_instance.record_response_time(duration)
        return wrapper
    return decorator

def monitor_cache_access(monitor_instance: PerformanceMonitor):
    """Decorator to automatically monitor cache hit/miss rates."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Detect cache hit/miss based on result or function name
            if hasattr(result, 'get') and result.get('from_cache'):
                monitor_instance.record_cache_hit()
            elif 'cache' in func.__name__.lower():
                # Heuristic: if function has 'cache' in name and returns None, it's likely a miss
                if result is None:
                    monitor_instance.record_cache_miss()
                else:
                    monitor_instance.record_cache_hit()
            
            return result
        return wrapper
    return decorator

# Global instance for application use
performance_monitor = PerformanceMonitor()

# Cleanup function
def cleanup_performance_monitor():
    """Clean up performance monitoring resources."""
    performance_monitor.stop_monitoring()
    logger.info("Performance monitor cleaned up")