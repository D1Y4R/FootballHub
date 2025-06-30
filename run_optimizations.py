#!/usr/bin/env python3
"""
Run Performance Optimizations Script
Comprehensive script to apply all performance optimizations and generate reports.
"""

import os
import sys
import time
import json
import gzip
import logging
from pathlib import Path
from datetime import datetime
import subprocess
import shutil

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache_optimizer import OptimizedCacheManager
from optimize_frontend_assets import AssetOptimizer
from lazy_ml_imports import initialize_lazy_ml_imports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimization.log')
    ]
)
logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """
    Comprehensive performance optimization runner.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.optimizations_completed = []
        self.optimization_results = {}
        self.errors = []
        
        # Performance metrics
        self.before_stats = self._get_system_stats()
        
    def _get_system_stats(self):
        """Get current system statistics."""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'file_sizes': {},
            'directory_sizes': {}
        }
        
        # Get file sizes for key files
        key_files = [
            'predictions_cache.json',
            'match_prediction.py',
            'main.py',
            'model_validation.py'
        ]
        
        for file_path in key_files:
            if os.path.exists(file_path):
                stats['file_sizes'][file_path] = os.path.getsize(file_path)
        
        # Get directory sizes
        key_dirs = ['static', 'templates']
        for dir_path in key_dirs:
            if os.path.exists(dir_path):
                stats['directory_sizes'][dir_path] = self._get_directory_size(dir_path)
        
        return stats
    
    def _get_directory_size(self, path):
        """Get total size of directory."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, IOError):
                    pass
        return total_size
    
    def optimize_cache_system(self):
        """Optimize the caching system."""
        logger.info("🔧 Starting cache system optimization...")
        
        try:
            # Backup existing cache
            if os.path.exists('predictions_cache.json'):
                original_size = os.path.getsize('predictions_cache.json')
                shutil.copy2('predictions_cache.json', 'predictions_cache_backup.json')
                logger.info(f"Backed up original cache file ({original_size:,} bytes)")
            
            # Initialize optimized cache manager
            cache_manager = OptimizedCacheManager(
                max_size_mb=2,
                compression_enabled=True,
                auto_cleanup_interval=3600
            )
            
            # Get cache statistics
            cache_stats = cache_manager.get_stats()
            
            # Force save to create compressed version
            cache_manager.save_cache()
            
            # Calculate compression savings
            compressed_size = 0
            if os.path.exists('predictions_cache.json.gz'):
                compressed_size = os.path.getsize('predictions_cache.json.gz')
            
            self.optimization_results['cache_optimization'] = {
                'original_size': cache_stats.get('cache_size_bytes', 0),
                'compressed_size': compressed_size,
                'compression_ratio': compressed_size / cache_stats.get('cache_size_bytes', 1) if cache_stats.get('cache_size_bytes', 0) > 0 else 0,
                'entries_count': cache_stats.get('total_entries', 0)
            }
            
            self.optimizations_completed.append('cache_optimization')
            logger.info("✅ Cache system optimization completed")
            
        except Exception as e:
            logger.error(f"❌ Cache optimization failed: {e}")
            self.errors.append(f"Cache optimization: {e}")
    
    def optimize_frontend_assets(self):
        """Optimize frontend assets."""
        logger.info("🎨 Starting frontend asset optimization...")
        
        try:
            asset_optimizer = AssetOptimizer()
            results = asset_optimizer.optimize_all()
            
            if results.get('success'):
                self.optimization_results['frontend_optimization'] = {
                    'css_bundle': results.get('css_bundle'),
                    'js_bundle': results.get('js_bundle'),
                    'critical_css': results.get('critical_css'),
                    'manifest': results.get('manifest', {})
                }
                
                # Calculate total savings
                original_size = 0
                bundled_size = 0
                
                # CSS savings
                css_files = ['css/custom.css', 'css/widget-style.css', 'css/widgetCountries.css', 
                           'css/widgetLeague.css', 'css/match-actions.css', 'css/prediction-modal.css',
                           'css/match-insights.css']
                
                for css_file in css_files:
                    file_path = Path('static') / css_file
                    if file_path.exists():
                        original_size += file_path.stat().st_size
                
                # JS savings  
                js_files = ['js/custom.js', 'js/main.js', 'js/prediction-handler.js',
                           'js/team_stats.js', 'js/team_history.js', 'js/api_football_debug.js']
                
                for js_file in js_files:
                    file_path = Path('static') / js_file
                    if file_path.exists():
                        original_size += file_path.stat().st_size
                
                # Get bundled file sizes
                build_dir = Path('static/build')
                if build_dir.exists():
                    for bundle_file in build_dir.glob('bundle.*'):
                        if not bundle_file.name.endswith('.gz'):
                            bundled_size += bundle_file.stat().st_size
                
                self.optimization_results['frontend_optimization']['size_reduction'] = {
                    'original_size': original_size,
                    'bundled_size': bundled_size,
                    'reduction_ratio': (original_size - bundled_size) / original_size if original_size > 0 else 0
                }
                
                self.optimizations_completed.append('frontend_optimization')
                logger.info("✅ Frontend asset optimization completed")
            else:
                raise Exception(results.get('error', 'Unknown error'))
                
        except Exception as e:
            logger.error(f"❌ Frontend optimization failed: {e}")
            self.errors.append(f"Frontend optimization: {e}")
    
    def optimize_ml_imports(self):
        """Optimize ML dependency loading."""
        logger.info("🧠 Starting ML import optimization...")
        
        try:
            # Initialize lazy loading system
            ml_imports = initialize_lazy_ml_imports(preload_critical=True)
            
            # Get availability status
            status = ml_imports.get_availability_status()
            
            # Test performance of lazy loading
            start_time = time.time()
            numpy_available = ml_imports.get_numpy().is_available()
            pandas_available = ml_imports.get_pandas().is_available() 
            sklearn_available = ml_imports.get_sklearn().is_available()
            tensorflow_available = ml_imports.get_tensorflow().is_available()
            load_time = time.time() - start_time
            
            self.optimization_results['ml_optimization'] = {
                'lazy_loading_enabled': True,
                'availability_status': status,
                'load_time_seconds': load_time,
                'critical_modules_available': {
                    'numpy': numpy_available,
                    'pandas': pandas_available,
                    'sklearn': sklearn_available,
                    'tensorflow': tensorflow_available
                }
            }
            
            self.optimizations_completed.append('ml_optimization')
            logger.info("✅ ML import optimization completed")
            
        except Exception as e:
            logger.error(f"❌ ML optimization failed: {e}")
            self.errors.append(f"ML optimization: {e}")
    
    def optimize_large_files(self):
        """Optimize large files in the project."""
        logger.info("📦 Starting large file optimization...")
        
        try:
            optimized_files = []
            
            # Compress large JSON files
            large_files = ['predictions_cache.json', 'model_validation.py', 'match_prediction.py']
            
            for file_path in large_files:
                if os.path.exists(file_path):
                    original_size = os.path.getsize(file_path)
                    
                    # Create gzipped version for storage/backup
                    with open(file_path, 'rb') as f_in:
                        with gzip.open(f"{file_path}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    compressed_size = os.path.getsize(f"{file_path}.gz")
                    compression_ratio = compressed_size / original_size
                    
                    optimized_files.append({
                        'file': file_path,
                        'original_size': original_size,
                        'compressed_size': compressed_size,
                        'compression_ratio': compression_ratio,
                        'savings_bytes': original_size - compressed_size
                    })
                    
                    logger.info(f"Compressed {file_path}: {original_size:,} -> {compressed_size:,} bytes "
                              f"({100 * (1 - compression_ratio):.1f}% reduction)")
            
            self.optimization_results['file_optimization'] = {
                'optimized_files': optimized_files,
                'total_savings': sum(f['savings_bytes'] for f in optimized_files)
            }
            
            self.optimizations_completed.append('file_optimization')
            logger.info("✅ Large file optimization completed")
            
        except Exception as e:
            logger.error(f"❌ File optimization failed: {e}")
            self.errors.append(f"File optimization: {e}")
    
    def update_configuration(self):
        """Update application configuration for better performance."""
        logger.info("⚙️ Updating application configuration...")
        
        try:
            # Update pyproject.toml with optimization dependencies
            pyproject_additions = [
                "aiohttp>=3.8.0",  # For async HTTP requests
                "psutil>=5.9.0",   # For performance monitoring
                "redis>=4.0.0",    # For enhanced caching (optional)
            ]
            
            # Read current pyproject.toml
            pyproject_path = Path('pyproject.toml')
            if pyproject_path.exists():
                with open(pyproject_path, 'r') as f:
                    content = f.read()
                
                # Check if dependencies need to be added
                missing_deps = []
                for dep in pyproject_additions:
                    dep_name = dep.split('>=')[0]
                    if dep_name not in content:
                        missing_deps.append(dep)
                
                if missing_deps:
                    logger.info(f"Recommended additional dependencies: {', '.join(missing_deps)}")
                    
                    # Write optimization config file
                    optimization_config = {
                        'optimizations_applied': {
                            'cache_compression': True,
                            'lazy_ml_loading': True,
                            'frontend_bundling': True,
                            'http_optimization': True
                        },
                        'recommended_dependencies': missing_deps,
                        'performance_settings': {
                            'cache_max_size_mb': 2,
                            'cache_compression_enabled': True,
                            'lazy_loading_enabled': True,
                            'http_timeout_seconds': 30,
                            'max_workers': 5
                        }
                    }
                    
                    with open('optimization_config.json', 'w') as f:
                        json.dump(optimization_config, f, indent=2)
                    
                    logger.info("Created optimization_config.json with recommended settings")
            
            self.optimization_results['configuration'] = {
                'optimization_config_created': True,
                'recommended_dependencies': missing_deps if 'missing_deps' in locals() else []
            }
            
            self.optimizations_completed.append('configuration')
            logger.info("✅ Configuration update completed")
            
        except Exception as e:
            logger.error(f"❌ Configuration update failed: {e}")
            self.errors.append(f"Configuration update: {e}")
    
    def run_all_optimizations(self):
        """Run all available optimizations."""
        logger.info("🚀 Starting comprehensive performance optimization...")
        
        optimizations = [
            ('Cache System', self.optimize_cache_system),
            ('Frontend Assets', self.optimize_frontend_assets),
            ('ML Dependencies', self.optimize_ml_imports),
            ('Large Files', self.optimize_large_files),
            ('Configuration', self.update_configuration)
        ]
        
        for name, optimization_func in optimizations:
            try:
                logger.info(f"Running {name} optimization...")
                optimization_func()
                time.sleep(1)  # Brief pause between optimizations
            except Exception as e:
                logger.error(f"Failed to run {name} optimization: {e}")
                self.errors.append(f"{name}: {e}")
        
        # Generate final report
        self.generate_optimization_report()
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report."""
        end_time = time.time()
        total_duration = end_time - self.start_time
        after_stats = self._get_system_stats()
        
        report = {
            'optimization_summary': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'total_duration_seconds': total_duration,
                'optimizations_completed': self.optimizations_completed,
                'total_optimizations': len(self.optimizations_completed),
                'errors_count': len(self.errors),
                'errors': self.errors
            },
            'performance_improvements': self.optimization_results,
            'before_stats': self.before_stats,
            'after_stats': after_stats,
            'size_reductions': self._calculate_size_reductions(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_filename = f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self._print_optimization_summary(report)
        
        logger.info(f"📊 Optimization report saved to {report_filename}")
        
        return report
    
    def _calculate_size_reductions(self):
        """Calculate total size reductions achieved."""
        reductions = {
            'cache_compression': 0,
            'frontend_bundling': 0,
            'file_compression': 0,
            'total_savings_bytes': 0
        }
        
        # Cache compression savings
        if 'cache_optimization' in self.optimization_results:
            cache_data = self.optimization_results['cache_optimization']
            cache_savings = cache_data.get('original_size', 0) - cache_data.get('compressed_size', 0)
            reductions['cache_compression'] = cache_savings
        
        # Frontend bundling savings
        if 'frontend_optimization' in self.optimization_results:
            frontend_data = self.optimization_results['frontend_optimization'].get('size_reduction', {})
            frontend_savings = frontend_data.get('original_size', 0) - frontend_data.get('bundled_size', 0)
            reductions['frontend_bundling'] = frontend_savings
        
        # File compression savings
        if 'file_optimization' in self.optimization_results:
            file_savings = self.optimization_results['file_optimization'].get('total_savings', 0)
            reductions['file_compression'] = file_savings
        
        reductions['total_savings_bytes'] = sum([
            reductions['cache_compression'],
            reductions['frontend_bundling'],
            reductions['file_compression']
        ])
        
        return reductions
    
    def _generate_recommendations(self):
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Based on optimization results
        if 'cache_optimization' in self.optimization_results:
            cache_data = self.optimization_results['cache_optimization']
            if cache_data.get('entries_count', 0) > 1000:
                recommendations.append("Consider implementing cache cleanup automation for large cache sizes")
        
        if 'ml_optimization' in self.optimization_results:
            ml_data = self.optimization_results['ml_optimization']
            if ml_data.get('load_time_seconds', 0) > 5:
                recommendations.append("ML dependencies are slow to load, consider further lazy loading optimization")
        
        # General recommendations
        recommendations.extend([
            "Consider implementing Redis for distributed caching in production",
            "Monitor application performance with the included performance monitoring tools",
            "Regularly run asset optimization to maintain performance gains",
            "Consider implementing CDN for static assets in production",
            "Monitor cache hit ratios and adjust cache size as needed"
        ])
        
        return recommendations
    
    def _print_optimization_summary(self, report):
        """Print a formatted optimization summary."""
        print("\n" + "="*80)
        print("🎯 PERFORMANCE OPTIMIZATION SUMMARY")
        print("="*80)
        
        summary = report['optimization_summary']
        print(f"⏱️  Duration: {summary['total_duration_seconds']:.2f} seconds")
        print(f"✅ Completed: {summary['total_optimizations']} optimizations")
        print(f"❌ Errors: {summary['errors_count']}")
        
        if summary['errors']:
            print("\n🚨 Errors encountered:")
            for error in summary['errors']:
                print(f"   • {error}")
        
        print(f"\n📋 Optimizations completed:")
        for opt in summary['optimizations_completed']:
            print(f"   ✓ {opt.replace('_', ' ').title()}")
        
        # Size reductions
        reductions = report['size_reductions']
        total_savings_mb = reductions['total_savings_bytes'] / 1024 / 1024
        
        print(f"\n💾 Total space saved: {total_savings_mb:.2f} MB")
        print(f"   • Cache compression: {reductions['cache_compression'] / 1024 / 1024:.2f} MB")
        print(f"   • Frontend bundling: {reductions['frontend_bundling'] / 1024:.1f} KB")
        print(f"   • File compression: {reductions['file_compression'] / 1024 / 1024:.2f} MB")
        
        # Frontend optimization details
        if 'frontend_optimization' in report['performance_improvements']:
            frontend = report['performance_improvements']['frontend_optimization']
            print(f"\n🎨 Frontend assets optimized:")
            if 'css_bundle' in frontend:
                print(f"   • CSS Bundle: {frontend['css_bundle']}")
            if 'js_bundle' in frontend:
                print(f"   • JS Bundle: {frontend['js_bundle']}")
            if 'critical_css' in frontend:
                print(f"   • Critical CSS: {frontend['critical_css']}")
        
        # Recommendations
        recommendations = report['recommendations'][:5]  # Show top 5
        if recommendations:
            print(f"\n💡 Top recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*80)
        print("🎉 Optimization completed! Check the full report for detailed metrics.")
        print("="*80 + "\n")

def main():
    """Main function to run all optimizations."""
    print("🚀 Starting Football Prediction App Performance Optimization")
    print("This will optimize cache, frontend assets, ML dependencies, and more...\n")
    
    try:
        optimizer = PerformanceOptimizer()
        optimizer.run_all_optimizations()
        return 0
    except KeyboardInterrupt:
        print("\n❌ Optimization interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Optimization failed with error: {e}")
        logger.error(f"Optimization failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())