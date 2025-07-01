"""
Static Asset Optimizer for Football Prediction System
Minifies and optimizes JavaScript and CSS files for better performance
"""
import os
import re
import gzip
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class StaticOptimizer:
    """Optimizes static assets for better performance"""
    
    def __init__(self, static_dir: str = 'static'):
        self.static_dir = Path(static_dir)
        self.js_dir = self.static_dir / 'js'
        self.css_dir = self.static_dir / 'css'
        self.output_dir = self.static_dir / 'optimized'
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'js').mkdir(exist_ok=True)
        (self.output_dir / 'css').mkdir(exist_ok=True)
    
    def minify_javascript(self, content: str) -> str:
        """Simple JavaScript minification"""
        # Remove comments
        content = re.sub(r'//.*', '', content)
        content = re.sub(r'/\*[\s\S]*?\*/', '', content)
        
        # Remove unnecessary whitespace
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r';\s*}', '}', content)
        content = re.sub(r'{\s*', '{', content)
        content = re.sub(r'}\s*', '}', content)
        content = re.sub(r';\s*', ';', content)
        
        # Remove whitespace around operators
        content = re.sub(r'\s*([+\-*/=<>!&|,;:{}()[\]])\s*', r'\1', content)
        
        return content.strip()
    
    def minify_css(self, content: str) -> str:
        """Simple CSS minification"""
        # Remove comments
        content = re.sub(r'/\*[\s\S]*?\*/', '', content)
        
        # Remove unnecessary whitespace
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r';\s*}', '}', content)
        content = re.sub(r'{\s*', '{', content)
        content = re.sub(r'}\s*', '}', content)
        content = re.sub(r';\s*', ';', content)
        content = re.sub(r':\s*', ':', content)
        
        # Remove whitespace around selectors
        content = re.sub(r'\s*([+>~,{}();:])\s*', r'\1', content)
        
        return content.strip()
    
    def optimize_file(self, file_path: Path, output_path: Path, file_type: str) -> Dict[str, Any]:
        """Optimize a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            original_size = len(original_content.encode('utf-8'))
            
            # Minify based on file type
            if file_type == 'js':
                minified_content = self.minify_javascript(original_content)
            elif file_type == 'css':
                minified_content = self.minify_css(original_content)
            else:
                minified_content = original_content
            
            minified_size = len(minified_content.encode('utf-8'))
            
            # Save minified version
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(minified_content)
            
            # Create gzipped version
            gzip_path = output_path.with_suffix(output_path.suffix + '.gz')
            with open(output_path, 'rb') as f_in:
                with gzip.open(gzip_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            gzip_size = os.path.getsize(gzip_path)
            
            savings = ((original_size - minified_size) / original_size) * 100
            gzip_savings = ((original_size - gzip_size) / original_size) * 100
            
            return {
                'file': file_path.name,
                'original_size': original_size,
                'minified_size': minified_size,
                'gzip_size': gzip_size,
                'savings_percent': round(savings, 1),
                'gzip_savings_percent': round(gzip_savings, 1),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error optimizing {file_path}: {e}")
            return {
                'file': file_path.name,
                'error': str(e),
                'success': False
            }
    
    def optimize_js_files(self) -> List[Dict[str, Any]]:
        """Optimize all JavaScript files"""
        results = []
        
        if not self.js_dir.exists():
            logger.warning(f"JavaScript directory {self.js_dir} does not exist")
            return results
        
        js_files = list(self.js_dir.glob('*.js'))
        logger.info(f"Found {len(js_files)} JavaScript files to optimize")
        
        for js_file in js_files:
            output_file = self.output_dir / 'js' / js_file.name
            result = self.optimize_file(js_file, output_file, 'js')
            results.append(result)
            
            if result['success']:
                logger.info(f"Optimized {js_file.name}: "
                           f"{result['original_size']} -> {result['minified_size']} bytes "
                           f"({result['savings_percent']}% savings)")
        
        return results
    
    def optimize_css_files(self) -> List[Dict[str, Any]]:
        """Optimize all CSS files"""
        results = []
        
        if not self.css_dir.exists():
            logger.warning(f"CSS directory {self.css_dir} does not exist")
            return results
        
        css_files = list(self.css_dir.glob('*.css'))
        logger.info(f"Found {len(css_files)} CSS files to optimize")
        
        for css_file in css_files:
            output_file = self.output_dir / 'css' / css_file.name
            result = self.optimize_file(css_file, output_file, 'css')
            results.append(result)
            
            if result['success']:
                logger.info(f"Optimized {css_file.name}: "
                           f"{result['original_size']} -> {result['minified_size']} bytes "
                           f"({result['savings_percent']}% savings)")
        
        return results
    
    def create_bundle(self, file_list: List[str], output_name: str, file_type: str) -> Dict[str, Any]:
        """Create a bundled and minified file from multiple files"""
        try:
            combined_content = ""
            original_size = 0
            
            base_dir = self.js_dir if file_type == 'js' else self.css_dir
            
            for filename in file_list:
                file_path = base_dir / filename
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        combined_content += f"\n/* {filename} */\n{content}\n"
                        original_size += len(content.encode('utf-8'))
                else:
                    logger.warning(f"File not found for bundling: {file_path}")
            
            # Minify combined content
            if file_type == 'js':
                minified_content = self.minify_javascript(combined_content)
            else:
                minified_content = self.minify_css(combined_content)
            
            # Save bundle
            output_path = self.output_dir / file_type / output_name
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(minified_content)
            
            # Create gzipped version
            gzip_path = output_path.with_suffix(output_path.suffix + '.gz')
            with open(output_path, 'rb') as f_in:
                with gzip.open(gzip_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            minified_size = len(minified_content.encode('utf-8'))
            gzip_size = os.path.getsize(gzip_path)
            
            savings = ((original_size - minified_size) / original_size) * 100
            gzip_savings = ((original_size - gzip_size) / original_size) * 100
            
            return {
                'bundle': output_name,
                'files_count': len(file_list),
                'original_size': original_size,
                'minified_size': minified_size,
                'gzip_size': gzip_size,
                'savings_percent': round(savings, 1),
                'gzip_savings_percent': round(gzip_savings, 1),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error creating bundle {output_name}: {e}")
            return {
                'bundle': output_name,
                'error': str(e),
                'success': False
            }
    
    def optimize_all(self) -> Dict[str, Any]:
        """Optimize all static assets"""
        logger.info("Starting static asset optimization...")
        
        # Optimize individual files
        js_results = self.optimize_js_files()
        css_results = self.optimize_css_files()
        
        # Create common bundles
        core_js_files = ['main.js', 'custom.js', 'prediction-handler.js']
        stats_js_files = ['team_stats.js', 'team_history.js']
        
        js_bundles = []
        css_bundles = []
        
        # Create JS bundles if files exist
        available_core_files = [f for f in core_js_files if (self.js_dir / f).exists()]
        if available_core_files:
            core_bundle = self.create_bundle(available_core_files, 'app-core.min.js', 'js')
            js_bundles.append(core_bundle)
        
        available_stats_files = [f for f in stats_js_files if (self.js_dir / f).exists()]
        if available_stats_files:
            stats_bundle = self.create_bundle(available_stats_files, 'app-stats.min.js', 'js')
            js_bundles.append(stats_bundle)
        
        # Create CSS bundle
        all_css_files = [f.name for f in self.css_dir.glob('*.css') if f.is_file()]
        if all_css_files:
            css_bundle = self.create_bundle(all_css_files, 'app-styles.min.css', 'css')
            css_bundles.append(css_bundle)
        
        # Calculate total savings
        total_original = sum(r.get('original_size', 0) for r in js_results + css_results if r['success'])
        total_minified = sum(r.get('minified_size', 0) for r in js_results + css_results if r['success'])
        total_gzip = sum(r.get('gzip_size', 0) for r in js_results + css_results if r['success'])
        
        total_savings = ((total_original - total_minified) / total_original * 100) if total_original > 0 else 0
        total_gzip_savings = ((total_original - total_gzip) / total_original * 100) if total_original > 0 else 0
        
        return {
            'js_files': js_results,
            'css_files': css_results,
            'js_bundles': js_bundles,
            'css_bundles': css_bundles,
            'summary': {
                'total_files': len(js_results) + len(css_results),
                'successful_optimizations': len([r for r in js_results + css_results if r['success']]),
                'total_original_size': total_original,
                'total_minified_size': total_minified,
                'total_gzip_size': total_gzip,
                'total_savings_percent': round(total_savings, 1),
                'total_gzip_savings_percent': round(total_gzip_savings, 1),
                'output_directory': str(self.output_dir)
            }
        }
    
    def generate_usage_templates(self) -> str:
        """Generate template code for using optimized assets"""
        template = """
<!-- Use optimized static assets -->
<!-- CSS Bundle -->
<link rel="stylesheet" href="{{ url_for('static', filename='optimized/css/app-styles.min.css') }}">

<!-- Core JavaScript Bundle -->
<script src="{{ url_for('static', filename='optimized/js/app-core.min.js') }}" defer></script>

<!-- Stats JavaScript Bundle (load on demand) -->
<script src="{{ url_for('static', filename='optimized/js/app-stats.min.js') }}" defer></script>

<!-- Enable gzip compression in Flask -->
<!-- Add to your app configuration: -->
"""
        
        flask_config = """
# In your Flask app
from flask_compress import Compress

app = Flask(__name__)
Compress(app)  # Enable gzip compression

# Or set up nginx/apache to serve .gz files directly
"""
        
        return template + flask_config

def optimize_static_assets(static_dir: str = 'static') -> Dict[str, Any]:
    """Convenience function to optimize all static assets"""
    optimizer = StaticOptimizer(static_dir)
    return optimizer.optimize_all()

if __name__ == "__main__":
    # CLI usage
    import sys
    import json
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if len(sys.argv) > 1 and sys.argv[1] == 'optimize':
        static_dir = sys.argv[2] if len(sys.argv) > 2 else 'static'
        print(f"Optimizing static assets in: {static_dir}")
        
        results = optimize_static_assets(static_dir)
        
        print("\nOptimization Results:")
        print(f"Files processed: {results['summary']['total_files']}")
        print(f"Successful optimizations: {results['summary']['successful_optimizations']}")
        print(f"Total size reduction: {results['summary']['total_savings_percent']}%")
        print(f"With gzip: {results['summary']['total_gzip_savings_percent']}%")
        print(f"Output directory: {results['summary']['output_directory']}")
        
        # Show detailed results
        if '--verbose' in sys.argv:
            print("\nDetailed Results:")
            print(json.dumps(results, indent=2))
        
        # Generate usage template
        optimizer = StaticOptimizer(static_dir)
        print("\nUsage Template:")
        print(optimizer.generate_usage_templates())
        
    else:
        print("Usage:")
        print("  python static_optimizer.py optimize [static_dir] [--verbose]")
        print("  python static_optimizer.py optimize static --verbose")