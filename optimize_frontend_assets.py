#!/usr/bin/env python3
"""
Frontend Asset Optimization Script
Bundles and minifies CSS/JS files for better performance.
"""

import os
import gzip
import shutil
from pathlib import Path
import logging
import hashlib
import json

logger = logging.getLogger(__name__)

class AssetOptimizer:
    """
    Optimize frontend assets for better performance.
    """
    
    def __init__(self, static_dir: str = "static"):
        self.static_dir = Path(static_dir)
        self.build_dir = self.static_dir / "build"
        self.build_dir.mkdir(exist_ok=True)
        
        # Asset manifest for cache busting
        self.manifest_file = self.build_dir / "manifest.json"
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> dict:
        """Load asset manifest for cache busting."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")
        return {}
    
    def _save_manifest(self):
        """Save asset manifest."""
        try:
            with open(self.manifest_file, 'w') as f:
                json.dump(self.manifest, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file content for cache busting."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()[:8]
        except Exception as e:
            logger.error(f"Failed to hash file {file_path}: {e}")
            return "unknown"
    
    def _minify_css(self, content: str) -> str:
        """Simple CSS minification."""
        # Remove comments
        import re
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Remove unnecessary whitespace
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r';\s*}', '}', content)
        content = re.sub(r'{\s*', '{', content)
        content = re.sub(r';\s*', ';', content)
        content = re.sub(r':\s*', ':', content)
        content = re.sub(r',\s*', ',', content)
        
        return content.strip()
    
    def _minify_js(self, content: str) -> str:
        """Simple JavaScript minification."""
        import re
        
        # Remove single line comments (but preserve URLs)
        content = re.sub(r'^\s*//.*$', '', content, flags=re.MULTILINE)
        
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Remove unnecessary whitespace
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r';\s*', ';', content)
        content = re.sub(r'{\s*', '{', content)
        content = re.sub(r'}\s*', '}', content)
        content = re.sub(r',\s*', ',', content)
        
        return content.strip()
    
    def bundle_css_files(self) -> str:
        """Bundle all CSS files into one optimized file."""
        css_files = [
            'css/custom.css',
            'css/widget-style.css', 
            'css/widgetCountries.css',
            'css/widgetLeague.css',
            'css/match-actions.css',
            'css/prediction-modal.css',
            'css/match-insights.css'
        ]
        
        bundled_content = []
        bundled_content.append("/* Bundled CSS - Generated automatically */\n")
        
        for css_file in css_files:
            file_path = self.static_dir / css_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    bundled_content.append(f"/* === {css_file} === */")
                    bundled_content.append(self._minify_css(content))
                    bundled_content.append("")
                    
                    logger.info(f"Added {css_file} to bundle ({file_path.stat().st_size} bytes)")
                    
                except Exception as e:
                    logger.error(f"Failed to process {css_file}: {e}")
        
        final_content = "\n".join(bundled_content)
        
        # Generate hash for cache busting
        content_hash = hashlib.md5(final_content.encode()).hexdigest()[:8]
        output_filename = f"bundle.{content_hash}.css"
        output_path = self.build_dir / output_filename
        
        # Write bundled file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        # Create gzipped version
        with open(output_path, 'rb') as f_in:
            with gzip.open(f"{output_path}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Update manifest
        self.manifest['bundle.css'] = output_filename
        
        original_size = sum((self.static_dir / css_file).stat().st_size 
                          for css_file in css_files 
                          if (self.static_dir / css_file).exists())
        
        bundled_size = output_path.stat().st_size
        gzipped_size = Path(f"{output_path}.gz").stat().st_size
        
        logger.info(f"CSS bundling complete:")
        logger.info(f"  Original: {original_size:,} bytes")
        logger.info(f"  Bundled: {bundled_size:,} bytes ({100-bundled_size/original_size*100:.1f}% reduction)")
        logger.info(f"  Gzipped: {gzipped_size:,} bytes ({100-gzipped_size/original_size*100:.1f}% reduction)")
        
        return output_filename
    
    def bundle_js_files(self) -> str:
        """Bundle all JavaScript files into one optimized file."""
        js_files = [
            'js/custom.js',
            'js/main.js',
            'js/prediction-handler.js',
            'js/team_stats.js',
            'js/team_history.js',
            'js/api_football_debug.js'
        ]
        
        bundled_content = []
        bundled_content.append("/* Bundled JavaScript - Generated automatically */\n")
        bundled_content.append("(function() {\n")  # IIFE wrapper
        
        for js_file in js_files:
            file_path = self.static_dir / js_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    bundled_content.append(f"/* === {js_file} === */")
                    bundled_content.append(self._minify_js(content))
                    bundled_content.append("")
                    
                    logger.info(f"Added {js_file} to bundle ({file_path.stat().st_size} bytes)")
                    
                except Exception as e:
                    logger.error(f"Failed to process {js_file}: {e}")
        
        bundled_content.append("})();")  # Close IIFE wrapper
        final_content = "\n".join(bundled_content)
        
        # Generate hash for cache busting
        content_hash = hashlib.md5(final_content.encode()).hexdigest()[:8]
        output_filename = f"bundle.{content_hash}.js"
        output_path = self.build_dir / output_filename
        
        # Write bundled file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        # Create gzipped version
        with open(output_path, 'rb') as f_in:
            with gzip.open(f"{output_path}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Update manifest
        self.manifest['bundle.js'] = output_filename
        
        original_size = sum((self.static_dir / js_file).stat().st_size 
                          for js_file in js_files 
                          if (self.static_dir / js_file).exists())
        
        bundled_size = output_path.stat().st_size
        gzipped_size = Path(f"{output_path}.gz").stat().st_size
        
        logger.info(f"JavaScript bundling complete:")
        logger.info(f"  Original: {original_size:,} bytes")
        logger.info(f"  Bundled: {bundled_size:,} bytes ({100-bundled_size/original_size*100:.1f}% reduction)")
        logger.info(f"  Gzipped: {gzipped_size:,} bytes ({100-gzipped_size/original_size*100:.1f}% reduction)")
        
        return output_filename
    
    def create_critical_css(self) -> str:
        """Extract critical CSS for above-the-fold content."""
        critical_css_rules = [
            # Basic layout and typography
            "body, html { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }",
            
            # Navigation
            ".navbar { background-color: #343a40; padding: 0.5rem 1rem; }",
            ".navbar-brand { color: #fff; text-decoration: none; font-weight: bold; }",
            ".navbar-nav .nav-link { color: #rgba(255,255,255,.55); }",
            
            # Container and grid
            ".container { max-width: 1200px; margin: 0 auto; padding: 0 15px; }",
            ".row { display: flex; flex-wrap: wrap; }",
            ".col, .col-md-6, .col-lg-4 { flex: 1; padding: 0 15px; }",
            
            # Basic styling
            ".btn { display: inline-block; padding: 0.375rem 0.75rem; background: #007bff; color: #fff; border: none; border-radius: 0.25rem; cursor: pointer; }",
            ".btn:hover { background: #0056b3; }",
            
            # Loading states
            ".loading { opacity: 0.6; pointer-events: none; }",
            
            # Hide non-critical content initially
            ".prediction-modal, .team-stats-modal { display: none; }",
        ]
        
        critical_css = "/* Critical CSS - Above the fold */\n" + "\n".join(critical_css_rules)
        
        # Generate hash for cache busting
        content_hash = hashlib.md5(critical_css.encode()).hexdigest()[:8]
        output_filename = f"critical.{content_hash}.css"
        output_path = self.build_dir / output_filename
        
        # Write critical CSS
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(critical_css)
        
        # Update manifest
        self.manifest['critical.css'] = output_filename
        
        logger.info(f"Critical CSS created: {output_filename} ({len(critical_css)} bytes)")
        return output_filename
    
    def optimize_images(self):
        """Optimize images in the static directory."""
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.ico'}
        image_dir = self.static_dir / 'img'
        
        if not image_dir.exists():
            logger.info("No image directory found, skipping image optimization")
            return
        
        optimized_count = 0
        total_saved = 0
        
        for image_file in image_dir.rglob('*'):
            if image_file.suffix.lower() in image_extensions:
                try:
                    original_size = image_file.stat().st_size
                    
                    # For now, just create gzipped versions of SVG-like content
                    if image_file.suffix.lower() in {'.svg'}:
                        with open(image_file, 'rb') as f_in:
                            with gzip.open(f"{image_file}.gz", 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        
                        gzipped_size = Path(f"{image_file}.gz").stat().st_size
                        saved = original_size - gzipped_size
                        total_saved += saved
                        optimized_count += 1
                        
                        logger.debug(f"Compressed {image_file.name}: {original_size} -> {gzipped_size} bytes")
                
                except Exception as e:
                    logger.error(f"Failed to optimize {image_file}: {e}")
        
        if optimized_count > 0:
            logger.info(f"Optimized {optimized_count} images, saved {total_saved:,} bytes")
        else:
            logger.info("No images optimized")
    
    def clean_old_bundles(self):
        """Remove old bundled files to save space."""
        if not self.build_dir.exists():
            return
        
        current_files = set(self.manifest.values())
        removed_count = 0
        
        for file_path in self.build_dir.glob('*'):
            if file_path.is_file():
                # Keep manifest and current files
                if file_path.name == 'manifest.json' or file_path.name in current_files:
                    continue
                
                # Remove old bundle files
                if any(pattern in file_path.name for pattern in ['bundle.', 'critical.']):
                    try:
                        file_path.unlink()
                        # Also remove gzipped version
                        gz_file = Path(f"{file_path}.gz")
                        if gz_file.exists():
                            gz_file.unlink()
                        removed_count += 1
                        logger.debug(f"Removed old bundle: {file_path.name}")
                    except Exception as e:
                        logger.error(f"Failed to remove {file_path}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old bundle files")
    
    def optimize_all(self) -> dict:
        """Run all optimizations and return results."""
        logger.info("Starting frontend asset optimization...")
        
        results = {}
        
        try:
            # Bundle CSS
            results['css_bundle'] = self.bundle_css_files()
            
            # Bundle JavaScript  
            results['js_bundle'] = self.bundle_js_files()
            
            # Create critical CSS
            results['critical_css'] = self.create_critical_css()
            
            # Optimize images
            self.optimize_images()
            
            # Clean old files
            self.clean_old_bundles()
            
            # Save manifest
            self._save_manifest()
            
            results['manifest'] = self.manifest
            results['success'] = True
            
            logger.info("Frontend asset optimization completed successfully")
            
        except Exception as e:
            logger.error(f"Asset optimization failed: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results

def main():
    """Main function for running asset optimization."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    optimizer = AssetOptimizer()
    results = optimizer.optimize_all()
    
    if results.get('success'):
        print("✅ Asset optimization completed successfully!")
        print(f"📦 CSS Bundle: {results['css_bundle']}")
        print(f"📦 JS Bundle: {results['js_bundle']}")
        print(f"⚡ Critical CSS: {results['critical_css']}")
    else:
        print(f"❌ Asset optimization failed: {results.get('error', 'Unknown error')}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())