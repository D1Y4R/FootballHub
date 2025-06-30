#!/usr/bin/env python3
"""
Lazy ML Imports Module
Provides lazy loading for heavy ML dependencies to improve application startup time.
"""

import logging
import sys
import time
from typing import Optional, Any
from functools import lru_cache
import threading

logger = logging.getLogger(__name__)

class LazyImport:
    """
    Lazy import wrapper that loads modules only when first accessed.
    """
    
    def __init__(self, module_name: str, package: Optional[str] = None, 
                 fallback_available: bool = True):
        self.module_name = module_name
        self.package = package
        self.fallback_available = fallback_available
        self._module = None
        self._load_attempted = False
        self._load_success = False
        self._lock = threading.Lock()
    
    def __getattr__(self, name: str) -> Any:
        """Get attribute from the lazily loaded module."""
        module = self._get_module()
        if module is None:
            if self.fallback_available:
                return MockAttribute(f"{self.module_name}.{name}")
            else:
                raise AttributeError(f"Module {self.module_name} not available and no fallback")
        return getattr(module, name)
    
    def _get_module(self) -> Optional[Any]:
        """Get the module, loading it if necessary."""
        if self._module is not None:
            return self._module
        
        if self._load_attempted and not self._load_success:
            return None
        
        with self._lock:
            if self._module is not None:
                return self._module
            
            if self._load_attempted and not self._load_success:
                return None
            
            self._load_attempted = True
            
            try:
                start_time = time.time()
                
                if self.package:
                    module = __import__(f"{self.package}.{self.module_name}", 
                                      fromlist=[self.module_name])
                else:
                    module = __import__(self.module_name)
                
                load_time = time.time() - start_time
                logger.info(f"Loaded {self.module_name} in {load_time:.2f}s")
                
                self._module = module
                self._load_success = True
                return module
                
            except ImportError as e:
                logger.warning(f"Failed to import {self.module_name}: {e}")
                self._load_success = False
                return None
            except Exception as e:
                logger.error(f"Unexpected error importing {self.module_name}: {e}")
                self._load_success = False
                return None
    
    def is_available(self) -> bool:
        """Check if the module is available."""
        return self._get_module() is not None
    
    def force_load(self) -> bool:
        """Force load the module and return success status."""
        module = self._get_module()
        return module is not None

class MockAttribute:
    """
    Mock attribute that provides reasonable defaults when real modules aren't available.
    """
    
    def __init__(self, name: str):
        self.name = name
        logger.debug(f"Created mock for {name}")
    
    def __call__(self, *args, **kwargs):
        """Mock function call."""
        logger.warning(f"Mock call to {self.name} with args={args}, kwargs={kwargs}")
        return MockAttribute(f"{self.name}()")
    
    def __getattr__(self, name: str):
        """Mock attribute access."""
        return MockAttribute(f"{self.name}.{name}")
    
    def __getitem__(self, key):
        """Mock item access."""
        return MockAttribute(f"{self.name}[{key}]")
    
    def __str__(self):
        return f"Mock({self.name})"
    
    def __repr__(self):
        return f"MockAttribute('{self.name}')"

class LazyMLImports:
    """
    Manager for lazy loading of ML dependencies.
    """
    
    def __init__(self):
        self._imports = {}
        self._setup_lazy_imports()
    
    def _setup_lazy_imports(self):
        """Setup lazy imports for common ML libraries."""
        
        # Core ML libraries
        self._imports['numpy'] = LazyImport('numpy')
        self._imports['pandas'] = LazyImport('pandas')
        self._imports['sklearn'] = LazyImport('sklearn')
        self._imports['tensorflow'] = LazyImport('tensorflow')
        
        # Specific sklearn components
        self._imports['sklearn.ensemble'] = LazyImport('ensemble', 'sklearn')
        self._imports['sklearn.preprocessing'] = LazyImport('preprocessing', 'sklearn')
        self._imports['sklearn.model_selection'] = LazyImport('model_selection', 'sklearn')
        self._imports['sklearn.metrics'] = LazyImport('metrics', 'sklearn')
        
        # TensorFlow Keras components
        self._imports['tensorflow.keras'] = LazyImport('keras', 'tensorflow')
        
        logger.info("Lazy ML imports configured")
    
    def get_numpy(self):
        """Get numpy with lazy loading."""
        return self._imports['numpy']
    
    def get_pandas(self):
        """Get pandas with lazy loading."""
        return self._imports['pandas']
    
    def get_sklearn(self):
        """Get sklearn with lazy loading."""
        return self._imports['sklearn']
    
    def get_tensorflow(self):
        """Get tensorflow with lazy loading."""
        return self._imports['tensorflow']
    
    def get_sklearn_component(self, component: str):
        """Get specific sklearn component."""
        key = f'sklearn.{component}'
        if key not in self._imports:
            self._imports[key] = LazyImport(component, 'sklearn')
        return self._imports[key]
    
    def preload_critical_modules(self, modules: list = None):
        """Preload critical modules in background."""
        if modules is None:
            modules = ['numpy', 'pandas']  # Only most critical
        
        def preload_worker():
            for module_name in modules:
                if module_name in self._imports:
                    try:
                        self._imports[module_name].force_load()
                    except Exception as e:
                        logger.error(f"Failed to preload {module_name}: {e}")
        
        thread = threading.Thread(target=preload_worker, daemon=True)
        thread.start()
        logger.info(f"Started background preloading for: {modules}")
    
    def get_availability_status(self) -> dict:
        """Get availability status of all configured imports."""
        status = {}
        for name, lazy_import in self._imports.items():
            status[name] = lazy_import.is_available()
        return status
    
    def force_load_all(self) -> dict:
        """Force load all modules and return status."""
        results = {}
        for name, lazy_import in self._imports.items():
            results[name] = lazy_import.force_load()
        return results

# Specific lazy loaders for common ML components

@lru_cache(maxsize=1)
def get_numpy():
    """Get numpy with caching."""
    lazy_import = LazyImport('numpy')
    return lazy_import

@lru_cache(maxsize=1)
def get_pandas():
    """Get pandas with caching."""
    lazy_import = LazyImport('pandas')
    return lazy_import

@lru_cache(maxsize=1)
def get_sklearn_components():
    """Get commonly used sklearn components."""
    components = {}
    
    # RandomForestRegressor
    try:
        ensemble = LazyImport('ensemble', 'sklearn')
        components['RandomForestRegressor'] = getattr(ensemble, 'RandomForestRegressor')
    except:
        components['RandomForestRegressor'] = MockAttribute('sklearn.ensemble.RandomForestRegressor')
    
    # StandardScaler
    try:
        preprocessing = LazyImport('preprocessing', 'sklearn')
        components['StandardScaler'] = getattr(preprocessing, 'StandardScaler')
    except:
        components['StandardScaler'] = MockAttribute('sklearn.preprocessing.StandardScaler')
    
    # train_test_split
    try:
        model_selection = LazyImport('model_selection', 'sklearn')
        components['train_test_split'] = getattr(model_selection, 'train_test_split')
    except:
        components['train_test_split'] = MockAttribute('sklearn.model_selection.train_test_split')
    
    return components

@lru_cache(maxsize=1)
def get_tensorflow_components():
    """Get commonly used TensorFlow components."""
    components = {}
    
    try:
        tf = LazyImport('tensorflow')
        components['tf'] = tf
        
        # Keras components
        if hasattr(tf, 'keras'):
            keras = getattr(tf, 'keras')
            components['Sequential'] = getattr(keras.models, 'Sequential', None)
            components['Dense'] = getattr(keras.layers, 'Dense', None)
            components['Dropout'] = getattr(keras.layers, 'Dropout', None)
            components['load_model'] = getattr(keras.models, 'load_model', None)
            components['save_model'] = getattr(keras.models, 'save_model', None)
            components['EarlyStopping'] = getattr(keras.callbacks, 'EarlyStopping', None)
        
    except Exception as e:
        logger.warning(f"TensorFlow components not available: {e}")
        # Provide mock components
        components['tf'] = MockAttribute('tensorflow')
        components['Sequential'] = MockAttribute('tensorflow.keras.models.Sequential')
        components['Dense'] = MockAttribute('tensorflow.keras.layers.Dense')
        components['Dropout'] = MockAttribute('tensorflow.keras.layers.Dropout')
        components['load_model'] = MockAttribute('tensorflow.keras.models.load_model')
        components['save_model'] = MockAttribute('tensorflow.keras.models.save_model')
        components['EarlyStopping'] = MockAttribute('tensorflow.keras.callbacks.EarlyStopping')
    
    return components

# Safe import functions (backwards compatibility)

def safe_import_numpy():
    """Safe import for numpy with fallback."""
    try:
        np = get_numpy()
        if np.is_available():
            return np
        else:
            logger.warning("NumPy not available, using mock")
            return MockAttribute('numpy')
    except Exception as e:
        logger.error(f"Error importing numpy: {e}")
        return MockAttribute('numpy')

def safe_import_pandas():
    """Safe import for pandas with fallback."""
    try:
        pd = get_pandas()
        if pd.is_available():
            return pd
        else:
            logger.warning("Pandas not available, using mock")
            return MockAttribute('pandas')
    except Exception as e:
        logger.error(f"Error importing pandas: {e}")
        return MockAttribute('pandas')

def safe_import_sklearn():
    """Safe import for sklearn components with fallback."""
    try:
        components = get_sklearn_components()
        return (
            components.get('RandomForestRegressor'),
            components.get('StandardScaler'),
            components.get('train_test_split')
        )
    except Exception as e:
        logger.error(f"Error importing sklearn components: {e}")
        return (
            MockAttribute('sklearn.ensemble.RandomForestRegressor'),
            MockAttribute('sklearn.preprocessing.StandardScaler'),
            MockAttribute('sklearn.model_selection.train_test_split')
        )

def safe_import_tensorflow():
    """Safe import for tensorflow with fallback."""
    try:
        components = get_tensorflow_components()
        return components.get('tf', MockAttribute('tensorflow'))
    except Exception as e:
        logger.error(f"Error importing tensorflow: {e}")
        return MockAttribute('tensorflow')

# Performance monitoring

class ImportPerformanceMonitor:
    """Monitor import performance and provide recommendations."""
    
    def __init__(self):
        self.import_times = {}
        self.total_import_time = 0
    
    def record_import(self, module_name: str, import_time: float):
        """Record import time for a module."""
        self.import_times[module_name] = import_time
        self.total_import_time += import_time
        
        if import_time > 2.0:  # Slow import threshold
            logger.warning(f"Slow import detected: {module_name} took {import_time:.2f}s")
    
    def get_performance_report(self) -> dict:
        """Get performance report for imports."""
        return {
            'total_import_time': self.total_import_time,
            'import_times': self.import_times,
            'slow_imports': {k: v for k, v in self.import_times.items() if v > 1.0},
            'recommendations': self._get_recommendations()
        }
    
    def _get_recommendations(self) -> list:
        """Get performance optimization recommendations."""
        recommendations = []
        
        if self.total_import_time > 10.0:
            recommendations.append("Consider lazy loading for non-critical ML dependencies")
        
        slow_imports = [k for k, v in self.import_times.items() if v > 2.0]
        if slow_imports:
            recommendations.append(f"Consider background loading for: {', '.join(slow_imports)}")
        
        if 'tensorflow' in self.import_times and self.import_times['tensorflow'] > 5.0:
            recommendations.append("TensorFlow is very slow to load, consider using lighter alternatives")
        
        return recommendations

# Global instances
ml_imports = LazyMLImports()
import_monitor = ImportPerformanceMonitor()

# Initialization function
def initialize_lazy_ml_imports(preload_critical: bool = True):
    """Initialize lazy ML imports system."""
    logger.info("Initializing lazy ML imports system")
    
    if preload_critical:
        ml_imports.preload_critical_modules(['numpy', 'pandas'])
    
    # Log availability status
    status = ml_imports.get_availability_status()
    available = [k for k, v in status.items() if v]
    unavailable = [k for k, v in status.items() if not v]
    
    if available:
        logger.info(f"Available ML modules: {', '.join(available)}")
    if unavailable:
        logger.warning(f"Unavailable ML modules: {', '.join(unavailable)}")
    
    return ml_imports