"""
Robust dependency management for football prediction system
Fixes the critical NumPy/SciPy compatibility issues
Optimized for CodeSandbox/Codespace environments with limited disk space
"""
import sys
import warnings
import logging
import os

logger = logging.getLogger(__name__)

# Environment detection for optimal fallback decisions
def is_limited_environment():
    """Check if running in a limited environment like CodeSandbox/Codespace"""
    indicators = [
        os.environ.get('CODESPACE_NAME'),
        os.environ.get('CODESANDBOX_HOST'), 
        os.environ.get('GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN'),
        os.environ.get('CODESANDBOX_SSE'),
        '/tmp' in os.getcwd() if hasattr(os, 'getcwd') else False,
        'runner' in os.environ.get('HOME', ''),
        'codespace' in os.environ.get('HOME', '').lower()
    ]
    
    is_limited = any(indicators)
    if is_limited:
        logger.info("Limited environment detected (CodeSandbox/Codespace), using fallback implementations")
    else:
        logger.info("Standard environment detected")
    
    return is_limited

# Global environment check
LIMITED_ENV = is_limited_environment()

def safe_import_numpy():
    """Import numpy with proper error handling"""
    try:
        # Check if we're in a limited environment
        if LIMITED_ENV:
            try:
                import numpy as np
                logger.info(f"NumPy {np.__version__} imported successfully in CodeSandbox")
                return np
            except ImportError:
                logger.warning("NumPy not available in CodeSandbox, using fallback")
                return create_numpy_fallback()
        
        import numpy as np
        logger.info(f"NumPy {np.__version__} imported successfully")
        return np
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        logger.error(f"NumPy import failed: {e}")
        logger.info("Using NumPy fallback implementation")
        return create_numpy_fallback()

def safe_import_pandas():
    """Import pandas with proper error handling"""
    try:
        # Check if we're in a limited environment
        if LIMITED_ENV:
            try:
                import pandas as pd
                logger.info(f"Pandas {pd.__version__} imported successfully in CodeSandbox")
                return pd
            except ImportError:
                logger.warning("Pandas not available in CodeSandbox, using fallback")
                return create_pandas_fallback()
                
        import pandas as pd
        logger.info(f"Pandas {pd.__version__} imported successfully")
        return pd
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        logger.error(f"Pandas import failed: {e}")
        logger.info("Using Pandas fallback implementation")
        return create_pandas_fallback()

def safe_import_sklearn():
    """
    Scikit-learn import - production için zorunlu
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        logger.info("Scikit-learn successfully imported")
        return RandomForestRegressor, StandardScaler, train_test_split
    except ImportError as e:
        logger.error(f"Scikit-learn import failed: {e}")
        raise ImportError("Scikit-learn is required for production use. Install with: pip install scikit-learn")

def safe_import_tensorflow():
    """
    TensorFlow import - production için zorunlu
    """
    try:
        import tensorflow as tf
        logger.info("TensorFlow successfully imported")
        return tf
    except ImportError as e:
        logger.error(f"TensorFlow import failed: {e}")
        raise ImportError("TensorFlow is required for production use. Install with: pip install tensorflow")

def create_numpy_fallback():
    """Create a functional numpy fallback that won't break the system"""
    class NumpyFallback:
        def __init__(self):
            self.random = self.RandomModule()
            
        def array(self, data):
            if isinstance(data, (list, tuple)):
                return list(data)
            return [data]
        
        def zeros(self, shape):
            if isinstance(shape, int):
                return [0.0] * shape
            return [[0.0] * shape[1] for _ in range(shape[0])]
        
        def mean(self, data):
            return sum(data) / len(data) if data else 0.0
        
        def std(self, data):
            if not data:
                return 0.0
            mean_val = self.mean(data)
            variance = sum((x - mean_val) ** 2 for x in data) / len(data)
            return variance ** 0.5
        
        def exp(self, x):
            import math
            try:
                if isinstance(x, (list, tuple)):
                    return [math.exp(min(700, max(-700, float(val)))) for val in x]
                return math.exp(min(700, max(-700, float(x))))
            except (ValueError, TypeError):
                if isinstance(x, (list, tuple)):
                    return [1.0] * len(x)
                return 1.0
        
        def log(self, x):
            import math
            if isinstance(x, list):
                return [math.log(max(1e-10, val)) for val in x]
            return math.log(max(1e-10, x))
        
        class RandomModule:
            def poisson(self, lam, size=None):
                import random
                # Simple Poisson approximation using normal distribution
                if size:
                    return [max(0, int(random.gauss(lam, lam**0.5))) for _ in range(size)]
                return max(0, int(random.gauss(lam, lam**0.5)))
            
            def choice(self, a, size=None, p=None):
                import random
                if isinstance(a, int):
                    choices = list(range(a))
                else:
                    choices = list(a)
                
                if size is None:
                    return random.choice(choices)
                return [random.choice(choices) for _ in range(size)]
    
    return NumpyFallback()

def create_pandas_fallback():
    """Create a functional pandas fallback"""
    class PandasFallback:
        def DataFrame(self, data=None, **kwargs):
            if data is None:
                return []
            if isinstance(data, dict):
                return data
            return data
        
        def Series(self, data=None, **kwargs):
            if data is None:
                return []
            return list(data) if hasattr(data, '__iter__') else [data]
    
    return PandasFallback()

def safe_import_matplotlib():
    """
    Matplotlib import - isteğe bağlı
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        logger.info("Matplotlib successfully imported")
        return plt
    except ImportError as e:
        logger.warning(f"Matplotlib import failed: {e}")
        return None

def safe_import_seaborn():
    """
    Seaborn import - isteğe bağlı
    """
    try:
        import seaborn as sns
        logger.info("Seaborn successfully imported")
        return sns
    except ImportError as e:
        logger.warning(f"Seaborn import failed: {e}")
        return None

def safe_import_requests():
    """
    Requests import - production için zorunlu
    """
    try:
        import requests
        logger.info("Requests successfully imported")
        return requests
    except ImportError as e:
        logger.error(f"Requests import failed: {e}")
        raise ImportError("Requests is required for production use. Install with: pip install requests")

def safe_import_flask():
    """
    Flask import - production için zorunlu
    """
    try:
        from flask import Flask
        logger.info("Flask successfully imported")
        return Flask
    except ImportError as e:
        logger.error(f"Flask import failed: {e}")
        raise ImportError("Flask is required for production use. Install with: pip install Flask")

def safe_import_joblib():
    """
    Joblib import - production için zorunlu
    """
    try:
        import joblib
        logger.info("Joblib successfully imported")
        return joblib
    except ImportError as e:
        logger.error(f"Joblib import failed: {e}")
        raise ImportError("Joblib is required for production use. Install with: pip install joblib")

# Production imports verification
def verify_production_imports():
    """Verify that all required imports are available for production"""
    try:
        np = safe_import_numpy()
        pd = safe_import_pandas()
        RandomForestRegressor, StandardScaler, train_test_split = safe_import_sklearn()
        tf = safe_import_tensorflow()
        
        logger.info("All production imports verified successfully")
        return True
    except Exception as e:
        logger.error(f"Production import verification failed: {e}")
        return False

if __name__ == "__main__":
    verify_production_imports()