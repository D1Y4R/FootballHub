"""
Lazy Model Manager for FootballHub
On-demand model loading to reduce memory usage and startup time
"""

import os
import threading
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)

class LazyModelManager:
    """
    Lazy loading manager for ML models and heavy services
    """
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.loaders: Dict[str, Callable] = {}
        self.loading_status: Dict[str, str] = {}
        self.lock = threading.RLock()
        self.load_attempts: Dict[str, int] = {}
        self.max_load_attempts = 3
        
    def register_loader(self, name: str, loader_func: Callable, lazy: bool = True):
        """
        Register a loader function for a model/service
        
        Args:
            name: Unique identifier for the model/service
            loader_func: Function that loads and returns the model/service
            lazy: If True, load on first access. If False, load immediately
        """
        with self.lock:
            self.loaders[name] = loader_func
            self.loading_status[name] = "registered"
            self.load_attempts[name] = 0
            
            if not lazy:
                self._load_model(name)
    
    def _load_model(self, name: str) -> Any:
        """Internal method to load a model"""
        if name in self.models:
            return self.models[name]
        
        if name not in self.loaders:
            raise ValueError(f"No loader registered for model: {name}")
        
        if self.load_attempts[name] >= self.max_load_attempts:
            logger.error(f"Max load attempts reached for model: {name}")
            return None
        
        try:
            self.loading_status[name] = "loading"
            self.load_attempts[name] += 1
            
            logger.info(f"Loading model: {name}")
            model = self.loaders[name]()
            
            self.models[name] = model
            self.loading_status[name] = "loaded"
            
            logger.info(f"Successfully loaded model: {name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {name}: {str(e)}")
            self.loading_status[name] = f"error: {str(e)}"
            return None
    
    def get_model(self, name: str) -> Any:
        """
        Get a model, loading it if necessary
        """
        with self.lock:
            if name in self.models:
                return self.models[name]
            
            return self._load_model(name)
    
    def is_loaded(self, name: str) -> bool:
        """Check if a model is loaded"""
        return name in self.models
    
    def get_status(self, name: str) -> str:
        """Get loading status of a model"""
        return self.loading_status.get(name, "not_registered")
    
    def unload_model(self, name: str) -> bool:
        """Unload a model to free memory"""
        with self.lock:
            if name in self.models:
                del self.models[name]
                self.loading_status[name] = "unloaded"
                logger.info(f"Unloaded model: {name}")
                return True
            return False
    
    def reload_model(self, name: str) -> Any:
        """Reload a model"""
        with self.lock:
            self.unload_model(name)
            self.load_attempts[name] = 0
            return self._load_model(name)
    
    def get_all_status(self) -> Dict[str, str]:
        """Get status of all registered models"""
        return dict(self.loading_status)
    
    def preload_models(self, model_names: list) -> Dict[str, bool]:
        """Preload specific models"""
        results = {}
        for name in model_names:
            try:
                model = self.get_model(name)
                results[name] = model is not None
            except Exception as e:
                logger.error(f"Failed to preload {name}: {str(e)}")
                results[name] = False
        return results
    
    def cleanup(self):
        """Clean up all loaded models"""
        with self.lock:
            for name in list(self.models.keys()):
                self.unload_model(name)

# Global lazy manager instance
manager = LazyModelManager()

def lazy_load(model_name: str, loader_func: Optional[Callable] = None, lazy: bool = True):
    """
    Decorator for lazy loading models
    
    Args:
        model_name: Name of the model to load
        loader_func: Function to load the model (if None, uses decorated function)
        lazy: Whether to load lazily or immediately
    """
    def decorator(func):
        # Register the loader
        actual_loader = loader_func if loader_func else func
        manager.register_loader(model_name, actual_loader, lazy)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Return the loaded model
            return manager.get_model(model_name)
        
        return wrapper
    return decorator

def require_model(model_name: str):
    """
    Decorator that ensures a model is loaded before function execution
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            model = manager.get_model(model_name)
            if model is None:
                raise RuntimeError(f"Required model '{model_name}' could not be loaded")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Helper functions for common models
def load_match_predictor():
    """Loader for match predictor model"""
    try:
        from match_prediction import MatchPredictor
        return MatchPredictor()
    except ImportError as e:
        logger.warning(f"Could not import MatchPredictor: {e}")
        return None

def load_model_validator():
    """Loader for model validator"""
    try:
        from model_validation import ModelValidator
        return ModelValidator()
    except ImportError as e:
        logger.warning(f"Could not import ModelValidator: {e}")
        return None

def load_kg_service():
    """Loader for KG service"""
    try:
        from hybrid_kg_service import HybridKGService
        return HybridKGService()
    except ImportError as e:
        logger.warning(f"Could not import HybridKGService: {e}")
        return None

# Register common models
manager.register_loader("match_predictor", load_match_predictor)
manager.register_loader("model_validator", load_model_validator)
manager.register_loader("kg_service", load_kg_service)

# Environment-specific loading
def is_resource_constrained():
    """Check if running in resource-constrained environment"""
    return (
        os.getenv('CODESANDBOX_SSE') or
        os.getenv('CODESPACE_NAME') or
        os.getenv('GITPOD_WORKSPACE_ID') or
        os.getenv('REPLIT_DEPLOYMENT')
    )

def get_model_safely(model_name: str, default=None):
    """
    Safely get a model, returning default if loading fails
    """
    try:
        model = manager.get_model(model_name)
        return model if model is not None else default
    except Exception as e:
        logger.error(f"Error getting model {model_name}: {e}")
        return default

def preload_critical_models():
    """Preload critical models if not in resource-constrained environment"""
    if not is_resource_constrained():
        logger.info("Preloading critical models...")
        results = manager.preload_models(["match_predictor", "model_validator"])
        logger.info(f"Preload results: {results}")
    else:
        logger.info("Resource-constrained environment detected, skipping preload")

def get_manager_status():
    """Get status of all models in the manager"""
    return manager.get_all_status()