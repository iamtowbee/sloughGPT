"""
Lazy Import Utilities

Provides lazy loading for heavy dependencies like torch.
Prevents blocking on network during import.

Usage:
    from domains.shared.lazy_imports import lazy_import, get_torch
    
    # Lazy import - won't block
    torch = get_torch()  # Returns None if unavailable
    
    # Or use lazy_import decorator
    @lazy_import
    def heavy_function():
        import torch
        return torch.randn(3)
"""

import sys
import importlib
import logging
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class LazyLoader:
    """Lazy loader for heavy dependencies."""
    
    _cache: dict = {}
    _failed: set = set()
    
    @classmethod
    def get(cls, module_name: str, package: Optional[str] = None) -> Optional[Any]:
        """
        Get a module lazily. Returns None if import fails.
        
        Args:
            module_name: Name of module (e.g., 'torch', 'numpy')
            package: Package name for submodules
            
        Returns:
            Module object or None if import fails
        """
        if module_name in cls._cache:
            return cls._cache[module_name]
        
        if module_name in cls._failed:
            return None
        
        try:
            if package:
                module = importlib.import_module(module_name, package)
            else:
                module = importlib.import_module(module_name)
            cls._cache[module_name] = module
            return module
        except (ImportError, ModuleNotFoundError) as e:
            logger.debug(f"Lazy import failed for {module_name}: {e}")
            cls._failed.add(module_name)
            return None
        except Exception as e:
            logger.warning(f"Unexpected error importing {module_name}: {e}")
            cls._failed.add(module_name)
            return None
    
    @classmethod
    def clear_cache(cls):
        """Clear the import cache."""
        cls._cache.clear()
        cls._failed.clear()


def lazy_import(module_name: str):
    """
    Decorator for lazy imports inside functions.
    
    Usage:
        @lazy_import('torch')
        def train_model():
            torch.nn.Linear(10, 10)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            module = LazyLoader.get(module_name)
            if module is None:
                raise ImportError(
                    f"Required module '{module_name}' not available. "
                    f"Install with: pip install {module_name}"
                )
            return func(module, *args, **kwargs)
        return wrapper
    return decorator


def get_torch() -> Optional[Any]:
    """Get torch module lazily. Returns None if unavailable."""
    return LazyLoader.get('torch')


def get_numpy() -> Optional[Any]:
    """Get numpy module lazily."""
    return LazyLoader.get('numpy')


def safe_import(module_name: str, fallback: Any = None) -> Any:
    """
    Safely import a module with fallback.
    
    Usage:
        torch = safe_import('torch', fallback={})
    """
    return LazyLoader.get(module_name) or fallback


# Convenience functions for common imports
def get_transformers():
    """Get transformers module lazily."""
    return LazyLoader.get('transformers')


def get_sentence_transformers():
    """Get sentence_transformers module lazily."""
    return LazyLoader.get('sentence_transformers')


def get_openai():
    """Get openai module lazily."""
    return LazyLoader.get('openai')


__all__ = [
    'LazyLoader',
    'lazy_import',
    'get_torch',
    'get_numpy',
    'safe_import',
    'get_transformers',
    'get_sentence_transformers',
    'get_openai',
]
