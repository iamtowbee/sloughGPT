"""
Lazy Import Utilities

Provides lazy loading for heavy dependencies.
"""

import logging
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class LazyLoader:
    """Lazy loader for heavy dependencies."""
    
    _cache: dict = {}
    _failed: set = set()
    
    @classmethod
    def get(cls, module_name: str) -> Optional[Any]:
        if module_name in cls._cache:
            return cls._cache[module_name]
        if module_name in cls._failed:
            return None
        try:
            import importlib
            module = importlib.import_module(module_name)
            cls._cache[module_name] = module
            return module
        except ImportError:
            cls._failed.add(module_name)
            return None
    
    @classmethod
    def clear_cache(cls):
        cls._cache.clear()
        cls._failed.clear()


def get_torch() -> Optional[Any]:
    return LazyLoader.get('torch')


def get_numpy() -> Optional[Any]:
    return LazyLoader.get('numpy')


__all__ = ['LazyLoader', 'get_torch', 'get_numpy']
