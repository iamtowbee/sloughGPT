from fastapi import HTTPException, Depends
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core.model_manager import ModelManager
    from .core.cache_manager import CacheManager

# Global managers (will be set by main app)
_model_manager = None
_cache_manager = None

def set_managers(model_manager, cache_manager):
    """Set global manager instances"""
    global _model_manager, _cache_manager
    _model_manager = model_manager
    _cache_manager = cache_manager

async def get_model_manager():
    """Get model manager dependency"""
    if _model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    return _model_manager

async def get_cache_manager():
    """Get cache manager dependency"""
    if _cache_manager is None:
        raise HTTPException(status_code=503, detail="Cache manager not initialized")
    return _cache_manager