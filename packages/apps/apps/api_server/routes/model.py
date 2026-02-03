from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import time
import torch

from fastapi import HTTPException, Depends
from ..core.model_manager import ModelManager
from ..core.config import settings
from ..core.cache_manager import CacheManager
from ..dependencies import get_model_manager, get_cache_manager

logger = logging.getLogger(__name__)
router = APIRouter()

class ModelStatus(BaseModel):
    model_name: str = Field(..., description="Model name")
    status: str = Field(..., description="Model status")
    device: str = Field(..., description="Device model is running on")
    context_size: int = Field(..., description="Model context window size")
    memory_usage: Optional[Dict[str, Any]] = Field(None, description="Memory usage info")
    uptime: float = Field(..., description="Model uptime in seconds")

class ModelConfig(BaseModel):
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(1024, ge=1, le=4096)
    context_size: Optional[int] = Field(2048, ge=1, le=8192)

@router.get("/status", response_model=ModelStatus)
async def get_model_status(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get current model status and information"""
    
    if not model_manager or not model_manager.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        memory_info = {}
        if torch.cuda.is_available() and model_manager.device.type == "cuda":
            memory_info = {
                "allocated": torch.cuda.memory_allocated(model_manager.device) / 1024**3,  # GB
                "cached": torch.cuda.memory_reserved(model_manager.device) / 1024**3,      # GB
                "total": torch.cuda.get_device_properties(model_manager.device).total_memory / 1024**3  # GB
            }
        
        return ModelStatus(
            model_name="slogpt",
            status="loaded",
            device=str(model_manager.device),
            context_size=settings.MODEL_CONTEXT_SIZE,
            memory_usage=memory_info if memory_info else None,
            uptime=time.time() - getattr(model_manager, 'load_time', time.time())
        )
        
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model status")

@router.get("/config")
async def get_model_config():
    """Get current model configuration"""
    return {
        "model_path": settings.MODEL_PATH,
        "context_size": settings.MODEL_CONTEXT_SIZE,
        "batch_size": settings.MODEL_BATCH_SIZE,
        "max_tokens": settings.MODEL_MAX_TOKENS,
        "enable_cache": settings.ENABLE_CACHE,
        "enable_request_batching": settings.ENABLE_REQUEST_BATCHING,
        "batch_size": settings.BATCH_SIZE,
        "batch_timeout": settings.BATCH_TIMEOUT
    }

@router.post("/reload")
async def reload_model(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Reload the model"""
    
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not available")
    
    try:
        logger.info("Reloading model...")
        
        # Cleanup existing model
        await model_manager.cleanup()
        
        # Reload model
        await model_manager.load_model()
        
        return {
            "message": "Model reloaded successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(status_code=500, detail="Model reload failed")

@router.post("/optimize")
async def optimize_model(
    config: ModelConfig,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Update model optimization settings"""
    
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not available")
    
    try:
        # Update settings (you'd implement this properly)
        updated_settings = []
        
        if config.temperature is not None:
            updated_settings.append(f"temperature={config.temperature}")
        if config.top_p is not None:
            updated_settings.append(f"top_p={config.top_p}")
        if config.max_tokens is not None:
            updated_settings.append(f"max_tokens={config.max_tokens}")
        if config.context_size is not None:
            updated_settings.append(f"context_size={config.context_size}")
        
        logger.info(f"Updated model settings: {', '.join(updated_settings)}")
        
        return {
            "message": "Model optimization settings updated",
            "updated_settings": updated_settings,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to optimize model: {e}")
        raise HTTPException(status_code=500, detail="Model optimization failed")

@router.delete("/cache")
async def clear_model_cache(
    model_manager: ModelManager = Depends(get_model_manager),
    cache_manager: CacheManager = Depends(get_cache_manager)
):
    """Clear model and inference cache"""
    
    try:
        # Clear Redis cache
        if cache_manager:
            await cache_manager.clear()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model cache cleared")
        
        return {
            "message": "Model cache cleared successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

@router.get("/batch/stats")
async def get_batch_stats(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get batch processing statistics"""
    
    try:
        stats = await model_manager.get_batch_stats()
        return {
            "batch_stats": stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get batch stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get batch statistics")