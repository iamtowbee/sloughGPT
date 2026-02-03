from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any
import logging
import time
import psutil
import torch

from fastapi import HTTPException, Depends
from ..core.model_manager import ModelManager
from ..core.cache_manager import CacheManager
from ..core.config import settings
from ..dependencies import get_model_manager, get_cache_manager

logger = logging.getLogger(__name__)
router = APIRouter()

class HealthCheck(BaseModel):
    status: str = "healthy"
    timestamp: float
    uptime: float
    version: str = "1.0.0"
    components: Dict[str, Any]

class SystemMetrics(BaseModel):
    cpu_percent: float
    memory_percent: float
    gpu_utilization: float = 0.0
    gpu_memory_percent: float = 0.0
    disk_usage: float

# Store start time
start_time = time.time()

@router.get("/", response_model=HealthCheck)
async def health_check(
    model_manager: ModelManager = Depends(get_model_manager),
    cache_manager: CacheManager = Depends(get_cache_manager)
):
    """Comprehensive health check"""
    
    uptime = time.time() - start_time
    components = {}
    
    # Check model status
    try:
        if model_manager and model_manager.model:
            components["model"] = {
                "status": "healthy",
                "device": str(model_manager.device),
                "loaded": True
            }
        else:
            components["model"] = {
                "status": "unhealthy",
                "error": "Model not loaded"
            }
    except Exception as e:
        components["model"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Check cache status
    try:
        if cache_manager:
            cache_stats = await cache_manager.get_stats()
            components["cache"] = {
                "status": "healthy",
                "redis_connected": cache_stats.get("redis_connected", False),
                "local_cache_size": cache_stats.get("local_cache_size", 0)
            }
        else:
            components["cache"] = {
                "status": "warning",
                "message": "Cache manager not available"
            }
    except Exception as e:
        components["cache"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Check system resources
    try:
        system_metrics = get_system_metrics()
        components["system"] = {
            "status": "healthy" if system_metrics.cpu_percent < 90 and system_metrics.memory_percent < 90 else "warning",
            "metrics": system_metrics.dict()
        }
    except Exception as e:
        components["system"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Overall status
    overall_status = "healthy"
    if any(comp.get("status") == "unhealthy" for comp in components.values()):
        overall_status = "unhealthy"
    elif any(comp.get("status") == "warning" for comp in components.values()):
        overall_status = "degraded"
    
    return HealthCheck(
        status=overall_status,
        timestamp=time.time(),
        uptime=uptime,
        components=components
    )

@router.get("/live")
async def liveness_check():
    """Simple liveness check for Kubernetes"""
    return {"status": "alive", "timestamp": time.time()}

@router.get("/ready")
async def readiness_check(
    model_manager: ModelManager = Depends(get_model_manager),
    cache_manager: CacheManager = Depends(get_cache_manager)
):
    """Readiness check - is the service ready to serve traffic?"""
    
    # Check if model is loaded
    if not model_manager or not model_manager.model:
        return {
            "status": "not_ready",
            "reason": "Model not loaded",
            "timestamp": time.time()
        }
    
    # Check cache connection (optional)
    if cache_manager and settings.ENABLE_CACHE:
        try:
            await cache_manager.get("health_check")
        except Exception:
            logger.warning("Cache not available for readiness check")
    
    return {
        "status": "ready",
        "timestamp": time.time()
    }

@router.get("/metrics", response_model=SystemMetrics)
async def get_metrics():
    """Get system metrics"""
    return get_system_metrics()

def get_system_metrics() -> SystemMetrics:
    """Collect system metrics"""
    
    # CPU and Memory
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # GPU metrics if available
    gpu_utilization = 0.0
    gpu_memory_percent = 0.0
    
    if torch.cuda.is_available():
        try:
            # GPU utilization (requires nvidia-ml-py for proper monitoring)
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0
            gpu_memory_percent = gpu_memory * 100
            
            # Simplified GPU utilization - in production you'd use nvidia-ml-py
            gpu_utilization = 50.0  # Placeholder
        except Exception:
            pass
    
    return SystemMetrics(
        cpu_percent=cpu_percent,
        memory_percent=memory.percent,
        gpu_utilization=gpu_utilization,
        gpu_memory_percent=gpu_memory_percent,
        disk_usage=disk.percent
    )