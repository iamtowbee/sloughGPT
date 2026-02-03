from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
import time

from ..monitoring.metrics_collector import metrics_collector
from ..core.model_manager import ModelManager
from ..core.cache_manager import CacheManager
from ..dependencies import get_model_manager, get_cache_manager

logger = logging.getLogger(__name__)
router = APIRouter()

class MetricsResponse(BaseModel):
    timestamp: float = Field(..., description="Response timestamp")
    metrics: Dict[str, Any] = Field(..., description="Metrics data")

@router.get("/overview", response_model=MetricsResponse)
async def get_metrics_overview():
    """Get comprehensive metrics overview"""
    try:
        # Recent request metrics
        recent_5min = metrics_collector.get_recent_metrics(minutes=5)
        recent_1hr = metrics_collector.get_recent_metrics(minutes=60)
        
        # Current system and model metrics
        current_system = metrics_collector.get_current_system_metrics()
        current_model = metrics_collector.get_current_model_metrics()
        
        # Error summary
        error_summary = metrics_collector.get_error_summary(minutes=60)
        
        # Endpoint summaries
        endpoint_summaries = metrics_collector.get_all_endpoint_summaries()
        
        overview = {
            "recent_activity": {
                "last_5_minutes": recent_5min,
                "last_hour": recent_1hr
            },
            "current_status": {
                "system": {
                    "cpu_percent": current_system.cpu_percent if current_system else 0,
                    "memory_percent": current_system.memory_percent if current_system else 0,
                    "gpu_utilization": current_system.gpu_utilization if current_system else 0,
                    "gpu_memory_used_mb": current_system.gpu_memory_used_mb if current_system else 0
                } if current_system else {},
                "model": {
                    "active_requests": current_model.active_requests if current_model else 0,
                    "cache_hit_rate": current_model.cache_hit_rate if current_model else 0,
                    "total_requests": current_model.total_requests if current_model else 0
                } if current_model else {}
            },
            "performance": {
                "top_endpoints": sorted(
                    endpoint_summaries.items(),
                    key=lambda x: x[1].get("total_requests", 0),
                    reverse=True
                )[:10],
                "slow_endpoints": sorted(
                    endpoint_summaries.items(),
                    key=lambda x: x[1].get("avg_duration", 0),
                    reverse=True
                )[:5]
            },
            "errors": error_summary,
            "active_requests_count": len(metrics_collector.active_requests)
        }
        
        return MetricsResponse(
            timestamp=time.time(),
            metrics=overview
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics overview")

@router.get("/requests")
async def get_request_metrics(
    minutes: int = Query(5, ge=1, le=1440, description="Time range in minutes")
):
    """Get request metrics for specified time range"""
    try:
        metrics = metrics_collector.get_recent_metrics(minutes=minutes)
        return {
            "time_range_minutes": minutes,
            "metrics": metrics,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get request metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get request metrics")

@router.get("/system")
async def get_system_metrics():
    """Get current system metrics"""
    try:
        current = metrics_collector.get_current_system_metrics()
        if not current:
            return {"message": "No system metrics available"}
        
        return {
            "system_metrics": {
                "timestamp": current.timestamp,
                "cpu_percent": current.cpu_percent,
                "memory_percent": current.memory_percent,
                "memory_used_mb": current.memory_used_mb,
                "disk_usage_percent": current.disk_usage_percent,
                "gpu_utilization": current.gpu_utilization,
                "gpu_memory_used_mb": current.gpu_memory_used_mb,
                "gpu_memory_total_mb": current.gpu_memory_total_mb,
                "active_connections": current.active_connections
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system metrics")

@router.get("/model")
async def get_model_metrics():
    """Get current model metrics"""
    try:
        current = metrics_collector.get_current_model_metrics()
        if not current:
            return {"message": "No model metrics available"}
        
        return {
            "model_metrics": {
                "timestamp": current.timestamp,
                "model_name": current.model_name,
                "active_requests": current.active_requests,
                "queue_length": current.queue_length,
                "avg_inference_time": current.avg_inference_time,
                "cache_hit_rate": current.cache_hit_rate,
                "batch_utilization": current.batch_utilization,
                "total_requests": current.total_requests,
                "total_tokens_generated": current.total_tokens_generated
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get model metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model metrics")

@router.get("/endpoints")
async def get_endpoint_metrics(
    endpoint: Optional[str] = Query(None, description="Specific endpoint name")
):
    """Get endpoint performance metrics"""
    try:
        if endpoint:
            summary = metrics_collector.get_endpoint_summary(endpoint)
            return {
                "endpoint": endpoint,
                "metrics": summary,
                "timestamp": time.time()
            }
        else:
            summaries = metrics_collector.get_all_endpoint_summaries()
            return {
                "endpoint_summaries": summaries,
                "timestamp": time.time()
            }
            
    except Exception as e:
        logger.error(f"Failed to get endpoint metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get endpoint metrics")

@router.get("/errors")
async def get_error_metrics(
    minutes: int = Query(60, ge=1, le=1440, description="Time range in minutes")
):
    """Get error metrics and trends"""
    try:
        error_summary = metrics_collector.get_error_summary(minutes=minutes)
        return {
            "error_metrics": error_summary,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get error metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get error metrics")

@router.get("/performance")
async def get_performance_metrics(
    model_manager: ModelManager = Depends(get_model_manager),
    cache_manager: CacheManager = Depends(get_cache_manager)
):
    """Get detailed performance metrics"""
    try:
        # Cache stats
        cache_stats = await cache_manager.get_stats() if cache_manager else {}
        
        # Batch stats
        batch_stats = await model_manager.get_batch_stats() if model_manager else {}
        
        # Request performance
        recent_metrics = metrics_collector.get_recent_metrics(minutes=15)
        
        performance = {
            "cache": cache_stats,
            "batch_processing": batch_stats,
            "recent_performance": recent_metrics,
            "optimization_status": {
                "async_file_io": True,
                "redis_caching": cache_stats.get("redis", {}).get("connected", False),
                "request_batching": batch_stats.get("batch_processor_enabled", False),
                "monitoring_active": metrics_collector.is_collecting
            }
        }
        
        return {
            "performance_metrics": performance,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")

@router.post("/export")
async def export_metrics(
    format: str = Query("json", regex="^(json|csv)$", description="Export format")
):
    """Export all metrics data"""
    try:
        if format == "json":
            data = metrics_collector.export_metrics("json")
            return {
                "format": "json",
                "data": data,
                "timestamp": time.time()
            }
        else:
            # CSV export would require additional implementation
            raise HTTPException(status_code=400, detail="CSV export not yet implemented")
            
    except Exception as e:
        logger.error(f"Failed to export metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to export metrics")

@router.post("/reset")
async def reset_metrics():
    """Reset all collected metrics"""
    try:
        metrics_collector.reset_metrics()
        return {
            "message": "All metrics reset successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to reset metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset metrics")