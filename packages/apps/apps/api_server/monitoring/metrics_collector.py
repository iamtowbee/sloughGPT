import logging
import time
import asyncio
import psutil
import torch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import threading

from ..core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class RequestMetrics:
    """Individual request metrics"""
    request_id: str
    endpoint: str
    method: str
    status_code: int
    duration: float
    timestamp: float
    user_agent: str = ""
    ip_address: str = ""
    model_inference_time: float = 0.0
    cache_hit: bool = False
    batch_size: int = 1
    tokens_generated: int = 0

@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    gpu_utilization: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    active_connections: int = 0

@dataclass
class ModelMetrics:
    """Model-specific metrics"""
    timestamp: float
    model_name: str
    active_requests: int
    queue_length: int
    avg_inference_time: float
    cache_hit_rate: float
    batch_utilization: float
    total_requests: int = 0
    total_tokens_generated: int = 0

class MetricsCollector:
    """Comprehensive metrics collection system"""
    
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        
        # Metrics storage
        self.request_metrics: deque = deque(maxlen=max_history_size)
        self.system_metrics: deque = deque(maxlen=1000)  # Keep fewer system metrics
        self.model_metrics: deque = deque(maxlen=1000)
        
        # Aggregated statistics
        self.endpoint_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_requests": 0,
            "total_duration": 0.0,
            "avg_duration": 0.0,
            "status_codes": defaultdict(int),
            "last_request": None
        })
        
        # Real-time counters
        self.active_requests: Dict[str, float] = {}  # request_id -> start_time
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Background collection
        self.is_collecting = False
        self.collection_task = None
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    async def start_collection(self, interval: float = 5.0):
        """Start background metrics collection"""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop(interval))
        logger.info(f"Started metrics collection with {interval}s interval")
    
    async def stop_collection(self):
        """Stop background metrics collection"""
        self.is_collecting = False
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped metrics collection")
    
    async def _collection_loop(self, interval: float):
        """Background collection loop"""
        while self.is_collecting:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics.append(system_metrics)
                
                # Collect model metrics
                model_metrics = await self._collect_model_metrics()
                if model_metrics:
                    self.model_metrics.append(model_metrics)
                
                # Cleanup old active requests
                await self._cleanup_active_requests()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(interval)
    
    def record_request_start(self, request_id: str, endpoint: str, method: str,
                           user_agent: str = "", ip_address: str = ""):
        """Record the start of a request"""
        with self._lock:
            self.active_requests[request_id] = time.time()
    
    def record_request_end(self, request_id: str, endpoint: str, method: str,
                          status_code: int, user_agent: str = "", ip_address: str = "",
                          model_inference_time: float = 0.0, cache_hit: bool = False,
                          batch_size: int = 1, tokens_generated: int = 0):
        """Record the completion of a request"""
        start_time = self.active_requests.pop(request_id, time.time())
        duration = time.time() - start_time
        
        metrics = RequestMetrics(
            request_id=request_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration=duration,
            timestamp=time.time(),
            user_agent=user_agent,
            ip_address=ip_address,
            model_inference_time=model_inference_time,
            cache_hit=cache_hit,
            batch_size=batch_size,
            tokens_generated=tokens_generated
        )
        
        with self._lock:
            self.request_metrics.append(metrics)
            
            # Update endpoint statistics
            stats = self.endpoint_stats[endpoint]
            stats["total_requests"] += 1
            stats["total_duration"] += duration
            stats["avg_duration"] = stats["total_duration"] / stats["total_requests"]
            stats["status_codes"][status_code] += 1
            stats["last_request"] = time.time()
            
            # Track errors
            if status_code >= 400:
                self.error_counts[endpoint] += 1
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system resource metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics if available
        gpu_utilization = 0.0
        gpu_memory_used_mb = 0.0
        gpu_memory_total_mb = 0.0
        
        if torch.cuda.is_available():
            try:
                # GPU memory
                gpu_memory_used = torch.cuda.memory_allocated()
                gpu_memory_total = torch.cuda.max_memory_allocated()
                gpu_memory_used_mb = gpu_memory_used / (1024**2)
                gpu_memory_total_mb = gpu_memory_total / (1024**2)
                
                # GPU utilization (simplified - would use nvidia-ml-py in production)
                gpu_utilization = 50.0  # Placeholder
            except Exception:
                pass
        
        # Active network connections (simplified)
        active_connections = len(psutil.net_connections())
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024**2),
            disk_usage_percent=disk.percent,
            gpu_utilization=gpu_utilization,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            active_connections=active_connections
        )
    
    async def _collect_model_metrics(self) -> Optional[ModelMetrics]:
        """Collect model-specific metrics"""
        # This would integrate with your model manager
        # For now, return basic metrics
        try:
            from ..dependencies import _model_manager as model_manager, _cache_manager as cache_manager
            
            active_requests = len(self.active_requests)
            queue_length = 0  # Would get from batch processor
            
            # Calculate cache hit rate
            cache_hit_rate = 0.0
            if _cache_manager:
                stats = await _cache_manager.get_stats()
                cache_hit_rate = stats.get("performance", {}).get("hit_rate", 0.0)
            
            return ModelMetrics(
                timestamp=time.time(),
                model_name="slogpt",
                active_requests=active_requests,
                queue_length=queue_length,
                avg_inference_time=0.0,  # Would calculate from recent requests
                cache_hit_rate=cache_hit_rate,
                batch_utilization=0.0,  # Would get from batch processor
                total_requests=sum(stats["total_requests"] for stats in self.endpoint_stats.values()),
                total_tokens_generated=0  # Would sum from request metrics
            )
            
        except Exception as e:
            logger.debug(f"Failed to collect model metrics: {e}")
            return None
    
    async def _cleanup_active_requests(self):
        """Clean up stale active requests"""
        current_time = time.time()
        timeout = 300  # 5 minutes
        
        stale_requests = []
        for request_id, start_time in self.active_requests.items():
            if current_time - start_time > timeout:
                stale_requests.append(request_id)
        
        for request_id in stale_requests:
            del self.active_requests[request_id]
            logger.warning(f"Cleaned up stale request: {request_id}")
    
    def get_recent_metrics(self, minutes: int = 5) -> Dict[str, Any]:
        """Get metrics from the last N minutes"""
        cutoff_time = time.time() - (minutes * 60)
        
        # Filter recent requests
        recent_requests = [
            m for m in self.request_metrics 
            if m.timestamp >= cutoff_time
        ]
        
        # Calculate statistics
        total_requests = len(recent_requests)
        avg_duration = sum(r.duration for r in recent_requests) / total_requests if total_requests > 0 else 0
        cache_hits = sum(1 for r in recent_requests if r.cache_hit)
        cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0
        
        # Status code distribution
        status_codes = defaultdict(int)
        for r in recent_requests:
            status_codes[r.status_code] += 1
        
        return {
            "time_range_minutes": minutes,
            "total_requests": total_requests,
            "avg_duration": avg_duration,
            "cache_hit_rate": cache_hit_rate,
            "status_codes": dict(status_codes),
            "active_requests": len(self.active_requests),
            "timestamp": time.time()
        }
    
    def get_endpoint_summary(self, endpoint: str) -> Dict[str, Any]:
        """Get summary for specific endpoint"""
        return dict(self.endpoint_stats.get(endpoint, {}))
    
    def get_all_endpoint_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries for all endpoints"""
        return {k: dict(v) for k, v in self.endpoint_stats.items()}
    
    def get_current_system_metrics(self) -> Optional[SystemMetrics]:
        """Get most recent system metrics"""
        return self.system_metrics[-1] if self.system_metrics else None
    
    def get_current_model_metrics(self) -> Optional[ModelMetrics]:
        """Get most recent model metrics"""
        return self.model_metrics[-1] if self.model_metrics else None
    
    def get_error_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get error summary for recent period"""
        cutoff_time = time.time() - (minutes * 60)
        
        recent_errors = [
            m for m in self.request_metrics 
            if m.timestamp >= cutoff_time and m.status_code >= 400
        ]
        
        error_by_endpoint = defaultdict(list)
        for error in recent_errors:
            error_by_endpoint[error.endpoint].append(error)
        
        return {
            "time_range_minutes": minutes,
            "total_errors": len(recent_errors),
            "errors_by_endpoint": {
                endpoint: {
                    "count": len(errors),
                    "avg_status_code": sum(e.status_code for e in errors) / len(errors),
                    "recent_examples": [
                        {
                            "status_code": e.status_code,
                            "duration": e.duration,
                            "timestamp": e.timestamp
                        }
                        for e in errors[-3:]  # Last 3 errors
                    ]
                }
                for endpoint, errors in error_by_endpoint.items()
            },
            "timestamp": time.time()
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export all metrics in specified format"""
        data = {
            "export_timestamp": time.time(),
            "request_metrics": [asdict(m) for m in list(self.request_metrics)],
            "system_metrics": [asdict(m) for m in list(self.system_metrics)],
            "model_metrics": [asdict(m) for m in list(self.model_metrics)],
            "endpoint_summaries": self.get_all_endpoint_summaries(),
            "error_counts": dict(self.error_counts)
        }
        
        if format.lower() == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def reset_metrics(self):
        """Reset all collected metrics"""
        with self._lock:
            self.request_metrics.clear()
            self.system_metrics.clear()
            self.model_metrics.clear()
            self.endpoint_stats.clear()
            self.active_requests.clear()
            self.error_counts.clear()
        
        logger.info("Metrics reset")

# Global metrics collector instance
metrics_collector = MetricsCollector()