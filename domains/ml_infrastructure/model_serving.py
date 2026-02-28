"""
Model Serving - Production Model Deployment and Inference

Production-grade model serving with:
- Model server with REST API
- Batching and optimization
- Model versioning for A/B testing
- Health checks and monitoring
- Request queuing
"""

import time
import json
import logging
import threading
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger("sloughgpt.serving")


class ModelStatus(Enum):
    """Model deployment status."""
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNLOADED = "unloaded"


@dataclass
class InferenceRequest:
    """Single inference request."""
    request_id: str
    inputs: Any
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    timeout: float = 30.0


@dataclass
class InferenceResponse:
    """Inference response."""
    request_id: str
    outputs: Any
    latency_ms: float
    model_version: str
    error: Optional[str] = None


@dataclass
class ModelEndpoint:
    """Model endpoint configuration."""
    name: str
    model_name: str
    model_version: Optional[str] = None
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 1
    max_batch_size: int = 32
    batch_timeout_ms: int = 100
    workers: int = 1
    timeout: float = 30.0


class InferenceEngine:
    """Inference engine for model execution."""
    
    def __init__(self, model_fn: Callable, preproc: Optional[Callable] = None, postproc: Optional[Callable] = None):
        self.model_fn = model_fn
        self.preproc = preproc
        self.postproc = postproc
    
    def predict(self, inputs: Any) -> Any:
        """Run inference."""
        if self.preproc:
            inputs = self.preproc(inputs)
        
        outputs = self.model_fn(inputs)
        
        if self.postproc:
            outputs = self.postproc(outputs)
        
        return outputs


class BatchingManager:
    """Dynamic batching for inference."""
    
    def __init__(self, max_batch_size: int = 32, timeout_ms: int = 100):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.queue: List[InferenceRequest] = []
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
    
    def add_request(self, request: InferenceRequest) -> List[InferenceRequest]:
        """Add request to batch."""
        with self.condition:
            self.queue.append(request)
            
            if len(self.queue) >= self.max_batch_size:
                batch = self.queue[:self.max_batch_size]
                self.queue = self.queue[self.max_batch_size:]
                return batch
            
            self.condition.wait(timeout=self.timeout_ms / 1000)
            
            if self.queue:
                batch = self.queue
                self.queue = []
                return batch
            
            return []
    
    def get_batch(self) -> List[InferenceRequest]:
        """Get current batch."""
        with self.condition:
            if not self.queue:
                return []
            
            batch = self.queue[:self.max_batch_size]
            self.queue = self.queue[self.max_batch_size:]
            return batch


class ModelServer:
    """
    Production model server.
    
    Features:
    - Multiple model endpoints
    - Dynamic batching
    - A/B testing support
    - Health checks
    - Request queuing
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        
        self.endpoints: Dict[str, ModelEndpoint] = {}
        self.models: Dict[str, InferenceEngine] = {}
        self.batching: Dict[str, BatchingManager] = {}
        
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.request_queue: List[InferenceRequest] = []
        
        self._lock = threading.Lock()
        self._running = False
        
        self.stats = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_error": 0,
            "latency_sum": 0.0,
            "latency_count": 0,
        }
    
    def register_endpoint(self, endpoint: ModelEndpoint, model_fn: Callable,
                         preproc: Optional[Callable] = None, postproc: Optional[Callable] = None):
        """Register a model endpoint."""
        self.endpoints[endpoint.name] = endpoint
        
        engine = InferenceEngine(model_fn, preproc, postproc)
        self.models[endpoint.name] = engine
        
        self.batching[endpoint.name] = BatchingManager(
            max_batch_size=endpoint.max_batch_size,
            timeout_ms=endpoint.batch_timeout_ms
        )
        
        logger.info(f"Registered endpoint: {endpoint.name}")
    
    def predict(self, endpoint_name: str, inputs: Any, parameters: Optional[Dict[str, Any]] = None) -> InferenceResponse:
        """Run inference on endpoint."""
        start_time = time.time()
        request_id = hashlib.md5(f"{time.time()}{inputs}".encode()).hexdigest()[:12]
        
        endpoint = self.endpoints.get(endpoint_name)
        if not endpoint:
            return InferenceResponse(
                request_id=request_id,
                outputs=None,
                latency_ms=0,
                model_version="",
                error=f"Endpoint '{endpoint_name}' not found"
            )
        
        try:
            engine = self.models.get(endpoint_name)
            if not engine:
                return InferenceResponse(
                    request_id=request_id,
                    outputs=None,
                    latency_ms=0,
                    model_version=endpoint.model_version or "unknown",
                    error="Model not loaded"
                )
            
            outputs = engine.predict(inputs)
            
            latency_ms = (time.time() - start_time) * 1000
            
            with self._lock:
                self.stats["requests_total"] += 1
                self.stats["requests_success"] += 1
                self.stats["latency_sum"] += latency_ms
                self.stats["latency_count"] += 1
            
            return InferenceResponse(
                request_id=request_id,
                outputs=outputs,
                latency_ms=latency_ms,
                model_version=endpoint.model_version or "latest"
            )
        
        except Exception as e:
            with self._lock:
                self.stats["requests_total"] += 1
                self.stats["requests_error"] += 1
            
            return InferenceResponse(
                request_id=request_id,
                outputs=None,
                latency_ms=(time.time() - start_time) * 1000,
                model_version=endpoint.model_version or "unknown",
                error=str(e)
            )
    
    def predict_batch(self, endpoint_name: str, inputs: List[Any]) -> List[InferenceResponse]:
        """Run batch inference."""
        responses = []
        for inp in inputs:
            resp = self.predict(endpoint_name, inp)
            responses.append(resp)
        return responses
    
    def health_check(self, endpoint_name: Optional[str] = None) -> Dict[str, Any]:
        """Health check for endpoints."""
        if endpoint_name:
            endpoint = self.endpoints.get(endpoint_name)
            if not endpoint:
                return {"status": "not_found", "endpoint": endpoint_name}
            
            engine = self.models.get(endpoint_name)
            return {
                "status": "healthy" if engine else "unhealthy",
                "endpoint": endpoint_name,
                "model_loaded": engine is not None
            }
        
        return {
            "status": "healthy",
            "endpoints": list(self.endpoints.keys()),
            "total_requests": self.stats["requests_total"],
            "avg_latency_ms": self.stats["latency_sum"] / max(1, self.stats["latency_count"])
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            **self.stats,
            "avg_latency_ms": self.stats["latency_sum"] / max(1, self.stats["latency_count"]),
            "error_rate": self.stats["requests_error"] / max(1, self.stats["requests_total"])
        }


class ModelRouter:
    """Route requests to different model versions (A/B testing)."""
    
    def __init__(self):
        self.routes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def add_route(self, endpoint: str, model_version: str, weight: float, model_fn: Callable):
        """Add a route with traffic weight."""
        self.routes[endpoint].append({
            "version": model_version,
            "weight": weight,
            "model_fn": model_fn
        })
    
    def select_version(self, endpoint: str) -> Optional[Callable]:
        """Select model version based on weights."""
        routes = self.routes.get(endpoint, [])
        if not routes:
            return None
        
        total_weight = sum(r["weight"] for r in routes)
        r = np.random.random() * total_weight
        
        cumulative = 0
        for route in routes:
            cumulative += route["weight"]
            if r <= cumulative:
                return route["model_fn"]
        
        return routes[-1]["model_fn"]


server = ModelServer()


__all__ = [
    "ModelServer",
    "ModelEndpoint",
    "InferenceEngine",
    "InferenceRequest",
    "InferenceResponse",
    "BatchingManager",
    "ModelRouter",
    "ModelStatus",
    "server",
]
