"""
SloughGPT Real-time Model Serving
High-performance inference service with optimization and scaling
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from collections import defaultdict, deque

from sloughgpt.core.logging_system import get_logger, timer
from sloughgpt.core.performance import get_performance_optimizer

class InferenceStatus(Enum):
    """Inference request status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

class OptimizationType(Enum):
    """Inference optimization techniques"""
    BATCHING = "batching"
    CACHING = "caching"
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    EARLY_EXIT = "early_exit"

@dataclass
class InferenceRequest:
    """Individual inference request"""
    request_id: str
    prompt: str
    model_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: InferenceStatus = InferenceStatus.QUEUED
    result: Optional[str] = None
    error: Optional[str] = None
    tokens_generated: int = 0
    processing_time: float = 0.0
    
    @property
    def duration(self) -> float:
        """Calculate request processing duration"""
        if not self.started_at:
            return 0.0
        end_time = self.completed_at or time.time()
        return end_time - self.started_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary"""
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "model_name": self.model_name,
            "parameters": self.parameters,
            "priority": self.priority,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "tokens_generated": self.tokens_generated,
            "processing_time": self.processing_time,
            "duration": self.duration
        }

@dataclass
class ModelInstance:
    """Represents a loaded model instance"""
    model_id: str
    model_name: str
    model_path: str
    device: str  # "cpu", "cuda:0", "cuda:1", etc.
    loaded_at: float = field(default_factory=time.time)
    memory_usage_mb: float = 0.0
    is_quantized: bool = False
    optimization_type: Optional[OptimizationType] = None
    max_batch_size: int = 1
    current_batch_size: int = 0
    total_inferences: int = 0
    total_tokens_generated: int = 0
    avg_inference_time: float = 0.0
    cache_hit_rate: float = 0.0
    status: str = "idle"  # idle, loading, busy, error
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def utilization(self) -> float:
        """Calculate model utilization"""
        return self.current_batch_size / self.max_batch_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "loaded_at": self.loaded_at,
            "memory_usage_mb": self.memory_usage_mb,
            "is_quantized": self.is_quantized,
            "optimization_type": self.optimization_type.value if self.optimization_type else None,
            "max_batch_size": self.max_batch_size,
            "current_batch_size": self.current_batch_size,
            "total_inferences": self.total_inferences,
            "total_tokens_generated": self.total_tokens_generated,
            "avg_inference_time": self.avg_inference_time,
            "cache_hit_rate": self.cache_hit_rate,
            "utilization": self.utilization,
            "status": self.status,
            "metrics": self.metrics
        }

class InferenceOptimizer(ABC):
    """Abstract base class for inference optimizers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"optimizer_{self.__class__.__name__.lower()}")
    
    @abstractmethod
    async def optimize_request(self, request: InferenceRequest) -> InferenceRequest:
        """Optimize inference request"""
        pass
    
    @abstractmethod
    async def optimize_model(self, model: ModelInstance) -> ModelInstance:
        """Optimize model for inference"""
        pass

class BatchingOptimizer(InferenceOptimizer):
    """Batching optimizer for inference"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.batch_queue = deque()
        self.max_batch_size = config.get("max_batch_size", 8)
        self.batch_timeout = config.get("batch_timeout", 0.1)  # seconds
        self._batching_enabled = True
    
    async def optimize_request(self, request: InferenceRequest) -> InferenceRequest:
        """Add request to batch queue"""
        if not self._batching_enabled:
            return request
        
        # Add batching metadata
        request.metadata["batch_id"] = str(uuid.uuid4())
        request.metadata["batch_position"] = len(self.batch_queue)
        
        # Add to batch queue
        self.batch_queue.append(request)
        
        self.logger.debug(f"Request {request.request_id} added to batch queue (size: {len(self.batch_queue)})")
        
        # Process batch if full or timeout
        if len(self.batch_queue) >= self.max_batch_size:
            await self._process_batch()
        else:
            # Schedule batch processing
            asyncio.create_task(self._schedule_batch_processing())
        
        return request
    
    async def _schedule_batch_processing(self) -> None:
        """Schedule batch processing with timeout"""
        await asyncio.sleep(self.batch_timeout)
        
        if self.batch_queue:
            await self._process_batch()
    
    async def _process_batch(self) -> List[InferenceRequest]:
        """Process batch of requests"""
        if not self.batch_queue:
            return []
        
        batch = list(self.batch_queue)
        self.batch_queue.clear()
        
        self.logger.info(f"Processing batch of {len(batch)} requests")
        
        # Update batch metadata
        for i, request in enumerate(batch):
            request.metadata["batch_size"] = len(batch)
            request.metadata["batch_index"] = i
            request.started_at = time.time()
            request.status = InferenceStatus.PROCESSING
        
        return batch
    
    async def optimize_model(self, model: ModelInstance) -> ModelInstance:
        """Optimize model for batching"""
        model.max_batch_size = self.max_batch_size
        model.metrics["batching_enabled"] = True
        model.metrics["optimal_batch_size"] = self.max_batch_size
        
        return model

class CachingOptimizer(InferenceOptimizer):
    """Caching optimizer for inference"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cache_size = config.get("cache_size", 1000)
        self.cache_ttl = config.get("cache_ttl", 3600)  # seconds
        self._cache = {}
        self._cache_timestamps = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    async def optimize_request(self, request: InferenceRequest) -> InferenceRequest:
        """Check cache for request"""
        cache_key = self._generate_cache_key(request)
        
        # Check cache
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            
            # Check if cache entry is still valid
            if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                request.result = cache_entry["result"]
                request.tokens_generated = cache_entry["tokens_generated"]
                request.status = InferenceStatus.COMPLETED
                request.completed_at = time.time()
                request.metadata["cache_hit"] = True
                
                self._cache_hits += 1
                self.logger.debug(f"Cache hit for request {request.request_id}")
                return request
        
        self._cache_misses += 1
        request.metadata["cache_hit"] = False
        return request
    
    async def optimize_model(self, model: ModelInstance) -> ModelInstance:
        """Optimize model for caching"""
        model.metrics["cache_enabled"] = True
        model.metrics["cache_size"] = self.cache_size
        model.metrics["cache_ttl"] = self.cache_ttl
        
        return model
    
    def _generate_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request"""
        # Include relevant parameters in cache key
        key_data = {
            "prompt": request.prompt,
            "model_name": request.model_name,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p
        }
        
        # Create hash of key data
        key_str = json.dumps(key_data, sort_keys=True)
        return f"cache_{hash(key_str)}"
    
    def cache_result(self, request: InferenceRequest, result: str, tokens: int) -> None:
        """Cache inference result"""
        cache_key = self._generate_cache_key(request)
        
        self._cache[cache_key] = {
            "result": result,
            "tokens_generated": tokens,
            "timestamp": time.time()
        }
        
        self._cache_timestamps[cache_key] = time.time()
        
        # Cleanup old cache entries
        self._cleanup_cache()
    
    def _cleanup_cache(self) -> None:
        """Cleanup expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self._cache_timestamps.items():
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self._cache:
                del self._cache[key]
            if key in self._cache_timestamps:
                del self._cache_timestamps[key]
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_requests = self._cache_hits + self._cache_misses
        return self._cache_hits / total_requests if total_requests > 0 else 0.0

class QuantizationOptimizer(InferenceOptimizer):
    """Quantization optimizer for inference"""
    
    async def optimize_request(self, request: InferenceRequest) -> InferenceRequest:
        """Optimize request for quantized model"""
        request.metadata["quantized_inference"] = True
        return request
    
    async def optimize_model(self, model: ModelInstance) -> ModelInstance:
        """Optimize model with quantization"""
        self.logger.info(f"Quantizing model {model.model_id}")
        
        # Mock quantization process
        await asyncio.sleep(2.0)  # Simulate quantization time
        
        model.is_quantized = True
        model.optimization_type = OptimizationType.QUANTIZATION
        model.memory_usage_mb *= 0.5  # Quantized models use ~50% less memory
        model.metrics["quantization_type"] = "int8"
        model.metrics["compression_ratio"] = 4.0  # 4x compression
        
        self.logger.info(f"Model {model.model_id} quantized successfully")
        return model

class ModelServer:
    """Real-time model serving engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("model_server")
        self.optimizer = get_performance_optimizer()
        
        # Model management
        self.models: Dict[str, ModelInstance] = {}
        self.model_loaders: Dict[str, Callable] = {}
        
        # Request management
        self.request_queue = asyncio.PriorityQueue()
        self.active_requests: Dict[str, InferenceRequest] = {}
        self.completed_requests: Dict[str, InferenceRequest] = {}
        
        # Optimizers
        self.optimizers = {
            OptimizationType.BATCHING: BatchingOptimizer(config.get("batching", {})),
            OptimizationType.CACHING: CachingOptimizer(config.get("caching", {})),
            OptimizationType.QUANTIZATION: QuantizationOptimizer(config.get("quantization", {}))
        }
        
        # Performance tracking
        self._performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "tokens_per_second": 0.0,
            "cache_hit_rate": 0.0,
            "model_utilization": 0.0
        }
        
        # Background tasks
        self._background_tasks = set()
        self._running = False
    
    async def start(self) -> None:
        """Start the model server"""
        self.logger.info("Starting model server")
        self._running = True
        
        # Start background processing tasks
        self._background_tasks.add(asyncio.create_task(self._process_requests()))
        self._background_tasks.add(asyncio.create_task(self._monitor_performance()))
        self._background_tasks.add(asyncio.create_task(self._cleanup_completed_requests()))
        
        self.logger.info("Model server started successfully")
    
    async def stop(self) -> None:
        """Stop the model server"""
        self.logger.info("Stopping model server")
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Unload all models
        for model in self.models.values():
            await self.unload_model(model.model_id)
        
        self.logger.info("Model server stopped")
    
    async def load_model(self, model_name: str, model_path: str, device: str = "cpu",
                       optimizations: List[OptimizationType] = None) -> str:
        """Load a model for serving"""
        self.logger.info(f"Loading model {model_name} from {model_path} on {device}")
        
        model_id = str(uuid.uuid4())
        
        # Create model instance
        model = ModelInstance(
            model_id=model_id,
            model_name=model_name,
            model_path=model_path,
            device=device,
            status="loading"
        )
        
        try:
            # Apply optimizations
            optimizations = optimizations or []
            for opt_type in optimizations:
                optimizer = self.optimizers.get(opt_type)
                if optimizer:
                    model = await optimizer.optimize_model(model)
                    self.logger.info(f"Applied {opt_type.value} optimization to model {model_id}")
            
            # Mock model loading
            await asyncio.sleep(3.0)  # Simulate loading time
            
            # Set model metrics
            model.memory_usage_mb = 2048.0 if "cuda" in device else 1024.0
            model.max_batch_size = 16 if "cuda" in device else 4
            model.status = "idle"
            
            # Add to models
            self.models[model_id] = model
            
            self.logger.info(f"Model {model_id} loaded successfully")
            return model_id
            
        except Exception as e:
            model.status = "error"
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model"""
        self.logger.info(f"Unloading model {model_id}")
        
        if model_id not in self.models:
            self.logger.warning(f"Model {model_id} not found")
            return False
        
        model = self.models[model_id]
        
        # Wait for active requests to complete
        while model.current_batch_size > 0:
            await asyncio.sleep(0.1)
        
        # Remove from models
        del self.models[model_id]
        
        self.logger.info(f"Model {model_id} unloaded successfully")
        return True
    
    async def submit_request(self, prompt: str, model_name: str, **kwargs) -> str:
        """Submit inference request"""
        request_id = str(uuid.uuid4())
        
        request = InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            model_name=model_name,
            **kwargs
        )
        
        # Add to queue
        await self.request_queue.put((-request.priority, request_id, request))
        self.active_requests[request_id] = request
        
        self.logger.debug(f"Request {request_id} submitted for model {model_name}")
        
        return request_id
    
    async def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get request status"""
        request = self.active_requests.get(request_id, self.completed_requests.get(request_id))
        
        if not request:
            return {"error": "Request not found"}
        
        return request.to_dict()
    
    async def _process_requests(self) -> None:
        """Background task to process inference requests"""
        while self._running:
            try:
                # Get next request (with timeout)
                _, request_id, request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=1.0
                )
                
                # Process request
                await self._handle_request(request)
                
            except asyncio.TimeoutError:
                # No requests in queue
                continue
            except Exception as e:
                self.logger.error(f"Error processing request: {e}")
                continue
    
    async def _handle_request(self, request: InferenceRequest) -> None:
        """Handle individual inference request"""
        self.logger.debug(f"Processing request {request.request_id}")
        
        try:
            request.started_at = time.time()
            request.status = InferenceStatus.PROCESSING
            
            # Find suitable model
            model = await self._find_suitable_model(request)
            if not model:
                raise Exception(f"No suitable model found for {request.model_name}")
            
            # Apply optimizations
            for optimizer in self.optimizers.values():
                request = await optimizer.optimize_request(request)
            
            # Check if request was cached
            if request.status == InferenceStatus.COMPLETED:
                # Cache the result
                cache_optimizer = self.optimizers.get(OptimizationType.CACHING)
                if cache_optimizer:
                    cache_optimizer.cache_result(request, request.result, request.tokens_generated)
                
                self._complete_request(request)
                return
            
            # Perform inference
            model.current_batch_size += 1
            model.status = "busy"
            
            # Mock inference
            inference_time = await self._perform_inference(model, request)
            
            # Update model metrics
            model.total_inferences += 1
            model.total_tokens_generated += request.tokens_generated
            model.current_batch_size -= 1
            model.status = "idle" if model.current_batch_size == 0 else "busy"
            
            # Update average inference time
            model.avg_inference_time = (
                (model.avg_inference_time * (model.total_inferences - 1) + inference_time) /
                model.total_inferences
            )
            
            request.completed_at = time.time()
            request.status = InferenceStatus.COMPLETED
            
            # Cache the result
            cache_optimizer = self.optimizers.get(OptimizationType.CACHING)
            if cache_optimizer:
                cache_optimizer.cache_result(request, request.result, request.tokens_generated)
            
            self._complete_request(request)
            
        except Exception as e:
            request.status = InferenceStatus.FAILED
            request.error = str(e)
            request.completed_at = time.time()
            self.logger.error(f"Request {request.request_id} failed: {e}")
            
            self._complete_request(request)
    
    async def _find_suitable_model(self, request: InferenceRequest) -> Optional[ModelInstance]:
        """Find suitable model for request"""
        for model in self.models.values():
            if model.model_name == request.model_name and model.status in ["idle", "busy"]:
                # Check if model can handle more requests
                if model.current_batch_size < model.max_batch_size:
                    return model
        return None
    
    async def _perform_inference(self, model: ModelInstance, request: InferenceRequest) -> float:
        """Perform actual inference"""
        # Mock inference with realistic timing
        base_time = 0.5 if "cuda" in model.device else 2.0
        
        # Adjust for optimizations
        if model.is_quantized:
            base_time *= 0.7  # 30% faster
        
        if model.current_batch_size > 1:
            # Batch processing is more efficient
            batch_efficiency = min(model.current_batch_size / 4.0, 2.0)
            base_time /= batch_efficiency
        
        # Simulate inference time
        await asyncio.sleep(base_time)
        
        # Generate mock result
        request.result = f"Generated response for: {request.prompt[:50]}..."
        request.tokens_generated = request.max_tokens // 4  # Simulate token generation
        
        return base_time
    
    def _complete_request(self, request: InferenceRequest) -> None:
        """Complete a request"""
        # Move from active to completed
        if request.request_id in self.active_requests:
            del self.active_requests[request.request_id]
        
        self.completed_requests[request.request_id] = request
        
        # Update performance metrics
        self._update_performance_metrics(request)
    
    def _update_performance_metrics(self, request: InferenceRequest) -> None:
        """Update performance metrics"""
        self._performance_metrics["total_requests"] += 1
        
        if request.status == InferenceStatus.COMPLETED:
            self._performance_metrics["successful_requests"] += 1
        else:
            self._performance_metrics["failed_requests"] += 1
        
        # Update average response time
        total_successful = self._performance_metrics["successful_requests"]
        current_avg = self._performance_metrics["avg_response_time"]
        
        self._performance_metrics["avg_response_time"] = (
            (current_avg * (total_successful - 1) + request.duration) /
            total_successful
        )
        
        # Update tokens per second
        if request.duration > 0:
            tokens_per_sec = request.tokens_generated / request.duration
            current_tps = self._performance_metrics["tokens_per_second"]
            self._performance_metrics["tokens_per_second"] = (
                (current_tps * (total_successful - 1) + tokens_per_sec) /
                total_successful
            )
        
        # Update cache hit rate
        cache_optimizer = self.optimizers.get(OptimizationType.CACHING)
        if cache_optimizer:
            self._performance_metrics["cache_hit_rate"] = cache_optimizer.hit_rate
    
    async def _monitor_performance(self) -> None:
        """Background task to monitor performance"""
        while self._running:
            try:
                # Calculate model utilization
                if self.models:
                    total_utilization = sum(model.utilization for model in self.models.values())
                    self._performance_metrics["model_utilization"] = total_utilization / len(self.models)
                
                # Log performance metrics
                self.logger.debug(f"Performance metrics: {self._performance_metrics}")
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_completed_requests(self) -> None:
        """Background task to cleanup completed requests"""
        while self._running:
            try:
                current_time = time.time()
                
                # Remove requests completed more than 1 hour ago
                expired_requests = [
                    req_id for req_id, request in self.completed_requests.items()
                    if current_time - request.completed_at > 3600
                ]
                
                for req_id in expired_requests:
                    del self.completed_requests[req_id]
                
                if expired_requests:
                    self.logger.debug(f"Cleaned up {len(expired_requests)} expired requests")
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(300)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive server metrics"""
        return {
            "server_metrics": self._performance_metrics,
            "model_metrics": {
                model_id: model.to_dict() for model_id, model in self.models.items()
            },
            "queue_metrics": {
                "active_requests": len(self.active_requests),
                "queued_requests": self.request_queue.qsize(),
                "completed_requests": len(self.completed_requests)
            },
            "optimizer_metrics": {
                opt_type.value: {
                    "enabled": True,
                    "metrics": getattr(optimizer, 'metrics', {})
                }
                for opt_type, optimizer in self.optimizers.items()
            }
        }

# Global model server instance
_global_model_server: Optional[ModelServer] = None

def get_model_server(config: Optional[Dict[str, Any]] = None) -> ModelServer:
    """Get or create global model server"""
    global _global_model_server
    if _global_model_server is None:
        _global_model_server = ModelServer(config or {})
    return _global_model_server

# Decorators for easy use
def real_time_inference(model_name: str, optimizations: List[OptimizationType] = None):
    """Decorator for real-time inference"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            server = get_model_server()
            
            # Extract prompt from arguments
            prompt = kwargs.get("prompt", str(args[0]) if args else "")
            
            # Submit inference request
            request_id = await server.submit_request(
                prompt=prompt,
                model_name=model_name,
                optimizations=optimizations or [],
                **kwargs
            )
            
            # Wait for completion
            while True:
                status = await server.get_request_status(request_id)
                
                if status["status"] in ["completed", "failed", "timeout"]:
                    if status["status"] == "completed":
                        return status["result"]
                    else:
                        raise Exception(f"Inference failed: {status.get('error', 'Unknown error')}")
                
                await asyncio.sleep(0.1)  # Poll every 100ms
        
        return wrapper
    return decorator