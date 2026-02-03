"""
SloughGPT Performance Optimization System
Model quantization, caching, and performance optimization utilities
"""

import time
import hashlib
import pickle
import json
import threading
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Generic, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor
import asyncio
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    from torch.quantization import quantize_dynamic, prepare_qat
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from .logging_system import get_logger, timer
from .exceptions import PerformanceError, create_error, SloughGPTErrorCode

T = TypeVar('T')

class CacheLevel(Enum):
    """Cache performance levels"""
    NO_CACHE = "no_cache"
    MEMORY = "memory_cache"
    DISK = "disk_cache"
    DISTRIBUTED = "distributed_cache"

class QuantizationLevel(Enum):
    """Model quantization levels"""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    DYNAMIC = "dynamic"

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    value: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    size_bytes: int = 0
    hit_count: int = 0
    
    @property
    def is_expired(self) -> bool:
        return time.time() > (self.timestamp + self.ttl)
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

@dataclass
class PerformanceMetrics:
    """Performance metrics collection"""
    operation_times: Dict[str, List[float]] = field(default_factory=dict)
    cache_hit_rate: Dict[str, float] = field(default_factory=lambda: {"hit": 0, "total": 0})
    quantization_stats: Dict[str, Any] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    
    def record_operation(self, operation: str, duration_ms: float):
        """Record operation timing"""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        self.operation_times[operation].append(duration_ms)
        
        # Keep only last 1000 measurements
        if len(self.operation_times[operation]) > 1000:
            self.operation_times[operation] = self.operation_times[operation][-1000:]
    
    def get_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for an operation"""
        if operation not in self.operation_times:
            return {}
        
        times = self.operation_times[operation]
        if not times:
            return {}
        
        return {
            "count": len(times),
            "avg_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "p50_ms": sorted(times)[len(times)//2],
            "p95_ms": sorted(times)[int(len(times)*0.95)],
            "p99_ms": sorted(times)[int(len(times)*0.99)]
        }
    
    def record_cache_hit(self, hit: bool):
        """Record cache hit/miss"""
        if hit:
            self.cache_hit_rate["hit"] += 1
        self.cache_hit_rate["total"] += 1
    
    @property
    def cache_hit_rate_percent(self) -> float:
        total = self.cache_hit_rate["total"]
        if total == 0:
            return 0.0
        return (self.cache_hit_rate["hit"] / total) * 100

class MemoryCache:
    """Thread-safe in-memory cache with TTL"""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
        self.logger = get_logger("memory_cache")
        self.metrics = PerformanceMetrics()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            entry = self.cache.get(key)
            
            if entry is None:
                self.metrics.record_cache_hit(False)
                return None
            
            if entry.is_expired:
                del self.cache[key]
                self.metrics.record_cache_hit(False)
                return None
            
            entry.access_count += 1
            entry.hit_count += 1
            self.metrics.record_cache_hit(True)
            
            self.logger.debug(f"Cache hit: {key}", 
                           key=key, 
                           access_count=entry.access_count,
                           age_seconds=entry.age_seconds)
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache"""
        with self.lock:
            if ttl is None:
                ttl = self.default_ttl
            
            # Calculate size (rough estimation)
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = len(str(value))
            
            # Evict if necessary
            self._ensure_space(size_bytes)
            
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            self.cache[key] = entry
            
            self.logger.debug(f"Cache put: {key}",
                           key=key,
                           ttl=ttl,
                           size_bytes=size_bytes)
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.logger.debug(f"Cache delete: {key}", key=key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            self.logger.info(f"Cache cleared", entries_deleted=count)
    
    def _ensure_space(self, required_bytes: int) -> None:
        """Ensure enough space, evict if necessary"""
        current_size = sum(entry.size_bytes for entry in self.cache.values())
        
        while (len(self.cache) >= self.max_size or 
               current_size + required_bytes > self.max_size * 1024 * 1024):  # Max 1GB
            # Evict least recently used
            oldest_key = min(self.cache.keys(), 
                          key=lambda k: self.cache[k].access_count)
            
            if oldest_key in self.cache:
                current_size -= self.cache[oldest_key].size_bytes
                del self.cache[oldest_key]
                self.logger.debug(f"Cache evict: {oldest_key}", key=oldest_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_entries = len(self.cache)
            expired_entries = sum(1 for entry in self.cache.values() if entry.is_expired)
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            total_hits = sum(entry.hit_count for entry in self.cache.values())
            
            return {
                "total_entries": total_entries,
                "expired_entries": expired_entries,
                "active_entries": total_entries - expired_entries,
                "total_size_bytes": total_size,
                "average_entry_size": total_size / max(1, total_entries),
                "total_hits": total_hits,
                "cache_hit_rate": self.metrics.cache_hit_rate_percent,
                "memory_usage_mb": total_size / (1024 * 1024)
            }

class DiskCache:
    """Persistent disk cache with compression"""
    
    def __init__(self, cache_dir: str = ".cache", max_size_mb: int = 1024):
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        self.logger = get_logger("disk_cache")
        self.metrics = PerformanceMetrics()
        
        import os
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> str:
        """Get file path for cache key"""
        # Use hash for safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return f"{self.cache_dir}/{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        import os
        
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            self.metrics.record_cache_hit(False)
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                
            entry = data.get("entry")
            if entry and entry.is_expired:
                os.remove(cache_path)
                self.metrics.record_cache_hit(False)
                return None
            
            if entry:
                entry.access_count += 1
                entry.hit_count += 1
                self.metrics.record_cache_hit(True)
                return entry.value
                
        except Exception as e:
            self.logger.error(f"Disk cache get error: {e}", key=key, path=cache_path)
            # Remove corrupted cache file
            try:
                os.remove(cache_path)
            except:
                pass
        
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in disk cache"""
        import os
        
        cache_path = self._get_cache_path(key)
        
        # Calculate size
        try:
            data_bytes = pickle.dumps({"entry": CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl or 300,
                size_bytes=len(pickle.dumps(value))
            )})
            size_bytes = len(data_bytes)
        except Exception as e:
            self.logger.error(f"Disk cache serialization error: {e}", key=key)
            return
        
        # Ensure enough space
        self._ensure_space(size_bytes)
        
        try:
            with open(cache_path, 'wb') as f:
                f.write(data_bytes)
                
            self.logger.debug(f"Disk cache put: {key}", 
                           key=key,
                           ttl=ttl or 300,
                           size_bytes=size_bytes)
                           
        except Exception as e:
            self.logger.error(f"Disk cache put error: {e}", key=key, path=cache_path)
    
    def delete(self, key: str) -> bool:
        """Delete value from disk cache"""
        import os
        
        cache_path = self._get_cache_path(key)
        
        if os.path.exists(cache_path):
            os.remove(cache_path)
            self.logger.debug(f"Disk cache delete: {key}", key=key)
            return True
        return False
    
    def _ensure_space(self, required_bytes: int) -> None:
        """Ensure enough disk space, cleanup if necessary"""
        import os
        
        # Get current cache size
        total_size = 0
        file_count = 0
        
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    filepath = os.path.join(self.cache_dir, filename)
                    total_size += os.path.getsize(filepath)
                    file_count += 1
        
        max_bytes = self.max_size_mb * 1024 * 1024
        
        while total_size + required_bytes > max_bytes and file_count > 0:
            # Remove oldest file
            oldest_file = min(
                [f for f in os.listdir(self.cache_dir) if f.endswith('.cache')],
                key=lambda f: os.path.getmtime(os.path.join(self.cache_dir, f))
            )
            
            oldest_path = os.path.join(self.cache_dir, oldest_file)
            file_size = os.path.getsize(oldest_path)
            
            os.remove(oldest_path)
            total_size -= file_size
            file_count -= 1
            
            self.logger.debug(f"Disk cache evict: {oldest_file}", 
                           file=oldest_file,
                           freed_bytes=file_size)

class ModelQuantizer:
    """Model quantization utilities"""
    
    def __init__(self, level: QuantizationLevel = QuantizationLevel.DYNAMIC):
        self.level = level
        self.logger = get_logger("model_quantizer")
        self.quantized_models = {}
        self.metrics = PerformanceMetrics()
    
    def quantize_model(self, model: nn.Module, example_input: Any) -> nn.Module:
        """Quantize a PyTorch model"""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available for quantization")
            return model
        
        model_id = id(model)
        
        if model_id in self.quantized_models:
            self.logger.info(f"Model already quantized", model_id=model_id)
            return self.quantized_models[model_id]
        
        with timer("model_quantization") as perf_timer:
            try:
                if self.level == QuantizationLevel.NONE:
                    quantized = model
                elif self.level == QuantizationLevel.DYNAMIC:
                    quantized = quantize_dynamic(model, {example_input})
                    self.logger.info(f"Dynamic quantization completed", 
                                   level=self.level.value,
                                   input_shape=example_input.shape if hasattr(example_input, 'shape') else 'unknown')
                elif self.level == QuantizationLevel.FP16:
                    quantized = model.half()
                    self.logger.info(f"FP16 quantization completed", level=self.level.value)
                elif self.level == QuantizationLevel.INT8:
                    # Static quantization requires calibration
                    model.eval()
                    quantized = quantize_dynamic(model, {example_input})  # Simplified INT8
                    self.logger.info(f"INT8 quantization completed", level=self.level.value)
                else:
                    self.logger.warning(f"Unsupported quantization level: {self.level}")
                    return model
                
                self.quantized_models[model_id] = quantized
                
                # Record quantization stats
                self.metrics.quantization_stats[str(model_id)] = {
                    "level": self.level.value,
                    "original_size_mb": self._get_model_size(model),
                    "quantized_size_mb": self._get_model_size(quantized),
                    "compression_ratio": self._get_model_size(model) / max(1, self._get_model_size(quantized)),
                    "quantization_time_ms": perf_timer.duration * 1000 if perf_timer.duration else 0
                }
                
                return quantized
                
            except Exception as e:
                error = create_error(
                    PerformanceError,
                    f"Model quantization failed: {str(e)}",
                    SloughGPTErrorCode.MODEL_QUANTIZATION_FAILED,
                    cause=e,
                    context={"quantization_level": self.level.value}
                )
                error.log_error()
                return model
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Estimate model size in MB"""
        try:
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            return param_size / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    def get_quantization_stats(self, model: nn.Module) -> Dict[str, Any]:
        """Get quantization statistics for a model"""
        model_id = str(id(model))
        return self.metrics.quantization_stats.get(model_id, {})

class PerformanceOptimizer:
    """Main performance optimization coordinator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("performance_optimizer")
        
        # Initialize caches
        self.memory_cache = MemoryCache(
            max_size=self.config.get("memory_cache_size", 1000),
            default_ttl=self.config.get("memory_cache_ttl", 300)
        )
        
        self.disk_cache = DiskCache(
            cache_dir=self.config.get("disk_cache_dir", ".cache"),
            max_size_mb=self.config.get("disk_cache_size", 1024)
        )
        
        # Initialize quantizer
        quant_level = QuantizationLevel(self.config.get("quantization_level", "dynamic"))
        self.quantizer = ModelQuantizer(quant_level)
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get("max_workers", 4)
        )
        
        self.metrics = PerformanceMetrics()
    
    def cached(self, cache_level: CacheLevel = CacheLevel.MEMORY, ttl: Optional[float] = None):
        """Decorator for caching function results"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(func, args, kwargs)
                
                # Try to get from cache
                if cache_level == CacheLevel.MEMORY:
                    cached_value = self.memory_cache.get(cache_key)
                elif cache_level == CacheLevel.DISK:
                    cached_value = self.disk_cache.get(cache_key)
                else:
                    cached_value = None
                
                if cached_value is not None:
                    return cached_value
                
                # Compute result
                with timer(f"{func.__name__}_execution") as perf_timer:
                    result = func(*args, **kwargs)
                
                # Store in cache
                if cache_level == CacheLevel.MEMORY:
                    self.memory_cache.put(cache_key, result, ttl)
                elif cache_level == CacheLevel.DISK:
                    self.disk_cache.put(cache_key, result, ttl)
                
                # Record performance
                self.metrics.record_operation(func.__name__, perf_timer.duration * 1000 if perf_timer.duration else 0)
                
                return result
            
            return wrapper
        return decorator
    
    def parallel(self, max_workers: Optional[int] = None):
        """Decorator for parallel function execution"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if max_workers is None:
                    workers = self.executor._max_workers
                else:
                    workers = min(max_workers, self.executor._max_workers)
                
                # For simplicity, we'll use current implementation
                # In a full implementation, this would distribute work
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def optimize_model(self, model: nn.Module, example_input: Any) -> nn.Module:
        """Optimize model with quantization"""
        return self.quantizer.quantize_model(model, example_input)
    
    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function call"""
        try:
            # Create a deterministic key from function name and arguments
            key_data = {
                "func": func.__name__,
                "args": args,
                "kwargs": sorted(kwargs.items())
            }
            key_str = json.dumps(key_data, sort_keys=True, default=str)
            return hashlib.md5(key_str.encode()).hexdigest()
        except Exception:
            # Fallback to simple key
            return f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            "cache_stats": {
                "memory_cache": self.memory_cache.get_stats(),
                "disk_cache": {"cache_dir": self.disk_cache.cache_dir, "max_size_mb": self.disk_cache.max_size_mb}
            },
            "quantization_stats": self.quantizer.metrics.quantization_stats,
            "operation_stats": {
                name: self.metrics.get_stats(name) 
                for name in self.metrics.operation_times.keys()
            },
            "cache_hit_rate": self.metrics.cache_hit_rate_percent
        }
    
    def clear_caches(self) -> None:
        """Clear all caches"""
        self.memory_cache.clear()
        self.logger.info("All caches cleared")
    
    def shutdown(self) -> None:
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.logger.info("Performance optimizer shutdown complete")

# Decorators for easy use
def memory_cache(ttl: Optional[float] = None):
    """Memory cache decorator"""
    def decorator(func):
        optimizer = get_performance_optimizer()
        return optimizer.cached(CacheLevel.MEMORY, ttl)(func)
    return decorator

def disk_cache(ttl: Optional[float] = None):
    """Disk cache decorator"""
    def decorator(func):
        optimizer = get_performance_optimizer()
        return optimizer.cached(CacheLevel.DISK, ttl)(func)
    return decorator

def quantized(level: QuantizationLevel = QuantizationLevel.DYNAMIC):
    """Model quantization decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would apply quantization to returned model
            result = func(*args, **kwargs)
            
            if TORCH_AVAILABLE and hasattr(result, 'parameters'):
                optimizer = get_performance_optimizer()
                example_input = args[0] if args else kwargs.get('input', None)
                if example_input is not None:
                    result = optimizer.optimize_model(result, example_input)
            
            return result
        return wrapper
    return decorator

def performance_monitor(operation_name: Optional[str] = None):
    """Performance monitoring decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                
                optimizer = get_performance_optimizer()
                optimizer.metrics.record_operation(op_name, duration)
                
                return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                optimizer = get_performance_optimizer()
                optimizer.metrics.record_operation(f"{op_name}_error", duration)
                raise
        
        return wrapper
    return decorator

# Global performance optimizer instance
_global_optimizer: Optional[PerformanceOptimizer] = None

def get_performance_optimizer(config: Optional[Dict[str, Any]] = None) -> PerformanceOptimizer:
    """Get or create global performance optimizer"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer(config)
    return _global_optimizer

def initialize_performance(config: Optional[Dict[str, Any]] = None) -> PerformanceOptimizer:
    """Initialize global performance optimizer"""
    global _global_optimizer
    _global_optimizer = PerformanceOptimizer(config)
    return _global_optimizer