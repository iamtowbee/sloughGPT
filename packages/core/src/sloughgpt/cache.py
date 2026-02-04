"""Caching layer with Redis support for SloughGPT."""

from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
import json
import pickle
import hashlib
import asyncio
from datetime import datetime, timedelta
import logging

try:
    import aioredis
    from aioredis import Redis, ConnectionPool
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import functools
    HAS_FUNCTOOLS = True
except ImportError:
    HAS_FUNCTOOLS = False


@dataclass
class CacheConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    default_ttl: int = 3600  # 1 hour
    key_prefix: str = "sloughgpt:"


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    memory_usage: Optional[Dict[str, Any]] = None


class CacheManager:
    """Advanced caching manager with Redis support."""
    
    def __init__(self, config: CacheConfig):
        if not HAS_REDIS:
            raise ImportError("aioredis is required for caching")
        
        self.config = config
        self.redis = None
        self.connection_pool = None
        self.stats = CacheStats()
        self.initialized = False
        
    async def initialize(self) -> bool:
        """Initialize Redis connection."""
        try:
            # Create connection pool
            self.connection_pool = ConnectionPool.from_url(
                f"redis://{self.config.host}:{self.config.port}/{self.config.db}",
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                password=self.config.password
            )
            
            # Create Redis client
            self.redis = Redis(connection_pool=self.connection_pool)
            
            # Test connection
            await self.redis.ping()
            
            self.initialized = True
            logging.info(f"Cache connected to Redis at {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize cache: {e}")
            return False
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
        if self.connection_pool:
            await self.connection_pool.disconnect()
        self.initialized = False
        logging.info("Cache connection closed")
    
    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.config.key_prefix}{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            if isinstance(value, (str, int, float, bool)):
                return json.dumps(value).encode('utf-8')
            else:
                return pickle.dumps(value)
        except Exception as e:
            logging.error(f"Failed to serialize cache value: {e}")
            raise
    
    def _deserialize_value(self, data: bytes, use_pickle: bool = False) -> Any:
        """Deserialize value from storage."""
        try:
            if use_pickle:
                return pickle.loads(data)
            else:
                return json.loads(data.decode('utf-8'))
        except Exception as e:
            logging.error(f"Failed to deserialize cache value: {e}")
            return None
    
    async def get(self, key: str, default: Any = None, use_pickle: bool = False) -> Any:
        """Get value from cache."""
        if not self.initialized:
            return default
        
        try:
            cache_key = self._make_key(key)
            data = await self.redis.get(cache_key)
            
            if data is None:
                self.stats.misses += 1
                return default
            else:
                self.stats.hits += 1
                return self._deserialize_value(data, use_pickle)
                
        except Exception as e:
            self.stats.errors += 1
            logging.error(f"Cache get error for key {key}: {e}")
            return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None,
                  use_pickle: bool = False) -> bool:
        """Set value in cache."""
        if not self.initialized:
            return False
        
        try:
            cache_key = self._make_key(key)
            serialized_value = self._serialize_value(value)
            expire_time = ttl or self.config.default_ttl
            
            result = await self.redis.setex(cache_key, expire_time, serialized_value)
            
            if result:
                self.stats.sets += 1
                return True
            else:
                return False
                
        except Exception as e:
            self.stats.errors += 1
            logging.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.initialized:
            return False
        
        try:
            cache_key = self._make_key(key)
            result = await self.redis.delete(cache_key)
            
            if result:
                self.stats.deletes += 1
                return True
            else:
                return False
                
        except Exception as e:
            self.stats.errors += 1
            logging.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.initialized:
            return False
        
        try:
            cache_key = self._make_key(key)
            result = await self.redis.exists(cache_key)
            return result == 1
                
        except Exception as e:
            self.stats.errors += 1
            logging.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key."""
        if not self.initialized:
            return False
        
        try:
            cache_key = self._make_key(key)
            result = await self.redis.expire(cache_key, ttl)
            return result
                
        except Exception as e:
            self.stats.errors += 1
            logging.error(f"Cache expire error for key {key}: {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """Get time to live for key."""
        if not self.initialized:
            return -1
        
        try:
            cache_key = self._make_key(key)
            result = await self.redis.ttl(cache_key)
            return result
                
        except Exception as e:
            self.stats.errors += 1
            logging.error(f"Cache TTL error for key {key}: {e}")
            return -1
    
    async def get_many(self, keys: List[str], use_pickle: bool = False) -> Dict[str, Any]:
        """Get multiple values from cache."""
        if not self.initialized or not keys:
            return {}
        
        try:
            cache_keys = [self._make_key(key) for key in keys]
            values = await self.redis.mget(cache_keys)
            
            result = {}
            for i, key in enumerate(keys):
                if values[i] is not None:
                    self.stats.hits += 1
                    result[key] = self._deserialize_value(values[i], use_pickle)
                else:
                    self.stats.misses += 1
                    result[key] = None
            
            return result
                
        except Exception as e:
            self.stats.errors += 1
            logging.error(f"Cache get_many error: {e}")
            return {}
    
    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None,
                     use_pickle: bool = False) -> bool:
        """Set multiple values in cache."""
        if not self.initialized or not mapping:
            return False
        
        try:
            expire_time = ttl or self.config.default_ttl
            pipe = self.redis.pipeline()
            
            for key, value in mapping.items():
                cache_key = self._make_key(key)
                serialized_value = self._serialize_value(value)
                pipe.setex(cache_key, expire_time, serialized_value)
            
            results = await pipe.execute()
            
            # Check if all operations succeeded
            if all(results):
                self.stats.sets += len(mapping)
                return True
            else:
                return False
                
        except Exception as e:
            self.stats.errors += 1
            logging.error(f"Cache set_many error: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment numeric value."""
        if not self.initialized:
            return None
        
        try:
            cache_key = self._make_key(key)
            result = await self.redis.incrby(cache_key, amount)
            self.stats.sets += 1
            return result
                
        except Exception as e:
            self.stats.errors += 1
            logging.error(f"Cache increment error for key {key}: {e}")
            return None
    
    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        if not self.initialized:
            return self.stats
        
        try:
            # Get Redis info
            info = await self.redis.info()
            
            self.stats.memory_usage = {
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "used_memory_peak": info.get("used_memory_peak", 0),
                "used_memory_peak_human": info.get("used_memory_peak_human", "0B"),
                "maxmemory": info.get("maxmemory", 0),
                "maxmemory_human": info.get("maxmemory_human", "0B")
            }
            
            return self.stats
                
        except Exception as e:
            logging.error(f"Failed to get cache stats: {e}")
            return self.stats
    
    async def clear_all(self) -> bool:
        """Clear all cache data."""
        if not self.initialized:
            return False
        
        try:
            # Only clear keys with our prefix
            pattern = f"{self.config.key_prefix}*"
            keys = await self.redis.keys(pattern)
            
            if keys:
                result = await self.redis.delete(*keys)
                self.stats.deletes += result
                return result > 0
            
            return True
                
        except Exception as e:
            self.stats.errors += 1
            logging.error(f"Cache clear_all error: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check cache health."""
        if not self.initialized:
            return {
                "status": "not_initialized",
                "error": "Cache not initialized"
            }
        
        try:
            # Test basic operations
            test_key = self._make_key("health_check")
            await self.redis.setex(test_key, 10, "test")
            value = await self.redis.get(test_key)
            await self.redis.delete(test_key)
            
            if value != b"test":
                raise Exception("Health check failed")
            
            stats = await self.get_stats()
            
            return {
                "status": "healthy",
                "redis_info": await self.redis.info(),
                "cache_stats": {
                    "hits": stats.hits,
                    "misses": stats.misses,
                    "sets": stats.sets,
                    "deletes": stats.deletes,
                    "errors": stats.errors,
                    "hit_rate": stats.hits / (stats.hits + stats.misses) if (stats.hits + stats.misses) > 0 else 0,
                    "memory_usage": stats.memory_usage
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class CacheDecorator:
    """Decorator for caching function results."""
    
    def __init__(self, cache_manager: CacheManager, ttl: int = 3600,
                 key_func: Optional[Callable] = None, use_pickle: bool = False):
        self.cache_manager = cache_manager
        self.ttl = ttl
        self.key_func = key_func or self._default_key_func
        self.use_pickle = use_pickle
    
    def _default_key_func(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate default cache key."""
        # Create key from function name and arguments
        key_parts = [func.__name__]
        
        # Add args
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(hashlib.md5(str(arg).encode()).hexdigest())
        
        # Add kwargs
        for k, v in sorted(kwargs.items()):
            key_parts.append(k)
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(str(v))
            else:
                key_parts.append(hashlib.md5(str(v).encode()).hexdigest())
        
        return ":".join(key_parts)
    
    def __call__(self, func: Callable):
        async def wrapper(*args, **kwargs):
            if not self.cache_manager.initialized:
                # Fall back to direct function call if cache not available
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            # Generate cache key
            cache_key = self.key_func(func, args, kwargs)
            
            # Try to get from cache
            cached_result = await self.cache_manager.get(cache_key, use_pickle=self.use_pickle)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await self.cache_manager.set(cache_key, result, self.ttl, self.use_pickle)
            return result
        
        return wrapper


if HAS_FUNCTOOLS:
    class cache_result:
        """Decorator for caching function results (alternative syntax)."""
        
        def __init__(self, ttl: int = 3600, key_func: Optional[Callable] = None,
                     use_pickle: bool = False):
            self.ttl = ttl
            self.key_func = key_func
            self.use_pickle = use_pickle
        
        def __call__(self, func: Callable):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Get cache manager from global scope or create new one
                from . import cache_manager
                
                # Generate cache key
                if self.key_func:
                    cache_key = self.key_func(func, args, kwargs)
                else:
                    cache_key = f"{func.__name__}:{hashlib.md5(str(args + tuple(sorted(kwargs.items()))).encode()).hexdigest()}"
                
                # Try to get from cache
                cached_result = await cache_manager.get(cache_key, use_pickle=self.use_pickle)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                await cache_manager.set(cache_key, result, self.ttl, self.use_pickle)
                return result
            
            return wrapper


# Cache manager factory
def create_cache_manager(config: Optional[CacheConfig] = None) -> CacheManager:
    """Create cache manager with default config if not provided."""
    if config is None:
        config = CacheConfig()
    
    return CacheManager(config)


# Global cache manager instance
cache_manager = create_cache_manager()