import asyncio
import json
import logging
import time
import hashlib
from typing import Optional, Any, Dict, List, Union
import aioredis
from aioredis import Redis
import pickle
from dataclasses import dataclass, asdict
from enum import Enum

from .config import settings

logger = logging.getLogger(__name__)

class CacheScope(Enum):
    SHORT_TERM = "short"    # 5 minutes
    MEDIUM_TERM = "medium"  # 1 hour
    LONG_TERM = "long"      # 24 hours

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: Any
    created_at: float
    ttl: int
    scope: CacheScope
    hit_count: int = 0
    last_accessed: float = 0
    
    def __post_init__(self):
        if self.last_accessed == 0:
            self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update last accessed time and hit count"""
        self.last_accessed = time.time()
        self.hit_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "data": self.data,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "scope": self.scope.value,
            "hit_count": self.hit_count,
            "last_accessed": self.last_accessed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary"""
        return cls(
            data=data["data"],
            created_at=data["created_at"],
            ttl=data["ttl"],
            scope=CacheScope(data["scope"]),
            hit_count=data.get("hit_count", 0),
            last_accessed=data.get("last_accessed", data["created_at"])
        )

class CacheManager:
    """Enhanced Redis-based caching manager with async operations and multi-layer caching"""
    
    def __init__(self, redis_url: str, local_cache_size: int = None):
        self.redis_url = redis_url
        self.redis: Optional[Redis] = None
        self.local_cache: Dict[str, CacheEntry] = {}
        self.local_cache_size = local_cache_size or settings.DATASET_CACHE_SIZE
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "redis_hits": 0,
            "local_hits": 0,
            "redis_errors": 0,
            "local_evictions": 0
        }
        
        # TTL mappings for different scopes
        self.scope_ttls = {
            CacheScope.SHORT_TERM: 300,   # 5 minutes
            CacheScope.MEDIUM_TERM: 3600, # 1 hour
            CacheScope.LONG_TERM: 86400   # 24 hours
        }
        
    async def connect(self):
        """Connect to Redis with fallback to local cache"""
        try:
            self.redis = await aioredis.from_url(self.redis_url, decode_responses=False)
            await self.redis.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Redis connection failed, using local cache only: {e}")
            self.redis = None
            
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
            logger.info("Disconnected from Redis cache")
            
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with multi-layer strategy"""
        # Try local cache first (fastest)
        entry = self.local_cache.get(key)
        if entry and not entry.is_expired():
            entry.touch()
            self.stats["hits"] += 1
            self.stats["local_hits"] += 1
            return entry.data
        elif entry:
            # Remove expired local entry
            del self.local_cache[key]
        
        # Try Redis cache
        if self.redis:
            try:
                serialized = await self.redis.get(key)
                if serialized:
                    entry_data = pickle.loads(serialized)
                    
                    # Handle both old format (direct data) and new format (CacheEntry)
                    if isinstance(entry_data, dict) and "data" in entry_data:
                        entry = CacheEntry.from_dict(entry_data)
                    else:
                        # Legacy format - wrap it
                        entry = CacheEntry(
                            data=entry_data,
                            created_at=time.time(),
                            ttl=settings.CACHE_TTL,
                            scope=CacheScope.MEDIUM_TERM
                        )
                    
                    # Cache in local memory if not expired
                    if not entry.is_expired():
                        await self._add_to_local_cache(key, entry)
                        self.stats["hits"] += 1
                        self.stats["redis_hits"] += 1
                        return entry.data
                    
            except Exception as e:
                logger.debug(f"Redis get failed for key {key}: {e}")
                self.stats["redis_errors"] += 1
        
        # Cache miss
        self.stats["misses"] += 1
        return None
        
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                  scope: CacheScope = CacheScope.MEDIUM_TERM) -> bool:
        """Set value in cache with enhanced metadata"""
        
        # Determine TTL
        if ttl is None:
            ttl = self.scope_ttls.get(scope, settings.CACHE_TTL)
        
        # Create cache entry
        entry = CacheEntry(
            data=value,
            created_at=time.time(),
            ttl=ttl,
            scope=scope
        )
        
        # Set in Redis
        redis_success = False
        if self.redis:
            try:
                serialized = pickle.dumps(entry.to_dict())
                await self.redis.setex(key, ttl, serialized)
                redis_success = True
            except Exception as e:
                logger.debug(f"Redis set failed for key {key}: {e}")
                self.stats["redis_errors"] += 1
        
        # Set in local cache
        await self._add_to_local_cache(key, entry)
        
        return redis_success
    
    async def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None,
                      scope: CacheScope = CacheScope.MEDIUM_TERM) -> int:
        """Set multiple values in cache efficiently"""
        success_count = 0
        
        if self.redis and items:
            try:
                # Prepare pipeline for Redis
                pipe = self.redis.pipeline()
                
                for key, value in items.items():
                    entry = CacheEntry(
                        data=value,
                        created_at=time.time(),
                        ttl=ttl or self.scope_ttls.get(scope, settings.CACHE_TTL),
                        scope=scope
                    )
                    serialized = pickle.dumps(entry.to_dict())
                    pipe.setex(key, entry.ttl, serialized)
                
                await pipe.execute()
                success_count = len(items)
                
            except Exception as e:
                logger.debug(f"Redis mset failed: {e}")
                self.stats["redis_errors"] += 1
        
        # Add to local cache
        for key, value in items.items():
            entry = CacheEntry(
                data=value,
                created_at=time.time(),
                ttl=ttl or self.scope_ttls.get(scope, settings.CACHE_TTL),
                scope=scope
            )
            await self._add_to_local_cache(key, entry)
        
        return success_count
        
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        redis_success = False
        
        if self.redis:
            try:
                await self.redis.delete(key)
                redis_success = True
            except Exception as e:
                logger.debug(f"Redis delete failed for key {key}: {e}")
                
        self.local_cache.pop(key, None)
        return redis_success
        
    async def clear(self) -> bool:
        """Clear all cache entries"""
        redis_success = False
        
        if self.redis:
            try:
                await self.redis.flushdb()
                redis_success = True
            except Exception as e:
                logger.debug(f"Redis clear failed: {e}")
                
        self.local_cache.clear()
        return redis_success
        
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if self.redis:
            try:
                return bool(await self.redis.exists(key))
            except Exception as e:
                logger.debug(f"Redis exists check failed for key {key}: {e}")
                
        return key in self.local_cache
        
    async def _add_to_local_cache(self, key: str, entry: CacheEntry):
        """Add entry to local cache with eviction policy"""
        # Evict expired entries
        await self._cleanup_expired_entries()
        
        # Evict entries if cache is full
        while len(self.local_cache) >= self.local_cache_size:
            await self._evict_lru_entry()
        
        self.local_cache[key] = entry
    
    async def _cleanup_expired_entries(self):
        """Remove expired entries from local cache"""
        expired_keys = []
        for key, entry in self.local_cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.local_cache[key]
    
    async def _evict_lru_entry(self):
        """Evict least recently used entry from local cache"""
        if not self.local_cache:
            return
        
        lru_key = min(self.local_cache.keys(), 
                     key=lambda k: self.local_cache[k].last_accessed)
        del self.local_cache[lru_key]
        self.stats["local_evictions"] += 1
    
    async def get_pattern(self, pattern: str) -> Dict[str, Any]:
        """Get all keys matching a pattern"""
        results = {}
        
        # Search local cache
        import fnmatch
        for key in self.local_cache:
            if fnmatch.fnmatch(key, pattern):
                entry = self.local_cache[key]
                if not entry.is_expired():
                    results[key] = entry.data
        
        # Search Redis if available
        if self.redis:
            try:
                redis_keys = await self.redis.keys(pattern)
                for redis_key in redis_keys:
                    key_str = redis_key.decode('utf-8')
                    if key_str not in results:
                        value = await self.redis.get(redis_key)
                        if value:
                            entry_data = pickle.loads(value)
                            if isinstance(entry_data, dict) and "data" in entry_data:
                                entry = CacheEntry.from_dict(entry_data)
                                if not entry.is_expired():
                                    results[key_str] = entry.data
            except Exception as e:
                logger.debug(f"Redis pattern search failed: {e}")
                self.stats["redis_errors"] += 1
        
        return results
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern"""
        deleted_count = 0
        
        # Delete from local cache
        import fnmatch
        local_keys_to_delete = []
        for key in self.local_cache:
            if fnmatch.fnmatch(key, pattern):
                local_keys_to_delete.append(key)
        
        for key in local_keys_to_delete:
            del self.local_cache[key]
            deleted_count += 1
        
        # Delete from Redis
        if self.redis:
            try:
                redis_keys = await self.redis.keys(pattern)
                if redis_keys:
                    await self.redis.delete(*redis_keys)
                    deleted_count += len(redis_keys)
            except Exception as e:
                logger.debug(f"Redis pattern delete failed: {e}")
                self.stats["redis_errors"] += 1
        
        return deleted_count
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        stats = {
            "local_cache": {
                "size": len(self.local_cache),
                "max_size": self.local_cache_size,
                "hit_count": self.stats["local_hits"],
                "evictions": self.stats["local_evictions"]
            },
            "redis": {
                "connected": self.redis is not None,
                "hit_count": self.stats["redis_hits"],
                "errors": self.stats["redis_errors"]
            },
            "performance": {
                "total_requests": total_requests,
                "hit_rate": hit_rate,
                "total_hits": self.stats["hits"],
                "total_misses": self.stats["misses"]
            }
        }
        
        # Add Redis server stats if connected
        if self.redis:
            try:
                info = await self.redis.info()
                stats["redis"]["memory"] = info.get("used_memory_human", "unknown")
                stats["redis"]["keys"] = info.get("db0", {}).get("keys", 0)
                stats["redis"]["connections"] = info.get("connected_clients", 0)
            except Exception as e:
                logger.debug(f"Failed to get Redis stats: {e}")
        
        return stats
    
    async def reset_stats(self):
        """Reset cache statistics"""
        for key in self.stats:
            self.stats[key] = 0
    
    def generate_key(self, *parts, prefix: str = "") -> str:
        """Generate consistent cache key from parts"""
        key_data = ":".join(str(part) for part in parts)
        if prefix:
            key_data = f"{prefix}:{key_data}"
        
        # Hash long keys to avoid issues with Redis key length limits
        if len(key_data) > 200:
            key_hash = hashlib.md5(key_data.encode()).hexdigest()
            return f"{prefix[:50]}:{key_hash}" if prefix else key_hash
        
        return key_data