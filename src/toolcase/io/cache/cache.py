"""Tool result caching with TTL support.

Provides in-memory caching to prevent repeated API calls for identical queries.
Cache keys are generated from tool name + hashed parameters.

Storage:
    - MemoryCache: In-memory dict (no serialization overhead)
    - Redis/Memcached: msgpack binary by default (smaller, faster than JSON strings)
"""

from __future__ import annotations

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol, TypeVar, runtime_checkable

import msgpack
import orjson

from toolcase.foundation.errors import Err, ErrorTrace, JsonDict, Ok, Result

if TYPE_CHECKING:
    from pydantic import BaseModel

DEFAULT_TTL: float = 300.0  # 5 minutes
T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# Binary serialization for distributed caches (msgpack - faster than JSON)
# ═══════════════════════════════════════════════════════════════════════════════

def pack_value(value: str) -> bytes:
    """Pack cache value to msgpack bytes (~40% smaller than UTF-8, faster)."""
    return msgpack.packb(value, use_bin_type=True)


def unpack_value(data: bytes) -> str:
    """Unpack cache value from msgpack bytes."""
    return msgpack.unpackb(data, raw=False)


@dataclass(slots=True)
class CacheEntry:
    """A cached tool result with expiration tracking."""
    value: str
    expires_at: float
    
    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at


@runtime_checkable
class CacheBackend(Protocol):
    """Protocol for cache backends (enables custom implementations)."""
    
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl: float) -> None: ...
    def delete(self, key: str) -> bool: ...
    def clear(self) -> None: ...
    def ping(self) -> bool: ...


class ToolCache(ABC):
    """Abstract base for sync tool caches."""
    
    @abstractmethod
    def get(self, tool_name: str, params: BaseModel | JsonDict) -> str | None:
        """Get cached result if exists and not expired."""
        ...
    
    @abstractmethod
    def set(
        self,
        tool_name: str,
        params: BaseModel | JsonDict,
        value: str,
        ttl: float | None = None,
    ) -> None:
        """Store result in cache."""
        ...
    
    @abstractmethod
    def invalidate(self, tool_name: str, params: BaseModel | JsonDict) -> bool:
        """Remove specific entry from cache."""
        ...
    
    @abstractmethod
    def invalidate_tool(self, tool_name: str) -> int:
        """Remove all entries for a tool. Returns count removed."""
        ...
    
    @abstractmethod
    def clear(self) -> None:
        """Clear entire cache."""
        ...
    
    @abstractmethod
    def ping(self) -> bool:
        """Check if cache backend is available. Returns True if healthy."""
        ...
    
    @abstractmethod
    def stats(self) -> JsonDict:
        """Get cache statistics for monitoring."""
        ...
    
    @staticmethod
    def make_key(tool_name: str, params: BaseModel | JsonDict) -> str:
        """Generate cache key from tool name and parameters."""
        if hasattr(params, "model_dump"):
            params_dict = params.model_dump(mode="json")  # type: ignore[union-attr]
        else:
            params_dict = params
        
        # Sort keys for consistent hashing (orjson sorts by default)
        params_bytes = orjson.dumps(params_dict, option=orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS)
        params_hash = hashlib.md5(params_bytes, usedforsecurity=False).hexdigest()[:12]
        
        return f"{tool_name}:{params_hash}"


class AsyncToolCache(ABC):
    """Abstract base for async tool caches."""
    
    @abstractmethod
    async def aget(self, tool_name: str, params: BaseModel | JsonDict) -> str | None:
        """Get cached result if exists and not expired."""
        ...
    
    @abstractmethod
    async def aset(
        self,
        tool_name: str,
        params: BaseModel | JsonDict,
        value: str,
        ttl: float | None = None,
    ) -> None:
        """Store result in cache."""
        ...
    
    @abstractmethod
    async def ainvalidate(self, tool_name: str, params: BaseModel | JsonDict) -> bool:
        """Remove specific entry from cache."""
        ...
    
    @abstractmethod
    async def ainvalidate_tool(self, tool_name: str) -> int:
        """Remove all entries for a tool. Returns count removed."""
        ...
    
    @abstractmethod
    async def aclear(self) -> None:
        """Clear entire cache."""
        ...
    
    @abstractmethod
    async def aping(self) -> bool:
        """Check if cache backend is available. Returns True if healthy."""
        ...
    
    @abstractmethod
    async def astats(self) -> JsonDict:
        """Get cache statistics for monitoring."""
        ...
    
    @staticmethod
    def make_key(tool_name: str, params: BaseModel | JsonDict) -> str:
        """Generate cache key from tool name and parameters."""
        return ToolCache.make_key(tool_name, params)


class MemoryCache(ToolCache):
    """Thread-safe in-memory cache with TTL-based expiration.
    
    Uses RLock for synchronization, safe under concurrent access.
    Automatic eviction when capacity is reached.
    
    Args:
        default_ttl: Default TTL in seconds for entries
        max_entries: Maximum number of entries before eviction
    
    Example:
        >>> cache = MemoryCache(default_ttl=60)
        >>> cache.set("my_tool", {"q": "test"}, "result")
        >>> cache.get("my_tool", {"q": "test"})
        'result'
    """
    
    __slots__ = ("_cache", "_default_ttl", "_max_entries", "_lock")
    
    def __init__(self, default_ttl: float = DEFAULT_TTL, max_entries: int = 1000) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl
        self._max_entries = max_entries
        self._lock = threading.RLock()  # RLock allows reentrant calls (e.g. set -> _evict)
    
    def get(self, tool_name: str, params: BaseModel | JsonDict) -> str | None:
        key = self.make_key(tool_name, params)
        with self._lock:
            if not (entry := self._cache.get(key)):
                return None
            if entry.expired:
                del self._cache[key]
                return None
            return entry.value
    
    def set(
        self,
        tool_name: str,
        params: BaseModel | JsonDict,
        value: str,
        ttl: float | None = None,
    ) -> None:
        key = self.make_key(tool_name, params)
        with self._lock:
            if len(self._cache) >= self._max_entries:
                self._evict_unlocked()
            self._cache[key] = CacheEntry(
                value=value,
                expires_at=time.time() + (ttl or self._default_ttl),
            )
    
    def invalidate(self, tool_name: str, params: BaseModel | JsonDict) -> bool:
        with self._lock:
            return self._cache.pop(self.make_key(tool_name, params), None) is not None
    
    def invalidate_tool(self, tool_name: str) -> int:
        prefix = f"{tool_name}:"
        with self._lock:
            keys = [k for k in self._cache if k.startswith(prefix)]
            for k in keys:
                del self._cache[k]
            return len(keys)
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
    
    def _evict_unlocked(self) -> None:
        """Remove expired entries, then oldest if still over capacity. Caller must hold lock."""
        for k in [k for k, v in self._cache.items() if v.expired]:
            del self._cache[k]
        if len(self._cache) >= self._max_entries:
            for k in sorted(self._cache, key=lambda k: self._cache[k].expires_at)[:self._max_entries // 4]:
                del self._cache[k]
    
    def ping(self) -> bool:
        """Memory cache is always available."""
        return True
    
    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)
    
    def stats(self) -> JsonDict:
        """Get cache statistics for monitoring."""
        with self._lock:
            expired = sum(1 for v in self._cache.values() if v.expired)
            return {
                "backend": "memory",
                "total_entries": len(self._cache),
                "expired_entries": expired,
                "active_entries": len(self._cache) - expired,
                "default_ttl": self._default_ttl,
                "max_entries": self._max_entries,
            }


# Global cache instance
_cache: ToolCache | None = None


def get_cache() -> ToolCache:
    """Get the global tool cache instance (creates MemoryCache if unset)."""
    global _cache
    if _cache is None:
        _cache = MemoryCache()
    return _cache


def set_cache(cache: ToolCache) -> None:
    """Set a custom cache backend."""
    global _cache
    _cache = cache


def reset_cache() -> None:
    """Reset the global cache (useful for testing)."""
    global _cache
    if _cache is not None:
        _cache.clear()
    _cache = None


# ═══════════════════════════════════════════════════════════════════════════════
# Result-Based Cache Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def cache_through(
    cache: ToolCache,
    tool_name: str,
    params: BaseModel | JsonDict,
    operation: Callable[[], T],
    *,
    ttl: float | None = None,
) -> Result[T, ErrorTrace]:
    """Execute operation with cache-through pattern.
    
    Checks cache first, executes operation on miss, caches successful results.
    Returns Result for type-safe error handling.
    
    Args:
        cache: Cache instance to use
        tool_name: Tool name for cache key
        params: Parameters for cache key
        operation: Callable to execute on cache miss
        ttl: Optional TTL override
    
    Returns:
        Result[T, ErrorTrace] with cached or computed value
    
    Example:
        >>> result = cache_through(
        ...     cache, "search", params,
        ...     lambda: expensive_api_call(params),
        ... )
        >>> output = result.unwrap_or("fallback")
    """
    if (cached := cache.get(tool_name, params)) is not None:
        return Ok(cached)  # type: ignore[return-value]
    try:
        result = operation()
    except Exception as e:
        from toolcase.foundation.errors import classify_exception
        return Err(ErrorTrace(
            message=str(e),
            error_code=classify_exception(e).value,
            recoverable=True,
        ).with_operation(f"cache_through:{tool_name}"))
    if isinstance(result, str):
        cache.set(tool_name, params, result, ttl)
    return Ok(result)


async def cache_through_async(
    cache: ToolCache,
    tool_name: str,
    params: BaseModel | JsonDict,
    operation: Callable[[], T],
    *,
    ttl: float | None = None,
) -> Result[T, ErrorTrace]:
    """Async version of cache_through.
    
    Checks cache first, executes async operation on miss, caches successful results.
    
    Args:
        cache: Cache instance to use
        tool_name: Tool name for cache key
        params: Parameters for cache key
        operation: Async callable to execute on cache miss
        ttl: Optional TTL override
    
    Returns:
        Result[T, ErrorTrace] with cached or computed value
    """
    import asyncio
    if (cached := cache.get(tool_name, params)) is not None:
        return Ok(cached)  # type: ignore[return-value]
    try:
        result = await operation() if asyncio.iscoroutinefunction(operation) else await asyncio.to_thread(operation)  # type: ignore[misc]
    except Exception as e:
        from toolcase.foundation.errors import classify_exception
        return Err(ErrorTrace(
            message=str(e),
            error_code=classify_exception(e).value,
            recoverable=True,
        ).with_operation(f"cache_through:{tool_name}"))
    if isinstance(result, str):
        cache.set(tool_name, params, result, ttl)
    return Ok(result)
