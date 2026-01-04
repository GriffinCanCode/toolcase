"""Tool result caching with TTL support.

Provides in-memory caching to prevent repeated API calls for identical queries.
Cache keys are generated from tool name + hashed parameters.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol, TypeVar, runtime_checkable

from ..errors import Err, ErrorCode, ErrorTrace, Ok, Result

if TYPE_CHECKING:
    from pydantic import BaseModel

DEFAULT_TTL: float = 300.0  # 5 minutes
T = TypeVar("T")


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


class ToolCache(ABC):
    """Abstract base for tool caches."""
    
    @abstractmethod
    def get(self, tool_name: str, params: BaseModel | dict[str, object]) -> str | None:
        """Get cached result if exists and not expired."""
        ...
    
    @abstractmethod
    def set(
        self,
        tool_name: str,
        params: BaseModel | dict[str, object],
        value: str,
        ttl: float | None = None,
    ) -> None:
        """Store result in cache."""
        ...
    
    @abstractmethod
    def invalidate(self, tool_name: str, params: BaseModel | dict[str, object]) -> bool:
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
    
    @staticmethod
    def make_key(tool_name: str, params: BaseModel | dict[str, object]) -> str:
        """Generate cache key from tool name and parameters."""
        if hasattr(params, "model_dump"):
            params_dict = params.model_dump(mode="json")  # type: ignore[union-attr]
        else:
            params_dict = params
        
        # Sort keys for consistent hashing
        params_json = json.dumps(params_dict, sort_keys=True, default=str)
        params_hash = hashlib.md5(params_json.encode(), usedforsecurity=False).hexdigest()[:12]
        
        return f"{tool_name}:{params_hash}"


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
    
    def get(self, tool_name: str, params: BaseModel | dict[str, object]) -> str | None:
        key = self.make_key(tool_name, params)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.expired:
                del self._cache[key]
                return None
            return entry.value
    
    def set(
        self,
        tool_name: str,
        params: BaseModel | dict[str, object],
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
    
    def invalidate(self, tool_name: str, params: BaseModel | dict[str, object]) -> bool:
        key = self.make_key(tool_name, params)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def invalidate_tool(self, tool_name: str) -> int:
        prefix = f"{tool_name}:"
        with self._lock:
            keys = [k for k in self._cache if k.startswith(prefix)]
            for key in keys:
                del self._cache[key]
            return len(keys)
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
    
    def _evict_unlocked(self) -> None:
        """Remove expired entries, then oldest if still over capacity. Caller must hold lock."""
        # First pass: remove expired
        expired = [k for k, v in self._cache.items() if v.expired]
        for key in expired:
            del self._cache[key]
        
        # If still over capacity, remove oldest quarter
        if len(self._cache) >= self._max_entries:
            sorted_keys = sorted(self._cache, key=lambda k: self._cache[k].expires_at)
            for key in sorted_keys[: self._max_entries // 4]:
                del self._cache[key]
    
    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)
    
    def stats(self) -> dict[str, object]:
        """Get cache statistics for monitoring."""
        with self._lock:
            expired = sum(1 for v in self._cache.values() if v.expired)
            return {
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
    params: BaseModel | dict[str, object],
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
    # Check cache
    cached = cache.get(tool_name, params)
    if cached is not None:
        return Ok(cached)  # type: ignore[return-value]
    
    # Execute operation with exception handling
    try:
        result = operation()
    except Exception as e:
        from ..errors import classify_exception
        trace = ErrorTrace(
            message=str(e),
            error_code=classify_exception(e).value,
            recoverable=True,
        ).with_operation(f"cache_through:{tool_name}")
        return Err(trace)
    
    # Cache successful result (only strings are cached)
    if isinstance(result, str):
        cache.set(tool_name, params, result, ttl)
    
    return Ok(result)


async def cache_through_async(
    cache: ToolCache,
    tool_name: str,
    params: BaseModel | dict[str, object],
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
    
    # Check cache
    cached = cache.get(tool_name, params)
    if cached is not None:
        return Ok(cached)  # type: ignore[return-value]
    
    # Execute operation with exception handling
    try:
        if asyncio.iscoroutinefunction(operation):
            result = await operation()  # type: ignore[misc]
        else:
            result = await asyncio.to_thread(operation)
    except Exception as e:
        from ..errors import classify_exception
        trace = ErrorTrace(
            message=str(e),
            error_code=classify_exception(e).value,
            recoverable=True,
        ).with_operation(f"cache_through:{tool_name}")
        return Err(trace)
    
    # Cache successful result (only strings are cached)
    if isinstance(result, str):
        cache.set(tool_name, params, result, ttl)
    
    return Ok(result)
