"""Tool result caching with TTL support.

Provides in-memory caching to prevent repeated API calls for identical queries.
Cache keys are generated from tool name + hashed parameters.

Backends:
    - MemoryCache: Thread-safe in-memory (default)
    - RedisCache: Sync redis-py backend (requires toolcase[redis])
    - AsyncRedisCache: Async redis.asyncio backend (requires toolcase[redis])
"""

from .cache import (
    DEFAULT_TTL,
    CacheBackend,
    MemoryCache,
    ToolCache,
    cache_through,
    cache_through_async,
    get_cache,
    reset_cache,
    set_cache,
)

__all__ = [
    "ToolCache",
    "MemoryCache",
    "CacheBackend",
    "get_cache",
    "set_cache",
    "reset_cache",
    "cache_through",
    "cache_through_async",
    "DEFAULT_TTL",
    # Redis (lazy import)
    "RedisCache",
    "AsyncRedisCache",
]


def __getattr__(name: str) -> object:
    """Lazy import Redis backends to avoid import-time dependency."""
    if name in ("RedisCache", "AsyncRedisCache"):
        from .redis import AsyncRedisCache, RedisCache
        return RedisCache if name == "RedisCache" else AsyncRedisCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
