"""Tool result caching with TTL support.

Provides caching to prevent repeated API calls for identical queries.
Cache keys are generated from tool name + hashed parameters.

Backends:
    - MemoryCache: Thread-safe in-memory (default)
    - RedisCache: Sync redis-py backend (requires toolcase[redis])
    - AsyncRedisCache: Async redis.asyncio backend (requires toolcase[redis])
    - MemcachedCache: Sync pymemcache backend (requires toolcase[memcached])
    - AsyncMemcachedCache: Async aiomcache backend (requires aiomcache)

All backends implement ping() for health checks and stats() for monitoring.
"""

from .cache import (
    DEFAULT_TTL,
    AsyncToolCache,
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
    "AsyncToolCache",
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
    # Memcached (lazy import)
    "MemcachedCache",
    "AsyncMemcachedCache",
]


def __getattr__(name: str) -> object:
    """Lazy import Redis/Memcached backends to avoid import-time dependency."""
    if name in ("RedisCache", "AsyncRedisCache"):
        from .redis import AsyncRedisCache, RedisCache
        return RedisCache if name == "RedisCache" else AsyncRedisCache
    if name in ("MemcachedCache", "AsyncMemcachedCache"):
        from .memcached import AsyncMemcachedCache, MemcachedCache
        return MemcachedCache if name == "MemcachedCache" else AsyncMemcachedCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
