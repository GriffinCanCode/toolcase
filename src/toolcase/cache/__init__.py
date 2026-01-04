"""Tool result caching with TTL support.

Provides in-memory caching to prevent repeated API calls for identical queries.
Cache keys are generated from tool name + hashed parameters.
"""

from .cache import (
    DEFAULT_TTL,
    CacheBackend,
    MemoryCache,
    ToolCache,
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
    "DEFAULT_TTL",
]
