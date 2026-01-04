"""IO - Data input/output, caching, and streaming.

Contains: cache, progress, streaming.
"""

from __future__ import annotations

__all__ = [
    # Cache
    "ToolCache", "MemoryCache", "CacheBackend",
    "get_cache", "set_cache", "reset_cache",
    "cache_through", "cache_through_async", "DEFAULT_TTL",
    "RedisCache", "AsyncRedisCache",
    # Progress
    "ToolProgress", "ProgressKind", "ProgressCallback",
    "status", "step", "source_found", "complete", "error",
    # Streaming
    "StreamChunk", "StreamEvent", "StreamEventKind", "StreamState", "StreamResult",
    "chunk", "stream_start", "stream_complete", "stream_error",
    "StreamAdapter", "sse_adapter", "ws_adapter", "json_lines_adapter",
]


def __getattr__(name: str):
    """Lazy imports to avoid circular dependencies."""
    cache_attrs = {
        "ToolCache", "MemoryCache", "CacheBackend",
        "get_cache", "set_cache", "reset_cache",
        "cache_through", "cache_through_async", "DEFAULT_TTL",
    }
    if name in cache_attrs:
        from . import cache
        return getattr(cache, name)
    
    if name in ("RedisCache", "AsyncRedisCache"):
        from .cache.redis import AsyncRedisCache, RedisCache
        return RedisCache if name == "RedisCache" else AsyncRedisCache
    
    progress_attrs = {
        "ToolProgress", "ProgressKind", "ProgressCallback",
        "status", "step", "source_found", "complete", "error",
    }
    if name in progress_attrs:
        from . import progress
        return getattr(progress, name)
    
    streaming_attrs = {
        "StreamChunk", "StreamEvent", "StreamEventKind", "StreamState", "StreamResult",
        "chunk", "stream_start", "stream_complete", "stream_error",
        "StreamAdapter", "sse_adapter", "ws_adapter", "json_lines_adapter",
    }
    if name in streaming_attrs:
        from . import streaming
        return getattr(streaming, name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
