# IO

Data input/output, caching, and streaming. Handles data flow in and out of tools.

## Modules

| Module | Purpose |
|--------|---------|
| `cache/` | `MemoryCache`, `RedisCache` - result caching with TTL |
| `cache/strategy` | `TaggedMemoryCache`, `SWRCache` - advanced invalidation strategies |
| `progress/` | `ToolProgress`, `ProgressKind` - progress event streaming |
| `streaming/` | `StreamEvent`, adapters (SSE, WebSocket) - incremental result streaming |

## Quick Import

```python
from toolcase.io import get_cache, MemoryCache, RedisCache
from toolcase.io import ToolProgress, status, step, complete
from toolcase.io import StreamEvent, sse_adapter, ws_adapter
```

## Advanced Cache Invalidation

Tag-based invalidation enables bulk cache clearing by semantic group:

```python
from toolcase.io.cache import TaggedMemoryCache, SWRCache, SWRConfig

# Create cache with tag support
cache = TaggedMemoryCache(default_ttl=300)

# Store entries with tags
cache.set("search", {"q": "foo"}, "result", tags=frozenset({"user:123", "type:search"}))
cache.set("search", {"q": "bar"}, "result", tags=frozenset({"user:123", "type:search"}))
cache.set("profile", {"id": "123"}, "data", tags=frozenset({"user:123"}))

# Invalidate all entries for a user (removes all 3)
cache.invalidate_by_tag("user:123")

# Pattern-based invalidation (fnmatch globs)
cache.invalidate_by_pattern("search:*")  # Removes all search entries
```

## Stale-While-Revalidate (SWR)

Serve stale content while refreshing in the background for better latency:

```python
from toolcase.io.cache import TaggedMemoryCache, SWRCache, SWRConfig

async def fetch_fresh(tool: str, params: dict) -> str | None:
    return await api.call(tool, params)

# Create SWR-enabled cache
cache = TaggedMemoryCache(
    default_ttl=60,  # Fresh for 60s
    swr=SWRConfig(stale_multiplier=2.0)  # Stale but usable for 120s
)

# Wrap with SWR behavior
swr = SWRCache(cache, revalidator=fetch_fresh)

# Returns stale value immediately, refreshes in background
result = await swr.get_or_revalidate("search", {"q": "test"})
```
