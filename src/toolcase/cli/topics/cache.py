CACHE = """
TOPIC: cache
============

Result caching with TTL support and advanced invalidation strategies.

PER-TOOL CONFIGURATION:
    class MyTool(BaseTool[Params]):
        cache_enabled = True   # Default
        cache_ttl = 300.0      # 5 minutes (default)
    
    class NoCacheTool(BaseTool[Params]):
        cache_enabled = False  # Disable caching
    
    class ShortCacheTool(BaseTool[Params]):
        cache_ttl = 60.0       # 1 minute

GLOBAL CACHE MANAGEMENT:
    from toolcase import get_cache, set_cache, reset_cache
    
    cache = get_cache()        # Get global cache
    reset_cache()              # Clear all cached data

MEMORY CACHE (Default):
    from toolcase.io.cache import MemoryCache
    
    cache = MemoryCache(default_ttl=300)
    cache.set("tool", params, "value", ttl=60)
    value = cache.get("tool", params)
    cache.delete("tool", params)
    cache.clear()              # Clear all or by tool name
    
    stats = cache.stats()      # {"hits": 10, "misses": 5, "size": 100}
    cache.ping()               # Health check (returns True)

TAGGED CACHE (Tag-Based Invalidation):
    from toolcase.io.cache import TaggedMemoryCache, TagIndex
    
    cache = TaggedMemoryCache()
    
    # Store with tags
    cache.set_tagged("user:123", "data", ttl=300, tags=["user", "profile"])
    cache.set_tagged("user:456", "data", ttl=300, tags=["user", "premium"])
    
    # Invalidate by tag (O(1) via reverse index)
    cache.invalidate_tag("user")      # Clears both entries
    cache.invalidate_tags(["profile", "premium"])

STALE-WHILE-REVALIDATE (SWR):
    from toolcase.io.cache import SWRCache, SWRConfig
    
    config = SWRConfig(
        stale_multiplier=2.0,     # Serve stale for 2x TTL
        max_stale_ms=60000,       # Max stale window 60s
        background_refresh=True,  # Refresh in background
    )
    
    cache = SWRCache(MemoryCache(), config)
    
    # Get with SWR semantics
    value, state = cache.get_with_state("key")
    if state == SWRState.FRESH:
        pass  # Normal cache hit
    elif state == SWRState.STALE:
        pass  # Stale hit, background refresh triggered
    elif state == SWRState.MISS:
        pass  # Cache miss

PATTERN MATCHING:
    from toolcase.io.cache import PatternMatcher
    
    matcher = PatternMatcher()
    cache.invalidate_pattern("user:*")     # fnmatch pattern
    cache.invalidate_regex(r"user:\\d+")    # Regex pattern

REDIS BACKEND (Distributed):
    from toolcase.io.cache import RedisCache, AsyncRedisCache
    
    # Sync (redis-py)
    cache = RedisCache(
        host="localhost", port=6379, db=0,
        prefix="toolcase:",
        default_ttl=300,
    )
    
    # Async (redis.asyncio)
    cache = AsyncRedisCache(
        host="localhost", port=6379, db=0,
        prefix="toolcase:",
    )
    
    # With connection URL
    cache = RedisCache(url="redis://localhost:6379/0")

MEMCACHED BACKEND (Distributed):
    from toolcase.io.cache import MemcachedCache, AsyncMemcachedCache
    
    # Sync (pymemcache)
    cache = MemcachedCache(host="localhost", port=11211)
    
    # Async (aiomcache)
    cache = AsyncMemcachedCache(host="localhost", port=11211)

CACHE-THROUGH UTILITIES:
    from toolcase.io.cache import cache_through, cache_through_async
    
    # Sync cache-through
    def expensive_compute(x):
        return x * 2
    
    result = cache_through(
        cache, "compute", {"x": 5},
        compute_fn=lambda: expensive_compute(5),
        ttl=60
    )
    
    # Async cache-through
    result = await cache_through_async(
        cache, "fetch", {"id": 123},
        compute_fn=lambda: fetch_data(123),
        ttl=300
    )

BINARY SERIALIZATION (Distributed Backends):
    # Redis/Memcached use msgpack (~40% smaller than JSON)
    from toolcase.io.cache import pack_value, unpack_value
    
    packed = pack_value({"key": "value", "num": 42})
    data = unpack_value(packed)

HEALTH & MONITORING:
    # All backends support health checks
    if cache.ping():
        print("Cache is healthy")
    
    # Get cache statistics
    stats = cache.stats()
    print(f"Hit rate: {stats['hits']/(stats['hits']+stats['misses']):.0%}")

RELATED TOPICS:
    toolcase help tool       Tool creation
    toolcase help settings   Global configuration
    toolcase help batch      Idempotent batch with cache
"""
