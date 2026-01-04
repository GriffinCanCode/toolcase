CACHE = """
TOPIC: cache
============

Result caching with TTL support.

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

CUSTOM BACKEND:
    from toolcase import CacheBackend, set_cache
    
    class RedisCache(CacheBackend):
        def get(self, tool_name: str, params_hash: str):
            ...
        
        def set(self, tool_name: str, params_hash: str, 
                value: str, ttl: float):
            ...
        
        def delete(self, tool_name: str, params_hash: str):
            ...
        
        def clear(self, tool_name: str | None = None):
            ...
    
    set_cache(RedisCache())

REDIS BACKEND (Built-in):
    from toolcase.io.cache import RedisCache
    set_cache(RedisCache(host="localhost", port=6379))

RELATED TOPICS:
    toolcase help tool       Tool creation
    toolcase help settings   Global configuration
"""
