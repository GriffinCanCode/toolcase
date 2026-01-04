"""Tests for cache backends."""

import pytest

from toolcase.io.cache import MemoryCache, ToolCache, get_cache, reset_cache, set_cache


class MockRedisClient:
    """In-memory mock of sync Redis client for testing."""
    
    def __init__(self) -> None:
        self._data: dict[str, tuple[str, float]] = {}  # key -> (value, expire_at)
        self._time = 0.0
    
    def get(self, key: str) -> bytes | None:
        if key not in self._data:
            return None
        return self._data[key][0].encode()
    
    def setex(self, name: str, time: int, value: str) -> bool:
        self._data[name] = (value, self._time + time)
        return True
    
    def delete(self, *names: str) -> int:
        count = sum(1 for n in names if n in self._data)
        for name in names:
            self._data.pop(name, None)
        return count
    
    def scan_iter(self, match: str) -> list[str]:
        import fnmatch
        return [k for k in self._data if fnmatch.fnmatch(k, match)]


class MockAsyncRedisClient:
    """In-memory mock of async Redis client for testing."""
    
    def __init__(self) -> None:
        self._sync = MockRedisClient()
    
    async def get(self, key: str) -> bytes | None:
        return self._sync.get(key)
    
    async def setex(self, name: str, time: int, value: str) -> bool:
        return self._sync.setex(name, time, value)
    
    async def delete(self, *names: str) -> int:
        return self._sync.delete(*names)
    
    async def scan_iter(self, match: str) -> object:
        for k in self._sync.scan_iter(match):
            yield k


@pytest.fixture(autouse=True)
def clean_cache() -> object:
    """Reset global cache before each test."""
    reset_cache()
    yield
    reset_cache()


def test_memory_cache_basic() -> None:
    """Test basic get/set operations."""
    cache = MemoryCache()
    
    cache.set("tool", {"q": "test"}, "result")
    assert cache.get("tool", {"q": "test"}) == "result"
    assert cache.get("tool", {"q": "other"}) is None


def test_memory_cache_ttl() -> None:
    """Test TTL expiration."""
    import time
    cache = MemoryCache(default_ttl=0.01)  # 10ms
    
    cache.set("tool", {"q": "test"}, "result")
    assert cache.get("tool", {"q": "test"}) == "result"
    
    time.sleep(0.02)
    assert cache.get("tool", {"q": "test"}) is None


def test_memory_cache_invalidate() -> None:
    """Test single entry invalidation."""
    cache = MemoryCache()
    
    cache.set("tool", {"q": "a"}, "result_a")
    cache.set("tool", {"q": "b"}, "result_b")
    
    assert cache.invalidate("tool", {"q": "a"})
    assert cache.get("tool", {"q": "a"}) is None
    assert cache.get("tool", {"q": "b"}) == "result_b"


def test_memory_cache_invalidate_tool() -> None:
    """Test invalidating all entries for a tool."""
    cache = MemoryCache()
    
    cache.set("tool1", {"q": "a"}, "result_1a")
    cache.set("tool1", {"q": "b"}, "result_1b")
    cache.set("tool2", {"q": "a"}, "result_2a")
    
    assert cache.invalidate_tool("tool1") == 2
    assert cache.get("tool1", {"q": "a"}) is None
    assert cache.get("tool1", {"q": "b"}) is None
    assert cache.get("tool2", {"q": "a"}) == "result_2a"


def test_memory_cache_clear() -> None:
    """Test clearing entire cache."""
    cache = MemoryCache()
    
    cache.set("tool1", {"q": "a"}, "result")
    cache.set("tool2", {"q": "b"}, "result")
    cache.clear()
    
    assert cache.size == 0


def test_memory_cache_eviction() -> None:
    """Test eviction when max_entries reached."""
    cache = MemoryCache(max_entries=10)
    
    for i in range(15):
        cache.set("tool", {"i": i}, f"result_{i}")
    
    # Should have evicted some entries
    assert cache.size < 15


def test_make_key_consistency() -> None:
    """Test that cache keys are consistent."""
    key1 = ToolCache.make_key("tool", {"a": 1, "b": 2})
    key2 = ToolCache.make_key("tool", {"b": 2, "a": 1})  # Different order
    assert key1 == key2


def test_global_cache_singleton() -> None:
    """Test global cache instance management."""
    cache1 = get_cache()
    cache2 = get_cache()
    assert cache1 is cache2
    
    custom = MemoryCache(default_ttl=1.0)
    set_cache(custom)
    assert get_cache() is custom


def test_redis_cache_sync() -> None:
    """Test sync RedisCache with mock client."""
    from toolcase.io.cache import RedisCache
    
    client = MockRedisClient()
    cache = RedisCache(client, prefix="test:")
    
    cache.set("tool", {"q": "test"}, "result")
    assert cache.get("tool", {"q": "test"}) == "result"
    assert cache.get("tool", {"q": "other"}) is None


def test_redis_cache_invalidate() -> None:
    """Test RedisCache invalidation."""
    from toolcase.io.cache import RedisCache
    
    client = MockRedisClient()
    cache = RedisCache(client, prefix="test:")
    
    cache.set("tool", {"q": "a"}, "result_a")
    cache.set("tool", {"q": "b"}, "result_b")
    
    assert cache.invalidate("tool", {"q": "a"})
    assert cache.get("tool", {"q": "a"}) is None
    assert cache.get("tool", {"q": "b"}) == "result_b"


def test_redis_cache_invalidate_tool() -> None:
    """Test RedisCache tool invalidation."""
    from toolcase.io.cache import RedisCache
    
    client = MockRedisClient()
    cache = RedisCache(client, prefix="test:")
    
    cache.set("tool1", {"q": "a"}, "result")
    cache.set("tool1", {"q": "b"}, "result")
    cache.set("tool2", {"q": "a"}, "result")
    
    assert cache.invalidate_tool("tool1") == 2
    assert cache.get("tool1", {"q": "a"}) is None
    assert cache.get("tool2", {"q": "a"}) == "result"


def test_redis_cache_clear() -> None:
    """Test RedisCache clear."""
    from toolcase.io.cache import RedisCache
    
    client = MockRedisClient()
    cache = RedisCache(client, prefix="test:")
    
    cache.set("tool1", {"q": "a"}, "result")
    cache.set("tool2", {"q": "b"}, "result")
    cache.clear()
    
    assert cache.get("tool1", {"q": "a"}) is None
    assert cache.get("tool2", {"q": "b"}) is None


@pytest.mark.asyncio
async def test_async_redis_cache() -> None:
    """Test async RedisCache with mock client."""
    from toolcase.io.cache import AsyncRedisCache
    
    client = MockAsyncRedisClient()
    cache = AsyncRedisCache(client, prefix="test:")
    
    await cache.aset("tool", {"q": "test"}, "result")
    assert await cache.aget("tool", {"q": "test"}) == "result"
    assert await cache.aget("tool", {"q": "other"}) is None


@pytest.mark.asyncio
async def test_async_redis_invalidate() -> None:
    """Test async RedisCache invalidation."""
    from toolcase.io.cache import AsyncRedisCache
    
    client = MockAsyncRedisClient()
    cache = AsyncRedisCache(client, prefix="test:")
    
    await cache.aset("tool", {"q": "a"}, "result_a")
    await cache.aset("tool", {"q": "b"}, "result_b")
    
    assert await cache.ainvalidate("tool", {"q": "a"})
    assert await cache.aget("tool", {"q": "a"}) is None
    assert await cache.aget("tool", {"q": "b"}) == "result_b"


@pytest.mark.asyncio
async def test_async_redis_invalidate_tool() -> None:
    """Test async RedisCache tool invalidation."""
    from toolcase.io.cache import AsyncRedisCache
    
    client = MockAsyncRedisClient()
    cache = AsyncRedisCache(client, prefix="test:")
    
    await cache.aset("tool1", {"q": "a"}, "result")
    await cache.aset("tool1", {"q": "b"}, "result")
    await cache.aset("tool2", {"q": "a"}, "result")
    
    assert await cache.ainvalidate_tool("tool1") == 2
    assert await cache.aget("tool1", {"q": "a"}) is None
    assert await cache.aget("tool2", {"q": "a"}) == "result"


@pytest.mark.asyncio
async def test_async_redis_clear() -> None:
    """Test async RedisCache clear."""
    from toolcase.io.cache import AsyncRedisCache
    
    client = MockAsyncRedisClient()
    cache = AsyncRedisCache(client, prefix="test:")
    
    await cache.aset("tool1", {"q": "a"}, "result")
    await cache.aset("tool2", {"q": "b"}, "result")
    await cache.aclear()
    
    assert await cache.aget("tool1", {"q": "a"}) is None
    assert await cache.aget("tool2", {"q": "b"}) is None


def test_async_redis_sync_methods_raise() -> None:
    """Test that sync methods on async cache raise NotImplementedError."""
    from toolcase.io.cache import AsyncRedisCache
    
    client = MockAsyncRedisClient()
    cache = AsyncRedisCache(client)
    
    with pytest.raises(NotImplementedError):
        cache.get("tool", {"q": "test"})
    
    with pytest.raises(NotImplementedError):
        cache.set("tool", {"q": "test"}, "value")
    
    with pytest.raises(NotImplementedError):
        cache.invalidate("tool", {"q": "test"})
    
    with pytest.raises(NotImplementedError):
        cache.invalidate_tool("tool")
    
    with pytest.raises(NotImplementedError):
        cache.clear()
