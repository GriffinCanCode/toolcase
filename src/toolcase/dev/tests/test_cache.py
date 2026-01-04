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
    
    def ping(self) -> bool:
        return True
    
    def info(self, section: str = "") -> dict[str, object]:
        return {"used_memory_human": "1M"}


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
    
    def scan_iter(self, match: str) -> object:
        async def _gen() -> object:
            for k in self._sync.scan_iter(match):
                yield k
        return _gen()
    
    async def ping(self) -> bool:
        return True
    
    async def info(self, section: str = "") -> dict[str, object]:
        return {"used_memory_human": "1M"}


class MockMemcachedClient:
    """In-memory mock of sync Memcached client for testing."""
    
    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
    
    def get(self, key: str) -> bytes | None:
        return self._data.get(key)
    
    def set(self, key: str, value: bytes, expire: int = 0) -> bool:
        self._data[key] = value
        return True
    
    def delete(self, key: str) -> bool:
        return self._data.pop(key, None) is not None
    
    def stats(self) -> dict[bytes, dict[bytes, bytes]]:
        return {b"localhost:11211": {b"curr_items": b"1", b"bytes": b"100"}}


class MockAsyncMemcachedClient:
    """In-memory mock of async Memcached client for testing."""
    
    def __init__(self) -> None:
        self._sync = MockMemcachedClient()
    
    async def get(self, key: bytes) -> bytes | None:
        return self._sync.get(key.decode())
    
    async def set(self, key: bytes, value: bytes, exptime: int = 0) -> bool:
        return self._sync.set(key.decode(), value, exptime)
    
    async def delete(self, key: bytes) -> bool:
        return self._sync.delete(key.decode())
    
    async def stats(self) -> dict[bytes, bytes]:
        return {b"curr_items": b"1", b"bytes": b"100"}


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


def test_memory_cache_ping() -> None:
    """Test MemoryCache ping always returns True."""
    cache = MemoryCache()
    assert cache.ping() is True


def test_memory_cache_stats() -> None:
    """Test MemoryCache stats structure."""
    cache = MemoryCache(default_ttl=60, max_entries=100)
    cache.set("tool", {"q": "a"}, "result_a")
    cache.set("tool", {"q": "b"}, "result_b")
    
    stats = cache.stats()
    assert stats["backend"] == "memory"
    assert stats["total_entries"] == 2
    assert stats["active_entries"] == 2
    assert stats["expired_entries"] == 0
    assert stats["default_ttl"] == 60
    assert stats["max_entries"] == 100


def test_redis_cache_ping() -> None:
    """Test RedisCache ping."""
    from toolcase.io.cache import RedisCache
    
    client = MockRedisClient()
    cache = RedisCache(client, prefix="test:")
    assert cache.ping() is True


def test_redis_cache_stats() -> None:
    """Test RedisCache stats structure."""
    from toolcase.io.cache import RedisCache
    
    client = MockRedisClient()
    cache = RedisCache(client, prefix="test:")
    cache.set("tool", {"q": "test"}, "result")
    
    stats = cache.stats()
    assert stats["backend"] == "redis"
    assert stats["prefix"] == "test:"
    assert stats["connected"] is True
    assert "total_entries" in stats


@pytest.mark.asyncio
async def test_async_redis_cache_ping() -> None:
    """Test AsyncRedisCache ping."""
    from toolcase.io.cache import AsyncRedisCache
    
    client = MockAsyncRedisClient()
    cache = AsyncRedisCache(client, prefix="test:")
    assert await cache.aping() is True


@pytest.mark.asyncio
async def test_async_redis_cache_stats() -> None:
    """Test AsyncRedisCache stats structure."""
    from toolcase.io.cache import AsyncRedisCache
    
    client = MockAsyncRedisClient()
    cache = AsyncRedisCache(client, prefix="test:")
    await cache.aset("tool", {"q": "test"}, "result")
    
    stats = await cache.astats()
    assert stats["backend"] == "redis"
    assert stats["prefix"] == "test:"
    assert stats["connected"] is True


def test_memcached_cache_basic() -> None:
    """Test sync MemcachedCache with mock client."""
    from toolcase.io.cache import MemcachedCache
    
    client = MockMemcachedClient()
    cache = MemcachedCache(client, prefix="test:")
    
    cache.set("tool", {"q": "test"}, "result")
    assert cache.get("tool", {"q": "test"}) == "result"
    assert cache.get("tool", {"q": "other"}) is None


def test_memcached_cache_invalidate() -> None:
    """Test MemcachedCache invalidation."""
    from toolcase.io.cache import MemcachedCache
    
    client = MockMemcachedClient()
    cache = MemcachedCache(client, prefix="test:")
    
    cache.set("tool", {"q": "a"}, "result_a")
    cache.set("tool", {"q": "b"}, "result_b")
    
    assert cache.invalidate("tool", {"q": "a"})
    assert cache.get("tool", {"q": "a"}) is None
    assert cache.get("tool", {"q": "b"}) == "result_b"


def test_memcached_cache_ping() -> None:
    """Test MemcachedCache ping."""
    from toolcase.io.cache import MemcachedCache
    
    client = MockMemcachedClient()
    cache = MemcachedCache(client, prefix="test:")
    assert cache.ping() is True


def test_memcached_cache_stats() -> None:
    """Test MemcachedCache stats structure."""
    from toolcase.io.cache import MemcachedCache
    
    client = MockMemcachedClient()
    cache = MemcachedCache(client, prefix="test:")
    
    stats = cache.stats()
    assert stats["backend"] == "memcached"
    assert stats["prefix"] == "test:"
    assert stats["connected"] is True


@pytest.mark.asyncio
async def test_async_memcached_cache_basic() -> None:
    """Test async MemcachedCache with mock client."""
    from toolcase.io.cache import AsyncMemcachedCache
    
    client = MockAsyncMemcachedClient()
    cache = AsyncMemcachedCache(client, prefix="test:")
    
    await cache.aset("tool", {"q": "test"}, "result")
    assert await cache.aget("tool", {"q": "test"}) == "result"
    assert await cache.aget("tool", {"q": "other"}) is None


@pytest.mark.asyncio
async def test_async_memcached_cache_ping() -> None:
    """Test AsyncMemcachedCache ping."""
    from toolcase.io.cache import AsyncMemcachedCache
    
    client = MockAsyncMemcachedClient()
    cache = AsyncMemcachedCache(client, prefix="test:")
    assert await cache.aping() is True


@pytest.mark.asyncio
async def test_async_memcached_cache_stats() -> None:
    """Test AsyncMemcachedCache stats structure."""
    from toolcase.io.cache import AsyncMemcachedCache
    
    client = MockAsyncMemcachedClient()
    cache = AsyncMemcachedCache(client, prefix="test:")
    
    stats = await cache.astats()
    assert stats["backend"] == "memcached"
    assert stats["prefix"] == "test:"
    assert stats["connected"] is True
