"""Redis cache backend for tool result caching.

Lightweight adapter for existing Redis deployments.
Supports both sync redis-py and async redis.asyncio clients.

Requires: pip install toolcase[redis]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from toolcase.foundation.errors import JsonDict

from .cache import DEFAULT_TTL, AsyncToolCache, ToolCache

if TYPE_CHECKING:
    from pydantic import BaseModel


@runtime_checkable
class RedisClient(Protocol):
    """Protocol for sync Redis client (duck typing)."""
    def get(self, key: str) -> bytes | None: ...
    def setex(self, name: str, time: int, value: str) -> bool: ...
    def delete(self, *names: str) -> int: ...
    def scan_iter(self, match: str) -> object: ...
    def ping(self) -> bool: ...
    def info(self, section: str = "") -> dict[str, object]: ...


@runtime_checkable  
class AsyncRedisClient(Protocol):
    """Protocol for async Redis client (duck typing)."""
    async def get(self, key: str) -> bytes | None: ...
    async def setex(self, name: str, time: int, value: str) -> bool: ...
    async def delete(self, *names: str) -> int: ...
    def scan_iter(self, match: str) -> object: ...
    async def ping(self) -> bool: ...
    async def info(self, section: str = "") -> dict[str, object]: ...


def _import_redis() -> object:
    """Lazy import redis with clear error."""
    try:
        import redis
        return redis
    except ImportError as e:
        raise ImportError(
            "Redis cache requires redis package. "
            "Install with: pip install toolcase[redis]"
        ) from e


class RedisCache(ToolCache):
    """Redis-backed tool cache for distributed deployments.
    
    Adapts existing Redis connections for tool result caching.
    TTL handled natively by Redis SETEX. Thread-safe by design.
    
    Args:
        client: Existing Redis client instance (sync)
        prefix: Key prefix for namespacing (default: "toolcase:")
        default_ttl: Default TTL in seconds (default: 300)
    
    Example:
        >>> import redis
        >>> r = redis.from_url("redis://localhost:6379/0")
        >>> cache = RedisCache(r)
        >>> set_cache(cache)  # Use globally
        
        # Or from URL directly:
        >>> cache = RedisCache.from_url("redis://localhost:6379/0")
    """
    
    __slots__ = ("_client", "_prefix", "_default_ttl")
    
    def __init__(
        self,
        client: RedisClient,
        prefix: str = "toolcase:",
        default_ttl: float = DEFAULT_TTL,
    ) -> None:
        self._client = client
        self._prefix = prefix
        self._default_ttl = default_ttl
    
    @classmethod
    def from_url(
        cls,
        url: str,
        prefix: str = "toolcase:",
        default_ttl: float = DEFAULT_TTL,
        **redis_kwargs: object,
    ) -> RedisCache:
        """Create cache from Redis URL.
        
        Args:
            url: Redis connection URL (redis://host:port/db)
            prefix: Key prefix for namespacing
            default_ttl: Default TTL in seconds
            **redis_kwargs: Additional args passed to redis.from_url
        
        Example:
            >>> cache = RedisCache.from_url("redis://localhost:6379/0")
        """
        redis = _import_redis()
        client = redis.from_url(url, **redis_kwargs)  # type: ignore[union-attr]
        return cls(client, prefix, default_ttl)
    
    def _key(self, tool_name: str, params: BaseModel | JsonDict) -> str:
        """Generate prefixed cache key."""
        return f"{self._prefix}{self.make_key(tool_name, params)}"
    
    def get(self, tool_name: str, params: BaseModel | JsonDict) -> str | None:
        val = self._client.get(self._key(tool_name, params))
        return val.decode() if val else None
    
    def set(
        self,
        tool_name: str,
        params: BaseModel | JsonDict,
        value: str,
        ttl: float | None = None,
    ) -> None:
        self._client.setex(self._key(tool_name, params), int(ttl or self._default_ttl), value)
    
    def invalidate(self, tool_name: str, params: BaseModel | JsonDict) -> bool:
        return self._client.delete(self._key(tool_name, params)) > 0
    
    def invalidate_tool(self, tool_name: str) -> int:
        """Remove all entries for a tool using SCAN (production-safe)."""
        keys = list(self._client.scan_iter(match=f"{self._prefix}{tool_name}:*"))
        return self._client.delete(*keys) if keys else 0
    
    def clear(self) -> None:
        """Clear all toolcase keys using SCAN."""
        if keys := list(self._client.scan_iter(match=f"{self._prefix}*")):
            self._client.delete(*keys)
    
    def ping(self) -> bool:
        """Check Redis connection health."""
        try:
            return bool(self._client.ping())
        except Exception:
            return False
    
    def stats(self) -> JsonDict:
        """Get Redis cache statistics."""
        try:
            info = self._client.info("memory")
            keys = list(self._client.scan_iter(match=f"{self._prefix}*"))
            return {
                "backend": "redis",
                "total_entries": len(keys),
                "prefix": self._prefix,
                "default_ttl": self._default_ttl,
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected": True,
            }
        except Exception:
            return {
                "backend": "redis",
                "prefix": self._prefix,
                "default_ttl": self._default_ttl,
                "connected": False,
            }


class AsyncRedisCache(AsyncToolCache):
    """Async Redis-backed tool cache for async deployments.
    
    Uses redis.asyncio for non-blocking operations.
    
    Args:
        client: Existing async Redis client instance
        prefix: Key prefix for namespacing (default: "toolcase:")
        default_ttl: Default TTL in seconds (default: 300)
    
    Example:
        >>> import redis.asyncio as redis
        >>> r = redis.from_url("redis://localhost:6379/0")
        >>> cache = AsyncRedisCache(r)
    """
    
    __slots__ = ("_client", "_prefix", "_default_ttl")
    
    def __init__(
        self,
        client: AsyncRedisClient,
        prefix: str = "toolcase:",
        default_ttl: float = DEFAULT_TTL,
    ) -> None:
        self._client = client
        self._prefix = prefix
        self._default_ttl = default_ttl
    
    @classmethod
    async def from_url(
        cls,
        url: str,
        prefix: str = "toolcase:",
        default_ttl: float = DEFAULT_TTL,
        **redis_kwargs: object,
    ) -> AsyncRedisCache:
        """Create async cache from Redis URL.
        
        Example:
            >>> cache = await AsyncRedisCache.from_url("redis://localhost:6379/0")
        """
        try:
            import redis.asyncio as aioredis
        except ImportError as e:
            raise ImportError(
                "Async Redis cache requires redis package. "
                "Install with: pip install toolcase[redis]"
            ) from e
        client = aioredis.from_url(url, **redis_kwargs)  # type: ignore[arg-type]
        return cls(client, prefix, default_ttl)
    
    def _key(self, tool_name: str, params: BaseModel | JsonDict) -> str:
        return f"{self._prefix}{self.make_key(tool_name, params)}"
    
    async def aget(self, tool_name: str, params: BaseModel | JsonDict) -> str | None:
        val = await self._client.get(self._key(tool_name, params))
        return val.decode() if val else None
    
    async def aset(
        self,
        tool_name: str,
        params: BaseModel | JsonDict,
        value: str,
        ttl: float | None = None,
    ) -> None:
        await self._client.setex(self._key(tool_name, params), int(ttl or self._default_ttl), value)
    
    async def ainvalidate(self, tool_name: str, params: BaseModel | JsonDict) -> bool:
        return await self._client.delete(self._key(tool_name, params)) > 0
    
    async def ainvalidate_tool(self, tool_name: str) -> int:
        keys = [k async for k in self._client.scan_iter(match=f"{self._prefix}{tool_name}:*")]
        return await self._client.delete(*keys) if keys else 0
    
    async def aclear(self) -> None:
        if keys := [k async for k in self._client.scan_iter(match=f"{self._prefix}*")]:
            await self._client.delete(*keys)
    
    async def aping(self) -> bool:
        """Check Redis connection health."""
        try:
            return bool(await self._client.ping())
        except Exception:
            return False
    
    async def astats(self) -> JsonDict:
        """Get Redis cache statistics."""
        try:
            info = await self._client.info("memory")
            keys = [k async for k in self._client.scan_iter(match=f"{self._prefix}*")]
            return {
                "backend": "redis",
                "total_entries": len(keys),
                "prefix": self._prefix,
                "default_ttl": self._default_ttl,
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected": True,
            }
        except Exception:
            return {
                "backend": "redis",
                "prefix": self._prefix,
                "default_ttl": self._default_ttl,
                "connected": False,
            }