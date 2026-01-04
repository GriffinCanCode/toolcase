"""Redis cache backend for tool result caching.

Uses msgpack binary storage (faster, ~40% smaller than JSON strings).

Requires: pip install toolcase[redis]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from toolcase.foundation.errors import JsonDict

from .cache import DEFAULT_TTL, AsyncToolCache, ToolCache, pack_value, unpack_value

if TYPE_CHECKING:
    from pydantic import BaseModel


@runtime_checkable
class RedisClient(Protocol):
    """Protocol for sync Redis client (duck typing)."""
    def get(self, key: str) -> bytes | None: ...
    def set(self, name: str, value: bytes, ex: int | None = None) -> bool: ...
    def delete(self, *names: str) -> int: ...
    def scan_iter(self, match: str) -> object: ...
    def ping(self) -> bool: ...
    def info(self, section: str = "") -> dict[str, object]: ...


@runtime_checkable  
class AsyncRedisClient(Protocol):
    """Protocol for async Redis client (duck typing)."""
    async def get(self, key: str) -> bytes | None: ...
    async def set(self, name: str, value: bytes, ex: int | None = None) -> bool: ...
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
    """Redis-backed tool cache with msgpack binary storage.
    
    Uses msgpack for ~40% smaller payloads and faster serialization.
    TTL handled natively by Redis SET EX. Thread-safe by design.
    
    Args:
        client: Existing Redis client instance (sync)
        prefix: Key prefix for namespacing (default: "tc:")
        default_ttl: Default TTL in seconds (default: 300)
    
    Example:
        >>> import redis
        >>> r = redis.from_url("redis://localhost:6379/0")
        >>> cache = RedisCache(r)
        >>> set_cache(cache)
    """
    
    __slots__ = ("_client", "_prefix", "_default_ttl")
    
    def __init__(self, client: RedisClient, prefix: str = "tc:", default_ttl: float = DEFAULT_TTL) -> None:
        self._client, self._prefix, self._default_ttl = client, prefix, default_ttl
    
    @classmethod
    def from_url(cls, url: str, prefix: str = "tc:", default_ttl: float = DEFAULT_TTL, **kw: object) -> RedisCache:
        """Create cache from Redis URL."""
        return cls(_import_redis().from_url(url, **kw), prefix, default_ttl)  # type: ignore[union-attr]
    
    def _key(self, tool_name: str, params: BaseModel | JsonDict) -> str:
        return f"{self._prefix}{self.make_key(tool_name, params)}"
    
    def get(self, tool_name: str, params: BaseModel | JsonDict) -> str | None:
        return unpack_value(val) if (val := self._client.get(self._key(tool_name, params))) else None
    
    def set(self, tool_name: str, params: BaseModel | JsonDict, value: str, ttl: float | None = None) -> None:
        self._client.set(self._key(tool_name, params), pack_value(value), ex=int(ttl or self._default_ttl))
    
    def invalidate(self, tool_name: str, params: BaseModel | JsonDict) -> bool:
        return self._client.delete(self._key(tool_name, params)) > 0
    
    def invalidate_tool(self, tool_name: str) -> int:
        keys = list(self._client.scan_iter(match=f"{self._prefix}{tool_name}:*"))
        return self._client.delete(*keys) if keys else 0
    
    def clear(self) -> None:
        if keys := list(self._client.scan_iter(match=f"{self._prefix}*")):
            self._client.delete(*keys)
    
    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            return False
    
    def stats(self) -> JsonDict:
        try:
            info = self._client.info("memory")
            keys = list(self._client.scan_iter(match=f"{self._prefix}*"))
            return {"backend": "redis", "encoding": "msgpack", "total_entries": len(keys),
                    "prefix": self._prefix, "default_ttl": self._default_ttl,
                    "used_memory": info.get("used_memory_human", "unknown"), "connected": True}
        except Exception:
            return {"backend": "redis", "prefix": self._prefix, "default_ttl": self._default_ttl, "connected": False}


class AsyncRedisCache(AsyncToolCache):
    """Async Redis-backed tool cache with msgpack binary storage.
    
    Uses redis.asyncio for non-blocking operations with msgpack serialization.
    
    Example:
        >>> import redis.asyncio as redis
        >>> r = redis.from_url("redis://localhost:6379/0")
        >>> cache = AsyncRedisCache(r)
    """
    
    __slots__ = ("_client", "_prefix", "_default_ttl")
    
    def __init__(self, client: AsyncRedisClient, prefix: str = "tc:", default_ttl: float = DEFAULT_TTL) -> None:
        self._client, self._prefix, self._default_ttl = client, prefix, default_ttl
    
    @classmethod
    async def from_url(cls, url: str, prefix: str = "tc:", default_ttl: float = DEFAULT_TTL, **kw: object) -> AsyncRedisCache:
        """Create async cache from Redis URL."""
        try:
            import redis.asyncio as aioredis
        except ImportError as e:
            raise ImportError("Async Redis cache requires: pip install toolcase[redis]") from e
        return cls(aioredis.from_url(url, **kw), prefix, default_ttl)  # type: ignore[arg-type]
    
    def _key(self, tool_name: str, params: BaseModel | JsonDict) -> str:
        return f"{self._prefix}{self.make_key(tool_name, params)}"
    
    async def aget(self, tool_name: str, params: BaseModel | JsonDict) -> str | None:
        return unpack_value(val) if (val := await self._client.get(self._key(tool_name, params))) else None
    
    async def aset(self, tool_name: str, params: BaseModel | JsonDict, value: str, ttl: float | None = None) -> None:
        await self._client.set(self._key(tool_name, params), pack_value(value), ex=int(ttl or self._default_ttl))
    
    async def ainvalidate(self, tool_name: str, params: BaseModel | JsonDict) -> bool:
        return await self._client.delete(self._key(tool_name, params)) > 0
    
    async def ainvalidate_tool(self, tool_name: str) -> int:
        keys = [k async for k in self._client.scan_iter(match=f"{self._prefix}{tool_name}:*")]
        return await self._client.delete(*keys) if keys else 0
    
    async def aclear(self) -> None:
        if keys := [k async for k in self._client.scan_iter(match=f"{self._prefix}*")]:
            await self._client.delete(*keys)
    
    async def aping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except Exception:
            return False
    
    async def astats(self) -> JsonDict:
        try:
            info = await self._client.info("memory")
            keys = [k async for k in self._client.scan_iter(match=f"{self._prefix}*")]
            return {"backend": "redis", "encoding": "msgpack", "total_entries": len(keys),
                    "prefix": self._prefix, "default_ttl": self._default_ttl,
                    "used_memory": info.get("used_memory_human", "unknown"), "connected": True}
        except Exception:
            return {"backend": "redis", "prefix": self._prefix, "default_ttl": self._default_ttl, "connected": False}
