"""Memcached cache backend for tool result caching.

Uses msgpack binary storage (faster, ~40% smaller than JSON strings).

Requires: pip install toolcase[memcached]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from toolcase.foundation.errors import JsonDict, JsonMapping

from .cache import DEFAULT_TTL, AsyncToolCache, ToolCache, pack_value, unpack_value

if TYPE_CHECKING:
    from pydantic import BaseModel


@runtime_checkable
class MemcachedClient(Protocol):
    """Protocol for sync Memcached client (duck typing)."""
    def get(self, key: str) -> bytes | None: ...
    def set(self, key: str, value: bytes, expire: int = 0) -> bool: ...
    def delete(self, key: str) -> bool: ...
    def stats(self) -> dict[bytes, dict[bytes, bytes]]: ...


@runtime_checkable
class AsyncMemcachedClient(Protocol):
    """Protocol for async Memcached client (duck typing)."""
    async def get(self, key: bytes) -> bytes | None: ...
    async def set(self, key: bytes, value: bytes, exptime: int = 0) -> bool: ...
    async def delete(self, key: bytes) -> bool: ...
    async def stats(self) -> dict[bytes, bytes]: ...


def _import_pymemcache() -> object:
    try:
        from pymemcache import client as pymemcache_client
        return pymemcache_client
    except ImportError as e:
        raise ImportError("Memcached cache requires: pip install toolcase[memcached]") from e


class MemcachedCache(ToolCache):
    """Memcached-backed tool cache with msgpack binary storage.
    
    Uses msgpack for ~40% smaller payloads and faster serialization.
    TTL handled natively by Memcached. Thread-safe via pymemcache.
    
    Args:
        client: Existing pymemcache client instance
        prefix: Key prefix for namespacing (default: "tc:")
        default_ttl: Default TTL in seconds (default: 300)
    
    Example:
        >>> from pymemcache.client import base
        >>> mc = base.Client(("localhost", 11211))
        >>> cache = MemcachedCache(mc)
    """
    
    __slots__ = ("_client", "_prefix", "_default_ttl")
    
    def __init__(self, client: MemcachedClient, prefix: str = "tc:", default_ttl: float = DEFAULT_TTL) -> None:
        self._client, self._prefix, self._default_ttl = client, prefix, default_ttl
    
    @classmethod
    def from_server(cls, host: str = "localhost", port: int = 11211, prefix: str = "tc:",
                    default_ttl: float = DEFAULT_TTL, **kw: object) -> MemcachedCache:
        """Create cache from Memcached server address."""
        return cls(_import_pymemcache().Client((host, port), **kw), prefix, default_ttl)  # type: ignore[union-attr]
    
    def _key(self, tool_name: str, params: BaseModel | JsonMapping) -> str:
        return f"{self._prefix}{self.make_key(tool_name, params)}"
    
    def get(self, tool_name: str, params: BaseModel | JsonMapping) -> str | None:
        return unpack_value(val) if (val := self._client.get(self._key(tool_name, params))) else None
    
    def set(self, tool_name: str, params: BaseModel | JsonMapping, value: str, ttl: float | None = None) -> None:
        self._client.set(self._key(tool_name, params), pack_value(value), expire=int(ttl or self._default_ttl))
    
    def invalidate(self, tool_name: str, params: BaseModel | JsonMapping) -> bool:
        return self._client.delete(self._key(tool_name, params))
    
    def invalidate_tool(self, tool_name: str) -> int:
        return 0  # Memcached doesn't support SCAN
    
    def clear(self) -> None:
        pass  # No-op - Memcached doesn't support namespace clearing
    
    def ping(self) -> bool:
        try:
            return bool(self._client.stats())
        except Exception:
            return False
    
    def stats(self) -> JsonDict:
        try:
            raw = self._client.stats()
            s = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
                 for srv in raw.values() for k, v in srv.items()}
            return {"backend": "memcached", "encoding": "msgpack", "prefix": self._prefix,
                    "default_ttl": self._default_ttl, "curr_items": int(s.get("curr_items", 0)),
                    "bytes": int(s.get("bytes", 0)), "connected": True}
        except Exception:
            return {"backend": "memcached", "prefix": self._prefix, "default_ttl": self._default_ttl, "connected": False}


class AsyncMemcachedCache(AsyncToolCache):
    """Async Memcached-backed tool cache with msgpack binary storage.
    
    Uses aiomcache for non-blocking operations.
    
    Example:
        >>> import aiomcache
        >>> mc = aiomcache.Client("localhost", 11211)
        >>> cache = AsyncMemcachedCache(mc)
    """
    
    __slots__ = ("_client", "_prefix", "_default_ttl")
    
    def __init__(self, client: AsyncMemcachedClient, prefix: str = "tc:", default_ttl: float = DEFAULT_TTL) -> None:
        self._client, self._prefix, self._default_ttl = client, prefix, default_ttl
    
    @classmethod
    def from_server(cls, host: str = "localhost", port: int = 11211, prefix: str = "tc:",
                    default_ttl: float = DEFAULT_TTL) -> AsyncMemcachedCache:
        """Create async cache from Memcached server address."""
        try:
            import aiomcache
        except ImportError as e:
            raise ImportError("Async Memcached cache requires: pip install aiomcache") from e
        return cls(aiomcache.Client(host, port), prefix, default_ttl)  # type: ignore[arg-type]
    
    def _key(self, tool_name: str, params: BaseModel | JsonMapping) -> bytes:
        return f"{self._prefix}{self.make_key(tool_name, params)}".encode()
    
    async def aget(self, tool_name: str, params: BaseModel | JsonMapping) -> str | None:
        return unpack_value(val) if (val := await self._client.get(self._key(tool_name, params))) else None
    
    async def aset(self, tool_name: str, params: BaseModel | JsonMapping, value: str, ttl: float | None = None) -> None:
        await self._client.set(self._key(tool_name, params), pack_value(value), exptime=int(ttl or self._default_ttl))
    
    async def ainvalidate(self, tool_name: str, params: BaseModel | JsonMapping) -> bool:
        return await self._client.delete(self._key(tool_name, params))
    
    async def ainvalidate_tool(self, tool_name: str) -> int:
        return 0  # Memcached doesn't support pattern operations
    
    async def aclear(self) -> None:
        pass  # No-op
    
    async def aping(self) -> bool:
        try:
            return bool(await self._client.stats())
        except Exception:
            return False
    
    async def astats(self) -> JsonDict:
        try:
            raw = await self._client.stats()
            s = {(k.decode() if isinstance(k, bytes) else k): v for k, v in raw.items()}
            return {"backend": "memcached", "encoding": "msgpack", "prefix": self._prefix,
                    "default_ttl": self._default_ttl, "curr_items": int(s.get("curr_items", 0)),
                    "bytes": int(s.get("bytes", 0)), "connected": True}
        except Exception:
            return {"backend": "memcached", "prefix": self._prefix, "default_ttl": self._default_ttl, "connected": False}
