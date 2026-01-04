"""Memcached cache backend for tool result caching.

Lightweight adapter for Memcached deployments.
Uses pymemcache for sync and aiomcache for async.

Requires: pip install toolcase[memcached]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from toolcase.foundation.errors import JsonDict

from .cache import DEFAULT_TTL, AsyncToolCache, ToolCache

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
    """Lazy import pymemcache with clear error."""
    try:
        from pymemcache import client as pymemcache_client
        return pymemcache_client
    except ImportError as e:
        raise ImportError(
            "Memcached cache requires pymemcache package. "
            "Install with: pip install toolcase[memcached]"
        ) from e


def _import_aiomcache() -> object:
    """Lazy import aiomcache with clear error."""
    try:
        import aiomcache
        return aiomcache
    except ImportError as e:
        raise ImportError(
            "Async Memcached cache requires aiomcache package. "
            "Install with: pip install aiomcache"
        ) from e


class MemcachedCache(ToolCache):
    """Memcached-backed tool cache for distributed deployments.
    
    Adapts existing Memcached connections for tool result caching.
    TTL handled natively by Memcached. Thread-safe via pymemcache.
    
    Args:
        client: Existing pymemcache client instance
        prefix: Key prefix for namespacing (default: "tc:")
        default_ttl: Default TTL in seconds (default: 300)
    
    Example:
        >>> from pymemcache.client import base
        >>> mc = base.Client(("localhost", 11211))
        >>> cache = MemcachedCache(mc)
        >>> set_cache(cache)  # Use globally
        
        # Or from server directly:
        >>> cache = MemcachedCache.from_server("localhost", 11211)
    """
    
    __slots__ = ("_client", "_prefix", "_default_ttl")
    
    def __init__(
        self,
        client: MemcachedClient,
        prefix: str = "tc:",
        default_ttl: float = DEFAULT_TTL,
    ) -> None:
        self._client = client
        self._prefix = prefix
        self._default_ttl = default_ttl
    
    @classmethod
    def from_server(
        cls,
        host: str = "localhost",
        port: int = 11211,
        prefix: str = "tc:",
        default_ttl: float = DEFAULT_TTL,
        **mc_kwargs: object,
    ) -> MemcachedCache:
        """Create cache from Memcached server address.
        
        Args:
            host: Memcached server hostname
            port: Memcached server port
            prefix: Key prefix for namespacing
            default_ttl: Default TTL in seconds
            **mc_kwargs: Additional args passed to pymemcache.Client
        
        Example:
            >>> cache = MemcachedCache.from_server("localhost", 11211)
        """
        pymemcache = _import_pymemcache()
        client = pymemcache.Client((host, port), **mc_kwargs)  # type: ignore[union-attr]
        return cls(client, prefix, default_ttl)
    
    def _key(self, tool_name: str, params: BaseModel | JsonDict) -> str:
        """Generate prefixed cache key (Memcached keys limited to 250 bytes)."""
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
        self._client.set(
            self._key(tool_name, params),
            value.encode(),
            expire=int(ttl or self._default_ttl),
        )
    
    def invalidate(self, tool_name: str, params: BaseModel | JsonDict) -> bool:
        return self._client.delete(self._key(tool_name, params))
    
    def invalidate_tool(self, tool_name: str) -> int:
        """Memcached doesn't support key scanning - returns 0.
        
        Note: Memcached doesn't support pattern-based key deletion.
        Consider using Redis if you need this functionality.
        """
        return 0  # Memcached doesn't support SCAN
    
    def clear(self) -> None:
        """Memcached doesn't support namespace clearing.
        
        Note: Only full flush_all is available, which clears ALL keys.
        This method is a no-op to prevent unintended data loss.
        """
        pass  # No-op to avoid flushing all caches
    
    def ping(self) -> bool:
        """Check Memcached connection health via stats."""
        try:
            return bool(self._client.stats())
        except Exception:
            return False
    
    def stats(self) -> JsonDict:
        """Get Memcached cache statistics."""
        try:
            raw_stats = self._client.stats()
            # pymemcache returns {server_addr: {stat_name: value}}
            stats_dict = {}
            for server_stats in raw_stats.values():
                for k, v in server_stats.items():
                    key = k.decode() if isinstance(k, bytes) else k
                    val = v.decode() if isinstance(v, bytes) else v
                    stats_dict[key] = val
            return {
                "backend": "memcached",
                "prefix": self._prefix,
                "default_ttl": self._default_ttl,
                "curr_items": int(stats_dict.get("curr_items", 0)),
                "bytes": int(stats_dict.get("bytes", 0)),
                "get_hits": int(stats_dict.get("get_hits", 0)),
                "get_misses": int(stats_dict.get("get_misses", 0)),
                "connected": True,
            }
        except Exception:
            return {
                "backend": "memcached",
                "prefix": self._prefix,
                "default_ttl": self._default_ttl,
                "connected": False,
            }


class AsyncMemcachedCache(AsyncToolCache):
    """Async Memcached-backed tool cache for async deployments.
    
    Uses aiomcache for non-blocking operations.
    
    Args:
        client: Existing aiomcache client instance
        prefix: Key prefix for namespacing (default: "tc:")
        default_ttl: Default TTL in seconds (default: 300)
    
    Example:
        >>> import aiomcache
        >>> mc = aiomcache.Client("localhost", 11211)
        >>> cache = AsyncMemcachedCache(mc)
    """
    
    __slots__ = ("_client", "_prefix", "_default_ttl")
    
    def __init__(
        self,
        client: AsyncMemcachedClient,
        prefix: str = "tc:",
        default_ttl: float = DEFAULT_TTL,
    ) -> None:
        self._client = client
        self._prefix = prefix
        self._default_ttl = default_ttl
    
    @classmethod
    def from_server(
        cls,
        host: str = "localhost",
        port: int = 11211,
        prefix: str = "tc:",
        default_ttl: float = DEFAULT_TTL,
    ) -> AsyncMemcachedCache:
        """Create async cache from Memcached server address.
        
        Example:
            >>> cache = AsyncMemcachedCache.from_server("localhost", 11211)
        """
        aiomcache = _import_aiomcache()
        client = aiomcache.Client(host, port)  # type: ignore[union-attr]
        return cls(client, prefix, default_ttl)
    
    def _key(self, tool_name: str, params: BaseModel | JsonDict) -> bytes:
        """Generate prefixed cache key as bytes for aiomcache."""
        return f"{self._prefix}{self.make_key(tool_name, params)}".encode()
    
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
        await self._client.set(
            self._key(tool_name, params),
            value.encode(),
            exptime=int(ttl or self._default_ttl),
        )
    
    async def ainvalidate(self, tool_name: str, params: BaseModel | JsonDict) -> bool:
        return await self._client.delete(self._key(tool_name, params))
    
    async def ainvalidate_tool(self, tool_name: str) -> int:
        """Memcached doesn't support key scanning - returns 0."""
        return 0  # Memcached doesn't support pattern-based operations
    
    async def aclear(self) -> None:
        """Memcached doesn't support namespace clearing - no-op."""
        pass
    
    async def aping(self) -> bool:
        """Check Memcached connection health."""
        try:
            return bool(await self._client.stats())
        except Exception:
            return False
    
    async def astats(self) -> JsonDict:
        """Get Memcached cache statistics."""
        try:
            raw_stats = await self._client.stats()
            stats_dict = {
                k.decode(): v.decode() if isinstance(v, bytes) else v
                for k, v in raw_stats.items()
            }
            return {
                "backend": "memcached",
                "prefix": self._prefix,
                "default_ttl": self._default_ttl,
                "curr_items": int(stats_dict.get("curr_items", 0)),
                "bytes": int(stats_dict.get("bytes", 0)),
                "get_hits": int(stats_dict.get("get_hits", 0)),
                "get_misses": int(stats_dict.get("get_misses", 0)),
                "connected": True,
            }
        except Exception:
            return {
                "backend": "memcached",
                "prefix": self._prefix,
                "default_ttl": self._default_ttl,
                "connected": False,
            }
