"""Distributed state stores for circuit breaker.

Uses msgpack binary storage for efficient state serialization.

Requires: pip install toolcase[redis]
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import msgpack

from .breaker import CircuitState


@runtime_checkable
class RedisClient(Protocol):
    """Protocol for sync Redis client (duck typing)."""
    def get(self, key: str) -> bytes | None: ...
    def set(self, name: str, value: bytes, ex: int | None = None) -> bool: ...
    def delete(self, *names: str) -> int: ...
    def scan_iter(self, match: str) -> object: ...


def _import_redis() -> object:
    """Lazy import redis with clear error."""
    try:
        import redis
        return redis
    except ImportError as e:
        raise ImportError(
            "Redis state store requires redis package. "
            "Install with: pip install toolcase[redis]"
        ) from e


class RedisStateStore:
    """Redis-backed circuit state store with msgpack binary storage.
    
    Uses msgpack for efficient state serialization (~40% smaller than JSON).
    
    Example:
        >>> import redis
        >>> r = redis.from_url("redis://localhost:6379/0")
        >>> store = RedisStateStore(r)
        >>> registry.use(CircuitBreakerMiddleware(store=store))
    """
    
    __slots__ = ("_client", "_prefix", "_ttl")
    
    def __init__(self, client: RedisClient, prefix: str = "cb:", ttl: int | None = None) -> None:
        self._client, self._prefix, self._ttl = client, prefix, ttl
    
    @classmethod
    def from_url(cls, url: str, prefix: str = "cb:", ttl: int | None = None, **kw: object) -> RedisStateStore:
        """Create store from Redis URL."""
        return cls(_import_redis().from_url(url, **kw), prefix, ttl)  # type: ignore[union-attr]
    
    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"
    
    def get(self, key: str) -> CircuitState | None:
        return CircuitState.from_dict(msgpack.unpackb(data, raw=False)) if (data := self._client.get(self._key(key))) else None
    
    def set(self, key: str, state: CircuitState) -> None:
        self._client.set(self._key(key), msgpack.packb(state.to_dict(), use_bin_type=True), ex=self._ttl)
    
    def delete(self, key: str) -> bool:
        """Delete circuit state from Redis."""
        return self._client.delete(self._key(key)) > 0
    
    def keys(self) -> list[str]:
        """Get all circuit keys (without prefix)."""
        prefix_len = len(self._prefix)
        return [k.decode()[prefix_len:] if isinstance(k, bytes) else k[prefix_len:] 
                for k in self._client.scan_iter(match=f"{self._prefix}*")]
