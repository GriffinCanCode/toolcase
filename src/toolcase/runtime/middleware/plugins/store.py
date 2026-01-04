"""Distributed state stores for circuit breaker.

Provides Redis-backed state storage for circuit breaker middleware,
enabling distributed deployments where circuit state is shared
across multiple instances.

Requires: pip install toolcase[redis]
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from .breaker import CircuitState


@runtime_checkable
class RedisClient(Protocol):
    """Protocol for sync Redis client (duck typing)."""
    def get(self, key: str) -> bytes | None: ...
    def set(self, name: str, value: str, ex: int | None = None) -> bool: ...
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
    """Redis-backed circuit state store for distributed deployments.
    
    Shares circuit breaker state across multiple instances via Redis.
    State is JSON-serialized with optional TTL for auto-cleanup of stale circuits.
    
    Args:
        client: Existing Redis client instance
        prefix: Key prefix for namespacing (default: "breaker:")
        ttl: Optional TTL in seconds for state keys (default: None = no expiry)
    
    Example:
        >>> import redis
        >>> r = redis.from_url("redis://localhost:6379/0")
        >>> store = RedisStateStore(r)
        >>> registry.use(CircuitBreakerMiddleware(store=store))
        
        # Or from URL:
        >>> store = RedisStateStore.from_url("redis://localhost:6379/0")
    """
    
    __slots__ = ("_client", "_prefix", "_ttl")
    
    def __init__(self, client: RedisClient, prefix: str = "breaker:", ttl: int | None = None) -> None:
        self._client = client
        self._prefix = prefix
        self._ttl = ttl
    
    @classmethod
    def from_url(cls, url: str, prefix: str = "breaker:", ttl: int | None = None, **kwargs: object) -> RedisStateStore:
        """Create store from Redis URL.
        
        Args:
            url: Redis connection URL (redis://host:port/db)
            prefix: Key prefix for namespacing
            ttl: Optional TTL in seconds
            **kwargs: Additional args passed to redis.from_url
        """
        redis = _import_redis()
        client = redis.from_url(url, **kwargs)  # type: ignore[union-attr]
        return cls(client, prefix, ttl)
    
    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"
    
    def get(self, key: str) -> CircuitState | None:
        """Get circuit state from Redis."""
        if (data := self._client.get(self._key(key))):
            return CircuitState.from_dict(json.loads(data))
        return None
    
    def set(self, key: str, state: CircuitState) -> None:
        """Store circuit state in Redis."""
        self._client.set(self._key(key), json.dumps(state.to_dict()), ex=self._ttl)
    
    def delete(self, key: str) -> bool:
        """Delete circuit state from Redis."""
        return self._client.delete(self._key(key)) > 0
    
    def keys(self) -> list[str]:
        """Get all circuit keys (without prefix)."""
        prefix_len = len(self._prefix)
        return [k.decode()[prefix_len:] if isinstance(k, bytes) else k[prefix_len:] 
                for k in self._client.scan_iter(match=f"{self._prefix}*")]
