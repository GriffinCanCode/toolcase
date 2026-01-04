"""Advanced cache invalidation strategies.

Provides:
- Tag-based invalidation: Group entries by semantic tags for bulk invalidation
- Pattern-based invalidation: fnmatch/regex patterns for flexible key matching
- Stale-while-revalidate (SWR): Serve stale content while refreshing in background

All strategies compose with existing cache backends via wrapper pattern.
"""

from __future__ import annotations

import asyncio
import fnmatch
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Generic, Protocol, TypeVar, runtime_checkable

from toolcase.foundation.errors import JsonDict, JsonMapping

if TYPE_CHECKING:
    from pydantic import BaseModel

T = TypeVar("T")

# ═══════════════════════════════════════════════════════════════════════════════
# Tagged Entry - Entry with Tags and Stale Window
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class TaggedEntry:
    """Cache entry with tag tracking and stale-while-revalidate support.
    
    Attributes:
        value: Cached string value
        expires_at: Time when entry becomes stale (soft expiry)
        stale_at: Time when entry is unusable (hard expiry)
        tags: Set of tags for group invalidation
    """
    value: str
    expires_at: float
    stale_at: float
    tags: frozenset[str] = field(default_factory=frozenset)
    
    @property
    def expired(self) -> bool:
        """True if past soft expiry (stale but usable if SWR enabled)."""
        return time.time() > self.expires_at
    
    @property
    def stale(self) -> bool:
        """True if past hard expiry (unusable)."""
        return time.time() > self.stale_at
    
    @property
    def is_fresh(self) -> bool:
        """True if before soft expiry."""
        return time.time() <= self.expires_at


# ═══════════════════════════════════════════════════════════════════════════════
# Invalidation Strategy Protocol
# ═══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class InvalidationStrategy(Protocol):
    """Protocol for invalidation strategies."""
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with given tag. Returns count removed."""
        ...
    
    def invalidate_by_tags(self, tags: frozenset[str]) -> int:
        """Invalidate all entries matching ANY of the tags. Returns count removed."""
        ...
    
    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate entries with keys matching fnmatch pattern. Returns count removed."""
        ...


@runtime_checkable
class AsyncInvalidationStrategy(Protocol):
    """Protocol for async invalidation strategies."""
    
    async def ainvalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with given tag. Returns count removed."""
        ...
    
    async def ainvalidate_by_tags(self, tags: frozenset[str]) -> int:
        """Invalidate all entries matching ANY of the tags. Returns count removed."""
        ...
    
    async def ainvalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate entries with keys matching fnmatch pattern. Returns count removed."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# Tag Index - Thread-Safe Reverse Index for Tag Lookups
# ═══════════════════════════════════════════════════════════════════════════════


class TagIndex:
    """Thread-safe reverse index mapping tags to cache keys.
    
    Enables O(1) tag-based invalidation by maintaining bidirectional mapping:
    - tag → set of keys (for invalidation)
    - key → set of tags (for cleanup)
    
    Example:
        >>> idx = TagIndex()
        >>> idx.add("key1", frozenset({"user:123", "product"}))
        >>> idx.keys_for_tag("user:123")
        {'key1'}
        >>> idx.remove("key1")
    """
    
    __slots__ = ("_tag_to_keys", "_key_to_tags", "_lock")
    
    def __init__(self) -> None:
        self._tag_to_keys: dict[str, set[str]] = {}
        self._key_to_tags: dict[str, frozenset[str]] = {}
        self._lock = threading.RLock()
    
    def add(self, key: str, tags: frozenset[str]) -> None:
        """Register key with tags."""
        if not tags:
            return
        with self._lock:
            self._key_to_tags[key] = tags
            for tag in tags:
                self._tag_to_keys.setdefault(tag, set()).add(key)
    
    def remove(self, key: str) -> frozenset[str]:
        """Unregister key, return its tags. O(tags) complexity."""
        with self._lock:
            if (tags := self._key_to_tags.pop(key, None)) is None:
                return frozenset()
            for tag in tags:
                if (keys := self._tag_to_keys.get(tag)):
                    keys.discard(key)
                    if not keys:
                        del self._tag_to_keys[tag]
            return tags
    
    def keys_for_tag(self, tag: str) -> frozenset[str]:
        """Get all keys with given tag. Thread-safe snapshot."""
        with self._lock:
            return frozenset(self._tag_to_keys.get(tag, ()))
    
    def keys_for_tags(self, tags: frozenset[str]) -> frozenset[str]:
        """Get keys matching ANY of the tags (union). Thread-safe snapshot."""
        with self._lock:
            result: set[str] = set()
            for tag in tags:
                result.update(self._tag_to_keys.get(tag, ()))
            return frozenset(result)
    
    def tags_for_key(self, key: str) -> frozenset[str]:
        """Get tags for a key."""
        with self._lock:
            return self._key_to_tags.get(key, frozenset())
    
    def clear(self) -> None:
        """Clear all mappings."""
        with self._lock:
            self._tag_to_keys.clear()
            self._key_to_tags.clear()
    
    @property
    def tag_count(self) -> int:
        """Number of unique tags."""
        with self._lock:
            return len(self._tag_to_keys)
    
    @property
    def key_count(self) -> int:
        """Number of indexed keys."""
        with self._lock:
            return len(self._key_to_tags)


# ═══════════════════════════════════════════════════════════════════════════════
# Pattern Matcher - Efficient Pattern Compilation and Matching
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class PatternMatcher:
    """Compiled pattern for efficient repeated matching.
    
    Supports both fnmatch glob patterns and regex. Glob patterns are
    translated to regex for consistent O(n) matching performance.
    
    Example:
        >>> m = PatternMatcher.glob("user:*:profile")
        >>> m.matches("user:123:profile")
        True
        >>> m = PatternMatcher.regex(r"tool:\\d+:.*")
        >>> m.matches("tool:42:search")
        True
    """
    _regex: re.Pattern[str]
    _original: str
    _is_glob: bool
    
    @classmethod
    def glob(cls, pattern: str) -> PatternMatcher:
        """Create matcher from fnmatch glob pattern."""
        return cls(re.compile(fnmatch.translate(pattern)), pattern, True)
    
    @classmethod
    def regex(cls, pattern: str) -> PatternMatcher:
        """Create matcher from regex pattern."""
        return cls(re.compile(pattern), pattern, False)
    
    def matches(self, key: str) -> bool:
        """Test if key matches pattern."""
        return self._regex.fullmatch(key) is not None
    
    def filter(self, keys: frozenset[str]) -> frozenset[str]:
        """Return subset of keys matching pattern."""
        return frozenset(k for k in keys if self.matches(k))
    
    def __repr__(self) -> str:
        kind = "glob" if self._is_glob else "regex"
        return f"PatternMatcher({kind}:{self._original!r})"


# ═══════════════════════════════════════════════════════════════════════════════
# SWR Entry State
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SWRState:
    """State from stale-while-revalidate lookup.
    
    Attributes:
        value: The cached value (may be stale)
        is_fresh: True if before TTL expiry
        is_stale: True if past TTL but within stale window
        is_expired: True if past stale window (unusable)
        revalidating: True if background refresh in progress
    """
    value: str | None
    is_fresh: bool = False
    is_stale: bool = False
    is_expired: bool = True
    revalidating: bool = False
    
    @property
    def hit(self) -> bool:
        """True if value is usable (fresh or stale)."""
        return self.value is not None and not self.is_expired
    
    @property
    def should_revalidate(self) -> bool:
        """True if stale and not already revalidating."""
        return self.is_stale and not self.revalidating


# Revalidation callback type
Revalidator = Callable[[str, "BaseModel | JsonMapping"], Coroutine[None, None, str | None]]


# ═══════════════════════════════════════════════════════════════════════════════
# SWR Config
# ═══════════════════════════════════════════════════════════════════════════════

# Default stale window multiplier (stale_window = ttl * multiplier)
DEFAULT_STALE_MULTIPLIER: Final[float] = 2.0


@dataclass(frozen=True, slots=True)
class SWRConfig:
    """Configuration for stale-while-revalidate behavior.
    
    Args:
        stale_multiplier: stale_window = ttl * multiplier (default: 2.0)
        max_concurrent_revalidations: Limit concurrent background refreshes
        revalidation_timeout: Timeout for background refresh in seconds
    
    Example:
        >>> cfg = SWRConfig(stale_multiplier=3.0, max_concurrent_revalidations=10)
        >>> # Entry with TTL=60s will be stale at 60s, expired at 180s
    """
    stale_multiplier: float = DEFAULT_STALE_MULTIPLIER
    max_concurrent_revalidations: int = 100
    revalidation_timeout: float = 30.0
    
    def stale_window(self, ttl: float) -> float:
        """Calculate stale window from TTL."""
        return ttl * self.stale_multiplier


# ═══════════════════════════════════════════════════════════════════════════════
# Tagged Memory Cache
# ═══════════════════════════════════════════════════════════════════════════════


class TaggedMemoryCache:
    """In-memory cache with tag-based invalidation and SWR support.
    
    Extends MemoryCache with:
    - Tag-based invalidation via reverse index
    - Pattern-based invalidation (fnmatch globs)
    - Stale-while-revalidate with background refresh
    
    Thread-safe via RLock. Uses TagIndex for O(1) tag lookups.
    
    Example:
        >>> cache = TaggedMemoryCache()
        >>> cache.set("tool", {"q": "test"}, "result", tags=frozenset({"user:123"}))
        >>> cache.invalidate_by_tag("user:123")  # Removes entry
        1
        >>> 
        >>> # With SWR
        >>> cache = TaggedMemoryCache(swr=SWRConfig())
        >>> state = cache.get_swr("tool", {"q": "test"})
        >>> if state.should_revalidate:
        ...     asyncio.create_task(refresh_entry(...))
    """
    
    __slots__ = ("_cache", "_index", "_default_ttl", "_max_entries", "_lock", "_swr", "_revalidating")
    
    def __init__(
        self,
        default_ttl: float = 300.0,
        max_entries: int = 1000,
        swr: SWRConfig | None = None,
    ) -> None:
        self._cache: dict[str, TaggedEntry] = {}
        self._index = TagIndex()
        self._default_ttl = default_ttl
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self._swr = swr
        self._revalidating: set[str] = set()  # Keys currently being revalidated
    
    @staticmethod
    def make_key(tool_name: str, params: BaseModel | JsonMapping) -> str:
        """Generate cache key from tool name and parameters."""
        import hashlib
        import orjson
        params_dict = params.model_dump(mode="json") if hasattr(params, "model_dump") else dict(params)
        params_bytes = orjson.dumps(params_dict, option=orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS)
        return f"{tool_name}:{hashlib.md5(params_bytes, usedforsecurity=False).hexdigest()[:12]}"
    
    def get(self, tool_name: str, params: BaseModel | JsonMapping) -> str | None:
        """Get cached result if exists and not stale."""
        key = self.make_key(tool_name, params)
        with self._lock:
            if not (entry := self._cache.get(key)):
                return None
            if entry.stale:
                self._remove_key(key)
                return None
            return entry.value
    
    def get_swr(self, tool_name: str, params: BaseModel | JsonMapping) -> SWRState:
        """Get cached result with SWR state information.
        
        Returns SWRState with value and freshness info. Caller should check
        `state.should_revalidate` to trigger background refresh.
        """
        key = self.make_key(tool_name, params)
        with self._lock:
            if not (entry := self._cache.get(key)):
                return SWRState(value=None, is_expired=True)
            
            if entry.stale:
                self._remove_key(key)
                return SWRState(value=None, is_expired=True)
            
            revalidating = key in self._revalidating
            if entry.is_fresh:
                return SWRState(value=entry.value, is_fresh=True, revalidating=revalidating)
            
            return SWRState(value=entry.value, is_stale=True, is_expired=False, revalidating=revalidating)
    
    def set(
        self,
        tool_name: str,
        params: BaseModel | JsonMapping,
        value: str,
        ttl: float | None = None,
        tags: frozenset[str] | None = None,
    ) -> None:
        """Store result in cache with optional tags."""
        key = self.make_key(tool_name, params)
        actual_ttl = ttl or self._default_ttl
        now = time.time()
        
        # Calculate stale window
        stale_at = now + (self._swr.stale_window(actual_ttl) if self._swr else actual_ttl)
        
        with self._lock:
            # Remove old entry from index if exists
            if key in self._cache:
                self._index.remove(key)
            
            if len(self._cache) >= self._max_entries:
                self._evict_unlocked()
            
            entry = TaggedEntry(
                value=value,
                expires_at=now + actual_ttl,
                stale_at=stale_at,
                tags=tags or frozenset(),
            )
            self._cache[key] = entry
            self._index.add(key, entry.tags)
            self._revalidating.discard(key)  # Clear revalidation flag
    
    def invalidate(self, tool_name: str, params: BaseModel | JsonMapping) -> bool:
        """Remove specific entry from cache."""
        with self._lock:
            return self._remove_key(self.make_key(tool_name, params))
    
    def invalidate_tool(self, tool_name: str) -> int:
        """Remove all entries for a tool. Returns count removed."""
        prefix = f"{tool_name}:"
        with self._lock:
            keys = [k for k in self._cache if k.startswith(prefix)]
            for k in keys:
                self._remove_key(k)
            return len(keys)
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with given tag. Returns count removed."""
        with self._lock:
            keys = self._index.keys_for_tag(tag)
            for k in keys:
                self._remove_key(k)
            return len(keys)
    
    def invalidate_by_tags(self, tags: frozenset[str]) -> int:
        """Invalidate all entries matching ANY of the tags. Returns count removed."""
        with self._lock:
            keys = self._index.keys_for_tags(tags)
            for k in keys:
                self._remove_key(k)
            return len(keys)
    
    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate entries with keys matching fnmatch pattern. Returns count removed."""
        matcher = PatternMatcher.glob(pattern)
        with self._lock:
            keys = matcher.filter(frozenset(self._cache.keys()))
            for k in keys:
                self._remove_key(k)
            return len(keys)
    
    def mark_revalidating(self, tool_name: str, params: BaseModel | JsonMapping) -> bool:
        """Mark entry as being revalidated. Returns False if already revalidating."""
        key = self.make_key(tool_name, params)
        with self._lock:
            if key in self._revalidating:
                return False
            if self._swr and len(self._revalidating) >= self._swr.max_concurrent_revalidations:
                return False
            self._revalidating.add(key)
            return True
    
    def clear_revalidating(self, tool_name: str, params: BaseModel | JsonMapping) -> None:
        """Clear revalidation flag for entry."""
        with self._lock:
            self._revalidating.discard(self.make_key(tool_name, params))
    
    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._index.clear()
            self._revalidating.clear()
    
    def ping(self) -> bool:
        """Memory cache is always available."""
        return True
    
    def _remove_key(self, key: str) -> bool:
        """Remove key from cache and index. Caller must hold lock."""
        if key not in self._cache:
            return False
        del self._cache[key]
        self._index.remove(key)
        self._revalidating.discard(key)
        return True
    
    def _evict_unlocked(self) -> None:
        """Remove expired entries, then oldest if still over capacity. Caller must hold lock."""
        # Remove stale entries first
        for k in [k for k, v in self._cache.items() if v.stale]:
            self._remove_key(k)
        
        if len(self._cache) >= self._max_entries:
            # Remove oldest (by stale_at) quarter of entries
            for k in sorted(self._cache, key=lambda k: self._cache[k].stale_at)[:self._max_entries // 4]:
                self._remove_key(k)
    
    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)
    
    def stats(self) -> JsonDict:
        """Get cache statistics for monitoring."""
        now = time.time()
        with self._lock:
            stale = sum(1 for v in self._cache.values() if v.stale)
            expired = sum(1 for v in self._cache.values() if v.expired and not v.stale)
            fresh = sum(1 for v in self._cache.values() if v.is_fresh)
            return {
                "backend": "memory_tagged",
                "total_entries": len(self._cache),
                "fresh_entries": fresh,
                "stale_entries": expired,  # past TTL but within stale window
                "expired_entries": stale,  # past stale window
                "active_entries": len(self._cache) - stale,
                "tag_count": self._index.tag_count,
                "revalidating": len(self._revalidating),
                "default_ttl": self._default_ttl,
                "max_entries": self._max_entries,
                "swr_enabled": self._swr is not None,
            }


# ═══════════════════════════════════════════════════════════════════════════════
# SWR Cache Wrapper - Stale-While-Revalidate Pattern
# ═══════════════════════════════════════════════════════════════════════════════


class SWRCache(Generic[T]):
    """Stale-while-revalidate cache wrapper.
    
    Wraps TaggedMemoryCache with automatic background revalidation.
    On stale hit, returns cached value immediately and triggers async refresh.
    
    Example:
        >>> async def fetch_data(tool: str, params: JsonMapping) -> str | None:
        ...     return await api.call(tool, params)
        >>> 
        >>> cache = SWRCache(TaggedMemoryCache(), revalidator=fetch_data)
        >>> result = await cache.get_or_revalidate("search", {"q": "test"})
    """
    
    __slots__ = ("_cache", "_revalidator", "_config", "_tasks")
    
    def __init__(
        self,
        cache: TaggedMemoryCache,
        revalidator: Revalidator,
        config: SWRConfig | None = None,
    ) -> None:
        self._cache = cache
        self._revalidator = revalidator
        self._config = config or SWRConfig()
        self._tasks: set[asyncio.Task[None]] = set()
    
    async def get_or_revalidate(
        self,
        tool_name: str,
        params: BaseModel | JsonMapping,
        tags: frozenset[str] | None = None,
    ) -> str | None:
        """Get cached value, triggering background revalidation if stale.
        
        Returns immediately with cached value (if any). If stale, spawns
        background task to refresh. Returns None only on complete miss.
        """
        state = self._cache.get_swr(tool_name, params)
        
        if state.is_fresh:
            return state.value
        
        if state.should_revalidate and self._cache.mark_revalidating(tool_name, params):
            # Spawn background revalidation
            task = asyncio.create_task(self._revalidate(tool_name, params, tags))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        
        return state.value  # Return stale value (or None if expired)
    
    async def _revalidate(
        self,
        tool_name: str,
        params: BaseModel | JsonMapping,
        tags: frozenset[str] | None,
    ) -> None:
        """Background revalidation task."""
        try:
            async with asyncio.timeout(self._config.revalidation_timeout):
                if (result := await self._revalidator(tool_name, params)) is not None:
                    self._cache.set(tool_name, params, result, tags=tags)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        except Exception:
            pass  # Swallow errors in background task
        finally:
            self._cache.clear_revalidating(tool_name, params)
    
    async def close(self) -> None:
        """Cancel pending revalidation tasks."""
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
    
    @property
    def cache(self) -> TaggedMemoryCache:
        """Access underlying cache."""
        return self._cache
    
    @property
    def pending_revalidations(self) -> int:
        """Number of pending revalidation tasks."""
        return len(self._tasks)
