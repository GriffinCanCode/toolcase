"""Dead Letter Queue for batch items that exceed failure thresholds.

Provides:
- DLQEntry: Immutable record of failed items with context
- DLQStore: Protocol for DLQ storage backends
- DLQConfig: Configuration for poison message handling
- MemoryDLQStore: In-memory store (default, useful for testing)

Design: Items failing beyond max_poison_threshold are routed to DLQ
rather than retried. Integrates with IdempotentBatchConfig via callback.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Protocol, runtime_checkable

import orjson
from pydantic import BaseModel, ConfigDict, Field

from toolcase.foundation.errors import ErrorTrace, JsonDict

if TYPE_CHECKING:
    from .batch import BatchItem


# ═══════════════════════════════════════════════════════════════════════════════
# DLQ Entry
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class DLQEntry:
    """Immutable record of a poisoned batch item.
    
    Captures full context for debugging and potential manual reprocessing.
    
    Attributes:
        batch_id: Original batch identifier
        item_index: Index in original batch
        tool_name: Tool that produced the failure
        params: Serialized parameters
        error: Final error trace
        attempts: Total attempts before DLQ
        timestamp: Unix timestamp when queued
        metadata: Optional additional context
    """
    batch_id: str
    item_index: int
    tool_name: str
    params: bytes  # orjson-serialized for efficiency
    error: ErrorTrace
    attempts: int
    timestamp: float = field(default_factory=time.time)
    metadata: JsonDict = field(default_factory=dict)
    
    @property
    def params_dict(self) -> JsonDict:
        """Deserialize params on demand."""
        return orjson.loads(self.params)
    
    @property
    def key(self) -> str:
        """Unique key for this entry."""
        return f"{self.batch_id}:{self.item_index}"
    
    @classmethod
    def from_item(
        cls,
        item: BatchItem[BaseModel],
        batch_id: str,
        tool_name: str,
        params: BaseModel,
        attempts: int,
        **metadata: object,
    ) -> DLQEntry:
        """Create from BatchItem with full context."""
        return cls(
            batch_id=batch_id,
            item_index=item.index,
            tool_name=tool_name,
            params=orjson.dumps(params.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS),
            error=item.error or ErrorTrace(message="Unknown error"),
            attempts=attempts,
            metadata=dict(metadata) if metadata else {},
        )
    
    def to_dict(self) -> JsonDict:
        """Serialize for storage/transport."""
        return {
            "batch_id": self.batch_id,
            "item_index": self.item_index,
            "tool_name": self.tool_name,
            "params": self.params_dict,
            "error": self.error.model_dump(),
            "attempts": self.attempts,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DLQ Store Protocol
# ═══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class DLQStore(Protocol):
    """Protocol for DLQ storage backends.
    
    Implementations can use memory, Redis, databases, message queues, etc.
    """
    
    def put(self, entry: DLQEntry) -> None:
        """Add entry to DLQ."""
        ...
    
    def get(self, key: str) -> DLQEntry | None:
        """Get entry by key (batch_id:item_index)."""
        ...
    
    def pop(self, key: str) -> DLQEntry | None:
        """Remove and return entry."""
        ...
    
    def list(self, *, batch_id: str | None = None, tool_name: str | None = None, limit: int = 100) -> list[DLQEntry]:
        """List entries with optional filtering."""
        ...
    
    def count(self) -> int:
        """Total entries in DLQ."""
        ...
    
    def clear(self, *, batch_id: str | None = None) -> int:
        """Clear entries, optionally filtered by batch_id. Returns count removed."""
        ...


class MemoryDLQStore:
    """Thread-safe in-memory DLQ store.
    
    Suitable for testing and single-instance deployments.
    For production distributed systems, use Redis/database backed stores.
    
    Example:
        >>> store = MemoryDLQStore(max_entries=1000)
        >>> store.put(DLQEntry(...))
        >>> entries = store.list(tool_name="http_tool")
    """
    
    __slots__ = ("_entries", "_max_entries")
    
    def __init__(self, max_entries: int = 10_000) -> None:
        self._entries: dict[str, DLQEntry] = {}
        self._max_entries = max_entries
    
    def put(self, entry: DLQEntry) -> None:
        if len(self._entries) >= self._max_entries:
            self._evict_oldest()
        self._entries[entry.key] = entry
    
    def get(self, key: str) -> DLQEntry | None:
        return self._entries.get(key)
    
    def pop(self, key: str) -> DLQEntry | None:
        return self._entries.pop(key, None)
    
    def list(self, *, batch_id: str | None = None, tool_name: str | None = None, limit: int = 100) -> list[DLQEntry]:
        entries = (
            e for e in sorted(self._entries.values(), key=lambda x: -x.timestamp)
            if (batch_id is None or e.batch_id == batch_id) and (tool_name is None or e.tool_name == tool_name)
        )
        return list(entries)[:limit]
    
    def count(self) -> int:
        return len(self._entries)
    
    def clear(self, *, batch_id: str | None = None) -> int:
        if batch_id is None:
            count = len(self._entries)
            self._entries.clear()
            return count
        keys = [k for k, v in self._entries.items() if v.batch_id == batch_id]
        for k in keys:
            del self._entries[k]
        return len(keys)
    
    def _evict_oldest(self) -> None:
        """Remove oldest 10% when at capacity."""
        if not self._entries:
            return
        sorted_keys = sorted(self._entries, key=lambda k: self._entries[k].timestamp)
        for k in sorted_keys[:max(1, len(sorted_keys) // 10)]:
            del self._entries[k]


# ═══════════════════════════════════════════════════════════════════════════════
# DLQ Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Type alias for DLQ callback
DLQCallback = Callable[[DLQEntry], Awaitable[None]]


class DLQConfig(BaseModel):
    """Configuration for Dead Letter Queue handling.
    
    Integrates with batch execution to route poison messages to DLQ
    after exceeding failure thresholds.
    
    Attributes:
        max_poison_threshold: Consecutive failures before DLQ (default: 3)
        enabled: Whether DLQ routing is active
        include_params: Store full params in DLQ entry
        include_trace: Store full error trace
    
    Example:
        >>> config = DLQConfig(max_poison_threshold=5)
        >>> batch_config = IdempotentBatchConfig(dlq=config)
    """
    
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)
    
    max_poison_threshold: Annotated[int, Field(ge=1, le=100)] = 3
    enabled: bool = True
    include_params: bool = True
    include_trace: bool = True
    
    def should_dlq(self, consecutive_failures: int) -> bool:
        """Check if item should be routed to DLQ."""
        return self.enabled and consecutive_failures >= self.max_poison_threshold


# Default disabled config
NO_DLQ = DLQConfig(enabled=False)

# Default store singleton
_default_store: DLQStore | None = None


def get_dlq_store() -> DLQStore:
    """Get global DLQ store (creates MemoryDLQStore if unset)."""
    global _default_store
    if _default_store is None:
        _default_store = MemoryDLQStore()
    return _default_store


def set_dlq_store(store: DLQStore) -> None:
    """Set custom DLQ store backend."""
    global _default_store
    _default_store = store


def reset_dlq_store() -> None:
    """Reset global DLQ store (useful for testing)."""
    global _default_store
    if _default_store is not None:
        _default_store.clear()
    _default_store = None


# ═══════════════════════════════════════════════════════════════════════════════
# DLQ Utilities
# ═══════════════════════════════════════════════════════════════════════════════


async def route_to_dlq(
    item: BatchItem[BaseModel],
    batch_id: str,
    tool_name: str,
    params: BaseModel,
    attempts: int,
    store: DLQStore | None = None,
    callback: DLQCallback | None = None,
    **metadata: object,
) -> DLQEntry:
    """Route failed item to DLQ with optional callback.
    
    Args:
        item: Failed BatchItem
        batch_id: Batch identifier
        tool_name: Tool name
        params: Original parameters
        attempts: Total attempts made
        store: DLQ store (uses global if None)
        callback: Optional async callback for notifications
        **metadata: Additional context
    
    Returns:
        Created DLQEntry
    
    Example:
        >>> async def notify_slack(entry: DLQEntry) -> None:
        ...     await post_to_slack(f"DLQ: {entry.tool_name} failed {entry.attempts}x")
        >>> entry = await route_to_dlq(item, "batch-123", "http", params, 3, callback=notify_slack)
    """
    entry = DLQEntry.from_item(item, batch_id, tool_name, params, attempts, **metadata)
    (store or get_dlq_store()).put(entry)
    if callback:
        await callback(entry)
    return entry


def reprocess_entry(entry: DLQEntry) -> tuple[str, int, JsonDict]:
    """Extract reprocessing info from DLQ entry.
    
    Returns:
        Tuple of (tool_name, item_index, params_dict) for retry.
    """
    return entry.tool_name, entry.item_index, entry.params_dict
