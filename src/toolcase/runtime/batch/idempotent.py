"""Batch-level retry with idempotency for exactly-once semantics.

Provides:
- IdempotencyStore: Protocol for idempotency key storage
- CacheIdempotencyAdapter: Adapts existing ToolCache for idempotency
- BatchRetryPolicy: Batch-level retry configuration
- batch_execute_idempotent(): Main entry point with exactly-once guarantees

Design: Extends existing batch infrastructure with batch-level retry and
idempotency keys. Integrates with existing cache system (MemoryCache, Redis,
Memcached) for storage. Individual items get unique keys; successful results
are cached to prevent re-execution on batch retry.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Protocol, runtime_checkable

import orjson
from pydantic import BaseModel, ConfigDict, Field

from toolcase.foundation.errors import ErrorCode, Result
from toolcase.foundation.errors.result import _ERR, _OK
from toolcase.io.cache import MemoryCache, ToolCache, get_cache
from toolcase.runtime.concurrency import run_sync
from toolcase.runtime.retry import Backoff, ExponentialBackoff

from .batch import BatchConfig, BatchItem, BatchResult, _err_trace

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool
    from toolcase.foundation.errors import ToolResult

# ═══════════════════════════════════════════════════════════════════════════════
# Idempotency Store Protocol & Adapter
# ═══════════════════════════════════════════════════════════════════════════════

# Separator for idempotency metadata in cached value
_IDEM_SEP = "\x00\x01"


@runtime_checkable
class IdempotencyStore(Protocol):
    """Protocol for idempotency key storage backends."""
    
    def get(self, key: str) -> tuple[str, bool] | None:
        """Get (result, is_ok) by idempotency key. Returns None if not found."""
        ...
    
    def set(self, key: str, result: str, is_ok: bool, ttl: float) -> None:
        """Store result with idempotency key."""
        ...
    
    def delete(self, key: str) -> bool:
        """Delete stored result. Returns True if existed."""
        ...
    
    def clear(self, prefix: str | None = None) -> int:
        """Clear stored results. Returns count deleted."""
        ...


class CacheIdempotencyAdapter:
    """Adapter that uses existing ToolCache for idempotency storage.
    
    Bridges the ToolCache interface to IdempotencyStore protocol,
    allowing reuse of existing cache infrastructure (Memory, Redis, Memcached).
    
    Stores result + is_ok flag as a single value with separator.
    
    Args:
        cache: Existing ToolCache instance (or uses global cache if None)
        prefix: Key prefix for idempotency entries
    
    Example:
        >>> from toolcase import get_cache, MemoryCache
        >>> adapter = CacheIdempotencyAdapter(get_cache())
        >>> # Or with custom cache:
        >>> adapter = CacheIdempotencyAdapter(MemoryCache(default_ttl=3600))
    """
    
    __slots__ = ("_cache", "_prefix")
    
    def __init__(self, cache: ToolCache | None = None, prefix: str = "idem") -> None:
        self._cache = cache or get_cache()
        self._prefix = prefix
    
    def _encode(self, result: str, is_ok: bool) -> str:
        """Encode result and status into single cache value."""
        return f"{'1' if is_ok else '0'}{_IDEM_SEP}{result}"
    
    def _decode(self, value: str) -> tuple[str, bool]:
        """Decode cache value into (result, is_ok)."""
        flag, _, result = value.partition(_IDEM_SEP)
        return result, flag == "1"
    
    def get(self, key: str) -> tuple[str, bool] | None:
        if (value := self._cache.get(self._prefix, {key: key})) is not None:
            return self._decode(value)
        return None
    
    def set(self, key: str, result: str, is_ok: bool, ttl: float) -> None:
        self._cache.set(self._prefix, {key: key}, self._encode(result, is_ok), ttl)
    
    def delete(self, key: str) -> bool:
        return self._cache.invalidate(self._prefix, {key: key})
    
    def clear(self, prefix: str | None = None) -> int:
        # Clear all idempotency entries
        return self._cache.invalidate_tool(self._prefix)
    
    @classmethod
    def from_cache(cls, cache: ToolCache, prefix: str = "idem") -> CacheIdempotencyAdapter:
        """Create adapter from existing cache instance."""
        return cls(cache, prefix)


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Retry Policy
# ═══════════════════════════════════════════════════════════════════════════════


class BatchRetryStrategy(StrEnum):
    """Strategy for batch-level retries."""
    FAILED_ONLY = "failed_only"      # Retry only failed items
    ENTIRE_BATCH = "entire_batch"    # Retry entire batch on threshold breach


class BatchRetryPolicy(BaseModel):
    """Configuration for batch-level retry behavior.
    
    Determines when and how to retry at the batch level (vs item level).
    Use with IdempotentBatchConfig for exactly-once semantics.
    
    Attributes:
        max_retries: Maximum batch-level retry attempts
        backoff: Backoff strategy for delay calculation
        strategy: Retry failed items only or entire batch
        failure_threshold: Min failure rate to trigger batch retry (0.0-1.0)
        retryable_codes: Error codes that trigger batch retry
    
    Example:
        >>> policy = BatchRetryPolicy(
        ...     max_retries=3,
        ...     failure_threshold=0.5,  # Retry if >50% failed
        ...     strategy=BatchRetryStrategy.FAILED_ONLY,
        ... )
    """
    
    model_config = ConfigDict(
        frozen=True, validate_default=True, extra="forbid",
        arbitrary_types_allowed=True,  # For Backoff protocol
        json_schema_extra={"title": "Batch Retry Policy"},
    )
    
    max_retries: Annotated[int, Field(ge=0, le=10)] = 3
    backoff: Backoff = Field(default_factory=ExponentialBackoff, repr=False)
    strategy: BatchRetryStrategy = BatchRetryStrategy.FAILED_ONLY
    failure_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    retryable_codes: frozenset[ErrorCode] = frozenset({
        ErrorCode.RATE_LIMITED, ErrorCode.TIMEOUT, ErrorCode.NETWORK_ERROR,
    })
    
    def should_retry_batch(self, result: BatchResult[BaseModel], attempt: int) -> bool:
        """Determine if batch should be retried based on failure pattern."""
        if attempt >= self.max_retries or result.all_ok:
            return False
        failure_rate = len(result.failures) / len(result.items) if result.items else 0.0
        if failure_rate < self.failure_threshold:
            return False
        return any(
            item.error and item.error.error_code in {c.value for c in self.retryable_codes}
            for item in result.failures
        )
    
    def get_delay(self, attempt: int) -> float:
        """Get delay before next batch retry."""
        return self.backoff.delay(attempt)
    
    def __hash__(self) -> int:
        return hash((self.max_retries, self.strategy, self.failure_threshold))


NO_BATCH_RETRY = BatchRetryPolicy(max_retries=0, failure_threshold=1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Idempotent Batch Config
# ═══════════════════════════════════════════════════════════════════════════════


def _make_idempotency_key(batch_id: str, tool_name: str, params: BaseModel, index: int) -> str:
    """Generate idempotency key for a batch item."""
    params_bytes = orjson.dumps(params.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS)
    params_hash = hashlib.md5(params_bytes, usedforsecurity=False).hexdigest()[:12]
    return f"{batch_id}:{tool_name}:{index}:{params_hash}"


class IdempotentBatchConfig(BatchConfig):
    """Extended batch config with idempotency and batch-level retry.
    
    Adds exactly-once semantics via idempotency keys and batch-level
    retry for transient failures. Integrates with existing cache system.
    
    Attributes:
        batch_id: Unique batch identifier (auto-generated if None)
        retry_policy: Batch-level retry configuration
        idempotency_ttl: TTL for idempotency keys (seconds)
        skip_cached: Use cached results for already-executed items
    
    Example:
        >>> config = IdempotentBatchConfig(
        ...     concurrency=10,
        ...     batch_id="order-batch-123",
        ...     retry_policy=BatchRetryPolicy(max_retries=3),
        ...     idempotency_ttl=3600,
        ... )
    """
    
    batch_id: str | None = None
    retry_policy: BatchRetryPolicy = Field(default_factory=lambda: NO_BATCH_RETRY)
    idempotency_ttl: Annotated[float, Field(ge=60.0, le=86400.0)] = 3600.0
    skip_cached: bool = True
    
    def __hash__(self) -> int:
        return hash((super().__hash__(), self.batch_id, self.idempotency_ttl, self.skip_cached))


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Result with Retry Metadata
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class IdempotentBatchResult(BatchResult[BaseModel]):
    """Extended batch result with retry and idempotency metadata."""
    batch_id: str = ""
    batch_attempts: int = 1
    from_cache: list[int] = field(default_factory=list)
    retry_history: list[dict[str, int | float]] = field(default_factory=list)
    
    @property
    def cache_hit_rate(self) -> float:
        return len(self.from_cache) / len(self.items) if self.items else 0.0
    
    @property
    def was_retried(self) -> bool:
        return self.batch_attempts > 1


# ═══════════════════════════════════════════════════════════════════════════════
# Core Execution
# ═══════════════════════════════════════════════════════════════════════════════


async def batch_execute_idempotent(
    tool: BaseTool[BaseModel],
    params_list: list[BaseModel],
    config: IdempotentBatchConfig | None = None,
    store: IdempotencyStore | None = None,
    cache: ToolCache | None = None,
) -> IdempotentBatchResult:
    """Execute batch with idempotency keys and batch-level retry.
    
    Provides exactly-once execution semantics via idempotency keys.
    Already-executed items (from prior attempts) return cached results.
    Batch-level retry kicks in when transient failures exceed threshold.
    
    Uses existing cache infrastructure for storage. Pass `cache` to use
    a specific cache (Memory, Redis, Memcached), or omit to use global cache.
    
    Args:
        tool: Tool to execute
        params_list: List of parameter sets
        config: Idempotent batch configuration
        store: Custom IdempotencyStore (overrides cache if provided)
        cache: ToolCache for idempotency storage (uses global cache if None)
    
    Returns:
        IdempotentBatchResult with execution results and metadata
    
    Example:
        >>> # Using global cache (default)
        >>> results = await batch_execute_idempotent(tool, params_list, config)
        >>> 
        >>> # Using specific cache
        >>> from toolcase import MemoryCache
        >>> cache = MemoryCache(default_ttl=3600)
        >>> results = await batch_execute_idempotent(tool, params_list, config, cache=cache)
    """
    if not params_list:
        return IdempotentBatchResult([], 0.0, 0)
    
    cfg = config or IdempotentBatchConfig()
    batch_id = cfg.batch_id or str(uuid.uuid4())[:8]
    # Use provided store, or create adapter from cache
    idem_store = store or CacheIdempotencyAdapter(cache)
    tool_name = tool.metadata.name
    
    n = len(params_list)
    items: list[BatchItem[BaseModel]] = [BatchItem(i, Result("", _OK), 0.0) for i in range(n)]
    from_cache: list[int] = []
    retry_history: list[dict[str, int | float]] = []
    
    # Generate idempotency keys
    keys = [_make_idempotency_key(batch_id, tool_name, p, i) for i, p in enumerate(params_list)]
    
    async def run_with_idempotency(idx: int, params: BaseModel, key: str, sem: asyncio.Semaphore, cancel: asyncio.Event) -> BatchItem[BaseModel]:
        """Execute single item with idempotency check."""
        if cancel.is_set():
            return BatchItem(idx, _err_trace("Batch cancelled", ErrorCode.CANCELLED, False), 0.0)
        
        # Check idempotency store
        if cfg.skip_cached and (cached := idem_store.get(key)):
            from_cache.append(idx)
            result_str, is_ok = cached
            result: ToolResult = Result(result_str, _OK if is_ok else _ERR)
            return BatchItem(idx, result, 0.0)
        
        async with sem:
            t0 = time.perf_counter()
            try:
                result = await (
                    asyncio.wait_for(tool._async_run_result(params), cfg.timeout_per_item)
                    if cfg.timeout_per_item else tool._async_run_result(params)
                )
            except asyncio.TimeoutError:
                result = _err_trace(f"Timeout after {cfg.timeout_per_item}s", ErrorCode.TIMEOUT)
            except asyncio.CancelledError:
                result = _err_trace("Execution cancelled", ErrorCode.CANCELLED, False)
            except Exception as e:
                result = _err_trace(str(e), ErrorCode.UNKNOWN)
            
            elapsed = (time.perf_counter() - t0) * 1000
            
            # Store in idempotency store (only successful or terminal failures)
            if result.is_ok() or (result.is_err() and not _is_retryable_code(result.unwrap_err().error_code, cfg.retry_policy)):
                idem_store.set(
                    key,
                    result.unwrap() if result.is_ok() else result.unwrap_err().message,
                    result.is_ok(),
                    cfg.idempotency_ttl,
                )
            
            item = BatchItem(idx, result, elapsed)
            if cfg.fail_fast and result.is_err():
                cancel.set()
            if cfg.on_item_complete:
                cfg.on_item_complete(item)
            return item
    
    # Execute with batch-level retry
    start = time.perf_counter()
    batch_attempt = 0
    policy = cfg.retry_policy
    
    while True:
        sem = asyncio.Semaphore(cfg.concurrency or n)
        cancel = asyncio.Event()
        
        # Determine which items to execute
        indices_to_run = (
            list(range(n)) if batch_attempt == 0 or policy.strategy == BatchRetryStrategy.ENTIRE_BATCH
            else [i for i in range(n) if items[i].is_err and _is_retryable_code(items[i].error.error_code if items[i].error else None, policy)]
        )
        
        tasks = [asyncio.create_task(run_with_idempotency(i, params_list[i], keys[i], sem, cancel)) for i in indices_to_run]
        
        try:
            completed = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise
        
        for i, r in enumerate(completed):
            idx = indices_to_run[i]
            items[idx] = r if isinstance(r, BatchItem) else BatchItem(idx, _err_trace(str(r), ErrorCode.UNKNOWN), 0.0)
        
        batch_result = IdempotentBatchResult(
            items=items,
            total_ms=(time.perf_counter() - start) * 1000,
            concurrency=cfg.concurrency or n,
            batch_id=batch_id,
            batch_attempts=batch_attempt + 1,
            from_cache=from_cache.copy(),
            retry_history=retry_history.copy(),
        )
        
        # Check if batch-level retry needed
        if not policy.should_retry_batch(batch_result, batch_attempt):
            return batch_result
        
        # Record retry and wait
        delay = policy.get_delay(batch_attempt)
        retry_history.append({
            "attempt": batch_attempt + 1,
            "failures": len(batch_result.failures),
            "delay": delay,
        })
        batch_attempt += 1
        await asyncio.sleep(delay)


def batch_execute_idempotent_sync(
    tool: BaseTool[BaseModel],
    params_list: list[BaseModel],
    config: IdempotentBatchConfig | None = None,
    store: IdempotencyStore | None = None,
    cache: ToolCache | None = None,
) -> IdempotentBatchResult:
    """Synchronous wrapper for batch_execute_idempotent."""
    return run_sync(batch_execute_idempotent(tool, params_list, config, store, cache))


def _is_retryable_code(code: str | None, policy: BatchRetryPolicy) -> bool:
    """Check if error code is retryable per policy."""
    return code in {c.value for c in policy.retryable_codes} if code else False
