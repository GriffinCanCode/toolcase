"""Batch execution engine for tools.

Provides intelligent batching with:
- Configurable concurrency limits
- Partial failure handling (fail-fast or continue)
- Result aggregation with indices
- Integration with tool caching and retry
- **Streaming progress** via `batch_execute_stream()` for real-time visibility

Design: Uses existing concurrency primitives (map_async, semaphore)
for efficient parallel execution while preserving result ordering.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Callable, Generic, Iterator, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from toolcase.foundation.errors import ErrorCode, ErrorTrace, Result, ToolResult
from toolcase.foundation.errors.result import _ERR, _OK
from toolcase.runtime.concurrency import run_sync

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool

T = TypeVar("T", bound=BaseModel)


def _err_trace(msg: str, code: ErrorCode, recoverable: bool = True) -> ToolResult:
    """Create error Result with ErrorTrace."""
    return Result(ErrorTrace(message=msg, error_code=code.value, recoverable=recoverable), _ERR)


@dataclass(frozen=True, slots=True)
class BatchItem(Generic[T]):
    """Single item result from batch execution."""
    index: int
    result: ToolResult
    elapsed_ms: float
    
    @property
    def is_ok(self) -> bool: return self.result.is_ok()
    
    @property
    def is_err(self) -> bool: return self.result.is_err()
    
    @property
    def value(self) -> str: return self.result.unwrap() if self.is_ok else ""
    
    @property
    def error(self) -> ErrorTrace | None: return self.result.unwrap_err() if self.is_err else None


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming Event Types
# ═══════════════════════════════════════════════════════════════════════════════


class BatchEventKind(StrEnum):
    """Event types for batch streaming."""
    START = "start"       # Batch started
    ITEM = "item"         # Single item completed
    COMPLETE = "complete" # Batch finished


@dataclass(slots=True, frozen=True)
class BatchItemEvent(Generic[T]):
    """Streaming event emitted during batch execution.
    
    Provides real-time visibility into batch progress. Yielded by
    `batch_execute_stream()` as each item completes.
    
    Attributes:
        kind: Event type (start, item, complete)
        item: BatchItem for ITEM events, None otherwise
        completed: Count of completed items
        total: Total items in batch
        elapsed_ms: Time since batch start
        batch_result: Final BatchResult for COMPLETE events only
    
    Example:
        >>> async for event in batch_execute_stream(tool, params):
        ...     if event.kind == BatchEventKind.ITEM:
        ...         print(f"[{event.completed}/{event.total}] {event.progress:.0%}")
    """
    kind: BatchEventKind
    completed: int
    total: int
    elapsed_ms: float
    item: BatchItem[T] | None = None
    batch_result: "BatchResult[T] | None" = None
    timestamp: float = field(default_factory=lambda: time.time() * 1000)
    
    @property
    def progress(self) -> float:
        """Completion percentage (0.0 to 1.0)."""
        return self.completed / self.total if self.total else 0.0
    
    @property
    def is_complete(self) -> bool:
        return self.kind == BatchEventKind.COMPLETE


# Factory functions for clean event creation
def _batch_start(total: int) -> BatchItemEvent[BaseModel]:
    return BatchItemEvent(BatchEventKind.START, 0, total, 0.0)

def _batch_item(item: BatchItem[T], completed: int, total: int, elapsed_ms: float) -> BatchItemEvent[T]:
    return BatchItemEvent(BatchEventKind.ITEM, completed, total, elapsed_ms, item=item)

def _batch_complete(result: "BatchResult[T]", total: int, elapsed_ms: float) -> BatchItemEvent[T]:
    return BatchItemEvent(BatchEventKind.COMPLETE, total, total, elapsed_ms, batch_result=result)


@dataclass(slots=True)
class BatchResult(Generic[T]):
    """Aggregated results from batch execution. Provides access to successes, failures, and metrics."""
    items: list[BatchItem[T]]
    total_ms: float
    concurrency: int
    
    @property
    def successes(self) -> list[BatchItem[T]]: return [i for i in self.items if i.is_ok]
    
    @property
    def failures(self) -> list[BatchItem[T]]: return [i for i in self.items if i.is_err]
    
    @property
    def success_rate(self) -> float: return len(self.successes) / len(self.items) if self.items else 0.0
    
    @property
    def all_ok(self) -> bool: return all(i.is_ok for i in self.items)
    
    @property
    def all_err(self) -> bool: return all(i.is_err for i in self.items)
    
    def values(self) -> list[str]: return [i.value for i in self.items if i.is_ok]
    
    def errors(self) -> list[ErrorTrace]: return [e for i in self.items if i.is_err and (e := i.error)]
    
    def to_result(self) -> Result[list[str], list[ErrorTrace]]:
        """Convert to Result: Ok if all succeeded, Err with all errors otherwise."""
        return Result(self.values(), _OK) if self.all_ok else Result(self.errors(), _ERR)
    
    def __len__(self) -> int: return len(self.items)
    
    def __iter__(self) -> Iterator[BatchItem[T]]: return iter(self.items)


class BatchConfig(BaseModel):
    """Configuration for batch execution.
    
    Example:
        >>> config = BatchConfig(concurrency=10, fail_fast=False)
        >>> results = await batch_execute(tool, params_list, config)
    """
    
    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_default=True,
        json_schema_extra={"title": "Batch Configuration", "examples": [{"concurrency": 10, "fail_fast": False}]},
    )
    
    concurrency: Annotated[int | None, Field(ge=1, le=1000)] = 10
    fail_fast: bool = False
    timeout_per_item: Annotated[float | None, Field(ge=0.1, le=300.0)] = None
    preserve_order: bool = True
    on_item_complete: Callable[[BatchItem[BaseModel]], None] | None = Field(default=None, exclude=True)
    
    def __hash__(self) -> int: return hash((self.concurrency, self.fail_fast, self.timeout_per_item, self.preserve_order))


DEFAULT_BATCH_CONFIG = BatchConfig()


async def batch_execute(
    tool: BaseTool[T],
    params_list: list[T],
    config: BatchConfig | None = None,
) -> BatchResult[T]:
    """Execute tool against multiple parameter sets concurrently.
    
    Example:
        >>> urls = [HttpParams(url=u) for u in ["https://a.com", "https://b.com"]]
        >>> results = await batch_execute(http_tool, urls, BatchConfig(concurrency=5))
        >>> print(f"Success rate: {results.success_rate:.0%}")
    """
    if not params_list:
        return BatchResult([], 0.0, 0)
    
    cfg, start = config or DEFAULT_BATCH_CONFIG, time.perf_counter()
    n, sem, cancel = len(params_list), asyncio.Semaphore(config.concurrency if config and config.concurrency else len(params_list)), asyncio.Event()
    items: list[BatchItem[T]] = [BatchItem(i, Result("", _OK), 0.0) for i in range(n)]
    
    async def run_one(idx: int, params: T) -> BatchItem[T]:
        if cancel.is_set():
            return BatchItem(idx, _err_trace("Batch cancelled due to fail_fast", ErrorCode.CANCELLED, False), 0.0)
        
        async with sem:
            t0 = time.perf_counter()
            try:
                result = await (asyncio.wait_for(tool._async_run_result(params), cfg.timeout_per_item) 
                               if cfg.timeout_per_item else tool._async_run_result(params))
            except asyncio.TimeoutError:
                result = _err_trace(f"Timeout after {cfg.timeout_per_item}s", ErrorCode.TIMEOUT)
            except asyncio.CancelledError:
                result = _err_trace("Execution cancelled", ErrorCode.CANCELLED, False)
            except Exception as e:
                result = _err_trace(str(e), ErrorCode.UNKNOWN)
            
            item = BatchItem(idx, result, (time.perf_counter() - t0) * 1000)
            if cfg.fail_fast and result.is_err(): cancel.set()
            if cfg.on_item_complete: cfg.on_item_complete(item)
            return item
    
    tasks = [asyncio.create_task(run_one(i, p)) for i, p in enumerate(params_list)]
    try:
        completed = await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        for t in tasks: t.cancel()
        raise
    
    # Process results, handling any exceptions from gather
    for i, r in enumerate(completed):
        items[i] = r if isinstance(r, BatchItem) else BatchItem(i, _err_trace(str(r), ErrorCode.UNKNOWN), 0.0)
    
    return BatchResult(items, (time.perf_counter() - start) * 1000, cfg.concurrency or n)


async def batch_execute_stream(
    tool: BaseTool[T],
    params_list: list[T],
    config: BatchConfig | None = None,
) -> AsyncIterator[BatchItemEvent[T]]:
    """Stream batch execution progress, yielding events as items complete.
    
    Unlike `batch_execute()` which returns after completion, this yields
    `BatchItemEvent` in real-time, enabling progress UI updates for large batches.
    
    Events are yielded in completion order (not original index order).
    The final COMPLETE event includes the full BatchResult with preserved order.
    
    Args:
        tool: Tool to execute
        params_list: List of parameter sets
        config: Batch configuration
    
    Yields:
        BatchItemEvent: START → ITEM (per completion) → COMPLETE
    
    Example:
        >>> async for event in batch_execute_stream(http_tool, urls):
        ...     match event.kind:
        ...         case BatchEventKind.START:
        ...             print(f"Starting {event.total} items...")
        ...         case BatchEventKind.ITEM:
        ...             status = "✓" if event.item.is_ok else "✗"
        ...             print(f"[{event.completed}/{event.total}] {status} idx={event.item.index}")
        ...         case BatchEventKind.COMPLETE:
        ...             print(f"Done: {event.batch_result.success_rate:.0%} success")
    """
    if not params_list:
        yield _batch_start(0)
        yield _batch_complete(BatchResult([], 0.0, 0), 0, 0.0)
        return
    
    cfg, start, n = config or DEFAULT_BATCH_CONFIG, time.perf_counter(), len(params_list)
    sem, cancel = asyncio.Semaphore(cfg.concurrency or n), asyncio.Event()
    items: list[BatchItem[T]] = [BatchItem(i, Result("", _OK), 0.0) for i in range(n)]
    queue: asyncio.Queue[BatchItem[T]] = asyncio.Queue()
    
    yield _batch_start(n)
    
    async def run_one(idx: int, params: T) -> None:
        if cancel.is_set():
            item = BatchItem(idx, _err_trace("Batch cancelled due to fail_fast", ErrorCode.CANCELLED, False), 0.0)
            await queue.put(item)
            return
        
        async with sem:
            t0 = time.perf_counter()
            try:
                result = await (asyncio.wait_for(tool._async_run_result(params), cfg.timeout_per_item)
                               if cfg.timeout_per_item else tool._async_run_result(params))
            except asyncio.TimeoutError:
                result = _err_trace(f"Timeout after {cfg.timeout_per_item}s", ErrorCode.TIMEOUT)
            except asyncio.CancelledError:
                result = _err_trace("Execution cancelled", ErrorCode.CANCELLED, False)
            except Exception as e:
                result = _err_trace(str(e), ErrorCode.UNKNOWN)
            
            item = BatchItem(idx, result, (time.perf_counter() - t0) * 1000)
            if cfg.fail_fast and result.is_err():
                cancel.set()
            if cfg.on_item_complete:
                cfg.on_item_complete(item)
            await queue.put(item)
    
    # Start all tasks
    tasks = [asyncio.create_task(run_one(i, p)) for i, p in enumerate(params_list)]
    
    # Yield events as items complete
    completed_count = 0
    try:
        while completed_count < n:
            item = await queue.get()
            items[item.index] = item
            completed_count += 1
            yield _batch_item(item, completed_count, n, (time.perf_counter() - start) * 1000)
    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        raise
    
    # Yield final result
    result = BatchResult(items, (time.perf_counter() - start) * 1000, cfg.concurrency or n)
    yield _batch_complete(result, n, result.total_ms)


def batch_execute_sync(tool: BaseTool[T], params_list: list[T], config: BatchConfig | None = None) -> BatchResult[T]:
    """Synchronous batch execution. Wraps async batch_execute for sync contexts."""
    return run_sync(batch_execute(tool, params_list, config))
