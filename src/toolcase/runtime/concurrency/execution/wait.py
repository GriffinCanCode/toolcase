"""Wait strategies for concurrent operations.

Provides various patterns for waiting on multiple async operations:
    - race: First to complete wins, cancel others
    - gather: Wait for all, fail fast on first error
    - gather_settled: Wait for all regardless of errors
    - first_success: First successful result, ignore failures
    - map_async: Parallel map with concurrency limit
    - all_settled: All results with success/failure status

Example:
    >>> # Race multiple providers
    >>> result = await race(
    ...     fetch_from_api_a(),
    ...     fetch_from_api_b(),
    ...     fetch_from_cache(),
    ... )
    
    >>> # Parallel processing with limit
    >>> results = await map_async(process_item, items, limit=10)
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Generic, TypeVar, overload

from toolcase.runtime.concurrency.primitives.task import _ExceptionGroup

T = TypeVar("T")
U = TypeVar("U")

__all__ = [
    "SettledStatus",
    "Settled",
    "WaitResult",
    "race",
    "race_with_index",
    "gather",
    "gather_settled",
    "all_settled",
    "first_success",
    "map_async",
    "map_async_unordered",
    "wait_any",
    "wait_all",
    "retry_until_success",
]


class SettledStatus(StrEnum):
    """Status of a settled operation."""
    FULFILLED = "fulfilled"
    REJECTED = "rejected"


@dataclass(slots=True, frozen=True)
class Settled(Generic[T]):
    """Result of a settled operation (success or failure). Like JS Promise.allSettled().
    
    Attributes:
        status: 'fulfilled' or 'rejected'
        value: Result value if fulfilled
        error: Exception if rejected
    """
    
    status: SettledStatus
    value: T | None = None
    error: BaseException | None = None
    
    @property
    def is_fulfilled(self) -> bool: return self.status == SettledStatus.FULFILLED
    
    @property
    def is_rejected(self) -> bool: return self.status == SettledStatus.REJECTED
    
    def unwrap(self) -> T:
        """Get value or raise stored error."""
        if self.is_rejected: raise self.error or RuntimeError("Rejected with no error")
        return self.value  # type: ignore[return-value]
    
    def unwrap_or(self, default: T) -> T:
        """Get value or return default."""
        return self.value if self.is_fulfilled else default  # type: ignore[return-value]


def _fulfilled(value: T) -> Settled[T]: return Settled(SettledStatus.FULFILLED, value=value)
def _rejected(error: BaseException) -> Settled[T]: return Settled(SettledStatus.REJECTED, error=error)


@dataclass(slots=True)
class WaitResult(Generic[T]):
    """Result of a wait operation with metadata (value, index, elapsed, cancelled count)."""
    value: T
    index: int = 0
    elapsed: float = 0.0
    cancelled: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Race: First to complete wins
# ─────────────────────────────────────────────────────────────────────────────

async def _cancel_tasks(tasks: set[asyncio.Task[T]] | list[asyncio.Task[T]]) -> None:
    """Cancel tasks and wait for cancellation to complete."""
    for t in tasks: t.cancel()
    if tasks: await asyncio.gather(*tasks, return_exceptions=True)


async def race(*coros: Awaitable[T], timeout: float | None = None) -> T:
    """Race multiple coroutines - first to complete wins.
    
    Cancels all remaining coroutines after first completes.
    If the first to complete raises, that exception propagates.
    
    Args:
        *coros: Coroutines to race
        timeout: Optional timeout for all operations
    
    Returns:
        Result from first completing coroutine
    
    Raises:
        ValueError: If no coroutines provided
        asyncio.TimeoutError: If timeout expires
        Exception: If first completing coroutine raises
    
    Example:
        >>> result = await race(
        ...     fetch_from_api_a(),
        ...     fetch_from_api_b(),
        ...     fetch_from_cache(),
        ... )
    """
    if not coros: raise ValueError("race() requires at least one coroutine")
    tasks = [asyncio.ensure_future(c) for c in coros]
    
    try:
        done, pending = await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if not done: raise asyncio.TimeoutError()
        await _cancel_tasks(pending)
        return next(iter(done)).result()
    except asyncio.CancelledError:
        await _cancel_tasks(tasks)
        raise


async def race_with_index(*coros: Awaitable[T], timeout: float | None = None) -> WaitResult[T]:
    """Race with metadata about which operation won. Like race() but returns WaitResult with index."""
    import time
    if not coros: raise ValueError("race_with_index() requires at least one coroutine")
    start, tasks = time.monotonic(), [asyncio.ensure_future(c) for c in coros]
    idx = {id(t): i for i, t in enumerate(tasks)}
    try:
        done, pending = await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if not done: raise asyncio.TimeoutError()
        await _cancel_tasks(pending)
        w = next(iter(done))
        return WaitResult(w.result(), idx[id(w)], time.monotonic() - start, len(pending))
    except asyncio.CancelledError:
        await _cancel_tasks(tasks)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Gather: Wait for all
# ─────────────────────────────────────────────────────────────────────────────

async def gather(*coros: Awaitable[T], return_exceptions: bool = False) -> list[T | BaseException]:
    """Gather results from all coroutines. Thin wrapper around asyncio.gather.
    
    Args:
        *coros: Coroutines to execute
        return_exceptions: If True, exceptions are returned instead of raised
    
    Returns:
        List of results in same order as input
    
    Example:
        >>> results = await gather(
        ...     fetch_user(1),
        ...     fetch_user(2),
        ...     fetch_user(3),
        ... )
    """
    return await asyncio.gather(*coros, return_exceptions=return_exceptions)  # type: ignore[return-value]


async def gather_settled(*coros: Awaitable[T]) -> list[Settled[T]]:
    """Gather all results, never raising. Like Promise.allSettled().
    
    Args:
        *coros: Coroutines to execute
    
    Returns:
        List of Settled results in same order
    
    Example:
        >>> results = await gather_settled(
        ...     risky_operation_1(),
        ...     risky_operation_2(),
        ... )
        >>> for r in results:
        ...     if r.is_fulfilled:
        ...         print(f"Success: {r.value}")
        ...     else:
        ...         print(f"Failed: {r.error}")
    """
    results = await asyncio.gather(*(asyncio.ensure_future(c) for c in coros), return_exceptions=True)
    return [_rejected(r) if isinstance(r, BaseException) else _fulfilled(r) for r in results]


# Alias for compatibility
all_settled = gather_settled


# ─────────────────────────────────────────────────────────────────────────────
# First Success: First non-error wins
# ─────────────────────────────────────────────────────────────────────────────

async def first_success(*coros: Awaitable[T], timeout: float | None = None) -> T:
    """Get first successful result, ignoring failures.
    
    Unlike race(), continues to next operation if one fails.
    Only raises if ALL operations fail.
    
    Args:
        *coros: Coroutines to try
        timeout: Overall timeout for all attempts
    
    Returns:
        First successful result
    
    Raises:
        ExceptionGroup: If all operations fail
        asyncio.TimeoutError: If timeout expires
    
    Example:
        >>> # Try multiple providers, first success wins
        >>> result = await first_success(
        ...     unreliable_api_a(),
        ...     unreliable_api_b(),
        ...     fallback_api(),
        ... )
    """
    if not coros: raise ValueError("first_success() requires at least one coroutine")
    tasks, errors, pending = [asyncio.ensure_future(c) for c in coros], [], None
    pending, deadline = set(tasks), (asyncio.get_event_loop().time() + timeout) if timeout else None
    try:
        while pending:
            remaining = (deadline - asyncio.get_event_loop().time()) if deadline else None
            if remaining is not None and remaining <= 0: raise asyncio.TimeoutError()
            done, pending = await asyncio.wait(pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
            if not done and not pending: raise asyncio.TimeoutError()
            for task in done:
                try:
                    await _cancel_tasks(pending)
                    return task.result()
                except Exception as e: errors.append(e)
        raise _ExceptionGroup(f"All {len(coros)} operations failed", [e for e in errors if isinstance(e, Exception)])
    except asyncio.CancelledError:
        await _cancel_tasks(tasks)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Map Async: Parallel map with concurrency control
# ─────────────────────────────────────────────────────────────────────────────

@overload
async def map_async(
    func: Callable[[T], Awaitable[U]], items: list[T], *, limit: int | None = None, return_exceptions: bool = False
) -> list[U]: ...

@overload
async def map_async(
    func: Callable[[T], Awaitable[U]], items: list[T], *, limit: int | None = None, return_exceptions: bool = True
) -> list[U | BaseException]: ...

async def map_async(
    func: Callable[[T], Awaitable[U]], items: list[T], *, limit: int | None = None, return_exceptions: bool = False
) -> list[U] | list[U | BaseException]:
    """Apply async function to items with concurrency limit. Like map() for async with resource control.
    
    Args:
        func: Async function to apply
        items: Items to process
        limit: Maximum concurrent operations (None = unlimited)
        return_exceptions: If True, return exceptions instead of raising
    
    Returns:
        Results in same order as input items
    
    Example:
        >>> results = await map_async(fetch_data, urls, limit=10)
    """
    if not items: return []
    if not limit or limit >= len(items):
        return await asyncio.gather(*[func(item) for item in items], return_exceptions=return_exceptions)  # type: ignore[return-value]
    sem, out = asyncio.Semaphore(limit), [None] * len(items)
    async def call(i: int, x: T) -> None:
        async with sem:
            try: out[i] = await func(x)
            except BaseException as e:
                if return_exceptions: out[i] = e
                else: raise
    await asyncio.gather(*(call(i, x) for i, x in enumerate(items)), return_exceptions=return_exceptions)
    return out  # type: ignore[return-value]


async def map_async_unordered(func: Callable[[T], Awaitable[U]], items: list[T], *, limit: int | None = None) -> list[U]:
    """Map async function, yielding results as they complete. Doesn't preserve order."""
    if not items: return []
    sem = asyncio.Semaphore(limit or len(items))
    async def call(x: T) -> U:
        async with sem: return await func(x)
    return [await t for t in asyncio.as_completed([asyncio.ensure_future(call(x)) for x in items])]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

async def wait_any(*coros: Awaitable[T], timeout: float | None = None) -> tuple[set[asyncio.Task[T]], set[asyncio.Task[T]]]:
    """Wait for any coroutine to complete. Lower-level than race(), doesn't cancel pending."""
    if not coros: return set(), set()
    return await asyncio.wait({asyncio.ensure_future(c) for c in coros}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)


async def wait_all(*coros: Awaitable[T], timeout: float | None = None) -> tuple[set[asyncio.Task[T]], set[asyncio.Task[T]]]:
    """Wait for all coroutines to complete. Lower-level wrapper around asyncio.wait."""
    if not coros: return set(), set()
    return await asyncio.wait({asyncio.ensure_future(c) for c in coros}, timeout=timeout, return_when=asyncio.ALL_COMPLETED)


async def retry_until_success(
    coro_factory: Callable[[], Awaitable[T]], max_attempts: int = 3, delay: float = 1.0,
    backoff: float = 2.0, exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Retry coroutine until success.
    
    Args:
        coro_factory: Callable that creates the coroutine (called each attempt)
        max_attempts: Maximum attempts before giving up
        delay: Initial delay between attempts
        backoff: Delay multiplier for each retry
        exceptions: Exception types to catch and retry
    
    Returns:
        Successful result
    
    Raises:
        Last exception if all attempts fail
    """
    err, d = None, delay
    for i in range(max_attempts):
        try: return await coro_factory()
        except exceptions as e:
            err, d = e, d * backoff
            if i < max_attempts - 1: await asyncio.sleep(d / backoff)
    raise err or RuntimeError("All retry attempts failed")
