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
from collections.abc import Awaitable, Coroutine
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Callable, Generic, TypeVar, overload

if TYPE_CHECKING:
    pass

T = TypeVar("T")
U = TypeVar("U")


class SettledStatus(StrEnum):
    """Status of a settled operation."""
    FULFILLED = "fulfilled"
    REJECTED = "rejected"


@dataclass(slots=True, frozen=True)
class Settled(Generic[T]):
    """Result of a settled operation (success or failure).
    
    Similar to JavaScript's Promise.allSettled() results.
    
    Attributes:
        status: 'fulfilled' or 'rejected'
        value: Result value if fulfilled
        error: Exception if rejected
    """
    
    status: SettledStatus
    value: T | None = None
    error: BaseException | None = None
    
    @property
    def is_fulfilled(self) -> bool:
        return self.status == SettledStatus.FULFILLED
    
    @property
    def is_rejected(self) -> bool:
        return self.status == SettledStatus.REJECTED
    
    def unwrap(self) -> T:
        """Get value or raise stored error."""
        if self.is_rejected:
            raise self.error or RuntimeError("Rejected with no error")
        return self.value  # type: ignore[return-value]
    
    def unwrap_or(self, default: T) -> T:
        """Get value or return default."""
        return self.value if self.is_fulfilled else default  # type: ignore[return-value]


def _fulfilled(value: T) -> Settled[T]:
    """Create a fulfilled Settled."""
    return Settled(SettledStatus.FULFILLED, value=value)


def _rejected(error: BaseException) -> Settled[T]:
    """Create a rejected Settled."""
    return Settled(SettledStatus.REJECTED, error=error)


@dataclass(slots=True)
class WaitResult(Generic[T]):
    """Result of a wait operation with metadata.
    
    Attributes:
        value: The result value
        index: Index of the completing operation (for race/first)
        elapsed: Time taken in seconds
        cancelled: Number of operations cancelled
    """
    
    value: T
    index: int = 0
    elapsed: float = 0.0
    cancelled: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Race: First to complete wins
# ─────────────────────────────────────────────────────────────────────────────

async def race(
    *coros: Awaitable[T],
    timeout: float | None = None,
) -> T:
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
    if not coros:
        raise ValueError("race() requires at least one coroutine")
    
    tasks = [asyncio.ensure_future(c) for c in coros]
    
    try:
        if timeout:
            done, pending = await asyncio.wait(
                tasks,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise asyncio.TimeoutError()
        else:
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
        
        # Cancel pending
        for task in pending:
            task.cancel()
        
        # Wait for cancellations to complete
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        
        # Return first result (may raise)
        winner = next(iter(done))
        return winner.result()
    
    except asyncio.CancelledError:
        # Cancel all on external cancellation
        for task in tasks:
            task.cancel()
        raise


async def race_with_index(
    *coros: Awaitable[T],
    timeout: float | None = None,
) -> WaitResult[T]:
    """Race with metadata about which operation won.
    
    Like race() but returns WaitResult with index of winner.
    """
    import time
    
    if not coros:
        raise ValueError("race_with_index() requires at least one coroutine")
    
    start = time.monotonic()
    tasks = [asyncio.ensure_future(c) for c in coros]
    task_indices = {id(t): i for i, t in enumerate(tasks)}
    
    try:
        if timeout:
            done, pending = await asyncio.wait(
                tasks,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise asyncio.TimeoutError()
        else:
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
        
        # Cancel pending
        for task in pending:
            task.cancel()
        
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        
        winner = next(iter(done))
        return WaitResult(
            value=winner.result(),
            index=task_indices[id(winner)],
            elapsed=time.monotonic() - start,
            cancelled=len(pending),
        )
    
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Gather: Wait for all
# ─────────────────────────────────────────────────────────────────────────────

async def gather(
    *coros: Awaitable[T],
    return_exceptions: bool = False,
) -> list[T | BaseException]:
    """Gather results from all coroutines.
    
    Thin wrapper around asyncio.gather with consistent interface.
    
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
    """Gather all results, never raising.
    
    Like Promise.allSettled() - waits for all operations regardless
    of success or failure, returning status for each.
    
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
    tasks = [asyncio.ensure_future(c) for c in coros]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    settled: list[Settled[T]] = []
    for result in results:
        if isinstance(result, BaseException):
            settled.append(_rejected(result))
        else:
            settled.append(_fulfilled(result))
    
    return settled


# Alias for compatibility
all_settled = gather_settled


# ─────────────────────────────────────────────────────────────────────────────
# First Success: First non-error wins
# ─────────────────────────────────────────────────────────────────────────────

async def first_success(
    *coros: Awaitable[T],
    timeout: float | None = None,
) -> T:
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
    if not coros:
        raise ValueError("first_success() requires at least one coroutine")
    
    tasks = [asyncio.ensure_future(c) for c in coros]
    pending = set(tasks)
    errors: list[BaseException] = []
    
    try:
        deadline = asyncio.get_event_loop().time() + timeout if timeout else None
        
        while pending:
            remaining_timeout = None
            if deadline:
                remaining_timeout = deadline - asyncio.get_event_loop().time()
                if remaining_timeout <= 0:
                    raise asyncio.TimeoutError()
            
            done, pending = await asyncio.wait(
                pending,
                timeout=remaining_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            
            if not done and not pending:
                raise asyncio.TimeoutError()
            
            for task in done:
                try:
                    result = task.result()
                    # Success! Cancel remaining
                    for p in pending:
                        p.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    return result
                except Exception as e:
                    errors.append(e)
        
        # All failed
        raise ExceptionGroup(
            f"All {len(coros)} operations failed",
            [e for e in errors if isinstance(e, Exception)],
        )
    
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Map Async: Parallel map with concurrency control
# ─────────────────────────────────────────────────────────────────────────────

@overload
async def map_async(
    func: Callable[[T], Awaitable[U]],
    items: list[T],
    *,
    limit: int | None = None,
    return_exceptions: bool = False,
) -> list[U]: ...


@overload
async def map_async(
    func: Callable[[T], Awaitable[U]],
    items: list[T],
    *,
    limit: int | None = None,
    return_exceptions: bool = True,
) -> list[U | BaseException]: ...


async def map_async(
    func: Callable[[T], Awaitable[U]],
    items: list[T],
    *,
    limit: int | None = None,
    return_exceptions: bool = False,
) -> list[U] | list[U | BaseException]:
    """Apply async function to items with concurrency limit.
    
    Like map() but for async functions. Controls maximum concurrent
    operations to avoid overwhelming resources.
    
    Args:
        func: Async function to apply
        items: Items to process
        limit: Maximum concurrent operations (None = unlimited)
        return_exceptions: If True, return exceptions instead of raising
    
    Returns:
        Results in same order as input items
    
    Example:
        >>> # Process 100 items, max 10 concurrent
        >>> results = await map_async(
        ...     fetch_data,
        ...     urls,
        ...     limit=10,
        ... )
    """
    if not items:
        return []
    
    if limit is None or limit >= len(items):
        # No limit needed
        coros = [func(item) for item in items]
        return await asyncio.gather(*coros, return_exceptions=return_exceptions)  # type: ignore[return-value]
    
    # Use semaphore for concurrency limiting
    semaphore = asyncio.Semaphore(limit)
    results: list[U | BaseException | None] = [None] * len(items)
    
    async def limited_call(idx: int, item: T) -> None:
        async with semaphore:
            try:
                results[idx] = await func(item)
            except BaseException as e:
                if return_exceptions:
                    results[idx] = e
                else:
                    raise
    
    await asyncio.gather(
        *(limited_call(i, item) for i, item in enumerate(items)),
        return_exceptions=return_exceptions,
    )
    
    return results  # type: ignore[return-value]


async def map_async_unordered(
    func: Callable[[T], Awaitable[U]],
    items: list[T],
    *,
    limit: int | None = None,
) -> list[U]:
    """Map async function, yielding results as they complete.
    
    Like map_async but doesn't preserve order. Use when you want
    results as fast as possible without waiting for earlier items.
    
    Note: Returns a list for consistency, but items may be in any order.
    """
    if not items:
        return []
    
    if limit is None:
        limit = len(items)
    
    semaphore = asyncio.Semaphore(limit)
    results: list[U] = []
    
    async def limited_call(item: T) -> U:
        async with semaphore:
            return await func(item)
    
    tasks = [asyncio.ensure_future(limited_call(item)) for item in items]
    
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

async def wait_any(
    *coros: Awaitable[T],
    timeout: float | None = None,
) -> tuple[set[asyncio.Task[T]], set[asyncio.Task[T]]]:
    """Wait for any coroutine to complete.
    
    Lower-level than race() - returns (done, pending) sets like asyncio.wait.
    Doesn't cancel pending tasks.
    """
    if not coros:
        return set(), set()
    
    tasks = {asyncio.ensure_future(c) for c in coros}
    return await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)


async def wait_all(
    *coros: Awaitable[T],
    timeout: float | None = None,
) -> tuple[set[asyncio.Task[T]], set[asyncio.Task[T]]]:
    """Wait for all coroutines to complete.
    
    Lower-level wrapper around asyncio.wait with ALL_COMPLETED.
    """
    if not coros:
        return set(), set()
    
    tasks = {asyncio.ensure_future(c) for c in coros}
    return await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.ALL_COMPLETED)


async def retry_until_success(
    coro_factory: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
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
    last_error: BaseException | None = None
    current_delay = delay
    
    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except exceptions as e:
            last_error = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
                current_delay *= backoff
    
    raise last_error or RuntimeError("All retry attempts failed")
