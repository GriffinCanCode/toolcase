"""Unified concurrency facade for easy access across codebase.

Provides a single entry point to all concurrency utilities with
sensible defaults and convenience methods.

Usage:
    from toolcase.runtime.concurrency import Concurrency
    
    # Use as context manager for structured concurrency
    async with Concurrency.task_group() as tg:
        tg.spawn(fetch_data, url1)
        tg.spawn(fetch_data, url2)
    
    # Race multiple operations
    result = await Concurrency.race(op1(), op2(), op3())
    
    # Parallel map with limit
    results = await Concurrency.map(process, items, limit=10)
    
    # Run blocking code from async
    data = await Concurrency.to_thread(blocking_io)
    
    # Run async from sync
    result = Concurrency.run_sync(async_operation())
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, ParamSpec, TypeVar

T, U = TypeVar("T"), TypeVar("U")
P = ParamSpec("P")


@dataclass(slots=True)
class ConcurrencyConfig:
    """Configuration for Concurrency facade."""
    default_timeout: float = 30.0
    default_pool_size: int = 10
    default_thread_workers: int | None = None  # Uses system default
    default_process_workers: int | None = None  # Uses CPU count


class Concurrency:
    """Unified facade for concurrency operations.
    
    Provides clean, discoverable access to all concurrency primitives
    with sensible defaults. Use this as the primary import across codebase.
    
    Categories:
        - Tasks: task_group(), spawn(), shield()
        - Wait: race(), gather(), first_success(), map()
        - Pools: to_thread(), to_process(), thread_pool(), process_pool()
        - Sync: lock(), semaphore(), event(), barrier(), limiter()
        - Streams: merge(), buffer(), throttle(), batch()
        - Interop: run_sync(), run_async()
    
    Example:
        >>> from toolcase.runtime.concurrency import Concurrency
        >>> 
        >>> # Structured task group
        >>> async with Concurrency.task_group() as tg:
        ...     tg.spawn(fetch(url1))
        ...     tg.spawn(fetch(url2))
        >>> 
        >>> # Race with timeout
        >>> result = await Concurrency.race(api_a(), api_b(), timeout=5.0)
        >>> 
        >>> # Parallel map
        >>> results = await Concurrency.map(process, items, limit=10)
    """
    
    _config = ConcurrencyConfig()
    
    # ─────────────────────────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────────────────────────
    
    @classmethod
    def configure(
        cls, *, default_timeout: float | None = None, default_pool_size: int | None = None,
        default_thread_workers: int | None = None, default_process_workers: int | None = None,
    ) -> None:
        """Configure global defaults for concurrency operations."""
        if default_timeout is not None: cls._config.default_timeout = default_timeout
        if default_pool_size is not None: cls._config.default_pool_size = default_pool_size
        if default_thread_workers is not None: cls._config.default_thread_workers = default_thread_workers
        if default_process_workers is not None: cls._config.default_process_workers = default_process_workers
    
    # ─────────────────────────────────────────────────────────────────
    # Task Management
    # ─────────────────────────────────────────────────────────────────
    
    @staticmethod
    def task_group() -> "TaskGroup":
        """Create a structured task group.
        
        Tasks spawned in the group are automatically cancelled if any fails.
        All tasks complete before exiting the context.
        
        Example:
            >>> async with Concurrency.task_group() as tg:
            ...     h1 = tg.spawn(fetch_user(1))
            ...     h2 = tg.spawn(fetch_user(2))
            >>> print(h1.result(), h2.result())
        """
        from .primitives import TaskGroup
        return TaskGroup()
    
    @staticmethod
    def spawn(coro: Coroutine[object, object, T], *, name: str | None = None) -> "TaskHandle[T]":
        """Spawn unstructured task (prefer task_group for structured concurrency)."""
        from .primitives import spawn
        return spawn(coro, name=name)
    
    @staticmethod
    async def shield(coro: Awaitable[T]) -> T:
        """Shield coroutine from cancellation."""
        from .primitives import shield
        return await shield(coro)
    
    @staticmethod
    async def checkpoint() -> None:
        """Yield to event loop, allowing cancellation."""
        from .primitives import checkpoint
        await checkpoint()
    
    @staticmethod
    async def sleep(seconds: float) -> None:
        """Async sleep with cancellation support."""
        await asyncio.sleep(seconds)
    
    @staticmethod
    def cancel_scope(*, timeout: float | None = None, shield: bool = False) -> "CancelScope":
        """Create a cancellation scope with optional timeout."""
        from .primitives import CancelScope
        return CancelScope(timeout=timeout, shield=shield)
    
    # ─────────────────────────────────────────────────────────────────
    # Wait Strategies
    # ─────────────────────────────────────────────────────────────────
    
    @staticmethod
    async def race(*coros: Awaitable[T], timeout: float | None = None) -> T:
        """Race coroutines - first to complete wins.
        
        Cancels remaining after first completes.
        
        Example:
            >>> result = await Concurrency.race(
            ...     fetch_from_api_a(),
            ...     fetch_from_api_b(),
            ...     fetch_from_cache(),
            ... )
        """
        from .execution import race
        return await race(*coros, timeout=timeout)
    
    @staticmethod
    async def gather(*coros: Awaitable[T], return_exceptions: bool = False) -> list[T]:
        """Gather all results (like asyncio.gather)."""
        from .execution import gather
        return await gather(*coros, return_exceptions=return_exceptions)
    
    @staticmethod
    async def gather_settled(*coros: Awaitable[T]) -> "list[Settled[T]]":
        """Gather all, returning success/failure status for each."""
        from .execution import gather_settled
        return await gather_settled(*coros)
    
    @staticmethod
    async def first_success(*coros: Awaitable[T], timeout: float | None = None) -> T:
        """Get first successful result, skipping failures."""
        from .execution import first_success
        return await first_success(*coros, timeout=timeout)
    
    @classmethod
    async def map(
        cls, func: Callable[[T], Awaitable[U]], items: list[T], *, limit: int | None = None, return_exceptions: bool = False,
    ) -> list[U]:
        """Parallel map with optional concurrency limit.
        
        Example:
            >>> results = await Concurrency.map(fetch_url, urls, limit=10)
        """
        from .execution import map_async
        return await map_async(func, items, limit=limit or cls._config.default_pool_size, return_exceptions=return_exceptions)
    
    @staticmethod
    async def retry(
        factory: Callable[[], Awaitable[T]], *, max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
    ) -> T:
        """Retry operation until success."""
        from .execution import retry_until_success
        return await retry_until_success(factory, max_attempts=max_attempts, delay=delay, backoff=backoff)
    
    # ─────────────────────────────────────────────────────────────────
    # Thread/Process Pools
    # ─────────────────────────────────────────────────────────────────
    
    @staticmethod
    async def to_thread(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Run blocking function in thread pool.
        
        Example:
            >>> data = await Concurrency.to_thread(read_file, path)
        """
        from .interop import to_thread
        return await to_thread(func, *args, **kwargs)
    
    @staticmethod
    async def to_process(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Run CPU-bound function in process pool."""
        from .execution import run_in_process
        return await run_in_process(func, *args, **kwargs)
    
    @classmethod
    def thread_pool(cls, max_workers: int | None = None) -> "ThreadPool":
        """Create a thread pool context manager."""
        from .execution import ThreadPool
        return ThreadPool(max_workers=max_workers or cls._config.default_thread_workers or 4)
    
    @classmethod
    def process_pool(cls, max_workers: int | None = None) -> "ProcessPool":
        """Create a process pool context manager."""
        from .execution import ProcessPool
        return ProcessPool(max_workers=max_workers or cls._config.default_process_workers or 4)
    
    # ─────────────────────────────────────────────────────────────────
    # Synchronization Primitives
    # ─────────────────────────────────────────────────────────────────
    
    @staticmethod
    def lock() -> "Lock":
        """Create an async lock."""
        from .primitives import Lock
        return Lock()
    
    @staticmethod
    def semaphore(value: int = 1) -> "Semaphore":
        """Create a counting semaphore."""
        from .primitives import Semaphore
        return Semaphore(value)
    
    @staticmethod
    def event() -> "Event":
        """Create an async event."""
        from .primitives import Event
        return Event()
    
    @staticmethod
    def barrier(parties: int) -> "Barrier":
        """Create a barrier for N parties."""
        from .primitives import Barrier
        return Barrier(parties)
    
    @staticmethod
    def limiter(capacity: int) -> "CapacityLimiter":
        """Create a capacity limiter for resource control.
        
        Example:
            >>> limiter = Concurrency.limiter(10)  # Max 10 concurrent
            >>> async with limiter: await api_call()
        """
        from .primitives import CapacityLimiter
        return CapacityLimiter(capacity)
    
    # ─────────────────────────────────────────────────────────────────
    # Stream Utilities
    # ─────────────────────────────────────────────────────────────────
    
    @staticmethod
    async def merge(*streams: AsyncIterator[T]) -> AsyncIterator[T]:
        """Merge multiple async streams into one."""
        from .streams import merge_streams
        async for item in merge_streams(*streams):
            yield item
    
    @staticmethod
    async def buffer(stream: AsyncIterator[T], maxsize: int = 10) -> AsyncIterator[T]:
        """Buffer stream for smoother consumption."""
        from .streams import buffer_stream
        async for item in buffer_stream(stream, maxsize=maxsize):
            yield item
    
    @staticmethod
    async def throttle(stream: AsyncIterator[T], rate: float, *, per: float = 1.0) -> AsyncIterator[T]:
        """Throttle stream to rate items per second."""
        from .streams import throttle_stream
        async for item in throttle_stream(stream, rate, per=per):
            yield item
    
    @staticmethod
    async def batch(stream: AsyncIterator[T], size: int, *, timeout: float | None = None) -> AsyncIterator[list[T]]:
        """Batch stream items into groups."""
        from .streams import batch_stream
        async for items in batch_stream(stream, size, timeout=timeout):
            yield items
    
    # ─────────────────────────────────────────────────────────────────
    # Sync/Async Interop
    # ─────────────────────────────────────────────────────────────────
    
    @staticmethod
    def run_sync(coro: Coroutine[object, object, T]) -> T:
        """Run async code from sync context. Handles running from within existing event loops (e.g., Jupyter, FastAPI).
        
        Example:
            >>> result = Concurrency.run_sync(async_operation())  # From sync code
        """
        from .interop import run_sync
        return run_sync(coro)
    
    @staticmethod
    async def run_async(func: Callable[..., T], *args: object, **kwargs: object) -> T:
        """Run sync function from async context (in thread)."""
        from .interop import run_async
        return await run_async(func, *args, **kwargs)
    
    @staticmethod
    def async_adapter(func: Callable[P, T]) -> "AsyncAdapter[P, T]":
        """Wrap sync function to be async (runs in thread)."""
        from .interop import AsyncAdapter
        return AsyncAdapter(func)
    
    @staticmethod
    def sync_adapter(func: Callable[P, Awaitable[T]]) -> "SyncAdapter[P, T]":
        """Wrap async function to be sync."""
        from .interop import SyncAdapter
        return SyncAdapter(func)
    
    @staticmethod
    def thread_context() -> "ThreadContext":
        """Context for calling async from worker threads."""
        from .interop import ThreadContext
        return ThreadContext()
    
    # ─────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────
    
    @staticmethod
    def shutdown() -> None:
        """Shutdown all default pools and executors.
        
        Call at application shutdown for clean resource cleanup.
        """
        from .execution import shutdown_default_pools
        from .interop import shutdown_executor
        shutdown_default_pools()
        shutdown_executor()


# Type hints for return types (forward references)
if TYPE_CHECKING:
    from .primitives import (
        TaskGroup,
        TaskHandle,
        CancelScope,
        Lock,
        Semaphore,
        Event,
        Barrier,
        CapacityLimiter,
    )
    from .execution import ThreadPool, ProcessPool, Settled
    from .interop import AsyncAdapter, SyncAdapter, ThreadContext


# Convenience aliases at module level
task_group = Concurrency.task_group
race = Concurrency.race
gather = Concurrency.gather
first_success = Concurrency.first_success
map_async = Concurrency.map
to_thread = Concurrency.to_thread
run_sync = Concurrency.run_sync
sleep = Concurrency.sleep