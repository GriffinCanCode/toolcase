"""Thread and process pool executors for CPU-bound work.

Provides async-friendly wrappers around concurrent.futures executors
for offloading CPU-intensive or blocking operations.

Key Features:
    - ThreadPool: For blocking I/O and light CPU work
    - ProcessPool: For heavy CPU-bound work (bypasses GIL)
    - Context manager support for automatic cleanup
    - Graceful shutdown with cancellation support

Use Cases:
    - ThreadPool: Database queries, file I/O, external process calls
    - ProcessPool: Data processing, compression, crypto, ML inference

Example:
    >>> async with ThreadPool(max_workers=4) as pool:
    ...     result = await pool.run(cpu_intensive_function, arg1, arg2)
    
    >>> # One-off execution
    >>> result = await run_in_thread(blocking_io_function)
"""

from __future__ import annotations

import asyncio
import functools
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, ParamSpec, TypeVar

if TYPE_CHECKING:
    from types import TracebackType

T = TypeVar("T")
P = ParamSpec("P")

__all__ = [
    "ThreadPool",
    "ProcessPool",
    "run_in_thread",
    "run_in_process",
    "shutdown_default_pools",
    "threadpool",
    "processpool",
    "DEFAULT_THREAD_WORKERS",
    "DEFAULT_PROCESS_WORKERS",
]

# Default worker counts based on CPU cores
_CPU_COUNT = os.cpu_count() or 1
DEFAULT_THREAD_WORKERS = min(32, _CPU_COUNT + 4)  # I/O bound heuristic
DEFAULT_PROCESS_WORKERS = _CPU_COUNT


@dataclass(slots=True)
class ThreadPool:
    """Async-friendly thread pool for blocking operations.
    
    Wraps ThreadPoolExecutor with async interface. Use for:
    - Blocking I/O (file operations, database calls)
    - Sync library functions
    - Light CPU work (parsing, serialization)
    
    Example:
        >>> async with ThreadPool(4) as pool:
        ...     # Run blocking function in thread
        ...     data = await pool.run(read_large_file, path)
        ...     
        ...     # Run multiple in parallel
        ...     results = await asyncio.gather(
        ...         pool.run(process, item) for item in items
        ...     )
    """
    
    max_workers: int = DEFAULT_THREAD_WORKERS
    thread_name_prefix: str = "toolcase-"
    _executor: ThreadPoolExecutor | None = field(default=None, repr=False)
    _owned: bool = field(default=True, repr=False)  # Whether we created the executor
    
    def __post_init__(self) -> None:
        if self.max_workers < 1: raise ValueError("max_workers must be >= 1")
    
    @classmethod
    def from_executor(cls, executor: ThreadPoolExecutor) -> ThreadPool:
        """Wrap an existing ThreadPoolExecutor. Pool won't be shut down when context exits."""
        pool = cls.__new__(cls)
        pool.max_workers, pool.thread_name_prefix, pool._executor, pool._owned = getattr(executor, "_max_workers", 4), "", executor, False
        return pool
    
    @property
    def executor(self) -> ThreadPoolExecutor:
        """Get or create the underlying executor."""
        if not self._executor:
            self._executor = ThreadPoolExecutor(self.max_workers, thread_name_prefix=self.thread_name_prefix)
        return self._executor
    
    async def run(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Run a function in the thread pool.
        
        Args:
            func: Sync function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Exception: Any exception raised by the function
        """
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, functools.partial(func, **kwargs) if kwargs else func, *args
        )
    
    def map(self, func: Callable[[T], object], items: list[T], *, timeout: float | None = None) -> list[object]:
        """Map function over items (sync). For async: `await asyncio.gather(*(pool.run(func, x) for x in items))`"""
        return list(self.executor.map(func, items, timeout=timeout))
    
    def submit(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> asyncio.Future[T]:
        """Submit work and return a Future. Lower-level than run()."""
        return asyncio.get_running_loop().run_in_executor(  # type: ignore[return-value]
            self.executor, functools.partial(func, **kwargs) if kwargs else func, *args
        )
    
    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Shut down the pool."""
        if self._executor and self._owned:
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            self._executor = None
    
    async def __aenter__(self) -> ThreadPool: return (self.executor, self)[1]
    
    async def __aexit__(self, _t: type[BaseException] | None, exc: BaseException | None, _tb: TracebackType | None) -> None:
        self.shutdown(wait=True, cancel_futures=exc is not None)


@dataclass(slots=True)
class ProcessPool:
    """Async-friendly process pool for CPU-bound operations.
    
    Wraps ProcessPoolExecutor for heavy computation. Each worker
    runs in a separate process, bypassing the GIL.
    
    Limitations:
        - Functions and arguments must be picklable
        - Higher overhead than threads for small tasks
        - Process startup has latency
    
    Example:
        >>> async with ProcessPool(4) as pool:
        ...     # CPU-intensive work in separate process
        ...     result = await pool.run(heavy_computation, data)
    """
    
    max_workers: int = DEFAULT_PROCESS_WORKERS
    mp_context: str | None = None  # 'fork', 'spawn', 'forkserver'
    _executor: ProcessPoolExecutor | None = field(default=None, repr=False)
    _owned: bool = field(default=True, repr=False)
    
    def __post_init__(self) -> None:
        if self.max_workers < 1: raise ValueError("max_workers must be >= 1")
    
    @classmethod
    def from_executor(cls, executor: ProcessPoolExecutor) -> ProcessPool:
        """Wrap an existing ProcessPoolExecutor."""
        pool = cls.__new__(cls)
        pool.max_workers, pool.mp_context, pool._executor, pool._owned = getattr(executor, "_max_workers", _CPU_COUNT), None, executor, False
        return pool
    
    @property
    def executor(self) -> ProcessPoolExecutor:
        """Get or create the underlying executor."""
        if not self._executor:
            self._executor = ProcessPoolExecutor(self.max_workers, mp_context=multiprocessing.get_context(self.mp_context) if self.mp_context else None)
        return self._executor
    
    async def run(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Run a function in the process pool. Note: func and args must be picklable.
        
        Args:
            func: Sync function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        """
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, functools.partial(func, **kwargs) if kwargs else func, *args
        )
    
    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Shut down the pool."""
        if self._executor and self._owned:
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            self._executor = None
    
    async def __aenter__(self) -> ProcessPool: return (self.executor, self)[1]
    
    async def __aexit__(self, _t: type[BaseException] | None, exc: BaseException | None, _tb: TracebackType | None) -> None:
        self.shutdown(wait=True, cancel_futures=exc is not None)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────

# Global default pools (lazily initialized)
_default_thread_pool: ThreadPool | None = None
_default_process_pool: ProcessPool | None = None


def _get_default_thread_pool() -> ThreadPool:
    """Get or create default thread pool."""
    global _default_thread_pool
    return _default_thread_pool if _default_thread_pool else (_default_thread_pool := ThreadPool())


def _get_default_process_pool() -> ProcessPool:
    """Get or create default process pool."""
    global _default_process_pool
    return _default_process_pool if _default_process_pool else (_default_process_pool := ProcessPool())


async def run_in_thread(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run a blocking function in the default thread pool. Convenience for one-off execution.
    
    Example:
        >>> data = await run_in_thread(read_file, path)
        >>> await run_in_thread(write_file, path, data)
    """
    return await _get_default_thread_pool().run(func, *args, **kwargs)


async def run_in_process(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run a CPU-bound function in the default process pool. func and args must be picklable.
    
    Example:
        >>> result = await run_in_process(heavy_compute, data)
    """
    return await _get_default_process_pool().run(func, *args, **kwargs)


def shutdown_default_pools(wait: bool = True) -> None:
    """Shut down default pools. Call at application shutdown to clean up resources."""
    global _default_thread_pool, _default_process_pool
    for pool in (_default_thread_pool, _default_process_pool):
        if pool: pool.shutdown(wait=wait)
    _default_thread_pool = _default_process_pool = None


# ─────────────────────────────────────────────────────────────────────────────
# Decorators
# ─────────────────────────────────────────────────────────────────────────────

from collections.abc import Coroutine


def threadpool(
    func: Callable[P, T] | None = None, *, pool: ThreadPool | None = None
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, Coroutine[object, object, T]]]:
    """Decorator to run sync function in thread pool. Converts sync to async.
    
    Example:
        >>> @threadpool
        ... def blocking_operation(x: int) -> int:
        ...     time.sleep(1)
        ...     return x * 2
        >>> 
        >>> result = await blocking_operation(5)
    """
    def decorator(f: Callable[P, T]) -> Callable[P, Coroutine[object, object, T]]:
        @functools.wraps(f)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await (pool or _get_default_thread_pool()).run(f, *args, **kwargs)
        return wrapper
    return decorator(func) if func else decorator


def processpool(
    func: Callable[P, T] | None = None, *, pool: ProcessPool | None = None
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, Coroutine[object, object, T]]]:
    """Decorator to run sync function in process pool. Decorated function must be picklable.
    
    Example:
        >>> @processpool
        ... def cpu_intensive(data: bytes) -> bytes:
        ...     return expensive_computation(data)
        >>> 
        >>> result = await cpu_intensive(raw_data)
    """
    def decorator(f: Callable[P, T]) -> Callable[P, Coroutine[object, object, T]]:
        @functools.wraps(f)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await (pool or _get_default_process_pool()).run(f, *args, **kwargs)
        return wrapper
    return decorator(func) if func else decorator
