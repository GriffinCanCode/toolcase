"""Sync/async interoperability utilities.

Provides utilities for bridging synchronous and asynchronous code:
    - run_sync: Run async code from sync context
    - run_async: Run sync code from async context (in thread)
    - from_thread: Call async code from worker thread
    - to_thread: Offload sync code to thread pool
    - AsyncAdapter: Wrap sync callable as async
    - SyncAdapter: Wrap async callable as sync

These utilities handle the tricky edge cases:
    - Running in an existing event loop (e.g., FastAPI, Jupyter)
    - Running from worker threads
    - Proper cleanup and cancellation

Example:
    >>> # Call async from sync
    >>> result = run_sync(async_function())
    
    >>> # Call sync from async (in thread)
    >>> result = await to_thread(blocking_function, arg1, arg2)
    
    >>> # Adapt sync to async
    >>> async_fn = AsyncAdapter(sync_fn)
    >>> result = await async_fn(args)
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Coroutine, Generic, ParamSpec, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable

T = TypeVar("T")
P = ParamSpec("P")

# Thread-local storage for event loop reference
_thread_local = threading.local()

# Default thread pool for to_thread operations
_default_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_default_executor() -> ThreadPoolExecutor:
    """Get or create default thread pool executor."""
    global _default_executor
    if _default_executor is None:
        with _executor_lock:
            if _default_executor is None:
                _default_executor = ThreadPoolExecutor(thread_name_prefix="toolcase-interop-")
    return _default_executor


# ─────────────────────────────────────────────────────────────────────────────
# Sync → Async: Running async code from sync context
# ─────────────────────────────────────────────────────────────────────────────

def run_sync(
    coro: Coroutine[object, object, T],
    *,
    loop: asyncio.AbstractEventLoop | None = None,
) -> T:
    """Run async coroutine from synchronous context.
    
    Handles multiple scenarios:
    1. No running loop → Use asyncio.run()
    2. Called from within event loop → Use thread pool
    3. Custom loop provided → Run in that loop
    
    Args:
        coro: Coroutine to execute
        loop: Optional event loop to use
    
    Returns:
        Coroutine result
    
    Example:
        >>> async def async_operation():
        ...     await asyncio.sleep(1)
        ...     return "done"
        >>> 
        >>> # From sync code
        >>> result = run_sync(async_operation())
    """
    if loop is not None:
        # Use provided loop
        return loop.run_until_complete(coro)
    
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - simple case
        return asyncio.run(coro)
    
    # We're inside a running loop - need to run in thread
    # This commonly happens in FastAPI, Jupyter, or nested async calls
    return _run_in_thread_loop(coro)


def _run_in_thread_loop(coro: Coroutine[object, object, T]) -> T:
    """Run coroutine in a new thread with its own event loop."""
    result: T | None = None
    error: BaseException | None = None
    done = threading.Event()
    
    def runner() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(coro)
        except BaseException as e:
            error = e
        finally:
            done.set()
    
    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    done.wait()
    
    if error is not None:
        raise error
    return result  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Async → Sync: Running sync code from async context
# ─────────────────────────────────────────────────────────────────────────────

async def run_async(
    func: Callable[..., T],
    *args: object,
    **kwargs: object,
) -> T:
    """Run sync function from async context (in thread).
    
    Offloads blocking/sync function to thread pool to avoid
    blocking the event loop.
    
    Args:
        func: Sync function to call
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Function result
    
    Example:
        >>> def blocking_io():
        ...     time.sleep(1)
        ...     return "done"
        >>> 
        >>> # From async code
        >>> result = await run_async(blocking_io)
    """
    return await to_thread(func, *args, **kwargs)  # type: ignore[arg-type]


async def to_thread(
    func: Callable[..., T],
    *args: object,
    cancellable: bool = False,
    **kwargs: object,
) -> T:
    """Run sync function in thread pool.
    
    Similar to asyncio.to_thread but with additional options.
    
    Args:
        func: Sync function to call
        *args: Positional arguments
        cancellable: Whether function should be interruptible (via threading)
        **kwargs: Keyword arguments
    
    Returns:
        Function result
    """
    loop = asyncio.get_running_loop()
    
    # Copy context for thread
    ctx = contextvars.copy_context()
    
    if kwargs:
        func = functools.partial(func, **kwargs)
    
    return await loop.run_in_executor(
        _get_default_executor(),
        functools.partial(ctx.run, func, *args),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Thread ↔ Async: Cross-thread async calls
# ─────────────────────────────────────────────────────────────────────────────

def from_thread(
    coro: Coroutine[object, object, T],
    loop: asyncio.AbstractEventLoop | None = None,
) -> T:
    """Call async code from a worker thread.
    
    Use when you need to call async code from within a thread pool
    worker or other non-async thread.
    
    Args:
        coro: Coroutine to run
        loop: Event loop to use (uses main thread's loop by default)
    
    Returns:
        Coroutine result
    
    Example:
        >>> def worker_thread():
        ...     # Inside thread pool
        ...     result = from_thread(async_operation())
        ...     return result
    """
    if loop is None:
        # Try to get the loop from thread-local storage
        loop = getattr(_thread_local, "loop", None)
        if loop is None:
            raise RuntimeError(
                "No event loop available. Either pass loop explicitly "
                "or call from within an async context."
            )
    
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def from_thread_nowait(
    coro: Coroutine[object, object, T],
    loop: asyncio.AbstractEventLoop | None = None,
) -> Future[T]:
    """Schedule async code from thread without waiting.
    
    Returns a Future that can be checked later.
    """
    if loop is None:
        loop = getattr(_thread_local, "loop", None)
        if loop is None:
            raise RuntimeError("No event loop available")
    
    return asyncio.run_coroutine_threadsafe(coro, loop)


def set_thread_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Set event loop reference for current thread.
    
    Call this from threads that will use from_thread().
    """
    _thread_local.loop = loop


def clear_thread_loop() -> None:
    """Clear event loop reference for current thread."""
    _thread_local.loop = None


# ─────────────────────────────────────────────────────────────────────────────
# Adapters: Convert between sync and async callables
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AsyncAdapter(Generic[P, T]):
    """Wrap a sync function to be callable as async.
    
    Runs the sync function in a thread pool to avoid blocking.
    
    Example:
        >>> def sync_io(path: str) -> str:
        ...     return open(path).read()
        >>> 
        >>> async_io = AsyncAdapter(sync_io)
        >>> content = await async_io("/etc/hosts")
    """
    
    func: Callable[P, T]
    executor: ThreadPoolExecutor | None = None
    
    async def __call__(self, *args: object, **kwargs: object) -> T:
        """Call wrapped function asynchronously."""
        loop = asyncio.get_running_loop()
        
        if kwargs:
            wrapped = functools.partial(self.func, **kwargs)  # type: ignore[arg-type]
            return await loop.run_in_executor(
                self.executor or _get_default_executor(),
                wrapped,
                *args,
            )
        
        return await loop.run_in_executor(
            self.executor or _get_default_executor(),
            self.func,  # type: ignore[arg-type]
            *args,
        )
    
    def __repr__(self) -> str:
        return f"AsyncAdapter({self.func!r})"


@dataclass
class SyncAdapter(Generic[P, T]):
    """Wrap an async function to be callable synchronously.
    
    Handles event loop creation/reuse transparently.
    
    Example:
        >>> async def async_fetch(url: str) -> str:
        ...     async with httpx.AsyncClient() as client:
        ...         return (await client.get(url)).text
        >>> 
        >>> sync_fetch = SyncAdapter(async_fetch)
        >>> content = sync_fetch("https://example.com")
    """
    
    func: Callable[P, Awaitable[T]]
    
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Call wrapped async function synchronously."""
        coro = self.func(*args, **kwargs)
        return run_sync(coro)  # type: ignore[arg-type]
    
    def __repr__(self) -> str:
        return f"SyncAdapter({self.func!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Decorators
# ─────────────────────────────────────────────────────────────────────────────

def async_to_sync(func: Callable[P, Awaitable[T]]) -> Callable[P, T]:
    """Decorator: Convert async function to sync.
    
    Example:
        >>> @async_to_sync
        ... async def fetch_data(url: str) -> dict:
        ...     async with httpx.AsyncClient() as client:
        ...         return (await client.get(url)).json()
        >>> 
        >>> # Now callable as sync
        >>> data = fetch_data("https://api.example.com")
    """
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        coro = func(*args, **kwargs)
        return run_sync(coro)  # type: ignore[arg-type]
    return wrapper


def sync_to_async(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """Decorator: Convert sync function to async (runs in thread).
    
    Example:
        >>> @sync_to_async
        ... def blocking_operation(data: bytes) -> bytes:
        ...     return expensive_sync_process(data)
        >>> 
        >>> # Now callable as async
        >>> result = await blocking_operation(data)
    """
    @functools.wraps(func)
    async def wrapper(*args: object, **kwargs: object) -> T:
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        if kwargs:
            partial_func = functools.partial(func, **kwargs)
            return await loop.run_in_executor(
                _get_default_executor(),
                functools.partial(ctx.run, partial_func, *args),
            )
        return await loop.run_in_executor(
            _get_default_executor(),
            functools.partial(ctx.run, func, *args),
        )
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Context Managers
# ─────────────────────────────────────────────────────────────────────────────

class ThreadContext:
    """Context manager for thread-safe async access.
    
    Sets up thread-local event loop reference for from_thread() calls.
    
    Example:
        >>> async def main():
        ...     async with ThreadContext() as ctx:
        ...         # Worker threads can now use from_thread()
        ...         await run_threaded_work()
    """
    
    __slots__ = ("_loop",)
    
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
    
    async def __aenter__(self) -> ThreadContext:
        self._loop = asyncio.get_running_loop()
        set_thread_loop(self._loop)
        return self
    
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        clear_thread_loop()
    
    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        return self._loop


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────

def shutdown_executor(wait: bool = True) -> None:
    """Shut down the default interop executor.
    
    Call at application shutdown to clean up resources.
    """
    global _default_executor
    if _default_executor is not None:
        with _executor_lock:
            if _default_executor is not None:
                _default_executor.shutdown(wait=wait)
                _default_executor = None
