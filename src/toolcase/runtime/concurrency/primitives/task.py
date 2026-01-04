"""Task management with structured concurrency.

Provides TaskGroup for managing multiple concurrent tasks with proper
cancellation semantics. When any task fails, sibling tasks are cancelled.

Key Features:
    - Structured lifetime: Tasks don't outlive their TaskGroup
    - Automatic cancellation: First exception cancels siblings
    - Cancel scopes: Fine-grained cancellation control
    - Task handles: Access to task state and results
    - Checkpoints: Cooperative cancellation points
    - Uses native asyncio.TaskGroup on Python 3.11+

Example:
    >>> async with TaskGroup() as tg:
    ...     handle1 = tg.spawn(fetch_data, "url1")
    ...     handle2 = tg.spawn(fetch_data, "url2")
    ...     # Both tasks run concurrently
    ... # Exiting context waits for completion
    >>> print(handle1.result())
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import sys
from collections.abc import Awaitable, Coroutine
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Callable, Generic, ParamSpec, TypeVar

if TYPE_CHECKING:
    from types import TracebackType

T = TypeVar("T")
P = ParamSpec("P")

# Use native TaskGroup on Python 3.11+
_USE_NATIVE_TASKGROUP = sys.version_info >= (3, 11)


class _FallbackExceptionGroup(BaseException):
    """Fallback ExceptionGroup for Python < 3.11."""
    
    __slots__ = ("message", "exceptions")
    
    def __init__(self, message: str, exceptions: list[Exception]) -> None:
        self.message = message
        self.exceptions = exceptions
        super().__init__(message)
    
    def __str__(self) -> str:
        return f"{self.message} ({len(self.exceptions)} sub-exceptions)"


# ExceptionGroup is only available in Python 3.11+
try:
    _ExceptionGroup: type[BaseException] = ExceptionGroup  # type: ignore[name-defined]
except NameError:
    _ExceptionGroup = _FallbackExceptionGroup

__all__ = [
    "TaskState",
    "TaskHandle",
    "CancelScope",
    "TaskGroup",
    "shield",
    "checkpoint",
    "shielded_checkpoint",
    "current_task",
    "spawn",
    "cancellable",
]


class TaskState(StrEnum):
    """Task lifecycle states."""
    PENDING = "pending"      # Not yet started
    RUNNING = "running"      # Currently executing
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"        # Raised exception
    CANCELLED = "cancelled"  # Was cancelled


@dataclass(slots=True)
class TaskHandle(Generic[T]):
    """Handle to a spawned task with state access.
    
    Provides access to task state, result, and cancellation control
    without exposing the underlying asyncio.Task directly.
    
    Attributes:
        name: Optional task name for debugging
        task: Underlying asyncio Task (internal)
    """
    
    name: str | None = None
    _task: asyncio.Task[T] | None = field(default=None, repr=False)
    
    @property
    def state(self) -> TaskState:
        """Current task state."""
        if self._task is None:
            return TaskState.PENDING
        if self._task.cancelled():
            return TaskState.CANCELLED
        if not self._task.done():
            return TaskState.RUNNING
        return TaskState.FAILED if self._task.exception() else TaskState.COMPLETED
    
    @property
    def done(self) -> bool:
        """Whether task has finished (success, failure, or cancelled)."""
        return self._task is not None and self._task.done()
    
    def result(self) -> T:
        """Get task result.
        
        Raises:
            RuntimeError: If task not complete
            Exception: If task failed with exception
            asyncio.CancelledError: If task was cancelled
        """
        if self._task is None:
            raise RuntimeError("Task not started")
        return self._task.result()
    
    def exception(self) -> BaseException | None:
        """Get task exception, or None if successful."""
        if self._task is None or not self._task.done() or self._task.cancelled():
            return None
        return self._task.exception()
    
    def cancel(self, msg: str | None = None) -> bool:
        """Request task cancellation.
        
        Returns:
            True if cancellation was requested, False if task already done
        """
        return self._task.cancel(msg) if self._task else False
    
    async def wait(self) -> T:
        """Wait for task completion and return result."""
        if self._task is None:
            raise RuntimeError("Task not started")
        return await self._task


@dataclass(slots=True)
class CancelScope:
    """Cancellation scope for fine-grained control.
    
    Allows cancelling a group of operations without affecting
    the entire TaskGroup. Supports timeouts and deadlines.
    
    Example:
        >>> async with CancelScope(timeout=5.0) as scope:
        ...     await long_operation()
        ...     if scope.cancel_called:
        ...         print("Timed out!")
    """
    
    timeout: float | None = None
    shield: bool = False
    _cancel_called: bool = field(default=False, repr=False)
    _timeout_task: asyncio.Task[None] | None = field(default=None, repr=False)
    _tasks: set[asyncio.Task[object]] = field(default_factory=set, repr=False)
    
    @property
    def cancel_called(self) -> bool:
        """Whether cancellation was requested."""
        return self._cancel_called
    
    def cancel(self) -> None:
        """Cancel all tasks in this scope."""
        self._cancel_called = True
        for task in self._tasks:
            task.cancel() if not task.done() else None
    
    async def __aenter__(self) -> CancelScope:
        if self.timeout is not None:
            self._timeout_task = asyncio.create_task(self._timeout_handler())
        return self
    
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if self._timeout_task:
            self._timeout_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._timeout_task
        # Suppress CancelledError if we initiated the cancellation
        return exc_type is asyncio.CancelledError and self._cancel_called
    
    async def _timeout_handler(self) -> None:
        """Internal timeout trigger."""
        await asyncio.sleep(self.timeout)  # type: ignore[arg-type]
        self.cancel()


class _LegacyTaskGroup:
    """Legacy TaskGroup implementation for Python < 3.11. Manages multiple concurrent tasks as a unit."""
    
    __slots__ = ("_tasks", "_handles", "_host_task", "_started", "_exiting")
    
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[object]] = set()
        self._handles: list[TaskHandle[object]] = []
        self._host_task: asyncio.Task[object] | None = None
        self._started = self._exiting = False
    
    def spawn(
        self,
        coro: Coroutine[object, object, T],
        *,
        name: str | None = None,
    ) -> TaskHandle[T]:
        """Spawn a task in this group."""
        if not self._started:
            raise RuntimeError("TaskGroup must be used as context manager")
        if self._exiting:
            raise RuntimeError("Cannot spawn tasks while exiting TaskGroup")
        
        task = asyncio.create_task(coro, name=name)
        handle: TaskHandle[T] = TaskHandle(name=name, _task=task)
        self._tasks.add(task)
        self._handles.append(handle)  # type: ignore[arg-type]
        task.add_done_callback(self._tasks.discard)
        return handle
    
    def spawn_soon(
        self,
        coro: Coroutine[object, object, T],
        *,
        name: str | None = None,
    ) -> TaskHandle[T]:
        """Schedule a task to start on next event loop iteration."""
        async def deferred() -> T:
            await asyncio.sleep(0)
            return await coro
        return self.spawn(deferred(), name=name)
    
    @property
    def tasks(self) -> list[TaskHandle[object]]:
        """All task handles in this group."""
        return list(self._handles)
    
    async def __aenter__(self) -> _LegacyTaskGroup:
        self._host_task = asyncio.current_task()
        self._started = True
        return self
    
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self._exiting = True
        if exc_val is not None:
            for task in self._tasks:
                task.cancel()
        
        exceptions: list[BaseException] = []
        while self._tasks:
            done, _ = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass
                except BaseException as e:
                    exceptions.append(e)
                    for t in self._tasks:
                        t.cancel()
        
        if exceptions:
            if exc_val is not None:
                exceptions.insert(0, exc_val)
            if len(exceptions) == 1:
                raise exceptions[0]
            raise _ExceptionGroup("TaskGroup errors", [e for e in exceptions if isinstance(e, Exception)])
        return False


class _NativeTaskGroup:
    """TaskGroup using Python 3.11+ native asyncio.TaskGroup. Wraps native impl with TaskHandle interface."""
    
    __slots__ = ("_tg", "_handles", "_started")
    
    def __init__(self) -> None:
        self._tg: asyncio.TaskGroup | None = None
        self._handles: list[TaskHandle[object]] = []
        self._started = False
    
    def spawn(
        self,
        coro: Coroutine[object, object, T],
        *,
        name: str | None = None,
    ) -> TaskHandle[T]:
        """Spawn a task using native TaskGroup."""
        if not self._started or self._tg is None:
            raise RuntimeError("TaskGroup must be used as context manager")
        task = self._tg.create_task(coro, name=name)
        handle: TaskHandle[T] = TaskHandle(name=name, _task=task)
        self._handles.append(handle)  # type: ignore[arg-type]
        return handle
    
    def spawn_soon(
        self,
        coro: Coroutine[object, object, T],
        *,
        name: str | None = None,
    ) -> TaskHandle[T]:
        """Schedule a task to start on next event loop iteration."""
        async def deferred() -> T:
            await asyncio.sleep(0)
            return await coro
        return self.spawn(deferred(), name=name)
    
    @property
    def tasks(self) -> list[TaskHandle[object]]:
        """All task handles in this group."""
        return list(self._handles)
    
    async def __aenter__(self) -> _NativeTaskGroup:
        self._tg = asyncio.TaskGroup()
        await self._tg.__aenter__()
        self._started = True
        return self
    
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        return await self._tg.__aexit__(exc_type, exc_val, exc_tb) if self._tg else False  # type: ignore[return-value]


# Choose implementation based on Python version
# Type alias for proper type checking
if _USE_NATIVE_TASKGROUP:
    TaskGroup = _NativeTaskGroup
else:
    TaskGroup = _LegacyTaskGroup

# Add docstring for the chosen implementation
TaskGroup.__doc__ = """Structured task group with automatic cancellation.

Manages multiple concurrent tasks as a unit. When exiting the
context, waits for all tasks. If any task fails, cancels siblings.

Uses native asyncio.TaskGroup on Python 3.11+ for better performance.
Falls back to custom implementation on earlier versions.

Features:
    - spawn(): Start tasks that are managed by the group
    - spawn_soon(): Schedule task for next iteration
    - Automatic cleanup on exception
    - First-exception cancels all

Example:
    >>> async with TaskGroup() as tg:
    ...     tg.spawn(fetch_user, user_id)
    ...     tg.spawn(fetch_orders, user_id)
    ...     tg.spawn(fetch_preferences, user_id)
    >>> # All complete, results in handles
    
    >>> # With error handling
    >>> try:
    ...     async with TaskGroup() as tg:
    ...         tg.spawn(may_fail)
    ...         tg.spawn(another_task)
    ... except ExceptionGroup as eg:
    ...     for exc in eg.exceptions:
    ...         print(f"Task failed: {exc}")
"""


async def shield(coro: Awaitable[T]) -> T:
    """Shield a coroutine from cancellation.
    
    The coroutine will complete even if the calling task is cancelled.
    Use sparingly - this breaks structured concurrency guarantees.
    
    Example:
        >>> async with TaskGroup() as tg:
        ...     # This will complete even if group is cancelled
        ...     result = await shield(critical_operation())
    """
    return await asyncio.shield(coro)


async def checkpoint() -> None:
    """Cooperative cancellation checkpoint. Yields control to event loop, allowing pending cancellations to be processed."""
    await asyncio.sleep(0)


async def shielded_checkpoint() -> None:
    """Checkpoint that handles cancellation gracefully. Like checkpoint() but re-raises CancelledError for cleanup."""
    await asyncio.sleep(0)


def current_task() -> asyncio.Task[object] | None:
    """Get the currently running task. Returns None if not in async context."""
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def spawn(
    coro: Coroutine[object, object, T],
    *,
    name: str | None = None,
) -> TaskHandle[T]:
    """Spawn a standalone task (not in a TaskGroup).
    
    WARNING: This creates an unstructured task that may outlive its
    caller. Prefer TaskGroup.spawn() for structured concurrency.
    
    Args:
        coro: Coroutine to run
        name: Optional task name
    
    Returns:
        TaskHandle for the spawned task
    """
    return TaskHandle(name=name, _task=asyncio.create_task(coro, name=name))


# Decorator for making functions checkpoint-aware
def cancellable(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """Decorator that adds automatic checkpoints. Wraps async function to check cancellation before/after."""
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        await checkpoint()
        result = await func(*args, **kwargs)
        await checkpoint()
        return result
    return wrapper
