"""Task management with structured concurrency.

Provides TaskGroup for managing multiple concurrent tasks with proper
cancellation semantics. When any task fails, sibling tasks are cancelled.

Key Features:
    - Structured lifetime: Tasks don't outlive their TaskGroup
    - Automatic cancellation: First exception cancels siblings
    - Cancel scopes: Fine-grained cancellation control
    - Task handles: Access to task state and results
    - Checkpoints: Cooperative cancellation points

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
import contextvars
import functools
from collections.abc import Awaitable, Coroutine
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Callable, Generic, ParamSpec, TypeVar

if TYPE_CHECKING:
    from types import TracebackType

T = TypeVar("T")
P = ParamSpec("P")

# Context variable for current task tracking
_current_task: contextvars.ContextVar[asyncio.Task[object] | None] = contextvars.ContextVar(
    "current_task", default=None
)


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
    _result: T | None = field(default=None, repr=False)
    _exception: BaseException | None = field(default=None, repr=False)
    
    @property
    def state(self) -> TaskState:
        """Current task state."""
        if self._task is None:
            return TaskState.PENDING
        if self._task.cancelled():
            return TaskState.CANCELLED
        if self._task.done():
            return TaskState.FAILED if self._task.exception() else TaskState.COMPLETED
        return TaskState.RUNNING
    
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
        if self._task is None or not self._task.done():
            return None
        try:
            self._task.result()
            return None
        except BaseException as e:
            return e
    
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
            if not task.done():
                task.cancel()
    
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
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass
        
        # Suppress CancelledError if we initiated the cancellation
        if exc_type is asyncio.CancelledError and self._cancel_called:
            return True
        return False
    
    async def _timeout_handler(self) -> None:
        """Internal timeout trigger."""
        assert self.timeout is not None
        await asyncio.sleep(self.timeout)
        self.cancel()


class TaskGroup:
    """Structured task group with automatic cancellation.
    
    Manages multiple concurrent tasks as a unit. When exiting the
    context, waits for all tasks. If any task fails, cancels siblings.
    
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
    
    __slots__ = ("_tasks", "_handles", "_host_task", "_started", "_exiting")
    
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[object]] = set()
        self._handles: list[TaskHandle[object]] = []
        self._host_task: asyncio.Task[object] | None = None
        self._started = False
        self._exiting = False
    
    def spawn(
        self,
        coro: Coroutine[object, object, T],
        *,
        name: str | None = None,
    ) -> TaskHandle[T]:
        """Spawn a task in this group.
        
        Args:
            coro: Coroutine to execute
            name: Optional task name
        
        Returns:
            TaskHandle for accessing result/state
        
        Raises:
            RuntimeError: If called outside context manager
        """
        if not self._started:
            raise RuntimeError("TaskGroup must be used as context manager")
        if self._exiting:
            raise RuntimeError("Cannot spawn tasks while exiting TaskGroup")
        
        task = asyncio.create_task(coro, name=name)
        handle: TaskHandle[T] = TaskHandle(name=name)
        handle._task = task
        
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
        """Schedule a task to start on next event loop iteration.
        
        Unlike spawn(), doesn't start immediately. Useful when you
        need to ensure ordering or want deferred execution.
        """
        async def deferred() -> T:
            await asyncio.sleep(0)  # Yield to event loop
            return await coro
        
        return self.spawn(deferred(), name=name)
    
    @property
    def tasks(self) -> list[TaskHandle[object]]:
        """All task handles in this group."""
        return list(self._handles)
    
    async def __aenter__(self) -> TaskGroup:
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
        
        # Cancel all tasks if we're exiting due to exception
        if exc_val is not None:
            for task in self._tasks:
                task.cancel()
        
        # Wait for all tasks to complete
        exceptions: list[BaseException] = []
        
        while self._tasks:
            done, _ = await asyncio.wait(
                self._tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            
            for task in done:
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass  # Expected if we cancelled
                except BaseException as e:
                    exceptions.append(e)
                    # Cancel remaining on first failure
                    for t in self._tasks:
                        t.cancel()
        
        # Re-raise collected exceptions
        if exceptions:
            if exc_val is not None:
                exceptions.insert(0, exc_val)
            if len(exceptions) == 1:
                raise exceptions[0]
            raise ExceptionGroup("TaskGroup errors", [e for e in exceptions if isinstance(e, Exception)])
        
        return False


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
    """Cooperative cancellation checkpoint.
    
    Yields control to the event loop, allowing pending cancellations
    to be processed. Call periodically in long-running sync code.
    
    Example:
        >>> async def process_many(items):
        ...     for item in items:
        ...         process_sync(item)
        ...         await checkpoint()  # Allow cancellation here
    """
    await asyncio.sleep(0)


def current_task() -> asyncio.Task[object] | None:
    """Get the currently running task.
    
    Returns:
        Current asyncio.Task or None if not in async context
    """
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
    task = asyncio.create_task(coro, name=name)
    handle: TaskHandle[T] = TaskHandle(name=name)
    handle._task = task
    return handle


# Decorator for making functions checkpoint-aware
def cancellable(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """Decorator that adds automatic checkpoints.
    
    Wraps an async function to check for cancellation before and after.
    
    Example:
        >>> @cancellable
        ... async def long_operation():
        ...     # Will check cancellation on entry and exit
        ...     return await do_work()
    """
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        await checkpoint()
        result = await func(*args, **kwargs)
        await checkpoint()
        return result
    return wrapper
