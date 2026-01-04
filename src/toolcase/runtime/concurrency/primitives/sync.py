"""Async synchronization primitives.

Provides higher-level synchronization abstractions built on asyncio
primitives with additional features like timeouts and fairness.

Key Primitives:
    - Lock: Mutual exclusion with timeout support
    - RLock: Reentrant lock for recursive locking
    - Semaphore: Counting semaphore with optional bounds
    - Event: One-shot signaling mechanism
    - Condition: Condition variable for complex synchronization
    - Barrier: Synchronization point for multiple tasks
    - CapacityLimiter: Limit concurrent access to a resource

Example:
    >>> async with Lock() as lock:
    ...     # Exclusive access here
    ...     await update_shared_state()
    
    >>> limiter = CapacityLimiter(10)
    >>> async with limiter:
    ...     await process_request()  # Max 10 concurrent
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from types import TracebackType

__all__ = [
    "Lock",
    "RLock",
    "Semaphore",
    "BoundedSemaphore",
    "Event",
    "Condition",
    "Barrier",
    "CapacityLimiter",
]


class Lock:
    """Async mutual exclusion lock with timeout support.
    
    A coroutine-safe lock with optional timeout and statistics.
    
    Example:
        >>> lock = Lock()
        >>> async with lock:
        ...     # Exclusive access
        ...     await modify_resource()
        
        >>> # With timeout
        >>> acquired = await lock.acquire(timeout=5.0)
        >>> if acquired:
        ...     try:
        ...         await modify_resource()
        ...     finally:
        ...         lock.release()
    """
    
    __slots__ = ("_lock", "_owner", "_acquire_count", "_contention_count")
    
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[object] | None = None
        self._acquire_count = self._contention_count = 0
    
    @property
    def locked(self) -> bool:
        """Whether lock is currently held."""
        return self._lock.locked()
    
    @property
    def owner(self) -> asyncio.Task[object] | None:
        """Task currently holding the lock."""
        return self._owner
    
    @property
    def statistics(self) -> dict[str, int]:
        """Lock usage statistics."""
        return {
            "acquires": self._acquire_count,
            "contentions": self._contention_count,
        }
    
    async def acquire(self, *, timeout: float | None = None) -> bool:
        """Acquire the lock. Returns True if acquired, False on timeout."""
        if self._lock.locked():
            self._contention_count += 1
        
        try:
            if timeout is None:
                await self._lock.acquire()
            else:
                await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
            self._owner = asyncio.current_task()
            self._acquire_count += 1
            return True
        except asyncio.TimeoutError:
            return False
    
    def release(self) -> None:
        """Release the lock.
        
        Raises:
            RuntimeError: If lock not held by current task
        """
        if self._owner != asyncio.current_task():
            raise RuntimeError("Cannot release lock not owned by current task")
        self._owner = None
        self._lock.release()
    
    async def __aenter__(self) -> Lock:
        await self.acquire()
        return self
    
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()


class RLock:
    """Reentrant async lock.
    
    Can be acquired multiple times by the same task without deadlock.
    Must be released the same number of times it was acquired.
    
    Example:
        >>> lock = RLock()
        >>> async def recursive_op():
        ...     async with lock:
        ...         # Can acquire again in same task
        ...         async with lock:
        ...             await do_work()
    """
    
    __slots__ = ("_lock", "_owner", "_count")
    
    def __init__(self) -> None:
        self._lock, self._owner, self._count = asyncio.Lock(), None, 0
    
    @property
    def locked(self) -> bool:
        return self._count > 0
    
    async def acquire(self, *, timeout: float | None = None) -> bool:
        """Acquire the lock (reentrant)."""
        current = asyncio.current_task()
        if self._owner == current:
            self._count += 1
            return True
        
        try:
            await self._lock.acquire() if timeout is None else await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        self._owner, self._count = current, 1
        return True
    
    def release(self) -> None:
        """Release the lock once."""
        if self._owner != asyncio.current_task():
            raise RuntimeError("Cannot release lock not owned by current task")
        self._count -= 1
        if self._count == 0:
            self._owner = None
            self._lock.release()
    
    async def __aenter__(self) -> RLock:
        await self.acquire()
        return self
    
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()


class Semaphore:
    """Counting semaphore with timeout support.
    
    Controls access to a pool of resources. Allows up to N concurrent
    holders. Use for connection pools, rate limiting, etc.
    
    Example:
        >>> # Allow max 5 concurrent database connections
        >>> db_pool = Semaphore(5)
        >>> async with db_pool:
        ...     conn = await get_connection()
        ...     await query(conn)
    """
    
    __slots__ = ("_semaphore", "_initial_value", "_acquire_count")
    
    def __init__(self, value: int = 1) -> None:
        if value < 0:
            raise ValueError("Semaphore value must be >= 0")
        self._semaphore, self._initial_value, self._acquire_count = asyncio.Semaphore(value), value, 0
    
    @property
    def value(self) -> int:
        """Current semaphore value (available slots)."""
        return self._semaphore._value  # type: ignore[attr-defined]
    
    async def acquire(self, *, timeout: float | None = None) -> bool:
        """Acquire a slot. Returns True if acquired, False on timeout."""
        try:
            await self._semaphore.acquire() if timeout is None else await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
            self._acquire_count += 1
            return True
        except asyncio.TimeoutError:
            return False
    
    def release(self) -> None:
        """Release a slot."""
        self._semaphore.release()
    
    async def __aenter__(self) -> Semaphore:
        await self.acquire()
        return self
    
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()


class BoundedSemaphore(Semaphore):
    """Bounded semaphore that raises on over-release.
    
    Same as Semaphore but raises ValueError if release() is called
    more times than acquire(). Use when strict counting is required.
    """
    
    def release(self) -> None:
        """Release a slot.
        
        Raises:
            ValueError: If releasing more than acquired
        """
        if self.value >= self._initial_value:
            raise ValueError("BoundedSemaphore released too many times")
        self._semaphore.release()


class Event:
    """Async event for one-shot signaling.
    
    Tasks can wait for an event to be set. Once set, all waiters
    are released and future waits return immediately.
    
    Example:
        >>> ready = Event()
        >>> 
        >>> async def waiter():
        ...     await ready.wait()
        ...     print("Ready!")
        >>> 
        >>> async def signaler():
        ...     await do_initialization()
        ...     ready.set()  # Release all waiters
    """
    
    __slots__ = ("_event",)
    
    def __init__(self) -> None:
        self._event = asyncio.Event()
    
    def is_set(self) -> bool:
        """Check if event is set."""
        return self._event.is_set()
    
    def set(self) -> None:
        """Set the event, releasing all waiters."""
        self._event.set()
    
    def clear(self) -> None:
        """Clear the event, future waits will block."""
        self._event.clear()
    
    async def wait(self, *, timeout: float | None = None) -> bool:
        """Wait for event to be set. Returns True if set, False on timeout."""
        try:
            await self._event.wait() if timeout is None else await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


class Condition:
    """Async condition variable for complex synchronization.
    
    Allows tasks to wait for arbitrary conditions while holding a lock.
    
    Example:
        >>> cond = Condition()
        >>> items: list[int] = []
        >>> 
        >>> async def consumer():
        ...     async with cond:
        ...         while not items:
        ...             await cond.wait()
        ...         return items.pop()
        >>> 
        >>> async def producer(item: int):
        ...     async with cond:
        ...         items.append(item)
        ...         cond.notify()
    """
    
    __slots__ = ("_condition",)
    
    def __init__(self, lock: Lock | None = None) -> None:
        # Note: asyncio.Condition accepts asyncio.Lock, not our Lock
        self._condition = asyncio.Condition(lock._lock if lock else None)
    
    def locked(self) -> bool:
        """Check if underlying lock is held."""
        return self._condition.locked()
    
    async def acquire(self) -> bool:
        """Acquire the underlying lock."""
        return await self._condition.acquire()
    
    def release(self) -> None:
        """Release the underlying lock."""
        self._condition.release()
    
    async def wait(self, *, timeout: float | None = None) -> bool:
        """Wait for notification. Must hold lock. Releases while waiting, reacquires before return."""
        try:
            await self._condition.wait() if timeout is None else await asyncio.wait_for(self._condition.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
    
    async def wait_for(
        self,
        predicate: Callable[[], bool],
        *,
        timeout: float | None = None,
    ) -> bool:
        """Wait until predicate returns True."""
        try:
            await self._condition.wait_for(predicate) if timeout is None else await asyncio.wait_for(self._condition.wait_for(predicate), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
    
    def notify(self, n: int = 1) -> None:
        """Wake up n waiters."""
        self._condition.notify(n)
    
    def notify_all(self) -> None:
        """Wake up all waiters."""
        self._condition.notify_all()
    
    async def __aenter__(self) -> Condition:
        await self.acquire()
        return self
    
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()


@dataclass(slots=True)
class Barrier:
    """Synchronization barrier for multiple tasks.
    
    All tasks wait at the barrier until N tasks arrive, then all proceed.
    
    Example:
        >>> barrier = Barrier(3)
        >>> 
        >>> async def worker(id: int):
        ...     print(f"Worker {id} starting")
        ...     await barrier.wait()  # Wait for all workers
        ...     print(f"Worker {id} proceeding")
        >>> 
        >>> # All 3 workers must reach barrier before any proceed
    """
    
    parties: int
    _count: int = field(default=0, repr=False)
    _event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _generation: int = field(default=0, repr=False)
    
    def __post_init__(self) -> None:
        if self.parties < 1:
            raise ValueError("Barrier parties must be >= 1")
    
    @property
    def n_waiting(self) -> int:
        """Number of tasks currently waiting."""
        return self._count
    
    @property
    def broken(self) -> bool:
        """Whether barrier is broken (not implemented - always False)."""
        return False
    
    async def wait(self, *, timeout: float | None = None) -> int:
        """Wait at barrier until all parties arrive.
        
        Returns:
            Index of this task (0 to parties-1)
        
        Raises:
            asyncio.TimeoutError: If timeout expires
        """
        async with self._lock:
            gen = self._generation
            index = self._count
            self._count += 1
            
            if self._count == self.parties:
                # Last arrival - release all
                self._count = 0
                self._generation += 1
                self._event.set()
                self._event = asyncio.Event()  # Reset for next use
                return index
        
        # Wait for release
        if timeout is None:
            while self._generation == gen:
                await self._event.wait()
        else:
            deadline = time.monotonic() + timeout
            while self._generation == gen:
                if (remaining := deadline - time.monotonic()) <= 0:
                    raise asyncio.TimeoutError()
                await asyncio.wait_for(self._event.wait(), timeout=remaining)
        return index
    
    async def reset(self) -> None:
        """Reset the barrier, releasing any waiting tasks."""
        async with self._lock:
            self._count, self._generation = 0, self._generation + 1
            self._event.set()
            self._event = asyncio.Event()


class CapacityLimiter:
    """Limit concurrent access to a resource.
    
    Similar to Semaphore but with better ergonomics for limiting
    concurrent operations. Tracks borrowers and provides statistics.
    
    Example:
        >>> # Limit to 10 concurrent API calls
        >>> limiter = CapacityLimiter(10)
        >>> 
        >>> async def api_call(url: str):
        ...     async with limiter:
        ...         return await fetch(url)
        >>> 
        >>> # Check usage
        >>> print(f"Active: {limiter.borrowed}/{limiter.total}")
    """
    
    __slots__ = ("_total", "_semaphore", "_borrowed", "_borrowers")
    
    def __init__(self, total: int) -> None:
        if total < 1:
            raise ValueError("CapacityLimiter total must be >= 1")
        self._total, self._semaphore, self._borrowed = total, asyncio.Semaphore(total), 0
        self._borrowers: set[asyncio.Task[object]] = set()
    
    @property
    def total(self) -> int:
        """Total capacity."""
        return self._total
    
    @property
    def borrowed(self) -> int:
        """Currently borrowed (in use)."""
        return self._borrowed
    
    @property
    def available(self) -> int:
        """Available capacity."""
        return self._total - self._borrowed
    
    @property
    def statistics(self) -> dict[str, int]:
        """Usage statistics."""
        return {
            "total": self._total,
            "borrowed": self._borrowed,
            "available": self.available,
            "borrowers": len(self._borrowers),
        }
    
    async def acquire(self, *, timeout: float | None = None) -> bool:
        """Borrow capacity. Returns True if acquired, False on timeout."""
        try:
            await self._semaphore.acquire() if timeout is None else await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        self._borrowed += 1
        if task := asyncio.current_task():
            self._borrowers.add(task)
        return True
    
    def release(self) -> None:
        """Return borrowed capacity."""
        if task := asyncio.current_task():
            self._borrowers.discard(task)
        self._borrowed -= 1
        self._semaphore.release()
    
    def set_total(self, new_total: int) -> None:
        """Adjust total capacity dynamically. Increasing releases slots immediately; decreasing takes effect as slots are released."""
        if new_total < 1:
            raise ValueError("Total must be >= 1")
        delta = new_total - self._total
        self._total = new_total
        for _ in range(max(0, delta)):
            self._semaphore.release()
    
    async def __aenter__(self) -> CapacityLimiter:
        await self.acquire()
        return self
    
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()
