"""Structured concurrency primitives for async/parallel operations.

This module provides a comprehensive toolkit for managing concurrent
operations with proper cancellation, resource cleanup, and error propagation.

Key Components:
    - TaskGroup: Structured task management with automatic cancellation
    - Synchronization primitives: Lock, Semaphore, Event, Barrier
    - Pool executors: ThreadPool, ProcessPool for CPU-bound work
    - Wait strategies: race, gather_all, map_async, first_success
    - Stream utilities: merge, interleave, buffer, throttle
    - Sync/async interop: run_sync, run_async, from_thread

Design Philosophy:
    - Structured concurrency: Tasks don't outlive their scope
    - Fail-fast: First exception cancels sibling tasks
    - Cancellation-safe: Proper cleanup on cancellation
    - Type-safe: Full typing support with generics
    - Zero external dependencies: Pure asyncio (Python 3.11+)

Example:
    >>> from toolcase.runtime.concurrency import TaskGroup, gather, race
    >>> 
    >>> # Structured task group
    >>> async with TaskGroup() as tg:
    ...     tg.spawn(fetch_data, "url1")
    ...     tg.spawn(fetch_data, "url2")
    ...     tg.spawn(fetch_data, "url3")
    >>> # All tasks complete or all cancelled on error
    >>> 
    >>> # Race multiple operations
    >>> result = await race(provider_a(), provider_b(), provider_c())
    >>> 
    >>> # Parallel map with concurrency limit
    >>> results = await map_async(process, items, limit=10)
"""

from __future__ import annotations

# Task management
from .task import (
    TaskGroup,
    TaskHandle,
    TaskState,
    CancelScope,
    shield,
    checkpoint,
    current_task,
    spawn,
)

# Synchronization primitives
from .sync import (
    Lock,
    RLock,
    Semaphore,
    BoundedSemaphore,
    Event,
    Condition,
    Barrier,
    CapacityLimiter,
)

# Pool executors
from .pool import (
    ThreadPool,
    ProcessPool,
    run_in_thread,
    run_in_process,
)

# Wait strategies
from .wait import (
    race,
    gather,
    gather_settled,
    first_success,
    map_async,
    all_settled,
    WaitResult,
    Settled,
)

# Stream utilities
from .stream import (
    merge_streams,
    interleave_streams,
    buffer_stream,
    throttle_stream,
    batch_stream,
    timeout_stream,
    StreamMerger,
)

# Sync/async interop
from .interop import (
    run_sync,
    run_async,
    from_thread,
    to_thread,
    AsyncAdapter,
    SyncAdapter,
)

__all__ = [
    # Task management
    "TaskGroup",
    "TaskHandle",
    "TaskState",
    "CancelScope",
    "shield",
    "checkpoint",
    "current_task",
    "spawn",
    # Sync primitives
    "Lock",
    "RLock",
    "Semaphore",
    "BoundedSemaphore",
    "Event",
    "Condition",
    "Barrier",
    "CapacityLimiter",
    # Pools
    "ThreadPool",
    "ProcessPool",
    "run_in_thread",
    "run_in_process",
    # Wait strategies
    "race",
    "gather",
    "gather_settled",
    "first_success",
    "map_async",
    "all_settled",
    "WaitResult",
    "Settled",
    # Streams
    "merge_streams",
    "interleave_streams",
    "buffer_stream",
    "throttle_stream",
    "batch_stream",
    "timeout_stream",
    "StreamMerger",
    # Interop
    "run_sync",
    "run_async",
    "from_thread",
    "to_thread",
    "AsyncAdapter",
    "SyncAdapter",
]
