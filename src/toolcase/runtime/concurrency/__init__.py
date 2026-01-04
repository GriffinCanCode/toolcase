"""Structured concurrency primitives for async/parallel operations.

This module provides a comprehensive toolkit for managing concurrent
operations with proper cancellation, resource cleanup, and error propagation.

Organization:
    - primitives/: Task management and synchronization (TaskGroup, Lock, etc.)
    - execution/: Pools and wait strategies (ThreadPool, race, gather, etc.)
    - streams/: Async stream combinators (merge, buffer, throttle, etc.)
    - interop/: Sync/async bridging (run_sync, run_async, adapters)
    - facade.py: Unified Concurrency class for easy access

Primary Usage:
    >>> from toolcase.runtime.concurrency import Concurrency
    >>> 
    >>> # Structured task group
    >>> async with Concurrency.task_group() as tg:
    ...     tg.spawn(fetch_data, "url1")
    ...     tg.spawn(fetch_data, "url2")
    >>> 
    >>> # Race multiple operations
    >>> result = await Concurrency.race(provider_a(), provider_b())
    >>> 
    >>> # Parallel map with limit
    >>> results = await Concurrency.map(process, items, limit=10)
    >>> 
    >>> # Run blocking code from async
    >>> data = await Concurrency.to_thread(blocking_io)

Direct Imports (when you need specific primitives):
    >>> from toolcase.runtime.concurrency import TaskGroup, Lock, race, map_async
"""

from __future__ import annotations

# Unified facade - primary import
from .facade import (
    Concurrency,
    ConcurrencyConfig,
    # Convenience aliases
    task_group,
    race,
    gather,
    first_success,
    map_async,
    to_thread,
    run_sync,
    sleep,
)

# Primitives - task management
from .primitives import (
    TaskGroup,
    TaskHandle,
    TaskState,
    CancelScope,
    shield,
    checkpoint,
    current_task,
    spawn,
    cancellable,
)

# Primitives - synchronization
from .primitives import (
    Lock,
    RLock,
    Semaphore,
    BoundedSemaphore,
    Event,
    Condition,
    Barrier,
    CapacityLimiter,
)

# Execution - pools
from .execution import (
    ThreadPool,
    ProcessPool,
    run_in_thread,
    run_in_process,
    shutdown_default_pools,
    threadpool,
    processpool,
)

# Execution - wait strategies
from .execution import (
    race as race_coros,  # Aliased to avoid conflict with facade.race
    race_with_index,
    gather as gather_coros,
    gather_settled,
    all_settled,
    first_success as first_success_coros,
    map_async as map_async_coros,
    map_async_unordered,
    wait_any,
    wait_all,
    retry_until_success,
    SettledStatus,
    Settled,
    WaitResult,
)

# Streams
from .streams import (
    merge_streams,
    interleave_streams,
    buffer_stream,
    backpressure_stream,
    BackpressureController,
    throttle_stream,
    batch_stream,
    timeout_stream,
    take_stream,
    skip_stream,
    filter_stream,
    map_stream,
    flatten_stream,
    enumerate_stream,
    zip_streams,
    chain_streams,
    StreamMerger,
)

# Interop
from .interop import (
    run_sync as run_sync_coro,
    run_async as run_async_func,
    from_thread,
    from_thread_nowait,
    set_thread_loop,
    clear_thread_loop,
    to_thread as to_thread_func,
    AsyncAdapter,
    SyncAdapter,
    async_to_sync,
    sync_to_async,
    ThreadContext,
    shutdown_executor,
)

__all__ = [
    # === Unified Facade ===
    "Concurrency",
    "ConcurrencyConfig",
    # Facade convenience aliases
    "task_group",
    "race",
    "gather",
    "first_success",
    "map_async",
    "to_thread",
    "run_sync",
    
    # === Primitives: Task Management ===
    "TaskGroup",
    "TaskHandle",
    "TaskState",
    "CancelScope",
    "shield",
    "checkpoint",
    "current_task",
    "spawn",
    "cancellable",
    
    # === Primitives: Synchronization ===
    "Lock",
    "RLock",
    "Semaphore",
    "BoundedSemaphore",
    "Event",
    "Condition",
    "Barrier",
    "CapacityLimiter",
    
    # === Execution: Pools ===
    "ThreadPool",
    "ProcessPool",
    "run_in_thread",
    "run_in_process",
    "shutdown_default_pools",
    "threadpool",
    "processpool",
    
    # === Execution: Wait Strategies ===
    "race_coros",
    "race_with_index",
    "gather_coros",
    "gather_settled",
    "all_settled",
    "first_success_coros",
    "map_async_coros",
    "map_async_unordered",
    "wait_any",
    "wait_all",
    "retry_until_success",
    "SettledStatus",
    "Settled",
    "WaitResult",
    
    # === Streams ===
    "merge_streams",
    "interleave_streams",
    "buffer_stream",
    "backpressure_stream",
    "BackpressureController",
    "throttle_stream",
    "batch_stream",
    "timeout_stream",
    "take_stream",
    "skip_stream",
    "filter_stream",
    "map_stream",
    "flatten_stream",
    "enumerate_stream",
    "zip_streams",
    "chain_streams",
    "StreamMerger",
    
    # === Interop ===
    "run_sync_coro",
    "run_async_func",
    "from_thread",
    "from_thread_nowait",
    "set_thread_loop",
    "clear_thread_loop",
    "to_thread_func",
    "AsyncAdapter",
    "SyncAdapter",
    "async_to_sync",
    "sync_to_async",
    "ThreadContext",
    "shutdown_executor",
]
