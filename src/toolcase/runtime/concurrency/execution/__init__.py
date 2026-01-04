"""Execution patterns: pools and wait strategies.

Execution provides patterns for running concurrent operations:
- Pools: ThreadPool, ProcessPool for CPU-bound/blocking work
- Wait strategies: race, gather, map_async for coordinating operations
"""

from .pool import (
    ThreadPool,
    ProcessPool,
    run_in_thread,
    run_in_process,
    shutdown_default_pools,
    threadpool,
    processpool,
    DEFAULT_THREAD_WORKERS,
    DEFAULT_PROCESS_WORKERS,
)

from .wait import (
    race,
    race_with_index,
    gather,
    gather_settled,
    all_settled,
    first_success,
    map_async,
    map_async_unordered,
    wait_any,
    wait_all,
    retry_until_success,
    SettledStatus,
    Settled,
    WaitResult,
)

__all__ = [
    # Pools
    "ThreadPool",
    "ProcessPool",
    "run_in_thread",
    "run_in_process",
    "shutdown_default_pools",
    "threadpool",
    "processpool",
    "DEFAULT_THREAD_WORKERS",
    "DEFAULT_PROCESS_WORKERS",
    # Wait strategies
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
    "SettledStatus",
    "Settled",
    "WaitResult",
]
