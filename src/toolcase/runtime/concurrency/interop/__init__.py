"""Sync/async interoperability utilities.

Bridge between synchronous and asynchronous code:
- run_sync: Run async from sync context
- run_async/to_thread: Run sync from async context
- Adapters: Convert between sync and async callables
"""

from .bridge import (
    run_sync,
    run_async,
    from_thread,
    from_thread_nowait,
    set_thread_loop,
    clear_thread_loop,
    to_thread,
    configure_thread_limit,
    AsyncAdapter,
    SyncAdapter,
    async_to_sync,
    sync_to_async,
    ThreadContext,
    shutdown_executor,
)

__all__ = [
    "run_sync",
    "run_async",
    "from_thread",
    "from_thread_nowait",
    "set_thread_loop",
    "clear_thread_loop",
    "to_thread",
    "configure_thread_limit",
    "AsyncAdapter",
    "SyncAdapter",
    "async_to_sync",
    "sync_to_async",
    "ThreadContext",
    "shutdown_executor",
]
