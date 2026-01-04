CONCURRENCY = """
TOPIC: concurrency
==================

Structured concurrency primitives for async operations.

UNIFIED FACADE:
    from toolcase import Concurrency
    
    # All operations via class methods
    async with Concurrency.task_group() as tg:
        tg.spawn(fetch_data("url1"))
        tg.spawn(fetch_data("url2"))

TASK GROUPS (Structured concurrency):
    async with Concurrency.task_group() as tg:
        h1 = tg.spawn(task1())
        h2 = tg.spawn(task2())
    # All complete or all cancelled together

CANCEL SCOPES:
    from toolcase import CancelScope
    
    # Timeout with structured cancellation
    async with CancelScope(timeout=5.0) as scope:
        result = await slow_operation()
    
    if scope.cancelled:
        print("Operation timed out")

WAIT STRATEGIES:
    # Race - first wins, others cancelled
    result = await Concurrency.race(api_a(), api_b())
    
    # Gather - wait for all
    results = await Concurrency.gather(op1(), op2(), op3())
    
    # First success - skip failures
    result = await Concurrency.first_success(
        unreliable_a(), unreliable_b()
    )
    
    # Parallel map with limit
    results = await Concurrency.map(process, items, limit=10)

SYNC PRIMITIVES:
    lock = Concurrency.lock()
    semaphore = Concurrency.semaphore(5)
    event = Concurrency.event()
    barrier = Concurrency.barrier(3)
    limiter = Concurrency.limiter(10)  # CapacityLimiter

UTILITIES:
    from toolcase.runtime.concurrency import sleep
    
    # Async sleep
    await sleep(1.0)
    
    # Checkpoint for cooperative cancellation
    from toolcase.runtime.concurrency import checkpoint
    await checkpoint()

THREAD/PROCESS:
    from toolcase import to_thread
    
    # Run blocking in thread
    data = await to_thread(blocking_io)
    
    # Run CPU-bound in process
    result = await Concurrency.to_process(heavy_compute)

SYNC/ASYNC INTEROP:
    from toolcase import run_sync, AsyncAdapter, SyncAdapter
    
    # Async from sync
    result = run_sync(async_operation())
    
    # Adapters
    async_fn = AsyncAdapter(sync_function)
    sync_fn = SyncAdapter(async_function)

STREAM COMBINATORS:
    from toolcase.runtime.concurrency import (
        merge_streams, buffer_stream, throttle_stream, batch_stream
    )
    
    # Merge multiple async iterators
    async for item in merge_streams(stream1, stream2):
        print(item)
    
    # Buffer stream for backpressure
    async for item in buffer_stream(slow_stream, maxsize=100):
        process(item)

RELATED TOPICS:
    toolcase help agents     Agentic composition
    toolcase help retry      Retry with backoff
    toolcase help batch      Batch execution
"""
