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
    limiter = Concurrency.limiter(10)

THREAD/PROCESS:
    # Run blocking in thread
    data = await Concurrency.to_thread(blocking_io)
    
    # Run CPU-bound in process
    result = await Concurrency.to_process(heavy_compute)

SYNC/ASYNC INTEROP:
    # Async from sync
    result = Concurrency.run_sync(async_operation())
    
    # Adapters
    async_fn = Concurrency.async_adapter(sync_function)
    sync_fn = Concurrency.sync_adapter(async_function)

RELATED TOPICS:
    toolcase help agents     Agentic composition
    toolcase help retry      Retry with backoff
"""
