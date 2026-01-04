# Concurrency Module

Structured concurrency primitives for async/parallel operations with proper cancellation, resource cleanup, and error propagation.

## Quick Start

```python
from toolcase.runtime.concurrency import Concurrency

# Structured task group - tasks auto-cancel on failure
async with Concurrency.task_group() as tg:
    tg.spawn(fetch_data("url1"))
    tg.spawn(fetch_data("url2"))

# Race - first to complete wins
result = await Concurrency.race(api_a(), api_b(), cache())

# Parallel map with concurrency limit
results = await Concurrency.map(process, items, limit=10)

# Run blocking code from async
data = await Concurrency.to_thread(blocking_io)

# Run async from sync context
result = Concurrency.run_sync(async_operation())
```

## Module Structure

```
concurrency/
├── __init__.py          # Re-exports everything
├── facade.py            # Unified Concurrency class (PRIMARY IMPORT)
├── primitives/          # Core building blocks
│   ├── task.py          # TaskGroup, TaskHandle, CancelScope
│   └── sync.py          # Lock, Semaphore, Event, Barrier
├── execution/           # Execution patterns
│   ├── pool.py          # ThreadPool, ProcessPool
│   └── wait.py          # race, gather, map_async, first_success
├── streams/             # Async stream utilities
│   └── combinators.py   # merge, buffer, throttle, batch
└── interop/             # Sync/async bridging
    └── bridge.py        # run_sync, run_async, adapters
```

## Import Patterns

### Primary: Unified Facade (Recommended)

```python
from toolcase.runtime.concurrency import Concurrency

# All operations via class methods
async with Concurrency.task_group() as tg: ...
await Concurrency.race(op1(), op2())
await Concurrency.map(fn, items, limit=10)
await Concurrency.to_thread(blocking)
Concurrency.run_sync(async_coro())
```

### Direct Imports (When needed)

```python
from toolcase.runtime.concurrency import (
    # Primitives
    TaskGroup, Lock, Semaphore, Event, Barrier, CapacityLimiter,
    # Wait strategies
    race, gather, first_success, map_async,
    # Pools
    ThreadPool, ProcessPool, run_in_thread,
    # Streams  
    merge_streams, buffer_stream, throttle_stream,
    # Interop
    run_sync, to_thread, AsyncAdapter,
)
```

## Feature Reference

### Task Management

```python
# Structured task group
async with Concurrency.task_group() as tg:
    h1 = tg.spawn(fetch_user(1))
    h2 = tg.spawn(fetch_user(2))
# All tasks complete or cancelled together

# Cancel scope with timeout
async with Concurrency.cancel_scope(timeout=5.0) as scope:
    await long_operation()
    if scope.cancel_called:
        print("Timed out!")

# Shield from cancellation
result = await Concurrency.shield(critical_operation())

# Checkpoints for cooperative cancellation
async def process_many(items):
    for item in items:
        process(item)
        await Concurrency.checkpoint()
```

### Wait Strategies

```python
# Race - first to complete wins, others cancelled
result = await Concurrency.race(
    fetch_from_api_a(),
    fetch_from_api_b(),
    fetch_from_cache(),
    timeout=5.0,
)

# Gather - wait for all
results = await Concurrency.gather(op1(), op2(), op3())

# First success - skip failures
result = await Concurrency.first_success(
    unreliable_api_a(),
    unreliable_api_b(),
    fallback_api(),
)

# Parallel map with limit
results = await Concurrency.map(
    process_item, items, limit=10
)

# Retry with backoff
result = await Concurrency.retry(
    lambda: fetch_data(),
    max_attempts=3,
    delay=1.0,
    backoff=2.0,
)
```

### Thread/Process Pools

```python
# Run blocking code in thread
data = await Concurrency.to_thread(read_file, path)

# Run CPU-bound in process
result = await Concurrency.to_process(heavy_compute, data)

# Managed thread pool
async with Concurrency.thread_pool(max_workers=4) as pool:
    result = await pool.run(blocking_function)

# Managed process pool
async with Concurrency.process_pool(max_workers=4) as pool:
    result = await pool.run(cpu_intensive)
```

### Synchronization

```python
# Mutex lock
lock = Concurrency.lock()
async with lock:
    await modify_shared_resource()

# Semaphore for resource limiting
sem = Concurrency.semaphore(5)  # Max 5 concurrent
async with sem:
    await use_resource()

# Capacity limiter (friendlier API)
limiter = Concurrency.limiter(10)
async with limiter:
    await api_call()

# Event for signaling
ready = Concurrency.event()
ready.set()  # Signal
await ready.wait()  # Wait for signal

# Barrier for synchronization
barrier = Concurrency.barrier(3)
await barrier.wait()  # Wait for 3 tasks
```

### Stream Utilities

```python
# Merge multiple streams
async for item in Concurrency.merge(stream1, stream2):
    process(item)

# Buffer for smoothing
async for item in Concurrency.buffer(slow_producer, maxsize=100):
    fast_consumer(item)

# Rate limit
async for item in Concurrency.throttle(fast_source, rate=10):
    api_call(item)  # Max 10/second

# Batch processing
async for batch in Concurrency.batch(items, size=100, timeout=5.0):
    bulk_insert(batch)
```

### Sync/Async Interop

```python
# Async from sync context
result = Concurrency.run_sync(async_operation())

# Sync from async (in thread)
result = await Concurrency.run_async(blocking_function)

# Adapters
async_fn = Concurrency.async_adapter(sync_function)
result = await async_fn(args)

sync_fn = Concurrency.sync_adapter(async_function)
result = sync_fn(args)

# Thread context for worker threads
async with Concurrency.thread_context():
    # Worker threads can now use from_thread()
    await run_threaded_work()
```

## Configuration

```python
# Configure global defaults
Concurrency.configure(
    default_timeout=30.0,
    default_pool_size=10,
    default_thread_workers=8,
    default_process_workers=4,
)
```

## Cleanup

```python
# At application shutdown
Concurrency.shutdown()
```

## Design Philosophy

- **Structured concurrency**: Tasks don't outlive their scope
- **Fail-fast**: First exception cancels sibling tasks
- **Cancellation-safe**: Proper cleanup on cancellation
- **Type-safe**: Full typing support with generics
- **Zero external dependencies**: Pure asyncio (Python 3.11+)
