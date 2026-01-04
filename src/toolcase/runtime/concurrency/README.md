# Concurrency Module

Structured concurrency primitives for async/parallel operations with proper cancellation, resource cleanup, and error propagation.

## Design Philosophy

- **Structured Concurrency**: Tasks don't outlive their scope
- **Fail-Fast**: First exception cancels sibling tasks  
- **Cancellation-Safe**: Proper cleanup on cancellation
- **Type-Safe**: Full typing support with generics
- **Zero External Dependencies**: Pure Python 3.11+ asyncio

## Quick Start

```python
from toolcase.runtime.concurrency import (
    TaskGroup, race, gather, map_async,
    Lock, Semaphore, CapacityLimiter,
    ThreadPool, run_in_thread,
)

# Structured task management
async with TaskGroup() as tg:
    tg.spawn(fetch_data, "url1")
    tg.spawn(fetch_data, "url2")
    tg.spawn(fetch_data, "url3")
# All tasks complete or all cancelled on error

# Race multiple operations
result = await race(provider_a(), provider_b(), provider_c())

# Parallel map with concurrency limit
results = await map_async(process, items, limit=10)
```

## Module Overview

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `task` | Task lifecycle management | `TaskGroup`, `TaskHandle`, `CancelScope` |
| `sync` | Synchronization primitives | `Lock`, `Semaphore`, `Event`, `Barrier` |
| `pool` | Thread/process execution | `ThreadPool`, `ProcessPool`, `run_in_thread` |
| `wait` | Wait strategies | `race`, `gather`, `first_success`, `map_async` |
| `stream` | Async iterator utilities | `merge_streams`, `throttle_stream`, `batch_stream` |
| `interop` | Sync/async bridging | `run_sync`, `to_thread`, `AsyncAdapter` |

---

## Task Management (`task.py`)

### TaskGroup

Manages multiple concurrent tasks as a unit with automatic cancellation.

```python
async with TaskGroup() as tg:
    # Spawn tasks that are managed together
    handle1 = tg.spawn(fetch_user, user_id)
    handle2 = tg.spawn(fetch_orders, user_id)
    handle3 = tg.spawn(fetch_preferences, user_id)
# All complete, or if one fails, all are cancelled

# Access results via handles
user = handle1.result()
orders = handle2.result()
```

### TaskHandle

Handle to a spawned task with state access:

```python
handle = tg.spawn(long_operation())

# Check state
print(handle.state)  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
print(handle.done)   # True/False

# Wait for completion
result = await handle.wait()

# Cancel if needed
handle.cancel()
```

### CancelScope

Fine-grained cancellation control with timeouts:

```python
async with CancelScope(timeout=5.0) as scope:
    await long_operation()
    if scope.cancel_called:
        print("Timed out!")
```

### Utilities

```python
# Shield from cancellation (use sparingly!)
result = await shield(critical_operation())

# Cooperative cancellation checkpoint
await checkpoint()  # Allow pending cancellations

# Get current task
task = current_task()
```

---

## Synchronization Primitives (`sync.py`)

### Lock

Mutual exclusion with timeout support:

```python
lock = Lock()

async with lock:
    # Exclusive access
    await modify_shared_state()

# With timeout
if await lock.acquire(timeout=5.0):
    try:
        await work()
    finally:
        lock.release()
```

### RLock

Reentrant lock for recursive operations:

```python
lock = RLock()

async def recursive_op():
    async with lock:
        # Can acquire again in same task
        async with lock:
            await do_work()
```

### Semaphore

Control concurrent access to limited resources:

```python
# Max 5 concurrent database connections
db_pool = Semaphore(5)

async with db_pool:
    conn = await get_connection()
    await query(conn)
```

### Event

One-shot signaling between tasks:

```python
ready = Event()

async def waiter():
    await ready.wait()
    print("Ready!")

async def signaler():
    await do_initialization()
    ready.set()  # Release all waiters
```

### Barrier

Synchronization point for multiple tasks:

```python
barrier = Barrier(3)  # Wait for 3 tasks

async def worker(id: int):
    print(f"Worker {id} starting")
    await barrier.wait()  # Wait for all
    print(f"Worker {id} proceeding")
```

### CapacityLimiter

Limit concurrent access with usage tracking:

```python
limiter = CapacityLimiter(10)

async with limiter:
    await process_request()

# Check usage
print(f"Active: {limiter.borrowed}/{limiter.total}")
```

---

## Pool Executors (`pool.py`)

### ThreadPool

For blocking I/O and sync operations:

```python
async with ThreadPool(max_workers=4) as pool:
    # Run blocking function in thread
    data = await pool.run(read_large_file, path)
    
    # Multiple in parallel
    results = await asyncio.gather(
        *(pool.run(process, item) for item in items)
    )
```

### ProcessPool

For CPU-bound work (bypasses GIL):

```python
async with ProcessPool(4) as pool:
    # Heavy computation in separate process
    result = await pool.run(heavy_computation, data)
```

### Convenience Functions

```python
# One-off thread execution
data = await run_in_thread(blocking_io_function, arg1)

# One-off process execution  
result = await run_in_process(cpu_intensive_function, data)
```

### Decorators

```python
@threadpool
def blocking_operation(x: int) -> int:
    time.sleep(1)
    return x * 2

# Now callable as async
result = await blocking_operation(5)

@processpool
def cpu_intensive(data: bytes) -> bytes:
    return expensive_computation(data)
```

---

## Wait Strategies (`wait.py`)

### race

First to complete wins, others cancelled:

```python
result = await race(
    fetch_from_api_a(),
    fetch_from_api_b(),
    fetch_from_cache(),
    timeout=5.0,
)
```

### gather / gather_settled

Wait for all operations:

```python
# Standard gather (raises on first error)
results = await gather(op1(), op2(), op3())

# Settled (never raises, returns status)
results = await gather_settled(op1(), op2(), op3())
for r in results:
    if r.is_fulfilled:
        print(f"Success: {r.value}")
    else:
        print(f"Failed: {r.error}")
```

### first_success

First successful result, ignoring failures:

```python
result = await first_success(
    unreliable_api_a(),
    unreliable_api_b(),
    fallback_api(),
)
```

### map_async

Parallel map with concurrency limit:

```python
# Process 100 items, max 10 concurrent
results = await map_async(
    fetch_data,
    urls,
    limit=10,
)
```

---

## Stream Utilities (`stream.py`)

### merge_streams

Combine multiple streams into one:

```python
async for item in merge_streams(source1, source2, source3):
    process(item)  # Items arrive as available
```

### throttle_stream

Rate limit stream consumption:

```python
# Max 10 items per second
async for item in throttle_stream(fast_source, rate=10.0):
    await api_call(item)
```

### batch_stream

Group items into batches:

```python
# Process in batches of 100
async for batch in batch_stream(items, size=100, timeout=5.0):
    await bulk_insert(batch)
```

### buffer_stream

Pre-fetch items for smoother consumption:

```python
async for item in buffer_stream(slow_producer, maxsize=100):
    fast_process(item)
```

### Other Operations

```python
# Take first n
async for x in take_stream(stream, 10): ...

# Skip first n  
async for x in skip_stream(stream, 5): ...

# Filter
async for x in filter_stream(stream, lambda x: x > 0): ...

# Map
async for x in map_stream(stream, transform): ...

# Enumerate
async for i, x in enumerate_stream(stream): ...
```

---

## Sync/Async Interop (`interop.py`)

### run_sync

Run async code from sync context:

```python
# From sync code
result = run_sync(async_operation())

# Handles nested loops (FastAPI, Jupyter)
```

### to_thread / run_async

Run sync code from async context:

```python
# From async code
result = await to_thread(blocking_io)
result = await run_async(sync_function, arg1, arg2)
```

### Adapters

```python
# Wrap sync as async
async_io = AsyncAdapter(sync_io_function)
content = await async_io("/etc/hosts")

# Wrap async as sync
sync_fetch = SyncAdapter(async_fetch_function)
content = sync_fetch("https://example.com")
```

### Decorators

```python
@sync_to_async
def blocking_operation(data: bytes) -> bytes:
    return expensive_sync_process(data)

# Now callable as async
result = await blocking_operation(data)

@async_to_sync  
async def fetch_data(url: str) -> dict:
    ...

# Now callable as sync
data = fetch_data("https://api.example.com")
```

### Thread Context

Enable async calls from worker threads:

```python
async def main():
    async with ThreadContext():
        # Worker threads can now use from_thread()
        await run_threaded_work()

def worker_thread():
    # Inside thread pool
    result = from_thread(async_operation())
```

---

## Integration with Toolcase

The concurrency module integrates with existing toolcase patterns:

```python
from toolcase.runtime import TaskGroup, map_async, CapacityLimiter
from toolcase.runtime.pipeline import parallel

# Use with pipeline tools
async with TaskGroup() as tg:
    tg.spawn(tool1.arun_result, params1)
    tg.spawn(tool2.arun_result, params2)

# Rate-limited tool execution
limiter = CapacityLimiter(10)

async def limited_call(tool, params):
    async with limiter:
        return await tool.arun_result(params)

results = await map_async(
    lambda p: limited_call(my_tool, p),
    param_list,
    limit=20,
)
```

---

## Best Practices

1. **Use TaskGroup for related tasks**: Ensures proper cleanup and cancellation
2. **Set concurrency limits**: Avoid overwhelming resources with `map_async(limit=N)` or `CapacityLimiter`
3. **Handle cancellation**: Use `checkpoint()` in long-running operations
4. **Choose the right pool**: `ThreadPool` for I/O, `ProcessPool` for CPU
5. **Prefer structured concurrency**: Avoid bare `asyncio.create_task()` when possible
