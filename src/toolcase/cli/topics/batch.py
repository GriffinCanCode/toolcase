BATCH = """
TOPIC: batch
============

Batch execution for running tools against multiple parameter sets.

TOOL BATCH METHOD:
    from toolcase import BaseTool, BatchConfig
    
    # Run a tool against multiple parameter sets concurrently
    params_list = [SearchParams(query=q) for q in ["python", "rust", "go"]]
    results = await search_tool.batch_run(params_list)
    
    # Access results
    print(f"Success rate: {results.success_rate:.0%}")
    print(f"Total duration: {results.total_duration_ms:.0f}ms")
    
    for item in results.successes:
        print(f"[{item.index}] {item.value[:50]}...")
    
    for item in results.failures:
        print(f"[{item.index}] FAILED: {item.error}")

BATCH CONFIGURATION:
    from toolcase import BatchConfig
    
    config = BatchConfig(
        concurrency=5,           # Max parallel executions (default: 10)
        fail_fast=False,         # Stop on first failure (default: False)
        timeout_per_item=30.0,   # Timeout per item in seconds
        collect_errors=True,     # Collect errors vs raise immediately
    )
    
    results = await tool.batch_run(params_list, config)

BATCH RESULT ATTRIBUTES:
    results.items           All BatchItems (success or failure)
    results.successes       Only successful items
    results.failures        Only failed items
    results.success_rate    Ratio of successes (0.0 to 1.0)
    results.total_duration_ms  Total execution time
    results.is_partial      True if some items failed

BATCH ITEM:
    item.index      Original position in params_list
    item.params     The input parameters
    item.value      Result value (if success)
    item.error      Error message (if failure)
    item.is_ok      True if successful
    item.duration_ms  Execution time for this item

STANDALONE BATCH FUNCTION:
    from toolcase import batch_execute, batch_execute_sync
    
    # Async batch execution
    results = await batch_execute(
        tool,
        params_list,
        BatchConfig(concurrency=3),
    )
    
    # Sync wrapper
    results = batch_execute_sync(tool, params_list)

STREAMING BATCH PROGRESS:
    from toolcase.runtime.batch import batch_execute_stream, BatchEventKind
    
    # Stream progress events as items complete
    async for event in batch_execute_stream(tool, params_list, config):
        match event.kind:
            case BatchEventKind.START:
                print(f"Starting {event.total} items...")
            case BatchEventKind.ITEM:
                item = event.item
                status = "OK" if item.is_ok else f"FAIL: {item.error}"
                print(f"[{event.completed}/{event.total}] {status}")
            case BatchEventKind.COMPLETE:
                print(f"Done: {event.batch_result.success_rate:.0%} success")

IDEMPOTENT BATCH (EXACTLY-ONCE SEMANTICS):
    from toolcase import (
        batch_execute_idempotent,
        IdempotentBatchConfig,
        BatchRetryPolicy,
        BatchRetryStrategy,
        MemoryCache,  # Reuses existing cache infrastructure!
    )
    
    # Configure idempotency and batch-level retry
    config = IdempotentBatchConfig(
        concurrency=10,
        batch_id="order-batch-123",      # Unique batch identifier
        retry_policy=BatchRetryPolicy(
            max_retries=3,               # Batch-level retry attempts
            failure_threshold=0.3,       # Retry if >30% failed
            strategy=BatchRetryStrategy.FAILED_ONLY,
        ),
        idempotency_ttl=3600,            # TTL for cached results
    )
    
    # Execute with exactly-once guarantees
    # Uses existing cache system (Memory, Redis, Memcached)
    cache = MemoryCache(default_ttl=3600)  # Or use get_cache()
    results = await batch_execute_idempotent(tool, params, config, cache=cache)
    
    # Access idempotency metadata
    print(f"Cache hits: {results.cache_hit_rate:.0%}")
    print(f"Batch attempts: {results.batch_attempts}")
    print(f"Was retried: {results.was_retried}")
    print(f"Cache stats: {cache.stats()}")

BATCH RETRY STRATEGIES:
    FAILED_ONLY     Retry only failed items (default)
    ENTIRE_BATCH    Retry entire batch on threshold breach

CACHE INTEGRATION:
    Idempotency uses existing toolcase cache infrastructure:
    - MemoryCache      In-memory (single process)
    - RedisCache       Redis-backed (distributed)
    - MemcachedCache   Memcached-backed (distributed)
    - get_cache()      Global cache instance

CUSTOM IDEMPOTENCY STORE:
    from toolcase.runtime.batch import IdempotencyStore, CacheIdempotencyAdapter
    
    # Use cache adapter (recommended)
    store = CacheIdempotencyAdapter(get_cache())
    
    # Or implement custom store
    class MyStore(IdempotencyStore):
        async def get(self, key: str) -> str | None: ...
        async def set(self, key: str, value: str, ttl: float) -> None: ...
        async def delete(self, key: str) -> None: ...

USE CASES:
    - Parallel API calls to external services
    - Bulk data processing with rate limiting
    - Running the same analysis on multiple inputs
    - Concurrent validation of multiple items
    - Exactly-once payment/order processing
    - Distributed batch jobs with deduplication

RELATED TOPICS:
    toolcase help tool         Tool creation with batch_run
    toolcase help concurrency  Async primitives
    toolcase help pipeline     Pipeline composition
    toolcase help retry        Retry policies
"""
