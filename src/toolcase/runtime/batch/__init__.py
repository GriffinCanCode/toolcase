"""Intelligent batching for tool execution.

Provides configurable batch execution with concurrency control,
partial failure handling, result aggregation, and streaming progress.

Usage:
    from toolcase.runtime.batch import BatchConfig, batch_execute
    
    # Configure batching behavior
    config = BatchConfig(concurrency=10, fail_fast=False)
    
    # Execute batch
    results = await batch_execute(tool, params_list, config)
    
    # Check results
    for r in results:
        if r.is_ok:
            print(f"[{r.index}] Success: {r.value}")
        else:
            print(f"[{r.index}] Failed: {r.error}")

Streaming Progress:
    from toolcase.runtime.batch import batch_execute_stream, BatchEventKind
    
    # Stream progress events as items complete
    async for event in batch_execute_stream(tool, params_list, config):
        match event.kind:
            case BatchEventKind.START:
                print(f"Starting {event.total} items...")
            case BatchEventKind.ITEM:
                print(f"[{event.completed}/{event.total}] {event.progress:.0%}")
            case BatchEventKind.COMPLETE:
                print(f"Done: {event.batch_result.success_rate:.0%} success")

Idempotent Batch (Exactly-Once Semantics):
    from toolcase.runtime.batch import (
        IdempotentBatchConfig, BatchRetryPolicy, batch_execute_idempotent
    )
    
    # Configure with batch-level retry and idempotency
    config = IdempotentBatchConfig(
        concurrency=10,
        batch_id="order-batch-123",
        retry_policy=BatchRetryPolicy(max_retries=3, failure_threshold=0.3),
    )
    
    # Execute with exactly-once guarantees
    results = await batch_execute_idempotent(tool, params_list, config)
    print(f"Cache hits: {results.cache_hit_rate:.0%}")

Dead Letter Queue (Poison Message Handling):
    from toolcase.runtime.batch import DLQConfig, route_to_dlq, get_dlq_store
    
    # Route failed items to DLQ after 3 consecutive failures
    config = DLQConfig(max_poison_threshold=3)
    
    # Manual routing with callback
    async def on_dlq(entry):
        print(f"DLQ: {entry.tool_name} failed {entry.attempts}x")
    
    entry = await route_to_dlq(item, "batch-123", "http", params, 3, callback=on_dlq)
    
    # Query DLQ
    store = get_dlq_store()
    failed = store.list(tool_name="http")
"""

from .batch import (
    BatchConfig,
    BatchEventKind,
    BatchItem,
    BatchItemEvent,
    BatchResult,
    batch_execute,
    batch_execute_stream,
    batch_execute_sync,
)
from .idempotent import (
    BatchRetryPolicy,
    BatchRetryStrategy,
    CacheIdempotencyAdapter,
    IdempotencyStore,
    IdempotentBatchConfig,
    IdempotentBatchResult,
    NO_BATCH_RETRY,
    batch_execute_idempotent,
    batch_execute_idempotent_sync,
)
from .dlq import (
    DLQCallback,
    DLQConfig,
    DLQEntry,
    DLQStore,
    MemoryDLQStore,
    NO_DLQ,
    get_dlq_store,
    reset_dlq_store,
    reprocess_entry,
    route_to_dlq,
    set_dlq_store,
)

__all__ = [
    # Core batch
    "BatchConfig",
    "BatchItem",
    "BatchResult",
    "batch_execute",
    "batch_execute_sync",
    # Streaming batch
    "BatchEventKind",
    "BatchItemEvent",
    "batch_execute_stream",
    # Idempotent batch
    "IdempotentBatchConfig",
    "IdempotentBatchResult",
    "BatchRetryPolicy",
    "BatchRetryStrategy",
    "NO_BATCH_RETRY",
    "batch_execute_idempotent",
    "batch_execute_idempotent_sync",
    # Idempotency adapter (uses existing cache infrastructure)
    "IdempotencyStore",
    "CacheIdempotencyAdapter",
    # Dead Letter Queue
    "DLQConfig",
    "DLQEntry",
    "DLQStore",
    "DLQCallback",
    "MemoryDLQStore",
    "NO_DLQ",
    "get_dlq_store",
    "set_dlq_store",
    "reset_dlq_store",
    "route_to_dlq",
    "reprocess_entry",
]
