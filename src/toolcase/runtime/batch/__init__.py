"""Intelligent batching for tool execution.

Provides configurable batch execution with concurrency control,
partial failure handling, and result aggregation.

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
"""

from .batch import (
    BatchConfig,
    BatchItem,
    BatchResult,
    batch_execute,
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

__all__ = [
    # Core batch
    "BatchConfig",
    "BatchItem",
    "BatchResult",
    "batch_execute",
    "batch_execute_sync",
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
]
