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
"""

from .batch import (
    BatchConfig,
    BatchItem,
    BatchResult,
    batch_execute,
    batch_execute_sync,
)

__all__ = [
    "BatchConfig",
    "BatchItem",
    "BatchResult",
    "batch_execute",
    "batch_execute_sync",
]
