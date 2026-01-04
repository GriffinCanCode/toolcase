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

USE CASES:
    - Parallel API calls to external services
    - Bulk data processing with rate limiting
    - Running the same analysis on multiple inputs
    - Concurrent validation of multiple items

RELATED TOPICS:
    toolcase help tool         Tool creation with batch_run
    toolcase help concurrency  Async primitives
    toolcase help pipeline     Pipeline composition
"""
