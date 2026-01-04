DLQ = """
TOPIC: dlq
==========

Dead Letter Queue for handling poison messages in batch execution.

CONCEPT:
    Items that fail repeatedly (exceeding max_poison_threshold) are routed
    to a Dead Letter Queue rather than retried indefinitely. This prevents
    poison messages from blocking batch progress and enables manual review.

CONFIGURATION:
    from toolcase.runtime.batch import DLQConfig
    
    # Route to DLQ after 3 consecutive failures
    config = DLQConfig(max_poison_threshold=3)
    
    # Check if item should be routed to DLQ
    if config.should_dlq(consecutive_failures=4):
        await route_to_dlq(item, batch_id, tool_name, params, attempts)

DLQ ENTRY:
    from toolcase.runtime.batch import DLQEntry
    
    # Entry captures full context for debugging
    entry = DLQEntry.from_item(
        item=failed_item,
        batch_id="batch-123",
        tool_name="http_tool",
        params=original_params,
        attempts=3,
        error_context="API unavailable",
    )
    
    # Access entry data
    entry.batch_id      # Original batch identifier
    entry.item_index    # Index in original batch
    entry.tool_name     # Tool that produced failure
    entry.params_dict   # Deserialized parameters
    entry.error         # ErrorTrace with full context
    entry.attempts      # Total attempts before DLQ
    entry.timestamp     # Unix timestamp when queued
    entry.key           # "batch-123:0" for lookup

ROUTING TO DLQ:
    from toolcase.runtime.batch import route_to_dlq
    
    # Simple routing
    entry = await route_to_dlq(item, "batch-123", "http", params, 3)
    
    # With async notification callback
    async def notify_slack(entry: DLQEntry) -> None:
        await post_message(f"DLQ: {entry.tool_name} failed {entry.attempts}x")
    
    entry = await route_to_dlq(
        item, "batch-123", "http", params, 3,
        callback=notify_slack,
        error_context="Connection refused",
    )

DLQ STORE:
    from toolcase.runtime.batch import (
        get_dlq_store, set_dlq_store, MemoryDLQStore
    )
    
    # Get global store (MemoryDLQStore by default)
    store = get_dlq_store()
    
    # Query entries
    entries = store.list(tool_name="http")           # By tool
    entries = store.list(batch_id="batch-123")       # By batch
    entries = store.list(limit=10)                   # Latest 10
    
    # Get specific entry
    entry = store.get("batch-123:0")
    
    # Remove and return (for reprocessing)
    entry = store.pop("batch-123:0")
    
    # Count/clear
    total = store.count()
    removed = store.clear(batch_id="batch-123")

REPROCESSING:
    from toolcase.runtime.batch import reprocess_entry
    
    # Extract reprocessing info from DLQ entry
    tool_name, idx, params = reprocess_entry(entry)
    
    # Retry with original parameters
    result = await registry.get(tool_name).arun(params)

CUSTOM STORE BACKEND:
    from toolcase.runtime.batch import DLQStore, set_dlq_store
    
    # Implement DLQStore protocol for Redis/DB/queue
    class RedisDLQStore:
        def put(self, entry: DLQEntry) -> None: ...
        def get(self, key: str) -> DLQEntry | None: ...
        def pop(self, key: str) -> DLQEntry | None: ...
        def list(self, *, batch_id=None, tool_name=None, limit=100): ...
        def count(self) -> int: ...
        def clear(self, *, batch_id=None) -> int: ...
    
    set_dlq_store(RedisDLQStore(redis_client))

TESTING:
    from toolcase.runtime.batch import reset_dlq_store
    
    # Reset between tests
    reset_dlq_store()

RELATED TOPICS:
    toolcase help batch       Batch execution
    toolcase help retry       Retry policies
    toolcase help resilience  Circuit breaker patterns
"""
