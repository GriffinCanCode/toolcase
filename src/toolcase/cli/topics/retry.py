RETRY = """
TOPIC: retry
============

Retry policies using stamina for clean, battle-tested retry behavior.

RETRY POLICY (stamina-based):
    from toolcase import RetryPolicy
    from toolcase.errors import ErrorCode
    
    policy = RetryPolicy(
        max_retries=3,              # Max retry attempts
        wait_initial=1.0,           # Initial delay (seconds)
        wait_max=30.0,              # Max delay cap (seconds)
        timeout=45.0,               # Total timeout for all retries
        wait_jitter=True,           # Add randomization (prevents thundering herd)
        retryable_codes=frozenset({
            ErrorCode.RATE_LIMITED,
            ErrorCode.TIMEOUT,
            ErrorCode.NETWORK_ERROR,
        }),
    )

TOOL-LEVEL RETRY:
    from toolcase import BaseTool, RetryPolicy
    
    class SearchTool(BaseTool[SearchParams]):
        # Automatic retry on retryable errors
        retry_policy = RetryPolicy(max_retries=3)
        
        async def _async_run(self, params):
            return await fetch_results(params.query)

MIDDLEWARE RETRY (exception-based):
    from toolcase import RetryMiddleware
    
    # Retries on exceptions with retryable error codes
    retry = RetryMiddleware(
        max_attempts=3,
        wait_initial=1.0,
        wait_max=30.0,
        timeout=45.0,
    )
    registry.use(retry)

COMPOSABLE RETRY STRATEGIES:
    from toolcase import RetryStrategy, resilient_tool
    
    # Fluent builder: retry → fallback → escalate
    strategy = (
        RetryStrategy()
        .with_retry(max_retries=3)
        .with_fallback([BackupAPI()])
        .with_escalation("approval_queue")
    )
    resilient = strategy.wrap(PrimaryAPI())
    
    # Or use helper function
    resilient = resilient_tool(
        PrimaryAPI(),
        retry=3,
        fallback=[BackupAPI()],
        escalate="approval_queue",
    )

BATCH-LEVEL RETRY:
    from toolcase import BatchRetryPolicy, IdempotentBatchConfig
    
    config = IdempotentBatchConfig(
        batch_id="order-batch-123",
        retry_policy=BatchRetryPolicy(
            max_retries=3,
            failure_threshold=0.3,  # Retry if >30% failed
        ),
    )

BACKOFF (for batch retry):
    ExponentialBackoff(base=1.0, max_delay=30.0, jitter=True)
    LinearBackoff(base=1.0, increment=1.0, max_delay=30.0)
    ConstantBackoff(delay_seconds=1.0)
    DecorrelatedJitter(base=1.0, max_delay=30.0)  # AWS-style

DEFAULT RETRYABLE ERROR CODES:
    - RATE_LIMITED
    - TIMEOUT
    - NETWORK_ERROR
    
    All retry logic uses stamina's exponential backoff with jitter internally.

RELATED TOPICS:
    toolcase help middleware     Middleware composition
    toolcase help concurrency    Async primitives
    toolcase help batch          Batch execution
"""
