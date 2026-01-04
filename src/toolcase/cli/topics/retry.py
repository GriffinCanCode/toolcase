RETRY = """
TOPIC: retry
============

Retry policies and backoff strategies for resilient execution.

RETRY POLICY:
    from toolcase import RetryPolicy, ExponentialBackoff
    
    policy = RetryPolicy(
        max_retries=3,
        backoff=ExponentialBackoff(base=1.0, max_delay=30.0),
        retryable=lambda e: isinstance(e, (TimeoutError, ConnectionError)),
    )

BACKOFF STRATEGIES:
    ConstantBackoff(delay)              Fixed delay
    LinearBackoff(initial, increment)   Linearly increasing
    ExponentialBackoff(base, max)       Exponential growth
    DecorrelatedJitter(base, max)       Randomized for thundering herd

USAGE WITH MIDDLEWARE:
    from toolcase import RetryMiddleware, ExponentialBackoff
    
    retry = RetryMiddleware(
        max_retries=3,
        backoff=ExponentialBackoff(base=1.0),
    )
    registry.use(retry)

USAGE WITH CONCURRENCY:
    from toolcase import Concurrency
    
    result = await Concurrency.retry(
        lambda: fetch_data(),
        max_attempts=3,
        delay=1.0,
        backoff=2.0,  # Multiplier
    )

DEFAULT RETRYABLE ERRORS:
    - TimeoutError
    - ConnectionError
    - asyncio.TimeoutError
    
    Customize with retryable parameter.

RELATED TOPICS:
    toolcase help middleware     Middleware composition
    toolcase help concurrency    Async primitives
"""
