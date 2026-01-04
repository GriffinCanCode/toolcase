MIDDLEWARE = """
TOPIC: middleware
=================

Request/response middleware for cross-cutting concerns.

CONCEPT:
    Middleware wraps tool execution to add behavior like logging,
    retries, timeouts, rate limiting, and circuit breaking.

BUILT-IN MIDDLEWARE:
    LoggingMiddleware         Log tool calls and results
    MetricsMiddleware         Emit metrics (latency, success rate)
    RetryMiddleware           Retry failed calls with backoff
    TimeoutMiddleware         Enforce execution time limits
    RateLimitMiddleware       Throttle call frequency
    CircuitBreakerMiddleware  Fail fast on repeated failures

USAGE:
    from toolcase import (
        compose, LoggingMiddleware, TimeoutMiddleware,
        RetryMiddleware, Context
    )
    
    # Compose middleware chain
    chain = compose(
        LoggingMiddleware(),
        TimeoutMiddleware(5.0),
        RetryMiddleware(max_retries=3),
    )
    
    # Apply to tool execution
    result = await chain(tool, params, Context())

CUSTOM MIDDLEWARE:
    from toolcase import Middleware, Context, Next
    
    class TimingMiddleware(Middleware):
        async def __call__(
            self, tool, params, ctx: Context, next: Next
        ) -> str:
            start = time.time()
            result = await next(tool, params, ctx)
            print(f"Took {time.time() - start:.2f}s")
            return result

REGISTRY INTEGRATION:
    registry.use(LoggingMiddleware())
    registry.use(TimeoutMiddleware(30.0))

RELATED TOPICS:
    toolcase help retry      Retry policies and backoff
    toolcase help tracing    Distributed tracing
"""
