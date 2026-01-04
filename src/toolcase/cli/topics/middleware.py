MIDDLEWARE = """
TOPIC: middleware
=================

Request/response middleware for cross-cutting concerns.

CONCEPT:
    Middleware wraps tool execution to add behavior like logging,
    retries, timeouts, rate limiting, validation, and circuit breaking.

BUILT-IN MIDDLEWARE:
    ValidationMiddleware      Centralized param validation + custom rules
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

VALIDATION MIDDLEWARE:
    # Enable centralized validation (runs first in chain)
    validation = registry.use_validation()
    
    # Add custom field rules
    validation.add_rule("search", "query", min_length(3), "too short")
    validation.add_rule("http_request", "url", https_only, "must use HTTPS")
    
    # Cross-field constraints
    validation.add_constraint("report", lambda p: p.start <= p.end or "invalid range")
    
    # Preset validators: min_length, max_length, in_range, matches, one_of, not_empty, https_only

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
    validation = registry.use_validation()  # First (validation)
    registry.use(LoggingMiddleware())       # Second
    registry.use(TimeoutMiddleware(30.0))   # Third (innermost)

RELATED TOPICS:
    toolcase help retry      Retry policies and backoff
    toolcase help tracing    Distributed tracing
"""
