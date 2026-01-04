MIDDLEWARE = """
TOPIC: middleware
=================

Request/response middleware for cross-cutting concerns.

CONCEPT:
    Middleware wraps tool execution to add behavior like logging,
    retries, timeouts, rate limiting, validation, and circuit breaking.

BUILT-IN MIDDLEWARE:
    ValidationMiddleware      Centralized param validation + custom rules
    LoggingMiddleware         Log tool calls and results (with params option)
    MetricsMiddleware         Emit metrics (latency, success rate)
    RetryMiddleware           Retry failed calls with backoff
    TimeoutMiddleware         Enforce execution time limits
    RateLimitMiddleware       Throttle call frequency
    CircuitBreakerMiddleware  Fail fast on repeated failures
    CoalesceMiddleware        Deduplicate concurrent identical requests
    TracingMiddleware         Create spans for distributed tracing
    CorrelationMiddleware     Add correlation IDs to requests

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

REGISTRY INTEGRATION (Recommended):
    registry = get_registry()
    validation = registry.use_validation()  # First (returns ValidationMiddleware)
    registry.use(LoggingMiddleware())       # Second
    registry.use(TimeoutMiddleware(30.0))   # Third (innermost)
    
    # Execute through middleware chain
    result = await registry.execute("search", {"query": "python"})

VALIDATION MIDDLEWARE (Legacy API):
    from toolcase.runtime.middleware.plugins import (
        min_length, max_length, in_range, matches, one_of, not_empty, https_only
    )
    
    # Enable centralized validation (runs first in chain)
    validation = registry.use_validation()
    
    # Add custom field rules
    validation.add_rule("search", "query", min_length(3), "query too short")
    validation.add_rule("http_request", "url", https_only, "must use HTTPS")
    validation.add_rule("search", "limit", in_range(1, 100), "limit out of range")
    
    # Cross-field constraints
    validation.add_constraint("report", lambda p: p.start <= p.end or "invalid range")

RULE DSL (Recommended):
    See 'toolcase help validation' for the composable Rule DSL with
    algebraic combinators (&, |, ~), conditional validation, and schemas.

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

COALESCE MIDDLEWARE (Request Deduplication):
    from toolcase.runtime.middleware import CoalesceMiddleware
    
    # Deduplicate concurrent identical requests (singleflight pattern)
    registry.use(CoalesceMiddleware())
    
    # 10 concurrent identical requests → 1 execution, 10 results
    results = await asyncio.gather(*[
        registry.execute("expensive_tool", {"q": "same"})
        for _ in range(10)
    ])
    
    # Monitor coalescing effectiveness
    mw = CoalesceMiddleware()
    registry.use(mw)
    print(mw.stats)  # {total_requests, coalesced_requests, coalesce_ratio, ...}

STREAMING MIDDLEWARE:
    from toolcase.runtime.middleware import StreamMiddleware, compose_streaming
    
    # Streaming middleware for chunk-level hooks
    registry.use(StreamLoggingMiddleware())
    async for chunk in registry.stream_execute("generate", {"topic": "AI"}):
        print(chunk, end="")

RELATED TOPICS:
    toolcase help validation Composable validation DSL
    toolcase help retry      Retry policies and backoff
    toolcase help tracing    Distributed tracing
    toolcase help registry   Registry middleware integration
"""
