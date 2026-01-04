"""Middleware system for tool execution hooks.

Provides composable pre/post execution hooks for cross-cutting concerns:
logging, metrics, rate limiting, circuit breaking, retries, timeouts, tracing.

Example:
    >>> from toolcase import get_registry
    >>> from toolcase.middleware import LoggingMiddleware, RetryMiddleware
    >>>
    >>> registry = get_registry()
    >>> registry.use(LoggingMiddleware())
    >>> registry.use(RetryMiddleware(max_attempts=3))
    >>>
    >>> result = await registry.execute("my_tool", {"query": "test"})

Resilience Stack Example:
    >>> from toolcase.middleware import (
    ...     CircuitBreakerMiddleware, RetryMiddleware, TimeoutMiddleware
    ... )
    >>> # Order matters: timeout → retry → circuit breaker
    >>> registry.use(CircuitBreakerMiddleware(failure_threshold=5))
    >>> registry.use(RetryMiddleware(max_attempts=3))
    >>> registry.use(TimeoutMiddleware(timeout_seconds=30))

Streaming Middleware Example:
    >>> from toolcase.middleware import StreamLoggingMiddleware, compose_streaming
    >>> # Streaming middleware receives chunk-level hooks
    >>> registry.use(StreamLoggingMiddleware())
    >>> async for chunk in registry.stream_execute("generate", {"topic": "AI"}):
    ...     print(chunk, end="")

Tracing Example:
    >>> from toolcase.observability import configure_tracing, TracingMiddleware
    >>> configure_tracing(service_name="my-agent", exporter="console")
    >>> registry.use(TracingMiddleware())
"""

from .middleware import Context, Middleware, Next, compose
from .plugins import (
    CircuitBreakerMiddleware,
    FieldRule,
    LoggingMiddleware,
    LogMetricsBackend,
    MetricsBackend,
    MetricsMiddleware,
    RateLimitMiddleware,
    RetryMiddleware,
    TimeoutMiddleware,
    ValidationMiddleware,
    Validator,
    https_only,
    in_range,
    matches,
    max_length,
    min_length,
    not_empty,
    one_of,
)
from .streaming import (
    StreamMiddleware,
    StreamingAdapter,
    StreamingChain,
    StreamLoggingMiddleware,
    StreamMetricsMiddleware,
    compose_streaming,
)

__all__ = [
    # Core
    "Middleware",
    "Next",
    "Context",
    "compose",
    # Streaming
    "StreamMiddleware",
    "StreamingAdapter",
    "StreamingChain",
    "compose_streaming",
    "StreamLoggingMiddleware",
    "StreamMetricsMiddleware",
    # Resilience
    "CircuitBreakerMiddleware",
    "RetryMiddleware",
    "TimeoutMiddleware",
    "RateLimitMiddleware",
    # Observability
    "LoggingMiddleware",
    "LogMetricsBackend",
    "MetricsBackend",
    "MetricsMiddleware",
    # Validation
    "ValidationMiddleware",
    "FieldRule",
    "Validator",
    # Preset validators
    "min_length",
    "max_length",
    "in_range",
    "matches",
    "one_of",
    "not_empty",
    "https_only",
]
