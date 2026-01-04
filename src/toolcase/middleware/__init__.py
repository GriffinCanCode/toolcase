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

Tracing Example:
    >>> from toolcase.observability import configure_tracing, TracingMiddleware
    >>> configure_tracing(service_name="my-agent", exporter="console")
    >>> registry.use(TracingMiddleware())
"""

from .middleware import Context, Middleware, Next, compose
from .plugins import (
    CircuitBreakerMiddleware,
    LoggingMiddleware,
    LogMetricsBackend,
    MetricsBackend,
    MetricsMiddleware,
    RateLimitMiddleware,
    RetryMiddleware,
    TimeoutMiddleware,
)

__all__ = [
    # Core
    "Middleware",
    "Next",
    "Context",
    "compose",
    # Plugins
    "CircuitBreakerMiddleware",
    "LoggingMiddleware",
    "LogMetricsBackend",
    "MetricsBackend",
    "MetricsMiddleware",
    "RateLimitMiddleware",
    "RetryMiddleware",
    "TimeoutMiddleware",
]
