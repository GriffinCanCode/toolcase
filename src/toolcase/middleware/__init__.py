"""Middleware system for tool execution hooks.

Provides composable pre/post execution hooks for cross-cutting concerns:
logging, metrics, rate limiting, auth, retries, timeouts.

Example:
    >>> from toolcase import get_registry
    >>> from toolcase.middleware import LoggingMiddleware, RetryMiddleware
    >>>
    >>> registry = get_registry()
    >>> registry.use(LoggingMiddleware())
    >>> registry.use(RetryMiddleware(max_attempts=3))
    >>>
    >>> result = await registry.execute("my_tool", {"query": "test"})
"""

from .middleware import (
    Context,
    Middleware,
    Next,
    compose,
)
from .builtins import (
    LoggingMiddleware,
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
    # Built-ins
    "LoggingMiddleware",
    "MetricsMiddleware",
    "RateLimitMiddleware",
    "RetryMiddleware",
    "TimeoutMiddleware",
]
