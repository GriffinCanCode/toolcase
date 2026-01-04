"""Built-in middleware plugins for common cross-cutting concerns.

These middleware are ready to use and demonstrate the middleware pattern.
They have no external dependencies beyond the standard library.
"""

from .breaker import CircuitBreakerMiddleware, CircuitState, State
from .logging import LoggingMiddleware
from .metrics import LogMetricsBackend, MetricsBackend, MetricsMiddleware
from .rate_limit import RateLimitMiddleware
from .retry import RETRYABLE_CODES, RetryMiddleware
from .timeout import TimeoutMiddleware
from .validation import (
    FieldRule,
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

__all__ = [
    # Resilience
    "CircuitBreakerMiddleware",
    "CircuitState",
    "State",
    "RetryMiddleware",
    "RETRYABLE_CODES",
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
