"""Built-in middleware plugins for common cross-cutting concerns.

These middleware are ready to use and demonstrate the middleware pattern.
They have no external dependencies beyond the standard library.
"""

from .breaker import CircuitBreakerMiddleware, CircuitState, MemoryStateStore, State, StateStore
from .logging import LoggingMiddleware
from .metrics import LogMetricsBackend, MetricsBackend, MetricsMiddleware
from .rate_limit import RateLimitMiddleware
from .retry import RETRYABLE_CODES, RetryMiddleware
from .rules import (
    # Combinators
    AllOf,
    AnyOf,
    # Cross-field
    CrossField,
    CrossFieldCondition,
    Email,
    FieldComparison,
    InRange,
    IsType,
    Matches,
    MaxLength,
    MinLength,
    MutualExclusion,
    Not,
    OneOf,
    Optional,
    Predicate,
    # Atomic rules
    Required,
    RequiredTogether,
    # Core types
    Rule,
    Schema,
    Unless,
    Url,
    ValidationResult,
    Violation,
    # Conditional
    When,
    XorOf,
    # Combinator factories
    all_of,
    any_of,
    at_least_one,
    # Cross-field factories
    cross,
    depends_on,
    email,
    equals,
    https,
    is_bool,
    is_dict,
    is_float,
    is_int,
    is_list,
    is_str,
    is_type,
    length,
    less_than,
    less_than_or_eq,
    max_len,
    min_len,
    mutex,
    optional,
    predicate,
    # Factory functions
    required,
    together,
    unless,
    url,
    # Conditional factories
    when,
    when_absent,
    when_eq,
    when_present,
    xor_of,
)
from .rules import (
    in_range as rule_in_range,  # Alias to avoid conflict with legacy validator
)
from .rules import (
    matches as rule_matches,  # Alias to avoid conflict with legacy validator
)
from .rules import (
    not_ as rule_not,
)
from .rules import (
    one_of as rule_one_of,  # Alias to avoid conflict with legacy validator
)
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
    "StateStore",
    "MemoryStateStore",
    "RedisStateStore",  # Lazy import
    "RetryMiddleware",
    "RETRYABLE_CODES",
    "TimeoutMiddleware",
    "RateLimitMiddleware",
    # Observability
    "LoggingMiddleware",
    "LogMetricsBackend",
    "MetricsBackend",
    "MetricsMiddleware",
    # Validation Middleware
    "ValidationMiddleware",
    "FieldRule",
    "Validator",
    # Preset validators (legacy)
    "min_length",
    "max_length",
    "in_range",
    "matches",
    "one_of",
    "not_empty",
    "https_only",
    # ─────────────────────────────────────────────────────────────
    # Rule DSL
    # ─────────────────────────────────────────────────────────────
    # Core types
    "Rule",
    "Schema",
    "ValidationResult",
    "Violation",
    # Atomic rules (classes)
    "Required",
    "Optional",
    "IsType",
    "InRange",
    "MinLength",
    "MaxLength",
    "Matches",
    "OneOf",
    "Url",
    "Email",
    "Predicate",
    # Combinators (classes)
    "AllOf",
    "AnyOf",
    "XorOf",
    "Not",
    # Conditional (classes)
    "When",
    "Unless",
    "CrossFieldCondition",
    # Cross-field (classes)
    "CrossField",
    "FieldComparison",
    "MutualExclusion",
    "RequiredTogether",
    # Factory functions (lowercase - ergonomic API)
    "required",
    "optional",
    "is_type",
    "is_str",
    "is_int",
    "is_float",
    "is_bool",
    "is_list",
    "is_dict",
    "rule_in_range",
    "min_len",
    "max_len",
    "length",
    "rule_matches",
    "rule_one_of",
    "url",
    "https",
    "email",
    "predicate",
    # Combinator factories
    "all_of",
    "any_of",
    "xor_of",
    "rule_not",
    # Conditional factories
    "when",
    "unless",
    "when_eq",
    "when_present",
    "when_absent",
    # Cross-field factories
    "cross",
    "less_than",
    "less_than_or_eq",
    "equals",
    "mutex",
    "together",
    "at_least_one",
    "depends_on",
]


def __getattr__(name: str) -> object:
    """Lazy import Redis state store to avoid import-time dependency."""
    if name == "RedisStateStore":
        from .store import RedisStateStore
        return RedisStateStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
