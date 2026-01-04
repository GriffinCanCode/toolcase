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
from .rules import (
    # Core types
    Rule,
    Schema,
    ValidationResult,
    Violation,
    # Atomic rules
    Required,
    Optional,
    IsType,
    InRange,
    MinLength,
    MaxLength,
    Matches,
    OneOf,
    Url,
    Email,
    Predicate,
    # Combinators
    AllOf,
    AnyOf,
    XorOf,
    Not,
    # Conditional
    When,
    Unless,
    CrossFieldCondition,
    # Cross-field
    CrossField,
    FieldComparison,
    MutualExclusion,
    RequiredTogether,
    # Factory functions
    required,
    optional,
    is_type,
    is_str,
    is_int,
    is_float,
    is_bool,
    is_list,
    is_dict,
    in_range as rule_in_range,  # Alias to avoid conflict with legacy validator
    min_len,
    max_len,
    length,
    matches as rule_matches,  # Alias to avoid conflict with legacy validator
    one_of as rule_one_of,  # Alias to avoid conflict with legacy validator
    url,
    https,
    email,
    predicate,
    # Combinator factories
    all_of,
    any_of,
    xor_of,
    not_ as rule_not,
    # Conditional factories
    when,
    unless,
    when_eq,
    when_present,
    when_absent,
    # Cross-field factories
    cross,
    less_than,
    less_than_or_eq,
    equals,
    mutex,
    together,
    at_least_one,
    depends_on,
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
