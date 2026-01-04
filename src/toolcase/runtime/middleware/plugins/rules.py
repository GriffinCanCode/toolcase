"""Validation Rule DSL for composable, type-safe parameter validation.

Provides a fluent API for building complex validation rules:
- Atomic rules: type checks, range constraints, patterns
- Combinators: all_of, any_of, xor_of (logical composition)
- Conditional: when/unless guards for context-dependent validation
- Cross-field: constraints spanning multiple fields
- Schema builder: declarative tool validation schemas

Design Principles:
- Algebraic composition (rules are monoids under AND/OR)
- Parser combinator style (small primitives → complex validators)
- Railway-oriented error accumulation
- Zero external dependencies beyond Pydantic

Example:
    >>> schema = (
    ...     Schema("http_request")
    ...     .field("url", required() & url() & https())
    ...     .field("timeout", optional() & in_range(1, 60))
    ...     .field("method", one_of("GET", "POST", "PUT", "DELETE"))
    ...     .cross(when(eq("method", "POST"), required("body")))
    ... )
    >>> middleware.add_schema(schema)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar, overload

from pydantic import BaseModel

if TYPE_CHECKING:
    from typing import Any

# ═════════════════════════════════════════════════════════════════════════════
# Core Abstractions
# ═════════════════════════════════════════════════════════════════════════════

T = TypeVar("T")

# Validation result: True (pass) | str (fail with message)
CheckResult = bool | str

# Value extractor: params → value (for cross-field access)
Extractor = Callable[[BaseModel], object]


@dataclass(frozen=True, slots=True)
class Violation:
    """Single validation violation with field path and message."""
    field: str
    message: str
    code: str = "INVALID"
    
    def __str__(self) -> str:
        return f"'{self.field}' {self.message}"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Accumulated validation result. Monoid under concatenation."""
    violations: tuple[Violation, ...] = ()
    
    @property
    def is_valid(self) -> bool: return not self.violations
    
    def __bool__(self) -> bool: return self.is_valid
    
    def __add__(self, other: ValidationResult) -> ValidationResult:
        return ValidationResult(self.violations + other.violations)
    
    def __iter__(self):
        return iter(self.violations)
    
    def format(self, sep: str = "; ") -> str:
        return sep.join(str(v) for v in self.violations) or "valid"


# Singletons
_VALID = ValidationResult()


def valid() -> ValidationResult:
    """Return singleton valid result."""
    return _VALID


def invalid(field: str, message: str, code: str = "INVALID") -> ValidationResult:
    """Create single-violation result."""
    return ValidationResult((Violation(field, message, code),))


# ═════════════════════════════════════════════════════════════════════════════
# Rule Protocol
# ═════════════════════════════════════════════════════════════════════════════


class Rule(ABC):
    """Abstract validation rule. Composable via & (AND) and | (OR)."""
    
    __slots__ = ()
    
    @abstractmethod
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        """Validate value, returning violations or empty result."""
        ...
    
    def __and__(self, other: Rule) -> Rule:
        """Compose with AND semantics (both must pass)."""
        return AllOf((self, other))
    
    def __or__(self, other: Rule) -> Rule:
        """Compose with OR semantics (either passes)."""
        return AnyOf((self, other))
    
    def __invert__(self) -> Rule:
        """Negate rule (passes if original fails)."""
        return Not(self)
    
    def when(self, condition: CrossFieldCondition) -> Rule:
        """Apply rule only when condition is met."""
        return When(condition, self)
    
    def unless(self, condition: CrossFieldCondition) -> Rule:
        """Apply rule only when condition is NOT met."""
        return Unless(condition, self)
    
    def with_message(self, message: str) -> Rule:
        """Override error message."""
        return MessageOverride(self, message)
    
    def with_code(self, code: str) -> Rule:
        """Override error code."""
        return CodeOverride(self, code)


# ═════════════════════════════════════════════════════════════════════════════
# Atomic Rules (Primitives)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class Predicate(Rule):
    """Rule from predicate function. Most flexible primitive."""
    check_fn: Callable[[object], CheckResult]
    message: str
    code: str = "INVALID"
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        result = self.check_fn(value)
        if result is True:
            return _VALID
        return invalid(field, result if isinstance(result, str) else self.message, self.code)


@dataclass(frozen=True, slots=True)
class Required(Rule):
    """Value must be present and non-empty."""
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        if value is None:
            return invalid(field, "is required", "REQUIRED")
        if isinstance(value, (str, list, dict)) and not value:
            return invalid(field, "cannot be empty", "EMPTY")
        return _VALID


@dataclass(frozen=True, slots=True)
class Optional(Rule):
    """Value may be None/missing. Always passes (for documentation)."""
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        return _VALID


@dataclass(frozen=True, slots=True)
class IsType(Rule):
    """Value must be instance of given type(s)."""
    types: tuple[type, ...]
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        if value is None or isinstance(value, self.types):
            return _VALID
        names = " | ".join(t.__name__ for t in self.types)
        return invalid(field, f"must be {names}, got {type(value).__name__}", "TYPE_ERROR")


@dataclass(frozen=True, slots=True)
class InRange(Rule):
    """Numeric value in [low, high] range."""
    low: float
    high: float
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        if value is None:
            return _VALID
        if not isinstance(value, (int, float)):
            return invalid(field, "must be numeric", "TYPE_ERROR")
        if not self.low <= value <= self.high:
            return invalid(field, f"must be between {self.low} and {self.high}", "OUT_OF_RANGE")
        return _VALID


@dataclass(frozen=True, slots=True)
class MinLength(Rule):
    """String/collection minimum length."""
    length: int
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        if value is None:
            return _VALID
        if not hasattr(value, "__len__"):
            return invalid(field, "must have length", "TYPE_ERROR")
        if len(value) < self.length:  # type: ignore[arg-type]
            return invalid(field, f"must have at least {self.length} items/characters", "TOO_SHORT")
        return _VALID


@dataclass(frozen=True, slots=True)
class MaxLength(Rule):
    """String/collection maximum length."""
    length: int
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        if value is None:
            return _VALID
        if not hasattr(value, "__len__"):
            return invalid(field, "must have length", "TYPE_ERROR")
        if len(value) > self.length:  # type: ignore[arg-type]
            return invalid(field, f"must have at most {self.length} items/characters", "TOO_LONG")
        return _VALID


@dataclass(frozen=True, slots=True)
class Matches(Rule):
    """String matches regex pattern."""
    pattern: re.Pattern[str]
    description: str | None = None
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        if value is None:
            return _VALID
        if not isinstance(value, str):
            return invalid(field, "must be string", "TYPE_ERROR")
        if not self.pattern.match(value):
            desc = self.description or self.pattern.pattern
            return invalid(field, f"must match pattern: {desc}", "PATTERN_MISMATCH")
        return _VALID


@dataclass(frozen=True, slots=True)
class OneOf(Rule):
    """Value must be one of allowed options."""
    allowed: frozenset[object]
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        if value is None:
            return _VALID
        if value not in self.allowed:
            opts = ", ".join(repr(v) for v in sorted(self.allowed, key=str))
            return invalid(field, f"must be one of: {opts}", "NOT_ALLOWED")
        return _VALID


@dataclass(frozen=True, slots=True)
class Url(Rule):
    """Value must be valid URL with optional scheme restriction."""
    schemes: frozenset[str] | None = None
    
    _URL_PATTERN: re.Pattern[str] = field(default=re.compile(r"^https?://[^\s<>\"]+$"), repr=False)
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        if value is None:
            return _VALID
        if not isinstance(value, str):
            return invalid(field, "must be string", "TYPE_ERROR")
        if not self._URL_PATTERN.match(value):
            return invalid(field, "must be valid URL", "INVALID_URL")
        if self.schemes:
            scheme = value.split("://")[0].lower()
            if scheme not in self.schemes:
                return invalid(field, f"must use scheme: {', '.join(self.schemes)}", "INVALID_SCHEME")
        return _VALID


@dataclass(frozen=True, slots=True)
class Email(Rule):
    """Value must be valid email format."""
    
    _EMAIL_PATTERN: re.Pattern[str] = field(default=re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"), repr=False)
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        if value is None:
            return _VALID
        if not isinstance(value, str):
            return invalid(field, "must be string", "TYPE_ERROR")
        if not self._EMAIL_PATTERN.match(value):
            return invalid(field, "must be valid email", "INVALID_EMAIL")
        return _VALID


# ═════════════════════════════════════════════════════════════════════════════
# Composite Rules (Combinators)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class AllOf(Rule):
    """All rules must pass (AND combinator). Accumulates violations."""
    rules: tuple[Rule, ...]
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        result = _VALID
        for rule in self.rules:
            result = result + rule.check(value, field, params)
        return result


@dataclass(frozen=True, slots=True)
class AnyOf(Rule):
    """At least one rule must pass (OR combinator). Short-circuits on success."""
    rules: tuple[Rule, ...]
    message: str = "none of the alternatives matched"
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        for rule in self.rules:
            if (result := rule.check(value, field, params)).is_valid:
                return _VALID
        return invalid(field, self.message, "NO_MATCH")


@dataclass(frozen=True, slots=True)
class XorOf(Rule):
    """Exactly one rule must pass (XOR combinator)."""
    rules: tuple[Rule, ...]
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        passing = sum(1 for r in self.rules if r.check(value, field, params).is_valid)
        if passing == 1:
            return _VALID
        if passing == 0:
            return invalid(field, "exactly one alternative must match (none did)", "XOR_NONE")
        return invalid(field, f"exactly one alternative must match ({passing} did)", "XOR_MULTIPLE")


@dataclass(frozen=True, slots=True)
class Not(Rule):
    """Negates rule result."""
    rule: Rule
    message: str = "must not satisfy condition"
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        if self.rule.check(value, field, params).is_valid:
            return invalid(field, self.message, "NEGATION_FAILED")
        return _VALID


# ═════════════════════════════════════════════════════════════════════════════
# Conditional Rules
# ═════════════════════════════════════════════════════════════════════════════


# Cross-field condition: params → bool
CrossFieldCondition = Callable[[BaseModel], bool]


@dataclass(frozen=True, slots=True)
class When(Rule):
    """Apply rule only when condition is True."""
    condition: CrossFieldCondition
    rule: Rule
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        return self.rule.check(value, field, params) if self.condition(params) else _VALID


@dataclass(frozen=True, slots=True)
class Unless(Rule):
    """Apply rule only when condition is False."""
    condition: CrossFieldCondition
    rule: Rule
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        return _VALID if self.condition(params) else self.rule.check(value, field, params)


# ═════════════════════════════════════════════════════════════════════════════
# Message/Code Overrides
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class MessageOverride(Rule):
    """Wrap rule with custom error message."""
    rule: Rule
    message: str
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        result = self.rule.check(value, field, params)
        if result.is_valid:
            return result
        return ValidationResult(tuple(Violation(v.field, self.message, v.code) for v in result.violations))


@dataclass(frozen=True, slots=True)
class CodeOverride(Rule):
    """Wrap rule with custom error code."""
    rule: Rule
    code: str
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        result = self.rule.check(value, field, params)
        if result.is_valid:
            return result
        return ValidationResult(tuple(Violation(v.field, v.message, self.code) for v in result.violations))


# ═════════════════════════════════════════════════════════════════════════════
# Cross-Field Rules
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class CrossField(Rule):
    """Validate relationship between multiple fields."""
    check_fn: Callable[[BaseModel], CheckResult]
    fields: tuple[str, ...]
    message: str
    code: str = "CROSS_FIELD"
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        result = self.check_fn(params)
        if result is True:
            return _VALID
        msg = result if isinstance(result, str) else self.message
        # Report on first field by convention
        return invalid(self.fields[0] if self.fields else field, msg, self.code)


@dataclass(frozen=True, slots=True)
class FieldComparison(Rule):
    """Compare two fields with given operator."""
    other_field: str
    op: Callable[[object, object], bool]
    op_name: str
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        other = getattr(params, self.other_field, None)
        if value is None or other is None:
            return _VALID
        if not self.op(value, other):
            return invalid(field, f"must be {self.op_name} '{self.other_field}'", "COMPARISON_FAILED")
        return _VALID


@dataclass(frozen=True, slots=True)
class MutualExclusion(Rule):
    """Only one of the specified fields may be present."""
    fields: tuple[str, ...]
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        present = [f for f in self.fields if getattr(params, f, None) is not None]
        if len(present) > 1:
            return invalid(present[0], f"mutually exclusive with {', '.join(present[1:])}", "MUTUAL_EXCLUSION")
        return _VALID


@dataclass(frozen=True, slots=True)
class RequiredTogether(Rule):
    """All specified fields must be present together or all absent."""
    fields: tuple[str, ...]
    
    def check(self, value: object, field: str, params: BaseModel) -> ValidationResult:
        present = [f for f in self.fields if getattr(params, f, None) is not None]
        absent = [f for f in self.fields if f not in present]
        if present and absent:
            return invalid(absent[0], f"required when {', '.join(present)} is present", "REQUIRED_TOGETHER")
        return _VALID


# ═════════════════════════════════════════════════════════════════════════════
# Schema Builder (Fluent API)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class FieldSpec:
    """Specification for a single field's validation rules."""
    name: str
    rules: list[Rule] = field(default_factory=list)
    
    def add(self, rule: Rule) -> FieldSpec:
        self.rules.append(rule)
        return self


@dataclass
class Schema:
    """Fluent builder for tool validation schema.
    
    Example:
        >>> schema = (
        ...     Schema("my_tool")
        ...     .field("url", required() & url() & https())
        ...     .field("count", optional() & is_type(int) & in_range(1, 100))
        ...     .cross(less_than("start", "end"))
        ...     .cross(mutex("sync", "async_mode"))
        ... )
    """
    tool_name: str
    _fields: dict[str, FieldSpec] = field(default_factory=dict)
    _cross: list[Rule] = field(default_factory=list)
    
    def field(self, name: str, *rules: Rule) -> Schema:
        """Add field with validation rules."""
        spec = self._fields.setdefault(name, FieldSpec(name))
        for rule in rules:
            spec.add(rule)
        return self
    
    def cross(self, rule: Rule) -> Schema:
        """Add cross-field validation rule."""
        self._cross.append(rule)
        return self
    
    def validate(self, params: BaseModel) -> ValidationResult:
        """Validate params against schema. Returns accumulated violations."""
        result = _VALID
        
        # Field rules
        for name, spec in self._fields.items():
            value = getattr(params, name, None)
            for rule in spec.rules:
                result = result + rule.check(value, name, params)
        
        # Cross-field rules
        for rule in self._cross:
            result = result + rule.check(None, "", params)
        
        return result
    
    @property
    def fields(self) -> dict[str, FieldSpec]:
        """Access field specifications."""
        return self._fields
    
    @property
    def cross_rules(self) -> list[Rule]:
        """Access cross-field rules."""
        return self._cross


# ═════════════════════════════════════════════════════════════════════════════
# Factory Functions (Ergonomic API)
# ═════════════════════════════════════════════════════════════════════════════


def required() -> Rule:
    """Value must be present and non-empty."""
    return Required()


def optional() -> Rule:
    """Value may be None/missing (documentation rule)."""
    return Optional()


def is_type(*types: type) -> Rule:
    """Value must be instance of given type(s)."""
    return IsType(types)


def is_str() -> Rule:
    """Value must be string."""
    return IsType((str,))


def is_int() -> Rule:
    """Value must be integer."""
    return IsType((int,))


def is_float() -> Rule:
    """Value must be float."""
    return IsType((float, int))


def is_bool() -> Rule:
    """Value must be boolean."""
    return IsType((bool,))


def is_list() -> Rule:
    """Value must be list."""
    return IsType((list,))


def is_dict() -> Rule:
    """Value must be dict."""
    return IsType((dict,))


def in_range(low: float, high: float) -> Rule:
    """Numeric value in [low, high] range."""
    return InRange(low, high)


def min_len(n: int) -> Rule:
    """Minimum string/collection length."""
    return MinLength(n)


def max_len(n: int) -> Rule:
    """Maximum string/collection length."""
    return MaxLength(n)


def length(n: int) -> Rule:
    """Exact string/collection length."""
    return MinLength(n) & MaxLength(n)


def matches(pattern: str, description: str | None = None) -> Rule:
    """String matches regex pattern."""
    return Matches(re.compile(pattern), description)


def one_of(*values: object) -> Rule:
    """Value must be one of allowed options."""
    return OneOf(frozenset(values))


def url(schemes: Iterable[str] | None = None) -> Rule:
    """Value must be valid URL."""
    return Url(frozenset(schemes) if schemes else None)


def https() -> Rule:
    """Value must be HTTPS URL."""
    return Url(frozenset(("https",)))


def email() -> Rule:
    """Value must be valid email."""
    return Email()


def predicate(fn: Callable[[object], CheckResult], message: str, code: str = "INVALID") -> Rule:
    """Rule from predicate function."""
    return Predicate(fn, message, code)


# ─────────────────────────────────────────────────────────────────────────────
# Combinator Factories
# ─────────────────────────────────────────────────────────────────────────────


def all_of(*rules: Rule) -> Rule:
    """All rules must pass (AND)."""
    return AllOf(rules)


def any_of(*rules: Rule, message: str = "none of the alternatives matched") -> Rule:
    """At least one rule must pass (OR)."""
    return AnyOf(rules, message)


def xor_of(*rules: Rule) -> Rule:
    """Exactly one rule must pass (XOR)."""
    return XorOf(rules)


def not_(rule: Rule, message: str = "must not satisfy condition") -> Rule:
    """Negate rule result."""
    return Not(rule, message)


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Factories
# ─────────────────────────────────────────────────────────────────────────────


def when(condition: CrossFieldCondition, rule: Rule) -> Rule:
    """Apply rule only when condition is True."""
    return When(condition, rule)


def unless(condition: CrossFieldCondition, rule: Rule) -> Rule:
    """Apply rule only when condition is False."""
    return Unless(condition, rule)


def when_eq(field_name: str, expected: object) -> CrossFieldCondition:
    """Condition: field equals expected value."""
    return lambda p: getattr(p, field_name, None) == expected


def when_present(field_name: str) -> CrossFieldCondition:
    """Condition: field is not None."""
    return lambda p: getattr(p, field_name, None) is not None


def when_absent(field_name: str) -> CrossFieldCondition:
    """Condition: field is None."""
    return lambda p: getattr(p, field_name, None) is None


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Field Factories
# ─────────────────────────────────────────────────────────────────────────────


def cross(
    check_fn: Callable[[BaseModel], CheckResult],
    *fields: str,
    message: str = "cross-field validation failed",
    code: str = "CROSS_FIELD",
) -> Rule:
    """Custom cross-field validation."""
    return CrossField(check_fn, fields, message, code)


def less_than(field_a: str, field_b: str) -> Rule:
    """Field A must be less than field B."""
    def check(params: BaseModel) -> CheckResult:
        a, b = getattr(params, field_a, None), getattr(params, field_b, None)
        if a is None or b is None:
            return True
        return a < b or f"'{field_a}' must be less than '{field_b}'"
    return CrossField(check, (field_a, field_b), f"'{field_a}' must be less than '{field_b}'")


def less_than_or_eq(field_a: str, field_b: str) -> Rule:
    """Field A must be ≤ field B."""
    def check(params: BaseModel) -> CheckResult:
        a, b = getattr(params, field_a, None), getattr(params, field_b, None)
        if a is None or b is None:
            return True
        return a <= b or f"'{field_a}' must be ≤ '{field_b}'"
    return CrossField(check, (field_a, field_b), f"'{field_a}' must be ≤ '{field_b}'")


def equals(field_a: str, field_b: str) -> Rule:
    """Field A must equal field B."""
    def check(params: BaseModel) -> CheckResult:
        a, b = getattr(params, field_a, None), getattr(params, field_b, None)
        if a is None or b is None:
            return True
        return a == b or f"'{field_a}' must equal '{field_b}'"
    return CrossField(check, (field_a, field_b), f"'{field_a}' must equal '{field_b}'")


def mutex(*fields: str) -> Rule:
    """Only one of the specified fields may be present (mutual exclusion)."""
    return MutualExclusion(fields)


def together(*fields: str) -> Rule:
    """All specified fields must be present together or all absent."""
    return RequiredTogether(fields)


def at_least_one(*fields: str) -> Rule:
    """At least one of the specified fields must be present."""
    def check(params: BaseModel) -> CheckResult:
        present = any(getattr(params, f, None) is not None for f in fields)
        return present or f"at least one of {', '.join(fields)} is required"
    return CrossField(check, fields, f"at least one of {', '.join(fields)} is required")


def depends_on(field_a: str, *dependencies: str) -> Rule:
    """If field A is present, all dependencies must also be present."""
    def check(params: BaseModel) -> CheckResult:
        if getattr(params, field_a, None) is None:
            return True
        missing = [d for d in dependencies if getattr(params, d, None) is None]
        return not missing or f"'{field_a}' requires: {', '.join(missing)}"
    return CrossField(check, (field_a, *dependencies), f"'{field_a}' has missing dependencies")
