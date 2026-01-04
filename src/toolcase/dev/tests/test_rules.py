"""Tests for Validation Rule DSL.

Validates:
- Atomic rule correctness
- Combinator algebraic laws (associativity, identity, distribution)
- Conditional rule logic
- Cross-field constraint behavior
- Schema composition and validation
- Integration with ValidationMiddleware
"""

from __future__ import annotations

from pydantic import BaseModel

from toolcase.runtime.middleware.plugins.rules import (
    # Core
    Schema, ValidationResult, valid, invalid,
    # Atomic factories
    required, optional, is_type, is_str, is_int, is_float, is_bool, is_list, is_dict,
    in_range, min_len, max_len, length, matches, one_of, url, https, email, predicate,
    # Combinators
    all_of, any_of, xor_of, not_,
    # Conditional
    when, unless, when_eq, when_present, when_absent,
    # Cross-field
    cross, less_than, less_than_or_eq, equals, mutex, together, at_least_one, depends_on,
)


# ═════════════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═════════════════════════════════════════════════════════════════════════════


class TestParams(BaseModel):
    """Test parameter model."""
    name: str | None = None
    age: int | None = None
    email: str | None = None
    url: str | None = None
    method: str | None = None
    body: str | None = None
    start: int | None = None
    end: int | None = None
    sync: bool | None = None
    async_mode: bool | None = None
    tags: list[str] | None = None


def params(**kwargs: object) -> TestParams:
    """Helper to create test params."""
    return TestParams(**kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# ValidationResult Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_valid_result() -> None:
    """Test valid result singleton."""
    v = valid()
    assert v.is_valid
    assert bool(v) is True
    assert list(v) == []


def test_invalid_result() -> None:
    """Test single-violation result."""
    v = invalid("name", "is required")
    assert not v.is_valid
    assert bool(v) is False
    assert len(list(v)) == 1


def test_result_concatenation() -> None:
    """Test result monoid concatenation."""
    a = invalid("name", "is required")
    b = invalid("age", "must be positive")
    combined = a + b
    
    assert len(combined.violations) == 2
    assert combined.violations[0].field == "name"
    assert combined.violations[1].field == "age"


def test_result_format() -> None:
    """Test result formatting."""
    v = invalid("name", "is required") + invalid("age", "must be positive")
    formatted = v.format()
    
    assert "'name'" in formatted
    assert "'age'" in formatted
    assert "is required" in formatted


# ═════════════════════════════════════════════════════════════════════════════
# Atomic Rule Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_required_passes_on_value() -> None:
    """Required passes when value exists."""
    r = required()
    assert r.check("hello", "name", params()).is_valid


def test_required_fails_on_none() -> None:
    """Required fails on None."""
    r = required()
    result = r.check(None, "name", params())
    assert not result.is_valid
    assert "required" in str(result.violations[0])


def test_required_fails_on_empty() -> None:
    """Required fails on empty string/list/dict."""
    r = required()
    
    assert not r.check("", "name", params()).is_valid
    assert not r.check([], "tags", params()).is_valid
    assert not r.check({}, "meta", params()).is_valid


def test_optional_always_passes() -> None:
    """Optional always passes (documentation rule)."""
    r = optional()
    assert r.check(None, "name", params()).is_valid
    assert r.check("value", "name", params()).is_valid


def test_is_type_checks_correctly() -> None:
    """Type checking validates correctly."""
    r = is_str()
    assert r.check("hello", "name", params()).is_valid
    assert r.check(None, "name", params()).is_valid  # None allowed
    assert not r.check(123, "name", params()).is_valid


def test_is_type_multiple() -> None:
    """Type check with multiple types."""
    r = is_type(int, float)
    assert r.check(42, "n", params()).is_valid
    assert r.check(3.14, "n", params()).is_valid
    assert not r.check("string", "n", params()).is_valid


def test_in_range() -> None:
    """Range validation works correctly."""
    r = in_range(0, 100)
    
    assert r.check(50, "age", params()).is_valid
    assert r.check(0, "age", params()).is_valid  # inclusive
    assert r.check(100, "age", params()).is_valid  # inclusive
    assert r.check(None, "age", params()).is_valid  # None allowed
    
    assert not r.check(-1, "age", params()).is_valid
    assert not r.check(101, "age", params()).is_valid


def test_min_len() -> None:
    """Minimum length validation."""
    r = min_len(3)
    
    assert r.check("abc", "name", params()).is_valid
    assert r.check("abcd", "name", params()).is_valid
    assert r.check(None, "name", params()).is_valid
    
    assert not r.check("ab", "name", params()).is_valid
    assert not r.check("", "name", params()).is_valid


def test_max_len() -> None:
    """Maximum length validation."""
    r = max_len(5)
    
    assert r.check("abc", "name", params()).is_valid
    assert r.check("abcde", "name", params()).is_valid
    assert r.check(None, "name", params()).is_valid
    
    assert not r.check("abcdef", "name", params()).is_valid


def test_length_exact() -> None:
    """Exact length validation."""
    r = length(4)
    
    assert r.check("abcd", "code", params()).is_valid
    assert not r.check("abc", "code", params()).is_valid
    assert not r.check("abcde", "code", params()).is_valid


def test_matches_pattern() -> None:
    """Regex pattern matching."""
    r = matches(r"^\d{3}-\d{4}$", "XXX-XXXX format")
    
    assert r.check("123-4567", "phone", params()).is_valid
    assert r.check(None, "phone", params()).is_valid
    
    assert not r.check("12-4567", "phone", params()).is_valid
    assert not r.check("abc-defg", "phone", params()).is_valid


def test_one_of() -> None:
    """Enumeration validation."""
    r = one_of("GET", "POST", "PUT", "DELETE")
    
    assert r.check("GET", "method", params()).is_valid
    assert r.check("POST", "method", params()).is_valid
    assert r.check(None, "method", params()).is_valid
    
    assert not r.check("PATCH", "method", params()).is_valid
    assert not r.check("get", "method", params()).is_valid  # case sensitive


def test_url_validation() -> None:
    """URL format validation."""
    r = url()
    
    assert r.check("http://example.com", "url", params()).is_valid
    assert r.check("https://example.com/path", "url", params()).is_valid
    assert r.check(None, "url", params()).is_valid
    
    assert not r.check("not-a-url", "url", params()).is_valid
    assert not r.check("ftp://example.com", "url", params()).is_valid  # http(s) only


def test_https_only() -> None:
    """HTTPS-only URL validation."""
    r = https()
    
    assert r.check("https://example.com", "url", params()).is_valid
    assert r.check(None, "url", params()).is_valid
    
    assert not r.check("http://example.com", "url", params()).is_valid


def test_email_validation() -> None:
    """Email format validation."""
    r = email()
    
    assert r.check("user@example.com", "email", params()).is_valid
    assert r.check("test.user@sub.domain.org", "email", params()).is_valid
    assert r.check(None, "email", params()).is_valid
    
    assert not r.check("not-an-email", "email", params()).is_valid
    assert not r.check("@missing-user.com", "email", params()).is_valid


def test_predicate_custom() -> None:
    """Custom predicate validation."""
    r = predicate(lambda v: v > 0 if isinstance(v, int) else False, "must be positive")
    
    assert r.check(42, "n", params()).is_valid
    assert not r.check(-1, "n", params()).is_valid
    assert not r.check(0, "n", params()).is_valid


# ═════════════════════════════════════════════════════════════════════════════
# Combinator Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_all_of_accumulates() -> None:
    """AllOf accumulates all violations."""
    r = all_of(min_len(3), max_len(5))
    
    # "a" fails min_len(3)
    result = r.check("a", "name", params())
    assert not result.is_valid
    assert len(result.violations) == 1
    
    # "abcdef" fails max_len(5)
    result = r.check("abcdef", "name", params())
    assert not result.is_valid
    assert len(result.violations) == 1
    
    # "abc" passes both
    assert r.check("abc", "name", params()).is_valid


def test_any_of_short_circuits() -> None:
    """AnyOf short-circuits on first success."""
    r = any_of(one_of("A", "B"), one_of("X", "Y"))
    
    assert r.check("A", "val", params()).is_valid
    assert r.check("Y", "val", params()).is_valid
    
    result = r.check("Z", "val", params())
    assert not result.is_valid


def test_xor_of_exactly_one() -> None:
    """XorOf requires exactly one passing."""
    # Use predicates that check specific values
    r = xor_of(
        predicate(lambda v: v == "A", "must be A"),
        predicate(lambda v: v == "B", "must be B"),
    )
    
    assert r.check("A", "val", params()).is_valid
    assert r.check("B", "val", params()).is_valid
    
    # Neither matches
    result = r.check("C", "val", params())
    assert not result.is_valid
    assert "none" in str(result.violations[0]).lower()


def test_not_negates() -> None:
    """Not negates rule result."""
    r = not_(one_of("admin", "root"), message="cannot use reserved name")
    
    assert r.check("user", "name", params()).is_valid
    
    result = r.check("admin", "name", params())
    assert not result.is_valid
    assert "cannot use reserved name" in str(result.violations[0])


def test_and_operator() -> None:
    """& operator creates AllOf."""
    r = required() & is_str() & min_len(3)
    
    assert r.check("hello", "name", params()).is_valid
    assert not r.check("ab", "name", params()).is_valid
    assert not r.check(None, "name", params()).is_valid


def test_or_operator() -> None:
    """| operator creates AnyOf."""
    r = is_int() | is_str()
    
    assert r.check(42, "val", params()).is_valid
    assert r.check("hello", "val", params()).is_valid


def test_invert_operator() -> None:
    """~ operator creates Not."""
    r = ~one_of("admin", "root")
    
    assert r.check("user", "name", params()).is_valid
    assert not r.check("admin", "name", params()).is_valid


# ═════════════════════════════════════════════════════════════════════════════
# Conditional Rule Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_when_applies_conditionally() -> None:
    """When applies rule only when condition is True."""
    # Require body when method is POST
    r = when(when_eq("method", "POST"), required())
    
    # Condition met, rule applied
    result = r.check(None, "body", params(method="POST"))
    assert not result.is_valid
    
    # Condition not met, rule skipped
    result = r.check(None, "body", params(method="GET"))
    assert result.is_valid


def test_unless_applies_inverted() -> None:
    """Unless applies rule only when condition is False."""
    r = unless(when_present("email"), required())
    
    # email present → skip rule
    result = r.check(None, "phone", params(email="x@x.com"))
    assert result.is_valid
    
    # email absent → apply rule
    result = r.check(None, "phone", params())
    assert not result.is_valid


def test_when_present_condition() -> None:
    """when_present condition."""
    cond = when_present("email")
    
    assert cond(params(email="x@x.com"))
    assert not cond(params())


def test_when_absent_condition() -> None:
    """when_absent condition."""
    cond = when_absent("email")
    
    assert cond(params())
    assert not cond(params(email="x@x.com"))


def test_rule_when_method() -> None:
    """Rule.when() method for fluent API."""
    r = required().when(when_eq("method", "POST"))
    
    assert not r.check(None, "body", params(method="POST")).is_valid
    assert r.check(None, "body", params(method="GET")).is_valid


def test_rule_unless_method() -> None:
    """Rule.unless() method for fluent API."""
    r = required().unless(when_present("email"))
    
    assert r.check(None, "phone", params(email="x@x.com")).is_valid
    assert not r.check(None, "phone", params()).is_valid


# ═════════════════════════════════════════════════════════════════════════════
# Cross-Field Rule Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_less_than() -> None:
    """less_than cross-field constraint."""
    r = less_than("start", "end")
    
    assert r.check(None, "", params(start=1, end=10)).is_valid
    assert r.check(None, "", params(start=None, end=10)).is_valid  # missing ok
    
    result = r.check(None, "", params(start=10, end=5))
    assert not result.is_valid


def test_less_than_or_eq() -> None:
    """less_than_or_eq cross-field constraint."""
    r = less_than_or_eq("start", "end")
    
    assert r.check(None, "", params(start=10, end=10)).is_valid  # equal ok
    assert r.check(None, "", params(start=5, end=10)).is_valid
    
    result = r.check(None, "", params(start=11, end=10))
    assert not result.is_valid


def test_equals_constraint() -> None:
    """equals cross-field constraint."""
    r = equals("password", "confirm_password")
    
    # Create params with both fields
    class PwParams(BaseModel):
        password: str | None = None
        confirm_password: str | None = None
    
    assert r.check(None, "", PwParams(password="secret", confirm_password="secret")).is_valid
    assert not r.check(None, "", PwParams(password="secret", confirm_password="different")).is_valid


def test_mutex_exclusion() -> None:
    """mutex (mutual exclusion) constraint."""
    r = mutex("sync", "async_mode")
    
    assert r.check(None, "", params(sync=True)).is_valid
    assert r.check(None, "", params(async_mode=True)).is_valid
    assert r.check(None, "", params()).is_valid  # neither present
    
    result = r.check(None, "", params(sync=True, async_mode=True))
    assert not result.is_valid
    assert "mutually exclusive" in str(result.violations[0])


def test_together_constraint() -> None:
    """together (required together) constraint."""
    r = together("start", "end")
    
    assert r.check(None, "", params(start=1, end=10)).is_valid  # both present
    assert r.check(None, "", params()).is_valid  # both absent
    
    result = r.check(None, "", params(start=1))  # only one present
    assert not result.is_valid


def test_at_least_one() -> None:
    """at_least_one constraint."""
    r = at_least_one("email", "name")
    
    assert r.check(None, "", params(email="x@x.com")).is_valid
    assert r.check(None, "", params(name="John")).is_valid
    assert r.check(None, "", params(email="x@x.com", name="John")).is_valid
    
    result = r.check(None, "", params())
    assert not result.is_valid
    assert "at least one" in str(result.violations[0])


def test_depends_on() -> None:
    """depends_on constraint."""
    r = depends_on("body", "method")
    
    assert r.check(None, "", params()).is_valid  # body absent
    assert r.check(None, "", params(body="data", method="POST")).is_valid  # dep present
    
    result = r.check(None, "", params(body="data"))  # dep missing
    assert not result.is_valid
    assert "requires" in str(result.violations[0])


def test_custom_cross_field() -> None:
    """Custom cross-field validation."""
    def check_range(p: TestParams) -> bool | str:
        if p.start is None or p.end is None:
            return True
        gap = p.end - p.start
        return gap >= 5 or f"range must be at least 5, got {gap}"
    
    r = cross(check_range, "start", "end", message="invalid range")
    
    assert r.check(None, "", params(start=0, end=10)).is_valid
    
    result = r.check(None, "", params(start=0, end=3))
    assert not result.is_valid
    assert "range must be at least 5" in str(result.violations[0])


# ═════════════════════════════════════════════════════════════════════════════
# Message/Code Override Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_with_message_override() -> None:
    """Override error message."""
    r = required().with_message("please provide a value")
    
    result = r.check(None, "name", params())
    assert not result.is_valid
    assert "please provide a value" in str(result.violations[0])


def test_with_code_override() -> None:
    """Override error code."""
    r = required().with_code("MISSING_FIELD")
    
    result = r.check(None, "name", params())
    assert not result.is_valid
    assert result.violations[0].code == "MISSING_FIELD"


# ═════════════════════════════════════════════════════════════════════════════
# Schema Builder Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_schema_field_validation() -> None:
    """Schema validates field rules."""
    schema = (
        Schema("test_tool")
        .field("name", required() & is_str())
        .field("age", optional() & in_range(0, 150))
    )
    
    # Valid params
    result = schema.validate(params(name="John", age=30))
    assert result.is_valid
    
    # Invalid: missing required name
    result = schema.validate(params(age=30))
    assert not result.is_valid
    
    # Invalid: age out of range
    result = schema.validate(params(name="John", age=200))
    assert not result.is_valid


def test_schema_cross_field_validation() -> None:
    """Schema validates cross-field rules."""
    schema = (
        Schema("range_tool")
        .field("start", optional() & is_int())
        .field("end", optional() & is_int())
        .cross(less_than("start", "end"))
    )
    
    assert schema.validate(params(start=1, end=10)).is_valid
    assert not schema.validate(params(start=10, end=5)).is_valid


def test_schema_accumulates_violations() -> None:
    """Schema accumulates all violations."""
    schema = (
        Schema("multi_error")
        .field("name", required() & min_len(3))
        .field("age", required() & in_range(0, 150))
    )
    
    result = schema.validate(params())  # both missing
    assert not result.is_valid
    assert len(result.violations) >= 2


def test_schema_conditional_validation() -> None:
    """Schema with conditional rules."""
    schema = (
        Schema("http_tool")
        .field("method", required() & one_of("GET", "POST"))
        .field("body", when(when_eq("method", "POST"), required()))
    )
    
    # GET without body is fine
    assert schema.validate(params(method="GET")).is_valid
    
    # POST without body fails
    result = schema.validate(params(method="POST"))
    assert not result.is_valid
    
    # POST with body passes
    assert schema.validate(params(method="POST", body="data")).is_valid


def test_schema_complex_example() -> None:
    """Complex schema demonstrating full DSL capability."""
    schema = (
        Schema("api_request")
        .field("url", required() & url() & https())
        .field("method", required() & one_of("GET", "POST", "PUT", "DELETE"))
        .field("body", when(when_eq("method", "POST"), required()))
        .field("tags", optional() & is_list() & max_len(10))
        .cross(mutex("sync", "async_mode"))
    )
    
    # Valid GET request
    assert schema.validate(params(url="https://api.com", method="GET")).is_valid
    
    # Valid POST with body
    assert schema.validate(params(url="https://api.com", method="POST", body="data")).is_valid
    
    # Invalid: HTTP URL
    result = schema.validate(params(url="http://api.com", method="GET"))
    assert not result.is_valid
    
    # Invalid: POST without body
    result = schema.validate(params(url="https://api.com", method="POST"))
    assert not result.is_valid
    
    # Invalid: both sync modes
    result = schema.validate(params(url="https://api.com", method="GET", sync=True, async_mode=True))
    assert not result.is_valid


# ═════════════════════════════════════════════════════════════════════════════
# Algebraic Law Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_all_of_associativity() -> None:
    """AllOf is associative: (a & b) & c = a & (b & c)."""
    a, b, c = is_str(), min_len(1), max_len(10)
    
    left = (a & b) & c
    right = a & (b & c)
    
    test_values = ["", "a", "hello", "x" * 15]
    for v in test_values:
        l_result = left.check(v, "f", params())
        r_result = right.check(v, "f", params())
        assert l_result.is_valid == r_result.is_valid


def test_any_of_associativity() -> None:
    """AnyOf is associative: (a | b) | c = a | (b | c)."""
    a = predicate(lambda v: v == "A", "must be A")
    b = predicate(lambda v: v == "B", "must be B")
    c = predicate(lambda v: v == "C", "must be C")
    
    left = (a | b) | c
    right = a | (b | c)
    
    test_values = ["A", "B", "C", "D"]
    for v in test_values:
        l_result = left.check(v, "f", params())
        r_result = right.check(v, "f", params())
        assert l_result.is_valid == r_result.is_valid


def test_valid_is_identity() -> None:
    """valid() acts as identity under +."""
    v = invalid("f", "error")
    
    assert (v + valid()).violations == v.violations
    assert (valid() + v).violations == v.violations


# ═════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_middleware_integration() -> None:
    """Test integration with ValidationMiddleware."""
    from toolcase.runtime.middleware.plugins import ValidationMiddleware
    
    middleware = ValidationMiddleware()
    schema = (
        Schema("test_tool")
        .field("name", required() & is_str() & min_len(2))
    )
    middleware.add_schema(schema)
    
    # Verify schema is registered
    assert "test_tool" in middleware._schemas
    assert middleware._schemas["test_tool"] is schema


def test_middleware_schema_method() -> None:
    """Test middleware.schema() fluent builder."""
    from toolcase.runtime.middleware.plugins import ValidationMiddleware
    
    middleware = ValidationMiddleware()
    
    # Build schema in-place
    middleware.schema("my_tool").field("url", required() & https())
    
    assert "my_tool" in middleware._schemas
    assert "url" in middleware._schemas["my_tool"].fields


if __name__ == "__main__":
    import sys
    
    tests = [
        # ValidationResult
        test_valid_result,
        test_invalid_result,
        test_result_concatenation,
        test_result_format,
        # Atomic rules
        test_required_passes_on_value,
        test_required_fails_on_none,
        test_required_fails_on_empty,
        test_optional_always_passes,
        test_is_type_checks_correctly,
        test_is_type_multiple,
        test_in_range,
        test_min_len,
        test_max_len,
        test_length_exact,
        test_matches_pattern,
        test_one_of,
        test_url_validation,
        test_https_only,
        test_email_validation,
        test_predicate_custom,
        # Combinators
        test_all_of_accumulates,
        test_any_of_short_circuits,
        test_xor_of_exactly_one,
        test_not_negates,
        test_and_operator,
        test_or_operator,
        test_invert_operator,
        # Conditional
        test_when_applies_conditionally,
        test_unless_applies_inverted,
        test_when_present_condition,
        test_when_absent_condition,
        test_rule_when_method,
        test_rule_unless_method,
        # Cross-field
        test_less_than,
        test_less_than_or_eq,
        test_equals_constraint,
        test_mutex_exclusion,
        test_together_constraint,
        test_at_least_one,
        test_depends_on,
        test_custom_cross_field,
        # Overrides
        test_with_message_override,
        test_with_code_override,
        # Schema
        test_schema_field_validation,
        test_schema_cross_field_validation,
        test_schema_accumulates_violations,
        test_schema_conditional_validation,
        test_schema_complex_example,
        # Algebraic laws
        test_all_of_associativity,
        test_any_of_associativity,
        test_valid_is_identity,
        # Integration
        test_middleware_integration,
        test_middleware_schema_method,
    ]
    
    failed = 0
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {type(e).__name__}: {e}")
            failed += 1
    
    print(f"\nRan {len(tests)} tests: {len(tests) - failed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
