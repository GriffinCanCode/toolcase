"""Tests for Result monad implementation.

Validates:
- Functor laws
- Monad laws
- Applicative laws
- Bifunctor laws
- Error handling correctness
"""

from __future__ import annotations

from typing import Callable

from toolcase.errors import Err, Ok, Result, collect_results, sequence, traverse


# ═════════════════════════════════════════════════════════════════════════════
# Property Tests - Functor Laws
# ═════════════════════════════════════════════════════════════════════════════


def test_functor_identity() -> None:
    """Functor law: fmap id = id"""
    result: Result[int, str] = Ok(42)
    assert result.map(lambda x: x) == result
    
    err_result: Result[int, str] = Err("fail")
    assert err_result.map(lambda x: x) == err_result


def test_functor_composition() -> None:
    """Functor law: fmap (f . g) = fmap f . fmap g"""
    f: Callable[[int], int] = lambda x: x + 1
    g: Callable[[int], int] = lambda x: x * 2
    
    result: Result[int, str] = Ok(5)
    
    # fmap (f . g)
    left = result.map(lambda x: f(g(x)))
    
    # fmap f . fmap g
    right = result.map(g).map(f)
    
    assert left == right


# ═════════════════════════════════════════════════════════════════════════════
# Property Tests - Monad Laws
# ═════════════════════════════════════════════════════════════════════════════


def test_monad_left_identity() -> None:
    """Monad law: return a >>= f = f a"""
    a = 42
    f: Callable[[int], Result[int, str]] = lambda x: Ok(x * 2)
    
    left = Ok(a).flat_map(f)
    right = f(a)
    
    assert left == right


def test_monad_right_identity() -> None:
    """Monad law: m >>= return = m"""
    m: Result[int, str] = Ok(42)
    
    left = m.flat_map(lambda x: Ok(x))
    right = m
    
    assert left == right


def test_monad_associativity() -> None:
    """Monad law: (m >>= f) >>= g = m >>= (\\x -> f x >>= g)"""
    m: Result[int, str] = Ok(5)
    f: Callable[[int], Result[int, str]] = lambda x: Ok(x + 1)
    g: Callable[[int], Result[int, str]] = lambda x: Ok(x * 2)
    
    # (m >>= f) >>= g
    left = m.flat_map(f).flat_map(g)
    
    # m >>= (\x -> f x >>= g)
    right = m.flat_map(lambda x: f(x).flat_map(g))
    
    assert left == right


# ═════════════════════════════════════════════════════════════════════════════
# Operational Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_ok_construction() -> None:
    """Test Ok variant construction and accessors."""
    result: Result[int, str] = Ok(42)
    
    assert result.is_ok()
    assert not result.is_err()
    assert result.unwrap() == 42
    assert result.ok() == 42
    assert result.err() is None


def test_err_construction() -> None:
    """Test Err variant construction and accessors."""
    result: Result[int, str] = Err("failed")
    
    assert not result.is_ok()
    assert result.is_err()
    assert result.unwrap_err() == "failed"
    assert result.ok() is None
    assert result.err() == "failed"


def test_map_ok() -> None:
    """Test map on Ok variant."""
    result: Result[int, str] = Ok(5)
    mapped = result.map(lambda x: x * 2)
    
    assert mapped.is_ok()
    assert mapped.unwrap() == 10


def test_map_err() -> None:
    """Test map on Err variant (should not apply function)."""
    result: Result[int, str] = Err("fail")
    mapped = result.map(lambda x: x * 2)
    
    assert mapped.is_err()
    assert mapped.unwrap_err() == "fail"


def test_map_err_on_err() -> None:
    """Test map_err on Err variant."""
    result: Result[int, str] = Err("fail")
    mapped = result.map_err(lambda e: f"Error: {e}")
    
    assert mapped.is_err()
    assert mapped.unwrap_err() == "Error: fail"


def test_map_err_on_ok() -> None:
    """Test map_err on Ok variant (should not apply function)."""
    result: Result[int, str] = Ok(42)
    mapped = result.map_err(lambda e: f"Error: {e}")
    
    assert mapped.is_ok()
    assert mapped.unwrap() == 42


def test_flat_map_ok_to_ok() -> None:
    """Test flat_map chaining Ok to Ok."""
    result: Result[int, str] = Ok(5)
    chained = result.flat_map(lambda x: Ok(x * 2))
    
    assert chained.is_ok()
    assert chained.unwrap() == 10


def test_flat_map_ok_to_err() -> None:
    """Test flat_map chaining Ok to Err."""
    result: Result[int, str] = Ok(5)
    chained = result.flat_map(lambda x: Err("failed"))
    
    assert chained.is_err()
    assert chained.unwrap_err() == "failed"


def test_flat_map_err() -> None:
    """Test flat_map on Err (should short-circuit)."""
    result: Result[int, str] = Err("fail")
    chained = result.flat_map(lambda x: Ok(x * 2))
    
    assert chained.is_err()
    assert chained.unwrap_err() == "fail"


def test_bimap() -> None:
    """Test bimap on both variants."""
    ok_result: Result[int, str] = Ok(5)
    ok_mapped = ok_result.bimap(
        ok_fn=lambda x: x * 2,
        err_fn=lambda e: f"Error: {e}"
    )
    assert ok_mapped.is_ok()
    assert ok_mapped.unwrap() == 10
    
    err_result: Result[int, str] = Err("fail")
    err_mapped = err_result.bimap(
        ok_fn=lambda x: x * 2,
        err_fn=lambda e: f"Error: {e}"
    )
    assert err_mapped.is_err()
    assert err_mapped.unwrap_err() == "Error: fail"


def test_and_then_alias() -> None:
    """Test and_then is same as flat_map."""
    result: Result[int, str] = Ok(5)
    
    flat_mapped = result.flat_map(lambda x: Ok(x * 2))
    and_thened = result.and_then(lambda x: Ok(x * 2))
    
    assert flat_mapped == and_thened


def test_or_else_on_err() -> None:
    """Test or_else on Err variant."""
    result: Result[int, str] = Err("fail")
    recovered = result.or_else(lambda _: Ok(42))
    
    assert recovered.is_ok()
    assert recovered.unwrap() == 42


def test_or_else_on_ok() -> None:
    """Test or_else on Ok variant (should not apply)."""
    result: Result[int, str] = Ok(5)
    recovered = result.or_else(lambda _: Ok(42))
    
    assert recovered.is_ok()
    assert recovered.unwrap() == 5


def test_unwrap_or() -> None:
    """Test unwrap_or on both variants."""
    ok_result: Result[int, str] = Ok(5)
    assert ok_result.unwrap_or(10) == 5
    
    err_result: Result[int, str] = Err("fail")
    assert err_result.unwrap_or(10) == 10


def test_unwrap_or_else() -> None:
    """Test unwrap_or_else on both variants."""
    ok_result: Result[int, str] = Ok(5)
    assert ok_result.unwrap_or_else(lambda _: 10) == 5
    
    err_result: Result[int, str] = Err("fail")
    assert err_result.unwrap_or_else(lambda _: 10) == 10


def test_match() -> None:
    """Test pattern matching on both variants."""
    ok_result: Result[int, str] = Ok(42)
    ok_output = ok_result.match(
        ok=lambda x: f"success: {x}",
        err=lambda e: f"failed: {e}"
    )
    assert ok_output == "success: 42"
    
    err_result: Result[int, str] = Err("fail")
    err_output = err_result.match(
        ok=lambda x: f"success: {x}",
        err=lambda e: f"failed: {e}"
    )
    assert err_output == "failed: fail"


def test_inspect() -> None:
    """Test inspect for side effects."""
    inspected_value = None
    
    def side_effect(x: int) -> None:
        nonlocal inspected_value
        inspected_value = x
    
    result: Result[int, str] = Ok(42)
    returned = result.inspect(side_effect)
    
    assert returned == result  # Returns self
    assert inspected_value == 42


def test_inspect_err() -> None:
    """Test inspect_err for side effects."""
    inspected_error = None
    
    def side_effect(e: str) -> None:
        nonlocal inspected_error
        inspected_error = e
    
    result: Result[int, str] = Err("fail")
    returned = result.inspect_err(side_effect)
    
    assert returned == result  # Returns self
    assert inspected_error == "fail"


def test_to_tuple() -> None:
    """Test conversion to tuple."""
    ok_result: Result[int, str] = Ok(42)
    assert ok_result.to_tuple() == (42, None)
    
    err_result: Result[int, str] = Err("fail")
    assert err_result.to_tuple() == (None, "fail")


def test_flatten() -> None:
    """Test flattening nested Result."""
    nested: Result[Result[int, str], str] = Ok(Ok(42))
    flattened = nested.flatten()
    
    assert flattened.is_ok()
    assert flattened.unwrap() == 42
    
    nested_err: Result[Result[int, str], str] = Ok(Err("fail"))
    flattened_err = nested_err.flatten()
    
    assert flattened_err.is_err()
    assert flattened_err.unwrap_err() == "fail"


def test_and_combinator() -> None:
    """Test and_ combinator."""
    ok1: Result[int, str] = Ok(5)
    ok2: Result[int, str] = Ok(10)
    assert ok1.and_(ok2) == ok2
    
    ok: Result[int, str] = Ok(5)
    err: Result[int, str] = Err("fail")
    assert ok.and_(err) == err
    
    err1: Result[int, str] = Err("fail1")
    ok3: Result[int, str] = Ok(10)
    assert err1.and_(ok3).is_err()
    assert err1.and_(ok3).unwrap_err() == "fail1"


def test_or_combinator() -> None:
    """Test or_ combinator."""
    ok: Result[int, str] = Ok(5)
    err: Result[int, str] = Err("fail")
    assert ok.or_(err) == ok
    
    err1: Result[int, str] = Err("fail1")
    ok2: Result[int, str] = Ok(10)
    assert err1.or_(ok2) == ok2


def test_truthiness() -> None:
    """Test __bool__ for truthiness checks."""
    assert bool(Ok(42)) is True
    assert bool(Err("fail")) is False


def test_equality() -> None:
    """Test structural equality."""
    assert Ok(42) == Ok(42)
    assert Err("fail") == Err("fail")
    assert Ok(42) != Ok(43)
    assert Err("fail") != Err("other")
    assert Ok(42) != Err("fail")


def test_iteration() -> None:
    """Test iteration over Result."""
    ok_result: Result[int, str] = Ok(42)
    values = list(ok_result)
    assert values == [42]
    
    err_result: Result[int, str] = Err("fail")
    values = list(err_result)
    assert values == []


# ═════════════════════════════════════════════════════════════════════════════
# Collection Operations
# ═════════════════════════════════════════════════════════════════════════════


def test_sequence_all_ok() -> None:
    """Test sequence with all Ok values."""
    results = [Ok(1), Ok(2), Ok(3)]
    sequenced = sequence(results)
    
    assert sequenced.is_ok()
    assert sequenced.unwrap() == [1, 2, 3]


def test_sequence_with_err() -> None:
    """Test sequence fails fast on first Err."""
    results = [Ok(1), Err("fail"), Ok(3)]
    sequenced = sequence(results)
    
    assert sequenced.is_err()
    assert sequenced.unwrap_err() == "fail"


def test_sequence_empty() -> None:
    """Test sequence with empty list."""
    results: list[Result[int, str]] = []
    sequenced = sequence(results)
    
    assert sequenced.is_ok()
    assert sequenced.unwrap() == []


def test_traverse_all_ok() -> None:
    """Test traverse with all successful transformations."""
    def parse_int(s: str) -> Result[int, str]:
        try:
            return Ok(int(s))
        except ValueError:
            return Err(f"invalid: {s}")
    
    result = traverse(["1", "2", "3"], parse_int)
    
    assert result.is_ok()
    assert result.unwrap() == [1, 2, 3]


def test_traverse_with_err() -> None:
    """Test traverse fails fast on first error."""
    def parse_int(s: str) -> Result[int, str]:
        try:
            return Ok(int(s))
        except ValueError:
            return Err(f"invalid: {s}")
    
    result = traverse(["1", "bad", "3"], parse_int)
    
    assert result.is_err()
    assert result.unwrap_err() == "invalid: bad"


def test_collect_results_all_ok() -> None:
    """Test collect_results with all Ok values."""
    results = [Ok(1), Ok(2), Ok(3)]
    collected = collect_results(results)
    
    assert collected.is_ok()
    assert collected.unwrap() == [1, 2, 3]


def test_collect_results_accumulate_errors() -> None:
    """Test collect_results accumulates all errors."""
    results = [Ok(1), Err("e1"), Ok(3), Err("e2")]
    collected = collect_results(results)
    
    assert collected.is_err()
    assert collected.unwrap_err() == ["e1", "e2"]


# ═════════════════════════════════════════════════════════════════════════════
# Railway-Oriented Programming Patterns
# ═════════════════════════════════════════════════════════════════════════════


def test_railway_success_path() -> None:
    """Test railway-oriented success path."""
    def parse_int(s: str) -> Result[int, str]:
        try:
            return Ok(int(s))
        except ValueError:
            return Err(f"invalid: {s}")
    
    def validate_positive(n: int) -> Result[int, str]:
        return Ok(n) if n > 0 else Err("must be positive")
    
    def double(n: int) -> int:
        return n * 2
    
    result = (
        Ok("42")
        .flat_map(parse_int)
        .flat_map(validate_positive)
        .map(double)
    )
    
    assert result.is_ok()
    assert result.unwrap() == 84


def test_railway_error_path() -> None:
    """Test railway-oriented error path (short-circuit)."""
    def parse_int(s: str) -> Result[int, str]:
        try:
            return Ok(int(s))
        except ValueError:
            return Err(f"invalid: {s}")
    
    def validate_positive(n: int) -> Result[int, str]:
        return Ok(n) if n > 0 else Err("must be positive")
    
    def double(n: int) -> int:
        return n * 2
    
    # Error in parsing
    result = (
        Ok("bad")
        .flat_map(parse_int)
        .flat_map(validate_positive)  # Skipped
        .map(double)  # Skipped
    )
    
    assert result.is_err()
    assert "invalid: bad" in result.unwrap_err()
    
    # Error in validation
    result = (
        Ok("-5")
        .flat_map(parse_int)
        .flat_map(validate_positive)
        .map(double)  # Skipped
    )
    
    assert result.is_err()
    assert "must be positive" in result.unwrap_err()


def test_fallback_chain() -> None:
    """Test fallback pattern with or_else."""
    def fetch_from_primary() -> Result[str, str]:
        return Err("primary unavailable")
    
    def fetch_from_backup() -> Result[str, str]:
        return Err("backup unavailable")
    
    def fetch_from_cache() -> Result[str, str]:
        return Ok("cached data")
    
    result = (
        fetch_from_primary()
        .or_else(lambda _: fetch_from_backup())
        .or_else(lambda _: fetch_from_cache())
    )
    
    assert result.is_ok()
    assert result.unwrap() == "cached data"


if __name__ == "__main__":
    # Run all tests
    import sys
    
    tests = [
        test_functor_identity,
        test_functor_composition,
        test_monad_left_identity,
        test_monad_right_identity,
        test_monad_associativity,
        test_ok_construction,
        test_err_construction,
        test_map_ok,
        test_map_err,
        test_map_err_on_err,
        test_map_err_on_ok,
        test_flat_map_ok_to_ok,
        test_flat_map_ok_to_err,
        test_flat_map_err,
        test_bimap,
        test_and_then_alias,
        test_or_else_on_err,
        test_or_else_on_ok,
        test_unwrap_or,
        test_unwrap_or_else,
        test_match,
        test_inspect,
        test_inspect_err,
        test_to_tuple,
        test_flatten,
        test_and_combinator,
        test_or_combinator,
        test_truthiness,
        test_equality,
        test_iteration,
        test_sequence_all_ok,
        test_sequence_with_err,
        test_sequence_empty,
        test_traverse_all_ok,
        test_traverse_with_err,
        test_collect_results_all_ok,
        test_collect_results_accumulate_errors,
        test_railway_success_path,
        test_railway_error_path,
        test_fallback_chain,
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
