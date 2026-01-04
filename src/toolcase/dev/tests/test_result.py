"""Tests for Result monad implementation.

Validates via property-based testing:
- Functor laws
- Monad laws
- Applicative laws
- Bifunctor laws
- Error handling correctness
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from toolcase.foundation.errors import Err, Ok, Result, collect_results, sequence, traverse


# ═════════════════════════════════════════════════════════════════════════════
# Hypothesis Strategies
# ═════════════════════════════════════════════════════════════════════════════

# Arbitrary values for property testing
values = st.one_of(
    st.integers(),
    st.text(max_size=50),
    st.floats(allow_nan=False),
    st.booleans(),
    st.none(),
    st.lists(st.integers(), max_size=10),
)

# Result strategies
ok_results = st.builds(Ok, values)
err_results = st.builds(Err, st.text(min_size=1, max_size=20))
results = st.one_of(ok_results, err_results)

# Pure functions for law testing (deterministic, side-effect free)
int_fns = st.sampled_from([
    lambda x: x + 1,
    lambda x: x * 2,
    lambda x: x - 3,
    lambda x: abs(x) if isinstance(x, (int, float)) else x,
    lambda x: x,
])

str_fns = st.sampled_from([
    lambda s: f"E:{s}",
    lambda s: s.upper() if isinstance(s, str) else str(s),
    lambda s: f"[{s}]",
])

monadic_fns = st.sampled_from([
    lambda x: Ok(x + 1) if isinstance(x, (int, float)) else Ok(x),
    lambda x: Ok(x * 2) if isinstance(x, (int, float)) else Ok(x),
    lambda x: Err("fail") if x == 0 else Ok(x),
    lambda x: Ok(x),
])


# ═════════════════════════════════════════════════════════════════════════════
# Property Tests - Functor Laws
# ═════════════════════════════════════════════════════════════════════════════


@given(st.integers())
def test_functor_identity_ok(x: int) -> None:
    """Functor law: fmap id = id (Ok variant)"""
    result: Result[int, str] = Ok(x)
    assert result.map(lambda v: v) == result


@given(st.text(max_size=20))
def test_functor_identity_err(e: str) -> None:
    """Functor law: fmap id = id (Err variant)"""
    result: Result[int, str] = Err(e)
    assert result.map(lambda v: v) == result


@given(st.integers())
@settings(max_examples=200)
def test_functor_composition(x: int) -> None:
    """Functor law: fmap (f . g) = fmap f . fmap g"""
    f, g = lambda v: v + 1, lambda v: v * 2
    result: Result[int, str] = Ok(x)
    assert result.map(lambda v: f(g(v))) == result.map(g).map(f)


# ═════════════════════════════════════════════════════════════════════════════
# Property Tests - Monad Laws
# ═════════════════════════════════════════════════════════════════════════════


@given(st.integers())
def test_monad_left_identity(a: int) -> None:
    """Monad law: return a >>= f = f a"""
    f = lambda x: Ok(x * 2)
    assert Ok(a).flat_map(f) == f(a)


@given(st.integers())
def test_monad_right_identity(x: int) -> None:
    """Monad law: m >>= return = m"""
    m: Result[int, str] = Ok(x)
    assert m.flat_map(Ok) == m


@given(st.integers())
@settings(max_examples=200)
def test_monad_associativity(x: int) -> None:
    """Monad law: (m >>= f) >>= g = m >>= (λx -> f x >>= g)"""
    m: Result[int, str] = Ok(x)
    f, g = lambda v: Ok(v + 1), lambda v: Ok(v * 2)
    assert m.flat_map(f).flat_map(g) == m.flat_map(lambda v: f(v).flat_map(g))


@given(st.text(min_size=1, max_size=20))
def test_monad_err_short_circuits(e: str) -> None:
    """Err propagates through flat_map chain without calling functions."""
    called = []
    
    def tracking_fn(x: int) -> Result[int, str]:
        called.append(x)
        return Ok(x)
    
    result: Result[int, str] = Err(e)
    chained = result.flat_map(tracking_fn).flat_map(tracking_fn)
    
    assert chained.is_err()
    assert chained.unwrap_err() == e
    assert called == []  # Functions never called


# ═════════════════════════════════════════════════════════════════════════════
# Property Tests - Bifunctor Laws
# ═════════════════════════════════════════════════════════════════════════════


@given(st.integers())
def test_bifunctor_identity_ok(x: int) -> None:
    """Bifunctor identity: bimap id id = id"""
    result: Result[int, str] = Ok(x)
    assert result.bimap(lambda v: v, lambda e: e) == result


@given(st.text(min_size=1, max_size=20))
def test_bifunctor_identity_err(e: str) -> None:
    """Bifunctor identity: bimap id id = id"""
    result: Result[int, str] = Err(e)
    assert result.bimap(lambda v: v, lambda err: err) == result


@given(st.integers())
def test_bifunctor_composition_ok(x: int) -> None:
    """Bifunctor composition on Ok: bimap (f.g) (h.i) = bimap f h . bimap g i"""
    f, g = lambda v: v + 1, lambda v: v * 2
    h, i = lambda e: f"A:{e}", lambda e: f"B:{e}"
    
    result: Result[int, str] = Ok(x)
    left = result.bimap(lambda v: f(g(v)), lambda e: h(i(e)))
    right = result.bimap(g, i).bimap(f, h)
    assert left == right


# ═════════════════════════════════════════════════════════════════════════════
# Property Tests - Applicative Laws
# ═════════════════════════════════════════════════════════════════════════════


@given(st.integers())
def test_applicative_identity(x: int) -> None:
    """Applicative identity: pure id <*> v = v"""
    v: Result[int, str] = Ok(x)
    id_fn: Result[type[[int], int], str] = Ok(lambda a: a)
    assert v.apply(id_fn) == v


@given(st.integers())
def test_applicative_homomorphism(x: int) -> None:
    """Applicative homomorphism: pure f <*> pure x = pure (f x)"""
    f = lambda v: v * 2
    assert Ok(x).apply(Ok(f)) == Ok(f(x))


# ═════════════════════════════════════════════════════════════════════════════
# Operational Tests
# ═════════════════════════════════════════════════════════════════════════════


@given(st.integers())
def test_ok_construction(x: int) -> None:
    """Test Ok variant construction and accessors."""
    result: Result[int, str] = Ok(x)
    assert result.is_ok() and not result.is_err()
    assert result.unwrap() == x == result.ok()
    assert result.err() is None


@given(st.text(min_size=1, max_size=50))
def test_err_construction(e: str) -> None:
    """Test Err variant construction and accessors."""
    result: Result[int, str] = Err(e)
    assert result.is_err() and not result.is_ok()
    assert result.unwrap_err() == e == result.err()
    assert result.ok() is None


@given(st.integers(), st.integers())
def test_map_ok(x: int, mult: int) -> None:
    """Test map on Ok variant."""
    result: Result[int, str] = Ok(x)
    mapped = result.map(lambda v: v * mult)
    assert mapped.is_ok() and mapped.unwrap() == x * mult


@given(st.text(min_size=1, max_size=20))
def test_map_err_passthrough(e: str) -> None:
    """Test map on Err variant (should not apply function)."""
    result: Result[int, str] = Err(e)
    mapped = result.map(lambda v: v * 2)
    assert mapped.is_err() and mapped.unwrap_err() == e


@given(st.text(min_size=1, max_size=20))
def test_map_err_transforms(e: str) -> None:
    """Test map_err on Err variant."""
    result: Result[int, str] = Err(e)
    mapped = result.map_err(lambda err: f"E:{err}")
    assert mapped.is_err() and mapped.unwrap_err() == f"E:{e}"


@given(st.integers())
def test_map_err_on_ok_passthrough(x: int) -> None:
    """Test map_err on Ok variant (should not apply function)."""
    result: Result[int, str] = Ok(x)
    mapped = result.map_err(lambda err: f"E:{err}")
    assert mapped.is_ok() and mapped.unwrap() == x


@given(st.integers())
def test_flat_map_ok_to_ok(x: int) -> None:
    """Test flat_map chaining Ok to Ok."""
    result: Result[int, str] = Ok(x)
    chained = result.flat_map(lambda v: Ok(v * 2))
    assert chained.is_ok() and chained.unwrap() == x * 2


@given(st.integers())
def test_flat_map_ok_to_err(x: int) -> None:
    """Test flat_map chaining Ok to Err."""
    result: Result[int, str] = Ok(x)
    chained = result.flat_map(lambda _: Err("failed"))
    assert chained.is_err() and chained.unwrap_err() == "failed"


@given(st.text(min_size=1, max_size=20))
def test_flat_map_err_short_circuit(e: str) -> None:
    """Test flat_map on Err (should short-circuit)."""
    result: Result[int, str] = Err(e)
    chained = result.flat_map(lambda v: Ok(v * 2))
    assert chained.is_err() and chained.unwrap_err() == e


@given(st.integers())
def test_bimap_ok(x: int) -> None:
    """Test bimap on Ok variant."""
    result: Result[int, str] = Ok(x)
    mapped = result.bimap(ok_fn=lambda v: v * 2, err_fn=lambda e: f"E:{e}")
    assert mapped.is_ok() and mapped.unwrap() == x * 2


@given(st.text(min_size=1, max_size=20))
def test_bimap_err(e: str) -> None:
    """Test bimap on Err variant."""
    result: Result[int, str] = Err(e)
    mapped = result.bimap(ok_fn=lambda v: v * 2, err_fn=lambda err: f"E:{err}")
    assert mapped.is_err() and mapped.unwrap_err() == f"E:{e}"


@given(st.integers())
def test_and_then_alias(x: int) -> None:
    """Test and_then is same as flat_map."""
    result: Result[int, str] = Ok(x)
    assert result.flat_map(lambda v: Ok(v * 2)) == result.and_then(lambda v: Ok(v * 2))


@given(st.text(min_size=1, max_size=20), st.integers())
def test_or_else_on_err(e: str, fallback: int) -> None:
    """Test or_else on Err variant."""
    result: Result[int, str] = Err(e)
    recovered = result.or_else(lambda _: Ok(fallback))
    assert recovered.is_ok() and recovered.unwrap() == fallback


@given(st.integers())
def test_or_else_on_ok(x: int) -> None:
    """Test or_else on Ok variant (should not apply)."""
    result: Result[int, str] = Ok(x)
    recovered = result.or_else(lambda _: Ok(42))
    assert recovered.is_ok() and recovered.unwrap() == x


@given(st.integers(), st.integers())
def test_unwrap_or(x: int, default: int) -> None:
    """Test unwrap_or on both variants."""
    assert Ok(x).unwrap_or(default) == x
    assert Err("fail").unwrap_or(default) == default


@given(st.integers(), st.integers())
def test_unwrap_or_else(x: int, fallback: int) -> None:
    """Test unwrap_or_else on both variants."""
    ok_result: Result[int, str] = Ok(x)
    err_result: Result[int, str] = Err("fail")
    assert ok_result.unwrap_or_else(lambda _: fallback) == x
    assert err_result.unwrap_or_else(lambda _: fallback) == fallback


@given(st.integers())
def test_match_ok(x: int) -> None:
    """Test pattern matching on Ok variant."""
    result: Result[int, str] = Ok(x)
    output = result.match(ok=lambda v: f"ok:{v}", err=lambda e: f"err:{e}")
    assert output == f"ok:{x}"


@given(st.text(min_size=1, max_size=20))
def test_match_err(e: str) -> None:
    """Test pattern matching on Err variant."""
    result: Result[int, str] = Err(e)
    output = result.match(ok=lambda v: f"ok:{v}", err=lambda err: f"err:{err}")
    assert output == f"err:{e}"


@given(st.integers())
def test_inspect(x: int) -> None:
    """Test inspect for side effects."""
    inspected = []
    result: Result[int, str] = Ok(x)
    returned = result.inspect(inspected.append)
    assert returned == result and inspected == [x]


@given(st.text(min_size=1, max_size=20))
def test_inspect_err(e: str) -> None:
    """Test inspect_err for side effects."""
    inspected = []
    result: Result[int, str] = Err(e)
    returned = result.inspect_err(inspected.append)
    assert returned == result and inspected == [e]


@given(st.integers())
def test_to_tuple_ok(x: int) -> None:
    """Test conversion to tuple (Ok)."""
    assert Ok(x).to_tuple() == (x, None)


@given(st.text(min_size=1, max_size=20))
def test_to_tuple_err(e: str) -> None:
    """Test conversion to tuple (Err)."""
    assert Err(e).to_tuple() == (None, e)


@given(st.integers())
def test_flatten(x: int) -> None:
    """Test flattening nested Result."""
    nested: Result[Result[int, str], str] = Ok(Ok(x))
    assert nested.flatten().unwrap() == x
    
    nested_err: Result[Result[int, str], str] = Ok(Err("fail"))
    assert nested_err.flatten().is_err()


@given(st.integers(), st.integers())
def test_and_combinator(x: int, y: int) -> None:
    """Test and_ combinator."""
    ok1: Result[int, str] = Ok(x)
    ok2: Result[int, str] = Ok(y)
    err: Result[int, str] = Err("fail")
    
    assert ok1.and_(ok2) == ok2
    assert ok1.and_(err) == err
    assert err.and_(ok1).is_err()


@given(st.integers(), st.integers())
def test_or_combinator(x: int, y: int) -> None:
    """Test or_ combinator."""
    ok: Result[int, str] = Ok(x)
    err: Result[int, str] = Err("fail")
    ok2: Result[int, str] = Ok(y)
    
    assert ok.or_(err) == ok
    assert err.or_(ok2) == ok2


@given(st.integers())
def test_truthiness(x: int) -> None:
    """Test __bool__ for truthiness checks."""
    assert bool(Ok(x)) is True
    assert bool(Err("fail")) is False


@given(st.integers())
def test_equality(x: int) -> None:
    """Test structural equality."""
    assert Ok(x) == Ok(x)
    assert Ok(x) != Ok(x + 1)
    assert Ok(x) != Err(str(x))
    assert Err("a") == Err("a")
    assert Err("a") != Err("b")


@given(st.integers())
def test_iteration(x: int) -> None:
    """Test iteration over Result."""
    assert list(Ok(x)) == [x]
    assert list(Err("fail")) == []


# ═════════════════════════════════════════════════════════════════════════════
# Collection Operations
# ═════════════════════════════════════════════════════════════════════════════


@given(st.lists(st.integers(), min_size=0, max_size=20))
def test_sequence_all_ok(xs: list[int]) -> None:
    """Test sequence with all Ok values."""
    results = [Ok(x) for x in xs]
    sequenced = sequence(results)
    assert sequenced.is_ok() and sequenced.unwrap() == xs


@given(st.lists(st.integers(), min_size=1, max_size=10), st.integers(min_value=0, max_value=9))
def test_sequence_with_err(xs: list[int], err_idx: int) -> None:
    """Test sequence fails fast on first Err."""
    idx = err_idx % len(xs)
    results: list[Result[int, str]] = [Ok(x) for x in xs]
    results[idx] = Err(f"fail@{idx}")
    
    sequenced = sequence(results)
    assert sequenced.is_err()
    # Should fail on first error (which might not be err_idx if there are earlier ones)
    first_err = next(i for i, r in enumerate(results) if r.is_err())
    assert sequenced.unwrap_err() == f"fail@{first_err}"


@given(st.lists(st.from_regex(r"[0-9]+", fullmatch=True), min_size=0, max_size=10))
def test_traverse_all_ok(strs: list[str]) -> None:
    """Test traverse with all successful transformations."""
    def parse_int(s: str) -> Result[int, str]:
        try:
            return Ok(int(s))
        except ValueError:
            return Err(f"invalid: {s}")
    
    result = traverse(strs, parse_int)
    assert result.is_ok()
    assert result.unwrap() == [int(s) for s in strs]


def test_traverse_with_err() -> None:
    """Test traverse fails fast on first error."""
    def parse_int(s: str) -> Result[int, str]:
        try:
            return Ok(int(s))
        except ValueError:
            return Err(f"invalid: {s}")
    
    result = traverse(["1", "bad", "3"], parse_int)
    assert result.is_err() and result.unwrap_err() == "invalid: bad"


@given(st.lists(st.integers(), min_size=0, max_size=20))
def test_collect_results_all_ok(xs: list[int]) -> None:
    """Test collect_results with all Ok values."""
    results = [Ok(x) for x in xs]
    collected = collect_results(results)
    assert collected.is_ok() and collected.unwrap() == xs


@given(st.lists(st.integers(), min_size=2, max_size=10))
def test_collect_results_accumulate_errors(xs: list[int]) -> None:
    """Test collect_results accumulates all errors."""
    results: list[Result[int, str]] = [
        Err(f"e{x}") if x % 2 == 0 else Ok(x) for x in xs
    ]
    collected = collect_results(results)
    
    expected_errs = [f"e{x}" for x in xs if x % 2 == 0]
    if expected_errs:
        assert collected.is_err() and collected.unwrap_err() == expected_errs
    else:
        assert collected.is_ok()


# ═════════════════════════════════════════════════════════════════════════════
# Railway-Oriented Programming Patterns
# ═════════════════════════════════════════════════════════════════════════════


@given(st.integers(min_value=1, max_value=1000))
def test_railway_success_path(x: int) -> None:
    """Test railway-oriented success path."""
    def validate_positive(n: int) -> Result[int, str]:
        return Ok(n) if n > 0 else Err("must be positive")
    
    result = Ok(x).flat_map(validate_positive).map(lambda n: n * 2)
    assert result.is_ok() and result.unwrap() == x * 2


@given(st.integers(max_value=0))
def test_railway_error_path(x: int) -> None:
    """Test railway-oriented error path (short-circuit)."""
    def validate_positive(n: int) -> Result[int, str]:
        return Ok(n) if n > 0 else Err("must be positive")
    
    result = Ok(x).flat_map(validate_positive).map(lambda n: n * 2)
    assert result.is_err() and "must be positive" in result.unwrap_err()


def test_fallback_chain() -> None:
    """Test fallback pattern with or_else."""
    fetch_primary = lambda: Err("primary unavailable")
    fetch_backup = lambda: Err("backup unavailable")
    fetch_cache = lambda: Ok("cached data")
    
    result = fetch_primary().or_else(lambda _: fetch_backup()).or_else(lambda _: fetch_cache())
    assert result.is_ok() and result.unwrap() == "cached data"


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
