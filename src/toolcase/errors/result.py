"""Result/Either monad for type-safe error handling.

Implements a discriminated union for success/failure with full monadic operations:
- Functor: map, map_err
- Applicative: apply
- Monad: flat_map (bind)
- Bifunctor: bimap
- Railway-oriented composition

Performance notes:
- Uses __slots__ for minimal memory footprint
- Direct attribute access (no method calls) in hot paths
- Optimized sequence/traverse with early bailout
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")
F = TypeVar("F")

# Sentinel for faster Ok/Err construction
_OK = True
_ERR = False


class Result(Generic[T, E]):
    """Discriminated union representing success (Ok) or failure (Err).
    
    Sum type enforcing exhaustive error handling. Implements Functor, Applicative,
    Monad, and Bifunctor interfaces for railway-oriented programming.
    
    Examples:
        >>> Ok(42).map(lambda x: x * 2).unwrap()
        84
        >>> Err("fail").map(lambda x: x * 2).unwrap_err()
        'fail'
        >>> Ok(5).flat_map(lambda x: Ok(x * 2) if x > 0 else Err("neg")).unwrap()
        10
    """
    
    __slots__ = ("_value", "_is_ok")
    __match_args__ = ("_value",)
    
    def __init__(self, value: T | E, is_ok: bool) -> None:
        self._value = value
        self._is_ok = is_ok
    
    # ─── Type Checking ───────────────────────────────────────────────
    
    def is_ok(self) -> bool:
        """Check if Result is Ok variant."""
        return self._is_ok
    
    def is_err(self) -> bool:
        """Check if Result is Err variant."""
        return not self._is_ok
    
    # ─── Value Extraction ──────────────────────────────────────────────
    
    def unwrap(self) -> T:
        """Extract Ok value. Raises RuntimeError on Err."""
        if self._is_ok:
            return self._value  # type: ignore[return-value]
        raise RuntimeError(f"unwrap() on Err: {self._value}")
    
    def unwrap_err(self) -> E:
        """Extract Err value. Raises RuntimeError on Ok."""
        if not self._is_ok:
            return self._value  # type: ignore[return-value]
        raise RuntimeError(f"unwrap_err() on Ok: {self._value}")
    
    def unwrap_or(self, default: T) -> T:
        """Extract Ok value or return default."""
        return self._value if self._is_ok else default  # type: ignore[return-value]
    
    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        """Extract Ok value or compute from error via f."""
        return self._value if self._is_ok else f(self._value)  # type: ignore[return-value,arg-type]
    
    def expect(self, msg: str) -> T:
        """Extract Ok value with custom error message."""
        if self._is_ok:
            return self._value  # type: ignore[return-value]
        raise RuntimeError(f"{msg}: {self._value}")
    
    def expect_err(self, msg: str) -> E:
        """Extract Err value with custom error message."""
        if not self._is_ok:
            return self._value  # type: ignore[return-value]
        raise RuntimeError(f"{msg}: {self._value}")
    
    # ─── Functor Operations ────────────────────────────────────────────
    
    def map(self, f: Callable[[T], U]) -> Result[U, E]:
        """Apply f to Ok value. Signature: Result[T,E] → (T→U) → Result[U,E]"""
        return Result(f(self._value), _OK) if self._is_ok else Result(self._value, _ERR)  # type: ignore[arg-type]
    
    def map_err(self, f: Callable[[E], F]) -> Result[T, F]:
        """Apply f to Err value. Signature: Result[T,E] → (E→F) → Result[T,F]"""
        return Result(f(self._value), _ERR) if not self._is_ok else Result(self._value, _OK)  # type: ignore[arg-type]
    
    # ─── Bifunctor Operations ──────────────────────────────────────────
    
    def bimap(self, ok_fn: Callable[[T], U], err_fn: Callable[[E], F]) -> Result[U, F]:
        """Apply ok_fn if Ok, err_fn if Err. Signature: Result[T,E] → (T→U, E→F) → Result[U,F]"""
        return Result(ok_fn(self._value), _OK) if self._is_ok else Result(err_fn(self._value), _ERR)  # type: ignore[arg-type]
    
    # ─── Monad Operations ──────────────────────────────────────────────
    
    def flat_map(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Monadic bind (>>=). Chain operations that can fail.
        
        Example:
            >>> Ok("42").flat_map(lambda s: Ok(int(s))).flat_map(lambda n: Ok(n*2) if n>0 else Err("neg"))
        """
        return f(self._value) if self._is_ok else Result(self._value, _ERR)  # type: ignore[arg-type]
    
    def and_then(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Alias for flat_map."""
        return f(self._value) if self._is_ok else Result(self._value, _ERR)  # type: ignore[arg-type]
    
    def or_else(self, f: Callable[[E], Result[T, F]]) -> Result[T, F]:
        """On Err, apply f to recover. On Ok, pass through."""
        return f(self._value) if not self._is_ok else Result(self._value, _OK)  # type: ignore[arg-type]
    
    # ─── Applicative Operations ─────────────────────────────────────────
    
    def apply(self, f_result: Result[Callable[[T], U], E]) -> Result[U, E]:
        """Apply wrapped function to wrapped value (Applicative)."""
        if f_result._is_ok and self._is_ok:
            return Result(f_result._value(self._value), _OK)  # type: ignore[operator]
        return Result(f_result._value if not f_result._is_ok else self._value, _ERR)  # type: ignore[arg-type]
    
    # ─── Logical Combinators ─────────────────────────────────────────────
    
    def and_(self, other: Result[U, E]) -> Result[U, E]:
        """Return other if Ok, else self's Err. Short-circuit AND."""
        return other if self._is_ok else Result(self._value, _ERR)  # type: ignore[arg-type]
    
    def or_(self, other: Result[T, F]) -> Result[T, F]:
        """Return self if Ok, else other. Short-circuit OR."""
        return Result(self._value, _OK) if self._is_ok else other  # type: ignore[arg-type]
    
    # ─── Inspection & Utilities ──────────────────────────────────────────
    
    def ok(self) -> T | None:
        """Some(T) if Ok, None if Err."""
        return self._value if self._is_ok else None  # type: ignore[return-value]
    
    def err(self) -> E | None:
        """Some(E) if Err, None if Ok."""
        return self._value if not self._is_ok else None  # type: ignore[return-value]
    
    def inspect(self, f: Callable[[T], None]) -> Result[T, E]:
        """Call f with Ok value for side effects, return self."""
        if self._is_ok:
            f(self._value)  # type: ignore[arg-type]
        return self
    
    def inspect_err(self, f: Callable[[E], None]) -> Result[T, E]:
        """Call f with Err value for side effects, return self."""
        if not self._is_ok:
            f(self._value)  # type: ignore[arg-type]
        return self
    
    # ─── Pattern Matching ────────────────────────────────────────────────
    
    def match(self, *, ok: Callable[[T], U], err: Callable[[E], U]) -> U:
        """Exhaustive pattern match. Forces handling both Ok and Err."""
        return ok(self._value) if self._is_ok else err(self._value)  # type: ignore[arg-type]
    
    # ─── Conversion ────────────────────────────────────────────────────────
    
    def to_tuple(self) -> tuple[T | None, E | None]:
        """Convert to (ok_value, err_value) tuple."""
        return (self._value, None) if self._is_ok else (None, self._value)  # type: ignore[return-value]
    
    def flatten(self: Result[Result[T, E], E]) -> Result[T, E]:
        """Flatten nested Result. Result[Result[T,E],E] → Result[T,E]"""
        return self._value if self._is_ok else Result(self._value, _ERR)  # type: ignore[return-value,arg-type]
    
    # ─── Dunder Methods ──────────────────────────────────────────────────
    
    __bool__ = lambda self: self._is_ok  # noqa: E731
    __hash__ = lambda self: hash((self._is_ok, self._value))  # noqa: E731
    __repr__ = lambda self: f"{'Ok' if self._is_ok else 'Err'}({self._value!r})"  # noqa: E731
    __str__ = __repr__
    
    def __eq__(self, other: object) -> bool:
        return self._is_ok == other._is_ok and self._value == other._value if isinstance(other, Result) else NotImplemented
    
    def __iter__(self) -> Iterator[T]:
        """Iterate: yields value if Ok, nothing if Err."""
        if self._is_ok:
            yield self._value  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════════
# Constructors
# ═══════════════════════════════════════════════════════════════════════════════


def Ok(value: T) -> Result[T, E]:  # noqa: N802
    """Construct Ok variant (success)."""
    return Result(value, _OK)


def Err(error: E) -> Result[T, E]:  # noqa: N802
    """Construct Err variant (failure)."""
    return Result(error, _ERR)


# ═══════════════════════════════════════════════════════════════════════════════
# Collection Operations
# ═══════════════════════════════════════════════════════════════════════════════


def sequence(results: list[Result[T, E]]) -> Result[list[T], E]:
    """List[Result[T,E]] → Result[List[T], E]. Fail-fast on first Err."""
    values: list[T] = []
    for r in results:
        if not r._is_ok:
            return Result(r._value, _ERR)  # type: ignore[arg-type]
        values.append(r._value)  # type: ignore[arg-type]
    return Result(values, _OK)


def traverse(items: list[T], f: Callable[[T], Result[U, E]]) -> Result[list[U], E]:
    """Map f over items, sequence results. Fail-fast on first Err."""
    values: list[U] = []
    for item in items:
        r = f(item)
        if not r._is_ok:
            return Result(r._value, _ERR)  # type: ignore[arg-type]
        values.append(r._value)  # type: ignore[arg-type]
    return Result(values, _OK)


def collect_results(results: list[Result[T, E]]) -> Result[list[T], list[E]]:
    """Collect all Results, accumulating ALL errors (not fail-fast)."""
    values: list[T] = []
    errors: list[E] = []
    for r in results:
        (values if r._is_ok else errors).append(r._value)  # type: ignore[arg-type]
    return Result(values, _OK) if not errors else Result(errors, _ERR)
