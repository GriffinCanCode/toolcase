"""Result/Either monad for type-safe error handling.

Implements a discriminated union for success/failure with full monadic operations:
- Functor: map, map_err
- Applicative: apply
- Monad: flat_map (bind)
- Bifunctor: bimap
- Railway-oriented composition
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    Literal,
    NoReturn,
    TypeVar,
    cast,
    overload,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# Type variables for generic Result
T = TypeVar("T")  # Success type
E = TypeVar("E")  # Error type
U = TypeVar("U")  # Mapped success type
F = TypeVar("F")  # Mapped error type


class Result(Generic[T, E]):
    """Discriminated union representing success (Ok) or failure (Err).
    
    This is a sum type that enforces exhaustive error handling at the type level.
    Inspired by Rust's Result and Haskell's Either.
    
    The Result type implements:
    - Functor: map, map_err
    - Applicative: apply
    - Monad: flat_map (>>=)
    - Bifunctor: bimap
    
    Examples:
        >>> result: Result[int, str] = Ok(42)
        >>> result.map(lambda x: x * 2).unwrap()
        84
        
        >>> error: Result[int, str] = Err("failed")
        >>> error.map(lambda x: x * 2).unwrap_err()
        'failed'
        
        Railway-oriented programming:
        >>> def validate_positive(x: int) -> Result[int, str]:
        ...     return Ok(x) if x > 0 else Err("must be positive")
        >>> 
        >>> result = (
        ...     Ok(5)
        ...     .flat_map(validate_positive)
        ...     .map(lambda x: x * 2)
        ... )
        >>> assert result.unwrap() == 10
    
    Notes:
        - Uses __slots__ for zero overhead
        - Immutable by design (all operations return new Result)
        - Pattern matching via is_ok()/is_err() + match()
    """
    
    __slots__ = ("_value", "_is_ok")
    __match_args__ = ("_value",)
    
    def __init__(self, value: T | E, is_ok: bool) -> None:
        """Private constructor. Use Ok() or Err() instead."""
        self._value: T | E = value
        self._is_ok: bool = is_ok
    
    # ─────────────────────────────────────────────────────────────────
    # Type Checking
    # ─────────────────────────────────────────────────────────────────
    
    def is_ok(self) -> bool:
        """Check if Result is Ok variant."""
        return self._is_ok
    
    def is_err(self) -> bool:
        """Check if Result is Err variant."""
        return not self._is_ok
    
    # ─────────────────────────────────────────────────────────────────
    # Value Extraction
    # ─────────────────────────────────────────────────────────────────
    
    def unwrap(self) -> T:
        """Extract Ok value, panic on Err.
        
        Raises:
            RuntimeError: If Result is Err
        """
        if self._is_ok:
            return cast(T, self._value)
        raise RuntimeError(f"Called unwrap() on Err value: {self._value}")
    
    def unwrap_err(self) -> E:
        """Extract Err value, panic on Ok.
        
        Raises:
            RuntimeError: If Result is Ok
        """
        if not self._is_ok:
            return cast(E, self._value)
        raise RuntimeError(f"Called unwrap_err() on Ok value: {self._value}")
    
    def unwrap_or(self, default: T) -> T:
        """Extract Ok value or return default."""
        return cast(T, self._value) if self._is_ok else default
    
    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        """Extract Ok value or compute from error."""
        return cast(T, self._value) if self._is_ok else f(cast(E, self._value))
    
    def expect(self, msg: str) -> T:
        """Extract Ok value with custom panic message.
        
        Raises:
            RuntimeError: If Result is Err with custom message
        """
        if self._is_ok:
            return cast(T, self._value)
        raise RuntimeError(f"{msg}: {self._value}")
    
    def expect_err(self, msg: str) -> E:
        """Extract Err value with custom panic message.
        
        Raises:
            RuntimeError: If Result is Ok with custom message
        """
        if not self._is_ok:
            return cast(E, self._value)
        raise RuntimeError(f"{msg}: {self._value}")
    
    # ─────────────────────────────────────────────────────────────────
    # Functor Operations
    # ─────────────────────────────────────────────────────────────────
    
    def map(self, f: Callable[[T], U]) -> Result[U, E]:
        """Map function over Ok value (Functor).
        
        This is the fundamental functor operation. Applies f only if Ok,
        preserves Err unchanged.
        
        Type signature: Result[T, E] -> (T -> U) -> Result[U, E]
        """
        if self._is_ok:
            return Ok(f(cast(T, self._value)))
        return Err(cast(E, self._value))
    
    def map_err(self, f: Callable[[E], F]) -> Result[T, F]:
        """Map function over Err value (Error Functor).
        
        Useful for transforming error types while preserving Ok values.
        
        Type signature: Result[T, E] -> (E -> F) -> Result[T, F]
        """
        if not self._is_ok:
            return Err(f(cast(E, self._value)))
        return Ok(cast(T, self._value))
    
    # ─────────────────────────────────────────────────────────────────
    # Bifunctor Operations
    # ─────────────────────────────────────────────────────────────────
    
    def bimap(self, ok_fn: Callable[[T], U], err_fn: Callable[[E], F]) -> Result[U, F]:
        """Map both Ok and Err values (Bifunctor).
        
        Applies ok_fn if Ok, err_fn if Err.
        
        Type signature: Result[T, E] -> (T -> U, E -> F) -> Result[U, F]
        """
        if self._is_ok:
            return Ok(ok_fn(cast(T, self._value)))
        return Err(err_fn(cast(E, self._value)))
    
    # ─────────────────────────────────────────────────────────────────
    # Monad Operations
    # ─────────────────────────────────────────────────────────────────
    
    def flat_map(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Monadic bind (>>=) - chain operations that can fail.
        
        This is the key operation for railway-oriented programming.
        Enables chaining operations where each step can fail.
        
        Type signature: Result[T, E] -> (T -> Result[U, E]) -> Result[U, E]
        
        Example:
            >>> def parse_int(s: str) -> Result[int, str]:
            ...     try:
            ...         return Ok(int(s))
            ...     except ValueError:
            ...         return Err(f"invalid int: {s}")
            >>> 
            >>> def validate_positive(n: int) -> Result[int, str]:
            ...     return Ok(n) if n > 0 else Err("must be positive")
            >>> 
            >>> result = (
            ...     Ok("42")
            ...     .flat_map(parse_int)
            ...     .flat_map(validate_positive)
            ... )
            >>> assert result.unwrap() == 42
        """
        if self._is_ok:
            return f(cast(T, self._value))
        return Err(cast(E, self._value))
    
    def and_then(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Alias for flat_map for better readability."""
        return self.flat_map(f)
    
    def or_else(self, f: Callable[[E], Result[T, F]]) -> Result[T, F]:
        """Chain alternative on Err.
        
        If Err, applies f to transform/recover. If Ok, passes through.
        
        Type signature: Result[T, E] -> (E -> Result[T, F]) -> Result[T, F]
        """
        if not self._is_ok:
            return f(cast(E, self._value))
        return Ok(cast(T, self._value))
    
    # ─────────────────────────────────────────────────────────────────
    # Applicative Operations
    # ─────────────────────────────────────────────────────────────────
    
    def apply(self, f_result: Result[Callable[[T], U], E]) -> Result[U, E]:
        """Apply wrapped function to wrapped value (Applicative).
        
        Enables parallel validation and accumulation patterns.
        
        Type signature: Result[T, E] -> Result[T -> U, E] -> Result[U, E]
        """
        if f_result.is_ok() and self._is_ok:
            fn = f_result.unwrap()
            return Ok(fn(cast(T, self._value)))
        if f_result.is_err():
            return Err(f_result.unwrap_err())
        return Err(cast(E, self._value))
    
    # ─────────────────────────────────────────────────────────────────
    # Logical Combinators
    # ─────────────────────────────────────────────────────────────────
    
    def and_(self, other: Result[U, E]) -> Result[U, E]:
        """Return other if self is Ok, otherwise return self's Err.
        
        Short-circuit AND - useful for sequencing.
        """
        return other if self._is_ok else Err(cast(E, self._value))
    
    def or_(self, other: Result[T, F]) -> Result[T, F]:
        """Return self if Ok, otherwise return other.
        
        Short-circuit OR - useful for fallbacks.
        """
        return Ok(cast(T, self._value)) if self._is_ok else other
    
    # ─────────────────────────────────────────────────────────────────
    # Inspection & Utilities
    # ─────────────────────────────────────────────────────────────────
    
    def ok(self) -> T | None:
        """Convert to Option-like: Some(T) if Ok, None if Err."""
        return cast(T, self._value) if self._is_ok else None
    
    def err(self) -> E | None:
        """Convert to Option-like: Some(E) if Err, None if Ok."""
        return cast(E, self._value) if not self._is_ok else None
    
    def inspect(self, f: Callable[[T], None]) -> Result[T, E]:
        """Call function with Ok value for side effects, return self."""
        if self._is_ok:
            f(cast(T, self._value))
        return self
    
    def inspect_err(self, f: Callable[[E], None]) -> Result[T, E]:
        """Call function with Err value for side effects, return self."""
        if not self._is_ok:
            f(cast(E, self._value))
        return self
    
    # ─────────────────────────────────────────────────────────────────
    # Pattern Matching
    # ─────────────────────────────────────────────────────────────────
    
    def match(
        self,
        *,
        ok: Callable[[T], U],
        err: Callable[[E], U],
    ) -> U:
        """Pattern match on Result variants.
        
        Exhaustive case analysis - forces handling both cases.
        
        Example:
            >>> result = Ok(42)
            >>> output = result.match(
            ...     ok=lambda x: f"success: {x}",
            ...     err=lambda e: f"failed: {e}"
            ... )
            >>> assert output == "success: 42"
        """
        if self._is_ok:
            return ok(cast(T, self._value))
        return err(cast(E, self._value))
    
    # ─────────────────────────────────────────────────────────────────
    # Conversion
    # ─────────────────────────────────────────────────────────────────
    
    def to_tuple(self) -> tuple[T | None, E | None]:
        """Convert to (ok_value, err_value) tuple."""
        if self._is_ok:
            return (cast(T, self._value), None)
        return (None, cast(E, self._value))
    
    def flatten(self: Result[Result[T, E], E]) -> Result[T, E]:
        """Flatten nested Result (join in monad terms).
        
        Result[Result[T, E], E] -> Result[T, E]
        """
        if self._is_ok:
            return cast(Result[T, E], self._value)
        return Err(cast(E, self._value))
    
    # ─────────────────────────────────────────────────────────────────
    # Dunder Methods
    # ─────────────────────────────────────────────────────────────────
    
    def __bool__(self) -> bool:
        """Enable truthiness checking (True if Ok)."""
        return self._is_ok
    
    def __repr__(self) -> str:
        """Debug representation."""
        variant = "Ok" if self._is_ok else "Err"
        return f"{variant}({self._value!r})"
    
    def __str__(self) -> str:
        """String representation."""
        return repr(self)
    
    def __eq__(self, other: object) -> bool:
        """Structural equality."""
        if not isinstance(other, Result):
            return NotImplemented
        return self._is_ok == other._is_ok and self._value == other._value
    
    def __hash__(self) -> int:
        """Make Result hashable."""
        return hash((self._is_ok, self._value))
    
    def __iter__(self) -> Iterator[T]:
        """Iterate over Ok value (yields 0 or 1 element).
        
        Enables use in for loops and comprehensions.
        """
        if self._is_ok:
            yield cast(T, self._value)


# ═════════════════════════════════════════════════════════════════════════════
# Constructor Functions
# ═════════════════════════════════════════════════════════════════════════════


def Ok(value: T) -> Result[T, E]:  # noqa: N802
    """Construct Ok variant (success).
    
    Type signature: T -> Result[T, E]
    """
    return Result(value, is_ok=True)


def Err(error: E) -> Result[T, E]:  # noqa: N802
    """Construct Err variant (failure).
    
    Type signature: E -> Result[T, E]
    """
    return Result(error, is_ok=False)


# ═════════════════════════════════════════════════════════════════════════════
# Collection Operations
# ═════════════════════════════════════════════════════════════════════════════


def sequence(results: list[Result[T, E]]) -> Result[list[T], E]:
    """Convert list of Results to Result of list.
    
    Fails fast on first Err, returns Ok with all values if all succeed.
    
    Type signature: [Result[T, E]] -> Result[[T], E]
    
    Example:
        >>> results = [Ok(1), Ok(2), Ok(3)]
        >>> sequence(results).unwrap()
        [1, 2, 3]
        
        >>> results = [Ok(1), Err("fail"), Ok(3)]
        >>> sequence(results).unwrap_err()
        'fail'
    """
    values: list[T] = []
    for result in results:
        if result.is_err():
            return Err(result.unwrap_err())
        values.append(result.unwrap())
    return Ok(values)


def traverse(
    items: list[T],
    f: Callable[[T], Result[U, E]],
) -> Result[list[U], E]:
    """Map function returning Result over list, collect into Result of list.
    
    Combines map and sequence - fails fast on first error.
    
    Type signature: [T] -> (T -> Result[U, E]) -> Result[[U], E]
    
    Example:
        >>> def parse_int(s: str) -> Result[int, str]:
        ...     try:
        ...         return Ok(int(s))
        ...     except ValueError:
        ...         return Err(f"invalid: {s}")
        >>> 
        >>> traverse(["1", "2", "3"], parse_int).unwrap()
        [1, 2, 3]
        
        >>> traverse(["1", "bad", "3"], parse_int).unwrap_err()
        'invalid: bad'
    """
    return sequence([f(item) for item in items])


def collect_results(results: list[Result[T, E]]) -> Result[list[T], list[E]]:
    """Collect all Results, accumulating all errors if any fail.
    
    Unlike sequence, this doesn't fail fast - it collects ALL errors.
    
    Type signature: [Result[T, E]] -> Result[[T], [E]]
    
    Example:
        >>> results = [Ok(1), Err("e1"), Ok(3), Err("e2")]
        >>> collect_results(results).unwrap_err()
        ['e1', 'e2']
    """
    values: list[T] = []
    errors: list[E] = []
    
    for result in results:
        if result.is_ok():
            values.append(result.unwrap())
        else:
            errors.append(result.unwrap_err())
    
    return Ok(values) if not errors else Err(errors)
