"""Result-based streaming for typed error propagation mid-stream.

Provides AsyncIterator[Result[str, ErrorTrace]] for graceful error handling
without raising exceptions. Errors propagate as typed values in the stream.

Example:
    >>> async def risky_stream() -> ResultStream:
    ...     yield Ok("chunk1")
    ...     yield Err(trace("network timeout"))
    ...     yield Ok("chunk3")  # Can continue after error
    >>>
    >>> async for result in risky_stream():
    ...     result.match(ok=print, err=lambda e: log.warn(e.message))
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, TypeVar

from toolcase.foundation.errors import Err, ErrorTrace, Ok, Result, trace, trace_from_exc

if TYPE_CHECKING:
    from .stream import StreamChunk

T = TypeVar("T")
U = TypeVar("U")

# ═══════════════════════════════════════════════════════════════════════════════
# Type Aliases
# ═══════════════════════════════════════════════════════════════════════════════

ResultStream: TypeAlias = AsyncIterator[Result[str, ErrorTrace]]
"""Async iterator yielding Result[str, ErrorTrace] for typed error propagation."""

ChunkResult: TypeAlias = Result[str, ErrorTrace]
"""Single chunk result - either Ok(content) or Err(trace)."""


# ═══════════════════════════════════════════════════════════════════════════════
# Factories
# ═══════════════════════════════════════════════════════════════════════════════

def ok_chunk(content: str) -> ChunkResult:
    """Create successful chunk result."""
    return Ok(content)


def err_chunk(message: str, *, code: str | None = None, recoverable: bool = True) -> ChunkResult:
    """Create error chunk result with optional code and recoverability flag."""
    return Err(trace(message, code=code, recoverable=recoverable))


def err_chunk_from_exc(exc: Exception, *, operation: str = "", code: str | None = None) -> ChunkResult:
    """Create error chunk from exception with context."""
    return Err(trace_from_exc(exc, operation=operation, code=code))


# ═══════════════════════════════════════════════════════════════════════════════
# Stream Converters
# ═══════════════════════════════════════════════════════════════════════════════

async def result_stream(source: AsyncIterator[str], *, operation: str = "") -> ResultStream:
    """Wrap raw stream, catching exceptions as Err chunks.
    
    Converts AsyncIterator[str] → AsyncIterator[Result[str, ErrorTrace]].
    Exceptions become Err values instead of propagating.
    
    Args:
        source: Raw string stream
        operation: Context for error traces
    
    Example:
        >>> async for result in result_stream(raw_llm_stream()):
        ...     if result.is_err():
        ...         handle_error(result.error)
        ...     else:
        ...         process(result.value)
    """
    try:
        async for chunk in source:
            yield Ok(chunk)
    except Exception as e:
        yield Err(trace_from_exc(e, operation=operation or "stream"))


async def result_stream_resilient(source: AsyncIterator[str], *, operation: str = "") -> ResultStream:
    """Like result_stream but continues after errors via per-chunk try/catch.
    
    Unlike result_stream which terminates on first exception,
    this variant attempts to continue iteration after errors.
    """
    it = source.__aiter__()
    while True:
        try:
            yield Ok(await it.__anext__())
        except StopAsyncIteration:
            break
        except Exception as e:
            yield Err(trace_from_exc(e, operation=operation or "stream"))


async def unwrap_stream(source: ResultStream, *, raise_on_err: bool = True) -> AsyncIterator[str]:
    """Convert ResultStream back to raw stream.
    
    Args:
        source: Result stream
        raise_on_err: If True, raises RuntimeError on Err. If False, skips Err chunks.
    
    Yields:
        Content from Ok chunks only
    """
    async for result in source:
        if result.is_ok():
            yield result.value
        elif raise_on_err:
            raise RuntimeError(result.error.format())


async def filter_ok(source: ResultStream) -> AsyncIterator[str]:
    """Yield only Ok values, silently dropping Err chunks."""
    async for result in source:
        if result.is_ok():
            yield result.value


async def filter_err(source: ResultStream) -> AsyncIterator[ErrorTrace]:
    """Yield only Err values for error aggregation/logging."""
    async for result in source:
        if result.is_err():
            yield result.error


# ═══════════════════════════════════════════════════════════════════════════════
# Collectors
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class StreamCollectResult:
    """Result of collecting a ResultStream - accumulated content and any errors."""
    content: str
    errors: tuple[ErrorTrace, ...]
    chunk_count: int
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def success(self) -> bool:
        return not self.has_errors
    
    def to_result(self) -> Result[str, tuple[ErrorTrace, ...]]:
        """Convert to Result - Err if any errors occurred."""
        return Err(self.errors) if self.errors else Ok(self.content)


async def collect_result_stream(source: ResultStream) -> StreamCollectResult:
    """Collect all chunks from ResultStream, separating content and errors.
    
    Unlike fail-fast approaches, this collects ALL results before returning,
    allowing partial content recovery even when errors occur.
    
    Example:
        >>> collected = await collect_result_stream(my_stream())
        >>> if collected.has_errors:
        ...     log_errors(collected.errors)
        >>> process_content(collected.content)  # May be partial
    """
    parts: list[str] = []
    errors: list[ErrorTrace] = []
    count = 0
    
    async for result in source:
        count += 1
        (parts if result.is_ok() else errors).append(
            result.value if result.is_ok() else result.error
        )
    
    return StreamCollectResult(content="".join(parts), errors=tuple(errors), chunk_count=count)


async def collect_or_first_error(source: ResultStream) -> Result[str, ErrorTrace]:
    """Collect stream, returning Err on first error (fail-fast).
    
    For resilient collection that accumulates all errors, use collect_result_stream.
    """
    parts: list[str] = []
    async for result in source:
        if result.is_err():
            return result
        parts.append(result.value)
    return Ok("".join(parts))


# ═══════════════════════════════════════════════════════════════════════════════
# Combinators
# ═══════════════════════════════════════════════════════════════════════════════

async def map_ok(source: ResultStream, f: Callable[[str], str]) -> ResultStream:
    """Map function over Ok values, passing Err through unchanged.
    
    Example:
        >>> async for chunk in map_ok(stream, str.upper):
        ...     print(chunk)  # Ok values uppercased, Err unchanged
    """
    async for result in source:
        yield result.map(f) if result.is_ok() else result


async def map_err(source: ResultStream, f: Callable[[ErrorTrace], ErrorTrace]) -> ResultStream:
    """Map function over Err values, passing Ok through unchanged."""
    async for result in source:
        yield result.map_err(f) if result.is_err() else result


async def tap_ok(source: ResultStream, f: Callable[[str], None]) -> ResultStream:
    """Execute side effect on Ok values without transforming."""
    async for result in source:
        if result.is_ok():
            f(result.value)
        yield result


async def tap_err(source: ResultStream, f: Callable[[ErrorTrace], None]) -> ResultStream:
    """Execute side effect on Err values without transforming."""
    async for result in source:
        if result.is_err():
            f(result.error)
        yield result


async def recover(source: ResultStream, handler: Callable[[ErrorTrace], str | None]) -> ResultStream:
    """Attempt to recover from errors by converting Err to Ok.
    
    Args:
        source: Result stream
        handler: (ErrorTrace) -> str | None. Returns replacement content or None to keep Err.
    
    Example:
        >>> async def fallback(err: ErrorTrace) -> str | None:
        ...     return "[redacted]" if err.recoverable else None
        >>> async for chunk in recover(stream, fallback):
        ...     print(chunk)  # Recoverable errors replaced with "[redacted]"
    """
    async for result in source:
        if result.is_err():
            replacement = handler(result.error)
            yield Ok(replacement) if replacement is not None else result
        else:
            yield result
