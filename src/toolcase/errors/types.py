"""Type aliases and error context tracking for monadic error handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .result import Result

# ═══════════════════════════════════════════════════════════════════════════════
# Type Aliases
# ═══════════════════════════════════════════════════════════════════════════════

ResultT: TypeAlias = "Result[str, str]"

# ═══════════════════════════════════════════════════════════════════════════════
# Error Context & Provenance
# ═══════════════════════════════════════════════════════════════════════════════

# Empty dict singleton to avoid allocation on each ErrorContext
_EMPTY_META: dict[str, object] = {}


@dataclass(frozen=True, slots=True)
class ErrorContext:
    """Context for error at a call site. Tracks operation, location, metadata."""

    operation: str
    location: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        meta = f" ({', '.join(f'{k}={v}' for k, v in self.metadata.items())})" if self.metadata else ""
        return f"{self.operation}{loc}{meta}"


# Pre-allocated empty tuple for default contexts (single allocation)
_EMPTY_CONTEXTS: tuple[ErrorContext, ...] = ()


@dataclass(frozen=True, slots=True)
class ErrorTrace:
    """Stack of error contexts forming call chain trace for provenance tracking."""

    message: str
    contexts: tuple[ErrorContext, ...] = _EMPTY_CONTEXTS
    error_code: str | None = None
    recoverable: bool = True
    details: str | None = None

    def with_context(self, ctx: ErrorContext) -> ErrorTrace:
        """Add context to trace (returns new trace, preserves immutability)."""
        return ErrorTrace(self.message, (*self.contexts, ctx), self.error_code, self.recoverable, self.details)

    def with_operation(self, operation: str, location: str = "", **metadata: object) -> ErrorTrace:
        """Add context with operation info."""
        return ErrorTrace(
            self.message,
            (*self.contexts, ErrorContext(operation, location, metadata or _EMPTY_META)),
            self.error_code,
            self.recoverable,
            self.details,
        )

    def with_code(self, code: str) -> ErrorTrace:
        """Return new trace with error code set."""
        return ErrorTrace(self.message, self.contexts, code, self.recoverable, self.details)

    def as_unrecoverable(self) -> ErrorTrace:
        """Return new trace marked as unrecoverable."""
        return ErrorTrace(self.message, self.contexts, self.error_code, False, self.details)

    def format(self, *, include_details: bool = False) -> str:
        """Format trace as human-readable string."""
        # Fast path: minimal error
        if not self.error_code and not self.contexts and not self.recoverable:
            return self.message if not (include_details and self.details) else f"{self.message}\nDetails:\n{self.details}"
        
        parts = [self.message]
        if self.error_code:
            parts.append(f" [{self.error_code}]")
        if self.contexts:
            parts.append("\nContext trace:\n" + "\n".join(f"  - {ctx}" for ctx in self.contexts))
        if self.recoverable:
            parts.append("\n(This error may be recoverable)")
        if include_details and self.details:
            parts.append(f"\nDetails:\n{self.details}")
        return "".join(parts)

    __str__ = format


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def context(operation: str, location: str = "", **metadata: object) -> ErrorContext:
    """Create ErrorContext concisely."""
    return ErrorContext(operation, location, metadata or _EMPTY_META)


def trace(message: str, *, code: str | None = None, recoverable: bool = True, details: str | None = None) -> ErrorTrace:
    """Create ErrorTrace concisely."""
    return ErrorTrace(message, _EMPTY_CONTEXTS, code, recoverable, details)


def trace_from_exc(exc: Exception, *, operation: str = "", code: str | None = None) -> ErrorTrace:
    """Create ErrorTrace from exception with optional operation context."""
    import traceback
    t = ErrorTrace(str(exc), _EMPTY_CONTEXTS, code, True, traceback.format_exc())
    return t.with_operation(operation) if operation else t
