"""Type aliases and error context tracking for monadic error handling.

Uses Pydantic models for validation and serialization where performance allows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer

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


class ErrorContext(BaseModel):
    """Context for error at a call site. Tracks operation, location, metadata.
    
    Attributes:
        operation: Name of the operation (e.g., "tool:web_search", "http:request")
        location: Optional location info (e.g., file:line)
        metadata: Additional context data
    """
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "title": "Error Context",
            "examples": [{"operation": "tool:web_search", "location": "", "metadata": {"attempt": 1}}],
        },
    )

    operation: Annotated[str, Field(min_length=1)]
    location: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)

    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        meta = f" ({', '.join(f'{k}={v}' for k, v in self.metadata.items())})" if self.metadata else ""
        return f"{self.operation}{loc}{meta}"
    
    def __hash__(self) -> int:
        """Enable hashing for use in sets/tuples."""
        return hash((self.operation, self.location, tuple(sorted(self.metadata.items()))))


# Pre-allocated empty tuple for default contexts (single allocation)
_EMPTY_CONTEXTS: tuple[ErrorContext, ...] = ()


class ErrorTrace(BaseModel):
    """Stack of error contexts forming call chain trace for provenance tracking.
    
    Provides immutable error propagation with rich context for debugging.
    
    Attributes:
        message: Human-readable error message
        contexts: Stack of ErrorContext forming the call chain
        error_code: Machine-readable error code
        recoverable: Whether retry might succeed
        details: Optional detailed info (e.g., stack trace)
    """
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
        json_schema_extra={
            "title": "Error Trace",
            "description": "Full error context with provenance tracking",
        },
    )

    message: Annotated[str, Field(min_length=1)]
    contexts: tuple[ErrorContext, ...] = _EMPTY_CONTEXTS
    error_code: str | None = None
    recoverable: bool = True
    details: str | None = None
    
    @field_serializer("contexts")
    def _serialize_contexts(self, v: tuple[ErrorContext, ...]) -> list[dict[str, object]]:
        """Serialize tuple of contexts to list of dicts."""
        return [ctx.model_dump() for ctx in v]
    
    @computed_field
    @property
    def depth(self) -> int:
        """Number of contexts in the trace."""
        return len(self.contexts)
    
    @computed_field
    @property
    def root_operation(self) -> str | None:
        """First operation in the trace (origin)."""
        return self.contexts[0].operation if self.contexts else None

    def with_context(self, ctx: ErrorContext) -> "ErrorTrace":
        """Add context to trace (returns new trace, preserves immutability)."""
        return ErrorTrace(
            message=self.message,
            contexts=(*self.contexts, ctx),
            error_code=self.error_code,
            recoverable=self.recoverable,
            details=self.details,
        )

    def with_operation(self, operation: str, location: str = "", **metadata: object) -> "ErrorTrace":
        """Add context with operation info."""
        return ErrorTrace(
            message=self.message,
            contexts=(*self.contexts, ErrorContext(operation=operation, location=location, metadata=metadata or _EMPTY_META)),
            error_code=self.error_code,
            recoverable=self.recoverable,
            details=self.details,
        )

    def with_code(self, code: str) -> "ErrorTrace":
        """Return new trace with error code set."""
        return ErrorTrace(
            message=self.message,
            contexts=self.contexts,
            error_code=code,
            recoverable=self.recoverable,
            details=self.details,
        )

    def as_unrecoverable(self) -> "ErrorTrace":
        """Return new trace marked as unrecoverable."""
        return ErrorTrace(
            message=self.message,
            contexts=self.contexts,
            error_code=self.error_code,
            recoverable=False,
            details=self.details,
        )

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
    return ErrorContext(operation=operation, location=location, metadata=metadata or _EMPTY_META)


def trace(message: str, *, code: str | None = None, recoverable: bool = True, details: str | None = None) -> ErrorTrace:
    """Create ErrorTrace concisely."""
    return ErrorTrace(message=message, contexts=_EMPTY_CONTEXTS, error_code=code, recoverable=recoverable, details=details)


def trace_from_exc(exc: Exception, *, operation: str = "", code: str | None = None) -> ErrorTrace:
    """Create ErrorTrace from exception with optional operation context."""
    import traceback
    t = ErrorTrace(message=str(exc), contexts=_EMPTY_CONTEXTS, error_code=code, recoverable=True, details=traceback.format_exc())
    return t.with_operation(operation) if operation else t
