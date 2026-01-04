"""Type aliases and error context tracking for monadic error handling.

Uses Pydantic models for validation/serialization. Optimized for high-frequency error paths with minimal allocations.
"""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING, Annotated, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, computed_field, field_serializer

if TYPE_CHECKING:
    from .result import Result

# ═══════════════════════════════════════════════════════════════════════════════
# Type Aliases
# ═══════════════════════════════════════════════════════════════════════════════

ResultT: TypeAlias = "Result[str, str]"

# JSON type aliases - using Any for recursive types to avoid Pydantic resolution issues
from typing import Any
JsonPrimitive = Union[str, int, float, bool, None]
JsonValue = Union[JsonPrimitive, list[Any], dict[str, Any]]  # Any for recursive slots
JsonDict = dict[str, Any]

# Threshold for using StringIO in format() (improves performance for large traces)
_FORMAT_STRINGIO_THRESHOLD = 10

# ═══════════════════════════════════════════════════════════════════════════════
# Error Context & Provenance
# ═══════════════════════════════════════════════════════════════════════════════

# Empty dict singleton to avoid allocation on each ErrorContext
_EMPTY_META: JsonDict = {}


class ErrorContext(BaseModel):
    """Context for error at a call site. Tracks operation, location, metadata. Pydantic frozen=True for immutability."""
    
    model_config = ConfigDict(
        frozen=True, str_strip_whitespace=True, extra="forbid",
        revalidate_instances="never", populate_by_name=True,
        json_schema_extra={"title": "Error Context", "examples": [{"operation": "tool:web_search", "location": "", "metadata": {"attempt": 1}}]},
    )

    operation: Annotated[str, Field(min_length=1)]
    location: str = Field(default="", repr=False)
    metadata: JsonDict = Field(default_factory=dict, repr=False)
    
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
    """Stack of error contexts forming call chain trace. Immutable error propagation with rich context for debugging."""
    
    model_config = ConfigDict(
        frozen=True, str_strip_whitespace=True, validate_default=True, extra="forbid",
        revalidate_instances="never",
        json_schema_extra={"title": "Error Trace", "description": "Full error context with provenance tracking"},
    )
    
    message: Annotated[str, Field(min_length=1)]
    contexts: tuple[ErrorContext, ...] = _EMPTY_CONTEXTS
    error_code: str | None = Field(default=None, repr=True)
    recoverable: bool = True
    details: str | None = Field(default=None, repr=False)  # Often verbose, hide from repr
    
    @field_serializer("contexts")
    def _serialize_contexts(self, v: tuple[ErrorContext, ...]) -> list[JsonDict]:
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
    
    def __hash__(self) -> int:
        """Hash for frozen model."""
        return hash((self.message, self.error_code, self.recoverable))

    def with_context(self, ctx: ErrorContext) -> "ErrorTrace":
        """Add context to trace (returns new trace, preserves immutability).
        
        Uses model_construct for performance when building from validated data.
        """
        return ErrorTrace.model_construct(
            message=self.message,
            contexts=(*self.contexts, ctx),
            error_code=self.error_code,
            recoverable=self.recoverable,
            details=self.details,
        )

    def with_operation(self, operation: str, location: str = "", **metadata: JsonValue) -> "ErrorTrace":
        """Add context with operation info."""
        ctx = ErrorContext.model_construct(
            operation=operation,
            location=location,
            metadata=metadata or _EMPTY_META,
        )
        return ErrorTrace.model_construct(
            message=self.message,
            contexts=(*self.contexts, ctx),
            error_code=self.error_code,
            recoverable=self.recoverable,
            details=self.details,
        )

    def with_code(self, code: str) -> "ErrorTrace":
        """Return new trace with error code set."""
        return ErrorTrace.model_construct(
            message=self.message,
            contexts=self.contexts,
            error_code=code,
            recoverable=self.recoverable,
            details=self.details,
        )

    def as_unrecoverable(self) -> "ErrorTrace":
        """Return new trace marked as unrecoverable."""
        return ErrorTrace.model_construct(
            message=self.message,
            contexts=self.contexts,
            error_code=self.error_code,
            recoverable=False,
            details=self.details,
        )

    def format(self, *, include_details: bool = False) -> str:
        """Format trace as human-readable string.
        
        Uses StringIO for traces with many contexts (>10) for better performance
        when building large strings.
        """
        # Fast path: minimal error
        if not self.error_code and not self.contexts and not self.recoverable:
            return self.message if not (include_details and self.details) else f"{self.message}\nDetails:\n{self.details}"
        
        # Use StringIO for large traces (many contexts)
        if len(self.contexts) > _FORMAT_STRINGIO_THRESHOLD:
            return self._format_large(include_details=include_details)
        
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
    
    def _format_large(self, *, include_details: bool = False) -> str:
        """Format large traces using StringIO for efficiency."""
        buf = StringIO()
        buf.write(self.message)
        if self.error_code:
            buf.write(f" [{self.error_code}]")
        if self.contexts:
            buf.write("\nContext trace:\n")
            for ctx in self.contexts:
                buf.write(f"  - {ctx}\n")
        if self.recoverable:
            buf.write("(This error may be recoverable)")
        if include_details and self.details:
            buf.write(f"\nDetails:\n{self.details}")
        return buf.getvalue()

    __str__ = format


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers (use model_construct for hot paths)
# ═══════════════════════════════════════════════════════════════════════════════

# TypeAdapters for validation when needed (cached at module level)
_ErrorContextAdapter: TypeAdapter[ErrorContext] = TypeAdapter(ErrorContext)
_ErrorTraceAdapter: TypeAdapter[ErrorTrace] = TypeAdapter(ErrorTrace)


def context(operation: str, location: str = "", **metadata: JsonValue) -> ErrorContext:
    """Create ErrorContext concisely (bypasses validation for performance)."""
    return ErrorContext.model_construct(
        operation=operation,
        location=location,
        metadata=metadata or _EMPTY_META,
    )


def trace(message: str, *, code: str | None = None, recoverable: bool = True, details: str | None = None) -> ErrorTrace:
    """Create ErrorTrace concisely (bypasses validation for performance)."""
    return ErrorTrace.model_construct(
        message=message,
        contexts=_EMPTY_CONTEXTS,
        error_code=code,
        recoverable=recoverable,
        details=details,
    )


def trace_from_exc(exc: Exception, *, operation: str = "", code: str | None = None) -> ErrorTrace:
    """Create ErrorTrace from exception with optional operation context."""
    import traceback
    t = ErrorTrace.model_construct(
        message=str(exc),
        contexts=_EMPTY_CONTEXTS,
        error_code=code,
        recoverable=True,
        details=traceback.format_exc(),
    )
    return t.with_operation(operation) if operation else t


def validate_context(data: JsonDict) -> ErrorContext:
    """Validate dict as ErrorContext (use when validation is needed)."""
    return _ErrorContextAdapter.validate_python(data)


def validate_trace(data: JsonDict) -> ErrorTrace:
    """Validate dict as ErrorTrace (use when validation is needed)."""
    return _ErrorTraceAdapter.validate_python(data)
