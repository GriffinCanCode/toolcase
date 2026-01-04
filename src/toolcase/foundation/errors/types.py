"""Type aliases and error context tracking for monadic error handling.

Uses Pydantic models for validation/serialization. Optimized for high-frequency error paths with minimal allocations.
Integrates beartype for O(1) runtime type checking via guard functions.
"""

from __future__ import annotations

from collections.abc import Mapping
from io import StringIO
from typing import TYPE_CHECKING, Annotated, TypeAlias, TypedDict

from beartype import beartype
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, computed_field, field_serializer
from pydantic import JsonValue as PydanticJsonValue

if TYPE_CHECKING:
    from .result import Result

# ═══════════════════════════════════════════════════════════════════════════════
# Type Aliases
# ═══════════════════════════════════════════════════════════════════════════════

ResultT: TypeAlias = "Result[str, str]"

# JSON type aliases
# Use Pydantic's JsonValue for model fields (handles recursion properly at runtime)
# For static typing, we define stricter aliases that preserve type narrowing
JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonArray: TypeAlias = list[PydanticJsonValue]
JsonObject: TypeAlias = dict[str, PydanticJsonValue]
JsonValue: TypeAlias = PydanticJsonValue  # Re-export Pydantic's properly recursive type
JsonDict: TypeAlias = dict[str, PydanticJsonValue]  # Mutable dict for building/modifying
JsonMapping: TypeAlias = Mapping[str, PydanticJsonValue]  # Read-only view for params/returns


# ═══════════════════════════════════════════════════════════════════════════════
# TypedDicts for Well-Known Structures (type-safe serialization)
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitStateDict(TypedDict):
    """Serialized circuit breaker state."""
    state: int
    failures: int
    successes: int
    last_failure: float
    last_state_change: float


class CacheStatsDict(TypedDict):
    """Cache statistics for monitoring."""
    backend: str
    total_entries: int
    expired_entries: int
    active_entries: int
    default_ttl: float
    max_entries: int


class SpanEventDict(TypedDict, total=False):
    """Serialized span event."""
    name: str
    timestamp: float
    attributes: JsonMapping


class SpanToolDict(TypedDict, total=False):
    """Serialized tool context within span."""
    name: str | None
    category: str | None
    params: JsonMapping | None
    result_preview: str | None


class ErrorTraceSerialized(TypedDict, total=False):
    """Serialized ErrorTrace for span export - stricter than JsonDict."""
    message: str
    code: str | None
    recoverable: bool
    contexts: list[str]
    details: str | None


class SpanDict(TypedDict, total=False):
    """Serialized span for export."""
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    kind: str
    start_time: float
    end_time: float | None
    duration_ms: float | None
    status: str
    error: str | None
    attributes: JsonMapping
    events: list[SpanEventDict]
    tool: SpanToolDict | None
    error_trace: ErrorTraceSerialized | None


class StreamChunkDict(TypedDict, total=False):
    """Serialized stream chunk."""
    content: str
    index: int
    timestamp: float
    metadata: JsonMapping


class StreamEventDict(TypedDict, total=False):
    """Serialized stream event."""
    kind: str
    tool: str
    timestamp: float
    data: StreamChunkDict
    accumulated: str
    error: str


class RequestRecordDict(TypedDict, total=False):
    """Recorded HTTP request in MockAPI."""
    method: str
    endpoint: str
    params: JsonMapping
    data: JsonMapping

# ═══════════════════════════════════════════════════════════════════════════════
# Runtime Type Checking (beartype)
# ═══════════════════════════════════════════════════════════════════════════════

# Re-export beartype decorator for function-level type checking
typechecked = beartype

# Import beartype exception base for unified error handling
from beartype.roar import BeartypeException


class TypeViolationError(TypeError):
    """Runtime type violation error for JSON type guards."""
    pass


# TypeViolation: tuple for except clauses catching both our guards and beartype violations
# Usage: except TypeViolation as e:
TypeViolation = (BeartypeException, TypeViolationError)

# JSON primitives tuple for O(1) isinstance checks
_JSON_PRIMITIVES = (str, int, float, bool, type(None))


def _is_json_recursive(value: object, _seen: set[int] | None = None) -> bool:
    """Deep check if value is valid JSON (handles cycles)."""
    if _seen is None:
        _seen = set()
    vid = id(value)
    if vid in _seen:
        return False  # Circular reference not allowed in JSON
    if isinstance(value, _JSON_PRIMITIVES):
        return True
    if isinstance(value, list):
        _seen.add(vid)
        return all(_is_json_recursive(v, _seen) for v in value)
    if isinstance(value, dict):
        _seen.add(vid)
        return all(isinstance(k, str) and _is_json_recursive(v, _seen) for k, v in value.items())
    return False


def is_json_value(value: object) -> bool:
    """Check if value conforms to JsonValue at runtime. Handles nested structures and cycles."""
    return _is_json_recursive(value)


def is_json_dict(value: object) -> bool:
    """Check if value conforms to JsonDict at runtime (dict with string keys and JSON values)."""
    return isinstance(value, dict) and all(isinstance(k, str) for k in value) and _is_json_recursive(value)


def as_json_value(value: object) -> JsonValue:
    """Narrow value to JsonValue, raising TypeViolation if invalid."""
    if not is_json_value(value):
        raise TypeViolationError(f"Expected JsonValue, got {type(value).__name__}: {value!r}")
    return value  # type: ignore[return-value]


def as_json_dict(value: object) -> JsonDict:
    """Narrow value to JsonDict, raising TypeViolation if invalid."""
    if not is_json_dict(value):
        raise TypeViolationError(f"Expected JsonDict, got {type(value).__name__}: {value!r}")
    return value  # type: ignore[return-value]


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
    def _serialize_contexts(self, v: tuple[ErrorContext, ...]) -> list[JsonMapping]:
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
        """Add context to trace (returns new trace). Uses model_construct for performance."""
        return ErrorTrace.model_construct(message=self.message, contexts=(*self.contexts, ctx),
                                          error_code=self.error_code, recoverable=self.recoverable, details=self.details)

    def with_operation(self, operation: str, location: str = "", **metadata: JsonValue) -> "ErrorTrace":
        """Add context with operation info."""
        ctx = ErrorContext.model_construct(operation=operation, location=location, metadata=metadata or _EMPTY_META)
        return ErrorTrace.model_construct(message=self.message, contexts=(*self.contexts, ctx),
                                          error_code=self.error_code, recoverable=self.recoverable, details=self.details)

    def with_code(self, code: str) -> "ErrorTrace":
        """Return new trace with error code set."""
        return ErrorTrace.model_construct(message=self.message, contexts=self.contexts, error_code=code,
                                          recoverable=self.recoverable, details=self.details)

    def as_unrecoverable(self) -> "ErrorTrace":
        """Return new trace marked as unrecoverable."""
        return ErrorTrace.model_construct(message=self.message, contexts=self.contexts, error_code=self.error_code,
                                          recoverable=False, details=self.details)

    def format(self, *, include_details: bool = False) -> str:
        """Format trace as human-readable string. Uses StringIO for traces with >10 contexts."""
        # Fast path: minimal error
        if not self.error_code and not self.contexts and not self.recoverable:
            return f"{self.message}\nDetails:\n{self.details}" if include_details and self.details else self.message
        
        # Use StringIO for large traces
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
    return ErrorContext.model_construct(operation=operation, location=location, metadata=metadata or _EMPTY_META)


def trace(message: str, *, code: str | None = None, recoverable: bool = True, details: str | None = None) -> ErrorTrace:
    """Create ErrorTrace concisely (bypasses validation for performance)."""
    return ErrorTrace.model_construct(message=message, contexts=_EMPTY_CONTEXTS, error_code=code,
                                      recoverable=recoverable, details=details)


def trace_from_exc(exc: Exception, *, operation: str = "", code: str | None = None) -> ErrorTrace:
    """Create ErrorTrace from exception with optional operation context."""
    import traceback
    t = ErrorTrace.model_construct(message=str(exc), contexts=_EMPTY_CONTEXTS, error_code=code,
                                   recoverable=True, details=traceback.format_exc())
    return t.with_operation(operation) if operation else t


def validate_context(data: JsonMapping) -> ErrorContext:
    """Validate dict as ErrorContext (use when validation is needed)."""
    return _ErrorContextAdapter.validate_python(data)


def validate_trace(data: JsonMapping) -> ErrorTrace:
    """Validate dict as ErrorTrace (use when validation is needed)."""
    return _ErrorTraceAdapter.validate_python(data)
