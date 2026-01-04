"""Unified error handling for toolcase.

- ErrorCode: Standard error codes for tool failures
- ToolError/ToolException: Structured errors and exceptions
- Result/Ok/Err: Monadic error handling with railway-oriented programming
- ErrorTrace/ErrorContext: Error context stacking and provenance tracking
- TypeAdapter utilities: validate_context, validate_trace for fast validation
- Runtime type checking: typechecked decorator, is_json_value/is_json_dict guards
"""

from .errors import (
    ErrorCode,
    ToolError,
    ToolException,
    RetryableToolException,
    AuthToolException,
    ValidationToolException,
    classify_exception,
    format_validation_error,
)
from .factory import (
    component_err,
    exc_err,
    make_trace,
    tool_err,
    trace_from_exception,
    validation_err,
)
from .result import Err, Ok, Result, collect_results, sequence, traverse, try_fn
from .tool import (
    ToolResult,
    batch_results,
    from_tool_error,
    ok_result,
    result_to_string,
    string_to_result,
    to_tool_error,
    tool_result,
    try_tool_operation,
    try_tool_operation_async,
)
from .types import (
    CacheStatsDict,
    CircuitStateDict,
    CoalesceStatsDict,
    ErrorContext,
    ErrorTrace,
    ErrorTraceSerialized,
    JsonArray,
    JsonDict,
    JsonMapping,
    JsonObject,
    JsonPrimitive,
    JsonValue,
    RequestRecordDict,
    ResultT,
    SpanDict,
    SpanEventDict,
    SpanToolDict,
    StreamChunkDict,
    StreamEventDict,
    TypeViolation,
    TypeViolationError,
    as_json_dict,
    as_json_value,
    context,
    is_json_dict,
    is_json_value,
    trace,
    trace_from_exc,
    typechecked,
    validate_context,
    validate_trace,
)

__all__ = [
    # Core errors
    "ErrorCode", "ToolError", "ToolException", "classify_exception", "format_validation_error",
    # Exception subclasses for specific error handling
    "RetryableToolException", "AuthToolException", "ValidationToolException",
    # Result monad
    "Result", "Ok", "Err", "ResultT", "try_fn",
    # Tool integration
    "ToolResult", "tool_result", "ok_result", "try_tool_operation", "try_tool_operation_async",
    "batch_results", "from_tool_error", "to_tool_error", "result_to_string", "string_to_result",
    # Error factories (DRY error creation)
    "tool_err", "component_err", "validation_err", "exc_err", "make_trace", "trace_from_exception",
    # Error context
    "ErrorContext", "ErrorTrace", "context", "trace", "trace_from_exc",
    # TypeAdapter validation utilities
    "validate_context", "validate_trace",
    # Collection ops
    "sequence", "traverse", "collect_results",
    # JSON types
    "JsonPrimitive", "JsonValue", "JsonArray", "JsonObject", "JsonDict", "JsonMapping",
    # TypedDicts for well-known structures
    "CircuitStateDict", "CacheStatsDict", "CoalesceStatsDict", "SpanDict", "SpanEventDict", "SpanToolDict",
    "StreamChunkDict", "StreamEventDict", "RequestRecordDict", "ErrorTraceSerialized",
    # Runtime type checking (beartype)
    "typechecked", "TypeViolation", "TypeViolationError", "is_json_value", "is_json_dict", "as_json_value", "as_json_dict",
]
