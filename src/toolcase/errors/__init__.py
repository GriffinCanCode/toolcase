"""Unified error handling for toolcase.

Provides:
- ErrorCode: Standard error codes for tool failures
- ToolError: Structured error responses (Pydantic model)
- ToolException: Exception wrapper for raising errors
- Result/Ok/Err: Monadic error handling with railway-oriented programming
- ErrorTrace/ErrorContext: Error context stacking and provenance tracking
"""

from .errors import ErrorCode, ToolError, ToolException, classify_exception
from .result import Err, Ok, Result, collect_results, sequence, traverse
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
from .types import ErrorContext, ErrorTrace, ResultT

__all__ = [
    # Core error types
    "ErrorCode",
    "ToolError",
    "ToolException",
    "classify_exception",
    # Result monad
    "Result",
    "Ok",
    "Err",
    "ResultT",
    # Tool integration
    "ToolResult",
    "tool_result",
    "ok_result",
    "try_tool_operation",
    "try_tool_operation_async",
    "batch_results",
    "from_tool_error",
    "to_tool_error",
    "result_to_string",
    "string_to_result",
    # Error context
    "ErrorContext",
    "ErrorTrace",
    # Collection operations
    "sequence",
    "traverse",
    "collect_results",
]
