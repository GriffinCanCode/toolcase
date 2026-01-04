"""Error factory functions for DRY error creation.

Provides concise factory functions to eliminate repeated ErrorTrace construction patterns
across the codebase. All return ToolResult for immediate use in Result chains.

Pattern hierarchy:
- tool_err:       For tool-level errors (tool:name operation)
- component_err:  For component-level errors (gate:name, fallback:name, etc.)
- validation_err: For Pydantic ValidationError conversion
- exc_err:        For Exception conversion with classification
"""

from __future__ import annotations

from pydantic import ValidationError

from .errors import ErrorCode, classify_exception, format_validation_error
from .result import Err, Result, _ERR
from .types import ErrorContext, ErrorTrace, JsonValue, _EMPTY_CONTEXTS, _EMPTY_META

# Type alias kept local to avoid circular imports
type ToolResult = Result[str, ErrorTrace]


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Functions
# ═══════════════════════════════════════════════════════════════════════════════


def tool_err(
    name: str,
    msg: str,
    code: ErrorCode,
    *,
    recoverable: bool = False,
    details: str | None = None,
    **ctx: JsonValue,
) -> ToolResult:
    """Create tool error result.
    
    Standard factory for tool-level errors. Operation is prefixed with 'tool:'.
    
    Args:
        name: Tool name
        msg: Error message
        code: ErrorCode enum value
        recoverable: Whether error is recoverable (default False)
        details: Optional detailed error info
        **ctx: Additional context metadata
    
    Returns:
        Err(ErrorTrace) with operation set to 'tool:{name}'
    
    Example:
        >>> return tool_err("http_request", "Connection refused", ErrorCode.NETWORK_ERROR, recoverable=True)
    """
    trace = ErrorTrace.model_construct(
        message=msg,
        contexts=_EMPTY_CONTEXTS,
        error_code=code.value,
        recoverable=recoverable,
        details=details,
    )
    return Result(trace.with_operation(f"tool:{name}", **ctx), _ERR)


def component_err(
    component: str,
    name: str,
    msg: str,
    code: ErrorCode,
    *,
    recoverable: bool = False,
    details: str | None = None,
    **ctx: JsonValue,
) -> ToolResult:
    """Create component error result.
    
    Factory for component-level errors (gate, fallback, router, etc.).
    Operation is prefixed with '{component}:{name}'.
    
    Args:
        component: Component type (gate, fallback, router, cache, middleware, etc.)
        name: Component instance name
        msg: Error message
        code: ErrorCode enum value
        recoverable: Whether error is recoverable (default False)
        details: Optional detailed error info
        **ctx: Additional context metadata
    
    Returns:
        Err(ErrorTrace) with operation set to '{component}:{name}'
    
    Example:
        >>> return component_err("gate", "auth_gate", "Access denied", ErrorCode.PERMISSION_DENIED, phase="pre")
    """
    trace = ErrorTrace.model_construct(
        message=msg,
        contexts=_EMPTY_CONTEXTS,
        error_code=code.value,
        recoverable=recoverable,
        details=details,
    )
    return Result(trace.with_operation(f"{component}:{name}", **ctx), _ERR)


def validation_err(exc: ValidationError, *, tool_name: str | None = None) -> ToolResult:
    """Create validation error result from Pydantic ValidationError.
    
    Converts Pydantic validation errors to human-readable format via format_validation_error.
    Always unrecoverable since it indicates malformed input.
    
    Args:
        exc: Pydantic ValidationError
        tool_name: Optional tool name for context
    
    Returns:
        Err(ErrorTrace) with INVALID_PARAMS code
    
    Example:
        >>> try:
        ...     params = MyParams(**input_dict)
        >>> except ValidationError as e:
        ...     return validation_err(e, tool_name="my_tool")
    """
    return Result(
        ErrorTrace.model_construct(
            message=format_validation_error(exc, tool_name=tool_name),
            contexts=_EMPTY_CONTEXTS,
            error_code=ErrorCode.INVALID_PARAMS.value,
            recoverable=False,
            details=None,
        ),
        _ERR,
    )


def exc_err(
    exc: Exception,
    operation: str,
    *,
    recoverable: bool = True,
    include_trace: bool = True,
    **ctx: JsonValue,
) -> ToolResult:
    """Create error result from exception with auto-classification.
    
    Classifies exception to ErrorCode and creates trace with optional stack trace.
    Default recoverable=True since most runtime exceptions are transient.
    
    Args:
        exc: Exception to convert
        operation: Operation name (e.g., 'tool:name', 'cache_through:name')
        recoverable: Whether error is recoverable (default True)
        include_trace: Include stack trace in details (default True)
        **ctx: Additional context metadata
    
    Returns:
        Err(ErrorTrace) with classified error code
    
    Example:
        >>> try:
        ...     await client.fetch(url)
        >>> except Exception as e:
        ...     return exc_err(e, f"tool:{tool_name}", url=url)
    """
    import traceback
    
    code = classify_exception(exc)
    trace = ErrorTrace.model_construct(
        message=str(exc),
        contexts=_EMPTY_CONTEXTS,
        error_code=code.value,
        recoverable=recoverable,
        details=traceback.format_exc() if include_trace else None,
    )
    return Result(trace.with_operation(operation, **ctx), _ERR)


# ═══════════════════════════════════════════════════════════════════════════════
# Trace Builders (for non-Result contexts)
# ═══════════════════════════════════════════════════════════════════════════════


def make_trace(
    msg: str,
    code: ErrorCode,
    *,
    recoverable: bool = False,
    details: str | None = None,
) -> ErrorTrace:
    """Create ErrorTrace without wrapping in Result.
    
    Use when you need the trace itself (e.g., for error aggregation).
    
    Args:
        msg: Error message
        code: ErrorCode enum value
        recoverable: Whether error is recoverable
        details: Optional detailed error info
    
    Returns:
        ErrorTrace instance (not wrapped in Result)
    """
    return ErrorTrace.model_construct(
        message=msg,
        contexts=_EMPTY_CONTEXTS,
        error_code=code.value,
        recoverable=recoverable,
        details=details,
    )


def trace_from_exception(
    exc: Exception,
    *,
    operation: str | None = None,
    recoverable: bool = True,
    include_trace: bool = True,
) -> ErrorTrace:
    """Create ErrorTrace from exception.
    
    Similar to exc_err but returns trace directly (not wrapped in Err).
    
    Args:
        exc: Exception to convert
        operation: Optional operation to add to context
        recoverable: Whether error is recoverable
        include_trace: Include stack trace in details
    
    Returns:
        ErrorTrace instance
    """
    import traceback
    
    code = classify_exception(exc)
    trace = ErrorTrace.model_construct(
        message=str(exc),
        contexts=_EMPTY_CONTEXTS,
        error_code=code.value,
        recoverable=recoverable,
        details=traceback.format_exc() if include_trace else None,
    )
    return trace.with_operation(operation) if operation else trace
