"""Integration between Result monad and toolcase's ToolError system.

Provides:
- ToolResult type alias for tool operations
- Conversion between Result and ToolError
- Tool-specific Result helpers
- Backwards compatibility adapters
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from ..errors import ErrorCode, ToolError
from .result import Err, Ok, Result
from .types import ErrorTrace

if TYPE_CHECKING:
    from collections.abc import Callable

# ═════════════════════════════════════════════════════════════════════════════
# Type Aliases for Tools
# ═════════════════════════════════════════════════════════════════════════════

# Tool operations return Result[success_string, ErrorTrace]
ToolResult: TypeAlias = Result[str, ErrorTrace]


# ═════════════════════════════════════════════════════════════════════════════
# ToolError Integration
# ═════════════════════════════════════════════════════════════════════════════


def tool_result(
    tool_name: str,
    message: str,
    *,
    code: ErrorCode = ErrorCode.UNKNOWN,
    recoverable: bool = True,
    details: str | None = None,
) -> ToolResult:
    """Create Err ToolResult from error parameters.
    
    Integrates with existing ToolError system by converting parameters
    to ErrorTrace format.
    
    Args:
        tool_name: Name of tool that encountered error
        message: Human-readable error message
        code: Machine-readable error code
        recoverable: Whether retry/fallback may succeed
        details: Stack trace or additional context
    
    Returns:
        Err variant with ErrorTrace
    
    Example:
        >>> result = tool_result(
        ...     "web_search",
        ...     "API rate limited",
        ...     code=ErrorCode.RATE_LIMITED,
        ...     recoverable=True
        ... )
        >>> assert result.is_err()
    """
    trace = ErrorTrace(
        message=message,
        contexts=[],
        error_code=code.value,
        recoverable=recoverable,
        details=details,
    )
    # Add tool name as first context
    trace = trace.with_operation(f"tool:{tool_name}")
    return Err(trace)


def from_tool_error(error: ToolError) -> ToolResult:
    """Convert ToolError to Result type.
    
    Enables migration from string-based to Result-based error handling.
    
    Args:
        error: Existing ToolError instance
    
    Returns:
        Err variant with ErrorTrace
    """
    trace = ErrorTrace(
        message=error.message,
        contexts=[],
        error_code=error.code.value,
        recoverable=error.recoverable,
        details=error.details,
    )
    trace = trace.with_operation(f"tool:{error.tool_name}")
    return Err(trace)


def to_tool_error(result: ToolResult, tool_name: str) -> ToolError:
    """Convert Err Result to ToolError.
    
    Provides backwards compatibility for code expecting ToolError.
    
    Args:
        result: Result to convert (must be Err)
        tool_name: Tool name for error context
    
    Returns:
        ToolError instance
    
    Raises:
        ValueError: If result is Ok
    """
    if result.is_ok():
        raise ValueError("Cannot convert Ok result to ToolError")
    
    trace = result.unwrap_err()
    
    # Extract error code, defaulting to UNKNOWN
    code = ErrorCode.UNKNOWN
    if trace.error_code:
        try:
            code = ErrorCode(trace.error_code)
        except ValueError:
            pass
    
    # Format contexts into message
    message = trace.message
    if trace.contexts:
        context_str = " <- ".join(str(ctx) for ctx in trace.contexts)
        message = f"{message} [{context_str}]"
    
    return ToolError(
        tool_name=tool_name,
        message=message,
        code=code,
        recoverable=trace.recoverable,
        details=trace.details,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Tool Result Helpers
# ═════════════════════════════════════════════════════════════════════════════


def ok_result(value: str) -> ToolResult:
    """Create Ok ToolResult.
    
    Simple wrapper for clarity in tool implementations.
    """
    return Ok(value)


def try_tool_operation(
    tool_name: str,
    operation: Callable[[], str],
    *,
    context: str = "",
) -> ToolResult:
    """Execute operation, catching exceptions and converting to Result.
    
    Provides automatic exception → ErrorTrace conversion with proper
    error code classification.
    
    Args:
        tool_name: Name of tool executing operation
        operation: Function to execute
        context: Description of what's being attempted
    
    Returns:
        Ok with result string or Err with ErrorTrace
    
    Example:
        >>> def risky_operation() -> str:
        ...     # might raise exception
        ...     return "success"
        >>> 
        >>> result = try_tool_operation(
        ...     "my_tool",
        ...     risky_operation,
        ...     context="fetching data"
        ... )
    """
    try:
        return Ok(operation())
    except Exception as e:
        # Import here to avoid circular dependency
        from ..errors import classify_exception
        
        code = classify_exception(e)
        message = f"{context}: {e}" if context else str(e)
        
        # Get stack trace
        import traceback
        details = traceback.format_exc()
        
        trace = ErrorTrace(
            message=message,
            contexts=[],
            error_code=code.value,
            recoverable=True,
            details=details,
        )
        trace = trace.with_operation(f"tool:{tool_name}")
        
        if context:
            trace = trace.with_operation(context)
        
        return Err(trace)


async def try_tool_operation_async(
    tool_name: str,
    operation: Callable[[], str],
    *,
    context: str = "",
) -> ToolResult:
    """Async version of try_tool_operation.
    
    Executes async operation, catching exceptions and converting to Result.
    """
    try:
        import asyncio
        if asyncio.iscoroutinefunction(operation):
            result = await operation()  # type: ignore[misc]
        else:
            result = await asyncio.to_thread(operation)
        return Ok(result)
    except Exception as e:
        from ..errors import classify_exception
        
        code = classify_exception(e)
        message = f"{context}: {e}" if context else str(e)
        
        import traceback
        details = traceback.format_exc()
        
        trace = ErrorTrace(
            message=message,
            contexts=[],
            error_code=code.value,
            recoverable=True,
            details=details,
        )
        trace = trace.with_operation(f"tool:{tool_name}")
        
        if context:
            trace = trace.with_operation(context)
        
        return Err(trace)


# ═════════════════════════════════════════════════════════════════════════════
# Backwards Compatibility
# ═════════════════════════════════════════════════════════════════════════════


def result_to_string(result: ToolResult, tool_name: str) -> str:
    """Convert ToolResult to string (for backwards compatibility).
    
    Ok results return the success string directly.
    Err results are formatted as ToolError render() output.
    
    This allows gradual migration from string-based to Result-based APIs.
    
    Args:
        result: Result to convert
        tool_name: Tool name for error formatting
    
    Returns:
        Success string or formatted error string
    """
    if result.is_ok():
        return result.unwrap()
    
    # Convert to ToolError and render
    error = to_tool_error(result, tool_name)
    return error.render()


def string_to_result(output: str, tool_name: str) -> ToolResult:
    """Convert string output to ToolResult (for backwards compatibility).
    
    Detects error strings by "**Tool Error" prefix and converts to Err.
    Success strings become Ok.
    
    Args:
        output: Tool output string
        tool_name: Tool name for error context
    
    Returns:
        ToolResult parsed from string
    """
    if output.startswith("**Tool Error"):
        # Parse error string back to trace (lossy but maintains compatibility)
        trace = ErrorTrace(
            message=output,
            error_code=ErrorCode.UNKNOWN.value,
            recoverable=True,
        )
        trace = trace.with_operation(f"tool:{tool_name}")
        return Err(trace)
    
    return Ok(output)


# ═════════════════════════════════════════════════════════════════════════════
# Batch Operations
# ═════════════════════════════════════════════════════════════════════════════


def batch_results(
    results: list[ToolResult],
    *,
    accumulate_errors: bool = False,
) -> ToolResult:
    """Combine multiple ToolResults into one.
    
    Args:
        results: List of ToolResults to combine
        accumulate_errors: If True, collect all errors. If False, fail fast.
    
    Returns:
        Ok with concatenated results, or Err with error(s)
    
    Example:
        >>> r1 = Ok("Result 1")
        >>> r2 = Ok("Result 2")
        >>> batch_results([r1, r2]).unwrap()
        'Result 1\\nResult 2'
    """
    from .result import collect_results, sequence
    
    if accumulate_errors:
        collected = collect_results(results)
        if collected.is_ok():
            return Ok("\n".join(collected.unwrap()))
        
        # Multiple errors - combine messages
        errors = collected.unwrap_err()
        combined_msg = "\n".join(e.message for e in errors)
        combined_trace = ErrorTrace(
            message=f"Multiple errors:\n{combined_msg}",
            contexts=[],
            error_code=errors[0].error_code if errors else None,
            recoverable=any(e.recoverable for e in errors),
        )
        return Err(combined_trace)
    
    # Fail fast
    sequenced = sequence(results)
    if sequenced.is_ok():
        return Ok("\n".join(sequenced.unwrap()))
    
    return Err(sequenced.unwrap_err())
