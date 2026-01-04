"""Integration between Result monad and toolcase's ToolError system."""

from __future__ import annotations

import traceback
from collections.abc import Awaitable, Callable
from typing import TypeAlias

from .errors import ErrorCode, ToolError, classify_exception
from .result import Err, Ok, Result, _ERR, _OK
from .types import ErrorTrace

# ═══════════════════════════════════════════════════════════════════════════════
# Type Aliases
# ═══════════════════════════════════════════════════════════════════════════════

ToolResult: TypeAlias = Result[str, ErrorTrace]


# ═══════════════════════════════════════════════════════════════════════════════
# ToolError Integration
# ═══════════════════════════════════════════════════════════════════════════════


def tool_result(
    tool_name: str,
    message: str,
    *,
    code: ErrorCode = ErrorCode.UNKNOWN,
    recoverable: bool = True,
    details: str | None = None,
) -> ToolResult:
    """Create Err ToolResult from error parameters."""
    return Err(
        ErrorTrace(message, (), code.value, recoverable, details).with_operation(f"tool:{tool_name}")
    )


def from_tool_error(error: ToolError) -> ToolResult:
    """Convert ToolError to Result type."""
    return Err(
        ErrorTrace(error.message, (), error.code.value, error.recoverable, error.details)
        .with_operation(f"tool:{error.tool_name}")
    )


def to_tool_error(result: ToolResult, tool_name: str) -> ToolError:
    """Convert Err Result to ToolError. Raises ValueError if Ok."""
    if result._is_ok:
        raise ValueError("Cannot convert Ok result to ToolError")
    
    trace: ErrorTrace = result._value  # type: ignore[assignment]
    
    # Extract error code
    code = ErrorCode.UNKNOWN
    if trace.error_code:
        try:
            code = ErrorCode(trace.error_code)
        except ValueError:
            pass
    
    # Format contexts into message
    message = f"{trace.message} [{' <- '.join(map(str, trace.contexts))}]" if trace.contexts else trace.message
    
    return ToolError(tool_name=tool_name, message=message, code=code, recoverable=trace.recoverable, details=trace.details)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Result Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def ok_result(value: str) -> ToolResult:
    """Create Ok ToolResult."""
    return Result(value, _OK)


def _make_error_trace(tool_name: str, e: Exception, context: str) -> ErrorTrace:
    """Internal helper to build ErrorTrace from exception."""
    message = f"{context}: {e}" if context else str(e)
    trace = ErrorTrace(message, (), classify_exception(e).value, True, traceback.format_exc())
    trace = trace.with_operation(f"tool:{tool_name}")
    return trace.with_operation(context) if context else trace


def try_tool_operation(tool_name: str, operation: Callable[[], str], *, context: str = "") -> ToolResult:
    """Execute operation, catching exceptions and converting to Result."""
    try:
        return Result(operation(), _OK)
    except Exception as e:
        return Result(_make_error_trace(tool_name, e, context), _ERR)


async def try_tool_operation_async(
    tool_name: str,
    operation: Callable[[], str] | Callable[[], Awaitable[str]],
    *,
    context: str = "",
) -> ToolResult:
    """Async version - executes sync or async operation, converts exceptions to Result."""
    import asyncio
    try:
        result = await operation() if asyncio.iscoroutinefunction(operation) else await asyncio.to_thread(operation)  # type: ignore[misc]
        return Result(result, _OK)
    except Exception as e:
        return Result(_make_error_trace(tool_name, e, context), _ERR)


# ═══════════════════════════════════════════════════════════════════════════════
# Backwards Compatibility
# ═══════════════════════════════════════════════════════════════════════════════


def result_to_string(result: ToolResult, tool_name: str) -> str:
    """Convert ToolResult to string. Ok returns value, Err renders as ToolError."""
    return result._value if result._is_ok else to_tool_error(result, tool_name).render()  # type: ignore[return-value]


def string_to_result(output: str, tool_name: str) -> ToolResult:
    """Parse string to ToolResult. Detects error strings by '**Tool Error' prefix."""
    if output.startswith("**Tool Error"):
        return Result(
            ErrorTrace(output, (), ErrorCode.UNKNOWN.value, True, None).with_operation(f"tool:{tool_name}"),
            _ERR,
        )
    return Result(output, _OK)


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Operations
# ═══════════════════════════════════════════════════════════════════════════════


def batch_results(results: list[ToolResult], *, accumulate_errors: bool = False) -> ToolResult:
    """Combine multiple ToolResults. accumulate_errors=True collects all errors, False fails fast."""
    from .result import collect_results, sequence
    
    if accumulate_errors:
        collected = collect_results(results)
        if collected._is_ok:
            return Result("\n".join(collected._value), _OK)  # type: ignore[arg-type]
        errors: list[ErrorTrace] = collected._value  # type: ignore[assignment]
        return Result(
            ErrorTrace(
                f"Multiple errors:\n{chr(10).join(e.message for e in errors)}",
                (),
                errors[0].error_code if errors else None,
                any(e.recoverable for e in errors),
            ),
            _ERR,
        )
    
    # Fail fast
    sequenced = sequence(results)
    return Result("\n".join(sequenced._value), _OK) if sequenced._is_ok else Result(sequenced._value, _ERR)  # type: ignore[arg-type]
