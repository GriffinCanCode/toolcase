"""Integration between Result monad and toolcase's ToolError system."""

from __future__ import annotations

import asyncio
import traceback
from collections.abc import Awaitable, Callable
from typing import TypeAlias

from .errors import ErrorCode, ToolError, classify_exception
from .result import Result, _ERR, _OK, collect_results, sequence
from .types import ErrorContext, ErrorTrace, JsonDict, _EMPTY_CONTEXTS
from toolcase.runtime.concurrency import to_thread

# ═══════════════════════════════════════════════════════════════════════════════
# Type Aliases
# ═══════════════════════════════════════════════════════════════════════════════

ToolResult: TypeAlias = Result[str, ErrorTrace]

# Empty metadata dict singleton
_EMPTY_META: JsonDict = {}


def _tool_ctx(tool_name: str) -> tuple[ErrorContext, ...]:
    """Create single-element context tuple for tool."""
    return (ErrorContext(operation=f"tool:{tool_name}", location="", metadata=_EMPTY_META),)


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
    return Result(ErrorTrace(message=message, contexts=_tool_ctx(tool_name), error_code=code.value, recoverable=recoverable, details=details), _ERR)


def from_tool_error(error: ToolError) -> ToolResult:
    """Convert ToolError to Result type."""
    return Result(ErrorTrace(message=error.message, contexts=_tool_ctx(error.tool_name), error_code=error.code.value, recoverable=error.recoverable, details=error.details), _ERR)


def to_tool_error(result: ToolResult, tool_name: str) -> ToolError:
    """Convert Err Result to ToolError. Raises ValueError if Ok."""
    if result._is_ok:
        raise ValueError("Cannot convert Ok result to ToolError")

    tr: ErrorTrace = result._value  # type: ignore[assignment]
    try:
        code = ErrorCode(tr.error_code) if tr.error_code else ErrorCode.UNKNOWN
    except ValueError:
        code = ErrorCode.UNKNOWN

    msg = f"{tr.message} [{' <- '.join(map(str, tr.contexts))}]" if tr.contexts else tr.message
    return ToolError(tool_name=tool_name, message=msg, code=code, recoverable=tr.recoverable, details=tr.details)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Result Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def ok_result(value: str) -> ToolResult:
    """Create Ok ToolResult."""
    return Result(value, _OK)


def _make_error_trace(tool_name: str, e: Exception, ctx: str) -> ErrorTrace:
    """Internal helper to build ErrorTrace from exception."""
    contexts = (ErrorContext(operation=f"tool:{tool_name}", location="", metadata=_EMPTY_META),
                ErrorContext(operation=ctx, location="", metadata=_EMPTY_META)) if ctx else _tool_ctx(tool_name)
    return ErrorTrace(message=f"{ctx}: {e}" if ctx else str(e), contexts=contexts, 
                      error_code=classify_exception(e).value, recoverable=True, details=traceback.format_exc())


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
    try:
        result = await operation() if asyncio.iscoroutinefunction(operation) else await to_thread(operation)  # type: ignore[misc]
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
        return Result(ErrorTrace(message=output, contexts=_tool_ctx(tool_name), error_code=ErrorCode.UNKNOWN.value, recoverable=True, details=None), _ERR)
    return Result(output, _OK)


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Operations
# ═══════════════════════════════════════════════════════════════════════════════


def batch_results(results: list[ToolResult], *, accumulate_errors: bool = False) -> ToolResult:
    """Combine multiple ToolResults. accumulate_errors=True collects all errors, False fails fast."""
    if not accumulate_errors:
        seq = sequence(results)
        return Result("\n".join(seq._value), _OK) if seq._is_ok else seq  # type: ignore[arg-type,return-value]
    
    collected = collect_results(results)
    if collected._is_ok:
        return Result("\n".join(collected._value), _OK)  # type: ignore[arg-type]
    errors: list[ErrorTrace] = collected._value  # type: ignore[assignment]
    return Result(ErrorTrace(
        message=f"Multiple errors:\n{chr(10).join(e.message for e in errors)}",
        contexts=_EMPTY_CONTEXTS,
        error_code=errors[0].error_code if errors else None,
        recoverable=any(e.recoverable for e in errors),
    ), _ERR)
