"""Timeout middleware for tool execution using structured concurrency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel

from toolcase.foundation.errors import ErrorCode, ErrorTrace, ToolError, ToolException
from toolcase.runtime.middleware import Context, Next
from toolcase.runtime.concurrency import CancelScope

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool


@dataclass(slots=True)
class TimeoutMiddleware:
    """Enforce execution timeout using CancelScope.
    
    Uses structured concurrency CancelScope for clean timeout handling.
    Raises ToolException with TIMEOUT code if exceeded.
    
    Args:
        timeout_seconds: Maximum execution time
        per_tool_overrides: Dict of tool_name -> timeout for specific tools
    
    Example:
        >>> registry.use(TimeoutMiddleware(
        ...     timeout_seconds=30.0,
        ...     per_tool_overrides={"slow_tool": 120.0}
        ... ))
    """
    
    timeout_seconds: float = 30.0
    per_tool_overrides: dict[str, float] = field(default_factory=dict)
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        timeout = self.per_tool_overrides.get(tool.metadata.name, self.timeout_seconds)
        ctx["timeout_configured"] = timeout
        
        async with CancelScope(timeout=timeout) as scope:
            result = await next(tool, params, ctx)
        
        if scope.cancel_called:
            trace = ErrorTrace(
                message=f"Execution timed out after {timeout}s", error_code=ErrorCode.TIMEOUT.value, recoverable=True,
            ).with_operation("middleware:timeout", tool=tool.metadata.name, timeout=timeout)
            ctx["error_trace"] = trace
            raise ToolException(ToolError.create(tool.metadata.name, trace.message, ErrorCode.TIMEOUT, recoverable=True))
        return result
