"""Timeout middleware for tool execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ...errors import ErrorCode, ErrorTrace, ToolError, ToolException
from ..middleware import Context, Next

if TYPE_CHECKING:
    from ...core import BaseTool


@dataclass(slots=True)
class TimeoutMiddleware:
    """Enforce execution timeout.
    
    Wraps execution in asyncio.wait_for. Raises ToolException with
    TIMEOUT code if exceeded. Stores ErrorTrace in context for observability.
    
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
        try:
            return await asyncio.wait_for(next(tool, params, ctx), timeout=timeout)
        except asyncio.TimeoutError:
            trace = ErrorTrace(
                message=f"Execution timed out after {timeout}s",
                error_code=ErrorCode.TIMEOUT.value,
                recoverable=True,
            ).with_operation(f"middleware:timeout", tool=tool.metadata.name, timeout=timeout)
            ctx["error_trace"] = trace
            raise ToolException(ToolError.create(
                tool.metadata.name,
                trace.message,
                ErrorCode.TIMEOUT,
                recoverable=True,
            )) from None
