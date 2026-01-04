"""Fallback primitive for graceful degradation chains.

Tries tools in order until one succeeds. Useful for:
- Provider redundancy (primary → backup)
- Graceful degradation (expensive → cheap)
- Timeout-based fallback (slow → fast)
- Error-specific fallback (rate limit → alternate)

Example:
    >>> resilient = fallback(
    ...     PrimaryAPI(),
    ...     BackupAPI(),
    ...     LocalCache(),
    ...     timeout=10.0,
    ... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError

from toolcase.foundation.core.base import BaseTool, ToolMetadata
from toolcase.foundation.errors import Err, ErrorCode, ErrorTrace, JsonDict, ToolResult, format_validation_error
from toolcase.runtime.concurrency import CancelScope

if TYPE_CHECKING:
    pass


# Default errors that trigger fallback (transient/recoverable)
DEFAULT_FALLBACK_CODES: frozenset[ErrorCode] = frozenset({
    ErrorCode.RATE_LIMITED,
    ErrorCode.TIMEOUT,
    ErrorCode.NETWORK_ERROR,
    ErrorCode.EXTERNAL_SERVICE_ERROR,
    ErrorCode.UNKNOWN,  # Generic failures
})


class FallbackParams(BaseModel):
    """Parameters for fallback execution."""
    
    input: JsonDict = Field(
        default_factory=dict,
        description="Input parameters passed to each fallback tool",
    )


# Rebuild model to resolve recursive JsonValue type
FallbackParams.model_rebuild()


class FallbackTool(BaseTool[FallbackParams]):
    """Fallback chain with timeout and error filtering.
    
    Tries tools sequentially until one succeeds. Can filter which
    error types trigger fallback vs immediate failure.
    
    Example:
        >>> chain = FallbackTool(
        ...     tools=[PrimaryTool(), BackupTool(), CacheTool()],
        ...     timeout=5.0,  # Per-tool timeout
        ...     fallback_on={ErrorCode.RATE_LIMITED, ErrorCode.TIMEOUT},
        ... )
    """
    
    __slots__ = ("_tools", "_timeout", "_fallback_codes", "_meta")
    
    params_schema = FallbackParams
    cache_enabled = False
    
    def __init__(
        self,
        tools: list[BaseTool[BaseModel]],
        *,
        timeout: float = 30.0,
        fallback_on: frozenset[ErrorCode] | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        if not tools:
            raise ValueError("Fallback requires at least one tool")
        
        self._tools = tools
        self._timeout = timeout
        self._fallback_codes = fallback_on or DEFAULT_FALLBACK_CODES
        
        tool_names = [t.metadata.name for t in tools]
        derived_name = name or f"fallback_{'_'.join(tool_names[:3])}"
        derived_desc = description or f"Fallback chain: {' → '.join(tool_names)}"
        
        self._meta = ToolMetadata(
            name=derived_name,
            description=derived_desc,
            category="agents",
            streaming=any(t.metadata.streaming for t in tools),
        )
    
    @property
    def metadata(self) -> ToolMetadata:
        return self._meta
    
    @property
    def tools(self) -> list[BaseTool[BaseModel]]:
        return self._tools
    
    def _should_fallback(self, trace: ErrorTrace) -> bool:
        """Determine if error should trigger fallback to next tool."""
        if not trace.error_code:
            return True  # Unknown errors fallback by default
        try:
            code = ErrorCode(trace.error_code)
            return code in self._fallback_codes
        except ValueError:
            return True  # Unknown codes fallback
    
    def _run(self, params: FallbackParams) -> str:
        return self._run_async_sync(self._async_run(params))
    
    async def _async_run(self, params: FallbackParams) -> str:
        result = await self._async_run_result(params)
        if result.is_ok():
            return result.unwrap()
        return result.unwrap_err().message
    
    async def _async_run_result(self, params: FallbackParams) -> ToolResult:
        """Execute fallback chain with Result-based handling using structured concurrency."""
        last_error: ErrorTrace | None = None
        errors: list[ErrorTrace] = []
        
        for i, tool in enumerate(self._tools):
            # Build params for this tool
            try:
                tool_params = tool.params_schema(**params.input)
            except ValidationError as e:
                trace = ErrorTrace(
                    message=format_validation_error(e, tool_name=tool.metadata.name),
                    error_code=ErrorCode.INVALID_PARAMS.value,
                    recoverable=False,
                )
                errors.append(trace)
                continue  # Try next tool
            
            # Execute with timeout using CancelScope
            async with CancelScope(timeout=self._timeout) as scope:
                result = await tool.arun_result(tool_params)
            
            if scope.cancel_called:
                trace = ErrorTrace(
                    message=f"Tool {tool.metadata.name} timed out after {self._timeout}s",
                    error_code=ErrorCode.TIMEOUT.value,
                    recoverable=True,
                )
                errors.append(trace)
                last_error = trace
                continue  # Timeout triggers fallback
            
            # Success - return immediately
            if result.is_ok():
                return result
            
            # Error - check if we should fallback
            trace = result.unwrap_err()
            errors.append(trace)
            last_error = trace
            
            if not self._should_fallback(trace):
                # Non-fallback error - stop chain
                return result.map_err(
                    lambda e: e.with_operation(f"fallback:{self._meta.name}", tool=tool.metadata.name)
                )
            
            # Continue to next fallback
        
        # All tools failed
        final_error = ErrorTrace(
            message=f"All {len(self._tools)} fallback tools failed",
            error_code=last_error.error_code if last_error else ErrorCode.UNKNOWN.value,
            recoverable=False,
            details="\n".join(f"- {e.message}" for e in errors),
        ).with_operation(f"fallback:{self._meta.name}")
        
        return Err(final_error)


def fallback(
    *tools: BaseTool[BaseModel],
    timeout: float = 30.0,
    fallback_on: frozenset[ErrorCode] | None = None,
    name: str | None = None,
    description: str | None = None,
) -> FallbackTool:
    """Create a fallback chain from tools.
    
    Tries each tool in order until one succeeds. Timeout and specific
    error codes trigger fallback to next tool.
    
    Args:
        *tools: Tools in fallback order (first = primary)
        timeout: Per-tool timeout in seconds
        fallback_on: Error codes that trigger fallback (default: transient errors)
        name: Optional fallback chain name
        description: Optional description
    
    Returns:
        FallbackTool instance
    
    Example:
        >>> # Basic fallback
        >>> search = fallback(GoogleAPI(), BingAPI(), LocalCache())
        >>>
        >>> # With timeout and specific errors
        >>> search = fallback(
        ...     ExpensiveAPI(),
        ...     CheapAPI(),
        ...     timeout=5.0,
        ...     fallback_on=frozenset({ErrorCode.RATE_LIMITED, ErrorCode.TIMEOUT}),
        ... )
    """
    if not tools:
        raise ValueError("fallback() requires at least one tool")
    
    return FallbackTool(
        list(tools),
        timeout=timeout,
        fallback_on=fallback_on,
        name=name,
        description=description,
    )
