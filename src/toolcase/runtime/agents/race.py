"""Race primitive for parallel execution with first-wins semantics.

Runs multiple tools concurrently, returns first successful result.
Uses structured concurrency for clean cancellation and error handling.

Useful for:
- Provider redundancy (fastest wins)
- Speculative execution (try multiple approaches)
- Latency optimization (hedge your bets)
- Load balancing across providers

Example:
    >>> fastest = race(
    ...     OpenAITool(),
    ...     AnthropicTool(),
    ...     LocalLLMTool(),
    ... )
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError

from toolcase.foundation.core.base import BaseTool, ToolMetadata
from toolcase.foundation.errors import Err, ErrorCode, ErrorTrace, JsonDict, ToolResult
from toolcase.runtime.concurrency import CancelScope, checkpoint

if TYPE_CHECKING:
    pass


class RaceParams(BaseModel):
    """Parameters for race execution."""
    
    input: JsonDict = Field(
        default_factory=dict,
        description="Input parameters broadcasted to all racing tools",
    )


# Rebuild model to resolve recursive JsonValue type
RaceParams.model_rebuild()


class RaceTool(BaseTool[RaceParams]):
    """Parallel execution with first-success-wins semantics.
    
    Runs all tools concurrently, returns as soon as one succeeds.
    Cancels remaining tools after first success.
    
    If all fail, returns combined error with all failures.
    
    Example:
        >>> race_search = RaceTool(
        ...     tools=[GoogleAPI(), BingAPI(), DuckDuckGoAPI()],
        ...     timeout=10.0,
        ... )
    """
    
    __slots__ = ("_tools", "_timeout", "_meta")
    
    params_schema = RaceParams
    cache_enabled = False
    
    def __init__(
        self,
        tools: list[BaseTool[BaseModel]],
        *,
        timeout: float = 30.0,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        if not tools:
            raise ValueError("Race requires at least one tool")
        
        self._tools = tools
        self._timeout = timeout
        
        tool_names = [t.metadata.name for t in tools]
        derived_name = name or f"race_{'_'.join(tool_names[:3])}"
        derived_desc = description or f"Race: {' | '.join(tool_names)}"
        
        self._meta = ToolMetadata(
            name=derived_name,
            description=derived_desc,
            category="agents",
            streaming=False,  # Race can't stream (first-wins semantics)
        )
    
    @property
    def metadata(self) -> ToolMetadata:
        return self._meta
    
    @property
    def tools(self) -> list[BaseTool[BaseModel]]:
        return self._tools
    
    def _run(self, params: RaceParams) -> str:
        return self._run_async_sync(self._async_run(params))
    
    async def _async_run(self, params: RaceParams) -> str:
        result = await self._async_run_result(params)
        if result.is_ok():
            return result.unwrap()
        return result.unwrap_err().message
    
    async def _async_run_result(self, params: RaceParams) -> ToolResult:
        """Execute all tools, return first success using structured concurrency."""
        input_dict = params.input
        errors: list[ErrorTrace | None] = [None] * len(self._tools)
        timed_out = False
        
        async def run_tool(idx: int, tool: BaseTool[BaseModel]) -> tuple[int, ToolResult]:
            await checkpoint()  # Cooperative cancellation point
            try:
                tool_params = tool.params_schema(**input_dict)
            except ValidationError as e:
                return idx, Err(ErrorTrace(
                    message=f"Tool {tool.metadata.name} params invalid: {e}",
                    error_code=ErrorCode.INVALID_PARAMS.value,
                    recoverable=False,
                ))
            return idx, await tool.arun_result(tool_params)
        
        tasks = [asyncio.create_task(run_tool(i, t)) for i, t in enumerate(self._tools)]
        pending = set(tasks)
        
        try:
            async with asyncio.timeout(self._timeout):
                while pending:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    
                    for task in done:
                        if task.cancelled():
                            continue
                        try:
                            idx, result = task.result()
                            if result.is_ok():
                                # Winner - cancel remaining
                                for p in pending:
                                    p.cancel()
                                return result
                            errors[idx] = result.unwrap_err()
                        except Exception:
                            pass  # Ignore exceptions from cancelled tasks
        except TimeoutError:
            timed_out = True
        finally:
            # Clean up any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for cancellations to complete
            if any(not t.done() for t in tasks):
                await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build error response
        valid_errors = [e for e in errors if e is not None]
        
        if timed_out and not valid_errors:
            trace = ErrorTrace(
                message=f"All {len(self._tools)} racing tools timed out",
                error_code=ErrorCode.TIMEOUT.value,
                recoverable=True,
            )
        elif not valid_errors:
            trace = ErrorTrace(
                message=f"All {len(self._tools)} racing tools failed",
                error_code=ErrorCode.UNKNOWN.value,
                recoverable=True,
            )
        else:
            trace = ErrorTrace(
                message=f"All {len(self._tools)} racing tools failed",
                error_code=valid_errors[0].error_code,
                recoverable=any(e.recoverable for e in valid_errors),
                details="\n".join(
                    f"- [{self._tools[i].metadata.name}] {e.message}"
                    for i, e in enumerate(errors) if e is not None
                ),
            )
        
        return Err(trace.with_operation(f"race:{self._meta.name}"))


def race(
    *tools: BaseTool[BaseModel],
    timeout: float = 30.0,
    name: str | None = None,
    description: str | None = None,
) -> RaceTool:
    """Create a race between tools - first success wins.
    
    Runs all tools concurrently. Returns as soon as any tool
    succeeds. Cancels remaining tools after winner.
    
    Args:
        *tools: Tools to race
        timeout: Maximum time to wait for any result
        name: Optional race name
        description: Optional description
    
    Returns:
        RaceTool instance
    
    Example:
        >>> # Race multiple providers
        >>> search = race(
        ...     GoogleSearchTool(),
        ...     BingSearchTool(),
        ...     DuckDuckGoTool(),
        ...     timeout=5.0,
        ... )
        >>>
        >>> # Race different strategies
        >>> answer = race(
        ...     RAGTool(),       # Try retrieval
        ...     WebSearchTool(),  # Try web search
        ...     CacheTool(),      # Try cache
        ... )
    """
    if not tools:
        raise ValueError("race() requires at least one tool")
    
    return RaceTool(
        list(tools),
        timeout=timeout,
        name=name,
        description=description,
    )
