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

from pydantic import BaseModel, Field, ValidationError

from toolcase.foundation.core.base import BaseTool, ToolMetadata
from toolcase.foundation.errors import Err, ErrorCode, ErrorTrace, JsonDict, ToolResult, format_validation_error
from toolcase.runtime.concurrency import checkpoint


class RaceParams(BaseModel):
    """Parameters for race execution."""
    input: JsonDict = Field(default_factory=dict, description="Input parameters broadcasted to all racing tools")


RaceParams.model_rebuild()  # Resolve recursive JsonValue type


class RaceTool(BaseTool[RaceParams]):
    """Parallel execution with first-success-wins. Cancels remaining tools after first success.
    
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
        
        self._tools, self._timeout = tools, timeout
        tool_names = [t.metadata.name for t in tools]
        self._meta = ToolMetadata(
            name=name or f"race_{'_'.join(tool_names[:3])}",
            description=description or f"Race: {' | '.join(tool_names)}",
            category="agents", streaming=False,  # Race can't stream (first-wins semantics)
        )
    
    @property
    def metadata(self) -> ToolMetadata:
        return self._meta
    
    @property
    def tools(self) -> list[BaseTool[BaseModel]]:
        return self._tools
    
    async def _async_run(self, params: RaceParams) -> str:
        r = await self._async_run_result(params)
        return r.unwrap() if r.is_ok() else r.unwrap_err().message
    
    async def _async_run_result(self, params: RaceParams) -> ToolResult:
        """Execute all tools, return first success using structured concurrency."""
        input_dict, errors, timed_out = params.input, [None] * len(self._tools), False
        
        async def run_tool(idx: int, tool: BaseTool[BaseModel]) -> tuple[int, ToolResult]:
            await checkpoint()
            try:
                tool_params = tool.params_schema(**input_dict)
            except ValidationError as e:
                return idx, Err(ErrorTrace(
                    message=format_validation_error(e, tool_name=tool.metadata.name),
                    error_code=ErrorCode.INVALID_PARAMS.value, recoverable=False,
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
                                for p in pending:
                                    p.cancel()
                                return result
                            errors[idx] = result.unwrap_err()
                        except Exception:
                            pass
        except TimeoutError:
            timed_out = True
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if any(not t.done() for t in tasks):
                await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_errors = [e for e in errors if e is not None]
        n = len(self._tools)
        
        if timed_out and not valid_errors:
            trace = ErrorTrace(message=f"All {n} racing tools timed out", error_code=ErrorCode.TIMEOUT.value, recoverable=True)
        elif not valid_errors:
            trace = ErrorTrace(message=f"All {n} racing tools failed", error_code=ErrorCode.UNKNOWN.value, recoverable=True)
        else:
            trace = ErrorTrace(
                message=f"All {n} racing tools failed", error_code=valid_errors[0].error_code,
                recoverable=any(e.recoverable for e in valid_errors),
                details="\n".join(f"- [{self._tools[i].metadata.name}] {e.message}" for i, e in enumerate(errors) if e),
            )
        return Err(trace.with_operation(f"race:{self._meta.name}"))


def race(
    *tools: BaseTool[BaseModel],
    timeout: float = 30.0,
    name: str | None = None,
    description: str | None = None,
) -> RaceTool:
    """Create a race between tools - first success wins.
    
    Runs all tools concurrently. Returns as soon as any tool succeeds. Cancels remaining tools after winner.
    
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
    return RaceTool(list(tools), timeout=timeout, name=name, description=description)
