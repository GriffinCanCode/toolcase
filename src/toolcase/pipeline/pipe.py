"""Tool composition via sequential and parallel pipelines.

Pipelines are tools themselves, enabling recursive composition.
Uses railway-oriented programming for error short-circuiting.

Sequential: tool1 >> tool2 >> tool3
Parallel: parallel(tool1, tool2, tool3)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

from pydantic import BaseModel, Field

from ..core.base import BaseTool, EmptyParams, ToolMetadata
from ..monads import Err, Ok, ToolResult, collect_results, sequence

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=BaseModel)
U = TypeVar("U", bound=BaseModel)

# ═════════════════════════════════════════════════════════════════════════════
# Transform Types
# ═════════════════════════════════════════════════════════════════════════════

# Transform: output string → next tool's params dict
Transform = Callable[[str], dict[str, object]]

# Merge: list of results → combined string
Merge = Callable[[list[str]], str]


def identity_dict(s: str) -> dict[str, object]:
    """Default transform: wrap result in 'input' key."""
    return {"input": s}


def concat_merge(results: list[str], sep: str = "\n\n") -> str:
    """Default merge: concatenate with separator."""
    return sep.join(results)


# ═════════════════════════════════════════════════════════════════════════════
# Step: Tool + Transform pair
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class Step:
    """A pipeline step: tool with optional output transform.
    
    The transform maps this step's output to the next step's input params.
    """
    
    tool: BaseTool[BaseModel]
    transform: Transform = field(default=identity_dict)
    
    async def execute(self, params: BaseModel) -> ToolResult:
        """Execute step and return Result."""
        return await self.tool.arun_result(params)
    
    def prepare_next(self, output: str) -> dict[str, object]:
        """Transform output for next step's params."""
        return self.transform(output)


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline Params
# ═════════════════════════════════════════════════════════════════════════════


class PipelineParams(BaseModel):
    """Parameters for pipeline execution - pass-through to first tool."""
    
    input: dict[str, object] = Field(default_factory=dict, description="Input params for first tool")


class ParallelParams(BaseModel):
    """Parameters for parallel execution - broadcast to all tools."""
    
    input: dict[str, object] = Field(default_factory=dict, description="Input params for all tools")


# ═════════════════════════════════════════════════════════════════════════════
# PipelineTool: Sequential Composition
# ═════════════════════════════════════════════════════════════════════════════


class PipelineTool(BaseTool[PipelineParams]):
    """Sequential tool composition with transform functions.
    
    Executes tools in order, passing each output through a transform
    to create the next tool's input. Short-circuits on first error.
    
    Example:
        >>> search = SearchTool()
        >>> summarize = SummarizeTool()
        >>> 
        >>> pipe = PipelineTool(
        ...     steps=[
        ...         Step(search),
        ...         Step(summarize, transform=lambda r: {"text": r}),
        ...     ],
        ...     name="search_and_summarize",
        ... )
        >>> 
        >>> # Or use >> operator:
        >>> pipe = search >> summarize
    """
    
    __slots__ = ("_steps", "_meta")
    
    params_schema = PipelineParams
    cache_enabled = False  # Pipelines delegate caching to inner tools
    
    def __init__(
        self,
        steps: list[Step],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        if not steps:
            raise ValueError("Pipeline requires at least one step")
        
        # Derive name from tools
        tool_names = [s.tool.metadata.name for s in steps]
        derived_name = name or "_then_".join(tool_names)
        derived_desc = description or f"Pipeline: {' → '.join(tool_names)}"
        
        self._steps = steps
        self._meta = ToolMetadata(
            name=derived_name,
            description=derived_desc,
            category="pipeline",
            streaming=any(s.tool.metadata.streaming for s in steps),
        )
    
    @property
    def metadata(self) -> ToolMetadata:
        return self._meta
    
    @property
    def steps(self) -> list[Step]:
        return self._steps
    
    def _run(self, params: PipelineParams) -> str:
        """Sync execution via async bridge."""
        return self._run_async_sync(self._async_run(params))
    
    async def _async_run(self, params: PipelineParams) -> str:
        """Execute pipeline sequentially."""
        result = await self._async_run_result(params)
        return result.unwrap_or(result.unwrap_err().message)
    
    async def _async_run_result(self, params: PipelineParams) -> ToolResult:
        """Execute with Result-based error handling."""
        current_params = params.input
        
        for step in self._steps:
            # Build params for this step
            step_params = step.tool.params_schema(**current_params)
            
            # Execute step
            result = await step.execute(step_params)
            
            # Short-circuit on error (railway-oriented)
            if result.is_err():
                return result.map_err(
                    lambda e: e.with_operation(f"pipeline:{self._meta.name}")
                )
            
            # Transform output for next step
            output = result.unwrap()
            current_params = step.prepare_next(output)
        
        # Return final output
        return Ok(output)
    
    def __rshift__(self, other: BaseTool[BaseModel] | Step) -> PipelineTool:
        """Chain another tool: self >> other."""
        next_step = other if isinstance(other, Step) else Step(other)
        return PipelineTool(
            steps=[*self._steps, next_step],
            name=None,  # Re-derive from combined steps
        )


# ═════════════════════════════════════════════════════════════════════════════
# ParallelTool: Concurrent Composition
# ═════════════════════════════════════════════════════════════════════════════


class ParallelTool(BaseTool[ParallelParams]):
    """Parallel tool execution with result merging.
    
    Executes all tools concurrently, then merges results.
    Can fail-fast or collect all errors.
    
    Example:
        >>> web = WebSearchTool()
        >>> news = NewsSearchTool()
        >>> academic = AcademicSearchTool()
        >>> 
        >>> multi = ParallelTool(
        ...     tools=[web, news, academic],
        ...     merge=lambda rs: "\\n---\\n".join(rs),
        ... )
        >>> 
        >>> # Or use factory:
        >>> multi = parallel(web, news, academic)
    """
    
    __slots__ = ("_tools", "_merge", "_fail_fast", "_meta")
    
    params_schema = ParallelParams
    cache_enabled = False
    
    def __init__(
        self,
        tools: list[BaseTool[BaseModel]],
        *,
        merge: Merge | None = None,
        fail_fast: bool = True,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        if not tools:
            raise ValueError("Parallel requires at least one tool")
        
        tool_names = [t.metadata.name for t in tools]
        derived_name = name or "_and_".join(tool_names)
        derived_desc = description or f"Parallel: {', '.join(tool_names)}"
        
        self._tools = tools
        self._merge = merge or concat_merge
        self._fail_fast = fail_fast
        self._meta = ToolMetadata(
            name=derived_name,
            description=derived_desc,
            category="pipeline",
            streaming=any(t.metadata.streaming for t in tools),
        )
    
    @property
    def metadata(self) -> ToolMetadata:
        return self._meta
    
    @property
    def tools(self) -> list[BaseTool[BaseModel]]:
        return self._tools
    
    def _run(self, params: ParallelParams) -> str:
        return self._run_async_sync(self._async_run(params))
    
    async def _async_run(self, params: ParallelParams) -> str:
        result = await self._async_run_result(params)
        return result.unwrap_or(result.unwrap_err().message)
    
    async def _async_run_result(self, params: ParallelParams) -> ToolResult:
        """Execute all tools concurrently."""
        # Create params for each tool
        async def run_tool(tool: BaseTool[BaseModel]) -> ToolResult:
            tool_params = tool.params_schema(**params.input)
            return await tool.arun_result(tool_params)
        
        # Execute all concurrently
        results = await asyncio.gather(
            *[run_tool(t) for t in self._tools],
            return_exceptions=False,
        )
        
        # Combine results
        if self._fail_fast:
            sequenced = sequence(list(results))
            if sequenced.is_err():
                return Err(
                    sequenced.unwrap_err().with_operation(f"parallel:{self._meta.name}")
                )
            return Ok(self._merge(sequenced.unwrap()))
        
        # Collect all (accumulate errors)
        collected = collect_results(list(results))
        if collected.is_err():
            errors = collected.unwrap_err()
            from ..monads.types import ErrorTrace
            combined = ErrorTrace(
                message=f"Multiple failures in {self._meta.name}",
                contexts=[],
                error_code=errors[0].error_code if errors else None,
                recoverable=any(e.recoverable for e in errors),
            )
            return Err(combined)
        
        return Ok(self._merge(collected.unwrap()))


# ═════════════════════════════════════════════════════════════════════════════
# Factory Functions
# ═════════════════════════════════════════════════════════════════════════════


def pipeline(
    *tools: BaseTool[BaseModel],
    transforms: list[Transform] | None = None,
    name: str | None = None,
    description: str | None = None,
) -> PipelineTool:
    """Create sequential pipeline from tools.
    
    Args:
        *tools: Tools to chain sequentially
        transforms: Optional list of transform functions (one per step)
        name: Override derived pipeline name
        description: Override derived description
    
    Returns:
        PipelineTool instance
    
    Example:
        >>> pipe = pipeline(
        ...     SearchTool(),
        ...     SummarizeTool(),
        ...     transforms=[
        ...         lambda r: {"query": r},  # search → summarize
        ...     ]
        ... )
    """
    if not tools:
        raise ValueError("pipeline() requires at least one tool")
    
    transforms = transforms or []
    steps: list[Step] = []
    
    for i, tool in enumerate(tools):
        transform = transforms[i] if i < len(transforms) else identity_dict
        steps.append(Step(tool, transform))
    
    return PipelineTool(steps, name=name, description=description)


def parallel(
    *tools: BaseTool[BaseModel],
    merge: Merge | None = None,
    fail_fast: bool = True,
    name: str | None = None,
    description: str | None = None,
) -> ParallelTool:
    """Create parallel execution from tools.
    
    Args:
        *tools: Tools to execute concurrently
        merge: Function to combine results (default: concat)
        fail_fast: Stop on first error (default: True)
        name: Override derived pipeline name
        description: Override derived description
    
    Returns:
        ParallelTool instance
    
    Example:
        >>> multi = parallel(
        ...     WebSearchTool(),
        ...     NewsSearchTool(),
        ...     merge=lambda rs: "Sources:\\n" + "\\n".join(rs),
        ... )
    """
    if not tools:
        raise ValueError("parallel() requires at least one tool")
    
    return ParallelTool(
        list(tools),
        merge=merge,
        fail_fast=fail_fast,
        name=name,
        description=description,
    )
