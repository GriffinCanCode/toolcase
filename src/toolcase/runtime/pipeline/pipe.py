"""Tool composition via sequential and parallel pipelines.

Pipelines are tools themselves, enabling recursive composition.
Uses railway-oriented programming for error short-circuiting.

Sequential: tool1 >> tool2 >> tool3
Parallel: parallel(tool1, tool2, tool3)
Streaming: streaming_pipeline(tool1, tool2)  # propagates async generators
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, TypeVar

from pydantic import BaseModel, Field, ValidationError

from toolcase.foundation.core.base import BaseTool, ToolMetadata
from toolcase.foundation.errors import Err, ErrorCode, ErrorTrace, Ok, ToolResult, collect_results, sequence
from toolcase.io.streaming import StreamChunk, StreamEvent, StreamEventKind, stream_error
from toolcase.runtime.concurrency import Concurrency

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=BaseModel)
U = TypeVar("U", bound=BaseModel)

# ═════════════════════════════════════════════════════════════════════════════
# Transform Types
# ═════════════════════════════════════════════════════════════════════════════

# Transform: accumulated output → next tool's params dict
Transform = Callable[[str], dict[str, object]]

# ChunkTransform: individual chunk → transformed chunk (in-flight)
ChunkTransform = Callable[[str], str]

# StreamTransform: full stream control (async generator → async generator)
StreamTransform = Callable[[AsyncIterator[str]], AsyncIterator[str]]

# Merge: list of results → combined string
Merge = Callable[[list[str]], str]


def identity_dict(s: str) -> dict[str, object]:
    """Default transform: wrap result in 'input' key."""
    return {"input": s}


def identity_chunk(s: str) -> str:
    """Default chunk transform: pass through unchanged."""
    return s


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


@dataclass(frozen=True, slots=True)
class StreamStep:
    """A streaming pipeline step with chunk-aware transforms.
    
    Supports both in-flight chunk transformation and accumulated output
    transformation for preparing next step's parameters.
    
    Attributes:
        tool: The tool to execute (should support streaming)
        chunk_transform: Applied to each chunk as it flows through
        accumulate_transform: Applied to accumulated output for next params
    """
    
    tool: BaseTool[BaseModel]
    chunk_transform: ChunkTransform = field(default=identity_chunk)
    accumulate_transform: Transform = field(default=identity_dict)
    
    async def stream(self, params: BaseModel) -> AsyncIterator[str]:
        """Stream result chunks from tool with chunk_transform applied."""
        async for chunk in self.tool.stream_result(params):
            yield self.chunk_transform(chunk)
    
    async def execute_collected(self, params: BaseModel) -> ToolResult:
        """Execute and collect full result (fallback for non-streaming)."""
        return await self.tool.arun_result(params)
    
    def prepare_next(self, accumulated: str) -> dict[str, object]:
        """Transform accumulated output for next step's params."""
        return self.accumulate_transform(accumulated)


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
        output = ""
        
        for i, step in enumerate(self._steps):
            # Build params for this step with validation error handling
            try:
                step_params = step.tool.params_schema(**current_params)
            except ValidationError as e:
                trace = ErrorTrace(
                    message=f"Step {i+1} ({step.tool.metadata.name}) params invalid: {e}",
                    error_code=ErrorCode.INVALID_PARAMS.value,
                    recoverable=False,
                ).with_operation(f"pipeline:{self._meta.name}")
                return Err(trace)
            
            # Execute step
            result = await step.execute(step_params)
            
            # Short-circuit on error (railway-oriented)
            if result.is_err():
                return result.map_err(
                    lambda e: e.with_operation(f"pipeline:{self._meta.name}")
                )
            
            # Transform output for next step with exception handling
            output = result.unwrap()
            try:
                current_params = step.prepare_next(output)
            except Exception as e:
                trace = ErrorTrace(
                    message=f"Transform after step {i+1} failed: {e}",
                    error_code=ErrorCode.PARSE_ERROR.value,
                    recoverable=False,
                ).with_operation(f"pipeline:{self._meta.name}")
                return Err(trace)
        
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
# StreamingPipelineTool: Streaming Composition
# ═════════════════════════════════════════════════════════════════════════════


class StreamingPipelineTool(BaseTool[PipelineParams]):
    """Sequential tool composition with streaming propagation.
    
    Propagates async generators through pipeline steps, allowing incremental
    output to flow end-to-end. Each step can transform chunks in-flight.
    
    Streaming Modes:
        - passthrough: Chunks flow through, accumulated for next step's params
        - transform: Apply chunk_transform to each chunk as it passes
        - collect: Accumulate first step, stream subsequent steps
    
    Example:
        >>> search = SearchTool()  # Returns full result
        >>> summarize = StreamingSummarizeTool()  # Yields chunks
        >>> 
        >>> pipe = StreamingPipelineTool(
        ...     steps=[
        ...         StreamStep(search),  # Collected then passed
        ...         StreamStep(summarize, chunk_transform=str.upper),
        ...     ],
        ...     name="search_and_stream_summarize",
        ... )
        >>> 
        >>> async for chunk in pipe.stream_result(params):
        ...     print(chunk, end="", flush=True)
    """
    
    __slots__ = ("_steps", "_meta")
    
    params_schema = PipelineParams
    cache_enabled = False
    
    def __init__(
        self,
        steps: list[StreamStep],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        if not steps:
            raise ValueError("StreamingPipeline requires at least one step")
        
        tool_names = [s.tool.metadata.name for s in steps]
        derived_name = name or "_stream_".join(tool_names)
        derived_desc = description or f"Streaming: {' → '.join(tool_names)}"
        
        self._steps = steps
        self._meta = ToolMetadata(
            name=derived_name,
            description=derived_desc,
            category="pipeline",
            streaming=True,  # Always streaming
        )
    
    @property
    def metadata(self) -> ToolMetadata:
        return self._meta
    
    @property
    def steps(self) -> list[StreamStep]:
        return self._steps
    
    @property
    def supports_result_streaming(self) -> bool:
        return True
    
    def _run(self, params: PipelineParams) -> str:
        """Sync execution collects all streaming output."""
        return self._run_async_sync(self._async_run(params))
    
    async def _async_run(self, params: PipelineParams) -> str:
        """Collect all chunks into final result."""
        chunks: list[str] = []
        async for chunk in self.stream_result(params):
            chunks.append(chunk)
        return "".join(chunks)
    
    async def _async_run_result(self, params: PipelineParams) -> ToolResult:
        """Execute with Result-based error handling."""
        try:
            result = await self._async_run(params)
            return Ok(result)
        except Exception as e:
            return self._err_from_exc(e, "streaming pipeline")
    
    async def stream_result(self, params: PipelineParams) -> AsyncIterator[str]:
        """Stream through pipeline steps, propagating chunks.
        
        For each step:
        1. If step's tool supports streaming, stream through with chunk_transform
        2. If not, collect result then pass accumulated to next step
        3. Final step's chunks are yielded to caller
        
        Yields:
            Transformed chunks from the final streaming step
        """
        current_params = params.input
        
        for i, step in enumerate(self._steps):
            is_final = i == len(self._steps) - 1
            
            # Build params for this step
            try:
                step_params = step.tool.params_schema(**current_params)
            except ValidationError as e:
                # Yield error as single chunk on validation failure
                yield f"[Pipeline Error] Step {i+1} ({step.tool.metadata.name}): {e}"
                return
            
            if is_final:
                # Final step: yield all chunks to caller
                async for chunk in step.stream(step_params):
                    yield chunk
            else:
                # Intermediate step: collect chunks, prepare next params
                accumulated: list[str] = []
                async for chunk in step.stream(step_params):
                    accumulated.append(chunk)
                
                # Transform accumulated for next step
                try:
                    current_params = step.prepare_next("".join(accumulated))
                except Exception as e:
                    yield f"[Pipeline Error] Transform after step {i+1} failed: {e}"
                    return
    
    async def stream_result_events(self, params: PipelineParams) -> AsyncIterator[StreamEvent]:
        """Stream as typed events with lifecycle management.
        
        Wraps stream_result() with start/chunk/complete/error events.
        """
        tool_name = self._meta.name
        yield StreamEvent(kind=StreamEventKind.START, tool_name=tool_name)
        
        accumulated: list[str] = []
        index = 0
        
        try:
            async for content in self.stream_result(params):
                accumulated.append(content)
                yield StreamEvent(
                    kind=StreamEventKind.CHUNK,
                    tool_name=tool_name,
                    data=StreamChunk(content=content, index=index),
                )
                index += 1
            
            yield StreamEvent(
                kind=StreamEventKind.COMPLETE,
                tool_name=tool_name,
                accumulated="".join(accumulated),
            )
        except Exception as e:
            yield stream_error(tool_name, str(e))
            raise
    
    def __rshift__(self, other: BaseTool[BaseModel] | StreamStep) -> StreamingPipelineTool:
        """Chain another tool: self >> other."""
        next_step = other if isinstance(other, StreamStep) else StreamStep(other)
        return StreamingPipelineTool(steps=[*self._steps, next_step], name=None)


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
        """Execute all tools concurrently using Concurrency.gather."""
        async def run_tool(tool: BaseTool[BaseModel]) -> ToolResult:
            try:
                tool_params = tool.params_schema(**params.input)
            except ValidationError as e:
                return Err(ErrorTrace(
                    message=f"Params invalid for {tool.metadata.name}: {e}",
                    error_code=ErrorCode.INVALID_PARAMS.value,
                    recoverable=False,
                ))
            return await tool.arun_result(tool_params)
        
        # Execute all concurrently using Concurrency facade
        results = await Concurrency.gather(*[run_tool(t) for t in self._tools])
        
        # Combine results
        if self._fail_fast:
            sequenced = sequence(list(results))
            if sequenced.is_err():
                return Err(sequenced.unwrap_err().with_operation(f"parallel:{self._meta.name}"))
            try:
                return Ok(self._merge(sequenced.unwrap()))
            except Exception as e:
                return Err(ErrorTrace(
                    message=f"Merge failed in {self._meta.name}: {e}",
                    error_code=ErrorCode.PARSE_ERROR.value,
                    recoverable=False,
                ))
        
        # Collect all (accumulate errors)
        collected = collect_results(list(results))
        if collected.is_err():
            errors = collected.unwrap_err()
            return Err(ErrorTrace(
                message=f"Multiple failures in {self._meta.name}",
                contexts=[],
                error_code=errors[0].error_code if errors else None,
                recoverable=any(e.recoverable for e in errors),
            ))
        
        try:
            return Ok(self._merge(collected.unwrap()))
        except Exception as e:
            return Err(ErrorTrace(
                message=f"Merge failed in {self._meta.name}: {e}",
                error_code=ErrorCode.PARSE_ERROR.value,
                recoverable=False,
            ))


# ═════════════════════════════════════════════════════════════════════════════
# StreamingParallelTool: Concurrent Streaming Composition
# ═════════════════════════════════════════════════════════════════════════════


# Merge that operates on async iterators instead of lists
StreamMerge = Callable[[list[AsyncIterator[str]]], AsyncIterator[str]]


async def interleave_streams(streams: list[AsyncIterator[str]]) -> AsyncIterator[str]:
    """Default stream merge: interleave chunks round-robin as they arrive.
    
    Uses cooperative cancellation via checkpoint() for clean shutdown.
    """
    from toolcase.runtime.concurrency import checkpoint
    
    pending: dict[int, asyncio.Task[tuple[int, str | None]]] = {}
    active_streams: set[int] = set(range(len(streams)))
    
    async def get_next(idx: int, stream: AsyncIterator[str]) -> tuple[int, str | None]:
        try:
            return (idx, await stream.__anext__())
        except StopAsyncIteration:
            return (idx, None)
    
    # Initialize tasks
    for i, stream in enumerate(streams):
        pending[i] = asyncio.create_task(get_next(i, stream))
    
    while pending:
        await checkpoint()  # Cooperative cancellation point
        done, _ = await asyncio.wait(pending.values(), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            idx, chunk = task.result()
            if chunk is None:
                active_streams.discard(idx)
                del pending[idx]
            else:
                yield chunk
                if idx in active_streams:
                    pending[idx] = asyncio.create_task(get_next(idx, streams[idx]))


class StreamingParallelTool(BaseTool[ParallelParams]):
    """Parallel streaming tool execution with stream merging.
    
    Runs multiple streaming tools concurrently, interleaving their
    chunks as they arrive. Perfect for multi-source aggregation where
    you want incremental output from all sources.
    
    Example:
        >>> web = StreamingWebSearch()
        >>> news = StreamingNewsSearch()
        >>> 
        >>> multi = StreamingParallelTool(
        ...     tools=[web, news],
        ...     stream_merge=interleave_streams,  # or custom merger
        ... )
        >>> 
        >>> async for chunk in multi.stream_result(params):
        ...     print(chunk)  # Interleaved chunks from both sources
    """
    
    __slots__ = ("_tools", "_stream_merge", "_meta")
    
    params_schema = ParallelParams
    cache_enabled = False
    
    def __init__(
        self,
        tools: list[BaseTool[BaseModel]],
        *,
        stream_merge: StreamMerge | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        if not tools:
            raise ValueError("StreamingParallel requires at least one tool")
        
        tool_names = [t.metadata.name for t in tools]
        derived_name = name or "_stream_and_".join(tool_names)
        derived_desc = description or f"StreamingParallel: {', '.join(tool_names)}"
        
        self._tools = tools
        self._stream_merge = stream_merge or interleave_streams
        self._meta = ToolMetadata(
            name=derived_name,
            description=derived_desc,
            category="pipeline",
            streaming=True,
        )
    
    @property
    def metadata(self) -> ToolMetadata:
        return self._meta
    
    @property
    def tools(self) -> list[BaseTool[BaseModel]]:
        return self._tools
    
    @property
    def supports_result_streaming(self) -> bool:
        return True
    
    def _run(self, params: ParallelParams) -> str:
        return self._run_async_sync(self._async_run(params))
    
    async def _async_run(self, params: ParallelParams) -> str:
        """Collect all streaming output."""
        chunks: list[str] = []
        async for chunk in self.stream_result(params):
            chunks.append(chunk)
        return "".join(chunks)
    
    async def _async_run_result(self, params: ParallelParams) -> ToolResult:
        try:
            return Ok(await self._async_run(params))
        except Exception as e:
            return self._err_from_exc(e, "streaming parallel")
    
    async def stream_result(self, params: ParallelParams) -> AsyncIterator[str]:
        """Stream merged output from all tools concurrently.
        
        Each tool's streaming output is merged according to stream_merge.
        Default: interleave chunks as they arrive from any source.
        """
        async def tool_stream(tool: BaseTool[BaseModel]) -> AsyncIterator[str]:
            try:
                tool_params = tool.params_schema(**params.input)
            except ValidationError as e:
                yield f"[Error {tool.metadata.name}]: {e}"
                return
            async for chunk in tool.stream_result(tool_params):
                yield chunk
        
        streams = [tool_stream(t) for t in self._tools]
        async for chunk in self._stream_merge(streams):
            yield chunk
    
    async def stream_result_events(self, params: ParallelParams) -> AsyncIterator[StreamEvent]:
        """Stream as typed events with lifecycle management."""
        tool_name = self._meta.name
        yield StreamEvent(kind=StreamEventKind.START, tool_name=tool_name)
        
        accumulated: list[str] = []
        index = 0
        
        try:
            async for content in self.stream_result(params):
                accumulated.append(content)
                yield StreamEvent(
                    kind=StreamEventKind.CHUNK,
                    tool_name=tool_name,
                    data=StreamChunk(content=content, index=index),
                )
                index += 1
            
            yield StreamEvent(
                kind=StreamEventKind.COMPLETE,
                tool_name=tool_name,
                accumulated="".join(accumulated),
            )
        except Exception as e:
            yield stream_error(tool_name, str(e))
            raise


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


def streaming_pipeline(
    *tools: BaseTool[BaseModel],
    chunk_transforms: list[ChunkTransform] | None = None,
    accumulate_transforms: list[Transform] | None = None,
    name: str | None = None,
    description: str | None = None,
) -> StreamingPipelineTool:
    """Create streaming pipeline that propagates async generators.
    
    Chains tools where streaming output flows through transforms
    and intermediate results are accumulated for next step's params.
    
    Args:
        *tools: Tools to chain (should support streaming for full benefit)
        chunk_transforms: Per-chunk transform functions (one per step)
        accumulate_transforms: Accumulated output → next params (one per step)
        name: Override derived pipeline name
        description: Override derived description
    
    Returns:
        StreamingPipelineTool instance
    
    Example:
        >>> # Basic streaming chain
        >>> pipe = streaming_pipeline(search, summarize, format_output)
        >>> 
        >>> # With chunk transforms (e.g., uppercase all chunks)
        >>> pipe = streaming_pipeline(
        ...     search,
        ...     summarize,
        ...     chunk_transforms=[None, str.upper],  # uppercase summary chunks
        ... )
        >>> 
        >>> # Consume stream
        >>> async for chunk in pipe.stream_result(params):
        ...     print(chunk, end="", flush=True)
    """
    if not tools:
        raise ValueError("streaming_pipeline() requires at least one tool")
    
    chunk_transforms = chunk_transforms or []
    accumulate_transforms = accumulate_transforms or []
    
    steps: list[StreamStep] = []
    for i, tool in enumerate(tools):
        chunk_fn = chunk_transforms[i] if i < len(chunk_transforms) and chunk_transforms[i] else identity_chunk
        accum_fn = accumulate_transforms[i] if i < len(accumulate_transforms) and accumulate_transforms[i] else identity_dict
        steps.append(StreamStep(tool, chunk_fn, accum_fn))
    
    return StreamingPipelineTool(steps, name=name, description=description)


def streaming_parallel(
    *tools: BaseTool[BaseModel],
    stream_merge: StreamMerge | None = None,
    name: str | None = None,
    description: str | None = None,
) -> StreamingParallelTool:
    """Create streaming parallel execution that interleaves outputs.
    
    Runs tools concurrently, merging streaming output as chunks arrive.
    
    Args:
        *tools: Tools to execute concurrently (should support streaming)
        stream_merge: Custom function to merge streams (default: interleave)
        name: Override derived pipeline name
        description: Override derived description
    
    Returns:
        StreamingParallelTool instance
    
    Example:
        >>> # Interleaved streaming from multiple sources
        >>> multi = streaming_parallel(
        ...     StreamingWebSearch(),
        ...     StreamingNewsSearch(),
        ... )
        >>> 
        >>> async for chunk in multi.stream_result(params):
        ...     print(chunk)  # Chunks arrive as they're ready
    """
    if not tools:
        raise ValueError("streaming_parallel() requires at least one tool")
    
    return StreamingParallelTool(
        list(tools),
        stream_merge=stream_merge,
        name=name,
        description=description,
    )
