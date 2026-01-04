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
from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from toolcase.foundation.core.base import BaseTool, ToolMetadata
from toolcase.foundation.errors import Err, ErrorCode, ErrorTrace, JsonDict, Ok, ToolResult, collect_results, format_validation_error, sequence
from toolcase.io.streaming import StreamChunk, StreamEvent, StreamEventKind, stream_error
from toolcase.runtime.concurrency import Concurrency

# ═════════════════════════════════════════════════════════════════════════════
# Transform Types
# ═════════════════════════════════════════════════════════════════════════════

# Transform: accumulated output → next tool's params dict
Transform = Callable[[str], JsonDict]

# ChunkTransform: individual chunk → transformed chunk (in-flight)
ChunkTransform = Callable[[str], str]

# StreamTransform: full stream control (async generator → async generator)
StreamTransform = Callable[[AsyncIterator[str]], AsyncIterator[str]]

# Merge: list of results → combined string
Merge = Callable[[list[str]], str]


def identity_dict(s: str) -> JsonDict:
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
    """A pipeline step: tool with optional output transform. Maps output to next step's input params."""
    tool: BaseTool[BaseModel]
    transform: Transform = field(default=identity_dict)
    
    async def execute(self, params: BaseModel) -> ToolResult:
        """Execute step and return Result."""
        return await self.tool.arun_result(params)
    
    def prepare_next(self, output: str) -> JsonDict:
        """Transform output for next step's params."""
        return self.transform(output)


@dataclass(frozen=True, slots=True)
class StreamStep:
    """Streaming pipeline step with chunk-aware transforms for in-flight and accumulated output."""
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
    
    def prepare_next(self, accumulated: str) -> JsonDict:
        """Transform accumulated output for next step's params."""
        return self.accumulate_transform(accumulated)


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline Params
# ═════════════════════════════════════════════════════════════════════════════


class PipelineParams(BaseModel):
    """Parameters for pipeline execution - pass-through to first tool."""
    
    input: JsonDict = Field(default_factory=dict, description="Input params for first tool")


class ParallelParams(BaseModel):
    """Parameters for parallel execution - broadcast to all tools."""
    
    input: JsonDict = Field(default_factory=dict, description="Input params for all tools")


# Rebuild models to resolve recursive JsonValue type
PipelineParams.model_rebuild()
ParallelParams.model_rebuild()


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
    
    def __init__(self, steps: list[Step], *, name: str | None = None, description: str | None = None) -> None:
        if not steps:
            raise ValueError("Pipeline requires at least one step")
        tool_names = [s.tool.metadata.name for s in steps]
        self._steps = steps
        self._meta = ToolMetadata(
            name=name or "_then_".join(tool_names),
            description=description or f"Pipeline: {' → '.join(tool_names)}",
            category="pipeline",
            streaming=any(s.tool.metadata.streaming for s in steps),
        )
    
    @property
    def metadata(self) -> ToolMetadata: return self._meta
    
    @property
    def steps(self) -> list[Step]: return self._steps
    
    def _run(self, params: PipelineParams) -> str:
        """Sync execution via async bridge."""
        return self._run_async_sync(self._async_run(params))
    
    async def _async_run(self, params: PipelineParams) -> str:
        """Execute pipeline sequentially."""
        return (r := await self._async_run_result(params)).unwrap_or(r.unwrap_err().message)
    
    async def _async_run_result(self, params: PipelineParams) -> ToolResult:
        """Execute with Result-based error handling."""
        current_params, output, op = params.input, "", f"pipeline:{self._meta.name}"
        for i, step in enumerate(self._steps):
            try: step_params = step.tool.params_schema(**current_params)
            except ValidationError as e:
                return Err(ErrorTrace(message=format_validation_error(e, tool_name=step.tool.metadata.name), error_code=ErrorCode.INVALID_PARAMS.value, recoverable=False).with_operation(op))
            if (result := await step.execute(step_params)).is_err():
                return result.map_err(lambda e: e.with_operation(op))
            output = result.unwrap()
            try: current_params = step.prepare_next(output)
            except Exception as e:
                return Err(ErrorTrace(message=f"Transform after step {i+1} failed: {e}", error_code=ErrorCode.PARSE_ERROR.value, recoverable=False).with_operation(op))
        return Ok(output)
    
    def __rshift__(self, other: BaseTool[BaseModel] | Step) -> PipelineTool:
        """Chain another tool: self >> other."""
        return PipelineTool(steps=[*self._steps, other if isinstance(other, Step) else Step(other)])


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
    
    def __init__(self, steps: list[StreamStep], *, name: str | None = None, description: str | None = None) -> None:
        if not steps:
            raise ValueError("StreamingPipeline requires at least one step")
        tool_names = [s.tool.metadata.name for s in steps]
        self._steps = steps
        self._meta = ToolMetadata(
            name=name or "_stream_".join(tool_names),
            description=description or f"Streaming: {' → '.join(tool_names)}",
            category="pipeline", streaming=True,
        )
    
    @property
    def metadata(self) -> ToolMetadata: return self._meta
    
    @property
    def steps(self) -> list[StreamStep]: return self._steps
    
    @property
    def supports_result_streaming(self) -> bool: return True
    
    def _run(self, params: PipelineParams) -> str:
        """Sync execution collects all streaming output."""
        return self._run_async_sync(self._async_run(params))
    
    async def _async_run(self, params: PipelineParams) -> str:
        """Collect all chunks into final result."""
        return "".join([chunk async for chunk in self.stream_result(params)])
    
    async def _async_run_result(self, params: PipelineParams) -> ToolResult:
        """Execute with Result-based error handling."""
        try: return Ok(await self._async_run(params))
        except Exception as e: return self._err_from_exc(e, "streaming pipeline")
    
    async def stream_result(self, params: PipelineParams) -> AsyncIterator[str]:
        """Stream through pipeline steps, propagating chunks. Final step's chunks yielded to caller."""
        current_params = params.input
        for i, step in enumerate(self._steps):
            try: step_params = step.tool.params_schema(**current_params)
            except ValidationError as e:
                yield f"[Pipeline Error] {format_validation_error(e, tool_name=step.tool.metadata.name)}"; return
            if i == len(self._steps) - 1:
                async for chunk in step.stream(step_params): yield chunk
            else:
                accumulated = [chunk async for chunk in step.stream(step_params)]
                try: current_params = step.prepare_next("".join(accumulated))
                except Exception as e:
                    yield f"[Pipeline Error] Transform after step {i+1} failed: {e}"; return
    
    async def stream_result_events(self, params: PipelineParams) -> AsyncIterator[StreamEvent]:
        """Stream as typed events with lifecycle management. Wraps stream_result() with start/chunk/complete/error events."""
        name, acc, idx = self._meta.name, [], 0
        yield StreamEvent(kind=StreamEventKind.START, tool_name=name)
        try:
            async for content in self.stream_result(params):
                acc.append(content)
                yield StreamEvent(kind=StreamEventKind.CHUNK, tool_name=name, data=StreamChunk(content=content, index=idx)); idx += 1
            yield StreamEvent(kind=StreamEventKind.COMPLETE, tool_name=name, accumulated="".join(acc))
        except Exception as e: yield stream_error(name, str(e)); raise
    
    def __rshift__(self, other: BaseTool[BaseModel] | StreamStep) -> StreamingPipelineTool:
        """Chain another tool: self >> other."""
        return StreamingPipelineTool(steps=[*self._steps, other if isinstance(other, StreamStep) else StreamStep(other)])


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
    
    def __init__(self, tools: list[BaseTool[BaseModel]], *, merge: Merge | None = None, fail_fast: bool = True, name: str | None = None, description: str | None = None) -> None:
        if not tools:
            raise ValueError("Parallel requires at least one tool")
        tool_names = [t.metadata.name for t in tools]
        self._tools, self._merge, self._fail_fast = tools, merge or concat_merge, fail_fast
        self._meta = ToolMetadata(
            name=name or "_and_".join(tool_names),
            description=description or f"Parallel: {', '.join(tool_names)}",
            category="pipeline", streaming=any(t.metadata.streaming for t in tools),
        )
    
    @property
    def metadata(self) -> ToolMetadata: return self._meta
    
    @property
    def tools(self) -> list[BaseTool[BaseModel]]: return self._tools
    
    def _run(self, params: ParallelParams) -> str:
        return self._run_async_sync(self._async_run(params))
    
    async def _async_run(self, params: ParallelParams) -> str:
        return (r := await self._async_run_result(params)).unwrap_or(r.unwrap_err().message)
    
    async def _async_run_result(self, params: ParallelParams) -> ToolResult:
        """Execute all tools concurrently using Concurrency.gather."""
        async def run_tool(tool: BaseTool[BaseModel]) -> ToolResult:
            try: return await tool.arun_result(tool.params_schema(**params.input))
            except ValidationError as e:
                return Err(ErrorTrace(message=format_validation_error(e, tool_name=tool.metadata.name), error_code=ErrorCode.INVALID_PARAMS.value, recoverable=False))
        
        results, name = await Concurrency.gather(*[run_tool(t) for t in self._tools]), self._meta.name
        def merge_safe(values: list[str]) -> ToolResult:
            try: return Ok(self._merge(values))
            except Exception as e: return Err(ErrorTrace(message=f"Merge failed in {name}: {e}", error_code=ErrorCode.PARSE_ERROR.value, recoverable=False))
        
        if self._fail_fast:
            return merge_safe((seq := sequence(list(results))).unwrap()) if seq.is_ok() else Err(seq.unwrap_err().with_operation(f"parallel:{name}"))
        if (collected := collect_results(list(results))).is_err():
            errs = collected.unwrap_err()
            return Err(ErrorTrace(message=f"Multiple failures in {name}", contexts=[], error_code=errs[0].error_code if errs else None, recoverable=any(e.recoverable for e in errs)))
        return merge_safe(collected.unwrap())


# ═════════════════════════════════════════════════════════════════════════════
# StreamingParallelTool: Concurrent Streaming Composition
# ═════════════════════════════════════════════════════════════════════════════


# Merge that operates on async iterators instead of lists
StreamMerge = Callable[[list[AsyncIterator[str]]], AsyncIterator[str]]


async def interleave_streams(streams: list[AsyncIterator[str]]) -> AsyncIterator[str]:
    """Default stream merge: interleave chunks round-robin as they arrive. Uses cooperative cancellation via checkpoint()."""
    from toolcase.runtime.concurrency import checkpoint
    
    async def get_next(idx: int, stream: AsyncIterator[str]) -> tuple[int, str | None]:
        try: return (idx, await stream.__anext__())
        except StopAsyncIteration: return (idx, None)
    
    active, pending = set(range(len(streams))), {i: asyncio.create_task(get_next(i, s)) for i, s in enumerate(streams)}
    while pending:
        await checkpoint()
        for task in (await asyncio.wait(pending.values(), return_when=asyncio.FIRST_COMPLETED))[0]:
            idx, chunk = task.result()
            if chunk is None:
                active.discard(idx); del pending[idx]
            else:
                yield chunk
                if idx in active: pending[idx] = asyncio.create_task(get_next(idx, streams[idx]))


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
    
    def __init__(self, tools: list[BaseTool[BaseModel]], *, stream_merge: StreamMerge | None = None, name: str | None = None, description: str | None = None) -> None:
        if not tools:
            raise ValueError("StreamingParallel requires at least one tool")
        tool_names = [t.metadata.name for t in tools]
        self._tools, self._stream_merge = tools, stream_merge or interleave_streams
        self._meta = ToolMetadata(
            name=name or "_stream_and_".join(tool_names),
            description=description or f"StreamingParallel: {', '.join(tool_names)}",
            category="pipeline", streaming=True,
        )
    
    @property
    def metadata(self) -> ToolMetadata: return self._meta
    
    @property
    def tools(self) -> list[BaseTool[BaseModel]]: return self._tools
    
    @property
    def supports_result_streaming(self) -> bool: return True
    
    def _run(self, params: ParallelParams) -> str:
        return self._run_async_sync(self._async_run(params))
    
    async def _async_run(self, params: ParallelParams) -> str:
        """Collect all streaming output."""
        return "".join([chunk async for chunk in self.stream_result(params)])
    
    async def _async_run_result(self, params: ParallelParams) -> ToolResult:
        try: return Ok(await self._async_run(params))
        except Exception as e: return self._err_from_exc(e, "streaming parallel")
    
    async def stream_result(self, params: ParallelParams) -> AsyncIterator[str]:
        """Stream merged output from all tools concurrently. Default: interleave chunks as they arrive."""
        async def tool_stream(tool: BaseTool[BaseModel]) -> AsyncIterator[str]:
            try: tool_params = tool.params_schema(**params.input)
            except ValidationError as e:
                yield f"[Error] {format_validation_error(e, tool_name=tool.metadata.name)}"; return
            async for chunk in tool.stream_result(tool_params): yield chunk
        async for chunk in self._stream_merge([tool_stream(t) for t in self._tools]): yield chunk
    
    async def stream_result_events(self, params: ParallelParams) -> AsyncIterator[StreamEvent]:
        """Stream as typed events with lifecycle management."""
        name, acc, idx = self._meta.name, [], 0
        yield StreamEvent(kind=StreamEventKind.START, tool_name=name)
        try:
            async for content in self.stream_result(params):
                acc.append(content)
                yield StreamEvent(kind=StreamEventKind.CHUNK, tool_name=name, data=StreamChunk(content=content, index=idx)); idx += 1
            yield StreamEvent(kind=StreamEventKind.COMPLETE, tool_name=name, accumulated="".join(acc))
        except Exception as e: yield stream_error(name, str(e)); raise


# ═════════════════════════════════════════════════════════════════════════════
# Factory Functions
# ═════════════════════════════════════════════════════════════════════════════


def pipeline(*tools: BaseTool[BaseModel], transforms: list[Transform] | None = None, name: str | None = None, description: str | None = None) -> PipelineTool:
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
    xforms = transforms or []
    steps = [Step(tool, xforms[i] if i < len(xforms) else identity_dict) for i, tool in enumerate(tools)]
    return PipelineTool(steps, name=name, description=description)


def parallel(*tools: BaseTool[BaseModel], merge: Merge | None = None, fail_fast: bool = True, name: str | None = None, description: str | None = None) -> ParallelTool:
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
    return ParallelTool(list(tools), merge=merge, fail_fast=fail_fast, name=name, description=description)


def streaming_pipeline(*tools: BaseTool[BaseModel], chunk_transforms: list[ChunkTransform] | None = None, accumulate_transforms: list[Transform] | None = None, name: str | None = None, description: str | None = None) -> StreamingPipelineTool:
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
    c_xforms, a_xforms = chunk_transforms or [], accumulate_transforms or []
    steps = [
        StreamStep(tool, c_xforms[i] if i < len(c_xforms) and c_xforms[i] else identity_chunk, a_xforms[i] if i < len(a_xforms) and a_xforms[i] else identity_dict)
        for i, tool in enumerate(tools)
    ]
    return StreamingPipelineTool(steps, name=name, description=description)


def streaming_parallel(*tools: BaseTool[BaseModel], stream_merge: StreamMerge | None = None, name: str | None = None, description: str | None = None) -> StreamingParallelTool:
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
    return StreamingParallelTool(list(tools), stream_merge=stream_merge, name=name, description=description)
