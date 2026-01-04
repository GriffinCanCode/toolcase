"""Tests for streaming pipeline propagation.

Tests StreamingPipelineTool and StreamingParallelTool for
async generator propagation through tool compositions.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel, Field

from toolcase import (
    BaseTool,
    StreamChunk,
    StreamEvent,
    StreamEventKind,
    StreamStep,
    StreamingParallelTool,
    StreamingPipelineTool,
    ToolMetadata,
    interleave_streams,
    streaming_parallel,
    streaming_pipeline,
    tool,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures: Streaming Tools
# ─────────────────────────────────────────────────────────────────────────────


class InputParams(BaseModel):
    """Simple input params."""
    input: str = Field(default="", description="Input text")


class ChunkyTool(BaseTool[InputParams]):
    """Tool that streams output in chunks."""
    
    metadata = ToolMetadata(
        name="chunky_tool",
        description="Streams output in word chunks",
        streaming=True,
    )
    params_schema = InputParams
    cache_enabled = False
    
    def _run(self, params: InputParams) -> str:
        return f"Processed: {params.input}"
    
    @property
    def supports_result_streaming(self) -> bool:
        return True
    
    async def stream_result(self, params: InputParams) -> AsyncIterator[str]:
        words = f"Processed: {params.input}".split()
        for word in words:
            await asyncio.sleep(0.001)
            yield word + " "


class UppercaseTool(BaseTool[InputParams]):
    """Tool that uppercases and streams."""
    
    metadata = ToolMetadata(
        name="uppercase_tool",
        description="Uppercases and streams input",
        streaming=True,
    )
    params_schema = InputParams
    cache_enabled = False
    
    def _run(self, params: InputParams) -> str:
        return params.input.upper()
    
    @property
    def supports_result_streaming(self) -> bool:
        return True
    
    async def stream_result(self, params: InputParams) -> AsyncIterator[str]:
        for char in params.input.upper():
            yield char


class NonStreamingTool(BaseTool[InputParams]):
    """Tool without streaming support."""
    
    metadata = ToolMetadata(
        name="non_streaming",
        description="Returns complete result without streaming",
    )
    params_schema = InputParams
    cache_enabled = False
    
    def _run(self, params: InputParams) -> str:
        return f"[{params.input}]"


# ─────────────────────────────────────────────────────────────────────────────
# StreamingPipelineTool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestStreamingPipelineTool:
    """Tests for StreamingPipelineTool."""
    
    @pytest.mark.asyncio
    async def test_basic_streaming_pipeline(self) -> None:
        """Pipeline streams chunks from final step."""
        pipe = StreamingPipelineTool(
            steps=[
                StreamStep(NonStreamingTool()),
                StreamStep(ChunkyTool()),
            ],
            name="test_pipe",
        )
        
        params = pipe.params_schema(input={"input": "hello"})
        chunks: list[str] = []
        async for chunk in pipe.stream_result(params):
            chunks.append(chunk)
        
        # Should have multiple chunks from ChunkyTool
        assert len(chunks) > 1
        result = "".join(chunks)
        assert "Processed:" in result
        assert "[hello]" in result  # From NonStreamingTool wrapping
    
    @pytest.mark.asyncio
    async def test_chunk_transform_applied(self) -> None:
        """Chunk transforms modify chunks in flight."""
        pipe = StreamingPipelineTool(
            steps=[
                StreamStep(ChunkyTool(), chunk_transform=str.upper),
            ],
        )
        
        params = pipe.params_schema(input={"input": "test"})
        chunks = [c async for c in pipe.stream_result(params)]
        
        # All chunks should be uppercase
        for chunk in chunks:
            assert chunk == chunk.upper()
    
    @pytest.mark.asyncio
    async def test_accumulate_transform_between_steps(self) -> None:
        """Accumulated output transforms for next step params."""
        
        def custom_transform(accumulated: str) -> dict[str, object]:
            return {"input": f"transformed:{accumulated.strip()}"}
        
        pipe = StreamingPipelineTool(
            steps=[
                StreamStep(ChunkyTool(), accumulate_transform=custom_transform),
                StreamStep(UppercaseTool()),
            ],
        )
        
        params = pipe.params_schema(input={"input": "data"})
        result = "".join([c async for c in pipe.stream_result(params)])
        
        # Should see uppercase version of transformed input
        assert "TRANSFORMED:" in result
    
    @pytest.mark.asyncio
    async def test_collect_to_final_result(self) -> None:
        """Sync run collects all chunks."""
        pipe = streaming_pipeline(ChunkyTool())
        
        # Use async run which collects
        params = pipe.params_schema(input={"input": "hello world"})
        result = await pipe.arun(params)
        
        assert "Processed:" in result
        assert "hello" in result
        assert "world" in result
    
    @pytest.mark.asyncio
    async def test_stream_result_events_lifecycle(self) -> None:
        """stream_result_events yields proper lifecycle events."""
        pipe = streaming_pipeline(ChunkyTool())
        
        params = pipe.params_schema(input={"input": "test"})
        events: list[StreamEvent] = []
        
        async for event in pipe.stream_result_events(params):
            events.append(event)
        
        # Should have START, chunks, COMPLETE
        assert events[0].kind == StreamEventKind.START
        assert events[-1].kind == StreamEventKind.COMPLETE
        assert events[-1].accumulated is not None
        
        # Middle events should be chunks
        chunk_events = [e for e in events if e.kind == StreamEventKind.CHUNK]
        assert len(chunk_events) > 0
        assert all(e.data is not None for e in chunk_events)
    
    @pytest.mark.asyncio
    async def test_pipeline_chaining_operator(self) -> None:
        """>> operator chains streaming pipelines."""
        pipe1 = StreamingPipelineTool(steps=[StreamStep(NonStreamingTool())])
        pipe2 = pipe1 >> ChunkyTool()
        
        assert len(pipe2.steps) == 2
        assert pipe2.metadata.streaming is True
    
    @pytest.mark.asyncio
    async def test_validation_error_yields_error_chunk(self) -> None:
        """Invalid params yield error chunk instead of raising."""
        
        class StrictParams(BaseModel):
            required_field: str
        
        class StrictTool(BaseTool[StrictParams]):
            metadata = ToolMetadata(
                name="strict_tool",
                description="Requires specific params",
            )
            params_schema = StrictParams
            cache_enabled = False
            
            def _run(self, params: StrictParams) -> str:
                return params.required_field
        
        pipe = streaming_pipeline(StrictTool())
        params = pipe.params_schema(input={})  # Missing required_field
        
        chunks = [c async for c in pipe.stream_result(params)]
        assert len(chunks) == 1
        assert "[Pipeline Error]" in chunks[0]


# ─────────────────────────────────────────────────────────────────────────────
# StreamingParallelTool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestStreamingParallelTool:
    """Tests for StreamingParallelTool."""
    
    @pytest.mark.asyncio
    async def test_basic_streaming_parallel(self) -> None:
        """Parallel streams chunks from all tools interleaved."""
        multi = StreamingParallelTool(
            tools=[ChunkyTool(), ChunkyTool()],
            name="test_parallel",
        )
        
        params = multi.params_schema(input={"input": "test"})
        chunks = [c async for c in multi.stream_result(params)]
        
        # Should have chunks from both tools
        assert len(chunks) > 2
    
    @pytest.mark.asyncio
    async def test_interleave_streams_helper(self) -> None:
        """interleave_streams merges multiple async generators."""
        
        async def stream_a() -> AsyncIterator[str]:
            for c in "ABC":
                await asyncio.sleep(0.001)
                yield c
        
        async def stream_b() -> AsyncIterator[str]:
            for c in "123":
                await asyncio.sleep(0.001)
                yield c
        
        chunks = [c async for c in interleave_streams([stream_a(), stream_b()])]
        
        # Should have all chunks from both streams
        assert set(chunks) == {"A", "B", "C", "1", "2", "3"}
        assert len(chunks) == 6
    
    @pytest.mark.asyncio
    async def test_custom_stream_merge(self) -> None:
        """Custom stream merge function is used."""
        
        async def prefix_merge(streams: list[AsyncIterator[str]]) -> AsyncIterator[str]:
            """Prefix each chunk with stream index."""
            for i, stream in enumerate(streams):
                async for chunk in stream:
                    yield f"[{i}]{chunk}"
        
        multi = streaming_parallel(
            ChunkyTool(),
            ChunkyTool(),
            stream_merge=prefix_merge,
        )
        
        params = multi.params_schema(input={"input": "x"})
        chunks = [c async for c in multi.stream_result(params)]
        
        # Chunks should be prefixed
        assert all(c.startswith("[0]") or c.startswith("[1]") for c in chunks)
    
    @pytest.mark.asyncio
    async def test_parallel_collect_result(self) -> None:
        """Sync/async run collects all parallel output."""
        multi = streaming_parallel(ChunkyTool(), UppercaseTool())
        
        params = multi.params_schema(input={"input": "data"})
        result = await multi.arun(params)
        
        # Should contain output from both tools
        assert "Processed:" in result or "DATA" in result
    
    @pytest.mark.asyncio
    async def test_parallel_stream_events(self) -> None:
        """stream_result_events has proper lifecycle."""
        multi = streaming_parallel(ChunkyTool())
        
        params = multi.params_schema(input={"input": "test"})
        events = [e async for e in multi.stream_result_events(params)]
        
        assert events[0].kind == StreamEventKind.START
        assert events[-1].kind == StreamEventKind.COMPLETE
    
    @pytest.mark.asyncio
    async def test_parallel_validation_error_in_stream(self) -> None:
        """Invalid params yield error chunk per tool."""
        
        class RequiredParams(BaseModel):
            required: str
        
        class RequiredTool(BaseTool[RequiredParams]):
            metadata = ToolMetadata(
                name="required_tool",
                description="Needs required param",
            )
            params_schema = RequiredParams
            cache_enabled = False
            
            def _run(self, params: RequiredParams) -> str:
                return params.required
        
        multi = streaming_parallel(RequiredTool())
        params = multi.params_schema(input={})
        
        chunks = [c async for c in multi.stream_result(params)]
        assert any("[Error" in c for c in chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Factory Function Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFactoryFunctions:
    """Tests for streaming_pipeline and streaming_parallel factories."""
    
    def test_streaming_pipeline_factory(self) -> None:
        """streaming_pipeline creates StreamingPipelineTool."""
        pipe = streaming_pipeline(ChunkyTool(), UppercaseTool())
        
        assert isinstance(pipe, StreamingPipelineTool)
        assert len(pipe.steps) == 2
        assert pipe.metadata.streaming is True
    
    def test_streaming_pipeline_with_transforms(self) -> None:
        """streaming_pipeline accepts chunk and accumulate transforms."""
        pipe = streaming_pipeline(
            ChunkyTool(),
            UppercaseTool(),
            chunk_transforms=[str.strip, None],
            accumulate_transforms=[lambda s: {"input": s.upper()}, None],
        )
        
        assert pipe.steps[0].chunk_transform is str.strip
        assert pipe.steps[1].chunk_transform("x") == "x"  # identity_chunk
    
    def test_streaming_pipeline_requires_tools(self) -> None:
        """streaming_pipeline raises on empty tools."""
        with pytest.raises(ValueError, match="requires at least one tool"):
            streaming_pipeline()
    
    def test_streaming_parallel_factory(self) -> None:
        """streaming_parallel creates StreamingParallelTool."""
        multi = streaming_parallel(ChunkyTool(), UppercaseTool())
        
        assert isinstance(multi, StreamingParallelTool)
        assert len(multi.tools) == 2
        assert multi.metadata.streaming is True
    
    def test_streaming_parallel_requires_tools(self) -> None:
        """streaming_parallel raises on empty tools."""
        with pytest.raises(ValueError, match="requires at least one tool"):
            streaming_parallel()


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests: Decorator-based Streaming Tools
# ─────────────────────────────────────────────────────────────────────────────


class TestDecoratorIntegration:
    """Test streaming pipelines with @tool decorated functions."""
    
    @pytest.mark.asyncio
    async def test_streaming_decorator_in_pipeline(self) -> None:
        """@tool(streaming=True) works in streaming pipelines."""
        
        @tool(description="Yields words one by one", streaming=True)
        async def word_streamer(input: str) -> AsyncIterator[str]:
            for word in input.split():
                yield word + " "
        
        @tool(description="Wraps input in brackets")
        def bracket_wrapper(input: str) -> str:
            return f"[{input}]"
        
        pipe = streaming_pipeline(bracket_wrapper, word_streamer)
        params = pipe.params_schema(input={"input": "hello world"})
        
        chunks = [c async for c in pipe.stream_result(params)]
        result = "".join(chunks)
        
        # Should have bracketed input split into words
        assert "[hello" in result or "hello]" in result
    
    @pytest.mark.asyncio
    async def test_llm_style_streaming_simulation(self) -> None:
        """Simulate LLM-style token streaming through pipeline."""
        
        @tool(description="Simulates LLM token streaming", streaming=True)
        async def llm_simulator(input: str) -> AsyncIterator[str]:
            response = f"Response to: {input}"
            for char in response:
                await asyncio.sleep(0.0001)
                yield char
        
        @tool(description="Prepares prompt")
        def prompt_builder(input: str) -> str:
            return f"[PROMPT] {input}"
        
        pipe = streaming_pipeline(prompt_builder, llm_simulator)
        params = pipe.params_schema(input={"input": "test query"})
        
        chunks: list[str] = []
        async for chunk in pipe.stream_result(params):
            chunks.append(chunk)
        
        # Should stream character by character
        assert len(chunks) > 10
        assert "".join(chunks).startswith("Response to:")


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests for streaming pipelines."""
    
    @pytest.mark.asyncio
    async def test_single_step_pipeline(self) -> None:
        """Single step pipeline works correctly."""
        pipe = streaming_pipeline(ChunkyTool())
        params = pipe.params_schema(input={"input": "solo"})
        
        result = "".join([c async for c in pipe.stream_result(params)])
        assert "Processed:" in result
        assert "solo" in result
    
    @pytest.mark.asyncio
    async def test_empty_stream_handling(self) -> None:
        """Handles tools that yield nothing gracefully."""
        
        class EmptyStreamer(BaseTool[InputParams]):
            metadata = ToolMetadata(
                name="empty_streamer",
                description="Yields nothing",
                streaming=True,
            )
            params_schema = InputParams
            cache_enabled = False
            
            def _run(self, params: InputParams) -> str:
                return ""
            
            @property
            def supports_result_streaming(self) -> bool:
                return True
            
            async def stream_result(self, params: InputParams) -> AsyncIterator[str]:
                return
                yield  # noqa: B901 - empty generator
        
        pipe = streaming_pipeline(EmptyStreamer())
        params = pipe.params_schema(input={"input": "x"})
        
        chunks = [c async for c in pipe.stream_result(params)]
        assert chunks == []
    
    @pytest.mark.asyncio
    async def test_mixed_streaming_non_streaming(self) -> None:
        """Pipeline handles mix of streaming and non-streaming tools."""
        pipe = streaming_pipeline(
            NonStreamingTool(),  # Not streaming
            ChunkyTool(),        # Streaming
            NonStreamingTool(),  # Not streaming (but uses base stream_result)
        )
        
        params = pipe.params_schema(input={"input": "mix"})
        # Should complete without error
        result = await pipe.arun(params)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
