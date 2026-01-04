"""Tests for result streaming functionality.

Tests both the streaming decorator pattern and registry consumption.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from toolcase import (
    ResultStreamingFunctionTool,
    StreamChunk,
    StreamEvent,
    StreamEventKind,
    StreamResult,
    ToolRegistry,
    get_registry,
    reset_registry,
    sse_adapter,
    stream_complete,
    stream_error,
    stream_start,
    tool,
    ws_adapter,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def registry() -> ToolRegistry:
    """Fresh registry for each test."""
    reset_registry()
    return get_registry()


# ─────────────────────────────────────────────────────────────────────────────
# Streaming Tool Creation
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamingToolCreation:
    """Test that streaming tools are created correctly."""
    
    def test_streaming_tool_detected_as_async_generator(self) -> None:
        """streaming=True with async generator creates ResultStreamingFunctionTool."""
        @tool(description="Generate streaming output for testing", streaming=True)
        async def stream_gen(topic: str) -> AsyncIterator[str]:
            yield f"Starting {topic}..."
            yield f"Processing {topic}..."
            yield f"Done with {topic}!"
        
        assert isinstance(stream_gen, ResultStreamingFunctionTool)
        assert stream_gen.supports_result_streaming is True
        assert stream_gen.metadata.streaming is True
    
    def test_non_streaming_tool_is_function_tool(self) -> None:
        """Regular async function without streaming flag is FunctionTool."""
        @tool(description="Non-streaming async tool for testing")
        async def regular_async(query: str) -> str:
            return f"Result: {query}"
        
        assert not isinstance(regular_async, ResultStreamingFunctionTool)
        assert regular_async.supports_result_streaming is False


# ─────────────────────────────────────────────────────────────────────────────
# Streaming Execution
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamingExecution:
    """Test streaming tool execution."""
    
    @pytest.mark.asyncio
    async def test_stream_result_yields_chunks(self) -> None:
        """stream_result() yields individual chunks."""
        @tool(description="Generate chunks for stream test", streaming=True)
        async def chunky(count: int) -> AsyncIterator[str]:
            for i in range(count):
                yield f"chunk{i}"
        
        # Validate params and stream
        params = chunky.params_schema(count=3)
        chunks = [c async for c in chunky.stream_result(params)]
        
        assert chunks == ["chunk0", "chunk1", "chunk2"]
    
    @pytest.mark.asyncio
    async def test_sync_run_collects_all_chunks(self) -> None:
        """Calling streaming tool synchronously collects all output."""
        @tool(description="Collect all chunks in sync mode", streaming=True)
        async def multi_chunk(parts: int) -> AsyncIterator[str]:
            for i in range(parts):
                yield f"part{i} "
        
        # Direct call should collect all chunks
        result = multi_chunk(parts=3)
        assert result == "part0 part1 part2 "
    
    @pytest.mark.asyncio
    async def test_async_run_collects_all_chunks(self) -> None:
        """Async execution collects all chunks into final result."""
        @tool(description="Async collect all chunks test", streaming=True)
        async def multi_yield(n: int) -> AsyncIterator[str]:
            for i in range(n):
                yield f"[{i}]"
        
        params = multi_yield.params_schema(n=4)
        result = await multi_yield.arun(params)
        assert result == "[0][1][2][3]"


# ─────────────────────────────────────────────────────────────────────────────
# Registry Streaming
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistryStreaming:
    """Test registry.stream_execute() method."""
    
    @pytest.mark.asyncio
    async def test_stream_execute_yields_chunks(self, registry: ToolRegistry) -> None:
        """registry.stream_execute() yields chunks from streaming tools."""
        @tool(description="Registry streaming test tool", streaming=True)
        async def gen_stream(topic: str) -> AsyncIterator[str]:
            yield f"Hello "
            yield f"{topic}"
            yield "!"
        
        registry.register(gen_stream)
        
        chunks: list[str] = []
        async for chunk in registry.stream_execute("gen_stream", {"topic": "World"}):
            chunks.append(chunk)
        
        assert chunks == ["Hello ", "World", "!"]
    
    @pytest.mark.asyncio
    async def test_stream_execute_falls_back_for_non_streaming(
        self, registry: ToolRegistry
    ) -> None:
        """stream_execute on non-streaming tool yields complete result."""
        @tool(description="Non-streaming tool for fallback test")
        async def regular(msg: str) -> str:
            return f"Result: {msg}"
        
        registry.register(regular)
        
        chunks: list[str] = []
        async for chunk in registry.stream_execute("regular", {"msg": "test"}):
            chunks.append(chunk)
        
        # Should yield single complete result
        assert len(chunks) == 1
        assert chunks[0] == "Result: test"
    
    @pytest.mark.asyncio
    async def test_stream_execute_events_lifecycle(
        self, registry: ToolRegistry
    ) -> None:
        """stream_execute_events yields proper start/chunk/complete events."""
        @tool(description="Event lifecycle streaming test", streaming=True)
        async def event_stream(text: str) -> AsyncIterator[str]:
            yield "A"
            yield "B"
        
        registry.register(event_stream)
        
        events: list[StreamEvent] = []
        async for event in registry.stream_execute_events("event_stream", {"text": "x"}):
            events.append(event)
        
        assert len(events) == 4  # start, chunk, chunk, complete
        
        assert events[0].kind == StreamEventKind.START
        assert events[0].tool_name == "event_stream"
        
        assert events[1].kind == StreamEventKind.CHUNK
        assert events[1].data is not None
        assert events[1].data.content == "A"
        assert events[1].data.index == 0
        
        assert events[2].kind == StreamEventKind.CHUNK
        assert events[2].data is not None
        assert events[2].data.content == "B"
        assert events[2].data.index == 1
        
        assert events[3].kind == StreamEventKind.COMPLETE
        assert events[3].accumulated == "AB"
    
    @pytest.mark.asyncio
    async def test_stream_execute_collected(self, registry: ToolRegistry) -> None:
        """stream_execute_collected returns StreamResult with metadata."""
        @tool(description="Collected stream result test", streaming=True)
        async def collect_test(items: int) -> AsyncIterator[str]:
            for i in range(items):
                yield f"item{i}"
        
        registry.register(collect_test)
        
        result = await registry.stream_execute_collected("collect_test", {"items": 3})
        
        assert isinstance(result, StreamResult)
        assert result.value == "item0item1item2"
        assert result.chunks == 3
        assert result.tool_name == "collect_test"
        assert result.duration_ms > 0


# ─────────────────────────────────────────────────────────────────────────────
# Transport Adapters
# ─────────────────────────────────────────────────────────────────────────────

class TestTransportAdapters:
    """Test SSE and WebSocket adapters."""
    
    def test_sse_adapter_formats_event(self) -> None:
        """SSE adapter formats events correctly."""
        event = stream_start("test_tool")
        formatted = sse_adapter.format_event(event)
        
        assert "event: start" in formatted
        assert '"tool": "test_tool"' in formatted
        assert formatted.endswith("\n\n")
    
    def test_sse_adapter_formats_chunk(self) -> None:
        """SSE adapter formats chunks as events."""
        chunk = StreamChunk(content="Hello", index=0)
        formatted = sse_adapter.format_chunk(chunk, "my_tool")
        
        assert "event: chunk" in formatted
        assert '"content": "Hello"' in formatted
        assert '"index": 0' in formatted
    
    def test_ws_adapter_formats_as_json(self) -> None:
        """WebSocket adapter formats as JSON."""
        event = stream_complete("ws_test", "Full content")
        formatted = ws_adapter.format_event(event)
        
        # Should be valid JSON
        import json
        data = json.loads(formatted)
        
        assert data["kind"] == "complete"
        assert data["tool"] == "ws_test"
        assert data["accumulated"] == "Full content"


# ─────────────────────────────────────────────────────────────────────────────
# Error Handling
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamingErrors:
    """Test error handling in streaming."""
    
    @pytest.mark.asyncio
    async def test_stream_execute_not_found(self, registry: ToolRegistry) -> None:
        """stream_execute yields error for unknown tool."""
        chunks: list[str] = []
        async for chunk in registry.stream_execute("nonexistent", {}):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert "not found" in chunks[0].lower() or "Tool Error" in chunks[0]
    
    @pytest.mark.asyncio
    async def test_stream_execute_invalid_params(self, registry: ToolRegistry) -> None:
        """stream_execute yields error for invalid params."""
        @tool(description="Tool requiring valid params for error test", streaming=True)
        async def needs_param(required: str) -> AsyncIterator[str]:
            yield required
        
        registry.register(needs_param)
        
        chunks: list[str] = []
        # Missing required param
        async for chunk in registry.stream_execute("needs_param", {}):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert "Invalid parameters" in chunks[0] or "Tool Error" in chunks[0]


# ─────────────────────────────────────────────────────────────────────────────
# Integration: Simulated LLM Streaming
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMStyleStreaming:
    """Test realistic LLM-style streaming patterns."""
    
    @pytest.mark.asyncio
    async def test_simulated_llm_streaming(self, registry: ToolRegistry) -> None:
        """Simulate LLM-style token-by-token streaming."""
        @tool(description="Simulate LLM streaming output", streaming=True)
        async def llm_generate(prompt: str) -> AsyncIterator[str]:
            # Simulate LLM streaming tokens
            response = f"Based on your prompt '{prompt}', here is the response."
            words = response.split()
            for word in words:
                await asyncio.sleep(0.001)  # Simulate token delay
                yield word + " "
        
        registry.register(llm_generate)
        
        # Collect with timing
        import time
        start = time.time()
        chunks: list[str] = []
        
        async for chunk in registry.stream_execute(
            "llm_generate", {"prompt": "test"}
        ):
            chunks.append(chunk)
        
        elapsed = time.time() - start
        
        # Should have multiple chunks
        assert len(chunks) > 1
        
        # Reconstructed should be the full response
        full = "".join(chunks)
        assert "Based on your prompt" in full
        assert "test" in full
    
    @pytest.mark.asyncio
    async def test_websocket_delivery_simulation(self, registry: ToolRegistry) -> None:
        """Simulate WebSocket message delivery."""
        @tool(description="Generate content for websocket delivery", streaming=True)
        async def ws_content(topic: str) -> AsyncIterator[str]:
            yield f"Starting analysis of {topic}...\n"
            yield f"Key findings for {topic}:\n"
            yield "1. First point\n"
            yield "2. Second point\n"
            yield "Conclusion complete."
        
        registry.register(ws_content)
        
        # Simulate WebSocket send
        messages: list[str] = []
        async for event in registry.stream_execute_events(
            "ws_content", {"topic": "AI trends"}
        ):
            # Would be: await websocket.send(event.to_json())
            messages.append(event.to_json())
        
        # Should have start, 5 chunks, and complete
        assert len(messages) == 7
        
        # All should be valid JSON
        import json
        for msg in messages:
            data = json.loads(msg)
            assert "kind" in data
            assert "tool" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
