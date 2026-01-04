"""Tests for streaming middleware integration."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel, Field

from toolcase.foundation.core import BaseTool, ToolMetadata
from toolcase.foundation.registry import get_registry, reset_registry
from toolcase.io.streaming import StreamChunk
from toolcase.runtime.middleware import (
    Context,
    LoggingMiddleware,
    Middleware,
    Next,
    StreamLoggingMiddleware,
    StreamMetricsMiddleware,
    StreamMiddleware,
    StreamingAdapter,
    StreamingChain,
    compose_streaming,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────────────────────────────────────

class EchoParams(BaseModel):
    message: str = Field(..., description="Message to echo")


class StreamingEchoTool(BaseTool[EchoParams]):
    """Streaming tool that yields words as chunks."""
    
    metadata = ToolMetadata(
        name="streaming_echo",
        description="Echo message as streaming chunks",
        streaming=True,
    )
    params_schema = EchoParams
    
    @property
    def supports_result_streaming(self) -> bool:
        return True
    
    async def _async_run(self, params: EchoParams) -> str:
        return params.message
    
    async def stream_result(self, params: EchoParams) -> AsyncIterator[str]:
        for word in params.message.split():
            yield word + " "


class NonStreamingTool(BaseTool[EchoParams]):
    """Non-streaming tool for fallback testing."""
    
    metadata = ToolMetadata(
        name="non_streaming",
        description="Non-streaming echo tool",
        streaming=False,
    )
    params_schema = EchoParams
    
    async def _async_run(self, params: EchoParams) -> str:
        return params.message


@dataclass(slots=True)
class ChunkRecorderMiddleware:
    """Test middleware that records all chunks for assertions."""
    
    chunks: list[StreamChunk] = field(default_factory=list)
    on_start_called: bool = False
    on_complete_called: bool = False
    accumulated: str = ""
    
    async def on_start(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
    ) -> None:
        self.on_start_called = True
        self.chunks.clear()
    
    async def on_chunk(
        self,
        chunk: StreamChunk,
        ctx: Context,
    ) -> StreamChunk:
        self.chunks.append(chunk)
        return chunk
    
    async def on_complete(
        self,
        accumulated: str,
        ctx: Context,
    ) -> None:
        self.on_complete_called = True
        self.accumulated = accumulated


@dataclass(slots=True)
class ChunkTransformMiddleware:
    """Test middleware that transforms chunks."""
    
    prefix: str = "[transformed] "
    
    async def on_chunk(
        self,
        chunk: StreamChunk,
        ctx: Context,
    ) -> StreamChunk:
        return StreamChunk(
            content=self.prefix + chunk.content,
            index=chunk.index,
            timestamp=chunk.timestamp,
            metadata=chunk.metadata,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compose_streaming
# ─────────────────────────────────────────────────────────────────────────────

class TestComposeStreaming:
    """Test streaming chain composition."""
    
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        reset_registry()
    
    async def test_empty_middleware_chain(self) -> None:
        """Chain with no middleware yields chunks directly."""
        chain = compose_streaming([])
        tool = StreamingEchoTool()
        params = EchoParams(message="hello world")
        ctx = Context()
        
        chunks = [c async for c in chain(tool, params, ctx)]
        
        assert len(chunks) == 2
        assert chunks[0].content == "hello "
        assert chunks[1].content == "world "
    
    async def test_single_stream_middleware(self) -> None:
        """Single streaming middleware receives hooks."""
        recorder = ChunkRecorderMiddleware()
        chain = compose_streaming([recorder])
        tool = StreamingEchoTool()
        params = EchoParams(message="a b c")
        ctx = Context()
        
        chunks = [c async for c in chain(tool, params, ctx)]
        
        assert recorder.on_start_called
        assert recorder.on_complete_called
        assert len(recorder.chunks) == 3
        assert recorder.accumulated == "a b c "
    
    async def test_multiple_middleware_order(self) -> None:
        """Middleware runs in correct order (outermost first)."""
        order: list[str] = []
        
        @dataclass
        class OrderTracker:
            name: str
            order_list: list[str] = field(default_factory=list)
            
            async def on_start(self, tool: BaseTool[BaseModel], params: BaseModel, ctx: Context) -> None:
                self.order_list.append(f"{self.name}:start")
            
            async def on_complete(self, accumulated: str, ctx: Context) -> None:
                self.order_list.append(f"{self.name}:complete")
        
        first = OrderTracker("first", order)
        second = OrderTracker("second", order)
        chain = compose_streaming([first, second])
        
        tool = StreamingEchoTool()
        params = EchoParams(message="test")
        ctx = Context()
        
        _ = [c async for c in chain(tool, params, ctx)]
        
        # First middleware's start runs first, complete runs first (wrapping)
        assert order == ["first:start", "second:start", "second:complete", "first:complete"]
    
    async def test_chunk_transformation(self) -> None:
        """Middleware can transform chunks."""
        transformer = ChunkTransformMiddleware(prefix="[X] ")
        chain = compose_streaming([transformer])
        tool = StreamingEchoTool()
        params = EchoParams(message="hello")
        ctx = Context()
        
        chunks = [c async for c in chain(tool, params, ctx)]
        
        assert len(chunks) == 1
        assert chunks[0].content == "[X] hello "
    
    async def test_regular_middleware_adapted(self) -> None:
        """Regular Middleware is auto-adapted via StreamingAdapter."""
        regular_mw = LoggingMiddleware(log_params=False)
        chain = compose_streaming([regular_mw])
        
        # Check that it's wrapped
        assert isinstance(chain._middleware[0], StreamingAdapter)
        
        tool = StreamingEchoTool()
        params = EchoParams(message="test")
        ctx = Context()
        
        # Should work without error
        chunks = [c async for c in chain(tool, params, ctx)]
        assert len(chunks) == 1


class TestStreamingAdapter:
    """Test StreamingAdapter for regular middleware."""
    
    async def test_adapter_wraps_regular_middleware(self) -> None:
        """Adapter correctly wraps Middleware protocol."""
        regular = LoggingMiddleware()
        adapter = StreamingAdapter(middleware=regular)
        
        assert adapter.middleware is regular
    
    async def test_adapter_has_lifecycle_hooks(self) -> None:
        """Adapter implements StreamMiddleware lifecycle."""
        regular = LoggingMiddleware()
        adapter = StreamingAdapter(middleware=regular)
        
        ctx = Context()
        # on_start should set marker
        await adapter.on_start(StreamingEchoTool(), EchoParams(message=""), ctx)
        assert ctx.get("_stream_started") is True


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Registry Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistryStreamingMiddleware:
    """Test streaming middleware through registry.stream_execute()."""
    
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        reset_registry()
        self.registry = get_registry()
        self.registry.register(StreamingEchoTool())
        self.registry.register(NonStreamingTool())
    
    async def test_stream_execute_with_stream_middleware(self) -> None:
        """stream_execute uses streaming middleware chain."""
        recorder = ChunkRecorderMiddleware()
        self.registry.use(recorder)
        
        chunks = []
        async for chunk in self.registry.stream_execute("streaming_echo", {"message": "hello world"}):
            chunks.append(chunk)
        
        assert recorder.on_start_called
        assert recorder.on_complete_called
        assert len(recorder.chunks) == 2
        assert "".join(chunks) == "hello world "
    
    async def test_stream_execute_with_regular_middleware(self) -> None:
        """Regular middleware works with streaming."""
        self.registry.use(LoggingMiddleware())
        
        chunks = []
        async for chunk in self.registry.stream_execute("streaming_echo", {"message": "test"}):
            chunks.append(chunk)
        
        assert chunks == ["test "]
    
    async def test_stream_execute_non_streaming_tool(self) -> None:
        """Non-streaming tools yield single chunk through middleware."""
        recorder = ChunkRecorderMiddleware()
        self.registry.use(recorder)
        
        chunks = []
        async for chunk in self.registry.stream_execute("non_streaming", {"message": "complete result"}):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert chunks[0] == "complete result"
    
    async def test_mixed_middleware_chain(self) -> None:
        """Mixed regular + streaming middleware works."""
        recorder = ChunkRecorderMiddleware()
        self.registry.use(LoggingMiddleware())
        self.registry.use(recorder)
        self.registry.use(StreamLoggingMiddleware())
        
        chunks = []
        async for chunk in self.registry.stream_execute("streaming_echo", {"message": "a b"}):
            chunks.append(chunk)
        
        assert recorder.on_start_called
        assert len(recorder.chunks) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Built-in Streaming Middleware
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamLoggingMiddleware:
    """Test StreamLoggingMiddleware."""
    
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        reset_registry()
        self.registry = get_registry()
        self.registry.register(StreamingEchoTool())
    
    async def test_logs_stream_metrics(self) -> None:
        """Middleware stores metrics in context."""
        mw = StreamLoggingMiddleware()
        ctx = Context()
        
        tool = StreamingEchoTool()
        params = EchoParams(message="hello world test")
        
        chain = compose_streaming([mw])
        _ = [c async for c in chain(tool, params, ctx)]
        
        assert ctx.get("stream_chunks") == 3
        assert ctx.get("stream_bytes") == 17  # "hello world test " = 17 bytes
        assert ctx.get("stream_duration_ms") is not None


class TestStreamMetricsMiddleware:
    """Test StreamMetricsMiddleware."""
    
    async def test_emits_metrics(self) -> None:
        """Middleware calls backend with metrics."""
        recorded: dict[str, list[tuple[str, int | float, dict[str, str] | None]]] = {
            "increment": [],
            "timing": [],
        }
        
        @dataclass
        class RecordingBackend:
            def increment(self, metric: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
                recorded["increment"].append((metric, value, tags))
            
            def timing(self, metric: str, value_ms: float, tags: dict[str, str] | None = None) -> None:
                recorded["timing"].append((metric, value_ms, tags))
        
        mw = StreamMetricsMiddleware(backend=RecordingBackend())
        chain = compose_streaming([StreamLoggingMiddleware(), mw])  # Need logging to set chunk counts
        
        tool = StreamingEchoTool()
        params = EchoParams(message="a b")
        ctx = Context()
        
        _ = [c async for c in chain(tool, params, ctx)]
        
        # Check metrics emitted
        assert any("started" in m[0] for m in recorded["increment"])
        assert any("chunks" in m[0] for m in recorded["increment"])
        assert any("duration_ms" in m[0] for m in recorded["timing"])


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Error Handling
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamingErrorHandling:
    """Test error handling in streaming middleware."""
    
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        reset_registry()
    
    async def test_on_error_called_on_exception(self) -> None:
        """on_error hook is called when stream fails."""
        
        class FailingTool(BaseTool[EchoParams]):
            metadata = ToolMetadata(
                name="failing_stream",
                description="Tool that fails during streaming",
                streaming=True,
            )
            params_schema = EchoParams
            
            @property
            def supports_result_streaming(self) -> bool:
                return True
            
            async def _async_run(self, params: EchoParams) -> str:
                return params.message
            
            async def stream_result(self, params: EchoParams) -> AsyncIterator[str]:
                yield "first "
                raise ValueError("Stream failure")
        
        error_captured: list[Exception] = []
        
        @dataclass
        class ErrorRecorder:
            async def on_error(self, error: Exception, ctx: Context) -> None:
                error_captured.append(error)
        
        chain = compose_streaming([ErrorRecorder()])
        tool = FailingTool()
        params = EchoParams(message="test")
        ctx = Context()
        
        with pytest.raises(ValueError, match="Stream failure"):
            _ = [c async for c in chain(tool, params, ctx)]
        
        assert len(error_captured) == 1
        assert isinstance(error_captured[0], ValueError)
    
    async def test_tool_not_found_yields_error(self) -> None:
        """stream_execute yields error for missing tool."""
        registry = get_registry()
        
        chunks = []
        async for chunk in registry.stream_execute("nonexistent", {"message": "test"}):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert "Tool Error" in chunks[0]
        assert "not found" in chunks[0]
    
    async def test_validation_error_yields_error(self) -> None:
        """stream_execute yields error for invalid params."""
        registry = get_registry()
        registry.register(StreamingEchoTool())
        
        chunks = []
        async for chunk in registry.stream_execute("streaming_echo", {}):  # Missing required field
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert "Tool Error" in chunks[0]
