"""Streaming middleware for chunk-aware execution hooks.

Extends the middleware system to support streaming tools with lifecycle
hooks: on_start, on_chunk, on_complete, on_error.

Regular Middleware is automatically adapted to streaming context via
StreamingAdapter - running "before" logic on start, "after" on complete.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

from toolcase.foundation.core.decorator import clear_injected_deps, set_injected_deps
from toolcase.foundation.errors import ToolError
from toolcase.io.streaming import StreamChunk

from .middleware import Context, Middleware

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool

# Type alias for streaming continuation
StreamNext = AsyncIterator[StreamChunk]


@runtime_checkable
class StreamMiddleware(Protocol):
    """Protocol for streaming-aware middleware.
    
    Provides lifecycle hooks for stream observation and transformation.
    All methods are optional - implement only what you need.
    
    Example:
        >>> class ChunkLoggerMiddleware:
        ...     async def on_start(self, tool, params, ctx):
        ...         ctx["chunk_count"] = 0
        ...     
        ...     async def on_chunk(self, chunk, ctx):
        ...         ctx["chunk_count"] += 1
        ...         return chunk  # Pass through
        ...     
        ...     async def on_complete(self, accumulated, ctx):
        ...         print(f"Streamed {ctx['chunk_count']} chunks")
    """
    
    async def on_start(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
    ) -> None:
        """Called before streaming begins. Use for setup/logging."""
        ...
    
    async def on_chunk(
        self,
        chunk: StreamChunk,
        ctx: Context,
    ) -> StreamChunk:
        """Called for each chunk. Return chunk (possibly transformed)."""
        ...
    
    async def on_complete(
        self,
        accumulated: str,
        ctx: Context,
    ) -> None:
        """Called when stream completes successfully."""
        ...
    
    async def on_error(
        self,
        error: Exception,
        ctx: Context,
    ) -> None:
        """Called when stream encounters an error."""
        ...


def _has_hook(obj: object, name: str) -> bool:
    """Check if object has an implemented hook method.
    
    Checks the object's class hierarchy (excluding Protocol base) for the method.
    """
    # Check if method exists and is callable
    attr = getattr(obj, name, None)
    if attr is None or not callable(attr):
        return False
    
    # Check if it's defined in the object's class (not just inherited from Protocol)
    obj_cls = type(obj)
    for cls in obj_cls.__mro__:
        # Skip Protocol classes and object
        if cls.__name__ in ("StreamMiddleware", "Protocol", "object"):
            continue
        if name in cls.__dict__:
            return True
    return False


@dataclass(slots=True)
class StreamingAdapter:
    """Adapts regular Middleware for streaming context.
    
    Executes the middleware's before-logic on stream start and
    after-logic on stream complete/error. This allows existing
    middleware like LoggingMiddleware to work with streaming.
    
    The adapted middleware receives a synthetic Next that accumulates
    chunks and returns the final result, preserving the original contract.
    """
    
    middleware: Middleware
    
    async def on_start(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
    ) -> None:
        """Mark stream start in context."""
        ctx["_stream_started"] = True
    
    async def on_complete(
        self,
        accumulated: str,
        ctx: Context,
    ) -> None:
        """No-op - actual middleware logic runs in wrap_stream."""
        pass
    
    async def on_error(
        self,
        error: Exception,
        ctx: Context,
    ) -> None:
        """No-op - errors propagate naturally."""
        pass


async def _stream_through_middleware(
    middleware: StreamMiddleware | StreamingAdapter,
    tool: BaseTool[BaseModel],
    params: BaseModel,
    ctx: Context,
    source: AsyncIterator[StreamChunk],
) -> AsyncIterator[StreamChunk]:
    """Stream chunks through a single middleware's hooks.
    
    Calls on_start before yielding, on_chunk for each chunk,
    and on_complete/on_error on termination.
    """
    accumulated: list[str] = []
    
    # on_start
    if _has_hook(middleware, "on_start"):
        await middleware.on_start(tool, params, ctx)
    
    try:
        async for chunk in source:
            # on_chunk - allow transformation
            if _has_hook(middleware, "on_chunk"):
                chunk = await middleware.on_chunk(chunk, ctx)
            accumulated.append(chunk.content)
            yield chunk
        
        # on_complete
        if _has_hook(middleware, "on_complete"):
            await middleware.on_complete("".join(accumulated), ctx)
    
    except Exception as e:
        # on_error
        if _has_hook(middleware, "on_error"):
            await middleware.on_error(e, ctx)
        raise


def _is_stream_middleware(mw: object) -> bool:
    """Check if object implements StreamMiddleware hooks (duck typing).
    
    We can't rely on isinstance(mw, StreamMiddleware) because Protocol
    runtime checking doesn't work well with dataclasses that don't
    explicitly inherit from the protocol.
    """
    # Has at least one streaming hook method
    return any(_has_hook(mw, name) for name in ("on_start", "on_chunk", "on_complete", "on_error"))


def _to_stream_middleware(mw: Middleware | StreamMiddleware) -> StreamMiddleware | StreamingAdapter:
    """Convert regular Middleware to streaming-compatible form.
    
    Objects with streaming hooks (on_start, on_chunk, etc.) are used directly.
    Regular Middleware objects are wrapped in StreamingAdapter.
    """
    if _is_stream_middleware(mw):
        return mw  # type: ignore[return-value]
    return StreamingAdapter(mw)  # type: ignore[arg-type]


async def _base_stream(
    tool: BaseTool[BaseModel],
    params: BaseModel,
    ctx: Context,
) -> AsyncIterator[StreamChunk]:
    """Base streaming executor - yields chunks from tool.stream_result().
    
    Errors are propagated to middleware on_error hooks rather than
    being caught here. This allows proper error handling through the chain.
    """
    from typing import cast as typing_cast
    
    from toolcase.foundation.errors import JsonDict
    
    # Set injected dependencies from context if present
    injected = ctx.get("injected")
    if injected and isinstance(injected, dict):
        set_injected_deps(typing_cast("JsonDict", injected))
    
    try:
        index = 0
        if hasattr(tool, "supports_result_streaming") and tool.supports_result_streaming:
            async for content in tool.stream_result(params):
                yield StreamChunk(content=content, index=index)
                index += 1
        else:
            # Non-streaming tool: yield single chunk with complete result
            result = await tool.arun(params)
            yield StreamChunk(content=result, index=0)
    finally:
        clear_injected_deps()


def compose_streaming(
    middleware: Sequence[Middleware | StreamMiddleware],
) -> "StreamingChain":
    """Compose middleware into a streaming execution chain.
    
    Creates a chain that streams chunks through each middleware's hooks.
    Regular Middleware is auto-adapted via StreamingAdapter.
    
    Args:
        middleware: Ordered list (first = outermost, runs first)
    
    Returns:
        StreamingChain callable: (tool, params, ctx) -> AsyncIterator[StreamChunk]
    
    Example:
        >>> chain = compose_streaming([LoggingMiddleware(), MetricsMiddleware()])
        >>> async for chunk in chain(tool, params, ctx):
        ...     print(chunk.content, end="")
    """
    return StreamingChain([_to_stream_middleware(mw) for mw in middleware])


@dataclass(slots=True)
class StreamingChain:
    """Composed streaming middleware chain.
    
    Implements the chain as nested async generators, with each middleware
    wrapping the output of the next.
    """
    
    _middleware: list[StreamMiddleware | StreamingAdapter]
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
    ) -> AsyncIterator[StreamChunk]:
        """Execute streaming chain, yielding chunks."""
        # Start with base executor
        stream: AsyncIterator[StreamChunk] = _base_stream(tool, params, ctx)
        
        # Wrap from innermost to outermost
        for mw in reversed(self._middleware):
            stream = _stream_through_middleware(mw, tool, params, ctx, stream)
        
        # Yield through composed chain
        async for chunk in stream:
            yield chunk


# ─────────────────────────────────────────────────────────────────────────────
# Streaming-Aware Middleware Implementations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class StreamLoggingMiddleware:
    """Log streaming execution with chunk-level observability.
    
    Logs stream start, chunk count, total bytes, and completion status.
    Stores metrics in context for downstream middleware.
    
    Example:
        >>> registry.use(StreamLoggingMiddleware())
    """
    
    import logging
    log: logging.Logger | None = None
    log_chunk_sizes: bool = False
    
    def __post_init__(self) -> None:
        import logging
        if self.log is None:
            self.log = logging.getLogger("toolcase.streaming")
    
    async def on_start(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
    ) -> None:
        import time
        ctx["_stream_start"] = time.perf_counter()
        ctx["_chunk_count"] = 0
        ctx["_total_bytes"] = 0
        self.log.info(f"[{tool.metadata.name}] Stream started")  # type: ignore[union-attr]
    
    async def on_chunk(
        self,
        chunk: StreamChunk,
        ctx: Context,
    ) -> StreamChunk:
        ctx["_chunk_count"] = ctx.get("_chunk_count", 0) + 1  # type: ignore[operator]
        ctx["_total_bytes"] = ctx.get("_total_bytes", 0) + len(chunk.content)  # type: ignore[operator]
        if self.log_chunk_sizes:
            self.log.debug(f"Chunk {chunk.index}: {len(chunk.content)} bytes")  # type: ignore[union-attr]
        return chunk
    
    async def on_complete(
        self,
        accumulated: str,
        ctx: Context,
    ) -> None:
        import time
        start = ctx.get("_stream_start", time.perf_counter())
        duration_ms = (time.perf_counter() - float(start)) * 1000  # type: ignore[arg-type]
        chunks = ctx.get("_chunk_count", 0)
        total_bytes = ctx.get("_total_bytes", 0)
        
        ctx["stream_duration_ms"] = duration_ms
        ctx["stream_chunks"] = chunks
        ctx["stream_bytes"] = total_bytes
        
        self.log.info(  # type: ignore[union-attr]
            f"Stream complete: {chunks} chunks, {total_bytes} bytes, {duration_ms:.1f}ms"
        )
    
    async def on_error(
        self,
        error: Exception,
        ctx: Context,
    ) -> None:
        import time
        start = ctx.get("_stream_start", time.perf_counter())
        duration_ms = (time.perf_counter() - float(start)) * 1000  # type: ignore[arg-type]
        chunks = ctx.get("_chunk_count", 0)
        
        self.log.error(  # type: ignore[union-attr]
            f"Stream error after {chunks} chunks, {duration_ms:.1f}ms: {error}"
        )


@dataclass(slots=True)
class StreamMetricsMiddleware:
    """Collect streaming metrics (chunk counts, bytes, timing).
    
    Uses same MetricsBackend protocol as MetricsMiddleware.
    
    Emits:
    - tool.stream.started: Counter per tool
    - tool.stream.chunks: Counter per stream
    - tool.stream.bytes: Counter per stream  
    - tool.stream.duration_ms: Timing
    - tool.stream.errors: Counter on failure
    
    Example:
        >>> registry.use(StreamMetricsMiddleware(backend=statsd))
    """
    
    from .plugins.metrics import LogMetricsBackend, MetricsBackend
    backend: MetricsBackend | None = None
    prefix: str = "tool.stream"
    
    def __post_init__(self) -> None:
        if self.backend is None:
            from .plugins.metrics import LogMetricsBackend
            self.backend = LogMetricsBackend()
    
    async def on_start(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
    ) -> None:
        import time
        ctx["_metrics_stream_start"] = time.perf_counter()
        ctx["_metrics_tool_name"] = tool.metadata.name
        ctx["_metrics_category"] = tool.metadata.category
        
        tags = {"tool": tool.metadata.name, "category": tool.metadata.category}
        self.backend.increment(f"{self.prefix}.started", tags=tags)  # type: ignore[union-attr]
    
    async def on_chunk(
        self,
        chunk: StreamChunk,
        ctx: Context,
    ) -> StreamChunk:
        return chunk  # Metrics collected on complete
    
    async def on_complete(
        self,
        accumulated: str,
        ctx: Context,
    ) -> None:
        import time
        start = ctx.get("_metrics_stream_start", time.perf_counter())
        duration_ms = (time.perf_counter() - float(start)) * 1000  # type: ignore[arg-type]
        
        tags = {
            "tool": str(ctx.get("_metrics_tool_name", "")),
            "category": str(ctx.get("_metrics_category", "")),
        }
        
        chunks = ctx.get("_chunk_count", 0)
        total_bytes = ctx.get("_total_bytes", 0)
        
        self.backend.increment(f"{self.prefix}.chunks", int(chunks), tags=tags)  # type: ignore[union-attr, arg-type]
        self.backend.increment(f"{self.prefix}.bytes", int(total_bytes), tags=tags)  # type: ignore[union-attr, arg-type]
        self.backend.timing(f"{self.prefix}.duration_ms", duration_ms, tags=tags)  # type: ignore[union-attr]
    
    async def on_error(
        self,
        error: Exception,
        ctx: Context,
    ) -> None:
        from toolcase.foundation.errors import classify_exception
        
        code = classify_exception(error)
        tags = {
            "tool": str(ctx.get("_metrics_tool_name", "")),
            "category": str(ctx.get("_metrics_category", "")),
            "error_code": code.value,
        }
        self.backend.increment(f"{self.prefix}.errors", tags=tags)  # type: ignore[union-attr]
