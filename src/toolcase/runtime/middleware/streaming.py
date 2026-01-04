"""Streaming middleware for chunk-aware execution hooks.

Extends the middleware system to support streaming tools with lifecycle
hooks: on_start, on_chunk, on_complete, on_error.

Regular Middleware is automatically adapted to streaming context via
StreamingAdapter - running "before" logic on start, "after" on complete.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from pydantic import BaseModel

from toolcase.foundation.core.decorator import InjectedDeps, clear_injected_deps, set_injected_deps
from toolcase.io.streaming import StreamChunk

from .middleware import Context, Middleware

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool

# Type alias for streaming continuation
StreamNext = AsyncIterator[StreamChunk]

# Hook names for duck typing checks
_STREAM_HOOKS = ("on_start", "on_chunk", "on_complete", "on_error")
_SKIP_CLASSES = frozenset(("StreamMiddleware", "Protocol", "object"))


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
    
    async def on_start(self, tool: BaseTool[BaseModel], params: BaseModel, ctx: Context) -> None:
        """Called before streaming begins. Use for setup/logging."""
        ...
    
    async def on_chunk(self, chunk: StreamChunk, ctx: Context) -> StreamChunk:
        """Called for each chunk. Return chunk (possibly transformed)."""
        ...
    
    async def on_complete(self, accumulated: str, ctx: Context) -> None:
        """Called when stream completes successfully."""
        ...
    
    async def on_error(self, error: Exception, ctx: Context) -> None:
        """Called when stream encounters an error."""
        ...


def _has_hook(obj: object, name: str) -> bool:
    """Check if object has implemented hook (not inherited from Protocol)."""
    if not callable(getattr(obj, name, None)):
        return False
    # Check if defined in object's class hierarchy (excluding Protocol bases)
    return any(
        name in cls.__dict__
        for cls in type(obj).__mro__
        if cls.__name__ not in _SKIP_CLASSES
    )


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
    
    async def on_start(self, tool: BaseTool[BaseModel], params: BaseModel, ctx: Context) -> None:
        """Mark stream start in context."""
        ctx["_stream_started"] = True
    
    async def on_complete(self, accumulated: str, ctx: Context) -> None:
        """No-op - actual middleware logic runs in wrap_stream."""
    
    async def on_error(self, error: Exception, ctx: Context) -> None:
        """No-op - errors propagate naturally."""


async def _stream_through_middleware(
    middleware: StreamMiddleware | StreamingAdapter,
    tool: BaseTool[BaseModel],
    params: BaseModel,
    ctx: Context,
    source: AsyncIterator[StreamChunk],
) -> AsyncIterator[StreamChunk]:
    """Stream chunks through middleware's hooks (on_start, on_chunk, on_complete/on_error)."""
    accumulated: list[str] = []
    
    if _has_hook(middleware, "on_start"):
        await middleware.on_start(tool, params, ctx)
    
    try:
        async for chunk in source:
            if _has_hook(middleware, "on_chunk"):
                chunk = await middleware.on_chunk(chunk, ctx)
            accumulated.append(chunk.content)
            yield chunk
        
        if _has_hook(middleware, "on_complete"):
            await middleware.on_complete("".join(accumulated), ctx)
    except Exception as e:
        if _has_hook(middleware, "on_error"):
            await middleware.on_error(e, ctx)
        raise


def _is_stream_middleware(mw: object) -> bool:
    """Check if object implements StreamMiddleware hooks (duck typing)."""
    return any(_has_hook(mw, name) for name in _STREAM_HOOKS)


def _to_stream_middleware(mw: Middleware | StreamMiddleware) -> StreamMiddleware | StreamingAdapter:
    """Convert regular Middleware to streaming-compatible form."""
    return mw if _is_stream_middleware(mw) else StreamingAdapter(mw)  # type: ignore[return-value, arg-type]


async def _base_stream(
    tool: BaseTool[BaseModel],
    params: BaseModel,
    ctx: Context,
) -> AsyncIterator[StreamChunk]:
    """Base streaming executor - yields chunks from tool.stream_result()."""
    # Set injected dependencies from context if present
    if (injected := ctx.get("injected")) and isinstance(injected, dict):
        set_injected_deps(cast(InjectedDeps, injected))
    
    try:
        if getattr(tool, "supports_result_streaming", False):
            idx = 0
            async for content in tool.stream_result(params):
                yield StreamChunk(content=content, index=idx)
                idx += 1
        else:
            # Non-streaming tool: yield single chunk with complete result
            yield StreamChunk(content=await tool.arun(params), index=0)
    finally:
        clear_injected_deps()


def compose_streaming(middleware: Sequence[Middleware | StreamMiddleware]) -> "StreamingChain":
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
        # Build chain: wrap from innermost to outermost
        def wrap(src: AsyncIterator[StreamChunk], mw: StreamMiddleware | StreamingAdapter) -> AsyncIterator[StreamChunk]:
            return _stream_through_middleware(mw, tool, params, ctx, src)
        
        stream = reduce(wrap, reversed(self._middleware), _base_stream(tool, params, ctx))
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
    
    log: logging.Logger | None = None
    log_chunk_sizes: bool = False
    
    def __post_init__(self) -> None:
        if self.log is None:
            self.log = logging.getLogger("toolcase.streaming")
    
    async def on_start(self, tool: BaseTool[BaseModel], params: BaseModel, ctx: Context) -> None:
        ctx["_stream_start"], ctx["_chunk_count"], ctx["_total_bytes"] = time.perf_counter(), 0, 0
        self.log.info(f"[{tool.metadata.name}] Stream started")  # type: ignore[union-attr]
    
    async def on_chunk(self, chunk: StreamChunk, ctx: Context) -> StreamChunk:
        ctx["_chunk_count"] = ctx.get("_chunk_count", 0) + 1  # type: ignore[operator]
        ctx["_total_bytes"] = ctx.get("_total_bytes", 0) + len(chunk.content)  # type: ignore[operator]
        if self.log_chunk_sizes:
            self.log.debug(f"Chunk {chunk.index}: {len(chunk.content)} bytes")  # type: ignore[union-attr]
        return chunk
    
    async def on_complete(self, accumulated: str, ctx: Context) -> None:
        duration_ms = (time.perf_counter() - float(ctx.get("_stream_start", 0))) * 1000  # type: ignore[arg-type]
        chunks, total_bytes = ctx.get("_chunk_count", 0), ctx.get("_total_bytes", 0)
        ctx["stream_duration_ms"], ctx["stream_chunks"], ctx["stream_bytes"] = duration_ms, chunks, total_bytes
        self.log.info(f"Stream complete: {chunks} chunks, {total_bytes} bytes, {duration_ms:.1f}ms")  # type: ignore[union-attr]
    
    async def on_error(self, error: Exception, ctx: Context) -> None:
        duration_ms = (time.perf_counter() - float(ctx.get("_stream_start", 0))) * 1000  # type: ignore[arg-type]
        self.log.error(f"Stream error after {ctx.get('_chunk_count', 0)} chunks, {duration_ms:.1f}ms: {error}")  # type: ignore[union-attr]


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
    
    async def on_start(self, tool: BaseTool[BaseModel], params: BaseModel, ctx: Context) -> None:
        ctx["_metrics_stream_start"] = time.perf_counter()
        ctx["_metrics_tool_name"], ctx["_metrics_category"] = tool.metadata.name, tool.metadata.category
        self.backend.increment(f"{self.prefix}.started", tags={"tool": tool.metadata.name, "category": tool.metadata.category})  # type: ignore[union-attr]
    
    async def on_chunk(self, chunk: StreamChunk, ctx: Context) -> StreamChunk:
        return chunk  # Metrics collected on complete
    
    async def on_complete(self, accumulated: str, ctx: Context) -> None:
        duration_ms = (time.perf_counter() - float(ctx.get("_metrics_stream_start", 0))) * 1000  # type: ignore[arg-type]
        tags = {"tool": str(ctx.get("_metrics_tool_name", "")), "category": str(ctx.get("_metrics_category", ""))}
        chunks, total_bytes = int(ctx.get("_chunk_count", 0)), int(ctx.get("_total_bytes", 0))  # type: ignore[arg-type]
        
        self.backend.increment(f"{self.prefix}.chunks", chunks, tags=tags)  # type: ignore[union-attr]
        self.backend.increment(f"{self.prefix}.bytes", total_bytes, tags=tags)  # type: ignore[union-attr]
        self.backend.timing(f"{self.prefix}.duration_ms", duration_ms, tags=tags)  # type: ignore[union-attr]
    
    async def on_error(self, error: Exception, ctx: Context) -> None:
        from toolcase.foundation.errors import classify_exception
        tags = {
            "tool": str(ctx.get("_metrics_tool_name", "")),
            "category": str(ctx.get("_metrics_category", "")),
            "error_code": classify_exception(error).value,
        }
        self.backend.increment(f"{self.prefix}.errors", tags=tags)  # type: ignore[union-attr]
