"""Central registry for tool discovery and management.

The registry provides:
- Tool registration and lookup by name
- Category-based filtering
- Formatted tool descriptions for LLM prompts
- Middleware pipeline for cross-cutting concerns
- Dependency injection for shared resources
- Integration adapters (e.g., LangChain)
- Centralized validation via ValidationMiddleware
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator

from pydantic import BaseModel, ValidationError

from toolcase.foundation.core import BaseTool, ToolMetadata
from toolcase.foundation.di import Container, Factory, Scope
from toolcase.foundation.errors import ErrorCode, ToolError, ToolException, format_validation_error
from toolcase.runtime.middleware import Context, Middleware, Next, ValidationMiddleware, compose, compose_streaming, StreamMiddleware
from toolcase.runtime.concurrency import run_sync
from toolcase.io.streaming import (
    StreamChunk, StreamEvent, StreamEventKind, StreamResult,
    stream_complete, stream_error, stream_start,
)


class ToolRegistry:
    """Central registry for all available tools.
    
    Provides tool discovery, filtering, middleware pipeline, dependency
    injection, and format conversion for agent use.
    
    Example:
        >>> registry = ToolRegistry()
        >>> registry.provide("db", lambda: AsyncpgPool(), Scope.SINGLETON)
        >>> registry.register(MyTool())
        >>> registry.use(LoggingMiddleware())
        >>> result = await registry.execute("my_tool", {"query": "test"})
        
    Streaming with Middleware:
        >>> registry.use(StreamLoggingMiddleware())
        >>> async for chunk in registry.stream_execute("gen", {"topic": "AI"}):
        ...     print(chunk, end="")
    
    Centralized Validation:
        >>> validation = registry.use_validation()  # Returns ValidationMiddleware
        >>> validation.add_rule("search", "query", min_length(3), "must be >= 3 chars")
    """
    
    __slots__ = ("_tools", "_middleware", "_chain", "_stream_chain", "_container", "_validation")
    
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool[BaseModel]] = {}
        self._middleware: list[Middleware | StreamMiddleware] = []
        self._chain: Next | None = None
        self._stream_chain: object | None = None  # StreamingChain, lazy import
        self._container = Container()
        self._validation: ValidationMiddleware | None = None
    
    def register(self, tool: BaseTool[BaseModel]) -> None:
        """Register a tool instance with validation."""
        name = tool.metadata.name
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered. Use unregister() first.")
        if len(tool.metadata.description) < 10:
            raise ValueError(f"Tool '{name}' description too short for LLM selection.")
        self._tools[name] = tool
    
    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if found."""
        return self._tools.pop(name, None) is not None
    
    def get(self, name: str) -> BaseTool[BaseModel] | None:
        """Get tool by name."""
        return self._tools.get(name)
    
    # ─────────────────────────────────────────────────────────────────
    # Dependency Injection
    # ─────────────────────────────────────────────────────────────────
    
    @property
    def container(self) -> Container:
        """Access the DI container for advanced configuration."""
        return self._container
    
    def provide(
        self,
        name: str,
        factory: Factory[object],
        scope: Scope = Scope.SINGLETON,
    ) -> None:
        """Register a dependency provider.
        
        Dependencies are automatically injected into tools that declare them
        via the `inject` parameter in the @tool decorator.
        
        Args:
            name: Dependency name (e.g., "db", "http_client")
            factory: Callable returning instance (sync or async supported)
            scope: Lifecycle scope (SINGLETON, SCOPED, TRANSIENT)
        
        Example:
            >>> registry.provide("db", lambda: AsyncpgPool(), Scope.SINGLETON)
            >>> registry.provide("http", httpx.AsyncClient, Scope.SCOPED)
        """
        self._container.provide(name, factory, scope)
    
    def __getitem__(self, name: str) -> BaseTool[BaseModel]:
        """Get tool by name, raises KeyError if not found."""
        return self._tools[name]
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __iter__(self) -> Iterator[BaseTool[BaseModel]]:
        return iter(self._tools.values())
    
    # ─────────────────────────────────────────────────────────────────
    # Middleware
    # ─────────────────────────────────────────────────────────────────
    
    def use(self, middleware: Middleware | StreamMiddleware) -> None:
        """Add middleware to the execution pipeline.
        
        Middleware is applied in order: first added = outermost (runs first).
        Invalidates the compiled chain, forcing recompilation on next execute.
        
        Both regular Middleware and StreamMiddleware are supported. Regular
        middleware is auto-adapted for streaming context.
        
        Args:
            middleware: Middleware instance implementing Middleware or StreamMiddleware
        
        Example:
            >>> registry.use(LoggingMiddleware())
            >>> registry.use(StreamLoggingMiddleware())  # Streaming-aware
            >>> registry.use(RateLimitMiddleware(max_calls=10, window_seconds=60))
        """
        self._middleware.append(middleware)
        self._chain = self._stream_chain = None  # Invalidate cached chains
    
    def use_validation(self, *, revalidate: bool = False) -> ValidationMiddleware:
        """Enable centralized validation via ValidationMiddleware.
        
        Creates and prepends ValidationMiddleware to the chain. Returns the
        instance for adding custom rules. Should be called before other middleware.
        
        When enabled, execute()/stream_execute() skip internal validation,
        delegating fully to the middleware chain.
        
        Args:
            revalidate: Re-run Pydantic validation on already-validated BaseModel
        
        Returns:
            ValidationMiddleware instance for adding custom rules
        
        Example:
            >>> validation = registry.use_validation()
            >>> validation.add_rule("search", "query", min_length(3), "must be >= 3 chars")
            >>> validation.add_constraint("report", lambda p: p.start <= p.end or "invalid range")
        """
        if self._validation is None:
            self._validation = ValidationMiddleware(revalidate=revalidate)
            self._middleware.insert(0, self._validation)  # First in chain
            self._chain = self._stream_chain = None
        return self._validation
    
    @property
    def validation(self) -> ValidationMiddleware | None:
        """Access the ValidationMiddleware if configured."""
        return self._validation
    
    @property
    def has_validation_middleware(self) -> bool:
        """Check if ValidationMiddleware is configured."""
        return self._validation is not None
    
    def _get_chain(self) -> Next:
        """Get or compile the middleware chain."""
        self._chain = self._chain or compose(self._middleware)  # type: ignore[arg-type]
        return self._chain
    
    def _get_stream_chain(self) -> object:
        """Get or compile the streaming middleware chain."""
        self._stream_chain = self._stream_chain or compose_streaming(self._middleware)
        return self._stream_chain
    
    async def execute(
        self,
        name: str,
        params: dict[str, object] | BaseModel,
        *,
        ctx: Context | None = None,
    ) -> str:
        """Execute a tool through the middleware pipeline.
        
        This is the primary execution method when middleware is configured.
        Validates params, builds context, resolves dependencies, and runs
        through the chain.
        
        When ValidationMiddleware is configured (via use_validation()), validation
        is delegated to the middleware chain. Otherwise, validates internally.
        
        Injected dependencies are resolved from the container and added to
        context["injected"] for tools to access. Scoped resources are
        automatically cleaned up after execution.
        
        Args:
            name: Tool name to execute
            params: Parameters as dict or BaseModel
            ctx: Optional pre-built context (default: new Context)
        
        Returns:
            Tool result string (or formatted error string on failure)
        
        Example:
            >>> result = await registry.execute("search", {"query": "python"})
        """
        # Tool not found
        if (tool := self._tools.get(name)) is None:
            return ToolError.create(name, f"Tool '{name}' not found in registry", ErrorCode.NOT_FOUND, recoverable=False).render()
        
        # Validate params if dict (skip if ValidationMiddleware handles it)
        validated: BaseModel | dict[str, object] = params
        if not self._validation:
            try:
                validated = tool.params_schema(**params) if isinstance(params, dict) else params
            except ValidationError as e:
                return ToolError.create(name, format_validation_error(e, tool_name=name), ErrorCode.INVALID_PARAMS, recoverable=False).render()
        
        # Build context
        context = ctx or Context()
        context["tool_name"] = name
        
        # Execute with scoped DI context
        async with self._container.scope() as di_ctx:
            context["_di_context"] = di_ctx
            context["_container"] = self._container
            
            # Resolve declared dependencies if tool has them
            if hasattr(tool, "_inject") and tool._inject:
                try:
                    context["injected"] = await self._container.resolve_many(tool._inject, di_ctx)
                except KeyError as e:
                    return ToolError.create(name, f"Missing dependency: {e}", ErrorCode.INVALID_PARAMS, recoverable=False).render()
            
            # Execute through chain with exception handling
            try:
                return await self._get_chain()(tool, validated, context)  # type: ignore[arg-type]
            except ToolException as e:
                return e.error.render()
            except Exception as e:
                return ToolError.from_exception(name, e, "Execution failed").render()
    
    def execute_sync(
        self,
        name: str,
        params: dict[str, object] | BaseModel,
        *,
        ctx: Context | None = None,
    ) -> str:
        """Synchronous wrapper for execute().
        
        For sync callers that need middleware support. Uses run_sync()
        which handles nested event loops (FastAPI, Jupyter).
        Returns structured error string on failure.
        """
        try:
            return run_sync(self.execute(name, params, ctx=ctx))
        except Exception as e:
            return ToolError.from_exception(name, e, "Sync execution failed").render()
    
    # ─────────────────────────────────────────────────────────────────
    # Result Streaming
    # ─────────────────────────────────────────────────────────────────
    
    async def stream_execute(
        self,
        name: str,
        params: dict[str, object] | BaseModel,
        *,
        ctx: Context | None = None,
    ) -> AsyncIterator[str]:
        """Stream tool execution through the middleware pipeline.
        
        For tools that support result streaming (streaming=True with async
        generator), this yields string chunks incrementally through each
        middleware's chunk hooks. For regular tools, yields complete result.
        
        When ValidationMiddleware is configured (via use_validation()), validation
        is delegated to the middleware chain. Otherwise, validates internally.
        
        Middleware receives lifecycle hooks: on_start, on_chunk, on_complete,
        on_error. Regular Middleware is auto-adapted to streaming context.
        
        Args:
            name: Tool name to execute
            params: Parameters as dict or BaseModel
            ctx: Optional pre-built context
        
        Yields:
            String chunks as they become available
        
        Example:
            >>> registry.use(StreamLoggingMiddleware())
            >>> async for chunk in registry.stream_execute("generate", {"topic": "AI"}):
            ...     print(chunk, end="", flush=True)
        """
        from toolcase.runtime.middleware.streaming import StreamingChain
        
        # Tool not found
        if (tool := self._tools.get(name)) is None:
            yield ToolError.create(name, f"Tool '{name}' not found in registry", ErrorCode.NOT_FOUND, recoverable=False).render()
            return
        
        # Validate params (skip if ValidationMiddleware handles it)
        validated: BaseModel | dict[str, object] = params
        if not self._validation:
            try:
                validated = tool.params_schema(**params) if isinstance(params, dict) else params
            except ValidationError as e:
                yield ToolError.create(name, format_validation_error(e, tool_name=name), ErrorCode.INVALID_PARAMS, recoverable=False).render()
                return
        
        # Build context
        context = ctx or Context()
        context["tool_name"] = name
        
        # Execute with scoped DI context
        async with self._container.scope() as di_ctx:
            context["_di_context"] = di_ctx
            context["_container"] = self._container
            
            # Resolve dependencies if needed
            if hasattr(tool, "_inject") and tool._inject:
                try:
                    context["injected"] = await self._container.resolve_many(tool._inject, di_ctx)
                except KeyError as e:
                    yield ToolError.create(name, f"Missing dependency: {e}", ErrorCode.INVALID_PARAMS, recoverable=False).render()
                    return
            
            try:
                # Get streaming chain and execute through middleware
                chain = self._get_stream_chain()
                if isinstance(chain, StreamingChain):
                    async for chunk in chain(tool, validated, context):  # type: ignore[arg-type]
                        yield chunk.content
                else:
                    # Fallback if chain type unexpected
                    if hasattr(tool, "supports_result_streaming") and tool.supports_result_streaming:
                        async for content in tool.stream_result(validated):  # type: ignore[arg-type]
                            yield content
                    else:
                        yield await tool.arun(validated)  # type: ignore[arg-type]
            except Exception as e:
                yield ToolError.from_exception(name, e, "Stream execution failed").render()
    
    async def stream_execute_events(
        self,
        name: str,
        params: dict[str, object] | BaseModel,
        *,
        ctx: Context | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream tool execution as typed events for transport.
        
        Wraps stream_execute with start/chunk/complete/error event lifecycle.
        Ideal for WebSocket/SSE delivery with full state tracking.
        
        Args:
            name: Tool name to execute
            params: Parameters as dict or BaseModel
            ctx: Optional pre-built context
        
        Yields:
            StreamEvent objects for transport serialization
        
        Example:
            >>> async for event in registry.stream_execute_events("gen", {"topic": "AI"}):
            ...     await websocket.send(event.to_json())
        """
        yield stream_start(name)
        accumulated: list[str] = []
        idx = 0
        try:
            async for content in self.stream_execute(name, params, ctx=ctx):
                accumulated.append(content)
                yield StreamEvent(kind=StreamEventKind.CHUNK, tool_name=name, data=StreamChunk(content=content, index=idx))
                idx += 1
            yield stream_complete(name, "".join(accumulated))
        except Exception as e:
            yield stream_error(name, str(e))
    
    async def stream_execute_collected(
        self,
        name: str,
        params: dict[str, object] | BaseModel,
        *,
        ctx: Context | None = None,
        timeout: float = 60.0,
    ) -> StreamResult[str]:
        """Stream and collect full result with metadata. Returns StreamResult with accumulated content and timing metadata."""
        from toolcase.runtime.concurrency import CancelScope
        start, parts = time.time(), []
        async with CancelScope(timeout=timeout):
            async for content in self.stream_execute(name, params, ctx=ctx):
                parts.append(content)
            return StreamResult(value="".join(parts), chunks=len(parts), duration_ms=(time.time() - start) * 1000, tool_name=name)
    
    # ─────────────────────────────────────────────────────────────────
    # Querying
    # ─────────────────────────────────────────────────────────────────
    
    def list_tools(self, *, enabled_only: bool = True) -> list[ToolMetadata]:
        """List metadata for all registered tools."""
        return [t.metadata for t in self._tools.values() if not enabled_only or t.metadata.enabled]
    
    def list_by_category(self, category: str, *, enabled_only: bool = True) -> list[ToolMetadata]:
        """List tools filtered by category."""
        return [t.metadata for t in self._tools.values() if t.metadata.category == category and (not enabled_only or t.metadata.enabled)]
    
    def categories(self) -> set[str]:
        """Get all unique categories."""
        return {t.metadata.category for t in self._tools.values()}
    
    # ─────────────────────────────────────────────────────────────────
    # Formatting
    # ─────────────────────────────────────────────────────────────────
    
    def describe(self, *, enabled_only: bool = True) -> str:
        """Get formatted descriptions of all tools for prompts."""
        lines = []
        for tool in self._tools.values():
            if enabled_only and not tool.metadata.enabled:
                continue
            m = tool.metadata
            flags = " ".join(f for f, c in [("⚡", m.requires_api_key), ("📡", m.streaming)] if c)
            lines.append(f"- **{m.name}** ({m.category}){' ' + flags if flags else ''}: {m.description}")
        return "\n".join(lines + (["\n_⚡ = requires API key | 📡 = supports streaming_"] if lines else []))
    
    # ─────────────────────────────────────────────────────────────────
    # Bulk Operations
    # ─────────────────────────────────────────────────────────────────
    
    def register_all(self, *tools: BaseTool[BaseModel]) -> None:
        """Register multiple tools at once."""
        for tool in tools:
            self.register(tool)
    
    def clear(self) -> None:
        """Remove all registered tools, middleware, and providers."""
        self._tools.clear()
        self._middleware.clear()
        self._chain = self._stream_chain = None
        self._validation = None
        self._container.clear()
    
    async def dispose(self) -> None:
        """Dispose all singleton resources in the container."""
        await self._container.dispose()


# ─────────────────────────────────────────────────────────────────────────────
# Global Registry
# ─────────────────────────────────────────────────────────────────────────────

_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _registry
    return _registry if _registry else (_registry := ToolRegistry())


def set_registry(registry: ToolRegistry) -> None:
    """Replace the global registry."""
    global _registry
    _registry = registry


def reset_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _registry
    _registry and _registry.clear()
    _registry = None
