"""Central registry for tool discovery and management.

The registry provides:
- Tool registration and lookup by name
- Category-based filtering
- Formatted tool descriptions for LLM prompts
- Middleware pipeline for cross-cutting concerns
- Dependency injection for shared resources
- Integration adapters (e.g., LangChain)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Iterator
from typing import TYPE_CHECKING, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from ..core import BaseTool, ToolMetadata
from ..di import Container, Factory, Scope, ScopedContext
from ..errors import ErrorCode, ToolError, ToolException
from ..middleware import Context, Middleware, Next, compose
from ..streaming import (
    StreamChunk,
    StreamEvent,
    StreamEventKind,
    StreamResult,
    stream_complete,
    stream_error,
    stream_start,
)

if TYPE_CHECKING:
    pass


T = TypeVar("T", bound=BaseTool[BaseModel])


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
    """
    
    __slots__ = ("_tools", "_middleware", "_chain", "_container")
    
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool[BaseModel]] = {}
        self._middleware: list[Middleware] = []
        self._chain: Next | None = None
        self._container = Container()
    
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
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
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
    
    def use(self, middleware: Middleware) -> None:
        """Add middleware to the execution pipeline.
        
        Middleware is applied in order: first added = outermost (runs first).
        Invalidates the compiled chain, forcing recompilation on next execute.
        
        Args:
            middleware: Middleware instance implementing the Middleware protocol
        
        Example:
            >>> registry.use(LoggingMiddleware())
            >>> registry.use(RateLimitMiddleware(max_calls=10, window_seconds=60))
        """
        self._middleware.append(middleware)
        self._chain = None  # Invalidate cached chain
    
    def _get_chain(self) -> Next:
        """Get or compile the middleware chain."""
        if self._chain is None:
            self._chain = compose(self._middleware)
        return self._chain
    
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
        if name not in self._tools:
            return ToolError.create(
                name, f"Tool '{name}' not found in registry",
                ErrorCode.NOT_FOUND, recoverable=False
            ).render()
        
        tool = self._tools[name]
        
        # Validate params if dict
        try:
            validated = tool.params_schema(**params) if isinstance(params, dict) else params
        except ValidationError as e:
            return ToolError.create(
                name, f"Invalid parameters: {e}",
                ErrorCode.INVALID_PARAMS, recoverable=False
            ).render()
        
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
                    injected = await self._container.resolve_many(tool._inject, di_ctx)
                    context["injected"] = injected
                except KeyError as e:
                    return ToolError.create(
                        name, f"Missing dependency: {e}",
                        ErrorCode.INVALID_PARAMS, recoverable=False
                    ).render()
            
            # Execute through chain with exception handling
            try:
                chain = self._get_chain()
                return await chain(tool, validated, context)
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
        
        For sync callers that need middleware support. Uses asyncio.run()
        internally, so cannot be called from within an async context.
        Returns structured error string on failure.
        """
        try:
            return asyncio.run(self.execute(name, params, ctx=ctx))
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
        """Stream tool execution results as they're generated.
        
        For tools that support result streaming (streaming=True with async
        generator), this yields string chunks incrementally. For regular
        tools, yields the complete result as a single chunk.
        
        This is the primary method for consuming LLM-powered tools that
        naturally produce incremental output.
        
        Args:
            name: Tool name to execute
            params: Parameters as dict or BaseModel
            ctx: Optional pre-built context
        
        Yields:
            String chunks as they become available
        
        Raises:
            ToolException: If tool not found or validation fails
        
        Example:
            >>> async for chunk in registry.stream_execute("generate_report", {"topic": "AI"}):
            ...     print(chunk, end="", flush=True)
        """
        # Tool not found
        if name not in self._tools:
            yield ToolError.create(
                name, f"Tool '{name}' not found in registry",
                ErrorCode.NOT_FOUND, recoverable=False
            ).render()
            return
        
        tool = self._tools[name]
        
        # Validate params
        try:
            validated = tool.params_schema(**params) if isinstance(params, dict) else params
        except ValidationError as e:
            yield ToolError.create(
                name, f"Invalid parameters: {e}",
                ErrorCode.INVALID_PARAMS, recoverable=False
            ).render()
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
                    injected = await self._container.resolve_many(tool._inject, di_ctx)
                    context["injected"] = injected
                    # Set in context var for tool access
                    from ..core.decorator import _injected_deps
                    token = _injected_deps.set(injected)
                except KeyError as e:
                    yield ToolError.create(
                        name, f"Missing dependency: {e}",
                        ErrorCode.CONFIG_ERROR, recoverable=False
                    ).render()
                    return
            else:
                token = None
            
            try:
                # Check if tool supports result streaming
                if hasattr(tool, "supports_result_streaming") and tool.supports_result_streaming:
                    async for chunk in tool.stream_result(validated):
                        yield chunk
                else:
                    # Fall back to regular execution, yield complete result
                    result = await tool.arun(validated)
                    yield result
            except Exception as e:
                yield ToolError.from_exception(name, e, "Stream execution failed").render()
            finally:
                if token is not None:
                    _injected_deps.reset(token)
    
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
        index = 0
        
        try:
            async for content in self.stream_execute(name, params, ctx=ctx):
                accumulated.append(content)
                chunk = StreamChunk(content=content, index=index)
                yield StreamEvent(
                    kind=StreamEventKind.CHUNK,
                    tool_name=name,
                    data=chunk,
                )
                index += 1
            
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
        """Stream and collect full result with metadata.
        
        Useful when you want streaming behavior but also need final stats.
        
        Returns:
            StreamResult with accumulated content and timing metadata
        """
        start = time.time()
        parts: list[str] = []
        chunk_count = 0
        
        async def collect() -> StreamResult[str]:
            nonlocal parts, chunk_count
            async for content in self.stream_execute(name, params, ctx=ctx):
                parts.append(content)
                chunk_count += 1
            return StreamResult(
                value="".join(parts),
                chunks=chunk_count,
                duration_ms=(time.time() - start) * 1000,
                tool_name=name,
            )
        
        return await asyncio.wait_for(collect(), timeout=timeout)
    
    # ─────────────────────────────────────────────────────────────────
    # Querying
    # ─────────────────────────────────────────────────────────────────
    
    def list_tools(self, *, enabled_only: bool = True) -> list[ToolMetadata]:
        """List metadata for all registered tools."""
        return [
            t.metadata for t in self._tools.values()
            if not enabled_only or t.metadata.enabled
        ]
    
    def list_by_category(self, category: str, *, enabled_only: bool = True) -> list[ToolMetadata]:
        """List tools filtered by category."""
        return [
            t.metadata for t in self._tools.values()
            if t.metadata.category == category and (not enabled_only or t.metadata.enabled)
        ]
    
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
            flags = []
            if m.requires_api_key:
                flags.append("⚡")
            if m.streaming:
                flags.append("📡")
            flag_str = " ".join(flags)
            lines.append(f"- **{m.name}** ({m.category}){' ' + flag_str if flag_str else ''}: {m.description}")
        
        if lines:
            lines.append("\n_⚡ = requires API key | 📡 = supports streaming_")
        
        return "\n".join(lines)
    
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
        self._chain = None
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
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def set_registry(registry: ToolRegistry) -> None:
    """Replace the global registry."""
    global _registry
    _registry = registry


def reset_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _registry
    if _registry is not None:
        _registry.clear()
    _registry = None
