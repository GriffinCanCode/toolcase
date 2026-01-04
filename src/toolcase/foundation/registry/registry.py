"""Central registry for tool discovery and management.

The registry provides:
- Tool registration and lookup by name
- Category-based filtering
- Formatted tool descriptions for LLM prompts
- Middleware pipeline for cross-cutting concerns
- Dependency injection for shared resources
- Integration adapters (e.g., LangChain)
- Centralized validation via ValidationMiddleware
- Capability-aware execution (respects max_concurrent, caching, etc.)
- Effect system verification (declare side effects, enable pure testing)
- Event signals for lifecycle hooks (on_register, on_unregister, on_execute)
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable, Iterator

from beartype import beartype as typechecked
from pydantic import BaseModel, ValidationError

from toolcase.foundation.core import AnyTool, BaseTool, ToolCapabilities, ToolMetadata, ToolProtocol
from toolcase.foundation.di import Container, Factory, Scope
from toolcase.foundation.errors import ErrorCode, ToolError, ToolException, format_validation_error
from toolcase.foundation.events import Signal
from toolcase.runtime.middleware import Context, Middleware, Next, ValidationMiddleware, compose, compose_streaming, StreamMiddleware
from toolcase.runtime.concurrency import run_sync, CapacityLimiter
from toolcase.io.streaming import (
    StreamChunk, StreamEvent, StreamEventKind, StreamResult,
    stream_complete, stream_error, stream_start,
)


class ToolRegistry:
    """Central registry for all available tools.
    
    Provides tool discovery, filtering, middleware pipeline, dependency
    injection, capability-aware execution, and format conversion for agent use.
    
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
    
    Capability-Aware Execution:
        >>> # Tools with max_concurrent are automatically rate-limited
        >>> @tool(description="API call", max_concurrent=5)
        >>> async def call_api(endpoint: str) -> str: ...
        >>> # Registry respects the limit automatically
    
    Effect System:
        >>> registry.require_effects()  # Enable effect verification
        >>> registry.provide_effect("db", InMemoryDB())  # Register handlers
        >>> @tool(description="Fetch data", effects=["db"])
        ... async def fetch(db: Database) -> str: ...
        >>> registry.register(fetch)  # Verifies db handler exists
    
    Event Signals:
        >>> registry.on_register += lambda tool: print(f"Registered: {tool.metadata.name}")
        >>> registry.on_unregister += lambda name: print(f"Unregistered: {name}")
        >>> registry.on_execute += lambda name, params, result: log_execution(name)
    """
    
    __slots__ = (
        "_tools", "_middleware", "_chain", "_stream_chain", "_container",
        "_validation", "_limiters", "_effect_handlers", "_require_effects",
        "on_register", "on_unregister", "on_execute",
    )
    
    # Event type signatures for external typing
    OnRegister = Callable[[AnyTool], None]
    OnUnregister = Callable[[str], None]
    OnExecute = Callable[[str, dict[str, object] | BaseModel, str], None]
    
    def __init__(self) -> None:
        self._tools: dict[str, AnyTool] = {}
        self._middleware: list[Middleware | StreamMiddleware] = []
        self._chain: Next | None = None
        self._stream_chain: object | None = None  # StreamingChain, lazy import
        self._container = Container()
        self._validation: ValidationMiddleware | None = None
        self._limiters: dict[str, CapacityLimiter] = {}  # Concurrency control per tool
        self._effect_handlers: dict[str, object] = {}  # Effect handlers (e.g., InMemoryDB)
        self._require_effects: bool = False  # Whether to verify effects on registration
        # Event signals for lifecycle hooks
        self.on_register: Signal[ToolRegistry.OnRegister] = Signal()
        self.on_unregister: Signal[ToolRegistry.OnUnregister] = Signal()
        self.on_execute: Signal[ToolRegistry.OnExecute] = Signal()
    
    @typechecked
    def register(self, tool: AnyTool) -> None:
        """Register a tool instance. Creates CapacityLimiter for max_concurrent tools. Verifies effects when enabled."""
        name, meta = tool.metadata.name, tool.metadata
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered. Use unregister() first.")
        if len(meta.description) < 10:
            raise ValueError(f"Tool '{name}' description too short for LLM selection.")
        self._require_effects and self._verify_tool_effects(tool)  # Verify effects if enabled
        self._tools[name] = tool
        if (max_conc := meta.max_concurrent) is not None:  # Create CapacityLimiter for rate-limited tools
            self._limiters[name] = CapacityLimiter(max_conc)
        self.on_register.fire(tool)
    
    @typechecked
    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if found. Fires on_unregister signal."""
        self._limiters.pop(name, None)
        if self._tools.pop(name, None) is not None:
            self.on_unregister.fire(name)
            return True
        return False
    
    @typechecked
    def get(self, name: str) -> AnyTool | None:
        """Get tool by name."""
        return self._tools.get(name)
    
    # ─────────────────────────────────────────────────────────────────
    # Dependency Injection
    # ─────────────────────────────────────────────────────────────────
    
    @property
    def container(self) -> Container:
        """Access the DI container for advanced configuration."""
        return self._container
    
    @typechecked
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
    
    # ─────────────────────────────────────────────────────────────────
    # Effect System
    # ─────────────────────────────────────────────────────────────────
    
    def require_effects(self, *, enabled: bool = True) -> None:
        """Enable/disable effect verification on tool registration.
        
        When enabled, tools declaring effects must have corresponding handlers
        registered via provide_effect(). This provides compile-time-style safety.
        
        Args:
            enabled: Whether to enforce effect verification
        
        Example:
            >>> registry.require_effects()
            >>> registry.provide_effect("db", InMemoryDB())
            >>> @tool(effects=["db"])  # OK - handler exists
            ... async def fetch(): ...
            >>> @tool(effects=["unknown"])  # Raises MissingEffectHandler
            ... async def broken(): ...
        """
        self._require_effects = enabled
    
    def provide_effect(self, effect: str, handler: object) -> None:
        """Register an effect handler.
        
        Effect handlers are automatically injected into tools that declare
        the effect. Pure handlers enable testing without mocks.
        
        Args:
            effect: Effect name (e.g., "db", "http", "cache")
            handler: Handler instance (e.g., InMemoryDB, RecordingHTTP)
        
        Example:
            >>> registry.provide_effect("db", InMemoryDB())
            >>> registry.provide_effect("http", RecordingHTTP())
        """
        self._effect_handlers[effect.lower()] = handler
    
    def get_effect(self, effect: str) -> object | None: return self._effect_handlers.get(effect.lower())  # Get registered effect handler
    def has_effect(self, effect: str) -> bool: return effect.lower() in self._effect_handlers  # Check if effect handler is registered
    registered_effects = property(lambda s: frozenset(s._effect_handlers.keys()))  # All registered effect types
    
    def _verify_tool_effects(self, tool: AnyTool) -> None:
        """Verify tool's declared effects have handlers. Raises MissingEffectHandler if missing."""
        from toolcase.foundation.effects import MissingEffectHandler, get_effects
        for effect in (get_effects(tool) or []):
            if effect not in self._effect_handlers:
                raise MissingEffectHandler(effect, tool.metadata.name)
    
    def verify_all_effects(self) -> list[tuple[str, str]]:
        """Verify all registered tools have their effects satisfied. Returns (tool_name, effect) for missing handlers."""
        from toolcase.foundation.effects import get_effects
        return [(name, effect) for name, tool in self._tools.items()
                for effect in get_effects(tool) if effect not in self._effect_handlers]
    
    def tools_by_effect(self, *effects: str) -> list[ToolMetadata]:
        """Get tools declaring all specified effects."""
        from toolcase.foundation.effects import has_effects
        return [t.metadata for t in self._tools.values() if has_effects(t, *effects)]
    
    def __getitem__(self, name: str) -> AnyTool: return self._tools[name]  # Get tool by name, raises KeyError
    def __contains__(self, name: str) -> bool: return name in self._tools
    def __len__(self) -> int: return len(self._tools)
    def __iter__(self) -> Iterator[AnyTool]: return iter(self._tools.values())
    
    # ─────────────────────────────────────────────────────────────────
    # Middleware
    # ─────────────────────────────────────────────────────────────────
    
    def use(self, middleware: Middleware | StreamMiddleware) -> None:
        """Add middleware to pipeline. First added = outermost. Invalidates compiled chain."""
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
    
    validation = property(lambda s: s._validation)  # Access the ValidationMiddleware if configured
    has_validation_middleware = property(lambda s: s._validation is not None)  # Check if configured
    
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
        
        Capability-Aware Execution:
        - Respects max_concurrent via semaphore (rate limiting)
        - Adds capability info to context for middleware inspection
        
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
        
        # Build context with capability info
        context = ctx or Context()
        context["tool_name"] = name
        context["capabilities"] = tool.metadata.capabilities
        
        # Execute with scoped DI context, effect handlers, and optional semaphore
        async with self._container.scope() as di_ctx:
            context["_di_context"] = di_ctx
            context["_container"] = self._container
            
            # Resolve declared dependencies if tool has them
            if hasattr(tool, "_inject") and tool._inject:
                try:
                    context["injected"] = await self._container.resolve_many(tool._inject, di_ctx)
                except KeyError as e:
                    return ToolError.create(name, f"Missing dependency: {e}", ErrorCode.INVALID_PARAMS, recoverable=False).render()
            
            # Set up effect handlers in context for tool execution
            from toolcase.foundation.effects import EffectScope
            async with EffectScope(**self._effect_handlers):
                # Execute through chain with exception handling (rate-limited if needed)
                try:
                    if (limiter := self._limiters.get(name)) is not None:
                        async with limiter:
                            result = await self._get_chain()(tool, validated, context)  # type: ignore[arg-type]
                    else:
                        result = await self._get_chain()(tool, validated, context)  # type: ignore[arg-type]
                    self.on_execute.fire(name, params, result)
                    return result
                except ToolException as e:
                    return e.error.render()
                except Exception as e:
                    return ToolError.from_exception(name, e, "Execution failed").render()
    
    def execute_sync(self, name: str, params: dict[str, object] | BaseModel, *, ctx: Context | None = None) -> str:
        """Synchronous wrapper for execute(). Uses run_sync() for nested event loops."""
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
        
        Capability-Aware: Respects max_concurrent via semaphore, adds capability
        info to context for middleware inspection.
        
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
        
        # Build context with capability info
        context = ctx or Context()
        context["tool_name"] = name
        context["capabilities"] = tool.metadata.capabilities
        
        # Execute with scoped DI context and effect handlers
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
            
            # Rate-limited streaming if limiter exists
            limiter = self._limiters.get(name)
            
            async def _stream() -> AsyncIterator[str]:
                chain = self._get_stream_chain()
                if isinstance(chain, StreamingChain):
                    async for chunk in chain(tool, validated, context):  # type: ignore[arg-type]
                        yield chunk.content
                elif hasattr(tool, "supports_result_streaming") and tool.supports_result_streaming:
                    async for content in tool.stream_result(validated):  # type: ignore[arg-type]
                        yield content
                else:
                    yield await tool.arun(validated)  # type: ignore[arg-type]
            
            # Set up effect handlers and execute
            from toolcase.foundation.effects import EffectScope
            async with EffectScope(**self._effect_handlers):
                try:
                    if limiter is not None:
                        async with limiter:
                            async for chunk in _stream():
                                yield chunk
                    else:
                        async for chunk in _stream():
                            yield chunk
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
    
    def categories(self) -> set[str]: return {t.metadata.category for t in self._tools.values()}  # Get all unique categories
    
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
    
    def register_all(self, *tools: AnyTool) -> None:
        """Register multiple tools at once."""
        for tool in tools: self.register(tool)
    
    def clear(self) -> None:
        """Remove all registered tools, middleware, providers, effect handlers, and event subscriptions."""
        for d in (self._tools, self._middleware, self._limiters, self._effect_handlers): d.clear()
        self._container.clear()
        self._chain = self._stream_chain = self._validation = None
        self._require_effects = False
        self.on_register.clear()
        self.on_unregister.clear()
        self.on_execute.clear()
    
    # ─────────────────────────────────────────────────────────────────
    # Capability Introspection
    # ─────────────────────────────────────────────────────────────────
    
    def get_limiter(self, name: str) -> CapacityLimiter | None: return self._limiters.get(name)  # Get CapacityLimiter for rate-limited tool
    def limiter_stats(self) -> dict[str, dict[str, int]]: return {k: v.statistics for k, v in self._limiters.items()}  # Get stats for all limiters
    
    def tools_by_capability(self, *, supports_caching: bool | None = None, supports_streaming: bool | None = None,
                            idempotent: bool | None = None, max_concurrent_lte: int | None = None) -> list[ToolMetadata]:
        """Filter tools by advertised capabilities. Returns list of matching ToolMetadata."""
        def matches(caps: ToolCapabilities) -> bool:
            if supports_caching is not None and caps.supports_caching != supports_caching: return False
            if supports_streaming is not None and caps.supports_streaming != supports_streaming: return False
            if idempotent is not None and caps.idempotent != idempotent: return False
            if max_concurrent_lte is not None and (caps.max_concurrent is None or caps.max_concurrent > max_concurrent_lte): return False
            return True
        return [t.metadata for t in self._tools.values() if matches(t.metadata.capabilities)]
    
    # ─────────────────────────────────────────────────────────────────
    # Circuit Breaker Observability
    # ─────────────────────────────────────────────────────────────────
    
    def _find_circuit_middleware(self) -> object | None:
        """Find CircuitBreakerMiddleware in the chain (lazy import to avoid cycles)."""
        from toolcase.runtime.middleware.plugins.breaker import CircuitBreakerMiddleware
        return next((m for m in self._middleware if isinstance(m, CircuitBreakerMiddleware)), None)
    
    def circuit_state(self, tool_name: str) -> object | None:
        """Get current circuit state. Returns State enum or None if no breaker."""
        return mw.get_state(tool_name) if (mw := self._find_circuit_middleware()) else None  # type: ignore[union-attr]
    
    def circuit_is_open(self, tool_name: str) -> bool:
        """Check if circuit is open (fail-fast). Returns False if no breaker."""
        from toolcase.runtime.resilience import State
        return self.circuit_state(tool_name) == State.OPEN
    
    def circuit_stats(self) -> dict[str, object]:
        """Get statistics for all circuit breakers. Returns empty dict if no breaker."""
        return mw.stats() if (mw := self._find_circuit_middleware()) else {}  # type: ignore[union-attr]
    
    def reset_circuit(self, tool_name: str | None = None) -> bool:
        """Manually reset circuit breaker(s). Returns True if breaker found and reset."""
        if (mw := self._find_circuit_middleware()):
            mw.reset(tool_name)  # type: ignore[union-attr]
            return True
        return False
    
    def get_circuit_breaker(self, tool_name: str) -> object | None:
        """Get CircuitBreaker instance for a tool. Returns None if no middleware."""
        return mw.get_breaker(tool_name) if (mw := self._find_circuit_middleware()) else None  # type: ignore[union-attr]
    
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
    return _registry or (_registry := ToolRegistry())


def set_registry(registry: ToolRegistry) -> None:
    """Replace the global registry."""
    global _registry
    _registry = registry


def reset_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _registry
    _registry and _registry.clear()
    _registry = None
