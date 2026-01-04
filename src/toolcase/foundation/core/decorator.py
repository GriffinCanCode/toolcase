"""Decorator-based tool definition for simple functions.

Transforms decorated functions into full BaseTool instances with
auto-generated parameter schemas from type hints.

Example:
    >>> @tool(
    ...     name="web_search",
    ...     description="Search the web for information",
    ...     category="search",
    ... )
    ... def web_search(query: str, limit: int = 5) -> str:
    ...     '''Search the web.
    ...     
    ...     Args:
    ...         query: Search query string
    ...         limit: Maximum results to return
    ...     '''
    ...     return f"Results for: {query}"
    ...
    >>> registry.register(web_search)  # Works - it's a BaseTool
    >>> web_search(query="python")     # Also works via __call__
    'Results for: python'

Dependency Injection Example:
    >>> @tool(description="Fetch user data", inject=["db", "http_client"])
    ... async def fetch_user(user_id: str, db: Database, http: HttpClient) -> str:
    ...     user = await db.fetch_one("SELECT * FROM users WHERE id = $1", user_id)
    ...     return f"User: {user['name']}"
    ...
    >>> registry.provide("db", lambda: AsyncpgPool())
    >>> registry.provide("http_client", httpx.AsyncClient)
    >>> await registry.execute("fetch_user", {"user_id": "123"})

Effect System Example:
    >>> @tool(description="Fetch user from DB", effects=["db", "cache"])
    ... async def fetch_user(user_id: str, db: Database) -> str:
    ...     return await db.fetch_one("SELECT * FROM users WHERE id = $1", user_id)
    ...
    >>> # Test without mocks using pure handlers
    >>> from toolcase.foundation.effects import test_effects, InMemoryDB
    >>> async with test_effects(db=InMemoryDB()):
    ...     result = await registry.execute("fetch_user", {"user_id": "123"})
"""

from __future__ import annotations

import asyncio
import inspect
import re
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Callable, ParamSpec, get_type_hints, overload

from pydantic import BaseModel, Field, create_model

from toolcase.runtime.concurrency import to_thread

from toolcase.io.cache import DEFAULT_TTL
from toolcase.foundation.errors import ToolResult, classify_exception, ErrorTrace, Result
from toolcase.foundation.errors.result import _ERR, _OK
from toolcase.foundation.errors.types import ErrorContext, typechecked
from .base import BaseTool, ToolCapabilities, ToolMetadata

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable

    from toolcase.io.progress import ToolProgress

P = ParamSpec("P")

# Type for injected dependencies (can be any runtime objects, not JSON)
InjectedDeps = dict[str, object]

# Context variable for passing injected dependencies
# Note: No default set to avoid mutable default dict shared across contexts
_injected_deps: ContextVar[InjectedDeps] = ContextVar("injected_deps")

# Empty dict singleton for clear operations (avoids allocations)
_EMPTY_DEPS: InjectedDeps = {}


@typechecked
def get_injected_deps() -> InjectedDeps:
    """Get dependencies for the current execution context. Returns empty dict if not set."""
    return _injected_deps.get(_EMPTY_DEPS)


@typechecked
def set_injected_deps(deps: InjectedDeps) -> None:
    """Set dependencies for the current execution context. Called by registry before tool execution."""
    _injected_deps.set(deps)


@typechecked
def clear_injected_deps() -> None:
    """Clear injected dependencies after execution."""
    _injected_deps.set(_EMPTY_DEPS)


# ─────────────────────────────────────────────────────────────────────────────
# Docstring Parsing
# ─────────────────────────────────────────────────────────────────────────────

_PARAM_PATTERN = re.compile(
    r"^\s*(?P<name>\w+)\s*(?:\([^)]*\))?\s*:\s*(?P<desc>.+?)(?=\n\s*\w+\s*:|$)",
    re.MULTILINE | re.DOTALL,
)


def _parse_docstring_params(docstring: str | None) -> dict[str, str]:
    """Extract parameter descriptions from Google/NumPy style docstrings."""
    if not docstring or len(sections := re.split(r"\n\s*(?:Args|Arguments|Parameters)\s*:\s*\n", docstring, flags=re.IGNORECASE)) < 2:
        return {}
    args_section = re.split(r"\n\s*(?:Returns|Raises|Examples?|Notes?|Yields)\s*:", sections[1], flags=re.IGNORECASE)[0]
    return {m.group("name"): " ".join(m.group("desc").split()) for m in _PARAM_PATTERN.finditer(args_section)}


# ─────────────────────────────────────────────────────────────────────────────
# Schema Generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_schema(func: Callable[..., str], model_name: str, exclude: list[str] | None = None) -> type[BaseModel]:
    """Generate Pydantic model from function signature. Introspects type hints and defaults to build Field definitions."""
    sig, hints, param_docs, excluded = inspect.signature(func), get_type_hints(func), _parse_docstring_params(func.__doc__), set(exclude or [])
    fields: dict[str, tuple[type, object]] = {}
    for name, param in sig.parameters.items():
        if name in ("self", "cls") or name in excluded:
            continue
        field_type, description = hints.get(name, str), param_docs.get(name, f"Parameter: {name}")
        fields[name] = (field_type, Field(..., description=description) if param.default is inspect.Parameter.empty else Field(default=param.default, description=description))
    return create_model(model_name, **fields)  # type: ignore[call-overload]


# ─────────────────────────────────────────────────────────────────────────────
# FunctionTool: BaseTool wrapper for functions
# ─────────────────────────────────────────────────────────────────────────────

class FunctionTool(BaseTool[BaseModel]):
    """BaseTool wrapper for decorated functions. Bridges function API with class-based system, supports DI via inject param and effect tracking."""
    
    __slots__ = ("_func", "_is_async", "_original_func", "_inject", "_effects", "_tool_ctx")
    
    def __init__(
        self,
        func: Callable[..., str] | Callable[..., Awaitable[str]],
        metadata: ToolMetadata,
        params_schema: type[BaseModel],
        *,
        cache_enabled: bool = True,
        cache_ttl: float = DEFAULT_TTL,
        inject: list[str] | None = None,
        effects: frozenset[str] | None = None,
    ) -> None:
        self._func, self._original_func = func, func
        self._is_async = asyncio.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)
        self._inject = inject or []
        self._effects = effects or frozenset()
        self._tool_ctx = (ErrorContext(operation=f"tool:{metadata.name}", location="", metadata={}),)
        self.__class__ = type(f"{type(self).__name__}_{metadata.name}", (type(self),),
                              {"metadata": metadata, "params_schema": params_schema, "cache_enabled": cache_enabled, "cache_ttl": cache_ttl})
    
    @property
    def declared_effects(self) -> frozenset[str]:
        """Effects declared by this tool."""
        return self._effects
    
    async def _async_run(self, params: BaseModel) -> str:
        """Execute the wrapped function. Async-first: sync funcs run via to_thread. Injects effect handlers if available."""
        from toolcase.foundation.effects import get_handler
        kwargs = params.model_dump() | (get_injected_deps() if self._inject else {})
        # Inject effect handlers for declared effects (replace None defaults)
        for effect in self._effects:
            if (handler := get_handler(effect)) and kwargs.get(effect) is None:
                kwargs[effect] = handler
        return await self._func(**kwargs) if self._is_async else await to_thread(self._func, **kwargs)  # type: ignore[misc, arg-type]
    
    async def _async_run_result(self, params: BaseModel) -> ToolResult:
        """Execute with Result-based error handling (optimized path)."""
        try:
            return Result(await self._async_run(params), _OK)
        except Exception as e:
            return self._make_err(e, "execution")
    
    def _make_err(self, exc: Exception, context: str) -> ToolResult:
        """Create Err result from exception (internal, optimized)."""
        import traceback
        return Result(ErrorTrace(
            message=f"{context}: {exc}" if context else str(exc), contexts=self._tool_ctx,
            error_code=classify_exception(exc).value, recoverable=True, details=traceback.format_exc(),
        ), _ERR)
    
    @property
    def func(self) -> Callable[..., str]:
        """Access the original wrapped function."""
        return self._original_func  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Streaming Function Tools
# ─────────────────────────────────────────────────────────────────────────────

class StreamingFunctionTool(FunctionTool):
    """FunctionTool variant for async generator functions yielding ToolProgress. For progress streaming."""
    
    __slots__ = ()
    
    async def stream_run(self, params: BaseModel) -> AsyncIterator[ToolProgress]:
        """Stream progress events from the wrapped generator function."""
        from toolcase.foundation.effects import get_handler
        kwargs = params.model_dump() | (get_injected_deps() if self._inject else {})
        # Inject effect handlers (replace None defaults)
        for effect in self._effects:
            if (handler := get_handler(effect)) and kwargs.get(effect) is None:
                kwargs[effect] = handler
        async for progress in self._func(**kwargs):  # type: ignore[union-attr]
            yield progress


class ResultStreamingFunctionTool(FunctionTool):
    """FunctionTool variant for async generators yielding string chunks. For true result streaming (LLM outputs, incremental content)."""
    
    __slots__ = ("_backpressure_buffer",)
    
    def __init__(
        self,
        func: Callable[..., str] | Callable[..., Awaitable[str]],
        metadata: ToolMetadata,
        params_schema: type[BaseModel],
        *,
        cache_enabled: bool = True,
        cache_ttl: float = DEFAULT_TTL,
        inject: list[str] | None = None,
        effects: frozenset[str] | None = None,
        backpressure_buffer: int | None = None,
    ) -> None:
        super().__init__(func, metadata, params_schema, cache_enabled=cache_enabled, cache_ttl=cache_ttl, inject=inject, effects=effects)
        self._backpressure_buffer = backpressure_buffer
    
    @property
    def supports_result_streaming(self) -> bool:
        return True
    
    @property
    def backpressure_buffer(self) -> int | None:
        """Buffer size for backpressure, or None if disabled."""
        return self._backpressure_buffer
    
    async def stream_result(self, params: BaseModel) -> AsyncIterator[str]:
        """Stream string chunks from the wrapped async generator."""
        from toolcase.foundation.effects import get_handler
        kwargs = params.model_dump() | (get_injected_deps() if self._inject else {})
        # Inject effect handlers (replace None defaults)
        for effect in self._effects:
            if (handler := get_handler(effect)) and kwargs.get(effect) is None:
                kwargs[effect] = handler
        async for chunk in self._func(**kwargs):  # type: ignore[union-attr]
            yield chunk
    
    async def stream_result_with_backpressure(self, params: BaseModel, buffer_size: int | None = None) -> AsyncIterator[str]:
        """Stream with backpressure - producer pauses when consumer is slow.
        
        Args:
            params: Tool parameters
            buffer_size: Override buffer size (default: tool's backpressure_buffer or 10)
        
        Yields:
            String chunks with backpressure applied
        """
        from toolcase.runtime.concurrency.streams import backpressure_stream
        
        size = buffer_size or self._backpressure_buffer or 10
        async for chunk in backpressure_stream(self.stream_result(params), maxsize=size):
            yield chunk
    
    async def _async_run(self, params: BaseModel) -> str:
        """Execute by collecting all stream chunks."""
        return "".join([chunk async for chunk in self.stream_result(params)])


# ─────────────────────────────────────────────────────────────────────────────
# The @tool Decorator
# ─────────────────────────────────────────────────────────────────────────────

@overload
def tool(func: Callable[P, str]) -> FunctionTool: ...

@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    category: str = "general",
    tags: list[str] | set[str] | None = None,
    requires_api_key: bool = False,
    streaming: bool = False,
    backpressure_buffer: int | None = None,
    cache_enabled: bool = True,
    cache_ttl: float = DEFAULT_TTL,
    inject: list[str] | None = None,
    effects: list[str] | set[str] | frozenset[str] | None = None,
    # Capability negotiation
    capabilities: ToolCapabilities | None = None,
    max_concurrent: int | None = None,
    idempotent: bool = True,
    requires_confirmation: bool = False,
    estimated_latency_ms: int | None = None,
) -> Callable[[Callable[P, str]], FunctionTool]: ...


def tool(
    func: Callable[P, str] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    category: str = "general",
    tags: list[str] | set[str] | None = None,
    requires_api_key: bool = False,
    streaming: bool = False,
    backpressure_buffer: int | None = None,
    cache_enabled: bool = True,
    cache_ttl: float = DEFAULT_TTL,
    inject: list[str] | None = None,
    effects: list[str] | set[str] | frozenset[str] | None = None,
    # Capability negotiation
    capabilities: ToolCapabilities | None = None,
    max_concurrent: int | None = None,
    idempotent: bool = True,
    requires_confirmation: bool = False,
    estimated_latency_ms: int | None = None,
) -> FunctionTool | Callable[[Callable[P, str]], FunctionTool]:
    """Decorator to create a tool from a function.
    
    Transforms a function into a full BaseTool instance with auto-generated
    parameter schema from type hints. Compatible with registry, cache, and
    LangChain integration.
    
    Args:
        func: The function to wrap (used when decorator called without parens)
        name: Tool name (defaults to function name, converted to snake_case)
        description: Tool description (defaults to first line of docstring)
        category: Tool category for grouping
        tags: Capability tags for discovery (e.g., ["search", "web"])
        requires_api_key: Whether tool needs external API credentials
        streaming: Whether tool supports progress streaming
        backpressure_buffer: Buffer size for backpressure (streaming tools only).
                            When set, producer pauses when buffer fills. None = no backpressure.
        cache_enabled: Enable result caching
        cache_ttl: Cache TTL in seconds
        inject: List of dependency names to inject from registry container
        effects: List of side effects declared by this tool (e.g., ["db", "http"])
                 Enables compile-time verification and testing without mocks
        capabilities: Full ToolCapabilities object (overrides individual capability params)
        max_concurrent: Max concurrent executions for rate-limited APIs
        idempotent: Whether repeated calls with same params are safe
        requires_confirmation: Whether user confirmation is required before execution
        estimated_latency_ms: Typical execution time hint for scheduling
    
    Returns:
        FunctionTool instance that wraps the function
    
    Example:
        >>> @tool(name="search", description="Search for information")
        ... def search(query: str, limit: int = 5) -> str:
        ...     return f"Results for: {query}"
        ...
        >>> search(query="python")  # Direct call
        'Results for: python'
        >>> registry.register(search)  # Register as BaseTool
    
    Effect System:
        >>> @tool(description="Fetch from DB", effects=["db", "cache"])
        ... async def fetch_user(user_id: str, db: Database) -> str:
        ...     return await db.fetch_one("SELECT * FROM users WHERE id = $1", user_id)
        ...
        >>> # Test without mocks using pure handlers
        >>> from toolcase.foundation.effects import test_effects, InMemoryDB
        >>> async with test_effects(db=InMemoryDB()):
        ...     result = await fetch_user(user_id="123")
    
    Capability Negotiation:
        >>> @tool(
        ...     description="Call external API",
        ...     max_concurrent=5,  # Rate-limited
        ...     idempotent=False,  # Has side effects
        ...     estimated_latency_ms=200,
        ... )
        ... async def call_api(endpoint: str) -> str:
        ...     ...
    
    Dependency Injection:
        >>> @tool(description="Fetch data from database", inject=["db"])
        ... async def fetch_data(query: str, db: Database) -> str:
        ...     result = await db.fetch(query)
        ...     return str(result)
    
    Streaming with Backpressure:
        >>> @tool(description="Generate report", streaming=True, backpressure_buffer=10)
        ... async def generate(topic: str) -> AsyncIterator[str]:
        ...     async for chunk in llm.stream(f"Report on {topic}"):
        ...         yield chunk  # Pauses if consumer is slow
        ...
        >>> async for chunk in registry.stream_execute("generate", {"topic": "AI"}):
        ...     await slow_process(chunk)  # Producer pauses when buffer fills
    
    Notes:
        - Function must return str (or AsyncIterator[str] for streaming)
        - All parameters must have type hints
        - Async functions are fully supported
        - streaming=True with async generator enables result streaming
        - backpressure_buffer prevents memory buildup with slow consumers
        - Injected parameters are excluded from the generated schema
        - Effects are tracked for verification and pure testing
    """
    def decorator(fn: Callable[P, str]) -> FunctionTool:
        tool_name = name or _to_snake_case(fn.__name__)
        tool_desc = description or _extract_description(fn.__doc__) or f"Execute {tool_name}"
        if len(tool_desc) < 10:
            tool_desc = f"{tool_desc} - automatically generated tool"
        
        # Build capabilities (explicit object or from individual params)
        caps = capabilities or ToolCapabilities(
            supports_caching=cache_enabled,
            supports_streaming=streaming,
            max_concurrent=max_concurrent,
            idempotent=idempotent,
            requires_confirmation=requires_confirmation,
            estimated_latency_ms=estimated_latency_ms,
        )
        
        meta = ToolMetadata(
            name=tool_name, description=tool_desc, category=category,
            tags=frozenset(tags or ()), requires_api_key=requires_api_key,
            streaming=streaming, capabilities=caps,
        )
        schema = _generate_schema(fn, f"{_to_pascal_case(tool_name)}Params", exclude=inject or [])
        
        # Normalize effects to frozenset
        effect_set = frozenset(e.lower() if isinstance(e, str) else e for e in (effects or ()))
        
        # Determine tool class based on function type and streaming flag
        is_result_streaming = streaming and inspect.isasyncgenfunction(fn)
        tool_cls = (ResultStreamingFunctionTool if is_result_streaming else
                    StreamingFunctionTool if streaming and asyncio.iscoroutinefunction(fn) else FunctionTool)
        
        # Pass backpressure_buffer only to ResultStreamingFunctionTool
        if is_result_streaming:
            tool_instance = tool_cls(fn, meta, schema, cache_enabled=cache_enabled, cache_ttl=cache_ttl, 
                                     inject=inject, effects=effect_set, backpressure_buffer=backpressure_buffer)
        else:
            tool_instance = tool_cls(fn, meta, schema, cache_enabled=cache_enabled, cache_ttl=cache_ttl, 
                                     inject=inject, effects=effect_set)
        wraps(fn)(tool_instance)
        return tool_instance
    
    return decorator(func) if func is not None else decorator


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _to_snake_case(name: str) -> str:
    """Convert CamelCase or mixed to snake_case."""
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)).lower()


def _to_pascal_case(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(word.capitalize() for word in name.split("_"))


def _extract_description(docstring: str | None) -> str | None:
    """Extract first line of docstring as description."""
    return docstring.strip().split("\n")[0].strip() if docstring else None
