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
"""

from __future__ import annotations

import asyncio
import inspect
import re
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Callable, ParamSpec, TypeVar, get_type_hints, overload

from pydantic import BaseModel, Field, create_model

from toolcase.runtime.concurrency import to_thread

from toolcase.io.cache import DEFAULT_TTL
from toolcase.foundation.errors import ToolResult, classify_exception, ErrorTrace, Result
from toolcase.foundation.errors.result import _ERR, _OK
from toolcase.foundation.errors.types import ErrorContext, JsonDict
from .base import BaseTool, ToolMetadata

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable

    from toolcase.io.progress import ToolProgress

P = ParamSpec("P")
T = TypeVar("T")

# Context variable for passing injected dependencies
# Note: No default set to avoid mutable default dict shared across contexts
_injected_deps: ContextVar[JsonDict] = ContextVar("injected_deps")

# Empty dict singleton for clear operations (avoids allocations)
_EMPTY_DEPS: JsonDict = {}


def get_injected_deps() -> JsonDict:
    """Get dependencies for the current execution context. Returns empty dict if not set."""
    return _injected_deps.get(_EMPTY_DEPS)


def set_injected_deps(deps: JsonDict) -> None:
    """Set dependencies for the current execution context. Called by registry before tool execution."""
    _injected_deps.set(deps)


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
    """BaseTool wrapper for decorated functions. Bridges function API with class-based system, supports DI via inject param."""
    
    __slots__ = ("_func", "_is_async", "_original_func", "_inject", "_tool_ctx")
    
    def __init__(
        self,
        func: Callable[..., str] | Callable[..., Awaitable[str]],
        metadata: ToolMetadata,
        params_schema: type[BaseModel],
        *,
        cache_enabled: bool = True,
        cache_ttl: float = DEFAULT_TTL,
        inject: list[str] | None = None,
    ) -> None:
        self._func, self._original_func = func, func
        self._is_async = asyncio.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)
        self._inject = inject or []
        self._tool_ctx = (ErrorContext(operation=f"tool:{metadata.name}", location="", metadata={}),)  # Pre-compute (avoid allocation)
        # Set class-level attrs (BaseTool expects ClassVars)
        self.__class__ = type(f"{type(self).__name__}_{metadata.name}", (type(self),),
                              {"metadata": metadata, "params_schema": params_schema, "cache_enabled": cache_enabled, "cache_ttl": cache_ttl})
    
    def _run(self, params: BaseModel) -> str:
        """Execute the wrapped function synchronously. Merges validated params with injected dependencies."""
        kwargs = params.model_dump() | (get_injected_deps() if self._inject else {})
        return self._run_async_sync(self._func(**kwargs)) if self._is_async else self._func(**kwargs)  # type: ignore[arg-type, return-value]
    
    def _run_result(self, params: BaseModel) -> ToolResult:
        """Execute with Result-based error handling (optimized path)."""
        try:
            kwargs = params.model_dump() | (get_injected_deps() if self._inject else {})
            return Result(self._run_async_sync(self._func(**kwargs)) if self._is_async else self._func(**kwargs), _OK)  # type: ignore[arg-type]
        except Exception as e:
            return self._make_err(e, "execution")
    
    async def _async_run(self, params: BaseModel) -> str:
        """Execute the wrapped function asynchronously. Merges validated params with injected dependencies."""
        kwargs = params.model_dump() | (get_injected_deps() if self._inject else {})
        return await self._func(**kwargs) if self._is_async else await to_thread(self._func, **kwargs)  # type: ignore[misc, arg-type]
    
    async def _async_run_result(self, params: BaseModel) -> ToolResult:
        """Execute asynchronously with Result-based error handling."""
        try:
            kwargs = params.model_dump() | (get_injected_deps() if self._inject else {})
            return Result(await self._func(**kwargs) if self._is_async else await asyncio.to_thread(self._func, **kwargs), _OK)  # type: ignore[misc, arg-type]
        except Exception as e:
            return self._make_err(e, "async execution")
    
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
        kwargs = params.model_dump() | (get_injected_deps() if self._inject else {})
        async for progress in self._func(**kwargs):  # type: ignore[union-attr]
            yield progress


class ResultStreamingFunctionTool(FunctionTool):
    """FunctionTool variant for async generators yielding string chunks. For true result streaming (LLM outputs, incremental content)."""
    
    __slots__ = ()
    
    @property
    def supports_result_streaming(self) -> bool:
        return True
    
    async def stream_result(self, params: BaseModel) -> AsyncIterator[str]:
        """Stream string chunks from the wrapped async generator."""
        kwargs = params.model_dump() | (get_injected_deps() if self._inject else {})
        async for chunk in self._func(**kwargs):  # type: ignore[union-attr]
            yield chunk
    
    async def _async_run(self, params: BaseModel) -> str:
        """Execute by collecting all stream chunks."""
        return "".join([chunk async for chunk in self.stream_result(params)])
    
    async def _async_run_result(self, params: BaseModel) -> ToolResult:
        """Execute by collecting all stream chunks, with Result handling."""
        try:
            return Result("".join([chunk async for chunk in self.stream_result(params)]), _OK)
        except Exception as e:
            return self._make_err(e, "async execution")
    
    def _run(self, params: BaseModel) -> str:
        """Sync execution via async collection."""
        return self._run_async_sync(self._async_run(params))
    
    def _run_result(self, params: BaseModel) -> ToolResult:
        """Sync execution via async collection, with Result handling."""
        try:
            return Result(self._run_async_sync(self._async_run(params)), _OK)
        except Exception as e:
            return self._make_err(e, "execution")


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
    requires_api_key: bool = False,
    streaming: bool = False,
    cache_enabled: bool = True,
    cache_ttl: float = DEFAULT_TTL,
    inject: list[str] | None = None,
) -> Callable[[Callable[P, str]], FunctionTool]: ...


def tool(
    func: Callable[P, str] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    category: str = "general",
    requires_api_key: bool = False,
    streaming: bool = False,
    cache_enabled: bool = True,
    cache_ttl: float = DEFAULT_TTL,
    inject: list[str] | None = None,
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
        requires_api_key: Whether tool needs external API credentials
        streaming: Whether tool supports progress streaming
        cache_enabled: Enable result caching
        cache_ttl: Cache TTL in seconds
        inject: List of dependency names to inject from registry container
    
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
    
    Dependency Injection:
        >>> @tool(description="Fetch data from database", inject=["db"])
        ... async def fetch_data(query: str, db: Database) -> str:
        ...     result = await db.fetch(query)
        ...     return str(result)
    
    Streaming Example:
        >>> @tool(description="Generate report", streaming=True)
        ... async def generate(topic: str) -> AsyncIterator[str]:
        ...     async for chunk in llm.stream(f"Report on {topic}"):
        ...         yield chunk
        ...
        >>> async for chunk in registry.stream_execute("generate", {"topic": "AI"}):
        ...     print(chunk, end="", flush=True)
    
    Notes:
        - Function must return str (or AsyncIterator[str] for streaming)
        - All parameters must have type hints
        - Async functions are fully supported
        - streaming=True with async generator enables result streaming
        - Injected parameters are excluded from the generated schema
    """
    def decorator(fn: Callable[P, str]) -> FunctionTool:
        tool_name = name or _to_snake_case(fn.__name__)
        tool_desc = description or _extract_description(fn.__doc__) or f"Execute {tool_name}"
        if len(tool_desc) < 10:
            tool_desc = f"{tool_desc} - automatically generated tool"
        
        meta = ToolMetadata(name=tool_name, description=tool_desc, category=category, requires_api_key=requires_api_key, streaming=streaming)
        schema = _generate_schema(fn, f"{_to_pascal_case(tool_name)}Params", exclude=inject or [])
        
        # Determine tool class based on function type and streaming flag
        tool_cls = (ResultStreamingFunctionTool if streaming and inspect.isasyncgenfunction(fn) else
                    StreamingFunctionTool if streaming and asyncio.iscoroutinefunction(fn) else FunctionTool)
        
        tool_instance = tool_cls(fn, meta, schema, cache_enabled=cache_enabled, cache_ttl=cache_ttl, inject=inject)
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
