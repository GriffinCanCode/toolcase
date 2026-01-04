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
from typing import (
    TYPE_CHECKING,
    Callable,
    ParamSpec,
    TypeVar,
    get_origin,
    get_type_hints,
    overload,
)

from pydantic import BaseModel, Field, create_model

from toolcase.runtime.concurrency import to_thread

from toolcase.io.cache import DEFAULT_TTL
from toolcase.foundation.errors import ToolError, ToolException, ToolResult, classify_exception, ErrorTrace, Result
from toolcase.foundation.errors.result import _ERR, _OK
from toolcase.foundation.errors.types import _EMPTY_CONTEXTS, ErrorContext, JsonDict
from .base import BaseTool, ToolMetadata

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable

    from toolcase.io.progress import ToolProgress

P = ParamSpec("P")
T = TypeVar("T")

# Context variable for passing injected dependencies
_injected_deps: ContextVar[JsonDict] = ContextVar("injected_deps", default={})


def set_injected_deps(deps: JsonDict) -> None:
    """Set dependencies for the current execution context.
    
    Called by registry before tool execution to provide resolved dependencies.
    """
    _injected_deps.set(deps)


def clear_injected_deps() -> None:
    """Clear injected dependencies after execution."""
    _injected_deps.set({})


# ─────────────────────────────────────────────────────────────────────────────
# Docstring Parsing
# ─────────────────────────────────────────────────────────────────────────────

_PARAM_PATTERN = re.compile(
    r"^\s*(?P<name>\w+)\s*(?:\([^)]*\))?\s*:\s*(?P<desc>.+?)(?=\n\s*\w+\s*:|$)",
    re.MULTILINE | re.DOTALL,
)


def _parse_docstring_params(docstring: str | None) -> dict[str, str]:
    """Extract parameter descriptions from Google/NumPy style docstrings."""
    if not docstring:
        return {}
    
    # Find Args/Parameters section
    sections = re.split(r"\n\s*(?:Args|Arguments|Parameters)\s*:\s*\n", docstring, flags=re.IGNORECASE)
    if len(sections) < 2:
        return {}
    
    # Parse until next section or end
    args_section = re.split(r"\n\s*(?:Returns|Raises|Examples?|Notes?|Yields)\s*:", sections[1], flags=re.IGNORECASE)[0]
    
    params: dict[str, str] = {}
    for match in _PARAM_PATTERN.finditer(args_section):
        name = match.group("name")
        desc = " ".join(match.group("desc").split())  # Normalize whitespace
        params[name] = desc
    
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Schema Generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_schema(
    func: Callable[..., str],
    model_name: str,
    exclude: list[str] | None = None,
) -> type[BaseModel]:
    """Generate Pydantic model from function signature.
    
    Introspects type hints and defaults to build Field definitions.
    Extracts descriptions from docstring if available.
    
    Args:
        func: Function to introspect
        model_name: Name for the generated model
        exclude: Parameter names to exclude (e.g., injected dependencies)
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    param_docs = _parse_docstring_params(func.__doc__)
    excluded = set(exclude or [])
    
    fields: dict[str, tuple[type, object]] = {}
    
    for name, param in sig.parameters.items():
        if name in ("self", "cls") or name in excluded:
            continue
        
        # Get type hint (default to str if missing)
        field_type = hints.get(name, str)
        
        # Get description from docstring
        description = param_docs.get(name, f"Parameter: {name}")
        
        # Build field with default if present
        if param.default is inspect.Parameter.empty:
            fields[name] = (field_type, Field(..., description=description))
        else:
            fields[name] = (field_type, Field(default=param.default, description=description))
    
    model: type[BaseModel] = create_model(model_name, **fields)  # type: ignore[call-overload]
    return model


# ─────────────────────────────────────────────────────────────────────────────
# FunctionTool: BaseTool wrapper for functions
# ─────────────────────────────────────────────────────────────────────────────

class FunctionTool(BaseTool[BaseModel]):
    """BaseTool implementation that wraps a decorated function.
    
    This class bridges the function-based API with the class-based BaseTool
    system, enabling full compatibility with registry, cache, and integrations.
    
    Supports dependency injection via the `inject` parameter, where declared
    dependencies are resolved from the registry's DI container at execution time.
    """
    
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
        self._func = func
        self._is_async = asyncio.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)
        self._original_func = func
        self._inject = inject or []
        # Pre-compute tool context tuple (avoid repeated allocation)
        self._tool_ctx = (ErrorContext(operation=f"tool:{metadata.name}", location="", metadata={}),)
        
        # Use the actual class being instantiated as base (supports subclasses)
        base_cls = type(self)
        
        # Set class-level attributes on the instance
        # This is necessary because BaseTool expects ClassVars
        self.__class__ = type(
            f"{base_cls.__name__}_{metadata.name}",
            (base_cls,),
            {
                "metadata": metadata,
                "params_schema": params_schema,
                "cache_enabled": cache_enabled,
                "cache_ttl": cache_ttl,
            },
        )
    
    def _run(self, params: BaseModel) -> str:
        """Execute the wrapped function synchronously.
        
        Merges validated params with any injected dependencies.
        Exceptions propagate to _run_result() which handles conversion to Result.
        """
        kwargs = params.model_dump()
        if self._inject:
            kwargs.update(_injected_deps.get())
        if self._is_async:
            return self._run_async_sync(self._func(**kwargs))  # type: ignore[arg-type]
        return self._func(**kwargs)  # type: ignore[return-value]
    
    def _run_result(self, params: BaseModel) -> ToolResult:
        """Execute with Result-based error handling (optimized path)."""
        try:
            kwargs = params.model_dump()
            if self._inject:
                kwargs.update(_injected_deps.get())
            if self._is_async:
                return Result(self._run_async_sync(self._func(**kwargs)), _OK)  # type: ignore[arg-type]
            return Result(self._func(**kwargs), _OK)  # type: ignore[arg-type]
        except Exception as e:
            return self._make_err(e, "execution")
    
    async def _async_run(self, params: BaseModel) -> str:
        """Execute the wrapped function asynchronously.
        
        Merges validated params with any injected dependencies.
        Exceptions propagate to _async_run_result() which handles conversion.
        """
        kwargs = params.model_dump()
        if self._inject:
            kwargs.update(_injected_deps.get())
        if self._is_async:
            result: str = await self._func(**kwargs)  # type: ignore[misc]
            return result
        return await to_thread(self._func, **kwargs)  # type: ignore[arg-type]
    
    async def _async_run_result(self, params: BaseModel) -> ToolResult:
        """Execute asynchronously with Result-based error handling."""
        try:
            kwargs = params.model_dump()
            if self._inject:
                kwargs.update(_injected_deps.get())
            if self._is_async:
                result: str = await self._func(**kwargs)  # type: ignore[misc]
            else:
                result = await asyncio.to_thread(self._func, **kwargs)  # type: ignore[arg-type]
            return Result(result, _OK)
        except Exception as e:
            return self._make_err(e, "async execution")
    
    def _make_err(self, exc: Exception, context: str) -> ToolResult:
        """Create Err result from exception (internal, optimized)."""
        import traceback
        msg = f"{context}: {exc}" if context else str(exc)
        return Result(
            ErrorTrace(message=msg, contexts=self._tool_ctx, error_code=classify_exception(exc).value, recoverable=True, details=traceback.format_exc()),
            _ERR,
        )
    
    @property
    def func(self) -> Callable[..., str]:
        """Access the original wrapped function."""
        return self._original_func  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Streaming Function Tools
# ─────────────────────────────────────────────────────────────────────────────

class StreamingFunctionTool(FunctionTool):
    """FunctionTool variant for async generator functions yielding ToolProgress.
    
    For progress streaming (status updates, step progress).
    """
    
    __slots__ = ()
    
    async def stream_run(self, params: BaseModel) -> AsyncIterator[ToolProgress]:
        """Stream progress events from the wrapped generator function."""
        kwargs = params.model_dump()
        if self._inject:
            kwargs.update(_injected_deps.get())
        gen = self._func(**kwargs)
        async for progress in gen:  # type: ignore[union-attr]
            yield progress


class ResultStreamingFunctionTool(FunctionTool):
    """FunctionTool variant for async generators yielding string chunks.
    
    For true result streaming (LLM outputs, incremental content).
    The wrapped function should be an async generator yielding strings.
    
    Example:
        >>> @tool(description="Generate report", streaming=True)
        ... async def generate(topic: str) -> AsyncIterator[str]:
        ...     async for chunk in llm.stream(prompt):
        ...         yield chunk
    """
    
    __slots__ = ()
    
    @property
    def supports_result_streaming(self) -> bool:
        return True
    
    async def stream_result(self, params: BaseModel) -> AsyncIterator[str]:
        """Stream string chunks from the wrapped async generator."""
        kwargs = params.model_dump()
        if self._inject:
            kwargs.update(_injected_deps.get())
        gen = self._func(**kwargs)
        async for chunk in gen:  # type: ignore[union-attr]
            yield chunk
    
    async def _async_run(self, params: BaseModel) -> str:
        """Execute by collecting all stream chunks."""
        parts: list[str] = []
        async for chunk in self.stream_result(params):
            parts.append(chunk)
        return "".join(parts)
    
    async def _async_run_result(self, params: BaseModel) -> ToolResult:
        """Execute by collecting all stream chunks, with Result handling."""
        try:
            parts: list[str] = []
            async for chunk in self.stream_result(params):
                parts.append(chunk)
            return Result("".join(parts), _OK)
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
        # Derive metadata from function if not provided
        tool_name = name or _to_snake_case(fn.__name__)
        tool_desc = description or _extract_description(fn.__doc__) or f"Execute {tool_name}"
        
        # Validate description length
        if len(tool_desc) < 10:
            tool_desc = f"{tool_desc} - automatically generated tool"
        
        # Build metadata
        meta = ToolMetadata(
            name=tool_name,
            description=tool_desc,
            category=category,
            requires_api_key=requires_api_key,
            streaming=streaming,
        )
        
        # Generate parameter schema (excluding injected params)
        schema_name = f"{_to_pascal_case(tool_name)}Params"
        schema = _generate_schema(fn, schema_name, exclude=inject or [])
        
        # Determine tool class based on function type and streaming flag
        is_async_gen = inspect.isasyncgenfunction(fn)
        
        if streaming and is_async_gen:
            # Async generator yielding strings -> result streaming
            tool_cls = ResultStreamingFunctionTool
        elif streaming and asyncio.iscoroutinefunction(fn):
            # Async coroutine with streaming -> progress streaming
            tool_cls = StreamingFunctionTool
        else:
            # Regular sync/async function
            tool_cls = FunctionTool
        
        tool_instance = tool_cls(
            fn,
            meta,
            schema,
            cache_enabled=cache_enabled,
            cache_ttl=cache_ttl,
            inject=inject,
        )
        
        # Preserve function metadata
        wraps(fn)(tool_instance)
        
        return tool_instance
    
    # Support both @tool and @tool(...) syntax
    if func is not None:
        return decorator(func)
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _to_snake_case(name: str) -> str:
    """Convert CamelCase or mixed to snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _to_pascal_case(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(word.capitalize() for word in name.split("_"))


def _extract_description(docstring: str | None) -> str | None:
    """Extract first line of docstring as description."""
    if not docstring:
        return None
    lines = docstring.strip().split("\n")
    return lines[0].strip() if lines else None
