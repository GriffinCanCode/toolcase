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
"""

from __future__ import annotations

import asyncio
import inspect
import re
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Callable,
    ParamSpec,
    TypeVar,
    get_type_hints,
    overload,
)

from pydantic import BaseModel, Field, create_model

from ..cache import DEFAULT_TTL
from ..errors import ToolError, ToolException
from .base import BaseTool, ToolMetadata

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable

    from .progress import ToolProgress

P = ParamSpec("P")
T = TypeVar("T")


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
) -> type[BaseModel]:
    """Generate Pydantic model from function signature.
    
    Introspects type hints and defaults to build Field definitions.
    Extracts descriptions from docstring if available.
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    param_docs = _parse_docstring_params(func.__doc__)
    
    fields: dict[str, tuple[type, object]] = {}
    
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
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
    """
    
    __slots__ = ("_func", "_is_async", "_original_func")
    
    def __init__(
        self,
        func: Callable[..., str] | Callable[..., Awaitable[str]],
        metadata: ToolMetadata,
        params_schema: type[BaseModel],
        *,
        cache_enabled: bool = True,
        cache_ttl: float = DEFAULT_TTL,
    ) -> None:
        self._func = func
        self._is_async = asyncio.iscoroutinefunction(func)
        self._original_func = func
        
        # Set class-level attributes on the instance
        # This is necessary because BaseTool expects ClassVars
        self.__class__ = type(
            f"FunctionTool_{metadata.name}",
            (FunctionTool,),
            {
                "metadata": metadata,
                "params_schema": params_schema,
                "cache_enabled": cache_enabled,
                "cache_ttl": cache_ttl,
            },
        )
    
    def _run(self, params: BaseModel) -> str:
        """Execute the wrapped function synchronously.
        
        Exceptions propagate to _run_result() which handles conversion to Result.
        """
        kwargs = params.model_dump()
        if self._is_async:
            return self._run_async_sync(self._func(**kwargs))  # type: ignore[arg-type]
        return self._func(**kwargs)  # type: ignore[return-value]
    
    async def _async_run(self, params: BaseModel) -> str:
        """Execute the wrapped function asynchronously.
        
        Exceptions propagate to _async_run_result() which handles conversion.
        """
        kwargs = params.model_dump()
        if self._is_async:
            result: str = await self._func(**kwargs)  # type: ignore[misc]
            return result
        return await asyncio.to_thread(self._func, **kwargs)  # type: ignore[arg-type]
    
    @property
    def func(self) -> Callable[..., str]:
        """Access the original wrapped function."""
        return self._original_func  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Streaming Function Tool
# ─────────────────────────────────────────────────────────────────────────────

class StreamingFunctionTool(FunctionTool):
    """FunctionTool variant for async generator functions.
    
    The wrapped function should be an async generator yielding ToolProgress events.
    """
    
    __slots__ = ()
    
    async def stream_run(self, params: BaseModel) -> AsyncIterator[ToolProgress]:
        """Stream progress events from the wrapped generator function."""
        kwargs = params.model_dump()
        gen = self._func(**kwargs)
        async for progress in gen:  # type: ignore[union-attr]
            yield progress


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
    
    Notes:
        - Function must return str
        - All parameters must have type hints
        - Async functions are fully supported
        - For streaming tools, use async generators yielding ToolProgress
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
        
        # Generate parameter schema
        schema_name = f"{_to_pascal_case(tool_name)}Params"
        schema = _generate_schema(fn, schema_name)
        
        # Create appropriate tool class
        tool_cls = StreamingFunctionTool if streaming and asyncio.iscoroutinefunction(fn) else FunctionTool
        
        tool_instance = tool_cls(
            fn,
            meta,
            schema,
            cache_enabled=cache_enabled,
            cache_ttl=cache_ttl,
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
