"""Toolcase - Type-safe, extensible tool framework for AI agents.

A minimal yet powerful framework for creating tools that AI agents can invoke.
Supports type-safe parameters, caching, progress streaming, and optional
LangChain integration.

Quick Start:
    >>> from toolcase import BaseTool, ToolMetadata, get_registry
    >>> from pydantic import BaseModel, Field
    >>>
    >>> class SearchParams(BaseModel):
    ...     query: str = Field(..., description="Search query")
    ...
    >>> class SearchTool(BaseTool[SearchParams]):
    ...     metadata = ToolMetadata(
    ...         name="search",
    ...         description="Search for information",
    ...         category="search",
    ...     )
    ...     params_schema = SearchParams
    ...
    ...     def _run(self, params: SearchParams) -> str:
    ...         return f"Results for: {params.query}"
    >>>
    >>> registry = get_registry()
    >>> registry.register(SearchTool())
    >>> registry["search"](query="python")
    'Results for: python'

For LangChain integration:
    >>> from toolcase.integrations import to_langchain_tools
    >>> lc_tools = to_langchain_tools(registry)
"""

from __future__ import annotations

__version__ = "0.1.0"

# Core
from .core import BaseTool, EmptyParams, ToolMetadata

# Errors
from .errors import ErrorCode, ToolError, ToolException, classify_exception

# Progress
from .progress import (
    ProgressCallback,
    ProgressKind,
    ToolProgress,
    complete,
    error,
    source_found,
    status,
    step,
)

# Cache
from .cache import (
    DEFAULT_TTL,
    CacheBackend,
    MemoryCache,
    ToolCache,
    get_cache,
    reset_cache,
    set_cache,
)

# Registry
from .registry import (
    ToolRegistry,
    get_registry,
    reset_registry,
    set_registry,
)

# Built-in tools
from .tools import DiscoveryParams, DiscoveryTool

__all__ = [
    # Version
    "__version__",
    # Core
    "BaseTool",
    "ToolMetadata",
    "EmptyParams",
    # Errors
    "ErrorCode",
    "ToolError",
    "ToolException",
    "classify_exception",
    # Progress
    "ToolProgress",
    "ProgressKind",
    "ProgressCallback",
    "status",
    "step",
    "source_found",
    "complete",
    "error",
    # Cache
    "ToolCache",
    "MemoryCache",
    "CacheBackend",
    "get_cache",
    "set_cache",
    "reset_cache",
    "DEFAULT_TTL",
    # Registry
    "ToolRegistry",
    "get_registry",
    "set_registry",
    "reset_registry",
    # Built-in tools
    "DiscoveryTool",
    "DiscoveryParams",
    # Convenience
    "init_tools",
]


def init_tools(*tools: BaseTool) -> ToolRegistry:  # type: ignore[type-arg]
    """Initialize the registry with tools.
    
    Convenience function that registers the discovery tool and any
    additional tools provided.
    
    Args:
        *tools: Additional tool instances to register
    
    Returns:
        The initialized global registry
    
    Example:
        >>> from toolcase import init_tools
        >>> registry = init_tools(MyTool(), AnotherTool())
    """
    registry = get_registry()
    
    # Register discovery tool first
    registry.register(DiscoveryTool())
    
    # Register user-provided tools
    for tool in tools:
        registry.register(tool)
    
    return registry
