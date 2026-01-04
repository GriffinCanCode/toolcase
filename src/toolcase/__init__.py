"""Toolcase - Type-safe, extensible tool framework for AI agents.

A minimal yet powerful framework for creating tools that AI agents can invoke.
Supports type-safe parameters, caching, progress streaming, and multi-framework
format converters for OpenAI, Anthropic, Google Gemini, LangChain, and MCP.

Quick Start (Decorator - Recommended):
    >>> from toolcase import tool, get_registry
    >>>
    >>> @tool(description="Search for information", category="search")
    ... def search(query: str, limit: int = 5) -> str:
    ...     '''Search the web.
    ...     
    ...     Args:
    ...         query: Search query string
    ...         limit: Max results to return
    ...     '''
    ...     return f"Results for: {query}"
    >>>
    >>> registry = get_registry()
    >>> registry.register(search)
    >>> search(query="python")
    'Results for: python'

Class-Based (For Complex Tools):
    >>> from toolcase import BaseTool, ToolMetadata
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

Multi-Framework Format Converters:
    >>> from toolcase.formats import to_openai, to_anthropic, to_google
    >>>
    >>> # OpenAI function calling format
    >>> openai_tools = to_openai(registry)
    >>> 
    >>> # Anthropic tool_use format
    >>> anthropic_tools = to_anthropic(registry)
    >>> 
    >>> # Google Gemini function declarations
    >>> gemini_tools = to_google(registry)

LangChain Integration:
    >>> from toolcase.integrations import to_langchain_tools
    >>> lc_tools = to_langchain_tools(registry)

MCP (Model Context Protocol) Integration:
    >>> from toolcase.mcp import serve_mcp
    >>> serve_mcp(registry, transport="sse", port=8080)

HTTP REST Server (Web Backends):
    >>> from toolcase.mcp import serve_http
    >>> serve_http(registry, port=8000)  # Simple HTTP endpoints
"""

from __future__ import annotations

__version__ = "0.2.0"

# Core
from .core import BaseTool, EmptyParams, FunctionTool, StreamingFunctionTool, ToolMetadata, tool

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

# Middleware
from .middleware import (
    Context,
    Middleware,
    Next,
    compose,
    LoggingMiddleware,
    MetricsMiddleware,
    RateLimitMiddleware,
    RetryMiddleware,
    TimeoutMiddleware,
)

# Retry policies
from .retry import (
    Backoff,
    ConstantBackoff,
    DecorrelatedJitter,
    ExponentialBackoff,
    LinearBackoff,
    RetryPolicy,
    DEFAULT_RETRYABLE,
    NO_RETRY,
)

# Pipeline composition
from .pipeline import (
    PipelineTool,
    ParallelTool,
    Step,
    pipeline,
    parallel,
)

# Built-in tools
from .tools import DiscoveryParams, DiscoveryTool

# Monadic error handling
from .monads import (
    Err,
    ErrorContext,
    ErrorTrace,
    Ok,
    Result,
    ResultT,
    ToolResult,
    batch_results,
    collect_results,
    sequence,
    tool_result,
    traverse,
    try_tool_operation,
    try_tool_operation_async,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "BaseTool",
    "ToolMetadata",
    "EmptyParams",
    "tool",
    "FunctionTool",
    "StreamingFunctionTool",
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
    # Middleware
    "Middleware",
    "Context",
    "Next",
    "compose",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "RateLimitMiddleware",
    "RetryMiddleware",
    "TimeoutMiddleware",
    # Retry policies
    "Backoff",
    "ExponentialBackoff",
    "LinearBackoff",
    "ConstantBackoff",
    "DecorrelatedJitter",
    "RetryPolicy",
    "DEFAULT_RETRYABLE",
    "NO_RETRY",
    # Pipeline composition
    "PipelineTool",
    "ParallelTool",
    "Step",
    "pipeline",
    "parallel",
    # Built-in tools
    "DiscoveryTool",
    "DiscoveryParams",
    # Monadic error handling
    "Result",
    "Ok",
    "Err",
    "ResultT",
    "ToolResult",
    "ErrorContext",
    "ErrorTrace",
    "tool_result",
    "try_tool_operation",
    "try_tool_operation_async",
    "batch_results",
    "sequence",
    "traverse",
    "collect_results",
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
