"""Built-in tools for toolcase.

A curated set of commonly-needed tools that work out of the box.
Every agent project reinvents file I/O, HTTP, shell access - these
well-tested defaults save that effort.

Quick Start:
    >>> from toolcase import get_registry
    >>> from toolcase.tools import standard_tools
    >>> 
    >>> registry = get_registry()
    >>> registry.register_all(*standard_tools())

Tool Discovery:
    >>> from toolcase.tools import ToolQuery, SchemaPattern, find_by_param
    >>> 
    >>> # Find tools accepting a 'query' string parameter
    >>> results = find_by_param(registry, "query", "string")
    >>> 
    >>> # Complex query with capabilities
    >>> results = (ToolQuery(registry)
    ...     .by_capability(["search", "web"])
    ...     .by_schema(SchemaPattern(type="string", required=True))
    ...     .execute())

Usage Statistics:
    >>> from toolcase.tools import get_stats, format_stats
    >>> 
    >>> stats = get_stats()
    >>> print(stats.get("search").success_rate)
    >>> print(format_stats(stats))

Individual Tools:
    >>> from toolcase.tools import HttpTool, HttpConfig, BearerAuth
    >>> 
    >>> http = HttpTool(HttpConfig(
    ...     allowed_hosts=["api.example.com"],
    ...     auth=BearerAuth(token="sk-xxx"),
    ... ))
    >>> registry.register(http)

Environment-Based Authentication:
    >>> from toolcase.tools import bearer_from_env, api_key_from_env
    >>> 
    >>> # Load secrets from environment variables (recommended for production)
    >>> http = HttpTool(HttpConfig(auth=bearer_from_env("OPENAI_API_KEY")))
    >>> 
    >>> # Or with custom header
    >>> http = HttpTool(HttpConfig(auth=api_key_from_env("ANTHROPIC_API_KEY", header="x-api-key")))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import (
    CapabilityFilter,
    ConfigurableTool,
    MatchMode,
    QueryResult,
    SchemaPattern,
    StatsMiddleware,
    ToolConfig,
    ToolQuery,
    ToolStats,
    UsageStats,
    find_by_input_type,
    find_by_max_concurrent,
    find_by_param,
    find_by_tags,
    find_cacheable,
    find_streamable,
    format_stats,
    get_stats,
    reset_stats,
    set_stats,
)
from .prebuilt import (
    ApiKeyAuth,
    BasicAuth,
    BearerAuth,
    CustomAuth,
    DiscoveryParams,
    DiscoveryTool,
    EnvApiKeyAuth,
    EnvBasicAuth,
    EnvBearerAuth,
    HttpConfig,
    HttpParams,
    HttpResponse,
    HttpTool,
    NoAuth,
    api_key_from_env,
    basic_from_env,
    bearer_from_env,
    get_no_auth,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from toolcase.foundation.core import BaseTool


def standard_tools() -> list[BaseTool[BaseModel]]:
    """Get all standard built-in tools with default configurations.
    
    Returns a list of tool instances ready for registration.
    Each tool uses sensible defaults but can be customized by
    creating instances directly with custom configs.
    
    Returns:
        List of tool instances to register
    
    Example:
        >>> registry.register_all(*standard_tools())
        
        >>> # Or selectively
        >>> for tool in standard_tools():
        ...     if tool.metadata.category == "network":
        ...         registry.register(tool)
    """
    return [
        DiscoveryTool(),
        HttpTool(),
    ]


__all__ = [
    # Discovery
    "DiscoveryTool",
    "DiscoveryParams",
    # Query engine
    "ToolQuery",
    "SchemaPattern",
    "CapabilityFilter",
    "QueryResult",
    "MatchMode",
    "find_by_param",
    "find_by_tags",
    "find_by_input_type",
    "find_streamable",
    "find_cacheable",
    "find_by_max_concurrent",
    # Statistics
    "UsageStats",
    "ToolStats",
    "StatsMiddleware",
    "get_stats",
    "set_stats",
    "reset_stats",
    "format_stats",
    # Base classes
    "ConfigurableTool",
    "ToolConfig",
    # HTTP Tool
    "HttpTool",
    "HttpConfig",
    "HttpParams",
    "HttpResponse",
    # Auth strategies (explicit secrets)
    "NoAuth",
    "BearerAuth",
    "BasicAuth",
    "ApiKeyAuth",
    "CustomAuth",
    "get_no_auth",
    # Auth strategies (environment-based)
    "EnvBearerAuth",
    "EnvApiKeyAuth",
    "EnvBasicAuth",
    "bearer_from_env",
    "api_key_from_env",
    "basic_from_env",
    # Utility
    "standard_tools",
]
