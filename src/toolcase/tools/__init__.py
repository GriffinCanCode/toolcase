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

Individual Tools:
    >>> from toolcase.tools import HttpTool, HttpConfig, BearerAuth
    >>> 
    >>> http = HttpTool(HttpConfig(
    ...     allowed_hosts=["api.example.com"],
    ...     auth=BearerAuth(token="sk-xxx"),
    ... ))
    >>> registry.register(http)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import ConfigurableTool, DiscoveryParams, DiscoveryTool, ToolConfig
from .prebuilt import (
    ApiKeyAuth,
    BasicAuth,
    BearerAuth,
    CustomAuth,
    HttpConfig,
    HttpParams,
    HttpResponse,
    HttpTool,
    NoAuth,
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
    # Base classes
    "ConfigurableTool",
    "ToolConfig",
    # HTTP Tool
    "HttpTool",
    "HttpConfig",
    "HttpParams",
    "HttpResponse",
    # Auth strategies
    "NoAuth",
    "BearerAuth",
    "BasicAuth",
    "ApiKeyAuth",
    "CustomAuth",
    "get_no_auth",
    # Utility
    "standard_tools",
]
