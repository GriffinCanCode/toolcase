"""Ext - External system integrations.

Contains: integrations (LangChain, Frontiers), mcp (Model Context Protocol).
"""

from __future__ import annotations

__all__ = [
    # Integrations
    "to_langchain", "to_langchain_tools",
    # MCP
    "ToolServer", "MCPServer", "HTTPToolServer",
    "serve_mcp", "serve_http", "create_mcp_server", "create_http_app", "create_tool_routes",
    "tool_to_handler", "registry_to_handlers",
]


def __getattr__(name: str):
    """Lazy imports to defer dependency loading."""
    # Integrations
    if name in ("to_langchain", "to_langchain_tools"):
        from .integrations import langchain
        return getattr(langchain, name)
    
    # MCP module
    mcp_attrs = {
        "ToolServer", "MCPServer", "HTTPToolServer",
        "serve_mcp", "serve_http", "create_mcp_server", "create_http_app", "create_tool_routes",
        "tool_to_handler", "registry_to_handlers",
    }
    if name in mcp_attrs:
        from . import mcp
        return getattr(mcp, name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
