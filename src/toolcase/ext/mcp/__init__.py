"""MCP and HTTP server integration for toolcase.

Exposes toolcase tools via multiple protocols for different use cases:

**MCP Protocol** (Cursor, Claude Desktop, VS Code):
    >>> from toolcase.mcp import serve_mcp
    >>> serve_mcp(registry, transport="sse", port=8080)

**Full MCP with Resources and Prompts**:
    >>> from toolcase.mcp import create_mcp_server
    >>> server = create_mcp_server(registry)
    >>> 
    >>> @server.resource("config://app")
    ... async def app_config() -> str:
    ...     return '{"version": "1.0"}'
    >>> 
    >>> @server.prompt("summarize")
    ... def summarize(text: str) -> str:
    ...     return f"Summarize:\\n{text}"
    >>> 
    >>> server.run(transport="sse", port=8080)

**HTTP REST** (Web backends, microservices, custom agents):
    >>> from toolcase.mcp import serve_http
    >>> serve_http(registry, port=8000)
    # GET  /tools         → List tools
    # POST /tools/{name}  → Invoke tool

**Embed in FastAPI/Starlette**:
    >>> from toolcase.mcp import create_http_app
    >>> app = create_http_app(registry)

Dependencies:
    pip install toolcase[mcp]   # FastMCP for MCP protocol
    pip install toolcase[http]  # Starlette + uvicorn for HTTP
"""

from __future__ import annotations

__all__ = [
    # Server classes
    "ToolServer",
    "MCPServer",
    "HTTPToolServer",
    # Factory functions
    "serve_mcp",
    "serve_http",
    "create_mcp_server",
    "create_http_app",
    "create_tool_routes",
    # Bridge utilities
    "tool_to_handler",
    "registry_to_handlers",
]


def __getattr__(name: str):
    """Lazy imports to defer dependency loading."""
    # Server classes
    if name == "ToolServer":
        from .server import ToolServer
        return ToolServer
    if name == "MCPServer":
        from .server import MCPServer
        return MCPServer
    if name == "HTTPToolServer":
        from .server import HTTPToolServer
        return HTTPToolServer
    
    # Factory functions
    if name == "serve_mcp":
        from .server import serve_mcp
        return serve_mcp
    if name == "serve_http":
        from .server import serve_http
        return serve_http
    if name == "create_mcp_server":
        from .server import create_mcp_server
        return create_mcp_server
    if name == "create_http_app":
        from .server import create_http_app
        return create_http_app
    if name == "create_tool_routes":
        from .server import create_tool_routes
        return create_tool_routes
    
    # Bridge utilities
    if name == "tool_to_handler":
        from .bridge import tool_to_handler
        return tool_to_handler
    if name == "registry_to_handlers":
        from .bridge import registry_to_handlers
        return registry_to_handlers
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
