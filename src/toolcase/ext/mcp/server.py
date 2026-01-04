"""MCP Server implementations for toolcase.

Provides multiple server adapters for different deployment scenarios:

1. **FastMCP** - Full MCP protocol (Cursor, Claude Desktop, VS Code)
2. **HTTP/REST** - Simple HTTP endpoints for web backend agents
3. **ASGI** - Mount tools into FastAPI/Starlette apps

Example - FastMCP (MCP clients):
    >>> from toolcase.mcp import serve_mcp
    >>> serve_mcp(registry, transport="sse", port=8080)

Example - HTTP endpoints (web backends):
    >>> from toolcase.mcp import create_http_app
    >>> app = create_http_app(registry)  # Returns Starlette/FastAPI app

Example - Mount into existing FastAPI:
    >>> from toolcase.mcp import mount_tools
    >>> mount_tools(app, registry, prefix="/tools")

Requires: pip install toolcase[mcp] (for FastMCP)
         pip install toolcase[http] (for HTTP endpoints)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ValidationError

from toolcase.foundation.errors import ErrorCode, ToolError, ToolException

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool
    from toolcase.foundation.registry import ToolRegistry

Transport = Literal["stdio", "sse", "streamable-http"]


# ═══════════════════════════════════════════════════════════════════════════════
# Abstract Server Protocol
# ═══════════════════════════════════════════════════════════════════════════════


class ToolServer(ABC):
    """Abstract base for tool server implementations.
    
    Subclasses implement different transport/protocol adapters while
    sharing the same tool registration logic from the bridge module.
    """
    
    __slots__ = ("_name", "_registry")
    
    def __init__(self, name: str, registry: ToolRegistry) -> None:
        self._name = name
        self._registry = registry
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def registry(self) -> ToolRegistry:
        return self._registry
    
    @abstractmethod
    def run(self, **kwargs: object) -> None:
        """Start the server (blocking)."""
        ...
    
    def list_tools(self) -> list[dict[str, object]]:
        """List all available tools with schemas."""
        from .bridge import get_required_params, get_tool_properties
        
        return [
            {
                "name": tool.metadata.name,
                "description": tool.metadata.description,
                "category": tool.metadata.category,
                "parameters": {
                    "type": "object",
                    "properties": get_tool_properties(tool),
                    "required": get_required_params(tool),
                },
            }
            for tool in self._registry
            if tool.metadata.enabled
        ]
    
    async def invoke(self, tool_name: str, params: dict[str, object]) -> str:
        """Invoke a tool by name with parameters.
        
        Returns structured error string on failure instead of raising.
        """
        tool = self._registry.get(tool_name)
        if tool is None:
            return ToolError.create(
                tool_name, f"Tool '{tool_name}' not found",
                ErrorCode.NOT_FOUND, recoverable=False
            ).render()
        
        try:
            validated = tool.params_schema(**params)
        except ValidationError as e:
            return ToolError.create(
                tool_name, f"Invalid parameters: {e}",
                ErrorCode.INVALID_PARAMS, recoverable=False
            ).render()
        
        try:
            return await tool.arun(validated)  # type: ignore[arg-type]
        except ToolException as e:
            return e.error.render()
        except Exception as e:
            return ToolError.from_exception(tool_name, e, "Execution failed").render()


# ═══════════════════════════════════════════════════════════════════════════════
# FastMCP Adapter (Full MCP Protocol)
# ═══════════════════════════════════════════════════════════════════════════════


class MCPServer(ToolServer):
    """FastMCP-backed server for MCP clients.
    
    Full MCP protocol support for Cursor, Claude Desktop, VS Code, etc.
    
    Example:
        >>> server = MCPServer("my-tools", registry)
        >>> server.run(transport="sse", port=8080)
    """
    
    __slots__ = ("_mcp",)
    
    def __init__(self, name: str, registry: ToolRegistry) -> None:
        super().__init__(name, registry)
        self._mcp = self._create_server()
    
    def _create_server(self):
        """Create FastMCP server and register tools."""
        try:
            from fastmcp import FastMCP
        except ImportError as e:
            raise ImportError(
                "MCP integration requires fastmcp. "
                "Install with: pip install toolcase[mcp]"
            ) from e
        
        mcp = FastMCP(self._name)
        self._register_tools(mcp)
        return mcp
    
    def _register_tools(self, mcp) -> None:
        """Register all registry tools with FastMCP."""
        for tool in self._registry:
            if not tool.metadata.enabled:
                continue
            
            schema = tool.params_schema
            
            async def handler(__tool=tool, __schema=schema, **kwargs: object) -> str:
                params = __schema(**kwargs)
                return await __tool.arun(params)  # type: ignore[arg-type]
            
            mcp.tool(
                name=tool.metadata.name,
                description=tool.metadata.description,
            )(handler)
    
    def run(
        self,
        transport: Transport = "stdio",
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        """Start MCP server.
        
        Args:
            transport: "stdio" (CLI), "sse" (HTTP), "streamable-http"
            host: Host for HTTP transports
            port: Port for HTTP transports
        """
        if transport == "stdio":
            self._mcp.run()
        else:
            self._mcp.run(transport=transport, host=host, port=port)
    
    @property
    def fastmcp(self):
        """Access underlying FastMCP instance."""
        return self._mcp


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP REST Adapter (Web Backends)
# ═══════════════════════════════════════════════════════════════════════════════


class HTTPToolServer(ToolServer):
    """HTTP/REST server for web backend integration.
    
    No MCP protocol overhead - just simple HTTP endpoints:
    - GET  /tools         → List available tools
    - POST /tools/{name}  → Invoke tool with JSON body
    
    Perfect for:
    - Web backend agents
    - Microservice architectures
    - Custom agent frameworks
    
    Example:
        >>> server = HTTPToolServer("api", registry)
        >>> server.run(host="0.0.0.0", port=8000)
    """
    
    __slots__ = ("_app",)
    
    def __init__(self, name: str, registry: ToolRegistry) -> None:
        super().__init__(name, registry)
        self._app = self._create_app()
    
    def _create_app(self):
        """Create Starlette/FastAPI app with tool endpoints."""
        try:
            from starlette.applications import Starlette
            from starlette.responses import JSONResponse
            from starlette.routing import Route
        except ImportError as e:
            raise ImportError(
                "HTTP server requires starlette. "
                "Install with: pip install toolcase[http]"
            ) from e
        
        async def list_tools(request):
            return JSONResponse({
                "server": self._name,
                "tools": self.list_tools(),
            })
        
        async def invoke_tool(request):
            tool_name = request.path_params["name"]
            try:
                body = await request.json()
            except Exception:
                body = {}
            
            result = await self.invoke(tool_name, body)
            
            # Check if result is an error (starts with **Tool Error)
            if result.startswith("**Tool Error"):
                # Determine status code from error content
                status = 404 if "not found" in result.lower() else 400 if "Invalid parameters" in result else 500
                return JSONResponse({"error": result}, status_code=status)
            
            return JSONResponse({"result": result})
        
        async def get_tool_schema(request):
            tool_name = request.path_params["name"]
            tool = self._registry.get(tool_name)
            if tool is None:
                return JSONResponse({"error": f"Tool '{tool_name}' not found"}, status_code=404)
            
            from .bridge import get_required_params, get_tool_properties
            return JSONResponse({
                "name": tool.metadata.name,
                "description": tool.metadata.description,
                "parameters": {
                    "type": "object",
                    "properties": get_tool_properties(tool),
                    "required": get_required_params(tool),
                },
            })
        
        routes = [
            Route("/tools", list_tools, methods=["GET"]),
            Route("/tools/{name}", invoke_tool, methods=["POST"]),
            Route("/tools/{name}/schema", get_tool_schema, methods=["GET"]),
        ]
        
        return Starlette(routes=routes)
    
    def run(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Start HTTP server."""
        try:
            import uvicorn
        except ImportError as e:
            raise ImportError(
                "HTTP server requires uvicorn. "
                "Install with: pip install toolcase[http]"
            ) from e
        
        uvicorn.run(self._app, host=host, port=port)
    
    @property
    def app(self):
        """Access ASGI app for embedding in larger applications."""
        return self._app


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Functions
# ═══════════════════════════════════════════════════════════════════════════════


def serve_mcp(
    registry: ToolRegistry,
    *,
    name: str = "toolcase",
    transport: Transport = "stdio",
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Expose tools via MCP protocol (Cursor, Claude Desktop, etc).
    
    Args:
        registry: Tool registry to expose
        name: Server name shown to clients
        transport: "stdio" (CLI), "sse" (HTTP), "streamable-http"
        host: Host for HTTP transports
        port: Port for HTTP transports
    """
    server = MCPServer(name, registry)
    server.run(transport=transport, host=host, port=port)


def serve_http(
    registry: ToolRegistry,
    *,
    name: str = "toolcase",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Expose tools via HTTP REST endpoints (web backends).
    
    Endpoints:
        GET  /tools         → List tools with schemas
        POST /tools/{name}  → Invoke tool with JSON body
        GET  /tools/{name}/schema → Get tool schema
    
    Args:
        registry: Tool registry to expose
        name: Server name
        host: Host address
        port: Port number
    """
    server = HTTPToolServer(name, registry)
    server.run(host=host, port=port)


def create_http_app(registry: ToolRegistry, name: str = "toolcase"):
    """Create ASGI app without running it.
    
    Use for embedding in existing FastAPI/Starlette apps.
    
    Example:
        >>> from fastapi import FastAPI
        >>> from toolcase.mcp import create_http_app
        >>>
        >>> main_app = FastAPI()
        >>> tools_app = create_http_app(registry)
        >>> main_app.mount("/tools", tools_app)
    
    Returns:
        Starlette ASGI application
    """
    return HTTPToolServer(name, registry).app


def create_mcp_server(registry: ToolRegistry, name: str = "toolcase") -> MCPServer:
    """Create MCP server without starting it."""
    return MCPServer(name, registry)


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI/Starlette Router Integration
# ═══════════════════════════════════════════════════════════════════════════════


def create_tool_routes(registry: ToolRegistry):
    """Create Starlette routes for tool endpoints.
    
    For manual integration into existing apps.
    
    Example:
        >>> from starlette.routing import Mount
        >>> routes = create_tool_routes(registry)
        >>> app = Starlette(routes=[Mount("/api/tools", routes=routes)])
    
    Returns:
        List of Starlette Route objects
    """
    try:
        from starlette.responses import JSONResponse
        from starlette.routing import Route
    except ImportError as e:
        raise ImportError(
            "Route creation requires starlette. "
            "Install with: pip install starlette"
        ) from e
    
    server = HTTPToolServer("tools", registry)
    return server._app.routes
