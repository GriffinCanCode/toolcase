"""MCP Server implementations for toolcase.

Provides multiple server adapters for different deployment scenarios:

1. **FastMCP** - Full MCP protocol (Cursor, Claude Desktop, VS Code)
2. **HTTP/REST** - Simple HTTP endpoints for web backend agents
3. **ASGI** - Mount tools into FastAPI/Starlette apps

Example - FastMCP (MCP clients):
    >>> from toolcase.mcp import serve_mcp
    >>> serve_mcp(registry, transport="sse", port=8080)

Example - Resources and Prompts:
    >>> server = create_mcp_server(registry)
    >>> @server.resource("config://app")
    ... async def app_config() -> str:
    ...     return json.dumps({"version": "1.0"})
    >>> @server.prompt("summarize")
    ... def summarize(text: str) -> str:
    ...     return f"Summarize:\\n{text}"

Example - HTTP endpoints (web backends):
    >>> from toolcase.mcp import create_http_app
    >>> app = create_http_app(registry)  # Returns Starlette/FastAPI app

Requires: pip install toolcase[mcp] (for FastMCP)
         pip install toolcase[http] (for HTTP endpoints)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Literal, TypeVar

from pydantic import ValidationError

from toolcase.foundation.errors import ErrorCode, JsonDict, ToolError, ToolException, format_validation_error

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool
    from toolcase.foundation.registry import ToolRegistry

Transport = Literal["stdio", "sse", "streamable-http"]
F = TypeVar("F", bound=Callable[..., object])


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
    
    def list_tools(self) -> list[JsonDict]:
        """List all available tools with schemas."""
        from .bridge import get_required_params, get_tool_properties
        return [{
            "name": (m := t.metadata).name, "description": m.description, "category": m.category,
            "parameters": {"type": "object", "properties": get_tool_properties(t), "required": get_required_params(t)},
        } for t in self._registry if t.metadata.enabled]
    
    async def invoke(self, tool_name: str, params: JsonDict) -> str:
        """Invoke a tool by name with parameters.
        
        Uses registry.execute() when middleware is configured (including
        ValidationMiddleware), otherwise validates and executes directly.
        Returns structured error string on failure.
        """
        # Use registry's execute if middleware chain exists (includes validation)
        if self._registry._middleware:
            return await self._registry.execute(tool_name, params)
        
        # Direct execution path (no middleware)
        if (tool := self._registry.get(tool_name)) is None:
            return ToolError.create(tool_name, f"Tool '{tool_name}' not found", ErrorCode.NOT_FOUND, recoverable=False).render()
        try:
            validated = tool.params_schema(**params)
        except ValidationError as e:
            return ToolError.create(tool_name, format_validation_error(e, tool_name=tool_name), ErrorCode.INVALID_PARAMS, recoverable=False).render()
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
    
    Full MCP protocol support including tools, resources, and prompts.
    Compatible with Cursor, Claude Desktop, VS Code, and other MCP clients.
    
    Example - Basic:
        >>> server = MCPServer("my-tools", registry)
        >>> server.run(transport="sse", port=8080)
    
    Example - With Resources:
        >>> server = MCPServer("my-tools", registry)
        >>> @server.resource("config://app")
        ... async def app_config() -> str:
        ...     return json.dumps({"version": "1.0", "env": "prod"})
        >>> @server.resource("file://{path}")  # URI templates supported
        ... async def read_file(path: str) -> str:
        ...     return Path(path).read_text()
    
    Example - With Prompts:
        >>> @server.prompt("summarize")
        ... def summarize_prompt(content: str) -> str:
        ...     return f"Summarize the following:\\n\\n{content}"
    """
    
    __slots__ = ("_mcp", "_pending_resources", "_pending_prompts")
    
    def __init__(self, name: str, registry: ToolRegistry) -> None:
        # Store pending registrations before FastMCP is created
        self._pending_resources: list[tuple[str, Callable[..., object]]] = []
        self._pending_prompts: list[tuple[str, Callable[..., object]]] = []
        super().__init__(name, registry)
        self._mcp = self._create_server()
    
    def _create_server(self):
        """Create FastMCP server and register tools, resources, prompts."""
        try:
            from fastmcp import FastMCP
        except ImportError as e:
            raise ImportError("MCP integration requires fastmcp. Install with: pip install toolcase[mcp]") from e
        
        mcp = FastMCP(self._name)
        self._register_tools(mcp)
        # Apply any pending resource/prompt registrations
        for uri, fn in self._pending_resources:
            mcp.resource(uri)(fn)
        for name, fn in self._pending_prompts:
            mcp.prompt(name)(fn)
        return mcp
    
    def _register_tools(self, mcp) -> None:
        """Register all registry tools with FastMCP."""
        for tool in (t for t in self._registry if t.metadata.enabled):
            schema, name = tool.params_schema, tool.metadata.name
            if self._registry._middleware:
                async def handler(__name=name, **kwargs: object) -> str:
                    return await self._registry.execute(__name, kwargs)
            else:
                async def handler(__tool=tool, __schema=schema, **kwargs: object) -> str:
                    return await __tool.arun(__schema(**kwargs))  # type: ignore[arg-type]
            mcp.tool(name=name, description=tool.metadata.description)(handler)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Resource Registration (MCP primitive for exposing data)
    # ─────────────────────────────────────────────────────────────────────────────
    
    def resource(self, uri: str) -> Callable[[F], F]:
        """Register a resource handler exposed via URI.
        
        Resources provide contextual data to LLMs. URIs can include
        {param} placeholders for dynamic resources.
        
        Args:
            uri: Resource URI (e.g., "config://app", "file://{path}")
        
        Example:
            >>> @server.resource("db://users/{user_id}")
            ... async def get_user(user_id: str) -> str:
            ...     return json.dumps(await db.get_user(user_id))
        """
        def decorator(fn: F) -> F:
            if hasattr(self, "_mcp") and self._mcp:
                self._mcp.resource(uri)(fn)
            else:
                self._pending_resources.append((uri, fn))
            return fn
        return decorator
    
    def add_resource(self, uri: str, fn: Callable[..., object]) -> None:
        """Programmatically register a resource handler."""
        if hasattr(self, "_mcp") and self._mcp:
            self._mcp.resource(uri)(fn)
        else:
            self._pending_resources.append((uri, fn))
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Prompt Registration (MCP primitive for reusable templates)
    # ─────────────────────────────────────────────────────────────────────────────
    
    def prompt(self, name: str) -> Callable[[F], F]:
        """Register a prompt template.
        
        Prompts are reusable conversational templates that guide LLM behavior.
        They're surfaced to users in MCP clients.
        
        Args:
            name: Prompt identifier (e.g., "summarize", "code-review")
        
        Example:
            >>> @server.prompt("analyze-code")
            ... def code_review(code: str, language: str = "python") -> str:
            ...     return f"Review this {language} code:\\n```{language}\\n{code}\\n```"
        """
        def decorator(fn: F) -> F:
            if hasattr(self, "_mcp") and self._mcp:
                self._mcp.prompt(name)(fn)
            else:
                self._pending_prompts.append((name, fn))
            return fn
        return decorator
    
    def add_prompt(self, name: str, fn: Callable[..., object]) -> None:
        """Programmatically register a prompt template."""
        if hasattr(self, "_mcp") and self._mcp:
            self._mcp.prompt(name)(fn)
        else:
            self._pending_prompts.append((name, fn))
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Server Lifecycle
    # ─────────────────────────────────────────────────────────────────────────────
    
    def run(self, transport: Transport = "stdio", *, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Start MCP server. transport: "stdio" (CLI), "sse" (HTTP), "streamable-http"."""
        self._mcp.run() if transport == "stdio" else self._mcp.run(transport=transport, host=host, port=port)
    
    @property
    def fastmcp(self):
        """Access underlying FastMCP instance for advanced configuration."""
        return self._mcp
    
    def list_resources(self) -> list[str]:
        """List registered resource URIs."""
        return [uri for uri, _ in self._pending_resources]
    
    def list_prompts(self) -> list[str]:
        """List registered prompt names."""
        return [name for name, _ in self._pending_prompts]


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
            raise ImportError("HTTP server requires starlette. Install with: pip install toolcase[http]") from e
        
        async def list_tools(request):
            return JSONResponse({"server": self._name, "tools": self.list_tools()})
        
        async def invoke_tool(request):
            tool_name = request.path_params["name"]
            body = await request.json() if request.headers.get("content-type") == "application/json" else {}
            result = await self.invoke(tool_name, body)
            # Check if result is an error (starts with **Tool Error)
            if result.startswith("**Tool Error"):
                status = 404 if "not found" in result.lower() else 400 if "Invalid parameters" in result else 500
                return JSONResponse({"error": result}, status_code=status)
            return JSONResponse({"result": result})
        
        async def get_tool_schema(request):
            if (tool := self._registry.get(request.path_params["name"])) is None:
                return JSONResponse({"error": f"Tool '{request.path_params['name']}' not found"}, status_code=404)
            from .bridge import get_required_params, get_tool_properties
            return JSONResponse({
                "name": tool.metadata.name, "description": tool.metadata.description,
                "parameters": {"type": "object", "properties": get_tool_properties(tool), "required": get_required_params(tool)},
            })
        
        return Starlette(routes=[
            Route("/tools", list_tools, methods=["GET"]),
            Route("/tools/{name}", invoke_tool, methods=["POST"]),
            Route("/tools/{name}/schema", get_tool_schema, methods=["GET"]),
        ])
    
    def run(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Start HTTP server."""
        try:
            import uvicorn
        except ImportError as e:
            raise ImportError("HTTP server requires uvicorn. Install with: pip install toolcase[http]") from e
        uvicorn.run(self._app, host=host, port=port)
    
    @property
    def app(self):
        """Access ASGI app for embedding in larger applications."""
        return self._app


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Functions
# ═══════════════════════════════════════════════════════════════════════════════


def serve_mcp(registry: ToolRegistry, *, name: str = "toolcase", transport: Transport = "stdio", host: str = "127.0.0.1", port: int = 8080) -> None:
    """Expose tools via MCP protocol (Cursor, Claude Desktop, etc).
    
    Args:
        registry: Tool registry to expose
        name: Server name shown to clients
        transport: "stdio" (CLI), "sse" (HTTP), "streamable-http"
        host: Host for HTTP transports
        port: Port for HTTP transports
    """
    MCPServer(name, registry).run(transport=transport, host=host, port=port)


def serve_http(registry: ToolRegistry, *, name: str = "toolcase", host: str = "127.0.0.1", port: int = 8000) -> None:
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
    HTTPToolServer(name, registry).run(host=host, port=port)


def create_http_app(registry: ToolRegistry, name: str = "toolcase"):
    """Create ASGI app without running it. Use for embedding in existing FastAPI/Starlette apps.
    
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
    """Create MCP server without starting it. Returns server for resource/prompt registration.
    
    Example:
        >>> server = create_mcp_server(registry, "my-api")
        >>> 
        >>> @server.resource("config://app")
        ... async def app_config() -> str:
        ...     return json.dumps({"version": "1.0"})
        >>> 
        >>> @server.prompt("summarize")
        ... def summarize(text: str) -> str:
        ...     return f"Summarize:\\n{text}"
        >>> 
        >>> server.run(transport="sse", port=8080)
    
    Returns:
        MCPServer instance with full tools/resources/prompts support
    """
    return MCPServer(name, registry)


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI/Starlette Router Integration
# ═══════════════════════════════════════════════════════════════════════════════


def create_tool_routes(registry: ToolRegistry):
    """Create Starlette routes for tool endpoints. For manual integration into existing apps.
    
    Example:
        >>> from starlette.routing import Mount
        >>> routes = create_tool_routes(registry)
        >>> app = Starlette(routes=[Mount("/api/tools", routes=routes)])
    
    Returns:
        List of Starlette Route objects
    """
    try:
        from starlette.routing import Route
    except ImportError as e:
        raise ImportError("Route creation requires starlette. Install with: pip install starlette") from e
    return HTTPToolServer("tools", registry)._app.routes
