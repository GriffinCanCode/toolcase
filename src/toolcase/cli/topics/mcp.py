MCP = """
TOPIC: mcp
==========

Model Context Protocol (MCP) and HTTP server integration.

CONCEPT:
    Expose toolcase tools via multiple protocols:
    - MCP: For Cursor, Claude Desktop, VS Code integrations
    - HTTP REST: For web backends, microservices, custom agents

MCP SERVER (Cursor, Claude Desktop):
    from toolcase import get_registry, init_tools
    from toolcase.ext.mcp import serve_mcp
    
    registry = init_tools(MyTool(), AnotherTool())
    
    # Start MCP server with SSE transport
    serve_mcp(registry, transport="sse", port=8080)
    
    # Or with stdio transport (for direct process communication)
    serve_mcp(registry, transport="stdio")

FULL MCP SERVER (Resources & Prompts):
    from toolcase.ext.mcp import create_mcp_server
    
    server = create_mcp_server(registry)
    
    # Add resources (static or dynamic content)
    @server.resource("config://app")
    async def app_config() -> str:
        return '{"version": "1.0", "env": "production"}'
    
    @server.resource("data://users/{user_id}")
    async def user_data(user_id: str) -> str:
        return f'{{"id": "{user_id}", "name": "User"}}'
    
    # Add prompt templates
    @server.prompt("summarize")
    def summarize(text: str) -> str:
        return f"Please summarize the following:\\n{text}"
    
    @server.prompt("analyze")
    def analyze(topic: str, depth: str = "brief") -> str:
        return f"Analyze {topic} with {depth} depth."
    
    # Run with transport
    server.run(transport="sse", port=8080)

HTTP REST SERVER (Web APIs):
    from toolcase.ext.mcp import serve_http
    
    # Starts HTTP server with standard REST endpoints:
    # GET  /tools         → List available tools
    # POST /tools/{name}  → Invoke tool with JSON body
    serve_http(registry, port=8000)

EMBED IN FASTAPI/STARLETTE:
    from toolcase.ext.mcp import create_http_app, create_tool_routes
    
    # Create standalone Starlette app
    app = create_http_app(registry)
    
    # Or get routes to mount in existing app
    routes = create_tool_routes(registry)
    
    # Mount in FastAPI
    from fastapi import FastAPI
    app = FastAPI()
    app.mount("/tools", create_http_app(registry))

SERVER CLASSES:
    from toolcase.ext.mcp import ToolServer, MCPServer, HTTPToolServer
    
    # Base class for custom servers
    class CustomServer(ToolServer):
        ...
    
    # MCP protocol server
    mcp = MCPServer(registry)
    mcp.run()
    
    # HTTP REST server
    http = HTTPToolServer(registry)
    http.run(port=8000)

BRIDGE UTILITIES:
    from toolcase.ext.mcp import tool_to_handler, registry_to_handlers
    
    # Convert single tool to MCP handler
    handler = tool_to_handler(my_tool)
    
    # Convert entire registry
    handlers = registry_to_handlers(registry)

DEPENDENCIES:
    # For MCP protocol support
    pip install toolcase[mcp]
    
    # For HTTP server support
    pip install toolcase[http]

EXAMPLE - CURSOR/CLAUDE DESKTOP CONFIG:
    Add to your MCP settings:
    
    {
        "mcpServers": {
            "my-tools": {
                "command": "python",
                "args": ["-m", "my_project.mcp_server"],
                "env": {}
            }
        }
    }
    
    # my_project/mcp_server.py:
    from toolcase import init_tools
    from toolcase.ext.mcp import serve_mcp
    from my_project.tools import MyTool, AnotherTool
    
    if __name__ == "__main__":
        registry = init_tools(MyTool(), AnotherTool())
        serve_mcp(registry, transport="stdio")

EXAMPLE - WEB BACKEND:
    # server.py
    from toolcase import init_tools
    from toolcase.ext.mcp import serve_http
    from my_tools import SearchTool, AnalyzeTool
    
    if __name__ == "__main__":
        registry = init_tools(SearchTool(), AnalyzeTool())
        serve_http(registry, port=8000)
    
    # Client usage:
    # curl http://localhost:8000/tools
    # curl -X POST http://localhost:8000/tools/search -d '{"query": "python"}'

RELATED TOPICS:
    toolcase help registry   Tool registration
    toolcase help formats    Format converters for other frameworks
    toolcase help http       HTTP tool for making requests
"""
