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

RELATED TOPICS:
    toolcase help registry   Tool registration
    toolcase help formats    Format converters for other frameworks
    toolcase help http       HTTP tool for making requests
"""
