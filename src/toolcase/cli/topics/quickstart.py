QUICKSTART = """
TOPIC: quickstart
=================

Complete guide to building an AI agent with toolcase.

STEP 1: INSTALL
    pip install toolcase
    
    # Optional backends
    pip install toolcase[mcp]       # MCP protocol support
    pip install toolcase[http]      # HTTP server support
    pip install toolcase[redis]     # Redis cache backend

STEP 2: CREATE TOOLS
    from toolcase import tool, BaseTool, ToolMetadata
    from pydantic import BaseModel, Field
    
    # Simple decorator-based tool
    @tool(description="Search the web for information")
    def search(query: str, limit: int = 5) -> str:
        return f"Results for: {query}"
    
    # Class-based tool (for complex logic)
    class AnalyzeParams(BaseModel):
        text: str = Field(..., description="Text to analyze")
        mode: str = Field(default="summary", description="Analysis mode")
    
    class AnalyzeTool(BaseTool[AnalyzeParams]):
        metadata = ToolMetadata(
            name="analyze",
            description="Analyze text content",
            category="analysis",
        )
        params_schema = AnalyzeParams
        
        async def _async_run(self, params: AnalyzeParams) -> str:
            return f"Analysis ({params.mode}): {params.text[:50]}..."

STEP 3: REGISTER & CONFIGURE
    from toolcase import get_registry, init_tools
    
    # Quick setup with DiscoveryTool
    registry = init_tools(search, AnalyzeTool())
    
    # Or manual registration
    registry = get_registry()
    registry.register(search)
    registry.register(AnalyzeTool())
    
    # Add middleware
    from toolcase import LoggingMiddleware, TimeoutMiddleware
    registry.use(LoggingMiddleware())
    registry.use(TimeoutMiddleware(30.0))

STEP 4: CONNECT TO AI PROVIDER
    # OpenAI
    from toolcase.ext.integrations.frontiers import to_openai
    openai_tools = to_openai(registry)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=openai_tools,
    )
    
    # Anthropic
    from toolcase.ext.integrations.frontiers import to_anthropic
    anthropic_tools = to_anthropic(registry)
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=messages,
        tools=anthropic_tools,
    )
    
    # Google Gemini
    from toolcase.ext.integrations.frontiers import to_google
    gemini_tools = to_google(registry)

STEP 5: EXECUTE TOOL CALLS
    # When AI requests a tool call
    async def handle_tool_call(name: str, params: dict) -> str:
        return await registry.execute(name, params)
    
    # With Result type for error handling
    from toolcase import Ok, Err
    
    result = await registry.execute_result(name, params)
    match result:
        case Ok(value):
            return {"status": "success", "content": value}
        case Err(error):
            return {"status": "error", "message": str(error)}

STEP 6: SERVE (OPTIONAL)
    # As MCP server (Cursor, Claude Desktop)
    from toolcase.ext.mcp import serve_mcp
    serve_mcp(registry, transport="sse", port=8080)
    
    # As HTTP REST API
    from toolcase.ext.mcp import serve_http
    serve_http(registry, port=8000)
    
    # Endpoints:
    # GET  /tools         → List available tools
    # POST /tools/{name}  → Invoke tool with JSON body

COMPLETE AGENT EXAMPLE:
    from toolcase import init_tools, get_registry
    from toolcase.foundation.config import load_env, get_env
    from toolcase.ext.integrations.frontiers import to_anthropic
    from toolcase.tools.prebuilt.web import (
        WebSearchTool, UrlFetchTool, HtmlParseTool, free_search
    )
    import anthropic
    
    # Load environment
    load_env()
    
    # Create registry with web tools
    registry = init_tools(
        free_search(),      # No API key needed
        UrlFetchTool(),
        HtmlParseTool(),
    )
    
    # Format tools for Claude
    tools = to_anthropic(registry)
    
    # Run agent loop
    client = anthropic.Anthropic(api_key=get_env("ANTHROPIC_API_KEY"))
    messages = [{"role": "user", "content": "Search for Python async patterns"}]
    
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            messages=messages,
            tools=tools,
        )
        
        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = await registry.execute(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        
        # Add to conversation
        messages.append({"role": "assistant", "content": response.content})
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            break  # Done when no more tool calls
    
    # Extract final answer
    for block in response.content:
        if block.type == "text":
            print(block.text)

ENVIRONMENT SETUP:
    # .env file
    ANTHROPIC_API_KEY=sk-ant-xxx
    OPENAI_API_KEY=sk-xxx
    TAVILY_API_KEY=tvly-xxx        # For premium search
    
    # Load in code
    from toolcase import load_env, get_env, require_env
    
    load_env()                      # Auto-detects .env files
    api_key = get_env("API_KEY")    # Optional with default
    api_key = require_env("API_KEY") # Required, raises if missing

RELATED TOPICS:
    toolcase help tool         Creating tools
    toolcase help registry     Tool registration
    toolcase help formats      Multi-provider format converters
    toolcase help web          Built-in web tools
    toolcase help mcp          MCP server setup
    toolcase help settings     Environment configuration
"""
