FORMATS = """
TOPIC: formats
==============

Multi-framework format converters for AI providers.

CONVERTERS:
    from toolcase.ext.integrations.frontiers import (
        to_openai, to_anthropic, to_google
    )
    
    registry = get_registry()
    
    # OpenAI function calling format
    openai_tools = to_openai(registry)
    
    # Anthropic tool_use format  
    anthropic_tools = to_anthropic(registry)
    
    # Google Gemini function declarations
    gemini_tools = to_google(registry)

SINGLE TOOL CONVERTERS:
    from toolcase.ext.integrations.frontiers import (
        tool_to_openai, tool_to_anthropic, tool_to_google
    )
    
    # Convert a single tool
    openai_spec = tool_to_openai(my_tool)
    anthropic_spec = tool_to_anthropic(my_tool)
    gemini_spec = tool_to_google(my_tool)

LANGCHAIN INTEGRATION:
    from toolcase.ext.integrations import to_langchain_tools
    
    lc_tools = to_langchain_tools(registry)
    
    # Use with LangChain agents
    from langchain.agents import AgentExecutor
    executor = AgentExecutor(agent=agent, tools=lc_tools)

MCP PROTOCOL:
    from toolcase.ext.mcp import serve_mcp
    
    # Expose via Model Context Protocol (Cursor, Claude Desktop)
    serve_mcp(registry, transport="sse", port=8080)

HTTP REST:
    from toolcase.ext.mcp import serve_http
    
    # Expose via HTTP REST endpoints
    serve_http(registry, port=8000)

RELATED TOPICS:
    toolcase help registry   Tool registration
    toolcase help tool       Creating tools
    toolcase help mcp        MCP protocol server
"""
