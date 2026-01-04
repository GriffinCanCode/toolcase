OVERVIEW = """
TOPIC: overview
===============

Toolcase is a type-safe, extensible tool framework for AI agents.

PURPOSE:
    Create tools that AI agents (LangChain, Claude, GPT, etc.) can invoke
    with type-safe parameters, automatic error handling, and caching.

KEY FEATURES:
    - Type-safe parameters via Pydantic
    - Monadic error handling (Result/Either types)
    - Built-in caching with TTL
    - Async/sync interoperability
    - Progress streaming for long operations
    - Multi-framework format converters (OpenAI, Anthropic, Google)
    - LangChain and MCP protocol integration

QUICK START:
    from toolcase import tool, get_registry
    
    @tool(description="Search the web")
    def search(query: str, limit: int = 5) -> str:
        return f"Results for: {query}"
    
    registry = get_registry()
    registry.register(search)
    result = search(query="python")

RELATED TOPICS:
    toolcase help tool         Creating tools
    toolcase help registry     Tool registration and discovery
    toolcase help formats      Multi-framework format conversion
"""
