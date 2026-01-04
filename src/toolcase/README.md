# Toolcase

**Type-safe, extensible tool framework for AI agents.**

A minimal yet powerful framework for creating tools that AI agents can invoke. Supports type-safe parameters, caching, progress streaming, and multi-framework format converters for OpenAI, Anthropic, Google Gemini, LangChain, and MCP.

## Architecture

The toolcase package is organized into modular components:

### Core (`toolcase.core`)
Foundation for building tools:
- **`BaseTool`** - Abstract base class for all tools
- **`ToolMetadata`** - Tool metadata and capabilities descriptor
- **`EmptyParams`** - Default parameter schema for parameterless tools
- **`@tool`** - Decorator to convert functions into tools
- **`FunctionTool`** / **`StreamingFunctionTool`** - Decorator implementations

### Registry (`toolcase.registry`)
Tool discovery and management:
- **`ToolRegistry`** - Central registry for all available tools
- **`get_registry()`** - Access the global registry singleton
- **`set_registry()`** / **`reset_registry()`** - Registry management

### Cache (`toolcase.cache`)
Result caching with TTL support:
- **`ToolCache`** - Abstract cache interface
- **`MemoryCache`** - Thread-safe in-memory cache implementation
- **`CacheBackend`** - Protocol for custom cache backends
- **`get_cache()`** / **`set_cache()`** / **`reset_cache()`** - Cache management

### Errors (`toolcase.errors`)
Standardized error handling:
- **`ToolError`** - Structured error response
- **`ErrorCode`** - Standard error codes (API_KEY_MISSING, RATE_LIMITED, etc.)
- **`ToolException`** - Exception wrapper for ToolError
- **`classify_exception()`** - Automatic error classification

### Progress (`toolcase.progress`)
Streaming progress for long-running operations:
- **`ToolProgress`** - Progress event model
- **`ProgressKind`** - Event types (STATUS, STEP, COMPLETE, ERROR)
- **`ProgressCallback`** - Callback protocol
- Factory functions: `status()`, `step()`, `source_found()`, `complete()`, `error()`

### Integrations (`toolcase.integrations`)
Multi-framework adapters:
- **`frontiers.py`** - Format converters for OpenAI, Anthropic, Google Gemini
  - `to_openai()`, `to_anthropic()`, `to_google()`, `to_provider()`
- **`langchain.py`** - LangChain StructuredTool integration
  - `to_langchain()`, `to_langchain_tools()`

### MCP (`toolcase.mcp`)
Server integration for multiple deployment scenarios:

**MCP Protocol** (Cursor, Claude Desktop, VS Code):
- **`MCPServer`** - Full MCP protocol server
- **`serve_mcp()`** - Run MCP server (stdio/sse transport)

**HTTP REST** (Web backends, microservices):
- **`HTTPToolServer`** - Simple HTTP endpoints
- **`serve_http()`** - Run HTTP server
- **`create_http_app()`** - ASGI app for embedding

**Shared**:
- **`ToolServer`** - Abstract base for custom servers
- **`tool_to_handler()`** - Convert BaseTool to handler

### Tools (`toolcase.tools`)
Built-in tools:
- **`DiscoveryTool`** - Meta-tool for listing available tools
- **`DiscoveryParams`** - Discovery tool parameters

### Formats (`toolcase.formats`)
Convenience re-export of format converters for simpler imports:
```python
from toolcase.formats import to_openai, to_anthropic, to_google
```

## Quick Start

### Decorator-Based (Recommended)
```python
from toolcase import tool, get_registry

@tool(description="Search for information", category="search")
def search(query: str, limit: int = 5) -> str:
    '''Search the web.
    
    Args:
        query: Search query string
        limit: Max results to return
    '''
    return f"Results for: {query}"

registry = get_registry()
registry.register(search)
search(query="python")
```

### Class-Based (For Complex Tools)
```python
from toolcase import BaseTool, ToolMetadata
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(..., description="Search query")

class SearchTool(BaseTool[SearchParams]):
    metadata = ToolMetadata(
        name="search",
        description="Search for information",
        category="search",
    )
    params_schema = SearchParams
    
    async def _async_run(self, params: SearchParams) -> str:
        return f"Results for: {params.query}"
```

### Multi-Framework Format Converters
```python
from toolcase.formats import to_openai, to_anthropic, to_google

# OpenAI function calling format
openai_tools = to_openai(registry)

# Anthropic tool_use format
anthropic_tools = to_anthropic(registry)

# Google Gemini function declarations
gemini_tools = to_google(registry)
```

### LangChain Integration
```python
from toolcase.integrations import to_langchain_tools

lc_tools = to_langchain_tools(registry)
```

### MCP Server (Cursor, Claude Desktop)
```python
from toolcase.mcp import serve_mcp

# stdio transport (default, for CLI tools)
serve_mcp(registry)

# SSE transport (for Cursor, Claude Desktop)
serve_mcp(registry, transport="sse", port=8080)
```

### HTTP REST Server (Web Backends)
```python
from toolcase.mcp import serve_http, create_http_app

# Standalone server
serve_http(registry, port=8000)
# GET  /tools         → List tools with schemas
# POST /tools/{name}  → Invoke tool with JSON body

# Or embed in existing FastAPI/Starlette app
from fastapi import FastAPI
app = FastAPI()
app.mount("/tools", create_http_app(registry))
```

## Design Principles

1. **Type Safety First** - Pydantic-powered schemas ensure type validation
2. **Minimal Dependencies** - Core functionality has minimal requirements
3. **Framework Agnostic** - Works with any LLM/agent framework
4. **Extensible** - Easy to add custom cache backends, error handlers, etc.
5. **Developer Experience** - Clean APIs with both decorator and class-based styles

## Module Guidelines

Each module is self-contained with:
- **Clear responsibility** - Single concern per module
- **Barrel exports** - `__init__.py` provides clean public API
- **Internal imports** - Uses relative imports within package
- **Documentation** - Module docstrings explain purpose and usage

## Testing

All modules can be imported and tested independently:
```python
# Test core abstractions
from toolcase.core import BaseTool, tool

# Test registry
from toolcase.registry import ToolRegistry

# Test cache
from toolcase.cache import MemoryCache

# Test integrations
from toolcase.integrations import to_openai

# Test MCP (requires pip install toolcase[mcp])
from toolcase.mcp import serve_mcp, MCPServer

# Test HTTP server (requires pip install toolcase[http])
from toolcase.mcp import serve_http, HTTPToolServer
```

## Contributing

When adding new functionality:
1. Place it in the appropriate module (or create a new one if needed)
2. Add exports to the module's `__init__.py`
3. Update the main `toolcase/__init__.py` if part of public API
4. Add documentation to this README
