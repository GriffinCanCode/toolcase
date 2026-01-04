# Toolcase

Type-safe, extensible tool framework for AI agents.

## Features

- **Async-first design** with sync compatibility
- **Type-safe parameters** via Pydantic generics
- **Monadic error handling** with Result types
- **Multi-framework converters** (OpenAI, Anthropic, Google)
- **MCP protocol & HTTP server** for Cursor/Claude Desktop
- **Middleware pipeline** (logging, retry, timeout, rate limiting)
- **Agentic primitives** (router, fallback, race, gate)
- **Structured concurrency** with TaskGroup and CancelScope
- **Distributed tracing** with multiple exporters
- **Built-in caching** with TTL and Redis support

## Installation

```bash
pip install toolcase

# With optional integrations
pip install toolcase[langchain]  # LangChain
pip install toolcase[mcp]        # MCP protocol
pip install toolcase[http]       # HTTP server
```

## Quick Start

```python
from toolcase import tool, get_registry

@tool(description="Search the web for information")
async def search(query: str, limit: int = 5) -> str:
    return f"Found {limit} results for: {query}"

registry = get_registry()
registry.register(search)
result = await registry.execute("search", {"query": "python", "limit": 3})
```

### Class-Based Tools

```python
from pydantic import BaseModel, Field
from toolcase import BaseTool, ToolMetadata

class SearchParams(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=5, ge=1, le=20)

class SearchTool(BaseTool[SearchParams]):
    metadata = ToolMetadata(
        name="web_search",
        description="Search the web for information",
        category="search",
    )
    params_schema = SearchParams

    async def _async_run(self, params: SearchParams) -> str:
        return f"Found {params.limit} results for: {params.query}"
```

## Core Concepts

### Middleware

```python
from toolcase import LoggingMiddleware, RetryMiddleware, TimeoutMiddleware

registry.use(LoggingMiddleware())
registry.use(RetryMiddleware(max_retries=3))
registry.use(TimeoutMiddleware(30.0))
```

### Error Handling

```python
from toolcase import Ok, Err, try_tool_operation

def _run_result(self, params):
    return (
        self._validate(params)
        .flat_map(lambda p: self._fetch(p))
        .map(lambda d: self._format(d))
    )

# Or automatic exception handling
result = try_tool_operation("my_tool", lambda: risky_call())
```

See [docs/MONADIC_ERRORS.md](docs/MONADIC_ERRORS.md) for complete guide.

### Agentic Composition

```python
from toolcase import router, fallback, race, Route

# Conditional routing
smart_search = router(
    Route(lambda p: "code" in p["query"], code_search),
    default=web_search,
)

# Fallback chain
resilient = fallback(primary_api, backup_api, cache)

# Race (first success wins)
fastest = race(api_a, api_b, timeout=5.0)
```

### Multi-Framework Export

```python
from toolcase.ext.integrations.frontiers import to_openai, to_anthropic, to_google

openai_tools = to_openai(registry)
anthropic_tools = to_anthropic(registry)
gemini_tools = to_google(registry)
```

### MCP & HTTP Server

```python
from toolcase.ext.mcp import serve_mcp, serve_http

# For Cursor/Claude Desktop
serve_mcp(registry, transport="stdio")

# HTTP REST API
serve_http(registry, port=8000)
```

### Batch Execution

```python
from toolcase import BatchConfig

params_list = [{"query": q} for q in ["python", "rust", "go"]]
results = await tool.batch_run(params_list, BatchConfig(concurrency=5))
print(f"Success rate: {results.success_rate:.0%}")
```

## CLI Help

```bash
toolcase help              # List topics
toolcase help tool         # Tool creation
toolcase help middleware   # Middleware pipeline
toolcase help agents       # Agentic patterns
toolcase help mcp          # MCP/HTTP server
```

## API Reference

### Core
- `BaseTool[T]`, `ToolMetadata`, `ToolCapabilities`, `@tool`

### Errors
- `Result`, `Ok`, `Err`, `ErrorCode`, `ToolError`

### Runtime
- `Middleware`, `compose`, `pipeline`, `parallel`
- `router`, `fallback`, `race`, `gate`
- `Concurrency`, `TaskGroup`, `CancelScope`
- `BatchConfig`, `batch_execute`

### Observability
- `configure_tracing`, `configure_logging`
- `TracingMiddleware`, `LoggingMiddleware`

## License

MIT
