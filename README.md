# Toolcase

Type-safe, extensible tool framework for AI agents.

## Features

- **Async-first** with sync compatibility
- **Type-safe parameters** via Pydantic generics
- **Monadic error handling** with `Result[T, E]` types
- **Multi-framework converters** (OpenAI, Anthropic, Google)
- **MCP protocol & HTTP server** for Cursor/Claude Desktop
- **Middleware pipeline** (logging, retry, timeout, rate limiting, circuit breaker)
- **Agentic primitives** (router, fallback, race, gate, escalation)
- **Structured concurrency** with TaskGroup and CancelScope
- **Distributed tracing** (OTLP, Datadog, Honeycomb, Zipkin)
- **Caching** with TTL (memory, Redis, Memcached)
- **Dependency injection** with scoped lifecycle
- **Streaming** with SSE, WebSocket, JSON Lines adapters

## Installation

```bash
pip install toolcase

# Optional integrations
pip install toolcase[mcp]        # MCP protocol (Cursor, Claude Desktop)
pip install toolcase[http]       # HTTP REST server
pip install toolcase[langchain]  # LangChain tools
pip install toolcase[redis]      # Redis cache backend
pip install toolcase[otel]       # OpenTelemetry exporters
pip install toolcase[all]        # Everything
```

## Quick Start

```python
from toolcase import tool, init_tools

@tool(description="Search the web")
async def search(query: str, limit: int = 5) -> str:
    return f"Found {limit} results for: {query}"

registry = init_tools(search)  # Registers discovery tool + your tools
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
    metadata = ToolMetadata(name="search", description="Search the web", category="search")
    params_schema = SearchParams

    async def _async_run(self, params: SearchParams) -> str:
        return f"Found {params.limit} results for: {params.query}"
```

## Core Concepts

### Middleware Pipeline

```python
from toolcase import LoggingMiddleware, RetryMiddleware, TimeoutMiddleware

registry.use(LoggingMiddleware())
registry.use(RetryMiddleware(max_retries=3))
registry.use(TimeoutMiddleware(30.0))
```

### Monadic Error Handling

```python
from toolcase import Ok, Err, Result, try_tool_operation

def _run_result(self, params) -> Result[str, ToolError]:
    return (
        self._validate(params)
        .flat_map(lambda p: self._fetch(p))
        .map(lambda d: self._format(d))
    )

# Auto-wrap exceptions
result = try_tool_operation("my_tool", lambda: risky_call())
match result:
    case Ok(value): print(value)
    case Err(error): print(error.message)
```

See [docs/MONADIC_ERRORS.md](docs/MONADIC_ERRORS.md) for complete guide.

### Agentic Composition

```python
from toolcase import router, fallback, race, gate, Route

smart_search = router(Route(lambda p: "code" in p["query"], code_search), default=web_search)
resilient = fallback(primary_api, backup_api, cache)  # Try until success
fastest = race(api_a, api_b, timeout=5.0)             # First success wins
premium = gate(lambda ctx: ctx.get("is_premium"), expensive_tool)
```

### Multi-Framework Export

```python
from toolcase.foundation.formats import to_openai, to_anthropic, to_google

openai_tools = to_openai(registry)      # OpenAI function calling
anthropic_tools = to_anthropic(registry) # Anthropic tool_use
gemini_tools = to_google(registry)       # Google Gemini
```

### MCP & HTTP Server

```python
from toolcase.ext.mcp import serve_mcp, serve_http

serve_mcp(registry, transport="sse", port=8080)  # Cursor, Claude Desktop
serve_http(registry, port=8000)                   # REST API
```

### Dependency Injection

```python
from toolcase import Container, Scope

container = Container()
container.provide("db", lambda: Database(), Scope.SINGLETON)
container.provide("http", lambda: httpx.AsyncClient(), Scope.SCOPED)

async with container.scope() as ctx:
    db = await container.resolve("db", ctx)
```

### Batch Execution

```python
from toolcase import BatchConfig

params_list = [{"query": q} for q in ["python", "rust", "go"]]
results = await tool.batch_run(params_list, BatchConfig(concurrency=5))
print(f"Success: {results.success_rate:.0%}")
```

### Streaming

```python
from toolcase import sse_adapter

@tool(description="Stream response", streaming=True)
async def stream_search(query: str):
    for i in range(10):
        yield f"Result {i} for {query}"

async for event in sse_adapter(stream_search.stream({"query": "test"})):
    print(event)
```

### Observability

```python
from toolcase import configure_tracing, configure_logging, TracingMiddleware

configure_tracing(exporter="otlp", endpoint="http://localhost:4317")
configure_logging(level="INFO", json=True)
registry.use(TracingMiddleware())
```

### Testing

```python
from toolcase import ToolTestCase, mock_tool

class TestSearch(ToolTestCase):
    async def test_search(self):
        tool = mock_tool("search", return_value="mocked")
        result = await tool.run({"query": "test"})
        self.assert_success(result)
```

### Settings & Environment

```python
from toolcase import get_settings, load_env

load_env()  # Load .env files
settings = get_settings()
print(settings.cache.ttl, settings.retry.max_retries)
```

## CLI Help

```bash
toolcase help              # List topics
toolcase help tool         # Tool creation
toolcase help middleware   # Middleware pipeline
toolcase help agents       # Agentic patterns
toolcase help mcp          # MCP/HTTP server
toolcase help result       # Monadic errors
```

## API Reference

### Core
`BaseTool[T]` · `ToolMetadata` · `ToolCapabilities` · `@tool` · `init_tools`

### Errors
`Result[T, E]` · `Ok` · `Err` · `ErrorCode` · `ToolError` · `try_tool_operation`

### Runtime
`Middleware` · `compose` · `pipeline` · `parallel` · `streaming_pipeline`

### Agents
`router` · `fallback` · `race` · `gate` · `retry_with_escalation`

### Concurrency
`TaskGroup` · `CancelScope` · `Lock` · `Semaphore` · `run_sync`

### Observability
`configure_tracing` · `configure_logging` · `TracingMiddleware` · `Span`

### IO
`ToolCache` · `MemoryCache` · `StreamChunk` · `StreamEvent` · `BatchConfig`

## License

MIT
