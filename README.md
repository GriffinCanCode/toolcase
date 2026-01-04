# Toolcase

Type-safe, extensible tool framework for AI agents.

## Features

- **Type-safe parameters** via Pydantic generics
- **Monadic error handling** with Result/Either types (NEW!)
- **Railway-oriented programming** for automatic error propagation
- **Standardized error handling** with error codes
- **Built-in caching** with TTL support
- **Async/sync interoperability**
- **Streaming progress** for long-running operations
- **Optional LangChain integration**
- **Central registry** for tool discovery

## Installation

```bash
pip install toolcase

# With LangChain support
pip install toolcase[langchain]
```

## Quick Start

```python
from pydantic import BaseModel, Field
from toolcase import BaseTool, ToolMetadata, init_tools

# Define parameters
class SearchParams(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=5, ge=1, le=20)

# Create a tool
class SearchTool(BaseTool[SearchParams]):
    metadata = ToolMetadata(
        name="web_search",
        description="Search the web for information",
        category="search",
    )
    params_schema = SearchParams

    def _run(self, params: SearchParams) -> str:
        # Your implementation here
        return f"Found {params.limit} results for: {params.query}"

# Register and use
registry = init_tools(SearchTool())
result = registry["web_search"](query="python tutorials", limit=3)
```

## Core Concepts

### BaseTool

All tools inherit from `BaseTool[TParams]` where `TParams` is a Pydantic model:

```python
class MyTool(BaseTool[MyParams]):
    metadata = ToolMetadata(...)
    params_schema = MyParams
    
    # Caching (optional)
    cache_enabled = True  # default
    cache_ttl = 300.0     # 5 minutes default
    
    def _run(self, params: MyParams) -> str:
        """Synchronous execution."""
        ...
    
    async def _async_run(self, params: MyParams) -> str:
        """Native async (optional override)."""
        ...
```

### Error Handling

#### String-Based (Traditional)

Use built-in error methods for consistent responses:

```python
def _run(self, params: MyParams) -> str:
    try:
        result = external_api_call()
        return format_result(result)
    except TimeoutError as e:
        return self._error("Request timed out", ErrorCode.TIMEOUT)
    except Exception as e:
        return self._error_from_exception(e, "API call failed")
```

#### Result-Based (Recommended - NEW!)

Use monadic Result types for type-safe error handling:

```python
from toolcase import Ok, ToolResult, try_tool_operation

def _run_result(self, params: MyParams) -> ToolResult:
    """Railway-oriented programming - errors propagate automatically."""
    return (
        self._validate_input(params)
        .flat_map(lambda p: self._fetch_data(p))
        .flat_map(lambda data: self._process_data(data))
        .map(lambda result: self._format_output(result))
    )

def _validate_input(self, params: MyParams) -> ToolResult:
    if not params.query:
        return tool_result(
            self.metadata.name,
            "Query required",
            code=ErrorCode.INVALID_PARAMS
        )
    return Ok(params)
```

Or use automatic exception handling:

```python
def _run_result(self, params: MyParams) -> ToolResult:
    return try_tool_operation(
        self.metadata.name,
        lambda: format_result(external_api_call()),
        context="calling external API"
    )
```

**Benefits**:
- ✅ Type-safe - compiler enforces error handling
- ✅ Composable - chain operations elegantly
- ✅ No manual error checking - errors propagate automatically
- ✅ Error context stacking - track provenance through call chains

See [MONADIC_ERRORS.md](MONADIC_ERRORS.md) for complete guide.

### Progress Streaming

For long-running tools, stream progress updates:

```python
from toolcase import ToolProgress, step, complete, status

class LongRunningTool(BaseTool[Params]):
    metadata = ToolMetadata(..., streaming=True)
    
    async def stream_run(self, params: Params) -> AsyncIterator[ToolProgress]:
        yield status("Starting...")
        
        for i, item in enumerate(params.items, 1):
            result = await process(item)
            yield step(f"Processed {item}", current=i, total=len(params.items))
        
        yield complete("All done!")
```

### Caching

Results are cached by default. Configure per-tool:

```python
class NoCacheTool(BaseTool[Params]):
    cache_enabled = False  # Disable caching

class ShortCacheTool(BaseTool[Params]):
    cache_ttl = 60.0  # 1 minute TTL
```

Or use a custom cache backend:

```python
from toolcase import set_cache, ToolCache

class RedisCache(ToolCache):
    def get(self, tool_name, params): ...
    def set(self, tool_name, params, value, ttl): ...
    # ... implement other methods

set_cache(RedisCache())
```

## LangChain Integration

```python
from toolcase import get_registry
from toolcase.integrations import to_langchain_tools

# Convert all tools to LangChain format
registry = get_registry()
lc_tools = to_langchain_tools(registry)

# Use with LangChain agents
from langchain.agents import AgentExecutor, create_tool_calling_agent
executor = AgentExecutor(agent=agent, tools=lc_tools)
```

## API Reference

### Core

- `BaseTool[TParams]` - Abstract base class for tools
- `ToolMetadata` - Tool metadata (name, description, category, etc.)
- `EmptyParams` - Default params for tools with no inputs

### Errors

- `ErrorCode` - Enum of standard error codes
- `ToolError` - Structured error response model
- `ToolException` - Exception wrapper for ToolError

### Progress

- `ToolProgress` - Progress event dataclass
- `ProgressKind` - Event types (status, step, complete, error)
- `status()`, `step()`, `complete()`, `error()` - Factory functions

### Cache

- `ToolCache` - Abstract cache interface
- `MemoryCache` - In-memory implementation
- `get_cache()`, `set_cache()`, `reset_cache()` - Global cache management

### Registry

- `ToolRegistry` - Tool container with discovery
- `get_registry()`, `set_registry()`, `reset_registry()` - Global registry
- `init_tools(*tools)` - Initialize registry with tools

## License

MIT
