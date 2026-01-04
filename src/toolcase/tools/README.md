# Tools

Built-in tools for toolcase—production-ready implementations that work out of the box.

## Structure

```
tools/
├── core/           # Base classes and infrastructure
│   ├── base.py     # ConfigurableTool, ToolConfig
│   └── discovery.py # DiscoveryTool (meta-tool for listing tools)
└── prebuilt/       # Ready-to-use tool implementations
    └── http.py     # HTTP requests with auth & security
```

## Quick Start

```python
from toolcase import get_registry
from toolcase.tools import standard_tools

registry = get_registry()
registry.register_all(*standard_tools())
```

## Available Tools

### DiscoveryTool
Meta-tool that lists all registered tools. Helps agents understand available capabilities.

```python
from toolcase.tools import DiscoveryTool

discovery = DiscoveryTool()
result = discovery.call(category="network", format="detailed")
```

### HttpTool
HTTP requests with security controls, authentication, and streaming.

```python
from toolcase.tools import HttpTool, HttpConfig, BearerAuth

# Basic usage
http = HttpTool()
result = await http.acall(url="https://api.github.com/users/octocat")

# Restricted configuration
http = HttpTool(HttpConfig(
    allowed_hosts=["api.example.com", "*.internal.corp"],
    allowed_methods=["GET", "POST"],
    auth=BearerAuth(token="sk-xxx"),
))
```

**Auth strategies:**
- `NoAuth` — No authentication (default)
- `BearerAuth` — OAuth2/JWT tokens
- `BasicAuth` — HTTP Basic auth
- `ApiKeyAuth` — API key in header
- `CustomAuth` — Arbitrary headers

## Creating Custom Tools

Extend `ConfigurableTool` for tools with runtime configuration:

```python
from toolcase.tools import ConfigurableTool, ToolConfig
from pydantic import BaseModel, Field

class MyConfig(ToolConfig):
    max_items: int = 100

class MyParams(BaseModel):
    query: str = Field(..., description="Search query")

class MyTool(ConfigurableTool[MyParams, MyConfig]):
    config_class = MyConfig
    
    def _run(self, params: MyParams) -> str:
        if len(params.query) > self.config.max_items:
            return self._error("Query too long")
        return f"Results for: {params.query}"
```

Runtime reconfiguration:
```python
tool = MyTool(MyConfig(max_items=50))
tool.configure(timeout=60.0)  # Mutate in place
new_tool = tool.with_config(max_items=200)  # Immutable variant
```
