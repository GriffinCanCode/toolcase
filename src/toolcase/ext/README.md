# Ext

External system integrations. Adapters for third-party frameworks and protocols.

## Modules

| Module | Purpose |
|--------|---------|
| `integrations/` | LangChain, Frontiers adapters |
| `mcp/` | MCP protocol server, HTTP REST server |

## Quick Import

```python
# LangChain integration
from toolcase.ext.integrations import to_langchain_tools

# MCP/HTTP servers
from toolcase.ext.mcp import serve_mcp, serve_http
```

## Usage

```python
# Expose tools via MCP (Cursor, Claude Desktop)
serve_mcp(registry, transport="sse", port=8080)

# Expose tools via HTTP REST
serve_http(registry, port=8000)
```
