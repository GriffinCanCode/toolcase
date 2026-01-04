# Foundation

Core building blocks for toolcase. Everything here is fundamental infrastructure that other modules depend on.

## Modules

| Module | Purpose |
|--------|---------|
| `core/` | `BaseTool`, `ToolMetadata`, `@tool` decorator - the base abstractions |
| `errors/` | `Result` monad, `ErrorCode`, `ToolError` - structured error handling |
| `di/` | `Container`, `Provider`, `Scope` - dependency injection |
| `registry/` | `ToolRegistry`, `get_registry()` - tool discovery and management |
| `testing/` | `ToolTestCase`, `MockTool`, fixtures - testing utilities |
| `formats/` | Format converters for OpenAI, Anthropic, Google, etc. |
| `config/` | `ToolcaseSettings` - centralized configuration |

## Quick Import

```python
from toolcase.foundation import BaseTool, ToolMetadata, tool
from toolcase.foundation import Result, Ok, Err, ErrorCode
from toolcase.foundation import Container, get_registry
```
