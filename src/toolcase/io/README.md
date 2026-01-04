# IO

Data input/output, caching, and streaming. Handles data flow in and out of tools.

## Modules

| Module | Purpose |
|--------|---------|
| `cache/` | `MemoryCache`, `RedisCache` - result caching with TTL |
| `progress/` | `ToolProgress`, `ProgressKind` - progress event streaming |
| `streaming/` | `StreamEvent`, adapters (SSE, WebSocket) - incremental result streaming |

## Quick Import

```python
from toolcase.io import get_cache, MemoryCache, RedisCache
from toolcase.io import ToolProgress, status, step, complete
from toolcase.io import StreamEvent, sse_adapter, ws_adapter
```
