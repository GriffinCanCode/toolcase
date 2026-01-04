# Runtime

Execution flow, control patterns, and monitoring. Everything related to how tools run.

## Modules

| Module | Purpose |
|--------|---------|
| `agents/` | `router`, `fallback`, `race`, `gate`, `escalation` - agentic composition |
| `middleware/` | `Middleware`, `StreamMiddleware`, plugins (retry, timeout, breaker, logging, metrics) |
| `pipeline/` | `pipeline()`, `parallel()` - tool composition |
| `retry/` | `RetryPolicy`, backoff strategies |
| `observability/` | `Tracer`, `Span`, exporters - distributed tracing |
| `concurrency/` | Task groups, sync primitives, stream combinators |

## Quick Import

```python
from toolcase.runtime import router, fallback, race, gate
from toolcase.runtime import RetryMiddleware, TimeoutMiddleware, CircuitBreakerMiddleware
from toolcase.runtime import StreamLoggingMiddleware, StreamMetricsMiddleware, compose_streaming
from toolcase.runtime import pipeline, parallel
from toolcase.runtime import configure_tracing, TracingMiddleware
```

## Streaming Middleware

Stream execution through the middleware pipeline with chunk-aware hooks:

```python
from toolcase.runtime import StreamLoggingMiddleware, StreamMetricsMiddleware

# Add streaming middleware to registry
registry.use(StreamLoggingMiddleware())  # Logs start, chunk count, bytes, duration
registry.use(StreamMetricsMiddleware())  # Emits tool.stream.* metrics

# Stream through middleware chain
async for chunk in registry.stream_execute("llm_tool", {"prompt": "Hello"}):
    print(chunk, end="", flush=True)
```

Custom streaming middleware implements lifecycle hooks:

```python
@dataclass
class ChunkTransformMiddleware:
    """Transform chunks as they stream through."""
    
    async def on_start(self, tool, params, ctx):
        ctx["start_time"] = time.time()
    
    async def on_chunk(self, chunk, ctx) -> StreamChunk:
        # Transform and return chunk
        return StreamChunk(content=chunk.content.upper(), index=chunk.index)
    
    async def on_complete(self, accumulated, ctx):
        print(f"Completed in {time.time() - ctx['start_time']:.2f}s")
    
    async def on_error(self, error, ctx):
        print(f"Stream failed: {error}")
```
