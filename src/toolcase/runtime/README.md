# Runtime

Execution flow, control patterns, and monitoring. Everything related to how tools run.

## Modules

| Module | Purpose |
|--------|---------|
| `agents/` | `router`, `fallback`, `race`, `gate`, `escalation` - agentic composition |
| `middleware/` | `Middleware`, plugins (retry, timeout, breaker, logging, metrics) |
| `pipeline/` | `pipeline()`, `parallel()` - tool composition |
| `retry/` | `RetryPolicy`, backoff strategies |
| `observability/` | `Tracer`, `Span`, exporters - distributed tracing |

## Quick Import

```python
from toolcase.runtime import router, fallback, race, gate
from toolcase.runtime import RetryMiddleware, TimeoutMiddleware, CircuitBreakerMiddleware
from toolcase.runtime import pipeline, parallel
from toolcase.runtime import configure_tracing, TracingMiddleware
```
