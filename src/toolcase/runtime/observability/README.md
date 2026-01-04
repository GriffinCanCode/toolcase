# Observability

Distributed tracing and structured logging optimized for AI agent debugging.

## Module Structure

```
observability/
├── tracing/          # Core tracing primitives
│   ├── context.py    # SpanContext, TraceContext, trace_context
│   ├── span.py       # Span, SpanEvent, SpanKind, SpanStatus
│   └── tracer.py     # Tracer, configure_tracing, get_tracer, traced
├── logging/          # Structured logging
│   └── logger.py     # BoundLogger, LogEntry, configure_logging, get_logger
├── middleware/       # Auto-instrumentation
│   └── tracing.py    # TracingMiddleware, CorrelationMiddleware
└── exporters/        # Span export backends
    ├── core/         # Base exporters (console, JSON, batch, filters)
    ├── clients/      # Protocol clients (OTLP)
    └── vendors/      # Vendor integrations (Datadog, Honeycomb, Zipkin)
```

## Quick Start

### Automatic Instrumentation

```python
from toolcase.runtime.observability import configure_tracing, TracingMiddleware
from toolcase import get_registry

# Configure tracing (once at startup)
configure_tracing(service_name="my-agent", exporter="console")

# Add middleware for automatic instrumentation
registry = get_registry()
registry.use(TracingMiddleware())

# All tool calls now emit traces
await registry.execute("search", {"query": "python"})
```

### Manual Instrumentation

```python
from toolcase.runtime.observability import get_tracer, traced, SpanKind

tracer = get_tracer()

# Context manager
with tracer.span("fetch_data", kind=SpanKind.EXTERNAL) as span:
    span.set_attribute("url", "https://api.example.com")
    data = fetch_data()

# Decorator
@traced(kind=SpanKind.EXTERNAL)
def fetch_data(url: str) -> dict:
    return requests.get(url).json()
```

### Structured Logging

```python
from toolcase.runtime.observability import get_logger, configure_logging

configure_logging(format="console")  # or "json" for production

log = get_logger("my-service")
log.info("processing request", user_id=123, path="/users")

# Bind tool context
log = log.bind_tool("web_search", "search")
log.info("executing", query="python tutorial")
```

## Exporters

| Exporter | Use Case |
|----------|----------|
| `console` | Development debugging |
| `json` | Log aggregation (ELK, Loki) |
| `otlp` | OpenTelemetry Collector (gRPC) |
| `otlp_http` | OpenTelemetry Collector (HTTP) |
| `datadog` | Datadog APM |
| `honeycomb` | Honeycomb.io |
| `zipkin` | Zipkin tracing |

### Production Configuration

```python
# OTLP export to OpenTelemetry Collector
configure_tracing(
    service_name="my-agent",
    exporter="otlp",
    endpoint="http://otel-collector:4317",
)

# Datadog with async export and 10% sampling
configure_tracing(
    service_name="my-agent",
    exporter="datadog",
    api_key="dd-xxx",
    env="production",
    async_export=True,
    sample_rate=0.1,
)
```

## Key Concepts

### Spans
A span represents a unit of work with timing, attributes, and events. Designed for AI tool observability:

- **name**: Human-readable operation name
- **kind**: `TOOL`, `INTERNAL`, `EXTERNAL`, `PIPELINE`
- **attributes**: Key-value metadata (params, results)
- **events**: Timestamped events during execution
- **status**: `OK`, `ERROR`, `UNSET`

### Context Propagation
Trace context automatically propagates across async boundaries via `contextvars`. Compatible with W3C Trace Context for external system integration.

### Filtering & Sampling
```python
from toolcase.runtime.observability import FilteredExporter, errors_only, slow_spans

# Only export error spans
configure_tracing(exporter=FilteredExporter(ConsoleExporter(), errors_only))

# Only export slow spans (>100ms)
configure_tracing(exporter=FilteredExporter(ConsoleExporter(), slow_spans(100)))
```
