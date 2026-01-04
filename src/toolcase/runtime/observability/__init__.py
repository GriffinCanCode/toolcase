"""Observability for tool execution: tracing, spans, and export.

Provides distributed tracing optimized for AI agent debugging:
- Automatic instrumentation via middleware
- Manual instrumentation via decorators/context managers
- Pluggable exporters (console, JSON, OTLP, Datadog, Honeycomb, Zipkin)
- Context propagation for correlated traces

Quick Start:
    >>> from toolcase.observability import configure_tracing, TracingMiddleware
    >>> from toolcase import get_registry
    >>> 
    >>> # Configure tracing (once at startup)
    >>> configure_tracing(service_name="my-agent", exporter="console")
    >>> 
    >>> # Add to registry for automatic instrumentation
    >>> registry = get_registry()
    >>> registry.use(TracingMiddleware())
    >>> 
    >>> # All tool calls now emit traces
    >>> await registry.execute("search", {"query": "python"})

Manual Instrumentation:
    >>> from toolcase.observability import get_tracer, traced, SpanKind
    >>> 
    >>> tracer = get_tracer()
    >>> with tracer.span("fetch_data", kind=SpanKind.EXTERNAL) as span:
    ...     span.set_attribute("url", "https://api.example.com")
    ...     data = fetch_data()
    >>> 
    >>> # Or use decorator
    >>> @traced(kind=SpanKind.EXTERNAL)
    ... def fetch_data(url: str) -> dict:
    ...     return requests.get(url).json()

Production Export:
    >>> configure_tracing(
    ...     service_name="my-agent",
    ...     exporter="otlp",  # or "json", "console", custom Exporter
    ...     endpoint="http://otel-collector:4317",
    ... )
"""

# Tracing
from .tracing import (
    BAGGAGE,
    Propagator,
    Span,
    SpanContext,
    SpanEvent,
    SpanKind,
    SpanStatus,
    TRACEPARENT,
    TRACESTATE,
    TraceContext,
    TraceFlags,
    Tracer,
    TraceState,
    configure_tracing,
    continue_trace,
    extract_trace_context,
    get_propagator,
    get_tracer,
    inject_trace_context,
    instrument_httpx,
    propagate_context,
    trace_context,
    traced,
    uninstrument_httpx,
)

# Logging
from .logging import (
    BoundLogger,
    LogScope,
    TracedLogger,
    configure_logging,
    configure_observability,
    get_logger,
    log_context,
    span_logger,
    timed,
)

# Middleware
from .middleware import CorrelationMiddleware, TracingMiddleware

# Exporters
from .exporters import (
    AsyncBatchExporter,
    BatchExporter,
    CompositeExporter,
    ConsoleExporter,
    DatadogExporter,
    Exporter,
    FilteredExporter,
    HoneycombExporter,
    JsonExporter,
    NoOpExporter,
    OTLPBridge,
    OTLPHttpBridge,
    SampledExporter,
    SpanPredicate,
    ZipkinExporter,
    create_otlp_exporter,
    datadog,
    errors_only,
    honeycomb,
    slow_spans,
    zipkin,
)

__all__ = [
    # Context
    "SpanContext",
    "TraceContext",
    "trace_context",
    # Propagation (W3C Trace Context)
    "inject_trace_context",
    "extract_trace_context",
    "propagate_context",
    "continue_trace",
    "Propagator",
    "get_propagator",
    "TraceFlags",
    "TraceState",
    "TRACEPARENT",
    "TRACESTATE",
    "BAGGAGE",
    # Span
    "Span",
    "SpanEvent",
    "SpanKind",
    "SpanStatus",
    # Tracer
    "Tracer",
    "get_tracer",
    "configure_tracing",
    "traced",
    # HTTPX Instrumentation
    "instrument_httpx",
    "uninstrument_httpx",
    # Logging
    "BoundLogger",
    "LogScope",
    "TracedLogger",
    "configure_logging",
    "configure_observability",
    "get_logger",
    "log_context",
    "span_logger",
    "timed",
    # Exporters - Core
    "Exporter",
    "ConsoleExporter",
    "JsonExporter",
    "NoOpExporter",
    "BatchExporter",
    "AsyncBatchExporter",
    "CompositeExporter",
    # Exporters - OTLP
    "OTLPBridge",
    "OTLPHttpBridge",
    "create_otlp_exporter",
    # Exporters - Vendors
    "DatadogExporter",
    "HoneycombExporter",
    "ZipkinExporter",
    "datadog",
    "honeycomb",
    "zipkin",
    # Exporters - Filtering
    "SampledExporter",
    "FilteredExporter",
    "SpanPredicate",
    "errors_only",
    "slow_spans",
    # Middleware
    "TracingMiddleware",
    "CorrelationMiddleware",
]
