"""Observability for tool execution: tracing, spans, and export.

Provides distributed tracing optimized for AI agent debugging:
- Automatic instrumentation via middleware
- Manual instrumentation via decorators/context managers
- Pluggable exporters (console, JSON, OTLP)
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

from .context import SpanContext, TraceContext, trace_context
from .exporter import (
    BatchExporter,
    CompositeExporter,
    ConsoleExporter,
    Exporter,
    JsonExporter,
    NoOpExporter,
    create_otlp_exporter,
)
from .middleware import CorrelationMiddleware, TracingMiddleware
from .span import Span, SpanEvent, SpanKind, SpanStatus
from .tracer import Tracer, configure_tracing, get_tracer, traced

__all__ = [
    # Context
    "SpanContext",
    "TraceContext",
    "trace_context",
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
    # Exporters
    "Exporter",
    "ConsoleExporter",
    "JsonExporter",
    "NoOpExporter",
    "BatchExporter",
    "CompositeExporter",
    "create_otlp_exporter",
    # Middleware
    "TracingMiddleware",
    "CorrelationMiddleware",
]
