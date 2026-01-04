"""Tracing module: spans, context, and tracer for distributed tracing."""

from .context import SpanContext, TraceContext, trace_context
from .span import Span, SpanEvent, SpanKind, SpanStatus
from .tracer import Tracer, configure_tracing, get_tracer, instrument_httpx, traced, uninstrument_httpx

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
    "configure_tracing",
    "get_tracer",
    "traced",
    # HTTPX Instrumentation
    "instrument_httpx",
    "uninstrument_httpx",
]
