"""Tracing module: spans, context, and tracer for distributed tracing."""

from .context import SpanContext, TraceContext, trace_context
from .propagate import (
    BAGGAGE,
    TRACEPARENT,
    TRACESTATE,
    Propagator,
    TraceFlags,
    TraceState,
    continue_trace,
    extract_trace_context,
    get_propagator,
    inject_trace_context,
    propagate_context,
)
from .span import Span, SpanEvent, SpanKind, SpanStatus
from .tracer import Tracer, configure_tracing, get_tracer, instrument_httpx, traced, uninstrument_httpx

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
    "configure_tracing",
    "get_tracer",
    "traced",
    # HTTPX Instrumentation
    "instrument_httpx",
    "uninstrument_httpx",
]
