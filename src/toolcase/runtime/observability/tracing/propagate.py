"""W3C Trace Context propagation for distributed tracing.

Implements W3C Trace Context (https://www.w3.org/TR/trace-context/) for
cross-service trace correlation. Provides inject/extract functions for
HTTP headers and generic carrier dictionaries.

Headers:
- traceparent: Required. Format: {version}-{trace_id}-{span_id}-{flags}
- tracestate: Optional. Vendor-specific key=value pairs
- baggage: Optional. Cross-cutting concerns (W3C Baggage spec)

Example:
    >>> from toolcase.observability import inject_trace_context, extract_trace_context
    >>> 
    >>> # Inject into outgoing HTTP request
    >>> headers = inject_trace_context({})
    >>> response = await client.get(url, headers=headers)
    >>> 
    >>> # Extract from incoming HTTP request
    >>> ctx = extract_trace_context(request.headers)
    >>> with trace_context(ctx):
    ...     await process_request()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

from .context import SpanContext, TraceContext, _current_context

# W3C Trace Context header names (lowercase for HTTP/2 compatibility)
TRACEPARENT = "traceparent"
TRACESTATE = "tracestate"
BAGGAGE = "baggage"

# W3C Trace Context version
_VERSION = "00"
# Default trace flags (sampled=01)
_SAMPLED = "01"
_NOT_SAMPLED = "00"


@dataclass(frozen=True, slots=True)
class TraceFlags:
    """W3C trace flags."""
    sampled: bool = True
    
    def __str__(self) -> str: return _SAMPLED if self.sampled else _NOT_SAMPLED
    
    @classmethod
    def parse(cls, flags: str) -> "TraceFlags": return cls(sampled=flags[-1] == "1")


@dataclass(frozen=True, slots=True)
class TraceState:
    """W3C tracestate: vendor-specific key-value pairs.
    
    Format: key1=value1,key2=value2 (max 32 entries, 512 bytes per entry)
    """
    entries: tuple[tuple[str, str], ...] = ()
    
    def __str__(self) -> str: return ",".join(f"{k}={v}" for k, v in self.entries)
    
    def with_entry(self, key: str, value: str) -> "TraceState":
        """Add/update entry (moves to front per spec)."""
        filtered = tuple((k, v) for k, v in self.entries if k != key)[:31]  # Max 32 entries
        return TraceState(entries=((key, value),) + filtered)
    
    def get(self, key: str) -> str | None:
        return next((v for k, v in self.entries if k == key), None)
    
    @classmethod
    def parse(cls, header: str) -> "TraceState":
        if not header: return cls()
        entries = tuple(
            (k.strip(), v.strip()) for e in header.split(",") if "=" in e
            for k, v in [e.split("=", 1)]
        )
        return cls(entries=entries[:32])


def inject_trace_context(
    carrier: MutableMapping[str, str],
    *,
    ctx: TraceContext | None = None,
    include_baggage: bool = True,
    tracestate: TraceState | None = None,
) -> MutableMapping[str, str]:
    """Inject W3C trace context headers into carrier (dict/headers).
    
    Adds traceparent header (required) and optionally tracestate/baggage.
    Uses current trace context if ctx not provided.
    
    Args:
        carrier: Mutable mapping to inject headers into (modified in-place)
        ctx: Optional explicit TraceContext (defaults to current context)
        include_baggage: Whether to include baggage header
        tracestate: Optional vendor-specific state
    
    Returns:
        The carrier with injected headers (for chaining)
    
    Example:
        >>> headers = inject_trace_context({})
        >>> await httpx.get(url, headers=headers)
        
        >>> # With explicit context
        >>> ctx = TraceContext.current()
        >>> inject_trace_context(request.headers, ctx=ctx)
    """
    trace_ctx = ctx or _current_context.get()
    if not trace_ctx: return carrier
    
    span_ctx = trace_ctx.span_context
    carrier[TRACEPARENT] = span_ctx.traceparent
    
    if tracestate: carrier[TRACESTATE] = str(tracestate)
    if include_baggage and trace_ctx.baggage:
        carrier[BAGGAGE] = ",".join(f"{k}={v}" for k, v in trace_ctx.baggage.items())
    
    return carrier


def extract_trace_context(carrier: Mapping[str, str]) -> TraceContext | None:
    """Extract W3C trace context from carrier (headers/dict).
    
    Parses traceparent header and optionally tracestate/baggage.
    Returns None if no valid traceparent found.
    
    Args:
        carrier: Mapping containing trace headers (case-insensitive lookup)
    
    Returns:
        TraceContext if valid traceparent found, else None
    
    Example:
        >>> ctx = extract_trace_context(request.headers)
        >>> if ctx:
        ...     with trace_context(ctx):
        ...         await handle_request()
    """
    # Case-insensitive header lookup
    get = lambda k: next((v for h, v in carrier.items() if h.lower() == k), None)
    
    traceparent = get(TRACEPARENT)
    if not traceparent: return None
    
    span_ctx = SpanContext.from_traceparent(traceparent)
    if not span_ctx: return None
    
    # Parse baggage
    baggage: dict[str, str] = {}
    if baggage_header := get(BAGGAGE):
        for entry in baggage_header.split(","):
            if "=" in entry:
                k, v = entry.split("=", 1)
                baggage[k.strip()] = v.strip()
    
    return TraceContext(span_context=span_ctx, baggage=baggage)


def propagate_context(carrier: MutableMapping[str, str]) -> MutableMapping[str, str]:
    """Convenience: inject current trace context into carrier.
    
    Shorthand for inject_trace_context with defaults.
    
    Example:
        >>> headers = propagate_context({"Content-Type": "application/json"})
    """
    return inject_trace_context(carrier)


def continue_trace(carrier: Mapping[str, str]) -> SpanContext | None:
    """Extract span context and create child for continuation.
    
    Use when you want to continue an incoming trace as a child span.
    
    Example:
        >>> child_ctx = continue_trace(request.headers)
        >>> with tracer.span("handle_request", context=child_ctx):
        ...     await process()
    """
    if ctx := extract_trace_context(carrier):
        return ctx.span_context.child()
    return None


class Propagator:
    """Pluggable propagator for different formats (W3C, B3, Jaeger, etc.).
    
    Default implementation is W3C Trace Context. Override for custom formats.
    
    Example:
        >>> propagator = Propagator()
        >>> propagator.inject(headers)
        >>> ctx = propagator.extract(headers)
    """
    
    __slots__ = ()
    
    def inject(self, carrier: MutableMapping[str, str], ctx: TraceContext | None = None) -> MutableMapping[str, str]:
        """Inject trace context into carrier."""
        return inject_trace_context(carrier, ctx=ctx)
    
    def extract(self, carrier: Mapping[str, str]) -> TraceContext | None:
        """Extract trace context from carrier."""
        return extract_trace_context(carrier)
    
    def propagate(self, carrier: MutableMapping[str, str]) -> MutableMapping[str, str]:
        """Inject current context into carrier (alias for inject with defaults)."""
        return self.inject(carrier)


# Default propagator instance
_propagator = Propagator()

def get_propagator() -> Propagator:
    """Get the default propagator."""
    return _propagator
