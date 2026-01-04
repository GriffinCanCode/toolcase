"""Trace context propagation for distributed tracing.

Uses contextvars for async-safe propagation across tool calls.
Compatible with W3C Trace Context when bridging to external systems.
"""

from __future__ import annotations

import secrets
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

# Global trace context - propagates automatically across async calls
_current_context: ContextVar[TraceContext | None] = ContextVar("trace_context", default=None)


def _generate_id(length: int = 16) -> str:
    """Generate cryptographically random hex ID."""
    return secrets.token_hex(length)


@dataclass(slots=True, frozen=True)
class SpanContext:
    """Immutable context identifying a span in a trace.
    
    Uses W3C-compatible IDs:
    - trace_id: 32 hex chars (128 bits)
    - span_id: 16 hex chars (64 bits)
    - parent_id: Optional parent span for hierarchical traces
    
    Example:
        >>> ctx = SpanContext.new()
        >>> child = ctx.child()
        >>> assert child.trace_id == ctx.trace_id
        >>> assert child.parent_id == ctx.span_id
    """
    
    trace_id: str
    span_id: str
    parent_id: str | None = None
    
    @classmethod
    def new(cls) -> SpanContext:
        """Create new root span context."""
        return cls(trace_id=_generate_id(16), span_id=_generate_id(8))
    
    def child(self) -> SpanContext:
        """Create child context with same trace_id."""
        return SpanContext(
            trace_id=self.trace_id,
            span_id=_generate_id(8),
            parent_id=self.span_id,
        )
    
    @property
    def traceparent(self) -> str:
        """W3C Trace Context format: version-trace_id-span_id-flags."""
        return f"00-{self.trace_id}-{self.span_id}-01"
    
    @classmethod
    def from_traceparent(cls, header: str) -> SpanContext | None:
        """Parse W3C traceparent header."""
        parts = header.split("-")
        if len(parts) != 4 or parts[0] != "00":
            return None
        return cls(trace_id=parts[1], span_id=parts[2])


@dataclass(slots=True)
class TraceContext:
    """Mutable container for current tracing state.
    
    Manages span stack and propagation. Use via context manager API.
    
    Example:
        >>> with TraceContext.current() as ctx:
        ...     with ctx.span("operation") as span:
        ...         # span is active here
        ...         pass
    """
    
    span_context: SpanContext = field(default_factory=SpanContext.new)
    baggage: dict[str, str] = field(default_factory=dict)
    _span_stack: list[SpanContext] = field(default_factory=list)
    
    @classmethod
    def current(cls) -> TraceContext:
        """Get or create current trace context."""
        ctx = _current_context.get()
        if ctx is None:
            ctx = cls()
            _current_context.set(ctx)
        return ctx
    
    @classmethod
    def get(cls) -> TraceContext | None:
        """Get current trace context if exists."""
        return _current_context.get()
    
    @classmethod
    def reset(cls) -> None:
        """Clear current trace context."""
        _current_context.set(None)
    
    def push_span(self, ctx: SpanContext) -> None:
        """Push new span onto stack."""
        self._span_stack.append(self.span_context)
        self.span_context = ctx
    
    def pop_span(self) -> SpanContext:
        """Pop span from stack, return to parent."""
        current = self.span_context
        self.span_context = self._span_stack.pop() if self._span_stack else SpanContext.new()
        return current
    
    @property
    def depth(self) -> int:
        """Current span nesting depth."""
        return len(self._span_stack)


class trace_context:
    """Context manager for trace context scope.
    
    Example:
        >>> with trace_context() as ctx:
        ...     # Operations here share the same trace_id
        ...     pass
    """
    
    __slots__ = ("_ctx", "_token")
    
    def __init__(self, ctx: TraceContext | None = None) -> None:
        self._ctx = ctx or TraceContext()
        self._token: object | None = None
    
    def __enter__(self) -> TraceContext:
        self._token = _current_context.set(self._ctx)
        return self._ctx
    
    def __exit__(self, *_: object) -> None:
        if self._token is not None:
            _current_context.reset(self._token)
