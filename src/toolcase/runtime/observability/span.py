"""Span types for tracing tool execution.

Spans represent units of work with timing, attributes, and events.
Optimized for AI tool debugging with rich context capture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from toolcase.foundation.errors import ErrorCode, ErrorTrace, JsonDict, JsonValue

if TYPE_CHECKING:
    from .context import SpanContext


class SpanKind(StrEnum):
    """Span type classification."""
    
    TOOL = "tool"        # Tool invocation
    INTERNAL = "internal"  # Internal operation
    EXTERNAL = "external"  # External API call
    PIPELINE = "pipeline"  # Multi-tool pipeline


class SpanStatus(StrEnum):
    """Span completion status."""
    
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass(slots=True)
class SpanEvent:
    """Point-in-time event within a span.
    
    Captures significant moments during execution (e.g., "cache_hit", "retry").
    """
    
    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class Span:
    """Represents a unit of work in a trace.
    
    Captures timing, attributes, events, and error information.
    Designed for AI tool observability with rich context.
    
    Attributes:
        name: Human-readable span name (e.g., "web_search")
        context: SpanContext with trace/span IDs
        kind: Type of work (tool, internal, external)
        start_time: Unix timestamp of span start
        end_time: Unix timestamp of span end (None if active)
        attributes: Key-value metadata (params, results, etc.)
        events: Timestamped events during execution
        status: Completion status
        error: Error message if failed
        error_trace: Full ErrorTrace for structured error info
    
    Example:
        >>> span = Span(name="search", context=SpanContext.new(), kind=SpanKind.TOOL)
        >>> span.set_attribute("query", "python tutorial")
        >>> span.add_event("cache_miss")
        >>> span.end(status=SpanStatus.OK)
    """
    
    name: str
    context: SpanContext
    kind: SpanKind = SpanKind.INTERNAL
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    attributes: JsonDict = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    status: SpanStatus = SpanStatus.UNSET
    error: str | None = None
    error_trace: ErrorTrace | None = None
    
    # Tool-specific fields (AI observability)
    tool_name: str | None = None
    tool_category: str | None = None
    params: JsonDict | None = None
    result_preview: str | None = None  # Truncated result for debugging
    
    @property
    def duration_ms(self) -> float | None:
        """Duration in milliseconds, or None if not ended."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000
    
    @property
    def is_active(self) -> bool:
        """Whether span is still running."""
        return self.end_time is None
    
    def set_attribute(self, key: str, value: JsonValue) -> Span:
        """Set attribute, returns self for chaining."""
        self.attributes[key] = value
        return self
    
    def set_attributes(self, attrs: JsonDict) -> Span:
        """Set multiple attributes."""
        self.attributes.update(attrs)
        return self
    
    def add_event(self, name: str, attributes: JsonDict | None = None) -> Span:
        """Add timestamped event to span."""
        self.events.append(SpanEvent(name=name, attributes=attributes or {}))
        return self
    
    def set_status(self, status: SpanStatus, error: str | None = None) -> Span:
        """Set completion status."""
        self.status = status
        if error:
            self.error = error
        return self
    
    def record_error(self, trace: ErrorTrace) -> Span:
        """Record structured error from ErrorTrace.
        
        Sets status to ERROR and captures full error context for debugging.
        Adds error event with code and recoverable status.
        
        Args:
            trace: ErrorTrace with full error context
        
        Returns:
            Self for chaining
        """
        self.status = SpanStatus.ERROR
        self.error = trace.message
        self.error_trace = trace
        self.add_event("error", {
            "message": trace.message,
            "code": trace.error_code,
            "recoverable": trace.recoverable,
            "contexts": [str(c) for c in trace.contexts],
        })
        return self
    
    def record_exception(self, exc: Exception, *, code: ErrorCode | None = None) -> Span:
        """Record error from exception.
        
        Creates ErrorTrace from exception and records it.
        
        Args:
            exc: Exception to record
            code: Optional error code override
        
        Returns:
            Self for chaining
        """
        from toolcase.foundation.errors import classify_exception, trace_from_exc
        actual_code = code or classify_exception(exc)
        trace = trace_from_exc(exc, operation=self.name, code=actual_code.value)
        return self.record_error(trace)
    
    def set_tool_context(
        self,
        tool_name: str,
        category: str,
        params: JsonDict | None = None,
    ) -> Span:
        """Set tool-specific context for AI observability."""
        self.tool_name = tool_name
        self.tool_category = category
        self.params = params
        self.kind = SpanKind.TOOL
        return self
    
    def set_result_preview(self, result: str, max_len: int = 200) -> Span:
        """Store truncated result for debugging."""
        self.result_preview = result[:max_len] + "..." if len(result) > max_len else result
        return self
    
    def end(
        self,
        status: SpanStatus | None = None,
        error: str | ErrorTrace | None = None,
    ) -> Span:
        """End the span with optional status.
        
        Args:
            status: Completion status
            error: Error string or ErrorTrace for structured error capture
        """
        self.end_time = time.time()
        if status:
            self.status = status
        if error:
            if isinstance(error, ErrorTrace):
                self.record_error(error)
            else:
                self.error = error
                self.status = SpanStatus.ERROR
        return self
    
    def to_dict(self) -> JsonDict:
        """Serialize span for export."""
        result: JsonDict = {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_id": self.context.parent_id,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "error": self.error,
            "attributes": self.attributes,
            "events": [
                {"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes}
                for e in self.events
            ],
            "tool": {
                "name": self.tool_name,
                "category": self.tool_category,
                "params": self.params,
                "result_preview": self.result_preview,
            } if self.tool_name else None,
        }
        # Include structured error info if present
        if self.error_trace:
            result["error_trace"] = {
                "message": self.error_trace.message,
                "code": self.error_trace.error_code,
                "recoverable": self.error_trace.recoverable,
                "contexts": [str(c) for c in self.error_trace.contexts],
                "details": self.error_trace.details,
            }
        return result