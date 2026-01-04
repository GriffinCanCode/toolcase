"""Span types for tracing tool execution.

Spans represent units of work with timing, attributes, and events.
Optimized for AI tool debugging with rich context capture.

Supports orjson (JSON) and msgpack (binary) serialization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import msgpack
import orjson

from toolcase.foundation.errors import (
    ErrorCode, ErrorTrace, ErrorTraceSerialized, JsonDict, JsonMapping, JsonValue, SpanDict, SpanEventDict,
)

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
    
    Captures timing, attributes, events, and error information. Designed for AI tool observability with rich context.
    
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
        return (self.end_time - self.start_time) * 1000 if self.end_time else None
    
    @property
    def is_active(self) -> bool:
        """Whether span is still running."""
        return self.end_time is None
    
    def set_attribute(self, key: str, value: JsonValue) -> Span:
        """Set attribute, returns self for chaining."""
        self.attributes[key] = value
        return self
    
    def set_attributes(self, attrs: JsonMapping) -> Span:
        """Set multiple attributes."""
        self.attributes.update(attrs)
        return self
    
    def add_event(self, name: str, attributes: JsonMapping | None = None) -> Span:
        """Add timestamped event to span."""
        self.events.append(SpanEvent(name=name, attributes=dict(attributes) if attributes else {}))
        return self
    
    def set_status(self, status: SpanStatus, error: str | None = None) -> Span:
        """Set completion status."""
        self.status, self.error = status, error or self.error
        return self
    
    def record_error(self, trace: ErrorTrace) -> Span:
        """Record structured error from ErrorTrace. Sets status to ERROR and captures full error context."""
        self.status, self.error, self.error_trace = SpanStatus.ERROR, trace.message, trace
        return self.add_event("error", {
            "message": trace.message, "code": trace.error_code,
            "recoverable": trace.recoverable, "contexts": [str(c) for c in trace.contexts],
        })
    
    def record_exception(self, exc: Exception, *, code: ErrorCode | None = None) -> Span:
        """Record error from exception. Creates ErrorTrace from exception and records it."""
        from toolcase.foundation.errors import classify_exception, trace_from_exc
        actual_code = code or classify_exception(exc)
        return self.record_error(trace_from_exc(exc, operation=self.name, code=actual_code.value))
    
    def set_tool_context(self, tool_name: str, category: str, params: JsonMapping | None = None) -> Span:
        """Set tool-specific context for AI observability."""
        self.tool_name, self.tool_category, self.params = tool_name, category, dict(params) if params else None
        self.kind = SpanKind.TOOL
        return self
    
    def set_result_preview(self, result: str, max_len: int = 200) -> Span:
        """Store truncated result for debugging."""
        self.result_preview = f"{result[:max_len]}..." if len(result) > max_len else result
        return self
    
    # ─────────────────────────────────────────────────────────────────────────
    # Span-Correlated Logging
    # ─────────────────────────────────────────────────────────────────────────
    
    def log(self, event: str, level: str = "info", **attrs: JsonValue) -> Span:
        """Emit a log entry correlated with this span.
        
        Logs are emitted to the configured logger with trace context automatically included,
        and also recorded as span events for bidirectional correlation.
        
        Args:
            event: Log message
            level: Log level (debug, info, warning, error)
            **attrs: Additional attributes
        
        Example:
            >>> with tracer.span("fetch") as span:
            ...     span.log("starting request", url="https://api.example.com")
            ...     # Both emits log AND records span event
        """
        from ..logging import get_logger
        log = get_logger().bind(span_name=self.name)
        getattr(log, level, log.info)(event, **attrs)
        self.add_event(f"log.{level}", {"message": event, **attrs})
        return self
    
    def log_debug(self, event: str, **attrs: JsonValue) -> Span:
        """Emit debug log correlated with this span."""
        return self.log(event, "debug", **attrs)
    
    def log_info(self, event: str, **attrs: JsonValue) -> Span:
        """Emit info log correlated with this span."""
        return self.log(event, "info", **attrs)
    
    def log_warning(self, event: str, **attrs: JsonValue) -> Span:
        """Emit warning log correlated with this span."""
        return self.log(event, "warning", **attrs)
    
    def log_error(self, event: str, **attrs: JsonValue) -> Span:
        """Emit error log correlated with this span."""
        return self.log(event, "error", **attrs)
    
    def end(self, status: SpanStatus | None = None, error: str | ErrorTrace | None = None) -> Span:
        """End the span with optional status."""
        self.end_time = time.time()
        if status:
            self.status = status
        if isinstance(error, ErrorTrace):
            self.record_error(error)
        elif error:
            self.error, self.status = error, SpanStatus.ERROR
        return self
    
    def to_dict(self) -> SpanDict:
        """Serialize span for export."""
        ctx = self.context
        events: list[SpanEventDict] = [
            {"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes}
            for e in self.events
        ]
        result: SpanDict = {
            "name": self.name, "trace_id": ctx.trace_id, "span_id": ctx.span_id,
            "parent_id": ctx.parent_id, "kind": self.kind.value,
            "start_time": self.start_time, "end_time": self.end_time,
            "duration_ms": self.duration_ms, "status": self.status.value, "error": self.error,
            "attributes": self.attributes, "events": events,
            "tool": {"name": self.tool_name, "category": self.tool_category,
                     "params": self.params, "result_preview": self.result_preview} if self.tool_name else None,
        }
        if self.error_trace:
            et = self.error_trace
            err_ser: ErrorTraceSerialized = {
                "message": et.message, "code": et.error_code, "recoverable": et.recoverable,
                "contexts": [str(c) for c in et.contexts], "details": et.details,
            }
            result["error_trace"] = err_ser
        return result
    
    def to_json(self) -> bytes:
        """Serialize span to JSON bytes (orjson)."""
        return orjson.dumps(self.to_dict(), option=orjson.OPT_NON_STR_KEYS)
    
    def to_msgpack(self) -> bytes:
        """Serialize span to msgpack bytes (~40% smaller than JSON)."""
        return msgpack.packb(self.to_dict(), use_bin_type=True)