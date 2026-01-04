"""Span exporters for different observability backends.

Provides pluggable export destinations:
- ConsoleExporter: Pretty-printed spans for development
- JsonExporter: JSON lines for log aggregation
- OTLPExporter: OpenTelemetry Protocol for production (optional dep)
- NoOpExporter: Silent export for testing
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, TextIO, runtime_checkable

from toolcase.foundation.errors import JsonDict

# Color constants for ConsoleExporter
_SPAN_COLORS = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
                "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m", "cyan": "\033[36m"}
_SPAN_NO_COLORS = {k: "" for k in _SPAN_COLORS}

if TYPE_CHECKING:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import Event
    from opentelemetry.sdk.util.instrumentation import InstrumentationScope
    from opentelemetry.trace import SpanContext, SpanKind
    from opentelemetry.trace.status import Status
    
    from .span import Span


@runtime_checkable
class Exporter(Protocol):
    """Protocol for span exporters.
    
    Exporters receive completed spans and send them to backends.
    Must be thread-safe for concurrent exports.
    """
    
    def export(self, spans: list[Span]) -> None:
        """Export batch of completed spans."""
        ...
    
    def shutdown(self) -> None:
        """Graceful shutdown, flush pending exports."""
        ...


@dataclass(slots=True)
class NoOpExporter:
    """Silent exporter for testing/disabled tracing."""
    
    def export(self, spans: list[Span]) -> None:
        pass
    
    def shutdown(self) -> None:
        pass


@dataclass(slots=True)
class ConsoleExporter:
    """Pretty-print spans to console for development.
    
    Args: output (stderr), colors (True if TTY), verbose (False)
    """
    
    output: TextIO = field(default_factory=lambda: sys.stderr)
    colors: bool = field(default=True)
    verbose: bool = False
    
    def __post_init__(self) -> None:
        if self.colors and not getattr(self.output, "isatty", lambda: False)():
            self.colors = False
    
    def export(self, spans: list[Span]) -> None:
        for s in spans: self._print_span(s)
    
    def _print_span(self, span: Span) -> None:
        c = _SPAN_COLORS if self.colors else _SPAN_NO_COLORS
        status_sym = {"ok": "✓", "error": "✗", "unset": "○"}.get(span.status.value, "?")
        status_color = {"ok": c["green"], "error": c["red"], "unset": c["dim"]}
        ts = datetime.fromtimestamp(span.start_time, tz=UTC).strftime("%H:%M:%S.%f")[:-3]
        dur = f"{span.duration_ms:.1f}ms" if span.duration_ms else "..."
        indent = "  " if span.context.parent_id else ""
        
        line = (f"{c['dim']}{ts}{c['reset']} "
                f"{status_color.get(span.status.value, c['dim'])}{status_sym}{c['reset']} "
                f"{indent}{c['bold']}{span.name}{c['reset']} "
                f"{c['cyan']}[{span.kind.value}]{c['reset']} "
                f"{c['yellow']}{dur}{c['reset']}")
        
        if span.tool_name:
            line += f" {c['dim']}tool={span.tool_name}{c['reset']}"
        if span.error:
            line += f" {c['red']}error={span.error[:50]}{c['reset']}"
        
        print(line, file=self.output)
        
        # Verbose: attributes and events
        if self.verbose and span.attributes:
            for k, v in span.attributes.items():
                print(f"    {c['dim']}{k}={v!r}{c['reset']}", file=self.output)
    
    def shutdown(self) -> None:
        self.output.flush()


@dataclass(slots=True)
class JsonExporter:
    """Export spans as JSON lines for log aggregation.
    
    Each span is a single JSON object per line (JSONL format).
    Suitable for shipping to Elasticsearch, Loki, etc.
    """
    
    output: TextIO = field(default_factory=lambda: sys.stdout)
    
    def export(self, spans: list[Span]) -> None:
        for s in spans: print(json.dumps(s.to_dict(), default=str), file=self.output)
    
    def shutdown(self) -> None:
        self.output.flush()


@dataclass(slots=True)
class BatchExporter:
    """Buffers spans and exports in batches. Flushes when batch_size reached or on shutdown."""
    
    exporter: Exporter
    batch_size: int = 100
    _buffer: list[Span] = field(default_factory=list)
    
    def export(self, spans: list[Span]) -> None:
        self._buffer.extend(spans)
        if len(self._buffer) >= self.batch_size:
            self.flush()
    
    def flush(self) -> None:
        if self._buffer:
            self.exporter.export(self._buffer)
            self._buffer.clear()
    
    def shutdown(self) -> None:
        self.flush()
        self.exporter.shutdown()


@dataclass(slots=True)
class CompositeExporter:
    """Fan-out to multiple exporters. Useful for dev console + production backend simultaneously."""
    
    exporters: list[Exporter] = field(default_factory=list)
    
    def export(self, spans: list[Span]) -> None:
        for e in self.exporters: e.export(spans)
    
    def shutdown(self) -> None:
        for e in self.exporters: e.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# OTLP Exporter (Optional - requires opentelemetry-* packages)
# ─────────────────────────────────────────────────────────────────────────────


def create_otlp_exporter(
    endpoint: str = "http://localhost:4317",
    service_name: str = "toolcase",
    insecure: bool = True,
    headers: dict[str, str] | None = None,
) -> Exporter:
    """Create OTLP exporter for OpenTelemetry backends. Requires: pip install toolcase[otel]"""
    try:
        import opentelemetry.exporter.otlp.proto.grpc.trace_exporter  # noqa: F401
    except ImportError as e:
        raise ImportError("OTLP exporter requires: pip install toolcase[otel]") from e
    return OTLPBridge(endpoint=endpoint, service_name=service_name, insecure=insecure, headers=headers)


@dataclass
class OTLPBridge:
    """Bridge toolcase Spans to OTel OTLP export, preserving timing, context, attributes, events, and status."""
    
    endpoint: str
    service_name: str
    insecure: bool = True
    headers: dict[str, str] | None = None
    _exporter: OTLPSpanExporter | None = field(default=None, init=False, repr=False)
    _resource: Resource | None = field(default=None, init=False, repr=False)
    
    def __post_init__(self) -> None:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        
        self._resource = Resource.create({SERVICE_NAME: self.service_name})
        self._exporter = OTLPSpanExporter(endpoint=self.endpoint, insecure=self.insecure, headers=self.headers or {})
    
    def export(self, spans: list[Span]) -> None:
        """Convert and export spans to OTLP backend."""
        if self._exporter is None:
            return
        otel_spans = [self._to_otel_span(s) for s in spans]
        self._exporter.export(otel_spans)
    
    def _to_otel_span(self, span: Span) -> _ReadableSpanAdapter:
        """Convert toolcase Span to OTel ReadableSpan."""
        from opentelemetry.sdk.trace import Event
        from opentelemetry.sdk.util.instrumentation import InstrumentationScope
        from opentelemetry.trace import SpanContext, TraceFlags
        from opentelemetry.trace import SpanKind as OtelSpanKind
        from opentelemetry.trace.status import Status, StatusCode
        
        kind_map = {"tool": OtelSpanKind.CLIENT, "internal": OtelSpanKind.INTERNAL,
                    "external": OtelSpanKind.CLIENT, "pipeline": OtelSpanKind.INTERNAL}
        otel_kind = kind_map.get(span.kind.value, OtelSpanKind.INTERNAL)
        
        # Parse trace/span IDs (hex strings -> int)
        trace_id = int(span.context.trace_id, 16) if span.context.trace_id else 0
        span_id = int(span.context.span_id, 16) if span.context.span_id else 0
        parent_id = int(span.context.parent_id, 16) if span.context.parent_id else None
        
        ctx = SpanContext(trace_id=trace_id, span_id=span_id, is_remote=False, trace_flags=TraceFlags.SAMPLED)
        parent_ctx = SpanContext(trace_id=trace_id, span_id=parent_id, is_remote=False,
                                  trace_flags=TraceFlags.SAMPLED) if parent_id else None
        
        # Convert timestamps (seconds -> nanoseconds)
        start_ns, end_ns = int(span.start_time * 1e9), int(span.end_time * 1e9) if span.end_time else int(span.start_time * 1e9)
        
        # Convert status
        status = (Status(StatusCode.ERROR, span.error or "") if span.status.value == "error"
                  else Status(StatusCode.OK) if span.status.value == "ok" else Status(StatusCode.UNSET))
        
        # Flatten attributes and add tool context
        attrs = self._flatten_attrs(span.attributes) | {
            k: v for k, v in [("tool.name", span.tool_name), ("tool.category", span.tool_category),
                              ("tool.result_preview", span.result_preview)] if v}
        
        events = tuple(Event(name=e.name, timestamp=int(e.timestamp * 1e9),
                             attributes=self._flatten_attrs(e.attributes)) for e in span.events)
        
        assert self._resource is not None  # Always set in __post_init__
        return _ReadableSpanAdapter(name=span.name, context=ctx, parent=parent_ctx, kind=otel_kind,
                                    start_time=start_ns, end_time=end_ns, attributes=attrs, events=events,
                                    status=status, resource=self._resource,
                                    instrumentation_scope=InstrumentationScope(name="toolcase", version="0.2.0"))
    
    def _flatten_attrs(self, attrs: JsonDict) -> dict[str, str | int | float | bool]:
        """Flatten attributes to OTel-compatible primitive types."""
        def _convert(v: object) -> str | int | float | bool:
            if isinstance(v, (str, int, float, bool)):
                return v
            return json.dumps(v) if isinstance(v, dict) else ("" if v is None else str(v))
        return {k: _convert(v) for k, v in attrs.items()}
    
    def shutdown(self) -> None:
        """Flush and shutdown the exporter."""
        self._exporter and self._exporter.shutdown()


@dataclass(slots=True)
class _ReadableSpanAdapter:
    """Adapter implementing OTel ReadableSpan protocol for direct export."""
    
    name: str
    context: SpanContext
    parent: SpanContext | None
    kind: SpanKind
    start_time: int  # nanoseconds
    end_time: int  # nanoseconds
    attributes: dict[str, str | int | float | bool]
    events: tuple[Event, ...]
    status: Status
    resource: Resource
    instrumentation_scope: InstrumentationScope | None = None
    
    # ReadableSpan protocol
    def get_span_context(self) -> SpanContext:
        return self.context
    
    @property
    def parent_span_context(self) -> SpanContext | None:
        return self.parent
    
    links: tuple[()] = ()
    dropped_attributes: int = 0
    dropped_events: int = 0
    dropped_links: int = 0
    
    def to_json(self, indent: int | None = None) -> str:
        return json.dumps({"name": self.name, "start_time": self.start_time,
                           "end_time": self.end_time, "attributes": self.attributes}, indent=indent)
