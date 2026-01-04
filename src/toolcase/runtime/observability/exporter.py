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
    
    Outputs readable trace visualization with timing and attributes.
    
    Args:
        output: Output stream (default: stderr)
        colors: Use ANSI colors (default: True if TTY)
        verbose: Show full attributes (default: False)
    """
    
    output: TextIO = field(default_factory=lambda: sys.stderr)
    colors: bool = field(default=True)
    verbose: bool = False
    
    def __post_init__(self) -> None:
        if self.colors and not (hasattr(self.output, "isatty") and self.output.isatty()):
            self.colors = False
    
    def export(self, spans: list[Span]) -> None:
        for span in spans:
            self._print_span(span)
    
    def _print_span(self, span: Span) -> None:
        # Colors
        c = self._colors if self.colors else self._no_colors
        
        # Status indicator
        status_sym = {"ok": "✓", "error": "✗", "unset": "○"}.get(span.status.value, "?")
        status_color = {"ok": c["green"], "error": c["red"], "unset": c["dim"]}
        
        # Format timing
        ts = datetime.fromtimestamp(span.start_time, tz=UTC).strftime("%H:%M:%S.%f")[:-3]
        dur = f"{span.duration_ms:.1f}ms" if span.duration_ms else "..."
        
        # Indent based on parent depth (approx from context)
        indent = "  " * (1 if span.context.parent_id else 0)
        
        # Build output
        line = (
            f"{c['dim']}{ts}{c['reset']} "
            f"{status_color.get(span.status.value, c['dim'])}{status_sym}{c['reset']} "
            f"{indent}{c['bold']}{span.name}{c['reset']} "
            f"{c['cyan']}[{span.kind.value}]{c['reset']} "
            f"{c['yellow']}{dur}{c['reset']}"
        )
        
        # Tool context
        if span.tool_name:
            line += f" {c['dim']}tool={span.tool_name}{c['reset']}"
        
        # Error
        if span.error:
            line += f" {c['red']}error={span.error[:50]}{c['reset']}"
        
        print(line, file=self.output)
        
        # Verbose: attributes and events
        if self.verbose and span.attributes:
            for k, v in span.attributes.items():
                print(f"    {c['dim']}{k}={v!r}{c['reset']}", file=self.output)
    
    @property
    def _colors(self) -> dict[str, str]:
        return {
            "reset": "\033[0m",
            "bold": "\033[1m",
            "dim": "\033[2m",
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "cyan": "\033[36m",
        }
    
    @property
    def _no_colors(self) -> dict[str, str]:
        return {k: "" for k in ("reset", "bold", "dim", "red", "green", "yellow", "cyan")}
    
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
        for span in spans:
            print(json.dumps(span.to_dict(), default=str), file=self.output)
    
    def shutdown(self) -> None:
        self.output.flush()


@dataclass(slots=True)
class BatchExporter:
    """Buffers spans and exports in batches for efficiency.
    
    Wraps another exporter and batches exports to reduce overhead.
    Flushes when batch_size reached or on shutdown.
    """
    
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
    """Fan-out to multiple exporters.
    
    Exports each span to all configured exporters.
    Useful for dev console + production backend simultaneously.
    """
    
    exporters: list[Exporter] = field(default_factory=list)
    
    def export(self, spans: list[Span]) -> None:
        for exp in self.exporters:
            exp.export(spans)
    
    def shutdown(self) -> None:
        for exp in self.exporters:
            exp.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# OTLP Exporter (Optional - requires opentelemetry-* packages)
# ─────────────────────────────────────────────────────────────────────────────


def create_otlp_exporter(
    endpoint: str = "http://localhost:4317",
    service_name: str = "toolcase",
    insecure: bool = True,
    headers: dict[str, str] | None = None,
) -> Exporter:
    """Create OTLP exporter for OpenTelemetry backends.
    
    Requires: pip install toolcase[otel]
    
    Args:
        endpoint: OTLP gRPC endpoint
        service_name: Service name for traces
        insecure: Use insecure connection (default for local dev)
        headers: Optional headers for authentication (e.g., API keys)
    
    Returns:
        OTLPSpanExporter wrapped in our Exporter protocol
    
    Raises:
        ImportError: If opentelemetry packages not installed
    """
    try:
        import opentelemetry.exporter.otlp.proto.grpc.trace_exporter  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "OTLP exporter requires: pip install toolcase[otel]"
        ) from e
    
    return OTLPBridge(
        endpoint=endpoint,
        service_name=service_name,
        insecure=insecure,
        headers=headers,
    )


@dataclass
class OTLPBridge:
    """Bridge toolcase Spans to OpenTelemetry OTLP export.
    
    Properly converts our Span format to OTel ReadableSpan for direct export,
    preserving timing, context, attributes, events, and status.
    """
    
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
        self._exporter = OTLPSpanExporter(
            endpoint=self.endpoint,
            insecure=self.insecure,
            headers=self.headers or {},
        )
    
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
        
        # Map our SpanKind to OTel SpanKind
        kind_map = {
            "tool": OtelSpanKind.CLIENT,
            "internal": OtelSpanKind.INTERNAL,
            "external": OtelSpanKind.CLIENT,
            "pipeline": OtelSpanKind.INTERNAL,
        }
        otel_kind = kind_map.get(span.kind.value, OtelSpanKind.INTERNAL)
        
        # Parse trace/span IDs (hex strings -> int)
        trace_id = int(span.context.trace_id, 16) if span.context.trace_id else 0
        span_id = int(span.context.span_id, 16) if span.context.span_id else 0
        parent_id = int(span.context.parent_id, 16) if span.context.parent_id else None
        
        # Build OTel SpanContext
        ctx = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=False,
            trace_flags=TraceFlags.SAMPLED,
        )
        
        # Build parent SpanContext if exists
        parent_ctx = None
        if parent_id:
            parent_ctx = SpanContext(
                trace_id=trace_id,
                span_id=parent_id,
                is_remote=False,
                trace_flags=TraceFlags.SAMPLED,
            )
        
        # Convert timestamps (seconds -> nanoseconds)
        start_ns = int(span.start_time * 1e9)
        end_ns = int(span.end_time * 1e9) if span.end_time else start_ns
        
        # Convert status
        if span.status.value == "error":
            status = Status(StatusCode.ERROR, span.error or "")
        elif span.status.value == "ok":
            status = Status(StatusCode.OK)
        else:
            status = Status(StatusCode.UNSET)
        
        # Flatten attributes (OTel only accepts primitive types)
        attrs = self._flatten_attrs(span.attributes)
        
        # Add tool context as attributes
        if span.tool_name:
            attrs["tool.name"] = span.tool_name
        if span.tool_category:
            attrs["tool.category"] = span.tool_category
        if span.result_preview:
            attrs["tool.result_preview"] = span.result_preview
        
        # Convert events
        events = tuple(
            Event(
                name=e.name,
                timestamp=int(e.timestamp * 1e9),
                attributes=self._flatten_attrs(e.attributes),
            )
            for e in span.events
        )
        
        # Create instrumentation scope
        scope = InstrumentationScope(name="toolcase", version="0.2.0")
        
        # _resource is always set in __post_init__, assert for type checker
        assert self._resource is not None
        return _ReadableSpanAdapter(
            name=span.name,
            context=ctx,
            parent=parent_ctx,
            kind=otel_kind,
            start_time=start_ns,
            end_time=end_ns,
            attributes=attrs,
            events=events,
            status=status,
            resource=self._resource,
            instrumentation_scope=scope,
        )
    
    def _flatten_attrs(self, attrs: dict[str, object]) -> dict[str, str | int | float | bool]:
        """Flatten attributes to OTel-compatible primitive types."""
        result: dict[str, str | int | float | bool] = {}
        for k, v in attrs.items():
            if isinstance(v, (str, int, float, bool)):
                result[k] = v
            elif isinstance(v, dict):
                result[k] = json.dumps(v)
            else:
                result[k] = str(v)
        return result
    
    def shutdown(self) -> None:
        """Flush and shutdown the exporter."""
        if self._exporter:
            self._exporter.shutdown()


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
    
    @property
    def links(self) -> tuple[()]:
        return ()
    
    @property
    def dropped_attributes(self) -> int:
        return 0
    
    @property
    def dropped_events(self) -> int:
        return 0
    
    @property
    def dropped_links(self) -> int:
        return 0
    
    def to_json(self, indent: int | None = None) -> str:
        return json.dumps({
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "attributes": self.attributes,
        }, indent=indent)
