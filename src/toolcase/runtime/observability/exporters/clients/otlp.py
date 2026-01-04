"""OpenTelemetry Protocol (OTLP) exporters - gRPC and HTTP variants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import orjson

from toolcase.foundation.errors import JsonDict

from ..core.protocol import Exporter

if TYPE_CHECKING:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import Event
    from opentelemetry.sdk.util.instrumentation import InstrumentationScope
    from opentelemetry.trace import SpanContext, SpanKind
    from opentelemetry.trace.status import Status
    
    from ...span import Span


def create_otlp_exporter(
    endpoint: str = "http://localhost:4317",
    service_name: str = "toolcase",
    insecure: bool = True,
    headers: dict[str, str] | None = None,
    *,
    use_http: bool = False,
) -> Exporter:
    """Create OTLP exporter for OpenTelemetry backends.
    
    Args:
        endpoint: OTLP collector endpoint
        service_name: Service name in traces
        insecure: Use insecure connection (gRPC only)
        headers: Additional headers (auth, etc.)
        use_http: Use HTTP/protobuf instead of gRPC
    
    Requires: pip install toolcase[otel]
    """
    try:
        import opentelemetry.exporter.otlp.proto.grpc.trace_exporter  # noqa: F401
    except ImportError as e:
        raise ImportError("OTLP exporter requires: pip install toolcase[otel]") from e
    if use_http:
        return OTLPHttpBridge(endpoint=endpoint, service_name=service_name, headers=headers)
    return OTLPBridge(endpoint=endpoint, service_name=service_name, insecure=insecure, headers=headers)


@dataclass
class OTLPBridge:
    """Bridge toolcase Spans to OTel OTLP gRPC export."""
    
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
        if self._exporter is None:
            return
        self._exporter.export([self._to_otel_span(s) for s in spans])
    
    def _to_otel_span(self, span: Span) -> _ReadableSpanAdapter:
        from opentelemetry.sdk.trace import Event
        from opentelemetry.sdk.util.instrumentation import InstrumentationScope
        from opentelemetry.trace import SpanContext, TraceFlags
        from opentelemetry.trace import SpanKind as OtelSpanKind
        from opentelemetry.trace.status import Status, StatusCode
        
        kind_map = {"tool": OtelSpanKind.CLIENT, "internal": OtelSpanKind.INTERNAL,
                    "external": OtelSpanKind.CLIENT, "pipeline": OtelSpanKind.INTERNAL}
        
        trace_id = int(span.context.trace_id, 16) if span.context.trace_id else 0
        span_id = int(span.context.span_id, 16) if span.context.span_id else 0
        parent_id = int(span.context.parent_id, 16) if span.context.parent_id else None
        
        ctx = SpanContext(trace_id=trace_id, span_id=span_id, is_remote=False, trace_flags=TraceFlags.SAMPLED)
        parent_ctx = SpanContext(trace_id=trace_id, span_id=parent_id, is_remote=False,
                                  trace_flags=TraceFlags.SAMPLED) if parent_id else None
        
        start_ns = int(span.start_time * 1e9)
        end_ns = int(span.end_time * 1e9) if span.end_time else start_ns
        
        status = (Status(StatusCode.ERROR, span.error or "") if span.status.value == "error"
                  else Status(StatusCode.OK) if span.status.value == "ok" else Status(StatusCode.UNSET))
        
        attrs = _flatten_attrs(span.attributes) | {
            k: v for k, v in [("tool.name", span.tool_name), ("tool.category", span.tool_category),
                              ("tool.result_preview", span.result_preview)] if v}
        
        events = tuple(Event(name=e.name, timestamp=int(e.timestamp * 1e9),
                             attributes=_flatten_attrs(e.attributes)) for e in span.events)
        
        assert self._resource is not None
        return _ReadableSpanAdapter(
            name=span.name, context=ctx, parent=parent_ctx, kind=kind_map.get(span.kind.value, OtelSpanKind.INTERNAL),
            start_time=start_ns, end_time=end_ns, attributes=attrs, events=events,
            status=status, resource=self._resource,
            instrumentation_scope=InstrumentationScope(name="toolcase", version="0.2.0"))
    
    def shutdown(self) -> None:
        self._exporter and self._exporter.shutdown()


@dataclass
class OTLPHttpBridge:
    """Bridge toolcase Spans to OTLP HTTP/protobuf export.
    
    Uses HTTP POST instead of gRPC. Compatible with:
    - OpenTelemetry Collector (http://collector:4318/v1/traces)
    - Grafana Tempo, Jaeger, Zipkin with OTLP receivers
    """
    
    endpoint: str = "http://localhost:4318/v1/traces"
    service_name: str = "toolcase"
    headers: dict[str, str] | None = None
    timeout: float = 10.0
    _resource: Resource | None = field(default=None, init=False, repr=False)
    
    def __post_init__(self) -> None:
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        self._resource = Resource.create({SERVICE_NAME: self.service_name})
    
    def export(self, spans: list[Span]) -> None:
        if not spans:
            return
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=self.endpoint, headers=self.headers or {}, timeout=int(self.timeout))
            exporter.export([self._to_otel_span(s) for s in spans])
            exporter.shutdown()
        except ImportError:
            self._export_manual(spans)
    
    def _export_manual(self, spans: list[Span]) -> None:
        """Manual JSON export when OTel HTTP exporter unavailable."""
        import urllib.request
        payload = {"resourceSpans": [{"resource": {"attributes": [{"key": "service.name", "value": {"stringValue": self.service_name}}]},
                                      "scopeSpans": [{"scope": {"name": "toolcase"}, "spans": [self._span_to_json(s) for s in spans]}]}]}
        req = urllib.request.Request(self.endpoint, data=orjson.dumps(payload),
                                      headers={"Content-Type": "application/json", **(self.headers or {})}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout):
                pass
        except Exception:  # noqa: BLE001
            pass
    
    def _span_to_json(self, span: Span) -> JsonDict:
        return {
            "traceId": span.context.trace_id, "spanId": span.context.span_id,
            "parentSpanId": span.context.parent_id or "", "name": span.name,
            "startTimeUnixNano": str(int(span.start_time * 1e9)),
            "endTimeUnixNano": str(int((span.end_time or span.start_time) * 1e9)),
            "kind": {"tool": 3, "internal": 1, "external": 3, "pipeline": 1}.get(span.kind.value, 1),
            "status": {"code": {"ok": 1, "error": 2}.get(span.status.value, 0), "message": span.error or ""},
            "attributes": [{"key": k, "value": {"stringValue": str(v)}} for k, v in span.attributes.items()],
        }
    
    def _to_otel_span(self, span: Span) -> _ReadableSpanAdapter:
        from opentelemetry.sdk.trace import Event
        from opentelemetry.sdk.util.instrumentation import InstrumentationScope
        from opentelemetry.trace import SpanContext, TraceFlags
        from opentelemetry.trace import SpanKind as OtelSpanKind
        from opentelemetry.trace.status import Status, StatusCode
        
        kind_map = {"tool": OtelSpanKind.CLIENT, "internal": OtelSpanKind.INTERNAL,
                    "external": OtelSpanKind.CLIENT, "pipeline": OtelSpanKind.INTERNAL}
        
        trace_id = int(span.context.trace_id, 16) if span.context.trace_id else 0
        span_id = int(span.context.span_id, 16) if span.context.span_id else 0
        parent_id = int(span.context.parent_id, 16) if span.context.parent_id else None
        
        ctx = SpanContext(trace_id=trace_id, span_id=span_id, is_remote=False, trace_flags=TraceFlags.SAMPLED)
        parent_ctx = SpanContext(trace_id=trace_id, span_id=parent_id, is_remote=False,
                                  trace_flags=TraceFlags.SAMPLED) if parent_id else None
        
        start_ns = int(span.start_time * 1e9)
        end_ns = int(span.end_time * 1e9) if span.end_time else start_ns
        
        status = (Status(StatusCode.ERROR, span.error or "") if span.status.value == "error"
                  else Status(StatusCode.OK) if span.status.value == "ok" else Status(StatusCode.UNSET))
        
        attrs = _flatten_attrs(span.attributes) | {
            k: v for k, v in [("tool.name", span.tool_name), ("tool.category", span.tool_category),
                              ("tool.result_preview", span.result_preview)] if v}
        
        events = tuple(Event(name=e.name, timestamp=int(e.timestamp * 1e9),
                             attributes=_flatten_attrs(e.attributes)) for e in span.events)
        
        assert self._resource is not None
        return _ReadableSpanAdapter(
            name=span.name, context=ctx, parent=parent_ctx, kind=kind_map.get(span.kind.value, OtelSpanKind.INTERNAL),
            start_time=start_ns, end_time=end_ns, attributes=attrs, events=events,
            status=status, resource=self._resource,
            instrumentation_scope=InstrumentationScope(name="toolcase", version="0.2.0"))
    
    def shutdown(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _flatten_attrs(attrs: JsonDict) -> dict[str, str | int | float | bool]:
    """Flatten attributes to OTel-compatible primitive types."""
    def _convert(v: object) -> str | int | float | bool:
        if isinstance(v, (str, int, float, bool)):
            return v
        return orjson.dumps(v).decode() if isinstance(v, dict) else ("" if v is None else str(v))
    return {k: _convert(v) for k, v in attrs.items()}


@dataclass(slots=True)
class _ReadableSpanAdapter:
    """Adapter implementing OTel ReadableSpan protocol for direct export."""
    
    name: str
    context: SpanContext
    parent: SpanContext | None
    kind: SpanKind
    start_time: int
    end_time: int
    attributes: dict[str, str | int | float | bool]
    events: tuple[Event, ...]
    status: Status
    resource: Resource
    instrumentation_scope: InstrumentationScope | None = None
    
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
        opts = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps({"name": self.name, "start_time": self.start_time,
                             "end_time": self.end_time, "attributes": self.attributes}, option=opts).decode()
