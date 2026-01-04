"""Zipkin v2 JSON exporter."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

import orjson

from toolcase.foundation.errors import JsonDict

if TYPE_CHECKING:
    from ...span import Span


@dataclass(slots=True)
class ZipkinExporter:
    """Export spans to Zipkin via v2 JSON API.
    
    Compatible with Zipkin, Jaeger (Zipkin collector), and other
    systems supporting Zipkin v2 format.
    
    Args:
        endpoint: Zipkin collector endpoint
        service_name: Local service name
        timeout: Request timeout in seconds
    """
    
    endpoint: str = "http://localhost:9411/api/v2/spans"
    service_name: str = "toolcase"
    timeout: float = 10.0
    
    def export(self, spans: list[Span]) -> None:
        if not spans:
            return
        try:
            with urllib.request.urlopen(urllib.request.Request(self.endpoint, data=orjson.dumps([self._to_zipkin_span(s) for s in spans]),
                                                               headers={"Content-Type": "application/json"}, method="POST"), timeout=self.timeout):
                pass
        except Exception:  # noqa: BLE001
            pass
    
    def _to_zipkin_span(self, span: Span) -> JsonDict:
        tags = {k: str(v) for k, v in span.attributes.items()} | {k: v for k, v in {"error": span.error, "tool.name": span.tool_name}.items() if v}
        return {
            "traceId": span.context.trace_id.lower(), "id": span.context.span_id.lower(), "name": span.name,
            "timestamp": int(span.start_time * 1e6), "duration": int((span.duration_ms or 0) * 1000),
            "localEndpoint": {"serviceName": self.service_name},
            "kind": {"tool": "CLIENT", "internal": "LOCAL", "external": "CLIENT", "pipeline": "LOCAL"}.get(span.kind.value, "LOCAL"),
            "tags": tags,
        } | ({"parentId": span.context.parent_id.lower()} if span.context.parent_id else {})
    
    def shutdown(self) -> None:
        pass


def zipkin(endpoint: str = "http://localhost:9411/api/v2/spans", *, service_name: str = "toolcase") -> ZipkinExporter:
    """Create Zipkin exporter with sensible defaults."""
    return ZipkinExporter(endpoint=endpoint, service_name=service_name)
