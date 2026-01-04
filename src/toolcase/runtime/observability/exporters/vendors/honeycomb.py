"""Honeycomb exporter via OTLP HTTP."""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import orjson

from toolcase.foundation.errors import JsonDict

if TYPE_CHECKING:
    from ...span import Span


@dataclass(slots=True)
class HoneycombExporter:
    """Export spans to Honeycomb via OTLP HTTP.
    
    Honeycomb natively supports OTLP, so this is a thin wrapper
    that adds the required Honeycomb headers.
    
    Args:
        api_key: Honeycomb API key (or set HONEYCOMB_API_KEY env var)
        dataset: Honeycomb dataset name (optional for new-style keys)
        service_name: Service name in traces
        timeout: Request timeout in seconds
    """
    
    api_key: str | None = None
    dataset: str = ""
    service_name: str = "toolcase"
    timeout: float = 10.0
    _endpoint: str = field(default="https://api.honeycomb.io/v1/traces", init=False, repr=False)
    
    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("HONEYCOMB_API_KEY")
        if not self.api_key:
            raise ValueError("HoneycombExporter requires api_key or HONEYCOMB_API_KEY env var")
    
    def export(self, spans: list[Span]) -> None:
        if not spans:
            return
        payload = self._build_otlp_payload(spans)
        headers = {"Content-Type": "application/json", "x-honeycomb-team": self.api_key or ""}
        if self.dataset:
            headers["x-honeycomb-dataset"] = self.dataset
        
        req = urllib.request.Request(self._endpoint, data=orjson.dumps(payload), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout):
                pass
        except Exception:  # noqa: BLE001
            pass
    
    def _build_otlp_payload(self, spans: list[Span]) -> JsonDict:
        return {
            "resourceSpans": [{
                "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": self.service_name}}]},
                "scopeSpans": [{"scope": {"name": "toolcase"}, "spans": [self._to_otlp_span(s) for s in spans]}],
            }],
        }
    
    def _to_otlp_span(self, span: Span) -> JsonDict:
        attrs = [{"key": k, "value": {"stringValue": str(v)}} for k, v in span.attributes.items()]
        if span.tool_name:
            attrs.append({"key": "tool.name", "value": {"stringValue": span.tool_name}})
        
        return {
            "traceId": span.context.trace_id, "spanId": span.context.span_id,
            "parentSpanId": span.context.parent_id or "",
            "name": span.name,
            "startTimeUnixNano": str(int(span.start_time * 1e9)),
            "endTimeUnixNano": str(int((span.end_time or span.start_time) * 1e9)),
            "kind": {"tool": 3, "internal": 1, "external": 3, "pipeline": 1}.get(span.kind.value, 1),
            "status": {"code": {"ok": 1, "error": 2}.get(span.status.value, 0), "message": span.error or ""},
            "attributes": attrs,
        }
    
    def shutdown(self) -> None:
        pass


def honeycomb(api_key: str | None = None, *, dataset: str = "", service_name: str = "toolcase") -> HoneycombExporter:
    """Create Honeycomb exporter with sensible defaults."""
    return HoneycombExporter(api_key=api_key, dataset=dataset, service_name=service_name)
