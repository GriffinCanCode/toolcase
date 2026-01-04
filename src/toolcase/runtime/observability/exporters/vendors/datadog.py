"""Datadog APM exporter."""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

import orjson

from toolcase.foundation.errors import JsonDict

if TYPE_CHECKING:
    from ...span import Span


@dataclass(slots=True)
class DatadogExporter:
    """Export spans to Datadog APM via Traces API.
    
    Uses Datadog's native trace format for full APM integration.
    Supports both direct API submission and Datadog Agent forwarding.
    
    Args:
        api_key: Datadog API key (or set DD_API_KEY env var)
        site: Datadog site (default: datadoghq.com)
        service_name: Service name in traces
        env: Environment name (prod, staging, etc.)
        version: Service version for deployment tracking
        agent_url: If set, forward to local DD Agent instead of API
        timeout: Request timeout in seconds
    """
    
    api_key: str | None = None
    site: str = "datadoghq.com"
    service_name: str = "toolcase"
    env: str = ""
    version: str = ""
    agent_url: str | None = None
    timeout: float = 10.0
    
    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("DD_API_KEY")
        if not self.api_key and not self.agent_url:
            raise ValueError("DatadogExporter requires api_key or agent_url")
    
    def export(self, spans: list[Span]) -> None:
        if not spans:
            return
        traces = self._group_by_trace(spans)
        payload = [[self._to_dd_span(s) for s in trace] for trace in traces.values()]
        self._send(payload)
    
    def _group_by_trace(self, spans: list[Span]) -> dict[str, list[Span]]:
        groups: dict[str, list[Span]] = {}
        for span in spans:
            groups.setdefault(span.context.trace_id, []).append(span)
        return groups
    
    def _to_dd_span(self, span: Span) -> JsonDict:
        trace_id = int(span.context.trace_id[:16], 16) if span.context.trace_id else 0
        span_id = int(span.context.span_id, 16) if span.context.span_id else 0
        parent_id = int(span.context.parent_id, 16) if span.context.parent_id else 0
        
        dd_span: JsonDict = {
            "trace_id": trace_id, "span_id": span_id, "parent_id": parent_id,
            "name": f"{span.kind.value}.{span.name}" if span.kind else span.name,
            "resource": span.tool_name or span.name,
            "service": self.service_name, "type": "custom",
            "start": int(span.start_time * 1e9),
            "duration": int((span.duration_ms or 0) * 1e6),
            "error": 1 if span.status.value == "error" else 0,
            "meta": {k: str(v) for k, v in span.attributes.items()},
            "metrics": {},
        }
        if self.env:
            dd_span["meta"]["env"] = self.env
        if self.version:
            dd_span["meta"]["version"] = self.version
        if span.error:
            dd_span["meta"]["error.msg"] = span.error
        if span.tool_name:
            dd_span["meta"]["tool.name"] = span.tool_name
        if span.duration_ms:
            dd_span["metrics"]["duration_ms"] = span.duration_ms
        return dd_span
    
    def _send(self, payload: list[list[JsonDict]]) -> None:
        if self.agent_url:
            url = f"{self.agent_url.rstrip('/')}/v0.3/traces"
            headers = {"Content-Type": "application/json"}
        else:
            url = f"https://trace.agent.{self.site}/api/v0.2/traces"
            headers = {"Content-Type": "application/json", "DD-API-KEY": self.api_key or ""}
        
        req = urllib.request.Request(url, data=orjson.dumps(payload), headers=headers, method="PUT")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout):
                pass
        except Exception:  # noqa: BLE001
            pass
    
    def shutdown(self) -> None:
        pass


def datadog(api_key: str | None = None, *, service_name: str = "toolcase", env: str = "", **kw: object) -> DatadogExporter:
    """Create Datadog exporter with sensible defaults."""
    return DatadogExporter(api_key=api_key, service_name=service_name, env=env, **kw)  # type: ignore[arg-type]
