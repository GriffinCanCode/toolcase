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
        api_key = self.api_key or os.environ.get("DD_API_KEY")
        if not api_key and not self.agent_url:
            raise ValueError("DatadogExporter requires api_key or agent_url")
        object.__setattr__(self, "api_key", api_key)
    
    def export(self, spans: list[Span]) -> None:
        if spans:
            traces = {s.context.trace_id: [] for s in spans}
            for s in spans:
                traces[s.context.trace_id].append(s)
            self._send([[self._to_dd_span(s) for s in t] for t in traces.values()])
    
    def _to_dd_span(self, span: Span) -> JsonDict:
        ctx, _int = span.context, lambda x: int(x, 16) if x else 0
        meta = {k: str(v) for k, v in span.attributes.items()}
        meta |= {k: v for k, v in {"env": self.env, "version": self.version, "error.msg": span.error, "tool.name": span.tool_name}.items() if v}
        return {
            "trace_id": _int(ctx.trace_id[:16]) if ctx.trace_id else 0,
            "span_id": _int(ctx.span_id), "parent_id": _int(ctx.parent_id),
            "name": f"{span.kind.value}.{span.name}" if span.kind else span.name,
            "resource": span.tool_name or span.name,
            "service": self.service_name, "type": "custom",
            "start": int(span.start_time * 1e9),
            "duration": int((span.duration_ms or 0) * 1e6),
            "error": int(span.status.value == "error"),
            "meta": meta,
            "metrics": {"duration_ms": span.duration_ms} if span.duration_ms else {},
        }
    
    def _send(self, payload: list[list[JsonDict]]) -> None:
        url, headers = (f"{self.agent_url.rstrip('/')}/v0.3/traces", {"Content-Type": "application/json"}) if self.agent_url else (
            f"https://trace.agent.{self.site}/api/v0.2/traces", {"Content-Type": "application/json", "DD-API-KEY": self.api_key or ""})
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=orjson.dumps(payload), headers=headers, method="PUT"), timeout=self.timeout):
                pass
        except Exception:  # noqa: BLE001
            pass
    
    def shutdown(self) -> None:
        pass


def datadog(api_key: str | None = None, *, service_name: str = "toolcase", env: str = "", **kw: object) -> DatadogExporter:
    """Create Datadog exporter with sensible defaults."""
    return DatadogExporter(api_key=api_key, service_name=service_name, env=env, **kw)  # type: ignore[arg-type]
