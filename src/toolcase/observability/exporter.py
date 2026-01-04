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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol, TextIO, runtime_checkable

if TYPE_CHECKING:
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
        ts = datetime.fromtimestamp(span.start_time, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]
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
) -> Exporter:
    """Create OTLP exporter for OpenTelemetry backends.
    
    Requires: pip install opentelemetry-exporter-otlp-proto-grpc
    
    Args:
        endpoint: OTLP gRPC endpoint
        service_name: Service name for traces
        insecure: Use insecure connection (default for local dev)
    
    Returns:
        OTLPSpanExporter wrapped in our Exporter protocol
    
    Raises:
        ImportError: If opentelemetry packages not installed
    """
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as e:
        raise ImportError(
            "OTLP exporter requires: pip install opentelemetry-exporter-otlp-proto-grpc"
        ) from e
    
    return _OTLPBridge(
        endpoint=endpoint,
        service_name=service_name,
        insecure=insecure,
    )


@dataclass
class _OTLPBridge:
    """Bridge our Span format to OpenTelemetry SDK."""
    
    endpoint: str
    service_name: str
    insecure: bool
    _provider: object = field(default=None, init=False)
    _exporter: object = field(default=None, init=False)
    
    def __post_init__(self) -> None:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        
        resource = Resource.create({"service.name": self.service_name})
        self._provider = TracerProvider(resource=resource)
        self._exporter = OTLPSpanExporter(endpoint=self.endpoint, insecure=self.insecure)
        self._provider.add_span_processor(BatchSpanProcessor(self._exporter))
    
    def export(self, spans: list[Span]) -> None:
        # Convert our spans to OTel format and export
        # This is a simplified bridge - production would do full conversion
        from opentelemetry.trace import Status, StatusCode
        
        tracer = self._provider.get_tracer("toolcase")
        for span in spans:
            with tracer.start_span(span.name) as otel_span:
                for k, v in span.attributes.items():
                    otel_span.set_attribute(k, str(v))
                if span.error:
                    otel_span.set_status(Status(StatusCode.ERROR, span.error))
    
    def shutdown(self) -> None:
        if self._provider:
            self._provider.shutdown()
