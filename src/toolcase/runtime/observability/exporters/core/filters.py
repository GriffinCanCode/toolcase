"""Sampling and filtering exporters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .protocol import Exporter

if TYPE_CHECKING:
    from ...span import Span

SpanPredicate = Callable[["Span"], bool]


@dataclass(slots=True)
class SampledExporter:
    """Head-based sampling decorator. Only exports a fraction of spans.
    
    Uses trace_id for deterministic sampling - all spans in a trace
    are either sampled or not, maintaining trace integrity.
    
    Args:
        exporter: Target exporter for sampled spans
        rate: Sampling rate 0.0-1.0 (default: 0.1 = 10%)
    """
    
    exporter: Exporter
    rate: float = 0.1
    
    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"rate must be 0.0-1.0, got {self.rate}")
    
    def export(self, spans: list[Span]) -> None:
        sampled = [s for s in spans if (int(s.context.trace_id[:8], 16) / 0xFFFFFFFF) < self.rate]
        if sampled:
            self.exporter.export(sampled)
    
    def shutdown(self) -> None:
        self.exporter.shutdown()


@dataclass(slots=True)
class FilteredExporter:
    """Predicate-based span filtering decorator.
    
    Only exports spans matching the predicate. Useful for:
    - Exporting only errors: lambda s: s.status == SpanStatus.ERROR
    - Filtering by duration: lambda s: (s.duration_ms or 0) > 100
    - Filtering by kind: lambda s: s.kind == SpanKind.EXTERNAL
    
    Args:
        exporter: Target exporter for filtered spans
        predicate: Function returning True for spans to export
    """
    
    exporter: Exporter
    predicate: SpanPredicate
    
    def export(self, spans: list[Span]) -> None:
        filtered = [s for s in spans if self.predicate(s)]
        if filtered:
            self.exporter.export(filtered)
    
    def shutdown(self) -> None:
        self.exporter.shutdown()


def errors_only(span: Span) -> bool:
    """Predicate: only error spans."""
    return span.status.value == "error"


def slow_spans(threshold_ms: float = 100.0) -> SpanPredicate:
    """Predicate factory: spans slower than threshold."""
    return lambda s: (s.duration_ms or 0) > threshold_ms
