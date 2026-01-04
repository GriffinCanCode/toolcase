"""Simple exporters: NoOp, Console, JSON."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from ...span import Span

# Color constants for ConsoleExporter
_COLORS = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
           "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m", "cyan": "\033[36m"}
_NO_COLORS = {k: "" for k in _COLORS}


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
        for s in spans:
            self._print_span(s)
    
    def _print_span(self, span: Span) -> None:
        c = _COLORS if self.colors else _NO_COLORS
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
        for s in spans:
            print(json.dumps(s.to_dict(), default=str), file=self.output)
    
    def shutdown(self) -> None:
        self.output.flush()
