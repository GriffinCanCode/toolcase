"""Exporter protocol definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ...span import Span


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
