"""Batch and composite exporters with async support."""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import TYPE_CHECKING

from .protocol import Exporter

if TYPE_CHECKING:
    from ...span import Span


@dataclass(slots=True)
class BatchExporter:
    """Buffers spans and exports in batches. Flushes when batch_size reached or on shutdown."""
    
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
    """Fan-out to multiple exporters. Useful for dev console + production backend simultaneously."""
    
    exporters: list[Exporter] = field(default_factory=list)
    
    def export(self, spans: list[Span]) -> None:
        for e in self.exporters:
            e.export(spans)
    
    def shutdown(self) -> None:
        for e in self.exporters:
            e.shutdown()


@dataclass
class AsyncBatchExporter:
    """Background queue-based export with configurable retry and backoff.
    
    Decouples span collection from export IO. Spans are queued and exported
    in batches by a background thread with exponential backoff on failures.
    
    Args:
        exporter: Target exporter for batched spans
        batch_size: Max spans per batch (default: 100)
        flush_interval: Seconds between flush attempts (default: 5.0)
        max_queue_size: Queue capacity before dropping spans (default: 10000)
        max_retries: Retry attempts on export failure (default: 3)
        base_delay: Initial retry delay in seconds (default: 1.0)
    """
    
    exporter: Exporter
    batch_size: int = 100
    flush_interval: float = 5.0
    max_queue_size: int = 10000
    max_retries: int = 3
    base_delay: float = 1.0
    _queue: Queue[Span] = field(default_factory=Queue, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _dropped: int = field(default=0, init=False, repr=False)
    
    def __post_init__(self) -> None:
        self._thread = threading.Thread(target=self._worker, daemon=True, name="exporter-worker")
        self._thread.start()
    
    def export(self, spans: list[Span]) -> None:
        """Queue spans for background export. Non-blocking."""
        for span in spans:
            if self._queue.qsize() < self.max_queue_size:
                self._queue.put_nowait(span)
            else:
                self._dropped += 1
    
    def _worker(self) -> None:
        """Background worker: drain queue in batches with retry."""
        batch: list[Span] = []
        while not self._stop.is_set():
            try:
                span = self._queue.get(timeout=self.flush_interval)
                batch.append(span)
                while len(batch) < self.batch_size:
                    try:
                        batch.append(self._queue.get_nowait())
                    except Empty:
                        break
                self._export_with_retry(batch)
                batch.clear()
            except Empty:
                if batch:
                    self._export_with_retry(batch)
                    batch.clear()
    
    def _export_with_retry(self, batch: list[Span]) -> None:
        """Export batch with exponential backoff retry."""
        for attempt in range(self.max_retries + 1):
            try:
                self.exporter.export(batch)
                return
            except Exception:  # noqa: BLE001
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt) * (0.5 + random.random())
                    self._stop.wait(timeout=delay)
    
    @property
    def dropped_count(self) -> int:
        """Number of spans dropped due to queue overflow."""
        return self._dropped
    
    def flush(self) -> None:
        """Force flush remaining spans. Blocks until queue is empty."""
        while not self._queue.empty():
            self._stop.wait(timeout=0.1)
    
    def shutdown(self) -> None:
        """Stop worker thread and flush remaining spans."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        batch = []
        while not self._queue.empty():
            try:
                batch.append(self._queue.get_nowait())
            except Empty:
                break
        if batch:
            self.exporter.export(batch)
        self.exporter.shutdown()
