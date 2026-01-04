"""Metrics middleware for tool execution."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

from ..middleware import Context, Next

if TYPE_CHECKING:
    from ...core import BaseTool

logger = logging.getLogger("toolcase.middleware")


@runtime_checkable
class MetricsBackend(Protocol):
    """Protocol for metrics collection backends."""
    
    def increment(self, metric: str, value: int = 1, tags: dict[str, str] | None = None) -> None: ...
    def timing(self, metric: str, value_ms: float, tags: dict[str, str] | None = None) -> None: ...


@dataclass(slots=True)
class LogMetricsBackend:
    """Default metrics backend that logs to Python logger."""
    
    log: logging.Logger = field(default_factory=lambda: logger)
    
    def increment(self, metric: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
        tag_str = f" {tags}" if tags else ""
        self.log.debug(f"METRIC {metric}={value}{tag_str}")
    
    def timing(self, metric: str, value_ms: float, tags: dict[str, str] | None = None) -> None:
        tag_str = f" {tags}" if tags else ""
        self.log.debug(f"METRIC {metric}={value_ms:.2f}ms{tag_str}")


@dataclass(slots=True)
class MetricsMiddleware:
    """Collect execution metrics (counters, timing).
    
    Emits:
    - tool.calls: Counter per tool
    - tool.errors: Counter for error results
    - tool.duration_ms: Timing histogram
    
    Args:
        backend: MetricsBackend implementation (defaults to logging)
        prefix: Metric name prefix
    
    Example:
        >>> from datadog import statsd
        >>> registry.use(MetricsMiddleware(backend=statsd, prefix="myapp"))
    """
    
    backend: MetricsBackend = field(default_factory=LogMetricsBackend)
    prefix: str = "tool"
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        name = tool.metadata.name
        tags = {"tool": name, "category": tool.metadata.category}
        start = time.perf_counter()
        
        try:
            result = await next(tool, params, ctx)
            duration_ms = (time.perf_counter() - start) * 1000
            
            self.backend.increment(f"{self.prefix}.calls", tags=tags)
            self.backend.timing(f"{self.prefix}.duration_ms", duration_ms, tags=tags)
            
            if result.startswith("**Tool Error"):
                self.backend.increment(f"{self.prefix}.errors", tags=tags)
            
            return result
            
        except Exception:
            self.backend.increment(f"{self.prefix}.calls", tags=tags)
            self.backend.increment(f"{self.prefix}.exceptions", tags=tags)
            raise
