"""Metrics middleware for tool execution."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

from toolcase.foundation.errors import ErrorCode, ToolException, classify_exception
from toolcase.runtime.middleware import Context, Next

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool

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
        self.log.debug(f"METRIC {metric}={value}{f' {tags}' if tags else ''}")
    
    def timing(self, metric: str, value_ms: float, tags: dict[str, str] | None = None) -> None:
        self.log.debug(f"METRIC {metric}={value_ms:.2f}ms{f' {tags}' if tags else ''}")


@dataclass(slots=True)
class MetricsMiddleware:
    """Collect execution metrics (counters, timing) with error code tracking.
    
    Emits:
    - tool.calls: Counter per tool
    - tool.errors: Counter for error results (with error_code tag)
    - tool.exceptions: Counter for exceptions (with error_code tag)
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
        tags = {"tool": tool.metadata.name, "category": tool.metadata.category}
        start = time.perf_counter()
        
        try:
            result = await next(tool, params, ctx)
            self.backend.increment(f"{self.prefix}.calls", tags=tags)
            self.backend.timing(f"{self.prefix}.duration_ms", (time.perf_counter() - start) * 1000, tags=tags)
            if result.startswith("**Tool Error"):
                self.backend.increment(f"{self.prefix}.errors", tags=tags)
            return result
        except ToolException as e:
            self.backend.increment(f"{self.prefix}.calls", tags=tags)
            self.backend.increment(f"{self.prefix}.exceptions", tags={**tags, "error_code": e.error.code.value})
            raise
        except Exception as e:
            self.backend.increment(f"{self.prefix}.calls", tags=tags)
            self.backend.increment(f"{self.prefix}.exceptions", tags={**tags, "error_code": classify_exception(e).value})
            raise
