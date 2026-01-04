"""Logging middleware for tool execution."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..middleware import Context, Next

if TYPE_CHECKING:
    from ...core import BaseTool

logger = logging.getLogger("toolcase.middleware")


@dataclass(slots=True)
class LoggingMiddleware:
    """Log tool execution with timing and result status.
    
    Logs at INFO level for successful calls, WARNING for errors.
    Duration is stored in context as 'duration_ms'.
    
    Args:
        logger: Logger instance to use (defaults to toolcase.middleware)
        log_params: Whether to include params in log (default False for privacy)
    
    Example:
        >>> registry.use(LoggingMiddleware(log_params=True))
    """
    
    log: logging.Logger = field(default_factory=lambda: logger)
    log_params: bool = False
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        name = tool.metadata.name
        start = time.perf_counter()
        
        param_str = f" params={params.model_dump()}" if self.log_params else ""
        self.log.info(f"[{name}] Starting{param_str}")
        
        try:
            result = await next(tool, params, ctx)
            duration_ms = (time.perf_counter() - start) * 1000
            ctx["duration_ms"] = duration_ms
            
            is_error = result.startswith("**Tool Error")
            level = logging.WARNING if is_error else logging.INFO
            status = "ERROR" if is_error else "OK"
            
            self.log.log(level, f"[{name}] {status} ({duration_ms:.1f}ms)")
            return result
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            ctx["duration_ms"] = duration_ms
            self.log.exception(f"[{name}] EXCEPTION ({duration_ms:.1f}ms): {e}")
            raise
