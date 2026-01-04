"""Logging middleware for tool execution."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel

from toolcase.foundation.errors import ErrorCode, ToolException, classify_exception
from toolcase.runtime.middleware import Context, Next

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool

logger = logging.getLogger("toolcase.middleware")


@dataclass(slots=True)
class LoggingMiddleware:
    """Log tool execution with timing and result status.
    
    Logs at INFO level for successful calls, WARNING for errors.
    Duration is stored in context as 'duration_ms'.
    Classifies exceptions using ErrorCode for structured logging.
    
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
        name, start = tool.metadata.name, time.perf_counter()
        self.log.info(f"[{name}] Starting{f' params={params.model_dump()}' if self.log_params else ''}")
        
        try:
            result = await next(tool, params, ctx)
            duration_ms = (time.perf_counter() - start) * 1000
            ctx["duration_ms"] = duration_ms
            is_error = result.startswith("**Tool Error")
            self.log.log(logging.WARNING if is_error else logging.INFO, f"[{name}] {'ERROR' if is_error else 'OK'} ({duration_ms:.1f}ms)")
            return result
        except ToolException as e:
            ctx.update(duration_ms=(time.perf_counter() - start) * 1000, error_code=e.error.code.value)
            self.log.error(f"[{name}] EXCEPTION ({ctx['duration_ms']:.1f}ms) [{e.error.code}]: {e.error.message}")
            raise
        except Exception as e:
            code = classify_exception(e)
            ctx.update(duration_ms=(time.perf_counter() - start) * 1000, error_code=code.value)
            self.log.exception(f"[{name}] EXCEPTION ({ctx['duration_ms']:.1f}ms) [{code}]: {e}")
            raise
