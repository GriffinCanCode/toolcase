"""Retry middleware for tool execution.

Handles exception-based retries at the middleware layer.
Complements RetryPolicy which handles error-code-based retries at the tool layer.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ...retry import Backoff, ExponentialBackoff
from ..middleware import Context, Next

if TYPE_CHECKING:
    from ...core import BaseTool

logger = logging.getLogger("toolcase.middleware")


@dataclass(slots=True)
class RetryMiddleware:
    """Retry failed executions with configurable backoff.
    
    Retries on exceptions only (not on error results). For error-code-based
    retries, use RetryPolicy on the tool class instead.
    
    Args:
        max_attempts: Total attempts including initial (minimum 1)
        backoff: Backoff strategy (default: ExponentialBackoff with jitter)
    
    Example:
        >>> from toolcase.retry import ExponentialBackoff, LinearBackoff
        >>> registry.use(RetryMiddleware(max_attempts=3))
        >>> # Or with custom backoff:
        >>> registry.use(RetryMiddleware(
        ...     max_attempts=5,
        ...     backoff=LinearBackoff(base=1.0, increment=2.0)
        ... ))
    """
    
    max_attempts: int = 3
    backoff: Backoff = field(default_factory=ExponentialBackoff)
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        last_exc: Exception | None = None
        
        for attempt in range(self.max_attempts):
            try:
                result = await next(tool, params, ctx)
                ctx["retry_attempts"] = attempt + 1
                return result
            except Exception as e:
                last_exc = e
                if attempt + 1 >= self.max_attempts:
                    break
                
                delay = self.backoff.delay(attempt)
                logger.warning(
                    f"[{tool.metadata.name}] Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
        
        ctx["retry_attempts"] = self.max_attempts
        raise last_exc  # type: ignore[misc]
