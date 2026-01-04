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

from ...errors import ErrorCode, ToolException, ToolError, classify_exception
from ...retry import Backoff, ExponentialBackoff
from ..middleware import Context, Next

if TYPE_CHECKING:
    from ...core import BaseTool

logger = logging.getLogger("toolcase.middleware")

# Default retryable error codes for exception-based retry
RETRYABLE_CODES: frozenset[ErrorCode] = frozenset({
    ErrorCode.RATE_LIMITED,
    ErrorCode.TIMEOUT,
    ErrorCode.NETWORK_ERROR,
})


@dataclass(slots=True)
class RetryMiddleware:
    """Retry failed executions with configurable backoff.
    
    Retries on exceptions with retryable error codes (RATE_LIMITED, TIMEOUT,
    NETWORK_ERROR by default). For error-code-based retries on Result types,
    use RetryPolicy on the tool class instead.
    
    Args:
        max_attempts: Total attempts including initial (minimum 1)
        backoff: Backoff strategy (default: ExponentialBackoff with jitter)
        retryable_codes: Error codes that trigger retry (default: transient errors)
    
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
    retryable_codes: frozenset[ErrorCode] = RETRYABLE_CODES
    
    def _should_retry(self, exc: Exception) -> bool:
        """Determine if exception is retryable based on error code."""
        # ToolExceptions already have classified codes
        if isinstance(exc, ToolException):
            return exc.error.code in self.retryable_codes
        # Classify other exceptions
        return classify_exception(exc) in self.retryable_codes
    
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
                code = classify_exception(e)
                ctx["last_error_code"] = code.value
                
                if attempt + 1 >= self.max_attempts or not self._should_retry(e):
                    break
                
                delay = self.backoff.delay(attempt)
                logger.warning(
                    f"[{tool.metadata.name}] Attempt {attempt + 1} failed ({code}): {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
        
        ctx["retry_attempts"] = self.max_attempts
        
        # Wrap non-ToolExceptions for consistent error handling
        if last_exc and not isinstance(last_exc, ToolException):
            raise ToolException(ToolError.from_exception(
                tool.metadata.name, last_exc, recoverable=False
            )) from last_exc
        raise last_exc  # type: ignore[misc]
