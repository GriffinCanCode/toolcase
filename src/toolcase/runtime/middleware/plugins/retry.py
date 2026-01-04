"""Retry middleware for tool execution using stamina.

Handles exception-based retries at the middleware layer.
Complements RetryPolicy which handles error-code-based retries at the tool layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING

import stamina
from pydantic import BaseModel

from toolcase.foundation.errors import ErrorCode, ErrorTrace, JsonDict, ToolError, ToolException, classify_exception
from toolcase.runtime.middleware import Context, Next

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool

logger = logging.getLogger("toolcase.middleware")

RETRYABLE_CODES: frozenset[ErrorCode] = frozenset({ErrorCode.RATE_LIMITED, ErrorCode.TIMEOUT, ErrorCode.NETWORK_ERROR})


class _RetryableException(Exception):
    """Wrapper to mark exceptions as retryable for stamina."""
    __slots__ = ("original", "code")
    
    def __init__(self, original: Exception, code: ErrorCode) -> None:
        self.original, self.code = original, code
        super().__init__(str(original))


@dataclass(slots=True)
class RetryMiddleware:
    """Retry failed executions with exponential backoff using stamina.
    
    Retries on exceptions with retryable error codes (RATE_LIMITED, TIMEOUT,
    NETWORK_ERROR by default). For error-code-based retries on Result types,
    use RetryPolicy on the tool class instead.
    
    Stores ErrorTrace in context for observability integration, tracking
    all retry attempts and their outcomes.
    
    Args:
        max_attempts: Total attempts including initial (minimum 1)
        wait_initial: Initial wait between retries (default: 1s)
        wait_max: Maximum wait between retries (default: 30s)
        timeout: Total timeout for all retries (default: 45s)
        retryable_codes: Error codes that trigger retry
    
    Example:
        >>> registry.use(RetryMiddleware(max_attempts=3))
        >>> # Or with custom settings:
        >>> registry.use(RetryMiddleware(
        ...     max_attempts=5,
        ...     wait_initial=2.0,
        ...     wait_max=60.0,
        ... ))
    """
    
    max_attempts: int = 3
    wait_initial: float = 1.0
    wait_max: float = 30.0
    timeout: float = 45.0
    retryable_codes: frozenset[ErrorCode] = field(default_factory=lambda: RETRYABLE_CODES)
    
    def _should_retry(self, exc: Exception) -> bool:
        """Determine if exception is retryable based on error code."""
        code = exc.error.code if isinstance(exc, ToolException) else classify_exception(exc)
        return code in self.retryable_codes
    
    def _make_trace(self, exc: Exception, tool_name: str, attempt: int) -> ErrorTrace:
        """Create ErrorTrace from exception with retry context."""
        code = classify_exception(exc)
        return ErrorTrace(
            message=str(exc), error_code=code.value, recoverable=code in self.retryable_codes,
        ).with_operation("middleware:retry", tool=tool_name, attempt=attempt + 1, max_attempts=self.max_attempts)
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        last_exc: Exception | None = None
        retry_history: list[JsonDict] = []
        attempt = 0
        
        async def _execute() -> str:
            nonlocal last_exc, attempt
            try:
                result = await next(tool, params, ctx)
                ctx.update(retry_attempts=attempt + 1, retry_history=retry_history)
                return result
            except Exception as e:
                last_exc = e
                code = classify_exception(e)
                ctx["last_error_code"] = code.value
                retry_history.append({
                    "attempt": attempt + 1,
                    "error_code": code.value,
                    "message": str(e),
                    "retryable": code in self.retryable_codes,
                })
                if self._should_retry(e):
                    attempt += 1
                    raise _RetryableException(e, code)
                raise
        
        try:
            async for attempt_info in stamina.retry_context(
                on=_RetryableException,
                attempts=self.max_attempts,
                timeout=timedelta(seconds=self.timeout) if self.timeout else None,
                wait_initial=timedelta(seconds=self.wait_initial),
                wait_max=timedelta(seconds=self.wait_max),
                wait_jitter=timedelta(seconds=self.wait_initial * 0.5),
            ):
                with attempt_info:
                    if attempt_info.num > 1:
                        logger.warning(
                            f"[{tool.metadata.name}] Attempt {attempt_info.num} after "
                            f"{retry_history[-1]['error_code']}: {retry_history[-1]['message']}"
                        )
                    return await _execute()
        except _RetryableException as e:
            last_exc = e.original
        
        ctx.update(retry_attempts=self.max_attempts, retry_history=retry_history)
        if last_exc:
            ctx["error_trace"] = self._make_trace(last_exc, tool.metadata.name, self.max_attempts - 1)
            if not isinstance(last_exc, ToolException):
                raise ToolException(ToolError.from_exception(tool.metadata.name, last_exc, recoverable=False)) from last_exc
        raise last_exc  # type: ignore[misc]
