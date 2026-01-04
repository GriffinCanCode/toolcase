"""Retry policy configuration for tools.

Provides declarative retry behavior at the tool class level.
Works with ToolResult error codes, complementing middleware (exception-based).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from ..errors import ErrorCode
from .backoff import Backoff, ExponentialBackoff

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from ..errors import ToolResult


logger = logging.getLogger("toolcase.retry")

# Default retryable error codes - transient errors that may succeed on retry
DEFAULT_RETRYABLE: frozenset[ErrorCode] = frozenset({
    ErrorCode.RATE_LIMITED,
    ErrorCode.TIMEOUT,
    ErrorCode.NETWORK_ERROR,
})


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Configurable retry policy for tool execution.
    
    Determines when and how to retry failed tool invocations based on
    error codes. Use as a class variable on BaseTool subclasses.
    
    This complements RetryMiddleware:
    - RetryPolicy: Retries based on ToolResult error codes
    - RetryMiddleware: Retries based on exceptions
    
    Attributes:
        max_retries: Maximum retry attempts (0 = no retries)
        backoff: Backoff strategy for delay calculation
        retryable_codes: Error codes that trigger retry
        on_retry: Optional callback for retry events
    
    Example:
        >>> class SearchTool(BaseTool[SearchParams]):
        ...     retry_policy = RetryPolicy(
        ...         max_retries=3,
        ...         backoff=ExponentialBackoff(base=1.0, max_delay=30.0),
        ...         retryable_codes=frozenset({ErrorCode.RATE_LIMITED}),
        ...     )
    """
    
    max_retries: int = 3
    backoff: Backoff = field(default_factory=ExponentialBackoff)
    retryable_codes: frozenset[ErrorCode] = DEFAULT_RETRYABLE
    on_retry: Callable[[int, ErrorCode, float], None] | None = None
    
    def should_retry(self, code: ErrorCode | str, attempt: int) -> bool:
        """Determine if retry should be attempted.
        
        Args:
            code: Error code from failed result
            attempt: Current attempt number (0-indexed, after first failure)
        
        Returns:
            True if retry should be attempted
        """
        if attempt >= self.max_retries:
            return False
        
        # Handle string error codes
        if isinstance(code, str):
            try:
                code = ErrorCode(code)
            except ValueError:
                return False
        
        return code in self.retryable_codes
    
    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry attempt."""
        return self.backoff.delay(attempt)


# Singleton for no-retry policy (optimization)
NO_RETRY = RetryPolicy(max_retries=0, retryable_codes=frozenset())


async def execute_with_retry(
    operation: Callable[[], Awaitable[ToolResult]],
    policy: RetryPolicy,
    tool_name: str,
) -> ToolResult:
    """Execute async operation with retry policy.
    
    Retries on retryable error codes up to max_retries times.
    
    Args:
        operation: Async callable returning ToolResult
        policy: Retry policy configuration
        tool_name: Tool name for logging
    
    Returns:
        ToolResult from successful attempt or last failed attempt
    """
    from ..errors import Err, Ok
    
    result = await operation()
    attempt = 0
    
    while result.is_err() and attempt < policy.max_retries:
        trace = result.unwrap_err()
        code = trace.error_code
        
        if not policy.should_retry(code, attempt):
            break
        
        delay = policy.get_delay(attempt)
        
        logger.info(
            f"[{tool_name}] Retry {attempt + 1}/{policy.max_retries} "
            f"after {delay:.1f}s (code: {code})"
        )
        
        if policy.on_retry:
            policy.on_retry(attempt, ErrorCode(code) if code else ErrorCode.UNKNOWN, delay)
        
        await asyncio.sleep(delay)
        result = await operation()
        attempt += 1
    
    return result


def execute_with_retry_sync(
    operation: Callable[[], ToolResult],
    policy: RetryPolicy,
    tool_name: str,
) -> ToolResult:
    """Execute sync operation with retry policy.
    
    Synchronous version for non-async tool implementations.
    """
    import time
    
    result = operation()
    attempt = 0
    
    while result.is_err() and attempt < policy.max_retries:
        trace = result.unwrap_err()
        code = trace.error_code
        
        if not policy.should_retry(code, attempt):
            break
        
        delay = policy.get_delay(attempt)
        
        logger.info(
            f"[{tool_name}] Retry {attempt + 1}/{policy.max_retries} "
            f"after {delay:.1f}s (code: {code})"
        )
        
        if policy.on_retry:
            policy.on_retry(attempt, ErrorCode(code) if code else ErrorCode.UNKNOWN, delay)
        
        time.sleep(delay)
        result = operation()
        attempt += 1
    
    return result
