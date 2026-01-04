"""Retry policy configuration for tools.

Provides declarative retry behavior at the tool class level.
Works with ToolResult error codes, complementing middleware (exception-based).

Optimizations:
- Frozen for immutability and hashability
- TypeAdapter for fast validation
- Pre-computed disabled state
- Enum value caching
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Annotated, Callable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    field_validator,
)

from toolcase.foundation.errors import ErrorCode
from toolcase.runtime.concurrency import checkpoint

from .backoff import Backoff, ExponentialBackoff

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from toolcase.foundation.errors import ToolResult


logger = logging.getLogger("toolcase.retry")

# Default retryable error codes - transient errors that may succeed on retry
DEFAULT_RETRYABLE: frozenset[ErrorCode] = frozenset({
    ErrorCode.RATE_LIMITED,
    ErrorCode.TIMEOUT,
    ErrorCode.NETWORK_ERROR,
})

# Pre-computed set of retryable code values for fast string lookup
_DEFAULT_RETRYABLE_VALUES: frozenset[str] = frozenset(c.value for c in DEFAULT_RETRYABLE)


class RetryPolicy(BaseModel):
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
    
    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,  # For Backoff protocol
        validate_default=True,
        extra="forbid",
        revalidate_instances="never",
        json_schema_extra={
            "title": "Retry Policy",
            "description": "Configuration for automatic retry behavior",
            "examples": [{
                "max_retries": 3,
                "retryable_codes": ["RATE_LIMITED", "TIMEOUT", "NETWORK_ERROR"],
            }],
        },
    )
    
    max_retries: Annotated[int, Field(ge=0, le=10)] = 3
    backoff: Backoff = Field(default_factory=ExponentialBackoff, repr=False)
    retryable_codes: frozenset[ErrorCode] = DEFAULT_RETRYABLE
    on_retry: Callable[[int, ErrorCode, float], None] | None = Field(default=None, exclude=True, repr=False)
    
    # Cached string values for fast lookup (computed once)
    _code_values: frozenset[str] | None = None
    
    @field_validator("retryable_codes", mode="before")
    @classmethod
    def _normalize_codes(cls, v: frozenset[ErrorCode] | set[str] | list[str] | tuple[str, ...]) -> frozenset[ErrorCode]:
        """Accept strings and convert to ErrorCode enum."""
        if isinstance(v, frozenset) and all(isinstance(c, ErrorCode) for c in v):
            return v
        return frozenset(ErrorCode(c) if isinstance(c, str) else c for c in v)
    
    @field_serializer("retryable_codes")
    def _serialize_codes(self, v: frozenset[ErrorCode]) -> list[str]:
        """Serialize error codes as sorted string list."""
        return sorted(c.value for c in v)
    
    @computed_field
    @property
    def is_disabled(self) -> bool:
        """Whether retries are effectively disabled."""
        return self.max_retries == 0 or not self.retryable_codes
    
    def _get_code_values(self) -> frozenset[str]:
        """Get cached set of code string values for fast lookup."""
        cached = object.__getattribute__(self, "_code_values")
        if cached is None:
            values = frozenset(c.value for c in self.retryable_codes)
            object.__setattr__(self, "_code_values", values)
            return values
        return cached
    
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
        
        # Fast path: string lookup without enum conversion
        code_str = code.value if isinstance(code, ErrorCode) else code
        return code_str in self._get_code_values()
    
    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry attempt."""
        return self.backoff.delay(attempt)
    
    def __hash__(self) -> int:
        """Hash for frozen model."""
        return hash((self.max_retries, tuple(sorted(c.value for c in self.retryable_codes))))


# Singleton for no-retry policy (optimization)
NO_RETRY = RetryPolicy(max_retries=0, retryable_codes=frozenset())


async def execute_with_retry(
    operation: Callable[[], Awaitable[ToolResult]],
    policy: RetryPolicy,
    tool_name: str,
) -> ToolResult:
    """Execute async operation with retry policy.
    
    Retries on retryable error codes up to max_retries times.
    Uses cooperative cancellation for clean shutdown.
    
    Args:
        operation: Async callable returning ToolResult
        policy: Retry policy configuration
        tool_name: Tool name for logging
    
    Returns:
        ToolResult from successful attempt or last failed attempt
    """
    result = await operation()
    attempt = 0
    
    while result.is_err() and attempt < policy.max_retries:
        await checkpoint()  # Cooperative cancellation point
        
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
        await checkpoint()  # Check cancellation after sleep
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
