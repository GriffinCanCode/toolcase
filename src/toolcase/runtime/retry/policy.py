"""Retry policy configuration for tools using stamina.

Provides declarative retry behavior at the tool class level.
Works with ToolResult error codes, complementing middleware (exception-based).

Stamina provides exponential backoff with jitter by default - cleaner than custom implementations.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Annotated, Callable

import stamina
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, computed_field, field_serializer, field_validator

from toolcase.foundation.errors import ErrorCode

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from toolcase.foundation.errors import JsonDict, ToolResult


logger = logging.getLogger("toolcase.retry")

# Default retryable error codes - transient errors that may succeed on retry
DEFAULT_RETRYABLE: frozenset[ErrorCode] = frozenset({
    ErrorCode.RATE_LIMITED, ErrorCode.TIMEOUT, ErrorCode.NETWORK_ERROR,
})


class RetryPolicy(BaseModel):
    """Configurable retry policy for tool execution using stamina.
    
    Determines when and how to retry failed tool invocations based on
    error codes. Use as a class variable on BaseTool subclasses.
    
    This complements RetryMiddleware:
    - RetryPolicy: Retries based on ToolResult error codes
    - RetryMiddleware: Retries based on exceptions
    
    Attributes:
        max_retries: Maximum retry attempts (0 = no retries)
        timeout: Total timeout for all retries (default: 45s)
        wait_initial: Initial wait between retries (default: 1s)
        wait_max: Maximum wait between retries (default: 30s)
        wait_jitter: Add jitter to delays (default: True)
        retryable_codes: Error codes that trigger retry
        on_retry: Optional callback for retry events
    
    Example:
        >>> class SearchTool(BaseTool[SearchParams]):
        ...     retry_policy = RetryPolicy(
        ...         max_retries=3,
        ...         wait_initial=1.0,
        ...         wait_max=30.0,
        ...         retryable_codes=frozenset({ErrorCode.RATE_LIMITED}),
        ...     )
    """
    
    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        extra="forbid",
        revalidate_instances="never",
        json_schema_extra={
            "title": "Retry Policy",
            "description": "Configuration for automatic retry behavior using stamina",
            "examples": [{"max_retries": 3, "retryable_codes": ["RATE_LIMITED", "TIMEOUT", "NETWORK_ERROR"]}],
        },
    )
    
    max_retries: Annotated[int, Field(ge=0, le=10)] = 3
    timeout: Annotated[float, Field(ge=0.0, le=300.0)] = 45.0
    wait_initial: Annotated[float, Field(ge=0.0, le=60.0)] = 1.0
    wait_max: Annotated[float, Field(ge=0.0, le=120.0)] = 30.0
    wait_jitter: bool = True
    retryable_codes: frozenset[ErrorCode] = DEFAULT_RETRYABLE
    on_retry: Callable[[int, ErrorCode, float], None] | None = Field(default=None, exclude=True, repr=False)
    
    @field_validator("retryable_codes", mode="before")
    @classmethod
    def _normalize_codes(cls, v: frozenset[ErrorCode] | set[str] | list[str] | tuple[str, ...]) -> frozenset[ErrorCode]:
        """Accept strings and convert to ErrorCode enum."""
        if isinstance(v, frozenset) and all(isinstance(c, ErrorCode) for c in v):
            return v
        return frozenset(ErrorCode(c) if isinstance(c, str) else c for c in v)
    
    @field_serializer("retryable_codes")
    def _serialize_codes(self, v: frozenset[ErrorCode]) -> list[str]:
        return sorted(c.value for c in v)
    
    @computed_field
    @property
    def is_disabled(self) -> bool:
        return self.max_retries == 0 or not self.retryable_codes
    
    @computed_field
    @property
    def code_values(self) -> frozenset[str]:
        return frozenset(c.value for c in self.retryable_codes)
    
    def should_retry(self, code: ErrorCode | str, attempt: int) -> bool:
        """Determine if retry should be attempted."""
        return attempt < self.max_retries and (code.value if isinstance(code, ErrorCode) else code) in self.code_values
    
    def __hash__(self) -> int:
        return hash((self.max_retries, self.code_values))


NO_RETRY = RetryPolicy(max_retries=0, retryable_codes=frozenset())
_RetryPolicyAdapter: TypeAdapter[RetryPolicy] = TypeAdapter(RetryPolicy)


def validate_policy(data: "JsonDict") -> RetryPolicy:
    """Validate dict as RetryPolicy (fast path for config parsing)."""
    return _RetryPolicyAdapter.validate_python(data)


class _RetryableResultError(Exception):
    """Internal exception to signal retryable ToolResult error to stamina."""
    __slots__ = ("code", "attempt")
    
    def __init__(self, code: str, attempt: int) -> None:
        self.code, self.attempt = code, attempt
        super().__init__(f"Retryable error: {code}")


def _on_retry_hook(policy: RetryPolicy, tool_name: str):
    """Create stamina retry hook for logging and callbacks."""
    def hook(details: stamina.RetryDetails) -> None:
        exc = details.exception
        if isinstance(exc, _RetryableResultError):
            code, delay = exc.code, details.idle_for
            logger.info(f"[{tool_name}] Retry {details.attempt}/{policy.max_retries} after {delay:.1f}s (code: {code})")
            if policy.on_retry:
                policy.on_retry(details.attempt - 1, ErrorCode(code) if code else ErrorCode.UNKNOWN, delay)
    return hook


async def execute_with_retry(
    operation: Callable[[], Awaitable[ToolResult]], policy: RetryPolicy, tool_name: str,
) -> ToolResult:
    """Execute async operation with retry policy using stamina.
    
    Retries on retryable error codes. Stamina handles exponential backoff with jitter.
    """
    if policy.is_disabled:
        return await operation()
    
    attempt = 0
    
    async def _wrapped() -> ToolResult:
        nonlocal attempt
        result = await operation()
        if result.is_err():
            code = result.unwrap_err().error_code or ErrorCode.UNKNOWN.value
            if policy.should_retry(code, attempt):
                attempt += 1
                raise _RetryableResultError(code, attempt)
        return result
    
    try:
        async for attempt_info in stamina.retry_context(
            on=_RetryableResultError,
            attempts=policy.max_retries + 1,
            timeout=timedelta(seconds=policy.timeout) if policy.timeout else None,
            wait_initial=timedelta(seconds=policy.wait_initial),
            wait_max=timedelta(seconds=policy.wait_max),
            wait_jitter=timedelta(seconds=policy.wait_initial * 0.5) if policy.wait_jitter else timedelta(0),
        ):
            with attempt_info:
                return await _wrapped()
    except _RetryableResultError:
        pass  # All retries exhausted, return last result
    
    return await operation()  # Final attempt without retry wrapper


def execute_with_retry_sync(
    operation: Callable[[], ToolResult], policy: RetryPolicy, tool_name: str,
) -> ToolResult:
    """Execute sync operation with retry policy using stamina."""
    if policy.is_disabled:
        return operation()
    
    attempt = 0
    
    def _wrapped() -> ToolResult:
        nonlocal attempt
        result = operation()
        if result.is_err():
            code = result.unwrap_err().error_code or ErrorCode.UNKNOWN.value
            if policy.should_retry(code, attempt):
                attempt += 1
                raise _RetryableResultError(code, attempt)
        return result
    
    try:
        for attempt_info in stamina.retry_context(
            on=_RetryableResultError,
            attempts=policy.max_retries + 1,
            timeout=timedelta(seconds=policy.timeout) if policy.timeout else None,
            wait_initial=timedelta(seconds=policy.wait_initial),
            wait_max=timedelta(seconds=policy.wait_max),
            wait_jitter=timedelta(seconds=policy.wait_initial * 0.5) if policy.wait_jitter else timedelta(0),
        ):
            with attempt_info:
                return _wrapped()
    except _RetryableResultError:
        pass
    
    return operation()
