"""Retry policies for tool execution.

Provides configurable retry behavior at the tool class level with
pluggable backoff strategies.

Example:
    >>> from toolcase import BaseTool, ToolMetadata
    >>> from toolcase.retry import RetryPolicy, ExponentialBackoff
    >>> from toolcase.errors import ErrorCode
    >>> 
    >>> class SearchTool(BaseTool[SearchParams]):
    ...     metadata = ToolMetadata(name="search", description="Search the web")
    ...     params_schema = SearchParams
    ...     
    ...     retry_policy = RetryPolicy(
    ...         max_retries=3,
    ...         backoff=ExponentialBackoff(base=1.0, max_delay=30.0),
    ...         retryable_codes=frozenset({ErrorCode.RATE_LIMITED, ErrorCode.TIMEOUT}),
    ...     )
    ...     
    ...     async def _async_run(self, params: SearchParams) -> str:
    ...         return search_api(params.query)
"""

from .backoff import (
    Backoff,
    ConstantBackoff,
    DecorrelatedJitter,
    ExponentialBackoff,
    LinearBackoff,
)
from .policy import (
    DEFAULT_RETRYABLE,
    NO_RETRY,
    RetryPolicy,
    execute_with_retry,
    execute_with_retry_sync,
    validate_policy,
)

__all__ = [
    # Backoff strategies
    "Backoff",
    "ExponentialBackoff",
    "LinearBackoff",
    "ConstantBackoff",
    "DecorrelatedJitter",
    # Policy
    "RetryPolicy",
    "DEFAULT_RETRYABLE",
    "NO_RETRY",
    "validate_policy",
    # Execution
    "execute_with_retry",
    "execute_with_retry_sync",
]
