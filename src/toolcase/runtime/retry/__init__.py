"""Retry policies for tool execution using stamina.

Provides configurable retry behavior at the tool class level using
stamina's clean retry API with exponential backoff and jitter.
Includes composable retry strategies for combining retry, fallback, and escalation.

Example:
    >>> from toolcase import BaseTool, ToolMetadata
    >>> from toolcase.retry import RetryPolicy
    >>> from toolcase.errors import ErrorCode
    >>> 
    >>> class SearchTool(BaseTool[SearchParams]):
    ...     metadata = ToolMetadata(name="search", description="Search the web")
    ...     params_schema = SearchParams
    ...     
    ...     retry_policy = RetryPolicy(
    ...         max_retries=3,
    ...         wait_initial=1.0,
    ...         wait_max=30.0,
    ...         retryable_codes=frozenset({ErrorCode.RATE_LIMITED, ErrorCode.TIMEOUT}),
    ...     )
    ...     
    ...     async def _async_run(self, params: SearchParams) -> str:
    ...         return search_api(params.query)

Composed strategies:
    >>> from toolcase.retry import RetryStrategy, resilient_tool
    >>> 
    >>> # Fluent builder
    >>> strategy = (
    ...     RetryStrategy()
    ...     .with_retry(max_retries=3)
    ...     .with_fallback([BackupAPI()])
    ...     .with_escalation("approval_queue")
    ... )
    >>> resilient = strategy.wrap(PrimaryAPI())
    >>> 
    >>> # Helper function
    >>> resilient = resilient_tool(
    ...     PrimaryAPI(),
    ...     retry=3,
    ...     fallback=[BackupAPI()],
    ...     escalate="approval_queue",
    ... )
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


# Lazy imports for strategy module to avoid circular import with BaseTool
def __getattr__(name: str):
    """Lazy load strategy module components."""
    _strategy_exports = {
        "DEFAULT_FALLBACK_CODES", "EscalateStage", "FallbackStage", "ResilientTool",
        "RetryStage", "RetryStrategy", "Stage", "resilient_tool",
    }
    if name in _strategy_exports:
        from . import strategy
        return getattr(strategy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Backoff strategies (for batch retry)
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
    # Strategy (composed) - lazy loaded
    "RetryStrategy",
    "ResilientTool",
    "RetryStage",
    "FallbackStage",
    "EscalateStage",
    "Stage",
    "DEFAULT_FALLBACK_CODES",
    "resilient_tool",
]
