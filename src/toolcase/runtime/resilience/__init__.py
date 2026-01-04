"""Resilience primitives for fault-tolerant tool execution.

Core building blocks for resilience patterns:
- CircuitBreaker: Fail-fast on repeated failures (state machine)
- StateStore/MemoryStateStore: Pluggable circuit state storage

These primitives are used by middleware but can also be used directly
for custom resilience patterns or monitoring.

Example:
    >>> from toolcase.runtime.resilience import CircuitBreaker, MemoryStateStore
    >>> 
    >>> # Standalone circuit breaker
    >>> breaker = CircuitBreaker(failure_threshold=3, recovery_time=30)
    >>> if breaker.allow():
    ...     try:
    ...         result = call_external_service()
    ...         breaker.record_success()
    ...     except Exception:
    ...         breaker.record_failure()
    ... else:
    ...     result = fallback()

Distributed state:
    >>> from toolcase.middleware.plugins import RedisStateStore
    >>> store = RedisStateStore.from_url("redis://localhost:6379/0")
    >>> breaker = CircuitBreaker(store=store, key="my_service")
"""

from .breaker import (
    CircuitBreaker,
    CircuitState,
    MemoryStateStore,
    State,
    StateStore,
)

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "State",
    "StateStore",
    "MemoryStateStore",
]
