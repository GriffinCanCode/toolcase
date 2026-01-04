"""Circuit breaker middleware for fault tolerance.

Adapts the core CircuitBreaker primitive to the middleware pattern.
Handles error code classification and middleware chain integration.

State Machine (delegated to core):
    CLOSED → failures exceed threshold → OPEN
    OPEN → recovery_time elapses → HALF_OPEN  
    HALF_OPEN → success → CLOSED
    HALF_OPEN → failure → OPEN

Distributed Support:
    By default, state is in-memory. For distributed deployments,
    inject a RedisStateStore to share state across instances.
    
    >>> from toolcase.middleware.plugins.store import RedisStateStore
    >>> store = RedisStateStore.from_url("redis://localhost:6379/0")
    >>> registry.use(CircuitBreakerMiddleware(store=store))
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel

from toolcase.foundation.errors import (
    CircuitStateDict,
    ErrorCode,
    ErrorTrace,
    ToolError,
    ToolException,
    classify_exception,
)
from toolcase.runtime.middleware import Context, Next
from toolcase.runtime.resilience import CircuitBreaker, CircuitState, MemoryStateStore, State, StateStore

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool


@dataclass(slots=True)
class CircuitBreakerMiddleware:
    """Fail fast when tools are experiencing failures.
    
    Tracks failure rates per-tool and opens circuit when threshold
    exceeded. Open circuits reject requests immediately, preventing
    resource exhaustion during outages. After recovery_time, allows
    probe requests to test if service recovered.
    
    Complements:
    - TimeoutMiddleware: Breaker opens after repeated timeouts
    - RetryMiddleware: Breaker prevents retry storms during outages
    - RateLimitMiddleware: Different purpose - breaker is reactive
    
    Args:
        failure_threshold: Failures before opening circuit (default: 5)
        recovery_time: Seconds before half-open probe (default: 30)
        success_threshold: Successes in half-open to close (default: 2)
        per_tool: Track per-tool (True) or global (False)
        trip_on: Error codes that trip the breaker (default: transient)
        store: State storage backend (default: MemoryStateStore)
    
    Example:
        >>> registry.use(CircuitBreakerMiddleware(
        ...     failure_threshold=3,
        ...     recovery_time=60,
        ... ))
        >>> 
        >>> # After 3 failures, circuit opens:
        >>> # "**Tool Error (search):** Circuit open - failing fast"
        
        # Distributed deployment with Redis:
        >>> from toolcase.middleware.plugins.store import RedisStateStore
        >>> store = RedisStateStore.from_url("redis://localhost:6379/0")
        >>> registry.use(CircuitBreakerMiddleware(store=store))
    """
    
    failure_threshold: int = 5
    recovery_time: float = 30.0
    success_threshold: int = 2
    per_tool: bool = True
    trip_on: frozenset[ErrorCode] = field(default_factory=lambda: frozenset({
        ErrorCode.TIMEOUT,
        ErrorCode.NETWORK_ERROR,
        ErrorCode.EXTERNAL_SERVICE_ERROR,
    }))
    store: StateStore = field(default_factory=MemoryStateStore, repr=False)
    _breakers: dict[str, CircuitBreaker] = field(default_factory=dict, repr=False, init=False)
    
    def _get_breaker(self, key: str) -> CircuitBreaker:
        """Get or create a CircuitBreaker for the given key."""
        if key not in self._breakers:
            self._breakers[key] = CircuitBreaker(
                failure_threshold=self.failure_threshold,
                recovery_time=self.recovery_time,
                success_threshold=self.success_threshold,
                store=self.store,
                key=key,
            )
        return self._breakers[key]
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        key = tool.metadata.name if self.per_tool else "_global_"
        breaker = self._get_breaker(key)
        state = breaker.state
        
        # Store circuit info in context for observability
        ctx["circuit_state"], ctx["circuit_failures"], ctx["circuit_key"] = state.name, breaker.failures, key
        
        # Fail fast if open
        if state == State.OPEN:
            retry_in = breaker.retry_after or 0
            trace = ErrorTrace(
                message=f"Circuit open - failing fast after {breaker.failures} failures. Retry in {retry_in:.0f}s",
                error_code=ErrorCode.EXTERNAL_SERVICE_ERROR.value,
                recoverable=True,
            ).with_operation(
                "middleware:circuit_breaker", tool=tool.metadata.name,
                state=state.name, failures=breaker.failures, retry_in_seconds=retry_in,
            )
            ctx["error_trace"] = trace
            raise ToolException(ToolError.create(
                tool.metadata.name, trace.message, ErrorCode.EXTERNAL_SERVICE_ERROR, recoverable=True,
            ))
        
        try:
            result = await next(tool, params, ctx)
            (self._record_failure(breaker, ErrorCode.EXTERNAL_SERVICE_ERROR) 
             if result.startswith("**Tool Error") else breaker.record_success())
        except ToolException as e:
            self._record_failure(breaker, e.error.code)
            ctx["circuit_state"] = breaker.state.name
            raise
        except Exception as e:
            self._record_failure(breaker, classify_exception(e))
            ctx["circuit_state"] = breaker.state.name
            raise
        ctx["circuit_state"] = breaker.state.name
        return result
    
    def _record_failure(self, breaker: CircuitBreaker, code: ErrorCode) -> None:
        """Record failure only if error code should trip the breaker."""
        if code in self.trip_on:
            breaker.record_failure()
    
    # ─────────────────────────────────────────────────────────────────
    # Observability API (delegates to core breakers)
    # ─────────────────────────────────────────────────────────────────
    
    def get_breaker(self, tool_name: str) -> CircuitBreaker:
        """Get the CircuitBreaker for a tool (creates if not exists).
        
        Provides direct access to the core primitive for advanced monitoring
        or manual intervention.
        
        Args:
            tool_name: Tool name (or "_global_" if per_tool=False)
        
        Returns:
            CircuitBreaker instance for the tool
        """
        key = tool_name if self.per_tool else "_global_"
        return self._get_breaker(key)
    
    def get_state(self, tool_name: str) -> State:
        """Get current circuit state for a tool (for monitoring)."""
        return self.get_breaker(tool_name).state
    
    def reset(self, tool_name: str | None = None) -> None:
        """Manually reset circuit(s) (for operations)."""
        if tool_name:
            key = tool_name if self.per_tool else "_global_"
            if key in self._breakers:
                self._breakers[key].reset()
        else:
            for breaker in self._breakers.values():
                breaker.reset()
    
    def stats(self) -> dict[str, CircuitStateDict]:
        """Get statistics for all circuits (for monitoring)."""
        return {key: breaker.stats for key, breaker in self._breakers.items()}


# Re-export core types for convenience
__all__ = [
    "CircuitBreakerMiddleware",
    "CircuitBreaker",
    "CircuitState",
    "State",
    "StateStore",
    "MemoryStateStore",
]
