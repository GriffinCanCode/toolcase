"""Circuit breaker middleware for fault tolerance.

Implements the circuit breaker pattern to prevent cascading failures
when external services are unavailable. Complements retry/timeout
by failing fast during known outages.

State Machine:
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
from enum import IntEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

from toolcase.foundation.errors import (
    CircuitStateDict,
    ErrorCode,
    ErrorTrace,
    JsonDict,
    ToolError,
    ToolException,
    classify_exception,
)
from toolcase.runtime.middleware import Context, Next

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool


class State(IntEnum):
    """Circuit breaker states."""
    CLOSED, OPEN, HALF_OPEN = 0, 1, 2  # Normal → Failing fast → Testing recovery


@dataclass(slots=True)
class CircuitState:
    """Per-circuit state tracking."""
    state: State = State.CLOSED
    failures: int = 0
    successes: int = 0
    last_failure: float = 0.0
    last_state_change: float = field(default_factory=time.time)
    
    def to_dict(self) -> CircuitStateDict:
        """Serialize for distributed storage."""
        return {"state": self.state, "failures": self.failures, "successes": self.successes,
                "last_failure": self.last_failure, "last_state_change": self.last_state_change}
    
    @classmethod
    def from_dict(cls, d: CircuitStateDict) -> CircuitState:
        """Deserialize from distributed storage."""
        return cls(State(d["state"]), int(d["failures"]), int(d["successes"]),
                   float(d["last_failure"]), float(d["last_state_change"]))


@runtime_checkable
class StateStore(Protocol):
    """Protocol for circuit state storage backends."""
    def get(self, key: str) -> CircuitState | None: ...
    def set(self, key: str, state: CircuitState) -> None: ...
    def delete(self, key: str) -> bool: ...
    def keys(self) -> list[str]: ...


class MemoryStateStore:
    """Thread-safe in-memory state store (default).
    
    Suitable for single-instance deployments. For distributed systems,
    use RedisStateStore to share circuit state across instances.
    """
    __slots__ = ("_states",)
    
    def __init__(self) -> None:
        self._states: dict[str, CircuitState] = {}
    
    def get(self, key: str) -> CircuitState | None:
        return self._states.get(key)
    
    def set(self, key: str, state: CircuitState) -> None:
        self._states[key] = state
    
    def delete(self, key: str) -> bool:
        return self._states.pop(key, None) is not None
    
    def keys(self) -> list[str]:
        return list(self._states)


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
    
    def _get_circuit(self, key: str) -> CircuitState:
        if (state := self.store.get(key)) is None:
            state = CircuitState()
            self.store.set(key, state)
        return state
    
    def _save_circuit(self, key: str, circuit: CircuitState) -> None:
        """Persist circuit state (required for distributed stores)."""
        self.store.set(key, circuit)
    
    def _check_state(self, key: str, circuit: CircuitState) -> State:
        """Evaluate and potentially transition circuit state."""
        if circuit.state == State.OPEN and time.time() - circuit.last_state_change >= self.recovery_time:
            circuit.state, circuit.successes, circuit.last_state_change = State.HALF_OPEN, 0, time.time()
            self._save_circuit(key, circuit)
        return circuit.state
    
    def _record_success(self, key: str, circuit: CircuitState) -> None:
        """Record successful execution."""
        if circuit.state == State.HALF_OPEN:
            circuit.successes += 1
            if circuit.successes >= self.success_threshold:  # Recovery confirmed
                circuit.state, circuit.failures, circuit.last_state_change = State.CLOSED, 0, time.time()
            self._save_circuit(key, circuit)
        elif circuit.state == State.CLOSED and circuit.failures > 0:
            circuit.failures = max(0, circuit.failures - 1)
            self._save_circuit(key, circuit)
    
    def _record_failure(self, key: str, circuit: CircuitState, code: ErrorCode) -> None:
        """Record failed execution."""
        if code not in self.trip_on:
            return
        circuit.failures += 1
        circuit.last_failure = time.time()
        if circuit.state == State.HALF_OPEN or (circuit.state == State.CLOSED and circuit.failures >= self.failure_threshold):
            circuit.state, circuit.last_state_change = State.OPEN, time.time()
        self._save_circuit(key, circuit)
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        key = tool.metadata.name if self.per_tool else "_global_"
        circuit = self._get_circuit(key)
        state = self._check_state(key, circuit)
        
        # Store circuit info in context for observability
        ctx.update(circuit_state=state.name, circuit_failures=circuit.failures, circuit_key=key)
        
        # Fail fast if open
        if state == State.OPEN:
            retry_in = self.recovery_time - (time.time() - circuit.last_state_change)
            trace = ErrorTrace(
                message=f"Circuit open - failing fast after {circuit.failures} failures. Retry in {retry_in:.0f}s",
                error_code=ErrorCode.EXTERNAL_SERVICE_ERROR.value,
                recoverable=True,
            ).with_operation(
                "middleware:circuit_breaker", tool=tool.metadata.name,
                state=state.name, failures=circuit.failures, retry_in_seconds=retry_in,
            )
            ctx["error_trace"] = trace
            raise ToolException(ToolError.create(
                tool.metadata.name, trace.message, ErrorCode.EXTERNAL_SERVICE_ERROR, recoverable=True,
            ))
        
        try:
            result = await next(tool, params, ctx)
            (self._record_failure(key, circuit, ErrorCode.EXTERNAL_SERVICE_ERROR) 
             if result.startswith("**Tool Error") else self._record_success(key, circuit))
        except ToolException as e:
            self._record_failure(key, circuit, e.error.code)
            ctx["circuit_state"] = circuit.state.name
            raise
        except Exception as e:
            self._record_failure(key, circuit, classify_exception(e))
            ctx["circuit_state"] = circuit.state.name
            raise
        ctx["circuit_state"] = circuit.state.name
        return result
    
    def get_state(self, tool_name: str) -> State:
        """Get current circuit state for a tool (for monitoring)."""
        key = tool_name if self.per_tool else "_global_"
        circuit = self._get_circuit(key)
        return self._check_state(key, circuit)
    
    def reset(self, tool_name: str | None = None) -> None:
        """Manually reset circuit(s) (for operations)."""
        if tool_name:
            key = tool_name if self.per_tool else "_global_"
            if self.store.get(key):
                self.store.set(key, CircuitState())
        else:
            for key in self.store.keys():
                self.store.delete(key)
    
    def stats(self) -> dict[str, CircuitStateDict]:
        """Get statistics for all circuits (for monitoring)."""
        return {
            key: circuit.to_dict()
            for key in self.store.keys()
            if (circuit := self.store.get(key))
        }
