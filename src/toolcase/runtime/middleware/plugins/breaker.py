"""Circuit breaker middleware for fault tolerance.

Implements the circuit breaker pattern to prevent cascading failures
when external services are unavailable. Complements retry/timeout
by failing fast during known outages.

State Machine:
    CLOSED → failures exceed threshold → OPEN
    OPEN → recovery_time elapses → HALF_OPEN  
    HALF_OPEN → success → CLOSED
    HALF_OPEN → failure → OPEN
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel

from toolcase.foundation.errors import ErrorCode, ErrorTrace, ToolError, ToolException, classify_exception
from toolcase.runtime.middleware import Context, Next

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool


class State(IntEnum):
    """Circuit breaker states."""
    CLOSED = 0      # Normal operation, requests pass through
    OPEN = 1        # Failing fast, requests rejected
    HALF_OPEN = 2   # Testing recovery, limited requests


@dataclass(slots=True)
class CircuitState:
    """Per-circuit state tracking."""
    state: State = State.CLOSED
    failures: int = 0
    successes: int = 0
    last_failure: float = 0.0
    last_state_change: float = field(default_factory=time.time)


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
    
    Example:
        >>> registry.use(CircuitBreakerMiddleware(
        ...     failure_threshold=3,
        ...     recovery_time=60,
        ... ))
        >>> 
        >>> # After 3 failures, circuit opens:
        >>> # "**Tool Error (search):** Circuit open - failing fast"
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
    _circuits: dict[str, CircuitState] = field(default_factory=dict, repr=False)
    
    def _get_circuit(self, key: str) -> CircuitState:
        if key not in self._circuits:
            self._circuits[key] = CircuitState()
        return self._circuits[key]
    
    def _should_trip(self, code: ErrorCode) -> bool:
        """Check if error code should trip the breaker."""
        return code in self.trip_on
    
    def _check_state(self, circuit: CircuitState) -> State:
        """Evaluate and potentially transition circuit state."""
        now = time.time()
        
        if circuit.state == State.OPEN:
            # Check if recovery time elapsed
            if now - circuit.last_state_change >= self.recovery_time:
                circuit.state = State.HALF_OPEN
                circuit.successes = 0
                circuit.last_state_change = now
        
        return circuit.state
    
    def _record_success(self, circuit: CircuitState) -> None:
        """Record successful execution."""
        if circuit.state == State.HALF_OPEN:
            circuit.successes += 1
            if circuit.successes >= self.success_threshold:
                # Recovery confirmed
                circuit.state = State.CLOSED
                circuit.failures = 0
                circuit.last_state_change = time.time()
        elif circuit.state == State.CLOSED:
            # Reset failure count on success
            circuit.failures = max(0, circuit.failures - 1)
    
    def _record_failure(self, circuit: CircuitState, code: ErrorCode) -> None:
        """Record failed execution."""
        if not self._should_trip(code):
            return
        
        circuit.failures += 1
        circuit.last_failure = time.time()
        
        if circuit.state == State.HALF_OPEN:
            # Probe failed, back to open
            circuit.state = State.OPEN
            circuit.last_state_change = time.time()
        elif circuit.state == State.CLOSED and circuit.failures >= self.failure_threshold:
            # Threshold exceeded, open circuit
            circuit.state = State.OPEN
            circuit.last_state_change = time.time()
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        key = tool.metadata.name if self.per_tool else "_global_"
        circuit = self._get_circuit(key)
        state = self._check_state(circuit)
        
        # Store circuit info in context for observability
        ctx["circuit_state"] = state.name
        ctx["circuit_failures"] = circuit.failures
        ctx["circuit_key"] = key
        
        # Fail fast if open
        if state == State.OPEN:
            retry_in = self.recovery_time - (time.time() - circuit.last_state_change)
            trace = ErrorTrace(
                message=f"Circuit open - failing fast after {circuit.failures} failures. "
                        f"Retry in {retry_in:.0f}s",
                error_code=ErrorCode.EXTERNAL_SERVICE_ERROR.value,
                recoverable=True,
            ).with_operation(
                "middleware:circuit_breaker",
                tool=tool.metadata.name,
                state=state.name,
                failures=circuit.failures,
                retry_in_seconds=retry_in,
            )
            ctx["error_trace"] = trace
            raise ToolException(ToolError.create(
                tool.metadata.name,
                trace.message,
                ErrorCode.EXTERNAL_SERVICE_ERROR,
                recoverable=True,
            ))
        
        try:
            result = await next(tool, params, ctx)
            
            # Check for error responses (string-based error detection)
            is_error = result.startswith("**Tool Error")
            if is_error:
                code = ErrorCode.EXTERNAL_SERVICE_ERROR
                self._record_failure(circuit, code)
            else:
                self._record_success(circuit)
            
            # Update context with final state
            ctx["circuit_state"] = circuit.state.name
            return result
            
        except ToolException as e:
            self._record_failure(circuit, e.error.code)
            ctx["circuit_state"] = circuit.state.name
            raise
        except Exception as e:
            code = classify_exception(e)
            self._record_failure(circuit, code)
            ctx["circuit_state"] = circuit.state.name
            raise
    
    def get_state(self, tool_name: str) -> State:
        """Get current circuit state for a tool (for monitoring)."""
        key = tool_name if self.per_tool else "_global_"
        circuit = self._get_circuit(key)
        return self._check_state(circuit)
    
    def reset(self, tool_name: str | None = None) -> None:
        """Manually reset circuit(s) (for operations)."""
        if tool_name:
            key = tool_name if self.per_tool else "_global_"
            if key in self._circuits:
                self._circuits[key] = CircuitState()
        else:
            self._circuits.clear()
    
    def stats(self) -> dict[str, dict[str, object]]:
        """Get statistics for all circuits (for monitoring)."""
        return {
            key: {
                "state": circuit.state.name,
                "failures": circuit.failures,
                "successes": circuit.successes,
                "last_failure": circuit.last_failure,
            }
            for key, circuit in self._circuits.items()
        }
