"""Core circuit breaker primitive for fault tolerance.

Implements the circuit breaker pattern as a standalone state machine.
Used by CircuitBreakerMiddleware but can be used directly for custom patterns.

State Machine:
    CLOSED → failures exceed threshold → OPEN
    OPEN → recovery_time elapses → HALF_OPEN  
    HALF_OPEN → success → CLOSED
    HALF_OPEN → failure → OPEN

Distributed Support:
    By default, state is in-memory. For distributed deployments,
    inject a StateStore (e.g., RedisStateStore) to share state across instances.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol, runtime_checkable

from toolcase.foundation.errors import CircuitStateDict


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
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is in OPEN state (fail-fast mode)."""
        return self.state == State.OPEN
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is in CLOSED state (normal operation)."""
        return self.state == State.CLOSED
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is in HALF_OPEN state (testing recovery)."""
        return self.state == State.HALF_OPEN
    
    def to_dict(self) -> CircuitStateDict:
        """Serialize for distributed storage."""
        return {
            "state": self.state, "failures": self.failures, "successes": self.successes,
            "last_failure": self.last_failure, "last_state_change": self.last_state_change,
        }
    
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
class CircuitBreaker:
    """Standalone circuit breaker state machine.
    
    Encapsulates the circuit breaker pattern independently of middleware.
    Can be used directly or composed into middleware/registry patterns.
    
    Args:
        failure_threshold: Failures before opening circuit (default: 5)
        recovery_time: Seconds before half-open probe (default: 30)
        success_threshold: Successes in half-open to close (default: 2)
        store: State storage backend (default: MemoryStateStore)
        key: Circuit identifier for multi-circuit scenarios (default: "_default_")
    
    Example (standalone):
        >>> breaker = CircuitBreaker(failure_threshold=3)
        >>> if breaker.allow():
        ...     try:
        ...         result = call_service()
        ...         breaker.record_success()
        ...     except Exception:
        ...         breaker.record_failure()
        ... else:
        ...     result = "Circuit open - using fallback"
    
    Example (monitoring):
        >>> breaker.state        # Current State enum
        >>> breaker.failures     # Current failure count
        >>> breaker.is_open      # Quick check if failing fast
        >>> breaker.retry_after  # Seconds until half-open probe
    """
    
    failure_threshold: int = 5
    recovery_time: float = 30.0
    success_threshold: int = 2
    store: StateStore = field(default_factory=MemoryStateStore, repr=False)
    key: str = "_default_"
    
    def _circuit(self) -> CircuitState:
        """Get or create circuit state."""
        if (state := self.store.get(self.key)) is None:
            state = CircuitState()
            self.store.set(self.key, state)
        return state
    
    def _save(self, circuit: CircuitState) -> None:
        """Persist circuit state (required for distributed stores)."""
        self.store.set(self.key, circuit)
    
    def _evaluate_state(self, circuit: CircuitState) -> State:
        """Evaluate and potentially transition circuit state."""
        if circuit.state == State.OPEN and time.time() - circuit.last_state_change >= self.recovery_time:
            circuit.state, circuit.successes, circuit.last_state_change = State.HALF_OPEN, 0, time.time()
            self._save(circuit)
        return circuit.state
    
    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────
    
    def allow(self) -> bool:
        """Check if request should be allowed through.
        
        Returns:
            True if request should proceed, False if circuit is open.
        
        Note: Also triggers state transitions (OPEN → HALF_OPEN when recovery_time elapsed).
        """
        return self._evaluate_state(self._circuit()) != State.OPEN
    
    def record_success(self) -> None:
        """Record successful execution."""
        circuit = self._circuit()
        if circuit.state == State.HALF_OPEN:
            circuit.successes += 1
            if circuit.successes >= self.success_threshold:  # Recovery confirmed
                circuit.state, circuit.failures, circuit.last_state_change = State.CLOSED, 0, time.time()
            self._save(circuit)
        elif circuit.state == State.CLOSED and circuit.failures > 0:
            circuit.failures = max(0, circuit.failures - 1)
            self._save(circuit)
    
    def record_failure(self) -> None:
        """Record failed execution."""
        circuit = self._circuit()
        circuit.failures += 1
        circuit.last_failure = time.time()
        if circuit.state == State.HALF_OPEN or (circuit.state == State.CLOSED and circuit.failures >= self.failure_threshold):
            circuit.state, circuit.last_state_change = State.OPEN, time.time()
        self._save(circuit)
    
    def reset(self) -> None:
        """Manually reset circuit to closed state."""
        self.store.set(self.key, CircuitState())
    
    # ─────────────────────────────────────────────────────────────────
    # Observability Properties
    # ─────────────────────────────────────────────────────────────────
    
    @property
    def state(self) -> State:
        """Current circuit state (evaluates transitions)."""
        return self._evaluate_state(self._circuit())
    
    @property
    def failures(self) -> int:
        """Current failure count."""
        return self._circuit().failures
    
    @property
    def successes(self) -> int:
        """Current success count (meaningful in HALF_OPEN)."""
        return self._circuit().successes
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (fail-fast mode)."""
        return self.state == State.OPEN
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == State.CLOSED
    
    @property
    def retry_after(self) -> float | None:
        """Seconds until circuit transitions to half-open, or None if not open."""
        circuit = self._circuit()
        if circuit.state != State.OPEN:
            return None
        remaining = self.recovery_time - (time.time() - circuit.last_state_change)
        return max(0.0, remaining)
    
    @property
    def stats(self) -> CircuitStateDict:
        """Get circuit statistics for monitoring."""
        return self._circuit().to_dict()
