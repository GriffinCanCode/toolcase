RESILIENCE = """
TOPIC: resilience
==================

Fault tolerance primitives for robust tool execution.

CIRCUIT BREAKER:
    Core primitive for the circuit breaker pattern. Tracks failures and
    opens circuit to fail fast when a service is unhealthy.
    
    State Machine:
        CLOSED → failures >= threshold → OPEN
        OPEN → recovery_time elapsed → HALF_OPEN  
        HALF_OPEN → success → CLOSED
        HALF_OPEN → failure → OPEN

STANDALONE USAGE:
    from toolcase.runtime.resilience import CircuitBreaker, State
    
    # Create a circuit breaker
    breaker = CircuitBreaker(failure_threshold=3, recovery_time=30)
    
    # Use in request handling
    if breaker.allow():
        try:
            result = call_external_service()
            breaker.record_success()
        except Exception:
            breaker.record_failure()
    else:
        result = fallback_response()
    
    # Observability
    breaker.state       # State enum (CLOSED, OPEN, HALF_OPEN)
    breaker.is_open     # Quick boolean check
    breaker.failures    # Current failure count
    breaker.retry_after # Seconds until half-open probe (or None)
    breaker.stats       # Full state dict for monitoring

MIDDLEWARE USAGE (Recommended):
    from toolcase import get_registry
    from toolcase.runtime.middleware import CircuitBreakerMiddleware
    
    registry = get_registry()
    registry.use(CircuitBreakerMiddleware(
        failure_threshold=5,   # Failures before opening
        recovery_time=30,      # Seconds before half-open
        success_threshold=2,   # Successes to close again
        per_tool=True,         # Separate circuit per tool
    ))

REGISTRY OBSERVABILITY:
    # Circuit state for a tool
    state = registry.circuit_state("search")
    if state and state == State.OPEN:
        print("Search circuit is open")
    
    # Quick check if failing fast
    if registry.circuit_is_open("search"):
        return fallback()
    
    # Get stats for all circuits
    stats = registry.circuit_stats()
    # {'search': {'state': 0, 'failures': 0, ...}}
    
    # Manual reset
    registry.reset_circuit("search")  # Reset one
    registry.reset_circuit()          # Reset all
    
    # Direct access to breaker primitive
    breaker = registry.get_circuit_breaker("search")
    if breaker and breaker.retry_after:
        print(f"Retry in {breaker.retry_after:.0f}s")

DISTRIBUTED STATE:
    from toolcase.runtime.middleware.plugins import RedisStateStore
    
    # Share circuit state across instances
    store = RedisStateStore.from_url("redis://localhost:6379/0")
    registry.use(CircuitBreakerMiddleware(store=store))

ERROR CODES THAT TRIP BREAKER:
    By default: TIMEOUT, NETWORK_ERROR, EXTERNAL_SERVICE_ERROR
    
    # Customize which errors trip the breaker
    from toolcase.foundation.errors import ErrorCode
    registry.use(CircuitBreakerMiddleware(
        trip_on=frozenset({ErrorCode.RATE_LIMITED, ErrorCode.TIMEOUT})
    ))

COMBINING WITH OTHER MIDDLEWARE:
    # Recommended order: circuit breaker → retry → timeout
    registry.use(CircuitBreakerMiddleware(failure_threshold=5))
    registry.use(RetryMiddleware(max_attempts=3))
    registry.use(TimeoutMiddleware(30.0))
    
    # Circuit breaker prevents retry storms during outages

RELATED TOPICS:
    toolcase help middleware  All middleware options
    toolcase help retry       Retry policies and backoff
    toolcase help registry    Registry integration
"""
