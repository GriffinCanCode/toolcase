"""Built-in middleware implementations for common cross-cutting concerns.

These middleware are ready to use and demonstrate the middleware pattern.
They have no external dependencies beyond the standard library.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

from .middleware import Context, Next

if TYPE_CHECKING:
    from ..core import BaseTool


logger = logging.getLogger("toolcase.middleware")


# ─────────────────────────────────────────────────────────────────────────────
# Logging Middleware
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class LoggingMiddleware:
    """Log tool execution with timing and result status.
    
    Logs at INFO level for successful calls, WARNING for errors.
    Duration is stored in context as 'duration_ms'.
    
    Args:
        logger: Logger instance to use (defaults to toolcase.middleware)
        log_params: Whether to include params in log (default False for privacy)
    
    Example:
        >>> registry.use(LoggingMiddleware(log_params=True))
    """
    
    log: logging.Logger = field(default_factory=lambda: logger)
    log_params: bool = False
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        name = tool.metadata.name
        start = time.perf_counter()
        
        param_str = f" params={params.model_dump()}" if self.log_params else ""
        self.log.info(f"[{name}] Starting{param_str}")
        
        try:
            result = await next(tool, params, ctx)
            duration_ms = (time.perf_counter() - start) * 1000
            ctx["duration_ms"] = duration_ms
            
            # Check if result is an error
            is_error = result.startswith("**Tool Error")
            level = logging.WARNING if is_error else logging.INFO
            status = "ERROR" if is_error else "OK"
            
            self.log.log(level, f"[{name}] {status} ({duration_ms:.1f}ms)")
            return result
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            ctx["duration_ms"] = duration_ms
            self.log.exception(f"[{name}] EXCEPTION ({duration_ms:.1f}ms): {e}")
            raise


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Middleware
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class MetricsBackend(Protocol):
    """Protocol for metrics collection backends."""
    
    def increment(self, metric: str, value: int = 1, tags: dict[str, str] | None = None) -> None: ...
    def timing(self, metric: str, value_ms: float, tags: dict[str, str] | None = None) -> None: ...


@dataclass(slots=True)
class LogMetricsBackend:
    """Default metrics backend that logs to Python logger."""
    
    log: logging.Logger = field(default_factory=lambda: logger)
    
    def increment(self, metric: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
        tag_str = f" {tags}" if tags else ""
        self.log.debug(f"METRIC {metric}={value}{tag_str}")
    
    def timing(self, metric: str, value_ms: float, tags: dict[str, str] | None = None) -> None:
        tag_str = f" {tags}" if tags else ""
        self.log.debug(f"METRIC {metric}={value_ms:.2f}ms{tag_str}")


@dataclass(slots=True)
class MetricsMiddleware:
    """Collect execution metrics (counters, timing).
    
    Emits:
    - tool.calls: Counter per tool
    - tool.errors: Counter for error results
    - tool.duration_ms: Timing histogram
    
    Args:
        backend: MetricsBackend implementation (defaults to logging)
        prefix: Metric name prefix
    
    Example:
        >>> from datadog import statsd
        >>> registry.use(MetricsMiddleware(backend=statsd, prefix="myapp"))
    """
    
    backend: MetricsBackend = field(default_factory=LogMetricsBackend)
    prefix: str = "tool"
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        name = tool.metadata.name
        tags = {"tool": name, "category": tool.metadata.category}
        start = time.perf_counter()
        
        try:
            result = await next(tool, params, ctx)
            duration_ms = (time.perf_counter() - start) * 1000
            
            self.backend.increment(f"{self.prefix}.calls", tags=tags)
            self.backend.timing(f"{self.prefix}.duration_ms", duration_ms, tags=tags)
            
            if result.startswith("**Tool Error"):
                self.backend.increment(f"{self.prefix}.errors", tags=tags)
            
            return result
            
        except Exception:
            self.backend.increment(f"{self.prefix}.calls", tags=tags)
            self.backend.increment(f"{self.prefix}.exceptions", tags=tags)
            raise


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limit Middleware
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RateLimitMiddleware:
    """Token bucket rate limiter per tool.
    
    Limits concurrent and per-window executions. Raises RuntimeError
    when limit exceeded (recoverable - caller can retry).
    
    Args:
        max_calls: Maximum calls per window
        window_seconds: Time window in seconds
        per_tool: Apply limits per-tool (True) or globally (False)
    
    Example:
        >>> registry.use(RateLimitMiddleware(max_calls=10, window_seconds=60))
    """
    
    max_calls: int = 10
    window_seconds: float = 60.0
    per_tool: bool = True
    _timestamps: dict[str, deque[float]] = field(default_factory=dict, repr=False)
    
    def _get_bucket(self, key: str) -> deque[float]:
        if key not in self._timestamps:
            self._timestamps[key] = deque()
        return self._timestamps[key]
    
    def _check_limit(self, key: str) -> bool:
        """Check and update rate limit. Returns True if allowed."""
        now = time.time()
        bucket = self._get_bucket(key)
        
        # Evict expired timestamps
        cutoff = now - self.window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        
        # Check limit
        if len(bucket) >= self.max_calls:
            return False
        
        bucket.append(now)
        return True
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        key = tool.metadata.name if self.per_tool else "_global_"
        
        if not self._check_limit(key):
            raise RuntimeError(
                f"Rate limit exceeded for '{tool.metadata.name}': "
                f"{self.max_calls} calls per {self.window_seconds}s"
            )
        
        return await next(tool, params, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Retry Middleware
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class RetryMiddleware:
    """Retry failed executions with exponential backoff.
    
    Retries on exceptions only (not on error results). Uses exponential
    backoff with jitter to prevent thundering herd.
    
    Args:
        max_attempts: Total attempts including initial (minimum 1)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Backoff multiplier (default 2)
    
    Example:
        >>> registry.use(RetryMiddleware(max_attempts=3, base_delay=1.0))
    """
    
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        import random
        
        last_exc: Exception | None = None
        
        for attempt in range(self.max_attempts):
            try:
                result = await next(tool, params, ctx)
                ctx["retry_attempts"] = attempt + 1
                return result
                
            except Exception as e:
                last_exc = e
                
                if attempt + 1 >= self.max_attempts:
                    break
                
                # Exponential backoff with jitter
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay,
                )
                delay *= 0.5 + random.random()  # Add jitter
                
                logger.warning(
                    f"[{tool.metadata.name}] Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
        
        ctx["retry_attempts"] = self.max_attempts
        raise last_exc  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# Timeout Middleware
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class TimeoutMiddleware:
    """Enforce execution timeout.
    
    Wraps execution in asyncio.wait_for. Raises asyncio.TimeoutError
    if exceeded.
    
    Args:
        timeout_seconds: Maximum execution time
        per_tool_overrides: Dict of tool_name -> timeout for specific tools
    
    Example:
        >>> registry.use(TimeoutMiddleware(
        ...     timeout_seconds=30.0,
        ...     per_tool_overrides={"slow_tool": 120.0}
        ... ))
    """
    
    timeout_seconds: float = 30.0
    per_tool_overrides: dict[str, float] = field(default_factory=dict)
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        timeout = self.per_tool_overrides.get(
            tool.metadata.name,
            self.timeout_seconds,
        )
        return await asyncio.wait_for(next(tool, params, ctx), timeout=timeout)
