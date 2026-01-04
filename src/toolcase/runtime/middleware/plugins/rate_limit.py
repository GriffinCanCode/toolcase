"""Rate limiting middleware for tool execution."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel

from toolcase.foundation.errors import ErrorCode, ErrorTrace, ToolError, ToolException
from toolcase.runtime.middleware import Context, Next

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool


@dataclass
class RateLimitMiddleware:
    """Token bucket rate limiter per tool.
    
    Limits concurrent and per-window executions. Raises ToolException
    with RATE_LIMITED code when limit exceeded (recoverable).
    Stores ErrorTrace in context for observability integration.
    
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
    
    def _check_limit(self, key: str) -> tuple[bool, int]:
        """Check and update rate limit. Returns (allowed, current_count)."""
        now = time.time()
        bucket = self._get_bucket(key)
        
        # Evict expired timestamps
        cutoff = now - self.window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        
        current = len(bucket)
        if current >= self.max_calls:
            return False, current
        
        bucket.append(now)
        return True, current + 1
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        key = tool.metadata.name if self.per_tool else "_global_"
        allowed, count = self._check_limit(key)
        ctx["rate_limit_count"] = count
        ctx["rate_limit_max"] = self.max_calls
        
        if not allowed:
            trace = ErrorTrace(
                message=f"Rate limit exceeded: {self.max_calls} calls per {self.window_seconds}s",
                error_code=ErrorCode.RATE_LIMITED.value,
                recoverable=True,
            ).with_operation(
                "middleware:rate_limit",
                tool=tool.metadata.name,
                limit=self.max_calls,
                window=self.window_seconds,
                current=count,
            )
            ctx["error_trace"] = trace
            raise ToolException(ToolError.create(
                tool.metadata.name,
                trace.message,
                ErrorCode.RATE_LIMITED,
                recoverable=True,
            ))
        
        return await next(tool, params, ctx)
