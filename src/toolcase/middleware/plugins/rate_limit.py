"""Rate limiting middleware for tool execution."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..middleware import Context, Next

if TYPE_CHECKING:
    from ...core import BaseTool


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
