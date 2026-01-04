"""Request coalescing middleware (singleflight pattern).

Deduplicates concurrent identical requests by sharing results. When multiple
requests with the same parameters arrive while one is in-flight, all waiters
receive the same result instead of executing separately.

Value: Prevents duplicate expensive operations (LLM calls, API requests)
during concurrent access with identical parameters.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel

from toolcase.foundation.errors import CoalesceStatsDict
from toolcase.io.cache import ToolCache
from toolcase.runtime.middleware import Context, Next

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool


@dataclass(slots=True)
class CoalesceMiddleware:
    """Deduplicate concurrent identical requests via singleflight pattern.
    
    When multiple requests arrive for the same tool+params while one is executing,
    subsequent requests wait on the in-flight Future rather than executing again.
    All waiters receive the same result.
    
    Uses asyncio.Future keyed by cache key (tool_name + hashed params).
    Automatically cleans up after completion/failure.
    
    Args:
        per_tool: Coalesce per-tool (True) or globally (False)
        include_in_ctx: Add coalesce info to context (default: True)
    
    Example:
        >>> registry.use(CoalesceMiddleware())
        >>> # 10 concurrent identical requests → 1 execution, 10 results
        >>> results = await asyncio.gather(*[
        ...     registry.execute("expensive_tool", {"q": "same"})
        ...     for _ in range(10)
        ... ])
    
    Monitoring:
        >>> mw = CoalesceMiddleware()
        >>> registry.use(mw)
        >>> # Later...
        >>> print(mw.stats)  # {"total_requests": 100, "coalesced_requests": 85, ...}
    """
    
    per_tool: bool = True
    include_in_ctx: bool = True
    _flights: dict[str, asyncio.Future[str]] = field(default_factory=dict, repr=False)
    _total: int = field(default=0, repr=False)
    _coalesced: int = field(default=0, repr=False)
    
    def _make_key(self, tool: BaseTool[BaseModel], params: BaseModel) -> str:
        """Generate coalesce key from tool + params."""
        base = ToolCache.make_key(tool.metadata.name, params)
        return base if self.per_tool else f"_global_:{base.split(':', 1)[1]}"
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        key = self._make_key(tool, params)
        self._total += 1
        
        # Check for in-flight request
        if (flight := self._flights.get(key)) is not None:
            self._coalesced += 1
            if self.include_in_ctx:
                ctx["coalesced"] = True
            return await flight
        
        # First request - create Future and execute
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._flights[key] = future
        
        if self.include_in_ctx:
            ctx["coalesced"] = False
            ctx["coalesce_key"] = key
        
        try:
            result = await next(tool, params, ctx)
            future.set_result(result)
            return result
        except BaseException as e:
            future.set_exception(e)
            raise
        finally:
            self._flights.pop(key, None)
    
    @property
    def in_flight(self) -> int:
        """Number of currently in-flight requests."""
        return len(self._flights)
    
    @property
    def stats(self) -> CoalesceStatsDict:
        """Get coalescing statistics."""
        return {
            "total_requests": self._total,
            "coalesced_requests": self._coalesced,
            "in_flight": self.in_flight,
            "coalesce_ratio": self._coalesced / self._total if self._total else 0.0,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._total = self._coalesced = 0
