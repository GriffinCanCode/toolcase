"""Usage statistics for tool execution tracking.

Lightweight in-memory statistics collector for tool invocations.
Designed to integrate with registry execution without middleware overhead.

Example:
    >>> from toolcase.tools.core.stats import UsageStats, get_stats
    >>> 
    >>> stats = get_stats()
    >>> stats.record_call("search", duration_ms=45.2, success=True)
    >>> 
    >>> tool_stats = stats.get("search")
    >>> print(f"Success rate: {tool_stats.success_rate:.1%}")

Middleware Integration:
    >>> from toolcase.tools.core.stats import StatsMiddleware
    >>> 
    >>> registry.use(StatsMiddleware())
    >>> await registry.execute("search", {"query": "python"})
    >>> 
    >>> print(get_stats().get("search").calls)  # 1
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool, ToolMetadata
    from toolcase.runtime.middleware import Context, Next


@dataclass(slots=True)
class ToolStats:
    """Statistics for a single tool.
    
    Attributes:
        name: Tool name
        calls: Total invocation count
        successes: Successful invocations
        errors: Failed invocations
        total_duration_ms: Cumulative execution time
        last_call_at: Timestamp of last invocation
        last_error_at: Timestamp of last error (None if no errors)
    """
    name: str
    calls: int = 0
    successes: int = 0
    errors: int = 0
    total_duration_ms: float = 0.0
    last_call_at: float | None = None
    last_error_at: float | None = None
    
    @property
    def success_rate(self) -> float:
        """Success rate (0.0-1.0). Returns 1.0 if no calls."""
        return self.successes / self.calls if self.calls else 1.0
    
    @property
    def error_rate(self) -> float:
        """Error rate (0.0-1.0). Returns 0.0 if no calls."""
        return self.errors / self.calls if self.calls else 0.0
    
    @property
    def avg_duration_ms(self) -> float:
        """Average execution time in milliseconds."""
        return self.total_duration_ms / self.calls if self.calls else 0.0
    
    def record(self, duration_ms: float, success: bool) -> None:
        """Record a single invocation."""
        now = time.time()
        self.calls += 1
        self.total_duration_ms += duration_ms
        self.last_call_at = now
        if success:
            self.successes += 1
        else:
            self.errors += 1
            self.last_error_at = now
    
    def to_dict(self) -> dict[str, object]:
        """Convert to dict for serialization."""
        return {
            "name": self.name,
            "calls": self.calls,
            "successes": self.successes,
            "errors": self.errors,
            "success_rate": round(self.success_rate, 4),
            "avg_duration_ms": round(self.avg_duration_ms, 2),
            "last_call_at": self.last_call_at,
            "last_error_at": self.last_error_at,
        }


@dataclass(slots=True)
class UsageStats:
    """Aggregate usage statistics for all tools.
    
    Thread-safe for basic operations. For high-concurrency scenarios,
    consider using atomic counters or external metrics backends.
    
    Example:
        >>> stats = UsageStats()
        >>> stats.record_call("search", 45.2, success=True)
        >>> stats.record_call("search", 32.1, success=True)
        >>> stats.record_call("summarize", 120.5, success=False)
        >>> 
        >>> print(stats.get("search").avg_duration_ms)  # 38.65
        >>> print(stats.summary())
    """
    
    _tools: dict[str, ToolStats] = field(default_factory=dict)
    _started_at: float = field(default_factory=time.time)
    
    def record_call(self, tool_name: str, duration_ms: float, *, success: bool) -> None:
        """Record a tool invocation.
        
        Args:
            tool_name: Name of the tool that was invoked
            duration_ms: Execution duration in milliseconds
            success: Whether the invocation succeeded
        """
        if tool_name not in self._tools:
            self._tools[tool_name] = ToolStats(name=tool_name)
        self._tools[tool_name].record(duration_ms, success)
    
    def get(self, tool_name: str) -> ToolStats | None:
        """Get statistics for a specific tool."""
        return self._tools.get(tool_name)
    
    def get_or_create(self, tool_name: str) -> ToolStats:
        """Get or create statistics for a tool."""
        if tool_name not in self._tools:
            self._tools[tool_name] = ToolStats(name=tool_name)
        return self._tools[tool_name]
    
    def all(self) -> list[ToolStats]:
        """Get all tool statistics."""
        return list(self._tools.values())
    
    def top_by_calls(self, n: int = 10) -> list[ToolStats]:
        """Get top N most called tools."""
        return sorted(self._tools.values(), key=lambda s: -s.calls)[:n]
    
    def top_by_errors(self, n: int = 10) -> list[ToolStats]:
        """Get tools with most errors."""
        return sorted(self._tools.values(), key=lambda s: -s.errors)[:n]
    
    def top_by_duration(self, n: int = 10) -> list[ToolStats]:
        """Get slowest tools by average duration."""
        return sorted(self._tools.values(), key=lambda s: -s.avg_duration_ms)[:n]
    
    @property
    def total_calls(self) -> int:
        """Total invocations across all tools."""
        return sum(s.calls for s in self._tools.values())
    
    @property
    def total_errors(self) -> int:
        """Total errors across all tools."""
        return sum(s.errors for s in self._tools.values())
    
    @property
    def overall_success_rate(self) -> float:
        """Overall success rate (0.0-1.0)."""
        total = self.total_calls
        return (total - self.total_errors) / total if total else 1.0
    
    @property
    def uptime_seconds(self) -> float:
        """Seconds since statistics collection started."""
        return time.time() - self._started_at
    
    def summary(self) -> dict[str, object]:
        """Get summary statistics."""
        return {
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "overall_success_rate": round(self.overall_success_rate, 4),
            "tools_tracked": len(self._tools),
            "uptime_seconds": round(self.uptime_seconds, 1),
        }
    
    def reset(self) -> None:
        """Clear all statistics."""
        self._tools.clear()
        self._started_at = time.time()
    
    def to_dict(self) -> dict[str, object]:
        """Convert all statistics to dict."""
        return {
            "summary": self.summary(),
            "tools": {name: stats.to_dict() for name, stats in self._tools.items()},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Global Instance
# ─────────────────────────────────────────────────────────────────────────────

_stats: UsageStats | None = None


def get_stats() -> UsageStats:
    """Get global usage statistics instance."""
    global _stats
    return _stats if _stats else (_stats := UsageStats())


def set_stats(stats: UsageStats) -> None:
    """Set global usage statistics instance."""
    global _stats
    _stats = stats


def reset_stats() -> None:
    """Reset global usage statistics."""
    global _stats
    _stats and _stats.reset()
    _stats = None


# ─────────────────────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class StatsMiddleware:
    """Middleware that records usage statistics for all tool executions.
    
    Automatically tracks call counts, success rates, and latencies.
    Data is stored in the global UsageStats instance.
    
    Example:
        >>> from toolcase import get_registry
        >>> from toolcase.tools import StatsMiddleware, get_stats
        >>> 
        >>> registry = get_registry()
        >>> registry.use(StatsMiddleware())
        >>> 
        >>> await registry.execute("search", {"query": "python"})
        >>> print(get_stats().get("search").calls)  # 1
    """
    
    stats: UsageStats | None = None
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        """Record execution statistics."""
        target = self.stats or get_stats()
        name = tool.metadata.name
        start = time.perf_counter()
        
        try:
            result = await next(tool, params, ctx)
            success = not result.startswith("**Tool Error")
            target.record_call(name, (time.perf_counter() - start) * 1000, success=success)
            return result
        except Exception:
            target.record_call(name, (time.perf_counter() - start) * 1000, success=False)
            raise


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_stats(stats: UsageStats, *, top_n: int = 5, include_tools: bool = True) -> str:
    """Format statistics as markdown for display.
    
    Example:
        >>> print(format_stats(get_stats(), top_n=3))
    """
    summary = stats.summary()
    lines = [
        "**Usage Statistics**\n",
        f"- Total Calls: {summary['total_calls']}",
        f"- Total Errors: {summary['total_errors']}",
        f"- Success Rate: {summary['overall_success_rate']:.1%}",
        f"- Tools Tracked: {summary['tools_tracked']}",
        f"- Uptime: {summary['uptime_seconds']:.0f}s",
    ]
    
    if not include_tools or not stats.all():
        return "\n".join(lines)
    
    lines.append("\n**Most Active Tools:**")
    for s in stats.top_by_calls(top_n):
        rate_emoji = "✓" if s.success_rate >= 0.95 else "⚠" if s.success_rate >= 0.8 else "✗"
        lines.append(f"- `{s.name}`: {s.calls} calls ({s.success_rate:.0%} {rate_emoji}, {s.avg_duration_ms:.0f}ms avg)")
    
    errored = [s for s in stats.top_by_errors(top_n) if s.errors > 0]
    if errored:
        lines.append("\n**Tools with Errors:**")
        for s in errored:
            lines.append(f"- `{s.name}`: {s.errors} errors ({s.error_rate:.0%})")
    
    return "\n".join(lines)
