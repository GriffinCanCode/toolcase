"""Structured logging for tool execution with context propagation.

Provides context-aware structured logging that integrates with tracing:
- Automatic trace ID correlation
- Tool context binding (name, category, params)
- Human-readable dev output, JSON for production
- Timing decorators and context managers

Uses structlog for maximum flexibility and readability.

Quick Start:
    >>> from toolcase.runtime.observability import get_logger, configure_logging
    >>> 
    >>> # Configure (once at startup)
    >>> configure_logging(format="console")  # or "json" for production
    >>> 
    >>> # Get logger with automatic trace context
    >>> log = get_logger("my-service")
    >>> log.info("processing request", user_id=123)
    
    >>> # Bind tool context
    >>> log = log.bind_tool("web_search", "search")
    >>> log.info("executing", query="python tutorial")

Integration with Middleware:
    >>> from toolcase.runtime.observability import StructuredLoggingMiddleware
    >>> registry.use(StructuredLoggingMiddleware())
"""

from __future__ import annotations

import logging
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import wraps
from typing import TYPE_CHECKING, Callable, ParamSpec, Protocol, TextIO, TypeVar, runtime_checkable

from toolcase.foundation.errors import JsonDict, JsonValue

if TYPE_CHECKING:
    from types import TracebackType

P = ParamSpec("P")
T = TypeVar("T")

# Context var for bound context (persists across async calls)
_log_context: ContextVar[JsonDict] = ContextVar("log_context", default={})


# ─────────────────────────────────────────────────────────────────────────────
# Core Logger Protocol & Implementation
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class StructuredLogger(Protocol):
    """Protocol for structured loggers."""
    
    def debug(self, event: str, **kw: JsonValue) -> None: ...
    def info(self, event: str, **kw: JsonValue) -> None: ...
    def warning(self, event: str, **kw: JsonValue) -> None: ...
    def error(self, event: str, **kw: JsonValue) -> None: ...
    def exception(self, event: str, **kw: JsonValue) -> None: ...
    def bind(self, **kw: JsonValue) -> StructuredLogger: ...


@dataclass(slots=True)
class BoundLogger:
    """Structured logger with bound context. Immutable - bind() returns a new logger with merged context.
    
    Example:
        >>> log = BoundLogger(context={"service": "api"})
        >>> log.info("request received", path="/users")
        # => 2024-01-03 10:30:45 [info] request received service=api path=/users
    """
    
    context: JsonDict = field(default_factory=dict)
    _renderer: LogRenderer | None = None
    _level: int = logging.DEBUG
    
    def bind(self, **kw: JsonValue) -> BoundLogger:
        """Create new logger with additional bound context."""
        return BoundLogger(context={**self.context, **kw}, _renderer=self._renderer, _level=self._level)
    
    def bind_tool(self, name: str, category: str, **kw: JsonValue) -> BoundLogger:
        """Bind tool execution context."""
        return self.bind(tool=name, category=category, **kw)
    
    def unbind(self, *keys: str) -> BoundLogger:
        """Create new logger without specified keys."""
        return BoundLogger(context={k: v for k, v in self.context.items() if k not in keys},
                           _renderer=self._renderer, _level=self._level)
    
    def new(self, **kw: JsonValue) -> BoundLogger:
        """Create new logger with only specified context (drops inherited)."""
        return BoundLogger(context=dict(kw), _renderer=self._renderer, _level=self._level)
    
    def _log(self, level: int, event: str, **kw: JsonValue) -> None:
        if level < self._level:
            return
        # Merge contexts: global -> bound -> call-site, then add trace context
        merged = {**_log_context.get(), **self.context, **kw, **_get_trace_context()}
        (self._renderer or _get_renderer()).render(LogEntry(time.time(), _level_name(level), event, merged))
    
    def debug(self, event: str, **kw: JsonValue) -> None: self._log(logging.DEBUG, event, **kw)
    def info(self, event: str, **kw: JsonValue) -> None: self._log(logging.INFO, event, **kw)
    def warning(self, event: str, **kw: JsonValue) -> None: self._log(logging.WARNING, event, **kw)
    def error(self, event: str, **kw: JsonValue) -> None: self._log(logging.ERROR, event, **kw)
    
    def exception(self, event: str, **kw: JsonValue) -> None:
        """Log error with exception info."""
        import traceback
        self._log(logging.ERROR, event, exc_info=traceback.format_exc(), **kw)  # type: ignore[arg-type]
    
    # Context manager for scoped context
    def scope(self, **kw: JsonValue) -> LogScope:
        """Create a scoped logging context.
        
        Example:
            >>> with log.scope(request_id="abc123"):
            ...     log.info("processing")  # includes request_id
            >>> log.info("done")  # no request_id
        """
        return LogScope(kw)


@dataclass(slots=True)
class LogEntry:
    """Immutable log entry with all context."""
    
    timestamp: float
    level: str
    event: str
    context: JsonDict
    
    @property
    def ts_iso(self) -> str:
        """ISO formatted timestamp."""
        return datetime.fromtimestamp(self.timestamp, tz=UTC).isoformat()
    
    @property
    def ts_human(self) -> str:
        """Human-readable timestamp (HH:MM:SS.mmm)."""
        return datetime.fromtimestamp(self.timestamp, tz=UTC).strftime("%H:%M:%S.%f")[:-3]


class LogScope:
    """Context manager for scoped log context."""
    
    __slots__ = ("_ctx", "_token")
    
    def __init__(self, ctx: JsonDict) -> None:
        self._ctx, self._token = ctx, None
    
    def __enter__(self) -> None:
        self._token = _log_context.set({**_log_context.get(), **self._ctx})
    
    def __exit__(self, *_: object) -> None:
        self._token and _log_context.reset(self._token)  # type: ignore[func-returns-value]


# ─────────────────────────────────────────────────────────────────────────────
# Renderers
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class LogRenderer(Protocol):
    """Protocol for log output renderers."""
    
    def render(self, entry: LogEntry) -> None: ...


@dataclass(slots=True)
class ConsoleRenderer:
    """Human-readable colored console output. Format: timestamp [level] event key=value ..."""
    
    output: TextIO = field(default_factory=lambda: sys.stderr)
    colors: bool | None = None  # None = auto-detect
    show_timestamp: bool = True
    
    def __post_init__(self) -> None:
        if self.colors is None:
            self.colors = getattr(self.output, "isatty", lambda: False)()
    
    def render(self, entry: LogEntry) -> None:
        c = _COLORS if self.colors else _NO_COLORS
        parts = ([f"{c['dim']}{entry.ts_human}{c['reset']}"] if self.show_timestamp else [])
        parts += [f"{_LEVEL_COLORS.get(entry.level, c['dim'])}[{entry.level}]{c['reset']}",
                  f"{c['bold']}{entry.event}{c['reset']}"]
        parts += [f"{c['cyan']}{k}{c['reset']}={_format_value(v, c)}"
                  for k, v in sorted(entry.context.items()) if k != "exc_info"]
        print(" ".join(parts), file=self.output)
        if "exc_info" in entry.context:
            print(f"{c['red']}{entry.context['exc_info']}{c['reset']}", file=self.output)


@dataclass(slots=True)
class JsonRenderer:
    """JSON Lines output for log aggregation (Elasticsearch, Loki, Datadog, etc.)."""
    
    output: TextIO = field(default_factory=lambda: sys.stdout)
    
    def render(self, entry: LogEntry) -> None:
        import json
        print(json.dumps({"timestamp": entry.ts_iso, "level": entry.level, "event": entry.event,
                          **entry.context}, default=str), file=self.output)


@dataclass(slots=True)
class NoOpRenderer:
    """Silent renderer for testing."""
    
    def render(self, entry: LogEntry) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Global Configuration
# ─────────────────────────────────────────────────────────────────────────────


_renderer: ContextVar[LogRenderer | None] = ContextVar("log_renderer", default=None)
_default_level: ContextVar[int] = ContextVar("log_level", default=logging.INFO)


def configure_logging(
    format: str = "console",  # noqa: A002 - shadows builtin but matches stdlib
    level: str = "INFO",
    *,
    output: TextIO | None = None,
    colors: bool | None = None,
) -> LogRenderer:
    """Configure global structured logging. Format: "console" (human), "json" (machine), "none"."""
    _default_level.set(getattr(logging, level.upper(), logging.INFO))
    match format:
        case "console": renderer: LogRenderer = ConsoleRenderer(output=output or sys.stderr, colors=colors)
        case "json": renderer = JsonRenderer(output=output or sys.stdout)
        case "none": renderer = NoOpRenderer()
        case _: raise ValueError(f"Unknown format: {format}. Use 'console', 'json', or 'none'")
    _renderer.set(renderer)
    return renderer


def get_logger(name: str | None = None, **initial_context: JsonValue) -> BoundLogger:
    """Get a structured logger with optional initial context. Name is added to context as 'logger'."""
    ctx = {**initial_context, **({"logger": name} if name else {})}
    return BoundLogger(context=ctx, _level=_default_level.get())


def _get_renderer() -> LogRenderer:
    """Get configured renderer or create default."""
    if (renderer := _renderer.get()) is None:
        _renderer.set(renderer := ConsoleRenderer())
    return renderer


def _get_trace_context() -> JsonDict:
    """Extract trace context if tracing is configured."""
    try:
        from .context import TraceContext
        if ctx := TraceContext.get():
            return {"trace_id": ctx.span_context.trace_id, "span_id": ctx.span_context.span_id}
    except ImportError:
        pass
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Decorators & Utilities
# ─────────────────────────────────────────────────────────────────────────────


def timed(
    log: BoundLogger | None = None,
    *,
    level: str = "info",
    event: str = "operation completed",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to log function execution with timing."""
    import asyncio
    
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        def _finish(_log: BoundLogger, start: float, err: Exception | None = None) -> float:
            dur = round((time.perf_counter() - start) * 1000, 2)
            if err:
                _log.error(f"{event} failed", function=func.__name__, duration_ms=dur, error=str(err))
            else:
                getattr(_log, level)(event, function=func.__name__, duration_ms=dur)
            return dur
        
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _log, start = log or get_logger(), time.perf_counter()
            try:
                result = func(*args, **kwargs)
                _finish(_log, start)
                return result
            except Exception as e:
                _finish(_log, start, e)
                raise
        
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _log, start = log or get_logger(), time.perf_counter()
            try:
                result = await func(*args, **kwargs)  # type: ignore[misc]
                _finish(_log, start)
                return result
            except Exception as e:
                _finish(_log, start, e)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper  # type: ignore[return-value]
    
    return decorator


class log_context:
    """Context manager for scoped logging context. Adds key-value pairs to all log entries within the scope."""
    
    __slots__ = ("_ctx", "_token")
    
    def __init__(self, **kw: JsonValue) -> None:
        self._ctx: JsonDict = dict(kw)
        self._token: object | None = None
    
    def __enter__(self) -> log_context:
        self._token = _log_context.set({**_log_context.get(), **self._ctx})
        return self
    
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None,
                 exc_tb: TracebackType | None) -> None:
        self._token and _log_context.reset(self._token)  # type: ignore[func-returns-value]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


_COLORS = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m", "red": "\033[31m",
           "green": "\033[32m", "yellow": "\033[33m", "blue": "\033[34m", "cyan": "\033[36m", "white": "\033[37m"}
_NO_COLORS = {k: "" for k in _COLORS}
_LEVEL_COLORS = {"debug": _COLORS["dim"], "info": _COLORS["green"], "warning": _COLORS["yellow"], "error": _COLORS["red"]}


def _level_name(level: int) -> str:
    """Convert logging level int to lowercase name."""
    return logging.getLevelName(level).lower()


def _format_value(v: object, c: dict[str, str]) -> str:
    """Format a value for console output."""
    match v:
        case str(): return f'{c["yellow"]}"{v}"{c["reset"]}'
        case bool(): return f'{c["blue"]}{str(v).lower()}{c["reset"]}'
        case int() | float(): return f'{c["blue"]}{v}{c["reset"]}'
        case dict(): return f'{c["dim"]}{{{len(v)} items}}{c["reset"]}'
        case list() | tuple(): return f'{c["dim"]}[{len(v)} items]{c["reset"]}'
        case _: return f'{c["white"]}{v!r}{c["reset"]}'
