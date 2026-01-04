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
    """Structured logger with bound context.
    
    Binds key-value pairs that appear in every log entry.
    Immutable - bind() returns a new logger with merged context.
    
    Example:
        >>> log = BoundLogger(context={"service": "api"})
        >>> log.info("request received", path="/users")
        # => 2024-01-03 10:30:45 [info] request received service=api path=/users
        
        >>> user_log = log.bind(user_id=42)
        >>> user_log.info("authenticated")
        # => 2024-01-03 10:30:46 [info] authenticated service=api user_id=42
    """
    
    context: JsonDict = field(default_factory=dict)
    _renderer: LogRenderer | None = None
    _level: int = logging.DEBUG
    
    def bind(self, **kw: JsonValue) -> BoundLogger:
        """Create new logger with additional bound context."""
        return BoundLogger(
            context={**self.context, **kw},
            _renderer=self._renderer,
            _level=self._level,
        )
    
    def bind_tool(self, name: str, category: str, **kw: JsonValue) -> BoundLogger:
        """Bind tool execution context."""
        return self.bind(tool=name, category=category, **kw)
    
    def unbind(self, *keys: str) -> BoundLogger:
        """Create new logger without specified keys."""
        return BoundLogger(
            context={k: v for k, v in self.context.items() if k not in keys},
            _renderer=self._renderer,
            _level=self._level,
        )
    
    def new(self, **kw: JsonValue) -> BoundLogger:
        """Create new logger with only specified context (drops inherited)."""
        return BoundLogger(context=dict(kw), _renderer=self._renderer, _level=self._level)
    
    def _log(self, level: int, event: str, **kw: JsonValue) -> None:
        if level < self._level:
            return
        
        # Merge contexts: global -> bound -> call-site
        global_ctx = _log_context.get()
        merged = {**global_ctx, **self.context, **kw}
        
        # Add trace context if available
        merged.update(_get_trace_context())
        
        entry = LogEntry(
            timestamp=time.time(),
            level=_level_name(level),
            event=event,
            context=merged,
        )
        
        renderer = self._renderer or _get_renderer()
        renderer.render(entry)
    
    def debug(self, event: str, **kw: JsonValue) -> None:
        self._log(logging.DEBUG, event, **kw)
    
    def info(self, event: str, **kw: JsonValue) -> None:
        self._log(logging.INFO, event, **kw)
    
    def warning(self, event: str, **kw: JsonValue) -> None:
        self._log(logging.WARNING, event, **kw)
    
    def error(self, event: str, **kw: JsonValue) -> None:
        self._log(logging.ERROR, event, **kw)
    
    def exception(self, event: str, **kw: JsonValue) -> None:
        """Log error with exception info."""
        import traceback
        kw["exc_info"] = traceback.format_exc()  # type: ignore[assignment]
        self._log(logging.ERROR, event, **kw)
    
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
        self._ctx = ctx
        self._token: object | None = None
    
    def __enter__(self) -> None:
        current = _log_context.get()
        self._token = _log_context.set({**current, **self._ctx})
    
    def __exit__(self, *_: object) -> None:
        if self._token is not None:
            _log_context.reset(self._token)


# ─────────────────────────────────────────────────────────────────────────────
# Renderers
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class LogRenderer(Protocol):
    """Protocol for log output renderers."""
    
    def render(self, entry: LogEntry) -> None: ...


@dataclass(slots=True)
class ConsoleRenderer:
    """Human-readable colored console output.
    
    Format: timestamp [level] event key=value key2=value2
    
    Colors are auto-detected based on TTY, can be forced on/off.
    """
    
    output: TextIO = field(default_factory=lambda: sys.stderr)
    colors: bool | None = None  # None = auto-detect
    show_timestamp: bool = True
    
    def __post_init__(self) -> None:
        if self.colors is None:
            self.colors = hasattr(self.output, "isatty") and self.output.isatty()
    
    def render(self, entry: LogEntry) -> None:
        c = _COLORS if self.colors else _NO_COLORS
        level_color = _LEVEL_COLORS.get(entry.level, c["dim"])
        
        parts: list[str] = []
        
        if self.show_timestamp:
            parts.append(f"{c['dim']}{entry.ts_human}{c['reset']}")
        
        parts.append(f"{level_color}[{entry.level}]{c['reset']}")
        parts.append(f"{c['bold']}{entry.event}{c['reset']}")
        
        # Format context key=value
        for k, v in sorted(entry.context.items()):
            if k == "exc_info":
                continue  # Handle separately
            parts.append(f"{c['cyan']}{k}{c['reset']}={_format_value(v, c)}")
        
        print(" ".join(parts), file=self.output)
        
        # Print exception if present
        if "exc_info" in entry.context:
            print(f"{c['red']}{entry.context['exc_info']}{c['reset']}", file=self.output)


@dataclass(slots=True)
class JsonRenderer:
    """JSON Lines output for log aggregation.
    
    Each entry is a single JSON object on its own line.
    Suitable for shipping to Elasticsearch, Loki, Datadog, etc.
    """
    
    output: TextIO = field(default_factory=lambda: sys.stdout)
    
    def render(self, entry: LogEntry) -> None:
        import json
        data = {
            "timestamp": entry.ts_iso,
            "level": entry.level,
            "event": entry.event,
            **entry.context,
        }
        print(json.dumps(data, default=str), file=self.output)


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
    """Configure global structured logging.
    
    Args:
        format: Output format - "console" (human), "json" (machine), "none"
        level: Minimum log level - DEBUG, INFO, WARNING, ERROR
        output: Output stream (default: stderr for console, stdout for json)
        colors: Force colors on/off (None = auto-detect)
    
    Returns:
        Configured renderer instance
    
    Example:
        >>> # Development (human-readable, colored)
        >>> configure_logging(format="console", level="DEBUG")
        
        >>> # Production (JSON for aggregation)
        >>> configure_logging(format="json", level="INFO")
    """
    level_int = getattr(logging, level.upper(), logging.INFO)
    _default_level.set(level_int)
    
    renderer: LogRenderer
    if format == "console":
        renderer = ConsoleRenderer(output=output or sys.stderr, colors=colors)
    elif format == "json":
        renderer = JsonRenderer(output=output or sys.stdout)
    elif format == "none":
        renderer = NoOpRenderer()
    else:
        raise ValueError(f"Unknown format: {format}. Use 'console', 'json', or 'none'")
    
    _renderer.set(renderer)
    return renderer


def get_logger(name: str | None = None, **initial_context: JsonValue) -> BoundLogger:
    """Get a structured logger with optional initial context.
    
    Args:
        name: Logger name (added to context as 'logger')
        **initial_context: Initial bound key-value pairs
    
    Returns:
        BoundLogger instance with trace context auto-populated
    
    Example:
        >>> log = get_logger("my-service")
        >>> log.info("starting up", version="1.0.0")
        
        >>> # With initial context
        >>> log = get_logger("api", environment="prod")
    """
    ctx = dict(initial_context)
    if name:
        ctx["logger"] = name
    return BoundLogger(context=ctx, _level=_default_level.get())


def _get_renderer() -> LogRenderer:
    """Get configured renderer or create default."""
    renderer = _renderer.get()
    if renderer is None:
        renderer = ConsoleRenderer()
        _renderer.set(renderer)
    return renderer


def _get_trace_context() -> JsonDict:
    """Extract trace context if tracing is configured."""
    try:
        from .context import TraceContext
        ctx = TraceContext.get()
        if ctx:
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
    """Decorator to log function execution with timing.
    
    Args:
        log: Logger to use (creates one if not provided)
        level: Log level (debug, info, warning, error)
        event: Event message to log
    
    Example:
        >>> @timed(event="data fetched")
        ... def fetch_data(url: str) -> dict:
        ...     return requests.get(url).json()
        
        >>> fetch_data("https://api.example.com")
        # => 10:30:45.123 [info] data fetched function=fetch_data duration_ms=45.2
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _log = log or get_logger()
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                getattr(_log, level)(event, function=func.__name__, duration_ms=round(duration_ms, 2))
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                _log.error(
                    f"{event} failed",
                    function=func.__name__,
                    duration_ms=round(duration_ms, 2),
                    error=str(e),
                )
                raise
        
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _log = log or get_logger()
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)  # type: ignore[misc]
                duration_ms = (time.perf_counter() - start) * 1000
                getattr(_log, level)(event, function=func.__name__, duration_ms=round(duration_ms, 2))
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                _log.error(
                    f"{event} failed",
                    function=func.__name__,
                    duration_ms=round(duration_ms, 2),
                    error=str(e),
                )
                raise
        
        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper  # type: ignore[return-value]
    
    return decorator


class log_context:
    """Context manager for scoped logging context.
    
    Adds key-value pairs to all log entries within the scope.
    
    Example:
        >>> with log_context(request_id="abc123", user_id=42):
        ...     log.info("processing")  # includes request_id and user_id
        >>> log.info("done")  # no longer includes them
    """
    
    __slots__ = ("_ctx", "_token")
    
    def __init__(self, **kw: JsonValue) -> None:
        self._ctx: JsonDict = dict(kw)
        self._token: object | None = None
    
    def __enter__(self) -> log_context:
        current = _log_context.get()
        self._token = _log_context.set({**current, **self._ctx})
        return self
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._token is not None:
            _log_context.reset(self._token)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}

_NO_COLORS = {k: "" for k in _COLORS}

_LEVEL_COLORS = {
    "debug": _COLORS["dim"],
    "info": _COLORS["green"],
    "warning": _COLORS["yellow"],
    "error": _COLORS["red"],
}


def _level_name(level: int) -> str:
    """Convert logging level int to lowercase name."""
    return logging.getLevelName(level).lower()


def _format_value(v: object, c: dict[str, str]) -> str:
    """Format a value for console output."""
    if isinstance(v, str):
        return f'{c["yellow"]}"{v}"{c["reset"]}'
    if isinstance(v, bool):
        return f'{c["blue"]}{str(v).lower()}{c["reset"]}'
    if isinstance(v, (int, float)):
        return f'{c["blue"]}{v}{c["reset"]}'
    if isinstance(v, dict):
        return f'{c["dim"]}{{{len(v)} items}}{c["reset"]}'
    if isinstance(v, (list, tuple)):
        return f'{c["dim"]}[{len(v)} items]{c["reset"]}'
    return f'{c["white"]}{v!r}{c["reset"]}'
