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
import time
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Callable, ParamSpec, Protocol, TypeVar, runtime_checkable

import structlog
from structlog.typing import FilteringBoundLogger

from toolcase.foundation.errors import JsonDict, JsonValue

if TYPE_CHECKING:
    from types import TracebackType

P = ParamSpec("P")
T = TypeVar("T")

# Context var for scoped context (used by log_context context manager)
_log_context: ContextVar[JsonDict] = ContextVar("log_context", default={})
_configured: ContextVar[bool] = ContextVar("log_configured", default=False)


# ─────────────────────────────────────────────────────────────────────────────
# Structlog Processors
# ─────────────────────────────────────────────────────────────────────────────


def _add_trace_context(
    logger: str, method: str, event_dict: dict[str, object]
) -> dict[str, object]:
    """Processor to inject trace context from active span."""
    try:
        from ..tracing import TraceContext, Tracer
        if ctx := TraceContext.get():
            sc = ctx.span_context
            event_dict["trace_id"] = sc.trace_id
            event_dict["span_id"] = sc.span_id
            if sc.parent_id:
                event_dict["parent_span_id"] = sc.parent_id
            if (tracer := Tracer.get_global()) and tracer.service_name:
                event_dict["service.name"] = tracer.service_name
    except ImportError:
        pass
    return event_dict


def _merge_scoped_context(
    logger: str, method: str, event_dict: dict[str, object]
) -> dict[str, object]:
    """Processor to merge log_context ContextVar into event."""
    if ctx := _log_context.get():
        event_dict = {**ctx, **event_dict}
    return event_dict


def _record_span_event(
    logger: str, method: str, event_dict: dict[str, object]
) -> dict[str, object]:
    """Processor to record log entries as span events (when enabled)."""
    if event_dict.pop("_record_to_span", False):
        try:
            from ..tracing import TraceContext
            from ..tracing.tracer import Tracer
            if (ctx := TraceContext.get()) and (tracer := Tracer.get_global()) and tracer.enabled:
                if tracer._spans:
                    attrs = {k: v for k, v in event_dict.items() 
                             if k not in {"event", "level", "timestamp", "trace_id", "span_id", 
                                         "parent_span_id", "service.name"}}
                    tracer._spans[-1].add_event(
                        f"log.{event_dict.get('level', 'info')}", 
                        {"message": event_dict.get("event", ""), **attrs}
                    )
        except (ImportError, IndexError):
            pass
    return event_dict


# ─────────────────────────────────────────────────────────────────────────────
# Core Logger Protocol & Wrapper
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


class BoundLogger:
    """Structured logger wrapping structlog. Immutable - bind() returns new logger.
    
    Example:
        >>> log = get_logger("my-service")
        >>> log.info("request received", path="/users")
        # => 2024-01-03 10:30:45 [info] request received service=api path=/users
    """
    
    __slots__ = ("_logger", "_record_to_span")
    
    def __init__(self, logger: FilteringBoundLogger, *, record_to_span: bool = False) -> None:
        self._logger = logger
        self._record_to_span = record_to_span
    
    def bind(self, **kw: JsonValue) -> BoundLogger:
        """Create new logger with additional bound context."""
        return BoundLogger(self._logger.bind(**kw), record_to_span=self._record_to_span)
    
    def bind_tool(self, name: str, category: str, **kw: JsonValue) -> BoundLogger:
        """Bind tool execution context."""
        return self.bind(tool=name, category=category, **kw)
    
    def unbind(self, *keys: str) -> BoundLogger:
        """Create new logger without specified keys."""
        return BoundLogger(self._logger.unbind(*keys), record_to_span=self._record_to_span)
    
    def new(self, **kw: JsonValue) -> BoundLogger:
        """Create new logger with only specified context (drops inherited)."""
        return BoundLogger(self._logger.new(**kw), record_to_span=self._record_to_span)
    
    def _log(self, method: str, event: str, **kw: JsonValue) -> None:
        if self._record_to_span:
            kw["_record_to_span"] = True  # type: ignore[assignment]
        getattr(self._logger, method)(event, **kw)
    
    def debug(self, event: str, **kw: JsonValue) -> None: self._log("debug", event, **kw)
    def info(self, event: str, **kw: JsonValue) -> None: self._log("info", event, **kw)
    def warning(self, event: str, **kw: JsonValue) -> None: self._log("warning", event, **kw)
    def error(self, event: str, **kw: JsonValue) -> None: self._log("error", event, **kw)
    def exception(self, event: str, **kw: JsonValue) -> None: self._log("exception", event, **kw)
    
    def scope(self, **kw: JsonValue) -> LogScope:
        """Create a scoped logging context.
        
        Example:
            >>> with log.scope(request_id="abc123"):
            ...     log.info("processing")  # includes request_id
            >>> log.info("done")  # no request_id
        """
        return LogScope(kw)


class TracedLogger(BoundLogger):
    """Logger that records logs as span events for bidirectional correlation.
    
    Example:
        >>> from toolcase.runtime.observability import get_tracer, span_logger
        >>> tracer = get_tracer()
        >>> with tracer.span("process") as span:
        ...     log = span_logger()  # Bound to current span
        ...     log.info("processing item", item_id=123)
        ...     # Log appears both in logs AND as span event
    """
    
    def __init__(self, logger: FilteringBoundLogger, *, record_to_span: bool = True) -> None:
        super().__init__(logger, record_to_span=record_to_span)
    
    def bind(self, **kw: JsonValue) -> TracedLogger:
        """Create new traced logger with additional bound context."""
        return TracedLogger(self._logger.bind(**kw), record_to_span=self._record_to_span)
    
    def with_span_events(self, enabled: bool = True) -> TracedLogger:
        """Return logger that records logs as span events."""
        return TracedLogger(self._logger, record_to_span=enabled)


# ─────────────────────────────────────────────────────────────────────────────
# Scoped Context
# ─────────────────────────────────────────────────────────────────────────────


class LogScope:
    """Context manager for scoped log context."""
    
    __slots__ = ("_ctx", "_token")
    
    def __init__(self, ctx: dict[str, JsonValue]) -> None:
        self._ctx = ctx
        self._token: object | None = None
    
    def __enter__(self) -> None:
        self._token = _log_context.set({**_log_context.get(), **self._ctx})
    
    def __exit__(self, *_: object) -> None:
        if self._token:
            _log_context.reset(self._token)


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
        if self._token:
            _log_context.reset(self._token)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


def configure_logging(
    format: str = "console",  # noqa: A002
    level: str = "INFO",
    *,
    colors: bool | None = None,
) -> None:
    """Configure global structured logging. Format: "console" (human), "json" (machine), "none"."""
    
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _merge_scoped_context,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_trace_context,
        _record_span_event,
    ]
    
    if format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    elif format == "none":
        renderer = structlog.dev.ConsoleRenderer(colors=False)  # Will be filtered by level anyway
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=colors)
    
    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured.set(True)


def _ensure_configured() -> None:
    """Ensure structlog is configured with defaults if not already done."""
    if not _configured.get():
        configure_logging()


def get_logger(name: str | None = None, **initial_context: JsonValue) -> BoundLogger:
    """Get a structured logger with optional initial context. Name is added to context as 'logger'."""
    _ensure_configured()
    ctx = {**initial_context, **({"logger": name} if name else {})}
    return BoundLogger(structlog.get_logger(**ctx))


def span_logger(name: str | None = None, record_to_span: bool = True, **initial_context: JsonValue) -> TracedLogger:
    """Get a logger bound to the current span context.
    
    Like get_logger(), but returns a TracedLogger that:
    - Is always correlated with the current trace (trace_id, span_id in every log)
    - Optionally records log entries as span events (record_to_span=True)
    
    Args:
        name: Logger name (added as 'logger' in context)
        record_to_span: If True, also record logs as span events
        **initial_context: Additional context to bind
    """
    _ensure_configured()
    ctx = {**initial_context, **({"logger": name} if name else {})}
    return TracedLogger(structlog.get_logger(**ctx), record_to_span=record_to_span)


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


# ─────────────────────────────────────────────────────────────────────────────
# Unified Observability Configuration
# ─────────────────────────────────────────────────────────────────────────────


def configure_observability(
    service_name: str = "toolcase",
    *,
    log_format: str = "console",
    log_level: str = "INFO",
    trace_exporter: str = "console",
    trace_endpoint: str | None = None,
    trace_api_key: str | None = None,
    trace_env: str = "",
    async_export: bool = False,
    sample_rate: float | None = None,
    colors: bool | None = None,
    verbose: bool = False,
) -> tuple[None, object]:
    """Configure both logging and tracing with automatic correlation.
    
    This is the recommended way to set up observability. It ensures:
    - Logs automatically include trace_id, span_id, parent_span_id
    - Service name is consistent across logs and traces
    - Proper correlation for downstream observability backends
    
    Args:
        service_name: Identifier for this service (used in both logs and traces)
        log_format: "console" (human-readable), "json" (machine-parseable), or "none"
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
        trace_exporter: "console", "json", "otlp", "otlp_http", "datadog", "honeycomb", "zipkin", "none"
        trace_endpoint: Collector endpoint (for otlp, zipkin)
        trace_api_key: API key (for datadog, honeycomb)
        trace_env: Environment name (for datadog)
        async_export: Use background batching for trace export
        sample_rate: Trace sampling rate (0.0-1.0)
        colors: Force color output (None = auto-detect)
        verbose: Show detailed attributes in console trace output
    
    Returns:
        Tuple of (None, Tracer) - None replaces renderer since structlog manages its own output
    
    Example:
        >>> from toolcase.runtime.observability import configure_observability, span_logger
        >>> 
        >>> # Development setup
        >>> configure_observability(service_name="my-agent")
        >>> 
        >>> # Production with OTLP
        >>> configure_observability(
        ...     service_name="my-agent",
        ...     log_format="json",
        ...     trace_exporter="otlp",
        ...     trace_endpoint="http://otel-collector:4317",
        ...     async_export=True,
        ... )
        >>> 
        >>> # Now logs automatically have trace context
        >>> log = span_logger("my-module")
        >>> with get_tracer().span("operation"):
        ...     log.info("processing")  # Includes trace_id, span_id, service.name
    """
    from ..tracing import configure_tracing
    
    # Configure logging first
    configure_logging(format=log_format, level=log_level, colors=colors)
    
    # Configure tracing with matching service name
    tracer = configure_tracing(
        service_name=service_name,
        exporter=trace_exporter,
        endpoint=trace_endpoint,
        verbose=verbose,
        async_export=async_export,
        sample_rate=sample_rate,
        api_key=trace_api_key,
        env=trace_env,
    )
    
    return None, tracer
