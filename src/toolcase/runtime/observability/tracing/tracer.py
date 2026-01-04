"""Tracer for creating and managing spans.

Provides the main API for instrumenting code with traces.
Uses context managers for automatic span lifecycle management.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Callable, ParamSpec, TypeVar

from toolcase.foundation.errors import JsonDict, JsonMapping

from .context import SpanContext, TraceContext
from .span import Span, SpanKind, SpanStatus
from ..exporters import ConsoleExporter, Exporter, NoOpExporter

if TYPE_CHECKING:
    from types import TracebackType

P = ParamSpec("P")
T = TypeVar("T")

# Global tracer instance
_tracer: ContextVar[Tracer | None] = ContextVar("tracer", default=None)


@dataclass(slots=True)
class Tracer:
    """Creates and manages spans for distributed tracing.
    
    The Tracer is the main entry point for instrumentation. It creates spans, manages context propagation, and exports completed spans.
    
    Usage:
        >>> tracer = Tracer(service_name="my-agent", exporter=ConsoleExporter())
        >>> tracer.configure_global()
        >>> 
        >>> with tracer.span("search_tool") as span:
        ...     span.set_attribute("query", "python")
        ...     # do work
    
    Args:
        service_name: Name identifying this service in traces
        exporter: Where to send completed spans
        enabled: Whether tracing is active (False = no-op)
    """
    
    service_name: str = "toolcase"
    exporter: Exporter = field(default_factory=ConsoleExporter)
    enabled: bool = True
    _spans: list[Span] = field(default_factory=list)
    
    def configure_global(self) -> None:
        """Set this tracer as the global instance."""
        _tracer.set(self)
    
    @classmethod
    def get_global(cls) -> Tracer | None:
        """Get the global tracer instance."""
        return _tracer.get()
    
    @classmethod
    def current(cls) -> Tracer:
        """Get global tracer or create disabled one."""
        return _tracer.get() or cls(enabled=False, exporter=NoOpExporter())
    
    def span(self, name: str, kind: SpanKind = SpanKind.INTERNAL, attributes: JsonMapping | None = None) -> SpanContextManager:
        """Create a span context manager.
        
        Example:
            >>> with tracer.span("operation", kind=SpanKind.EXTERNAL) as span:
            ...     span.set_attribute("url", "https://api.example.com")
            ...     result = fetch_data()
            ...     span.set_result_preview(result)
        """
        return SpanContextManager(self, name, kind, dict(attributes) if attributes else {})
    
    def start_span(self, name: str, kind: SpanKind = SpanKind.INTERNAL, attributes: JsonMapping | None = None) -> Span:
        """Start a span manually (caller must end it). Prefer `span()` context manager for automatic lifecycle."""
        if not self.enabled:
            return Span(name=name, context=SpanContext.new(), kind=kind)
        
        ctx = TraceContext.current()
        span_ctx = ctx.span_context.child()
        ctx.push_span(span_ctx)
        return Span(name=name, context=span_ctx, kind=kind,
                    attributes={"service.name": self.service_name, **(attributes or {})})
    
    def end_span(self, span: Span, status: SpanStatus = SpanStatus.OK, error: str | None = None) -> None:
        """End a manually started span and export it."""
        if not self.enabled:
            return
        span.end(status=status, error=error)
        self.exporter.export([span])
        if ctx := TraceContext.get():
            ctx.pop_span()
    
    def shutdown(self) -> None:
        """Shutdown tracer and flush exports."""
        self.exporter.shutdown()


@dataclass(slots=True)
class SpanContextManager:
    """Context manager for span lifecycle. Automatically ends span and sets status on exit."""
    
    tracer: Tracer
    name: str
    kind: SpanKind
    attributes: JsonDict
    _span: Span | None = None
    
    def __enter__(self) -> Span:
        self._span = self.tracer.start_span(self.name, self.kind, self.attributes)
        return self._span
    
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        if not self._span:
            return
        status = SpanStatus.ERROR if exc_val else (self._span.status if self._span.status != SpanStatus.UNSET else SpanStatus.OK)
        self.tracer.end_span(self._span, status, str(exc_val) if exc_val else None)
    
    async def __aenter__(self) -> Span:
        return self.__enter__()
    
    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


# ─────────────────────────────────────────────────────────────────────────────
# Decorator API
# ─────────────────────────────────────────────────────────────────────────────


def traced(
    name: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    capture_args: bool = True,
    capture_result: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to automatically trace a function.
    
    Creates a span for each function invocation with:
    - Function arguments as attributes (if capture_args)
    - Return value preview (if capture_result)
    - Exception handling with error status
    
    Example:
        >>> @traced(kind=SpanKind.EXTERNAL)
        ... def fetch_data(url: str) -> dict:
        ...     return requests.get(url).json()
    """
    import asyncio
    
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = name or func.__name__
        build_attrs: Callable[[dict[str, object]], JsonDict] = lambda kw: {
            "function": func.__name__, **({"args": {k: _safe_repr(v) for k, v in kw.items()}} if capture_args and kw else {})
        }
        
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with Tracer.current().span(span_name, kind, build_attrs(kwargs)) as span:
                result = func(*args, **kwargs)
                if capture_result and isinstance(result, str):
                    span.set_result_preview(result)
                return result
        
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with Tracer.current().span(span_name, kind, build_attrs(kwargs)) as span:
                result = await func(*args, **kwargs)
                if capture_result and isinstance(result, str):
                    span.set_result_preview(result)
                return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper  # type: ignore[return-value]
    
    return decorator


def _safe_repr(value: object, max_len: int = 100) -> str:
    """Safe string representation with length limit."""
    s = repr(value)
    return f"{s[:max_len]}..." if len(s) > max_len else s


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


def get_tracer() -> Tracer:
    """Get the global tracer (creates disabled one if not configured)."""
    return Tracer.current()


def configure_tracing(
    service_name: str = "toolcase",
    exporter: str | Exporter = "console",
    *,
    endpoint: str | None = None,
    verbose: bool = False,
    async_export: bool = False,
    sample_rate: float | None = None,
    api_key: str | None = None,
    env: str = "",
) -> Tracer:
    """Configure global tracing with common defaults.
    
    Args:
        service_name: Name for this service in traces
        exporter: "console", "json", "otlp", "otlp_http", "datadog", "honeycomb", 
                  "zipkin", "none", or Exporter instance
        endpoint: Collector endpoint (for otlp, zipkin)
        verbose: Show detailed attributes in console output
        async_export: Wrap in AsyncBatchExporter for background export
        sample_rate: If set, wrap in SampledExporter (0.0-1.0)
        api_key: API key (for datadog, honeycomb)
        env: Environment name (for datadog)
    
    Returns:
        Configured global Tracer instance
    
    Example:
        >>> from toolcase.observability import configure_tracing
        >>> configure_tracing(service_name="my-agent", exporter="console")
        >>> 
        >>> # Production with OTLP
        >>> configure_tracing(
        ...     service_name="my-agent",
        ...     exporter="otlp",
        ...     endpoint="http://otel-collector:4317",
        ... )
        >>> 
        >>> # Datadog with async export and 10% sampling
        >>> configure_tracing(
        ...     service_name="my-agent",
        ...     exporter="datadog",
        ...     api_key="dd-xxx",
        ...     env="production",
        ...     async_export=True,
        ...     sample_rate=0.1,
        ... )
    """
    from ..exporters import (
        AsyncBatchExporter,
        ConsoleExporter,
        DatadogExporter,
        HoneycombExporter,
        JsonExporter,
        NoOpExporter,
        SampledExporter,
        ZipkinExporter,
        create_otlp_exporter,
    )
    
    if isinstance(exporter, str):
        exporters: dict[str, object] = {
            "console": lambda: ConsoleExporter(verbose=verbose),
            "json": JsonExporter,
            "otlp": lambda: create_otlp_exporter(endpoint=endpoint or "http://localhost:4317", service_name=service_name),
            "otlp_http": lambda: create_otlp_exporter(endpoint=endpoint or "http://localhost:4318/v1/traces", service_name=service_name, use_http=True),
            "datadog": lambda: DatadogExporter(api_key=api_key, service_name=service_name, env=env),
            "honeycomb": lambda: HoneycombExporter(api_key=api_key, service_name=service_name),
            "zipkin": lambda: ZipkinExporter(endpoint=endpoint or "http://localhost:9411/api/v2/spans", service_name=service_name),
            "none": NoOpExporter,
        }
        if exporter not in exporters:
            raise ValueError(f"Unknown exporter: {exporter}. Use: {', '.join(sorted(exporters))}")
        exp: Exporter = exporters[exporter]()  # type: ignore[assignment]
    else:
        exp = exporter
    
    # Apply sampling if configured
    if sample_rate is not None:
        exp = SampledExporter(exp, rate=sample_rate)
    
    # Apply async batching if configured
    if async_export:
        exp = AsyncBatchExporter(exp)
    
    tracer = Tracer(service_name=service_name, exporter=exp, enabled=True)
    tracer.configure_global()
    return tracer
