"""Tracing middleware for automatic tool instrumentation.

Auto-instruments all tool calls with spans capturing:
- Tool name, category, parameters
- Execution timing and status
- Error details with context
- Result preview for debugging
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

from toolcase.foundation.errors import JsonDict, ToolException, classify_exception
from toolcase.runtime.middleware.middleware import Context, Next
from .span import SpanKind, SpanStatus
from .tracer import Tracer

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool


@dataclass(slots=True)
class TracingMiddleware:
    """Auto-instrument tool execution with distributed traces.
    
    Creates a span for each tool call capturing:
    - Tool metadata (name, category)
    - Input parameters (optionally sanitized)
    - Execution timing
    - Result preview (truncated for large outputs)
    - Error context with codes
    
    Integrates with any configured Exporter (console, OTLP, etc.).
    
    Args:
        tracer: Tracer instance (defaults to global)
        capture_params: Include params in span (default True, disable for PII)
        capture_result: Include result preview (default True)
        result_preview_len: Max chars for result preview
    
    Example:
        >>> from toolcase.observability import configure_tracing, TracingMiddleware
        >>> 
        >>> configure_tracing(service_name="my-agent", exporter="console")
        >>> registry.use(TracingMiddleware())
        >>> 
        >>> # Now all tool calls emit traces automatically
        >>> await registry.execute("search", {"query": "python"})
    """
    
    tracer: Tracer | None = None
    capture_params: bool = True
    capture_result: bool = True
    result_preview_len: int = 200
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        tracer = self.tracer or Tracer.current()
        meta, param_dump = tool.metadata, params.model_dump()
        
        # Build span attributes
        attrs: JsonDict = {"tool.name": meta.name, "tool.category": meta.category}
        if self.capture_params:
            attrs["tool.params"] = param_dump
        if "request_id" in ctx:  # Add correlation from context if present
            attrs["request.id"] = ctx["request_id"]
        
        with tracer.span(f"tool.{meta.name}", SpanKind.TOOL, attrs) as span:
            span.set_tool_context(meta.name, meta.category, param_dump if self.capture_params else None)
            ctx.update(trace_span=span, trace_id=span.context.trace_id, span_id=span.context.span_id)
            
            try:
                result = await next(tool, params, ctx)
                
                if self.capture_result:
                    span.set_result_preview(result, self.result_preview_len)
                
                # Detect error responses
                if result.startswith("**Tool Error"):
                    span.set_status(SpanStatus.ERROR, "tool returned error response")
                    span.add_event("tool_error", {"response_prefix": result[:100]})
                else:
                    span.set_status(SpanStatus.OK).add_event("tool_success")
                
                if span.duration_ms:
                    ctx["duration_ms"] = span.duration_ms
                
                return result
                
            except ToolException as e:
                err = e.error
                span.set_status(SpanStatus.ERROR, err.message).set_attributes(
                    {"error.code": err.code.value, "error.recoverable": err.recoverable}
                ).add_event("tool_exception", {"code": err.code.value, "message": err.message})
                raise
                
            except Exception as e:
                code, etype = classify_exception(e), type(e).__name__
                span.set_status(SpanStatus.ERROR, str(e)).set_attributes(
                    {"error.code": code.value, "error.type": etype}
                ).add_event("exception", {"type": etype, "message": str(e), "code": code.value})
                raise


@dataclass(slots=True)
class CorrelationMiddleware:
    """Add correlation IDs to context for request tracing.
    
    Generates or propagates request IDs for correlating logs, traces, and metrics across a request lifecycle.
    
    Args:
        header_name: Header to extract correlation ID from (for HTTP)
        generate_if_missing: Auto-generate ID if not present
    
    Example:
        >>> registry.use(CorrelationMiddleware())
        >>> registry.use(TracingMiddleware())  # Will pick up correlation ID
    """
    
    header_name: str = "X-Request-ID"
    generate_if_missing: bool = True
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        if "request_id" not in ctx and self.generate_if_missing:
            import secrets
            ctx["request_id"] = secrets.token_hex(8)
        return await next(tool, params, ctx)
