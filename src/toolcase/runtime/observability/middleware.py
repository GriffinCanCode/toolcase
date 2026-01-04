"""Tracing middleware for automatic tool instrumentation.

Auto-instruments all tool calls with spans capturing:
- Tool name, category, parameters
- Execution timing and status
- Error details with context
- Result preview for debugging
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel

from toolcase.foundation.errors import ErrorCode, JsonDict, ToolException, classify_exception
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
    
    def _get_tracer(self) -> Tracer:
        return self.tracer or Tracer.current()
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        tracer = self._get_tracer()
        
        # Build span attributes
        attrs: JsonDict = {
            "tool.name": tool.metadata.name,
            "tool.category": tool.metadata.category,
        }
        
        if self.capture_params:
            attrs["tool.params"] = params.model_dump()
        
        # Add correlation from context if present
        if "request_id" in ctx:
            attrs["request.id"] = ctx["request_id"]
        
        with tracer.span(
            name=f"tool.{tool.metadata.name}",
            kind=SpanKind.TOOL,
            attributes=attrs,
        ) as span:
            # Enrich span with tool context
            span.set_tool_context(
                tool.metadata.name,
                tool.metadata.category,
                params.model_dump() if self.capture_params else None,
            )
            
            # Store span in context for downstream access
            ctx["trace_span"] = span
            ctx["trace_id"] = span.context.trace_id
            ctx["span_id"] = span.context.span_id
            
            try:
                result = await next(tool, params, ctx)
                
                # Capture result preview
                if self.capture_result:
                    span.set_result_preview(result, self.result_preview_len)
                
                # Detect error responses
                if result.startswith("**Tool Error"):
                    span.set_status(SpanStatus.ERROR, "tool returned error response")
                    span.add_event("tool_error", {"response_prefix": result[:100]})
                else:
                    span.set_status(SpanStatus.OK)
                    span.add_event("tool_success")
                
                # Store duration in context for other middleware
                if span.duration_ms:
                    ctx["duration_ms"] = span.duration_ms
                
                return result
                
            except ToolException as e:
                span.set_status(SpanStatus.ERROR, e.error.message)
                span.set_attribute("error.code", e.error.code.value)
                span.set_attribute("error.recoverable", e.error.recoverable)
                span.add_event("tool_exception", {
                    "code": e.error.code.value,
                    "message": e.error.message,
                })
                raise
                
            except Exception as e:
                code = classify_exception(e)
                span.set_status(SpanStatus.ERROR, str(e))
                span.set_attribute("error.code", code.value)
                span.set_attribute("error.type", type(e).__name__)
                span.add_event("exception", {
                    "type": type(e).__name__,
                    "message": str(e),
                    "code": code.value,
                })
                raise


@dataclass(slots=True)
class CorrelationMiddleware:
    """Add correlation IDs to context for request tracing.
    
    Generates or propagates request IDs for correlating logs,
    traces, and metrics across a request lifecycle.
    
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
        # Check if correlation ID already exists
        if "request_id" not in ctx:
            if self.generate_if_missing:
                import secrets
                ctx["request_id"] = secrets.token_hex(8)
        
        return await next(tool, params, ctx)
