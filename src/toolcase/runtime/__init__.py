"""Runtime - Execution flow, control, and monitoring.

Contains: agents, middleware, retry, pipeline, observability.
"""

from __future__ import annotations

__all__ = [
    # Agents
    "Route", "RouterTool", "router",
    "FallbackTool", "fallback",
    "EscalationHandler", "EscalationResult", "EscalationStatus", "EscalationTool",
    "QueueEscalation", "retry_with_escalation",
    "RaceTool", "race",
    "GateTool", "gate",
    # Middleware
    "Middleware", "Next", "Context", "compose",
    "CircuitBreakerMiddleware", "LoggingMiddleware", "LogMetricsBackend", "MetricsBackend",
    "MetricsMiddleware", "RateLimitMiddleware", "RetryMiddleware", "TimeoutMiddleware",
    # Observability
    "SpanContext", "TraceContext", "trace_context",
    "Span", "SpanEvent", "SpanKind", "SpanStatus",
    "Tracer", "get_tracer", "configure_tracing", "traced",
    "Exporter", "ConsoleExporter", "JsonExporter", "NoOpExporter",
    "BatchExporter", "CompositeExporter", "OTLPBridge", "create_otlp_exporter",
    "TracingMiddleware", "CorrelationMiddleware",
    # Pipeline
    "Transform", "ChunkTransform", "StreamTransform", "Merge", "StreamMerge",
    "Step", "StreamStep",
    "PipelineTool", "ParallelTool", "StreamingPipelineTool", "StreamingParallelTool",
    "PipelineParams", "ParallelParams",
    "pipeline", "parallel", "streaming_pipeline", "streaming_parallel",
    "identity_dict", "identity_chunk", "concat_merge", "interleave_streams",
    # Retry
    "Backoff", "ExponentialBackoff", "LinearBackoff", "ConstantBackoff", "DecorrelatedJitter",
    "RetryPolicy", "DEFAULT_RETRYABLE", "NO_RETRY",
    "execute_with_retry", "execute_with_retry_sync",
]


def __getattr__(name: str):
    """Lazy imports to avoid circular dependencies."""
    agents_attrs = {
        "Route", "RouterTool", "router",
        "FallbackTool", "fallback",
        "EscalationHandler", "EscalationResult", "EscalationStatus", "EscalationTool",
        "QueueEscalation", "retry_with_escalation",
        "RaceTool", "race",
        "GateTool", "gate",
    }
    if name in agents_attrs:
        from . import agents
        return getattr(agents, name)
    
    middleware_attrs = {
        "Middleware", "Next", "Context", "compose",
        "CircuitBreakerMiddleware", "LoggingMiddleware", "LogMetricsBackend", "MetricsBackend",
        "MetricsMiddleware", "RateLimitMiddleware", "RetryMiddleware", "TimeoutMiddleware",
    }
    if name in middleware_attrs:
        from . import middleware
        return getattr(middleware, name)
    
    observability_attrs = {
        "SpanContext", "TraceContext", "trace_context",
        "Span", "SpanEvent", "SpanKind", "SpanStatus",
        "Tracer", "get_tracer", "configure_tracing", "traced",
        "Exporter", "ConsoleExporter", "JsonExporter", "NoOpExporter",
        "BatchExporter", "CompositeExporter", "OTLPBridge", "create_otlp_exporter",
        "TracingMiddleware", "CorrelationMiddleware",
    }
    if name in observability_attrs:
        from . import observability
        return getattr(observability, name)
    
    pipeline_attrs = {
        "Transform", "ChunkTransform", "StreamTransform", "Merge", "StreamMerge",
        "Step", "StreamStep",
        "PipelineTool", "ParallelTool", "StreamingPipelineTool", "StreamingParallelTool",
        "PipelineParams", "ParallelParams",
        "pipeline", "parallel", "streaming_pipeline", "streaming_parallel",
        "identity_dict", "identity_chunk", "concat_merge", "interleave_streams",
    }
    if name in pipeline_attrs:
        from . import pipeline
        return getattr(pipeline, name)
    
    retry_attrs = {
        "Backoff", "ExponentialBackoff", "LinearBackoff", "ConstantBackoff", "DecorrelatedJitter",
        "RetryPolicy", "DEFAULT_RETRYABLE", "NO_RETRY",
        "execute_with_retry", "execute_with_retry_sync",
    }
    if name in retry_attrs:
        from . import retry
        return getattr(retry, name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
