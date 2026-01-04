"""Runtime - Execution flow, control, and monitoring.

Contains: agents, middleware, retry, pipeline, observability, concurrency.
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
    "StreamMiddleware", "StreamingAdapter", "StreamingChain", "compose_streaming",
    "StreamLoggingMiddleware", "StreamMetricsMiddleware",
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
    # Concurrency (unified facade + direct exports)
    "Concurrency",  # Primary unified facade
    "TaskGroup", "TaskHandle", "TaskState", "CancelScope",
    "Lock", "RLock", "Semaphore", "BoundedSemaphore", "Event", "Condition", "Barrier", "CapacityLimiter",
    "ThreadPool", "ProcessPool", "run_in_thread", "run_in_process",
    "race_async", "gather_async", "gather_settled", "first_success", "map_async", "all_settled",
    "merge_streams", "interleave_streams_async", "buffer_stream", "throttle_stream", "batch_stream",
    "run_sync", "run_async", "from_thread", "to_thread", "AsyncAdapter", "SyncAdapter",
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
        "StreamMiddleware", "StreamingAdapter", "StreamingChain", "compose_streaming",
        "StreamLoggingMiddleware", "StreamMetricsMiddleware",
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
    
    concurrency_attrs = {
        "Concurrency",  # Unified facade
        "TaskGroup", "TaskHandle", "TaskState", "CancelScope",
        "Lock", "RLock", "Semaphore", "BoundedSemaphore", "Event", "Condition", "Barrier", "CapacityLimiter",
        "ThreadPool", "ProcessPool", "run_in_thread", "run_in_process",
        "race_async", "gather_async", "gather_settled", "first_success", "map_async", "all_settled",
        "merge_streams", "interleave_streams_async", "buffer_stream", "throttle_stream", "batch_stream",
        "run_sync", "run_async", "from_thread", "to_thread", "AsyncAdapter", "SyncAdapter",
    }
    if name in concurrency_attrs:
        from . import concurrency
        # Map renamed exports
        attr_map = {
            "race_async": "race",
            "gather_async": "gather",
            "interleave_streams_async": "interleave_streams",
        }
        return getattr(concurrency, attr_map.get(name, name))
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
