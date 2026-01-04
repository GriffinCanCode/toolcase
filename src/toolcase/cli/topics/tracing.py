TRACING = """
TOPIC: tracing
==============

Distributed tracing and observability.

EXPORTERS:
    console      Pretty-print to stderr (dev)
    json         JSON lines to stdout
    otlp         OpenTelemetry gRPC (production)
    otlp_http    OpenTelemetry HTTP/protobuf
    datadog      Datadog APM
    honeycomb    Honeycomb via OTLP HTTP
    zipkin       Zipkin v2 JSON format
    none         Disabled (no-op)

CONFIGURATION:
    from toolcase import configure_tracing
    
    # Development
    configure_tracing(service_name="my-service", exporter="console")
    
    # Production OTLP
    configure_tracing(
        exporter="otlp",
        endpoint="http://otel-collector:4317",
        async_export=True,      # Background export
        sample_rate=0.1,        # 10% sampling
    )
    
    # Datadog
    configure_tracing(
        exporter="datadog",
        api_key="dd-xxx",       # or DD_API_KEY env var
        env="production",
    )
    
    # Honeycomb
    configure_tracing(
        exporter="honeycomb",
        api_key="hcaik_xxx",    # or HONEYCOMB_API_KEY env var
    )

ADVANCED EXPORTERS:
    from toolcase import (
        AsyncBatchExporter,     # Background queue export
        SampledExporter,        # Head-based sampling
        FilteredExporter,       # Predicate filtering
        CompositeExporter,      # Fan-out to multiple
        errors_only,            # Only error spans
        slow_spans,             # Spans > threshold
    )
    
    # Export errors to console, sample 10% to production
    exporter = CompositeExporter([
        FilteredExporter(ConsoleExporter(), errors_only),
        SampledExporter(DatadogExporter(...), rate=0.1),
    ])

ENVIRONMENT VARIABLES:
    TOOLCASE_TRACING_ENABLED=true
    TOOLCASE_TRACING_SERVICE_NAME=my-service
    TOOLCASE_TRACING_OTLP_ENDPOINT=http://localhost:4317
    TOOLCASE_TRACING_SAMPLE_RATE=1.0
    DD_API_KEY=...                # Datadog
    HONEYCOMB_API_KEY=...         # Honeycomb

MIDDLEWARE:
    from toolcase import TracingMiddleware, CorrelationMiddleware
    
    registry.use(CorrelationMiddleware())  # Add correlation IDs
    registry.use(TracingMiddleware())       # Create spans

MANUAL TRACING:
    from toolcase import get_tracer, SpanStatus
    
    tracer = get_tracer()
    with tracer.span("my_operation") as span:
        span.set_attribute("key", "value")
        span.set_status(SpanStatus.OK)

DECORATOR:
    from toolcase import traced
    
    @traced(name="fetch_data")
    async def fetch_data(url: str):
        ...

RELATED TOPICS:
    toolcase help middleware   Middleware composition
    toolcase help logging      Structured logging with trace correlation
"""
