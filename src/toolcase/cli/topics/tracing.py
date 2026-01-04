TRACING = """
TOPIC: tracing
==============

Distributed tracing and observability.

CONFIGURATION:
    from toolcase import configure_tracing
    
    configure_tracing(
        service_name="my-service",
        exporter="otlp",  # or "console", "jaeger"
        endpoint="http://localhost:4317",
    )

MIDDLEWARE:
    from toolcase import TracingMiddleware, CorrelationMiddleware
    
    registry.use(CorrelationMiddleware())  # Add correlation IDs
    registry.use(TracingMiddleware())       # Create spans

MANUAL TRACING:
    from toolcase import get_tracer, traced, Span
    
    tracer = get_tracer()
    
    with tracer.start_span("my_operation") as span:
        span.set_attribute("key", "value")
        result = do_work()
        span.set_status(SpanStatus.OK)

DECORATOR:
    from toolcase import traced
    
    @traced(name="fetch_data")
    async def fetch_data(url: str):
        ...

SPAN ATTRIBUTES:
    span.set_attribute("user_id", user_id)
    span.set_attribute("query", query)
    span.add_event("cache_miss")

SPAN STATUS:
    SpanStatus.UNSET     Not set
    SpanStatus.OK        Success
    SpanStatus.ERROR     Failure

RELATED TOPICS:
    toolcase help middleware   Middleware composition
    toolcase help settings     Configuration
"""
