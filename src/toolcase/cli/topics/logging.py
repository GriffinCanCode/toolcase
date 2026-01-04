LOGGING = """
TOPIC: logging
==============

Structured logging with trace context correlation.

CONFIGURATION:
    from toolcase import configure_logging
    
    # Development (human-readable, colored)
    configure_logging(format="console", level="DEBUG")
    
    # Production (JSON for aggregation)
    configure_logging(format="json", level="INFO")

BASIC USAGE:
    from toolcase import get_logger
    
    log = get_logger("my-service")
    log.info("processing request", user_id=123, query="python")
    
    # Output: 10:30:45.123 [info] processing request user_id=123 query="python"

CONTEXT BINDING:
    # Bind context for all subsequent logs
    log = log.bind(request_id="abc123")
    log.info("fetching data")     # includes request_id
    log.info("processing")        # includes request_id
    
    # Bind tool context
    log = log.bind_tool("web_search", "search")
    log.info("executing", query="python")

SCOPED CONTEXT:
    from toolcase import log_context
    
    with log_context(request_id="abc123", user_id=42):
        log.info("processing")    # includes request_id and user_id
    log.info("done")              # no longer includes them

TIMING DECORATOR:
    from toolcase import timed
    
    @timed(event="data fetched")
    def fetch_data(url: str):
        return requests.get(url).json()
    
    # Output: 10:30:45.123 [info] data fetched function=fetch_data duration_ms=45.2

MIDDLEWARE:
    from toolcase import LoggingMiddleware
    
    # Structured logging for all tool executions
    registry.use(LoggingMiddleware())
    
    # With params logging (for debugging)
    registry.use(LoggingMiddleware(log_params=True))

TRACE CORRELATION:
    When tracing is enabled, logs automatically include trace_id and span_id:
    
    from toolcase import configure_tracing, configure_logging
    
    configure_tracing(service_name="my-agent")
    configure_logging(format="json")
    
    # Logs now include: {"trace_id": "...", "span_id": "...", ...}

LOG LEVELS:
    log.debug("verbose info")
    log.info("normal operation")
    log.warning("potential issue")
    log.error("something failed")
    log.exception("with stack trace")

RELATED TOPICS:
    toolcase help tracing      Distributed tracing
    toolcase help middleware   Middleware composition
    toolcase help settings     Configuration
"""
