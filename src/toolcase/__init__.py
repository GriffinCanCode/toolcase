"""Toolcase - Type-safe, extensible tool framework for AI agents.

A minimal yet powerful framework for creating tools that AI agents can invoke.
Supports type-safe parameters, caching, progress streaming, and multi-framework
format converters for OpenAI, Anthropic, Google Gemini, LangChain, and MCP.

Quick Start (Decorator - Recommended):
    >>> from toolcase import tool, get_registry
    >>>
    >>> @tool(description="Search for information", category="search")
    ... def search(query: str, limit: int = 5) -> str:
    ...     '''Search the web.
    ...     
    ...     Args:
    ...         query: Search query string
    ...         limit: Max results to return
    ...     '''
    ...     return f"Results for: {query}"
    >>>
    >>> registry = get_registry()
    >>> registry.register(search)
    >>> search(query="python")
    'Results for: python'

Class-Based (For Complex Tools):
    >>> from toolcase import BaseTool, ToolMetadata
    >>> from pydantic import BaseModel, Field
    >>>
    >>> class SearchParams(BaseModel):
    ...     query: str = Field(..., description="Search query")
    ...
    >>> class SearchTool(BaseTool[SearchParams]):
    ...     metadata = ToolMetadata(
    ...         name="search",
    ...         description="Search for information",
    ...         category="search",
    ...     )
    ...     params_schema = SearchParams
    ...
    ...     async def _async_run(self, params: SearchParams) -> str:
    ...         return f"Results for: {params.query}"

Multi-Framework Format Converters:
    >>> from toolcase.foundation.formats import to_openai, to_anthropic, to_google
    >>>
    >>> # OpenAI function calling format
    >>> openai_tools = to_openai(registry)
    >>> 
    >>> # Anthropic tool_use format
    >>> anthropic_tools = to_anthropic(registry)
    >>> 
    >>> # Google Gemini function declarations
    >>> gemini_tools = to_google(registry)

LangChain Integration:
    >>> from toolcase.ext.integrations import to_langchain_tools
    >>> lc_tools = to_langchain_tools(registry)

MCP (Model Context Protocol) Integration:
    >>> from toolcase.ext.mcp import serve_mcp
    >>> serve_mcp(registry, transport="sse", port=8080)

HTTP REST Server (Web Backends):
    >>> from toolcase.ext.mcp import serve_http
    >>> serve_http(registry, port=8000)  # Simple HTTP endpoints
"""

from __future__ import annotations

__version__ = "0.2.0"

# Foundation: Core
from .foundation.core import (
    BaseTool,
    EmptyParams,
    FunctionTool,
    ResultStreamingFunctionTool,
    StreamingFunctionTool,
    ToolCapabilities,
    ToolMetadata,
    tool,
)

# Foundation: Errors
from .foundation.errors import (
    Err,
    ErrorCode,
    ErrorContext,
    ErrorTrace,
    Ok,
    Result,
    ResultT,
    ToolError,
    ToolException,
    ToolResult,
    batch_results,
    classify_exception,
    collect_results,
    sequence,
    tool_result,
    traverse,
    try_tool_operation,
    try_tool_operation_async,
    validate_context,
    validate_trace,
)

# Foundation: DI
from .foundation.di import Container, Disposable, Factory, Provider, Scope, ScopedContext

# Foundation: Registry
from .foundation.registry import ToolRegistry, get_registry, reset_registry, set_registry

# Foundation: Config/Settings
from .foundation.config import (
    CacheSettings,
    HttpSettings,
    LoggingSettings,
    RateLimitSettings,
    RetrySettings,
    ToolcaseSettings,
    TracingSettings,
    clear_settings_cache,
    dotenv_values,
    env,
    get_env,
    get_env_files_loaded,
    get_env_prefix,
    get_settings,
    load_env,
    require_env,
)

# Foundation: Testing
from .foundation.testing import (
    Invocation,
    MockAPI,
    MockResponse,
    MockTool,
    ToolTestCase,
    fixture,
    mock_api,
    mock_api_slow,
    mock_api_with_errors,
    mock_tool,
)

# IO: Progress
from .io.progress import (
    ProgressCallback,
    ProgressKind,
    ToolProgress,
    complete,
    error,
    source_found,
    status,
    step,
    validate_progress,
)

# IO: Cache
from .io.cache import (
    DEFAULT_TTL,
    AsyncToolCache,
    CacheBackend,
    MemoryCache,
    ToolCache,
    get_cache,
    reset_cache,
    set_cache,
)

# IO: Streaming
from .io.streaming import (
    StreamAdapter,
    StreamChunk,
    StreamEvent,
    StreamEventKind,
    StreamResult,
    StreamState,
    chunk,
    json_lines_adapter,
    sse_adapter,
    stream_complete,
    stream_error,
    stream_start,
    ws_adapter,
)

# Runtime: Middleware
from .runtime.middleware import (
    CircuitBreakerMiddleware,
    Context,
    LoggingMiddleware,
    MetricsMiddleware,
    Middleware,
    Next,
    RateLimitMiddleware,
    RetryMiddleware,
    TimeoutMiddleware,
    compose,
)

# Runtime: Retry
from .runtime.retry import (
    Backoff,
    ConstantBackoff,
    DecorrelatedJitter,
    DEFAULT_RETRYABLE,
    ExponentialBackoff,
    LinearBackoff,
    NO_RETRY,
    RetryPolicy,
    validate_policy,
)

# Runtime: Pipeline
from .runtime.pipeline import (
    ChunkTransform,
    ParallelTool,
    PipelineTool,
    Step,
    StreamingParallelTool,
    StreamingPipelineTool,
    StreamMerge,
    StreamStep,
    interleave_streams,
    parallel,
    pipeline,
    streaming_parallel,
    streaming_pipeline,
)

# Runtime: Observability
from .runtime.observability import (
    AsyncBatchExporter,
    BatchExporter,
    BoundLogger,
    CompositeExporter,
    ConsoleExporter,
    CorrelationMiddleware,
    DatadogExporter,
    Exporter,
    FilteredExporter,
    HoneycombExporter,
    JsonExporter,
    NoOpExporter,
    OTLPBridge,
    OTLPHttpBridge,
    SampledExporter,
    Span,
    SpanKind,
    SpanPredicate,
    SpanStatus,
    Tracer,
    TracingMiddleware,
    ZipkinExporter,
    configure_logging,
    configure_tracing,
    create_otlp_exporter,
    datadog,
    errors_only,
    get_logger,
    get_tracer,
    honeycomb,
    log_context,
    slow_spans,
    timed,
    traced,
    zipkin,
)

# Runtime: Agents
from .runtime.agents import (
    EscalationHandler,
    EscalationResult,
    EscalationStatus,
    EscalationTool,
    FallbackTool,
    GateTool,
    QueueEscalation,
    RaceTool,
    Route,
    RouterTool,
    fallback,
    gate,
    race,
    retry_with_escalation,
    router,
)

# Runtime: Concurrency
from .runtime.concurrency import (
    Concurrency,
    ConcurrencyConfig,
    TaskGroup,
    TaskHandle,
    CancelScope,
    Lock,
    Semaphore,
    Event,
    Barrier,
    CapacityLimiter,
    run_sync,
    to_thread,
    AsyncAdapter,
    SyncAdapter,
)

# Runtime: Batch
from .runtime.batch import (
    BatchConfig,
    BatchItem,
    BatchResult,
    batch_execute,
    batch_execute_sync,
)

# Built-in tools
from .tools import (
    ApiKeyAuth,
    BasicAuth,
    BearerAuth,
    CapabilityFilter,
    ConfigurableTool,
    CustomAuth,
    DiscoveryParams,
    DiscoveryTool,
    EnvApiKeyAuth,
    EnvBasicAuth,
    EnvBearerAuth,
    HttpConfig,
    HttpParams,
    HttpResponse,
    HttpTool,
    MatchMode,
    NoAuth,
    QueryResult,
    SchemaPattern,
    StatsMiddleware,
    ToolConfig,
    ToolQuery,
    ToolStats,
    UsageStats,
    api_key_from_env,
    basic_from_env,
    bearer_from_env,
    find_by_input_type,
    find_by_max_concurrent,
    find_by_param,
    find_by_tags,
    find_cacheable,
    find_streamable,
    format_stats,
    get_stats,
    reset_stats,
    set_stats,
    standard_tools,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "BaseTool", "ToolMetadata", "ToolCapabilities", "EmptyParams", "tool",
    "FunctionTool", "StreamingFunctionTool", "ResultStreamingFunctionTool",
    # Errors
    "ErrorCode", "ToolError", "ToolException", "classify_exception",
    # Progress
    "ToolProgress", "ProgressKind", "ProgressCallback",
    "status", "step", "source_found", "complete", "error",
    # Cache
    "ToolCache", "AsyncToolCache", "MemoryCache", "CacheBackend",
    "get_cache", "set_cache", "reset_cache", "DEFAULT_TTL",
    # Registry
    "ToolRegistry", "get_registry", "set_registry", "reset_registry",
    # Middleware
    "Middleware", "Context", "Next", "compose",
    "CircuitBreakerMiddleware", "LoggingMiddleware", "MetricsMiddleware",
    "RateLimitMiddleware", "RetryMiddleware", "TimeoutMiddleware",
    # Retry policies
    "Backoff", "ExponentialBackoff", "LinearBackoff", "ConstantBackoff", "DecorrelatedJitter",
    "RetryPolicy", "DEFAULT_RETRYABLE", "NO_RETRY",
    # Pipeline composition
    "PipelineTool", "ParallelTool", "StreamingPipelineTool", "StreamingParallelTool",
    "Step", "StreamStep", "ChunkTransform", "StreamMerge",
    "pipeline", "parallel", "streaming_pipeline", "streaming_parallel", "interleave_streams",
    # Dependency Injection
    "Container", "Disposable", "Factory", "Provider", "Scope", "ScopedContext",
    # Settings/Config
    "ToolcaseSettings", "get_settings", "clear_settings_cache",
    "CacheSettings", "LoggingSettings", "RetrySettings",
    "HttpSettings", "TracingSettings", "RateLimitSettings",
    # Environment utilities
    "load_env", "get_env", "require_env", "env",
    "get_env_prefix", "get_env_files_loaded", "dotenv_values",
    # Built-in tools
    "DiscoveryTool", "DiscoveryParams",
    "ConfigurableTool", "ToolConfig",
    "HttpTool", "HttpConfig", "HttpParams", "HttpResponse",
    "NoAuth", "BearerAuth", "BasicAuth", "ApiKeyAuth", "CustomAuth",
    "EnvBearerAuth", "EnvApiKeyAuth", "EnvBasicAuth",
    "bearer_from_env", "api_key_from_env", "basic_from_env",
    "standard_tools",
    # Tool discovery & query
    "ToolQuery", "SchemaPattern", "CapabilityFilter", "QueryResult", "MatchMode",
    "find_by_param", "find_by_tags", "find_by_input_type",
    "find_streamable", "find_cacheable", "find_by_max_concurrent",
    # Usage statistics
    "UsageStats", "ToolStats", "StatsMiddleware",
    "get_stats", "set_stats", "reset_stats", "format_stats",
    # Monadic error handling
    "Result", "Ok", "Err", "ResultT", "ToolResult", "ErrorContext", "ErrorTrace",
    "tool_result", "try_tool_operation", "try_tool_operation_async",
    "batch_results", "sequence", "traverse", "collect_results",
    # TypeAdapter validation utilities (fast dict→model validation)
    "validate_context", "validate_trace", "validate_progress", "validate_policy",
    # Observability - Core
    "configure_tracing", "get_tracer", "traced",
    "configure_logging", "get_logger", "log_context", "timed", "BoundLogger",
    "CorrelationMiddleware", "TracingMiddleware",
    "Span", "SpanKind", "SpanStatus", "Tracer",
    # Observability - Exporters
    "Exporter", "ConsoleExporter", "JsonExporter", "NoOpExporter",
    "BatchExporter", "AsyncBatchExporter", "CompositeExporter",
    "OTLPBridge", "OTLPHttpBridge", "create_otlp_exporter",
    "DatadogExporter", "HoneycombExporter", "ZipkinExporter",
    "datadog", "honeycomb", "zipkin",
    "SampledExporter", "FilteredExporter", "SpanPredicate",
    "errors_only", "slow_spans",
    # Streaming (Result Streaming)
    "StreamAdapter", "StreamChunk", "StreamEvent", "StreamEventKind",
    "StreamResult", "StreamState",
    "chunk", "stream_start", "stream_complete", "stream_error",
    "sse_adapter", "ws_adapter", "json_lines_adapter",
    # Testing utilities
    "ToolTestCase", "mock_tool", "MockTool", "Invocation",
    "fixture", "MockAPI", "MockResponse",
    "mock_api", "mock_api_with_errors", "mock_api_slow",
    # Agentic composition primitives
    "Route", "RouterTool", "router",
    "FallbackTool", "fallback",
    "EscalationHandler", "EscalationResult", "EscalationStatus", "EscalationTool",
    "QueueEscalation", "retry_with_escalation",
    "RaceTool", "race",
    "GateTool", "gate",
    # Concurrency
    "Concurrency", "ConcurrencyConfig",
    "TaskGroup", "TaskHandle", "CancelScope",
    "Lock", "Semaphore", "Event", "Barrier", "CapacityLimiter",
    "run_sync", "to_thread", "AsyncAdapter", "SyncAdapter",
    # Batch
    "BatchConfig", "BatchItem", "BatchResult",
    "batch_execute", "batch_execute_sync",
    # Convenience
    "init_tools",
]


def init_tools(*tools: BaseTool) -> ToolRegistry:  # type: ignore[type-arg]
    """Initialize the registry with tools.
    
    Convenience function that registers the discovery tool and any
    additional tools provided.
    
    Args:
        *tools: Additional tool instances to register
    
    Returns:
        The initialized global registry
    
    Example:
        >>> from toolcase import init_tools
        >>> registry = init_tools(MyTool(), AnotherTool())
    """
    registry = get_registry()
    registry.register(DiscoveryTool())
    for t in tools:
        registry.register(t)
    return registry
