IMPORTS = """
TOPIC: imports
==============

Import patterns for toolcase.

TOP-LEVEL (Most common):
    from toolcase import (
        # Core
        tool, BaseTool, ToolMetadata, ToolCapabilities,
        # Registry
        get_registry, init_tools,
        # Errors
        Result, Ok, Err, ErrorCode, ToolError, ToolResult,
        # Middleware
        compose, RetryMiddleware, TimeoutMiddleware, LoggingMiddleware,
        # Pipeline
        pipeline, parallel, fallback, router, race, gate,
        # Concurrency
        Concurrency, TaskGroup, CancelScope,
        # Batch
        batch_execute, BatchConfig, BatchResult,
        # Settings
        get_settings, get_env, require_env, load_env,
        # Observability
        configure_tracing, configure_logging, get_tracer, get_logger,
    )

BUILT-IN TOOLS & AUTH:
    from toolcase import (
        # HTTP Tool
        HttpTool, HttpConfig, HttpParams, HttpResponse,
        # Auth strategies (direct)
        NoAuth, BearerAuth, BasicAuth, ApiKeyAuth, CustomAuth,
        # Auth strategies (env-based, recommended)
        bearer_from_env, api_key_from_env, basic_from_env,
        EnvBearerAuth, EnvApiKeyAuth, EnvBasicAuth,
        # Discovery
        DiscoveryTool, ToolQuery, find_by_param,
        # Statistics
        StatsMiddleware, get_stats, format_stats,
    )

WEB TOOLS:
    from toolcase.tools import (
        # Web Search
        WebSearchTool, WebSearchConfig, WebSearchParams,
        free_search, tavily_search, perplexity_search,
        SearchBackend, TavilyBackend, PerplexityBackend, DuckDuckGoBackend,
        # URL Fetch
        UrlFetchTool, UrlFetchConfig, UrlFetchParams,
        # HTML Parse
        HtmlParseTool, HtmlParseConfig, HtmlParseParams,
        # Extraction
        RegexExtractTool, RegexExtractConfig, RegexExtractParams,
        JsonExtractTool,
    )

SUBMODULE IMPORTS:

    # Foundation
    from toolcase.foundation.core import BaseTool, tool, ToolCapabilities
    from toolcase.foundation.errors import Result, Ok, Err, sequence, traverse
    from toolcase.foundation.di import Container, Scope, ScopedContext
    from toolcase.foundation.registry import get_registry
    from toolcase.foundation.testing import ToolTestCase, mock_tool
    from toolcase.foundation.config import get_settings, load_env, get_env, require_env
    from toolcase.foundation.effects import (
        effects, get_effects, test_effects, effect_scope,
        InMemoryDB, RecordingHTTP, InMemoryFS, FrozenTime,
        SeededRandom, CollectingLogger, NoOpCache,
    )
    
    # Format converters (multi-provider)
    from toolcase.ext.integrations.frontiers import (
        to_openai, to_anthropic, to_google,
        tool_to_openai, tool_to_anthropic, tool_to_google,
        to_provider,  # Universal converter
    )
    
    # IO
    from toolcase.io.cache import (
        get_cache, set_cache, reset_cache,
        MemoryCache, TaggedMemoryCache, SWRCache, SWRConfig,
        RedisCache, AsyncRedisCache, MemcachedCache,
    )
    from toolcase.io.progress import ToolProgress, status, step, complete
    from toolcase.io.streaming import (
        StreamEvent, StreamChunk, StreamResult,
        sse_adapter, ws_adapter, json_lines_adapter, binary_adapter,
        encode, decode, pack, unpack,  # Codecs
        result_stream, ok_chunk, err_chunk, filter_ok, recover,  # Result streaming
    )
    
    # Runtime
    from toolcase.runtime.middleware import compose, Middleware
    from toolcase.runtime.middleware.plugins import (
        ValidationMiddleware, Schema,
        required, optional, is_str, min_len, rule_in_range,
        when, when_eq, mutex, depends_on,  # Rule DSL
    )
    from toolcase.runtime.retry import RetryPolicy, ExponentialBackoff
    from toolcase.runtime.pipeline import pipeline, parallel, streaming_pipeline
    from toolcase.runtime.agents import router, fallback, race, gate, retry_with_escalation
    from toolcase.runtime.concurrency import Concurrency, TaskGroup, run_sync, to_thread
    from toolcase.runtime.observability import configure_tracing, configure_logging
    from toolcase.runtime.batch import (
        batch_execute, BatchConfig, batch_execute_stream,
        batch_execute_idempotent, IdempotentBatchConfig, BatchRetryPolicy,
    )
    
    # Extensions
    from toolcase.ext.integrations import to_langchain_tools
    from toolcase.ext.integrations.frontiers import to_openai, to_anthropic, to_google
    from toolcase.ext.mcp import serve_mcp, serve_http, create_mcp_server
    
    # Built-in tools
    from toolcase.tools import HttpTool, DiscoveryTool, standard_tools
    from toolcase.tools import WebSearchTool, UrlFetchTool, HtmlParseTool

CONCURRENCY:
    from toolcase import (
        Concurrency, ConcurrencyConfig,
        TaskGroup, TaskHandle, CancelScope,
        Lock, Semaphore, Event, Barrier, CapacityLimiter,
        run_sync, to_thread, AsyncAdapter, SyncAdapter,
    )
    
    from toolcase.runtime.concurrency.streams import (
        merge_streams, buffer_stream, throttle_stream, batch_stream,
        backpressure_stream,
    )

BATCH & IDEMPOTENCY:
    from toolcase import (
        batch_execute, batch_execute_sync,
        BatchConfig, BatchItem, BatchResult,
        batch_execute_idempotent, batch_execute_idempotent_sync,
        IdempotentBatchConfig, IdempotentBatchResult,
        BatchRetryPolicy, BatchRetryStrategy, NO_BATCH_RETRY,
        CacheIdempotencyAdapter, IdempotencyStore,
    )

OBSERVABILITY:
    from toolcase import (
        # Core
        configure_tracing, configure_logging,
        get_tracer, get_logger, traced, timed,
        # Exporters
        ConsoleExporter, JsonExporter, OTLPBridge,
        DatadogExporter, HoneycombExporter, ZipkinExporter,
        CompositeExporter, SampledExporter, FilteredExporter,
        # Predicates
        errors_only, slow_spans,
    )

RELATED TOPICS:
    toolcase help overview      What is toolcase
    toolcase help tool          Creating tools
    toolcase help architecture  Module structure
    toolcase help web           Web tools (search, fetch, parse)
"""
