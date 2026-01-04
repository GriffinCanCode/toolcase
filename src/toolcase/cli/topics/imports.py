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
        Concurrency, TaskGroup,
        # Batch
        batch_execute, BatchConfig, BatchResult,
        # Settings
        get_settings, get_env, require_env, load_env,
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
    )

SUBMODULE IMPORTS:

    # Foundation
    from toolcase.foundation.core import BaseTool, tool
    from toolcase.foundation.errors import Result, Ok, Err
    from toolcase.foundation.di import Container, Scope
    from toolcase.foundation.registry import get_registry
    from toolcase.foundation.formats import to_openai, to_anthropic, to_google
    from toolcase.foundation.testing import ToolTestCase, mock_tool
    from toolcase.foundation.config import get_settings
    
    # IO
    from toolcase.io.cache import get_cache, MemoryCache, set_cache
    from toolcase.io.progress import ToolProgress, status, step, complete
    from toolcase.io.streaming import StreamEvent, sse_adapter, ws_adapter
    
    # Runtime
    from toolcase.runtime.middleware import compose, Middleware, ValidationMiddleware
    from toolcase.runtime.retry import RetryPolicy, ExponentialBackoff
    from toolcase.runtime.pipeline import pipeline, parallel, streaming_pipeline
    from toolcase.runtime.agents import router, fallback, race, gate, retry_with_escalation
    from toolcase.runtime.concurrency import Concurrency, TaskGroup, run_sync
    from toolcase.runtime.observability import configure_tracing, configure_logging
    from toolcase.runtime.batch import batch_execute, BatchConfig
    
    # Extensions
    from toolcase.ext.integrations import to_langchain_tools
    from toolcase.ext.mcp import serve_mcp, serve_http
    
    # Built-in tools
    from toolcase.tools import HttpTool, DiscoveryTool

RELATED TOPICS:
    toolcase help overview      What is toolcase
    toolcase help tool          Creating tools
    toolcase help architecture  Module structure
"""
