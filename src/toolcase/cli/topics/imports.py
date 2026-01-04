IMPORTS = """
TOPIC: imports
==============

Import patterns for toolcase.

TOP-LEVEL (Most common):
    from toolcase import (
        # Core
        tool, BaseTool, ToolMetadata,
        # Registry
        get_registry, init_tools,
        # Errors
        Result, Ok, Err, ErrorCode, ToolError,
        # Middleware
        compose, RetryMiddleware, TimeoutMiddleware,
        # Pipeline
        pipeline, parallel, fallback, router,
        # Concurrency
        Concurrency,
    )

SUBMODULE IMPORTS:

    # Foundation
    from toolcase.foundation.core import BaseTool, tool
    from toolcase.foundation.errors import Result, Ok, Err
    from toolcase.foundation.di import Container
    from toolcase.foundation.registry import get_registry
    from toolcase.foundation.formats import to_openai
    from toolcase.foundation.testing import ToolTestCase
    from toolcase.foundation.config import get_settings
    
    # IO
    from toolcase.io.cache import get_cache, MemoryCache
    from toolcase.io.progress import ToolProgress, status
    from toolcase.io.streaming import StreamEvent, sse_adapter
    
    # Runtime
    from toolcase.runtime.middleware import compose, Middleware
    from toolcase.runtime.retry import RetryPolicy, ExponentialBackoff
    from toolcase.runtime.pipeline import pipeline, parallel
    from toolcase.runtime.agents import router, fallback, race
    from toolcase.runtime.concurrency import Concurrency
    from toolcase.runtime.observability import configure_tracing
    
    # Extensions
    from toolcase.ext.integrations import to_langchain_tools
    from toolcase.ext.mcp import serve_mcp, serve_http
    
    # Built-in tools
    from toolcase.tools import HttpTool, DiscoveryTool

RELATED TOPICS:
    toolcase help overview   What is toolcase
    toolcase help tool       Creating tools
"""
