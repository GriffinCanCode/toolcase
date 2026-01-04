"""Foundation - Core building blocks for toolcase.

Contains: core abstractions, error handling, DI, registry, testing, formats, config.
"""

from __future__ import annotations

__all__ = [
    # Core
    "BaseTool", "ToolMetadata", "EmptyParams", "tool",
    "FunctionTool", "StreamingFunctionTool", "ResultStreamingFunctionTool",
    "set_injected_deps", "clear_injected_deps",
    # Errors
    "ErrorCode", "ToolError", "ToolException", "classify_exception",
    "Result", "Ok", "Err", "ResultT", "try_fn",
    "ToolResult", "tool_result", "ok_result", "try_tool_operation", "try_tool_operation_async",
    "batch_results", "from_tool_error", "to_tool_error", "result_to_string", "string_to_result",
    "ErrorContext", "ErrorTrace", "context", "trace", "trace_from_exc",
    "sequence", "traverse", "collect_results",
    # DI
    "Container", "DIResult", "Disposable", "Factory", "Provider", "Scope", "ScopedContext",
    # Registry
    "ToolRegistry", "get_registry", "set_registry", "reset_registry",
    # Testing
    "ToolTestCase", "mock_tool", "MockTool", "Invocation",
    "fixture", "MockAPI", "MockResponse", "mock_api", "mock_api_with_errors", "mock_api_slow",
    # Config
    "ToolcaseSettings", "get_settings", "clear_settings_cache",
    "CacheSettings", "LoggingSettings", "RetrySettings", "HttpSettings", "TracingSettings", "RateLimitSettings",
]


def __getattr__(name: str):
    """Lazy imports to avoid circular dependencies."""
    if name in ("BaseTool", "ToolMetadata", "EmptyParams", "tool",
                "FunctionTool", "StreamingFunctionTool", "ResultStreamingFunctionTool",
                "set_injected_deps", "clear_injected_deps"):
        from . import core
        return getattr(core, name)
    
    if name in ("ErrorCode", "ToolError", "ToolException", "classify_exception",
                "Result", "Ok", "Err", "ResultT", "try_fn",
                "ToolResult", "tool_result", "ok_result", "try_tool_operation", "try_tool_operation_async",
                "batch_results", "from_tool_error", "to_tool_error", "result_to_string", "string_to_result",
                "ErrorContext", "ErrorTrace", "context", "trace", "trace_from_exc",
                "sequence", "traverse", "collect_results"):
        from . import errors
        return getattr(errors, name)
    
    if name in ("Container", "DIResult", "Disposable", "Factory", "Provider", "Scope", "ScopedContext"):
        from . import di
        return getattr(di, name)
    
    if name in ("ToolRegistry", "get_registry", "set_registry", "reset_registry"):
        from . import registry
        return getattr(registry, name)
    
    if name in ("ToolTestCase", "mock_tool", "MockTool", "Invocation",
                "fixture", "MockAPI", "MockResponse", "mock_api", "mock_api_with_errors", "mock_api_slow"):
        from . import testing
        return getattr(testing, name)
    
    if name in ("ToolcaseSettings", "get_settings", "clear_settings_cache",
                "CacheSettings", "LoggingSettings", "RetrySettings", "HttpSettings",
                "TracingSettings", "RateLimitSettings"):
        from . import config
        return getattr(config, name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
