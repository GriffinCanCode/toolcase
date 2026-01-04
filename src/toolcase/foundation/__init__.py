"""Foundation - Core building blocks for toolcase.

Contains: core abstractions, error handling, DI, registry, testing, formats, config, effects.
"""

from __future__ import annotations

__all__ = [
    # Core
    "BaseTool", "ToolProtocol", "AnyTool", "ToolMetadata", "EmptyParams", "tool",
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
    # Events
    "Signal", "SignalHandler", "one_shot",
    # Effects
    "Effect", "EffectSet", "EffectHandler", "EffectContext",
    "effects", "declare_effects", "get_effects", "has_effects", "get_handler",
    "EffectHandlerRegistry", "EffectScope", "effect_scope", "test_effects",
    "PureHandler", "InMemoryDB", "RecordingHTTP", "InMemoryFS", "NoOpCache",
    "FrozenTime", "SeededRandom", "CollectingLogger",
    "EffectViolation", "MissingEffectHandler", "UndeclaredEffect", "verify_effects",
    # Testing
    "ToolTestCase", "mock_tool", "MockTool", "Invocation",
    "fixture", "MockAPI", "MockResponse", "mock_api", "mock_api_with_errors", "mock_api_slow",
    # Config
    "ToolcaseSettings", "get_settings", "clear_settings_cache",
    "CacheSettings", "LoggingSettings", "RetrySettings", "HttpSettings", "TracingSettings", "RateLimitSettings",
    # Fast Validation (msgspec, 10-100x faster)
    "FastStruct", "fast", "fast_frozen",
    "validate", "validate_or_none", "validate_many", "FastValidator",
    "to_pydantic", "from_pydantic", "pydantic_to_fast",
]


# Module cache for lazy imports (prevents recursion)
_module_cache: dict[str, object] = {}


def __getattr__(name: str):
    """Lazy imports to avoid circular dependencies."""
    import importlib
    
    if name in ("BaseTool", "ToolProtocol", "AnyTool", "ToolMetadata", "EmptyParams", "tool",
                "FunctionTool", "StreamingFunctionTool", "ResultStreamingFunctionTool",
                "set_injected_deps", "clear_injected_deps"):
        if "core" not in _module_cache:
            _module_cache["core"] = importlib.import_module(".core", __name__)
        return getattr(_module_cache["core"], name)
    
    if name in ("ErrorCode", "ToolError", "ToolException", "classify_exception",
                "Result", "Ok", "Err", "ResultT", "try_fn",
                "ToolResult", "tool_result", "ok_result", "try_tool_operation", "try_tool_operation_async",
                "batch_results", "from_tool_error", "to_tool_error", "result_to_string", "string_to_result",
                "ErrorContext", "ErrorTrace", "context", "trace", "trace_from_exc",
                "sequence", "traverse", "collect_results"):
        if "errors" not in _module_cache:
            _module_cache["errors"] = importlib.import_module(".errors", __name__)
        return getattr(_module_cache["errors"], name)
    
    if name in ("Container", "DIResult", "Disposable", "Factory", "Provider", "Scope", "ScopedContext"):
        if "di" not in _module_cache:
            _module_cache["di"] = importlib.import_module(".di", __name__)
        return getattr(_module_cache["di"], name)
    
    if name in ("ToolRegistry", "get_registry", "set_registry", "reset_registry"):
        if "registry" not in _module_cache:
            _module_cache["registry"] = importlib.import_module(".registry", __name__)
        return getattr(_module_cache["registry"], name)
    
    if name in ("Signal", "SignalHandler", "one_shot"):
        if "events" not in _module_cache:
            _module_cache["events"] = importlib.import_module(".events", __name__)
        return getattr(_module_cache["events"], name)
    
    if name in ("Effect", "EffectSet", "EffectHandler", "EffectContext",
                "effects", "declare_effects", "get_effects", "has_effects", "get_handler",
                "EffectHandlerRegistry", "EffectScope", "effect_scope", "test_effects",
                "PureHandler", "InMemoryDB", "RecordingHTTP", "InMemoryFS", "NoOpCache",
                "FrozenTime", "SeededRandom", "CollectingLogger",
                "EffectViolation", "MissingEffectHandler", "UndeclaredEffect", "verify_effects"):
        if "effects" not in _module_cache:
            _module_cache["effects"] = importlib.import_module(".effects", __name__)
        return getattr(_module_cache["effects"], name)
    
    if name in ("ToolTestCase", "mock_tool", "MockTool", "Invocation",
                "fixture", "MockAPI", "MockResponse", "mock_api", "mock_api_with_errors", "mock_api_slow"):
        if "testing" not in _module_cache:
            _module_cache["testing"] = importlib.import_module(".testing", __name__)
        return getattr(_module_cache["testing"], name)
    
    if name in ("ToolcaseSettings", "get_settings", "clear_settings_cache",
                "CacheSettings", "LoggingSettings", "RetrySettings", "HttpSettings",
                "TracingSettings", "RateLimitSettings"):
        if "config" not in _module_cache:
            _module_cache["config"] = importlib.import_module(".config", __name__)
        return getattr(_module_cache["config"], name)
    
    if name in ("FastStruct", "fast", "fast_frozen",
                "validate", "validate_or_none", "validate_many", "FastValidator",
                "to_pydantic", "from_pydantic", "pydantic_to_fast"):
        if "fast" not in _module_cache:
            _module_cache["fast"] = importlib.import_module(".fast", __name__)
        return getattr(_module_cache["fast"], name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
