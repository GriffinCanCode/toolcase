"""Core tool abstractions and decorators.

This module provides the foundation for building type-safe, extensible tools:
- BaseTool: Abstract base class for all tools (recommended - includes caching, retry, batch, etc.)
- ToolProtocol: Duck typing interface for third-party tools without inheritance
- AnyTool: Type alias accepting either BaseTool or ToolProtocol-conforming objects
- ToolMetadata: Tool metadata and capabilities
- ToolCapabilities: Advertised capabilities for intelligent scheduling
- EmptyParams: Default parameter schema for parameterless tools
- @tool decorator: Convert functions to tools
- FunctionTool: Standard function wrapper
- StreamingFunctionTool: Progress streaming (ToolProgress events)
- ResultStreamingFunctionTool: Result streaming (string chunks for LLM output)
- Dependency injection helpers
"""

from .base import AnyTool, BaseTool, EmptyParams, ToolCapabilities, ToolMetadata, ToolProtocol
from .decorator import (
    FunctionTool,
    InjectedDeps,
    ResultStreamingFunctionTool,
    StreamingFunctionTool,
    clear_injected_deps,
    get_injected_deps,
    set_injected_deps,
    tool,
)

__all__ = [
    "BaseTool",
    "ToolProtocol",
    "AnyTool",
    "ToolMetadata",
    "ToolCapabilities",
    "EmptyParams",
    "tool",
    "FunctionTool",
    "StreamingFunctionTool",
    "ResultStreamingFunctionTool",
    "InjectedDeps",
    "set_injected_deps",
    "get_injected_deps",
    "clear_injected_deps",
]
