"""Core tool abstractions and decorators.

This module provides the foundation for building type-safe, extensible tools:
- BaseTool: Abstract base class for all tools
- ToolMetadata: Tool metadata and capabilities
- ToolCapabilities: Advertised capabilities for intelligent scheduling
- EmptyParams: Default parameter schema for parameterless tools
- @tool decorator: Convert functions to tools
- FunctionTool: Standard function wrapper
- StreamingFunctionTool: Progress streaming (ToolProgress events)
- ResultStreamingFunctionTool: Result streaming (string chunks for LLM output)
- Dependency injection helpers
"""

from .base import BaseTool, EmptyParams, ToolCapabilities, ToolMetadata
from .decorator import (
    FunctionTool,
    ResultStreamingFunctionTool,
    StreamingFunctionTool,
    clear_injected_deps,
    get_injected_deps,
    set_injected_deps,
    tool,
)

__all__ = [
    "BaseTool",
    "ToolMetadata",
    "ToolCapabilities",
    "EmptyParams",
    "tool",
    "FunctionTool",
    "StreamingFunctionTool",
    "ResultStreamingFunctionTool",
    "set_injected_deps",
    "get_injected_deps",
    "clear_injected_deps",
]
