"""Core tool abstractions and decorators.

This module provides the foundation for building type-safe, extensible tools:
- BaseTool: Abstract base class for all tools
- ToolMetadata: Tool metadata and capabilities
- EmptyParams: Default parameter schema for parameterless tools
- @tool decorator: Convert functions to tools
- FunctionTool/StreamingFunctionTool: Decorator implementations
"""

from .base import BaseTool, EmptyParams, ToolMetadata
from .decorator import FunctionTool, StreamingFunctionTool, tool

__all__ = [
    "BaseTool",
    "ToolMetadata",
    "EmptyParams",
    "tool",
    "FunctionTool",
    "StreamingFunctionTool",
]
