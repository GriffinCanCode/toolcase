"""LangChain integration for toolcase.

Provides adapters to convert toolcase tools to LangChain StructuredTools
for use with LangChain agents.

Requires: pip install toolcase[langchain]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from toolcase.foundation.errors import ErrorCode, ToolError, ToolException, format_validation_error

if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool

    from toolcase.foundation.core import AnyTool
    from toolcase.foundation.registry import ToolRegistry


def _wrap_error(name: str, exc: Exception) -> str:
    """Convert exception to rendered ToolError string."""
    if isinstance(exc, ValidationError):
        return ToolError.create(
            name, format_validation_error(exc, tool_name=name),
            ErrorCode.INVALID_PARAMS, recoverable=False
        ).render()
    if isinstance(exc, ToolException):
        return exc.error.render()
    return ToolError.from_exception(name, exc, "Execution failed").render()


def to_langchain(tool: AnyTool) -> StructuredTool:
    """Convert a toolcase tool to a LangChain StructuredTool.
    
    Wraps tool execution with proper error handling, returning
    structured error responses instead of raising exceptions.
    
    Args:
        tool: The toolcase tool instance to convert
    
    Returns:
        A LangChain StructuredTool that wraps the tool
    
    Example:
        >>> from toolcase.integrations import to_langchain
        >>> lc_tool = to_langchain(my_tool)
        >>> agent = create_tool_calling_agent(llm, [lc_tool], prompt)
    
    Raises:
        ImportError: If langchain-core is not installed
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as e:
        raise ImportError(
            "LangChain integration requires langchain-core. Install with: pip install toolcase[langchain]"
        ) from e
    
    schema, name = tool.params_schema, tool.metadata.name
    
    def _invoke(**kwargs: object) -> str:
        try:
            return tool.run(schema(**kwargs))  # type: ignore[arg-type, call-arg]
        except Exception as e:
            return _wrap_error(name, e)
    
    async def _ainvoke(**kwargs: object) -> str:
        try:
            return await tool.arun(schema(**kwargs))  # type: ignore[arg-type, call-arg]
        except Exception as e:
            return _wrap_error(name, e)
    
    return StructuredTool.from_function(
        func=_invoke, coroutine=_ainvoke, name=name,
        description=tool.metadata.description, args_schema=schema,
    )


def to_langchain_tools(registry: ToolRegistry, *, enabled_only: bool = True) -> list[StructuredTool]:
    """Convert all tools in a registry to LangChain format.
    
    Args:
        registry: The tool registry containing tools to convert
        enabled_only: Only include enabled tools (default True)
    
    Returns:
        List of LangChain StructuredTools
    
    Example:
        >>> from toolcase import get_registry
        >>> from toolcase.integrations import to_langchain_tools
        >>> lc_tools = to_langchain_tools(get_registry())
        >>> executor = AgentExecutor(agent=agent, tools=lc_tools)
    """
    return [
        to_langchain(tool)
        for tool in registry if not enabled_only or tool.metadata.enabled
    ]
