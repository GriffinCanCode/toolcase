"""LangChain integration for toolcase.

Provides adapters to convert toolcase tools to LangChain StructuredTools
for use with LangChain agents.

Requires: pip install toolcase[langchain]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from ..errors import ErrorCode, ToolError, ToolException

if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool

    from ..core import BaseTool
    from ..registry import ToolRegistry


def to_langchain(tool: BaseTool[BaseModel]) -> StructuredTool:
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
            "LangChain integration requires langchain-core. "
            "Install with: pip install toolcase[langchain]"
        ) from e
    
    schema = tool.params_schema
    name = tool.metadata.name
    
    def _invoke(**kwargs: object) -> str:
        try:
            return tool.run(schema(**kwargs))  # type: ignore[arg-type, call-arg]
        except ValidationError as e:
            return ToolError.create(
                name, f"Invalid parameters: {e}",
                ErrorCode.INVALID_PARAMS, recoverable=False
            ).render()
        except ToolException as e:
            return e.error.render()
        except Exception as e:
            return ToolError.from_exception(name, e, "Execution failed").render()
    
    async def _ainvoke(**kwargs: object) -> str:
        try:
            return await tool.arun(schema(**kwargs))  # type: ignore[arg-type, call-arg]
        except ValidationError as e:
            return ToolError.create(
                name, f"Invalid parameters: {e}",
                ErrorCode.INVALID_PARAMS, recoverable=False
            ).render()
        except ToolException as e:
            return e.error.render()
        except Exception as e:
            return ToolError.from_exception(name, e, "Execution failed").render()
    
    return StructuredTool.from_function(
        func=_invoke,
        coroutine=_ainvoke,
        name=name,
        description=tool.metadata.description,
        args_schema=schema,
    )


def to_langchain_tools(
    registry: ToolRegistry,
    *,
    enabled_only: bool = True,
) -> list[StructuredTool]:
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
        for tool in registry
        if not enabled_only or tool.metadata.enabled
    ]
