"""Bridge between toolcase BaseTool and MCP tool primitives.

Converts toolcase tools to MCP-compatible format, enabling seamless
registration with any MCP server implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..core import BaseTool
    from ..registry import ToolRegistry


def tool_to_handler(tool: BaseTool[BaseModel]) -> Callable[..., str]:
    """Convert BaseTool to an MCP-compatible handler function.
    
    Creates a callable that:
    - Accepts **kwargs matching the tool's params_schema
    - Validates inputs via Pydantic
    - Returns string result
    
    Args:
        tool: Toolcase tool instance to wrap
    
    Returns:
        Async function suitable for MCP tool registration
    """
    schema = tool.params_schema
    
    async def handler(**kwargs: object) -> str:
        params = schema(**kwargs)
        return await tool.arun(params)  # type: ignore[arg-type]
    
    # Preserve metadata for introspection
    handler.__name__ = tool.metadata.name
    handler.__doc__ = tool.metadata.description
    handler.__annotations__ = _extract_annotations(schema)
    
    return handler


def _extract_annotations(schema: type[BaseModel]) -> dict[str, type]:
    """Extract field annotations from Pydantic model for function signature."""
    return {
        name: info.annotation or str
        for name, info in schema.model_fields.items()
    }


def get_tool_schema(tool: BaseTool[BaseModel]) -> dict[str, object]:
    """Extract JSON schema from tool's params for MCP registration.
    
    Returns:
        JSON Schema dict compatible with MCP tool definition
    """
    schema = tool.params_schema.model_json_schema()
    # Strip Pydantic-specific metadata
    schema.pop("title", None)
    schema.pop("$defs", None)
    schema.pop("definitions", None)
    return schema


def get_tool_properties(tool: BaseTool[BaseModel]) -> dict[str, dict[str, object]]:
    """Extract cleaned property definitions for MCP."""
    schema = tool.params_schema.model_json_schema()
    properties = schema.get("properties", {})
    return {
        name: {k: v for k, v in prop.items() if k != "title"}
        for name, prop in properties.items()
    }


def get_required_params(tool: BaseTool[BaseModel]) -> list[str]:
    """Get list of required parameter names."""
    return tool.params_schema.model_json_schema().get("required", [])


def registry_to_handlers(
    registry: ToolRegistry,
    *,
    enabled_only: bool = True,
) -> dict[str, Callable[..., str]]:
    """Convert all registry tools to MCP handlers.
    
    Args:
        registry: Tool registry to convert
        enabled_only: Only include enabled tools
    
    Returns:
        Dict mapping tool names to handler functions
    """
    return {
        tool.metadata.name: tool_to_handler(tool)
        for tool in registry
        if not enabled_only or tool.metadata.enabled
    }
