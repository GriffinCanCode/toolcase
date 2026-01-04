"""Bridge between toolcase BaseTool and MCP tool primitives.

Converts toolcase tools to MCP-compatible format, enabling seamless
registration with any MCP server implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from pydantic import BaseModel, ValidationError

from toolcase.foundation.errors import ErrorCode, JsonDict, ToolError, ToolException, format_validation_error

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool
    from toolcase.foundation.registry import ToolRegistry


def tool_to_handler(tool: BaseTool[BaseModel]) -> Callable[..., str]:
    """Convert BaseTool to an MCP-compatible handler function.
    
    Creates a callable that:
    - Accepts **kwargs matching the tool's params_schema
    - Validates inputs via Pydantic
    - Returns string result (or structured error on failure)
    
    Args:
        tool: Toolcase tool instance to wrap
    
    Returns:
        Async function suitable for MCP tool registration
    """
    schema, meta = tool.params_schema, tool.metadata
    
    async def handler(**kwargs: object) -> str:
        try:
            params = schema(**kwargs)
        except ValidationError as e:
            return ToolError.create(
                meta.name, format_validation_error(e, tool_name=meta.name),
                ErrorCode.INVALID_PARAMS, recoverable=False
            ).render()
        try:
            return await tool.arun(params)  # type: ignore[arg-type]
        except ToolException as e:
            return e.error.render()
        except Exception as e:
            return ToolError.from_exception(meta.name, e, "Execution failed").render()
    
    # Preserve metadata for introspection
    handler.__name__, handler.__doc__ = meta.name, meta.description
    handler.__annotations__ = {n: i.annotation or str for n, i in schema.model_fields.items()}
    return handler


def get_tool_schema(tool: BaseTool[BaseModel]) -> JsonDict:
    """Extract JSON schema from tool's params for MCP registration."""
    schema = tool.params_schema.model_json_schema()
    # Strip Pydantic-specific metadata
    return {k: v for k, v in schema.items() if k not in {"title", "$defs", "definitions"}}


def get_tool_properties(tool: BaseTool[BaseModel]) -> dict[str, JsonDict]:
    """Extract cleaned property definitions for MCP."""
    props = tool.params_schema.model_json_schema().get("properties", {})
    return {n: {k: v for k, v in p.items() if k != "title"} for n, p in props.items()}


def get_required_params(tool: BaseTool[BaseModel]) -> list[str]:
    """Get list of required parameter names."""
    return tool.params_schema.model_json_schema().get("required", [])


def registry_to_handlers(registry: ToolRegistry, *, enabled_only: bool = True) -> dict[str, Callable[..., str]]:
    """Convert all registry tools to MCP handlers.
    
    Args:
        registry: Tool registry to convert
        enabled_only: Only include enabled tools
    
    Returns:
        Dict mapping tool names to handler functions
    """
    return {t.metadata.name: tool_to_handler(t) for t in registry if not enabled_only or t.metadata.enabled}
