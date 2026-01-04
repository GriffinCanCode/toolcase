"""Multi-framework output format converters for toolcase.

Provides native format converters for major AI providers:
- OpenAI (function calling format)
- Anthropic (tool_use format)
- Google Gemini (function declarations)

Example:
    >>> from toolcase import get_registry
    >>> from toolcase.integrations.frontiers import to_openai, to_anthropic, to_google
    >>>
    >>> registry = get_registry()
    >>> openai_tools = to_openai(registry)
    >>> anthropic_tools = to_anthropic(registry)
    >>> gemini_tools = to_google(registry)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, TypedDict

from pydantic import BaseModel

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool
    from toolcase.foundation.registry import ToolRegistry


# ─────────────────────────────────────────────────────────────────────────────
# Typed Schemas for Provider Formats
# ─────────────────────────────────────────────────────────────────────────────

# JSON Schema property value type
JsonSchemaValue = bool | int | float | str | list["JsonSchemaValue"] | dict[str, "JsonSchemaValue"] | None


class JsonSchemaProperty(TypedDict, total=False):
    """JSON Schema property definition."""
    type: str
    description: str
    enum: list[str]
    default: JsonSchemaValue
    items: "JsonSchemaProperty"
    properties: dict[str, "JsonSchemaProperty"]
    required: list[str]
    format: str
    minimum: int | float
    maximum: int | float
    minLength: int
    maxLength: int
    pattern: str


class ParametersSchema(TypedDict, total=False):
    """JSON Schema for function parameters."""
    type: Literal["object"]
    properties: dict[str, JsonSchemaProperty]
    required: list[str]
    additionalProperties: bool


class OpenAIFunctionDef(TypedDict, total=False):
    """OpenAI function definition within a tool."""
    name: str
    description: str
    parameters: ParametersSchema
    strict: bool


class OpenAITool(TypedDict):
    """OpenAI function calling tool format."""
    type: Literal["function"]
    function: OpenAIFunctionDef


class AnthropicTool(TypedDict):
    """Anthropic tool_use format."""
    name: str
    description: str
    input_schema: ParametersSchema


class GoogleTool(TypedDict):
    """Google Gemini function declaration format."""
    name: str
    description: str
    parameters: ParametersSchema


# JSON schema types are inherently dynamic; use dict[str, ...] for internal processing
_JsonSchema = dict[str, JsonSchemaValue]
_PropertyMap = dict[str, JsonSchemaProperty]


def _extract_schema_parts(tool: BaseTool[BaseModel]) -> tuple[_PropertyMap, list[str]]:
    """Extract cleaned properties and required fields from tool schema."""
    schema: _JsonSchema = tool.params_schema.model_json_schema()
    for key in ("title", "$defs", "definitions"):
        schema.pop(key, None)
    
    raw_props = schema.get("properties", {})
    raw_required = schema.get("required", [])
    
    # Clean properties: remove 'title' key from each property dict
    properties: _PropertyMap = {
        name: {k: v for k, v in prop.items() if k != "title"}  # type: ignore[misc]
        for name, prop in (raw_props if isinstance(raw_props, dict) else {}).items()
        if isinstance(prop, dict)
    }
    required = [str(r) for r in raw_required] if isinstance(raw_required, list) else []
    
    return properties, required


def _build_params_schema(tool: BaseTool[BaseModel]) -> ParametersSchema:
    """Build a ParametersSchema from tool's schema."""
    properties, required = _extract_schema_parts(tool)
    return {"type": "object", "properties": properties, "required": required}


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Format
# ─────────────────────────────────────────────────────────────────────────────

def tool_to_openai(tool: BaseTool[BaseModel], *, strict: bool = False) -> OpenAITool:
    """Convert a toolcase tool to OpenAI function calling format.
    
    OpenAI tools format (Chat Completions API):
    ```json
    {
        "type": "function",
        "function": {
            "name": "tool_name",
            "description": "Description of what the tool does",
            "parameters": {
                "type": "object",
                "properties": {...},
                "required": [...]
            },
            "strict": false
        }
    }
    ```
    
    Args:
        tool: The toolcase tool instance to convert
        strict: Enable strict mode for structured outputs (ensures schema adherence)
    
    Returns:
        OpenAI-compatible tool definition dict
    
    Reference:
        https://platform.openai.com/docs/guides/function-calling
    """
    params = _build_params_schema(tool)
    function_def: OpenAIFunctionDef = {
        "name": tool.metadata.name,
        "description": tool.metadata.description,
        "parameters": params,
    }
    if strict:
        function_def["strict"] = True
        params["additionalProperties"] = False
    
    return {"type": "function", "function": function_def}


def to_openai(
    registry: ToolRegistry, *, enabled_only: bool = True, strict: bool = False,
) -> list[OpenAITool]:
    """Convert all tools in registry to OpenAI format.
    
    Args:
        registry: Tool registry containing tools to convert
        enabled_only: Only include enabled tools (default True)
        strict: Enable strict mode for all tools
    
    Returns:
        List of OpenAI-compatible tool definitions
    
    Example:
        >>> from toolcase import get_registry
        >>> from toolcase.integrations.formats import to_openai
        >>> 
        >>> openai_tools = to_openai(get_registry())
        >>> response = client.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=messages,
        ...     tools=openai_tools,
        ... )
    """
    return [
        tool_to_openai(tool, strict=strict)
        for tool in registry if not enabled_only or tool.metadata.enabled
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic Format
# ─────────────────────────────────────────────────────────────────────────────

def tool_to_anthropic(tool: BaseTool[BaseModel]) -> AnthropicTool:
    """Convert a toolcase tool to Anthropic tool_use format.
    
    Anthropic tools format (Messages API):
    ```json
    {
        "name": "tool_name",
        "description": "Description of what the tool does",
        "input_schema": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
    ```
    
    Args:
        tool: The toolcase tool instance to convert
    
    Returns:
        Anthropic-compatible tool definition dict
    
    Reference:
        https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use
    """
    return {
        "name": tool.metadata.name,
        "description": tool.metadata.description,
        "input_schema": _build_params_schema(tool),
    }


def to_anthropic(registry: ToolRegistry, *, enabled_only: bool = True) -> list[AnthropicTool]:
    """Convert all tools in registry to Anthropic format.
    
    Args:
        registry: Tool registry containing tools to convert
        enabled_only: Only include enabled tools (default True)
    
    Returns:
        List of Anthropic-compatible tool definitions
    
    Example:
        >>> from toolcase import get_registry
        >>> from toolcase.integrations.formats import to_anthropic
        >>> 
        >>> anthropic_tools = to_anthropic(get_registry())
        >>> response = client.messages.create(
        ...     model="claude-3-opus-20240229",
        ...     messages=messages,
        ...     tools=anthropic_tools,
        ... )
    """
    return [
        tool_to_anthropic(tool)
        for tool in registry if not enabled_only or tool.metadata.enabled
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Google Gemini Format
# ─────────────────────────────────────────────────────────────────────────────

def tool_to_google(tool: BaseTool[BaseModel]) -> GoogleTool:
    """Convert a toolcase tool to Google Gemini function declaration format.
    
    Gemini function declarations format:
    ```json
    {
        "name": "tool_name",
        "description": "Description of what the tool does",
        "parameters": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
    ```
    
    Args:
        tool: The toolcase tool instance to convert
    
    Returns:
        Google Gemini-compatible function declaration dict
    
    Reference:
        https://ai.google.dev/gemini-api/docs/function-calling
    """
    return {
        "name": tool.metadata.name,
        "description": tool.metadata.description,
        "parameters": _build_params_schema(tool),
    }


def to_google(registry: ToolRegistry, *, enabled_only: bool = True) -> list[GoogleTool]:
    """Convert all tools in registry to Google Gemini format.
    
    Args:
        registry: Tool registry containing tools to convert
        enabled_only: Only include enabled tools (default True)
    
    Returns:
        List of Google Gemini-compatible function declarations
    
    Example:
        >>> from toolcase import get_registry
        >>> from toolcase.integrations.formats import to_google
        >>> from google import genai
        >>> from google.genai import types
        >>> 
        >>> gemini_tools = to_google(get_registry())
        >>> tools = types.Tool(function_declarations=gemini_tools)
        >>> response = client.models.generate_content(
        ...     model="gemini-2.5-flash",
        ...     contents=prompt,
        ...     config=types.GenerateContentConfig(tools=[tools]),
        ... )
    """
    return [
        tool_to_google(tool)
        for tool in registry if not enabled_only or tool.metadata.enabled
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Universal Converter
# ─────────────────────────────────────────────────────────────────────────────

Provider = Literal["openai", "anthropic", "google"]
ProviderTool = OpenAITool | AnthropicTool | GoogleTool  # Union type for all provider tool formats


def to_provider(
    registry: ToolRegistry, provider: Provider, *, enabled_only: bool = True, strict: bool = False,
) -> Sequence[ProviderTool]:
    """Convert tools to any supported provider format.
    
    Args:
        registry: Tool registry containing tools to convert
        provider: Target provider ("openai", "anthropic", "google")
        enabled_only: Only include enabled tools (default True)
        strict: Enable strict mode (OpenAI only, ignored for other providers)
    
    Returns:
        Sequence of provider-compatible tool definitions
    
    Raises:
        ValueError: If provider is not supported
    
    Example:
        >>> tools = to_provider(registry, "openai", strict=True)
    """
    converters = {
        "openai": lambda: to_openai(registry, enabled_only=enabled_only, strict=strict),
        "anthropic": lambda: to_anthropic(registry, enabled_only=enabled_only),
        "google": lambda: to_google(registry, enabled_only=enabled_only),
    }
    if provider not in converters:
        raise ValueError(f"Unsupported provider: {provider}. Supported: {', '.join(converters)}")
    return converters[provider]()
