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

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..core import BaseTool
    from ..registry import ToolRegistry

# Type aliases for clarity
OpenAITool = dict[str, Any]
AnthropicTool = dict[str, Any]
GoogleTool = dict[str, Any]


def _get_json_schema(tool: BaseTool[BaseModel]) -> dict[str, Any]:
    """Extract JSON schema from tool's params_schema."""
    schema = tool.params_schema.model_json_schema()
    # Remove Pydantic metadata not needed by providers
    schema.pop("title", None)
    schema.pop("$defs", None)
    schema.pop("definitions", None)
    return schema


def _clean_properties(properties: dict[str, Any]) -> dict[str, Any]:
    """Clean property definitions for provider compatibility."""
    cleaned = {}
    for name, prop in properties.items():
        clean_prop = {k: v for k, v in prop.items() if k != "title"}
        cleaned[name] = clean_prop
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Format
# ─────────────────────────────────────────────────────────────────────────────

def tool_to_openai(
    tool: BaseTool[BaseModel],
    *,
    strict: bool = False,
) -> OpenAITool:
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
    schema = _get_json_schema(tool)
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    function_def: dict[str, Any] = {
        "name": tool.metadata.name,
        "description": tool.metadata.description,
        "parameters": {
            "type": "object",
            "properties": _clean_properties(properties),
            "required": required,
        },
    }
    
    if strict:
        function_def["strict"] = True
        function_def["parameters"]["additionalProperties"] = False
    
    return {"type": "function", "function": function_def}


def to_openai(
    registry: ToolRegistry,
    *,
    enabled_only: bool = True,
    strict: bool = False,
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
        for tool in registry
        if not enabled_only or tool.metadata.enabled
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
    schema = _get_json_schema(tool)
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    return {
        "name": tool.metadata.name,
        "description": tool.metadata.description,
        "input_schema": {
            "type": "object",
            "properties": _clean_properties(properties),
            "required": required,
        },
    }


def to_anthropic(
    registry: ToolRegistry,
    *,
    enabled_only: bool = True,
) -> list[AnthropicTool]:
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
        for tool in registry
        if not enabled_only or tool.metadata.enabled
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
    schema = _get_json_schema(tool)
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    return {
        "name": tool.metadata.name,
        "description": tool.metadata.description,
        "parameters": {
            "type": "object",
            "properties": _clean_properties(properties),
            "required": required,
        },
    }


def to_google(
    registry: ToolRegistry,
    *,
    enabled_only: bool = True,
) -> list[GoogleTool]:
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
        for tool in registry
        if not enabled_only or tool.metadata.enabled
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Universal Converter
# ─────────────────────────────────────────────────────────────────────────────

Provider = Literal["openai", "anthropic", "google"]


def to_provider(
    registry: ToolRegistry,
    provider: Provider,
    *,
    enabled_only: bool = True,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Convert tools to any supported provider format.
    
    Args:
        registry: Tool registry containing tools to convert
        provider: Target provider ("openai", "anthropic", "google")
        enabled_only: Only include enabled tools (default True)
        **kwargs: Provider-specific options (e.g., strict=True for OpenAI)
    
    Returns:
        List of provider-compatible tool definitions
    
    Raises:
        ValueError: If provider is not supported
    
    Example:
        >>> tools = to_provider(registry, "openai", strict=True)
    """
    converters = {
        "openai": to_openai,
        "anthropic": to_anthropic,
        "google": to_google,
    }
    
    if provider not in converters:
        supported = ", ".join(converters.keys())
        raise ValueError(f"Unsupported provider: {provider}. Supported: {supported}")
    
    converter = converters[provider]
    
    # Only OpenAI supports strict mode
    if provider == "openai":
        return converter(registry, enabled_only=enabled_only, **kwargs)
    return converter(registry, enabled_only=enabled_only)
