"""Multi-framework output format converters for toolcase.

Convenience re-export from toolcase.integrations.frontiers for simpler imports.

Example:
    >>> from toolcase.formats import to_openai, to_anthropic, to_google
    >>>
    >>> registry = get_registry()
    >>> openai_tools = to_openai(registry)
    >>> anthropic_tools = to_anthropic(registry)
    >>> gemini_tools = to_google(registry)
"""

from .integrations.frontiers import (
    to_anthropic,
    to_google,
    to_openai,
    to_provider,
    tool_to_anthropic,
    tool_to_google,
    tool_to_openai,
)

__all__ = [
    # Registry converters
    "to_openai",
    "to_anthropic",
    "to_google",
    "to_provider",
    # Single tool converters
    "tool_to_openai",
    "tool_to_anthropic",
    "tool_to_google",
]
