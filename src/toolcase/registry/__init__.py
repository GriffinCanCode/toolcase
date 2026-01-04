"""Tool registry for discovery and management.

The registry provides:
- Tool registration and lookup by name
- Category-based filtering
- Formatted tool descriptions for LLM prompts
- Global registry singleton
"""

from .registry import ToolRegistry, get_registry, reset_registry, set_registry

__all__ = [
    "ToolRegistry",
    "get_registry",
    "set_registry",
    "reset_registry",
]
