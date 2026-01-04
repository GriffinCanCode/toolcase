"""Central registry for tool discovery and management.

The registry provides:
- Tool registration and lookup by name
- Category-based filtering
- Formatted tool descriptions for LLM prompts
- Integration adapters (e.g., LangChain)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from .core import BaseTool, ToolMetadata

if TYPE_CHECKING:
    pass


T = TypeVar("T", bound=BaseTool[BaseModel])


class ToolRegistry:
    """Central registry for all available tools.
    
    Provides tool discovery, filtering, and format conversion for agent use.
    
    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(MyTool())
        >>> registry.get("my_tool")
        <MyTool ...>
        >>> registry.list_tools()
        [ToolMetadata(name='my_tool', ...)]
    """
    
    __slots__ = ("_tools",)
    
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool[BaseModel]] = {}
    
    def register(self, tool: BaseTool[BaseModel]) -> None:
        """Register a tool instance with validation."""
        name = tool.metadata.name
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered. Use unregister() first.")
        if len(tool.metadata.description) < 10:
            raise ValueError(f"Tool '{name}' description too short for LLM selection.")
        self._tools[name] = tool
    
    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if found."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def get(self, name: str) -> BaseTool[BaseModel] | None:
        """Get tool by name."""
        return self._tools.get(name)
    
    def __getitem__(self, name: str) -> BaseTool[BaseModel]:
        """Get tool by name, raises KeyError if not found."""
        return self._tools[name]
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __iter__(self):
        return iter(self._tools.values())
    
    # ─────────────────────────────────────────────────────────────────
    # Querying
    # ─────────────────────────────────────────────────────────────────
    
    def list_tools(self, *, enabled_only: bool = True) -> list[ToolMetadata]:
        """List metadata for all registered tools."""
        return [
            t.metadata for t in self._tools.values()
            if not enabled_only or t.metadata.enabled
        ]
    
    def list_by_category(self, category: str, *, enabled_only: bool = True) -> list[ToolMetadata]:
        """List tools filtered by category."""
        return [
            t.metadata for t in self._tools.values()
            if t.metadata.category == category and (not enabled_only or t.metadata.enabled)
        ]
    
    def categories(self) -> set[str]:
        """Get all unique categories."""
        return {t.metadata.category for t in self._tools.values()}
    
    # ─────────────────────────────────────────────────────────────────
    # Formatting
    # ─────────────────────────────────────────────────────────────────
    
    def describe(self, *, enabled_only: bool = True) -> str:
        """Get formatted descriptions of all tools for prompts."""
        lines = []
        for tool in self._tools.values():
            if enabled_only and not tool.metadata.enabled:
                continue
            m = tool.metadata
            flags = []
            if m.requires_api_key:
                flags.append("⚡")
            if m.streaming:
                flags.append("📡")
            flag_str = " ".join(flags)
            lines.append(f"- **{m.name}** ({m.category}){' ' + flag_str if flag_str else ''}: {m.description}")
        
        if lines:
            lines.append("\n_⚡ = requires API key | 📡 = supports streaming_")
        
        return "\n".join(lines)
    
    # ─────────────────────────────────────────────────────────────────
    # Bulk Operations
    # ─────────────────────────────────────────────────────────────────
    
    def register_all(self, *tools: BaseTool[BaseModel]) -> None:
        """Register multiple tools at once."""
        for tool in tools:
            self.register(tool)
    
    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Global Registry
# ─────────────────────────────────────────────────────────────────────────────

_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def set_registry(registry: ToolRegistry) -> None:
    """Replace the global registry."""
    global _registry
    _registry = registry


def reset_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _registry
    if _registry is not None:
        _registry.clear()
    _registry = None
