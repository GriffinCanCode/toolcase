"""Central registry for tool discovery and management.

The registry provides:
- Tool registration and lookup by name
- Category-based filtering
- Formatted tool descriptions for LLM prompts
- Middleware pipeline for cross-cutting concerns
- Integration adapters (e.g., LangChain)
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ValidationError

from ..core import BaseTool, ToolMetadata
from ..errors import ErrorCode, ToolError, ToolException
from ..middleware import Context, Middleware, Next, compose

if TYPE_CHECKING:
    pass


T = TypeVar("T", bound=BaseTool[BaseModel])


class ToolRegistry:
    """Central registry for all available tools.
    
    Provides tool discovery, filtering, middleware pipeline, and format
    conversion for agent use.
    
    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(MyTool())
        >>> registry.use(LoggingMiddleware())
        >>> result = await registry.execute("my_tool", {"query": "test"})
    """
    
    __slots__ = ("_tools", "_middleware", "_chain")
    
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool[BaseModel]] = {}
        self._middleware: list[Middleware] = []
        self._chain: Next | None = None
    
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
    
    def __iter__(self) -> Iterator[BaseTool[BaseModel]]:
        return iter(self._tools.values())
    
    # ─────────────────────────────────────────────────────────────────
    # Middleware
    # ─────────────────────────────────────────────────────────────────
    
    def use(self, middleware: Middleware) -> None:
        """Add middleware to the execution pipeline.
        
        Middleware is applied in order: first added = outermost (runs first).
        Invalidates the compiled chain, forcing recompilation on next execute.
        
        Args:
            middleware: Middleware instance implementing the Middleware protocol
        
        Example:
            >>> registry.use(LoggingMiddleware())
            >>> registry.use(RateLimitMiddleware(max_calls=10, window_seconds=60))
        """
        self._middleware.append(middleware)
        self._chain = None  # Invalidate cached chain
    
    def _get_chain(self) -> Next:
        """Get or compile the middleware chain."""
        if self._chain is None:
            self._chain = compose(self._middleware)
        return self._chain
    
    async def execute(
        self,
        name: str,
        params: dict[str, object] | BaseModel,
        *,
        ctx: Context | None = None,
    ) -> str:
        """Execute a tool through the middleware pipeline.
        
        This is the primary execution method when middleware is configured.
        Validates params, builds context, and runs through the chain.
        Returns structured error responses for all failure modes.
        
        Args:
            name: Tool name to execute
            params: Parameters as dict or BaseModel
            ctx: Optional pre-built context (default: new Context)
        
        Returns:
            Tool result string (or formatted error string on failure)
        
        Example:
            >>> result = await registry.execute("search", {"query": "python"})
        """
        # Tool not found
        if name not in self._tools:
            return ToolError.create(
                name, f"Tool '{name}' not found in registry",
                ErrorCode.NOT_FOUND, recoverable=False
            ).render()
        
        tool = self._tools[name]
        
        # Validate params if dict
        try:
            validated = tool.params_schema(**params) if isinstance(params, dict) else params
        except ValidationError as e:
            return ToolError.create(
                name, f"Invalid parameters: {e}",
                ErrorCode.INVALID_PARAMS, recoverable=False
            ).render()
        
        # Build context
        context = ctx or Context()
        context["tool_name"] = name
        
        # Execute through chain with exception handling
        try:
            chain = self._get_chain()
            return await chain(tool, validated, context)
        except ToolException as e:
            return e.error.render()
        except Exception as e:
            return ToolError.from_exception(name, e, "Execution failed").render()
    
    def execute_sync(
        self,
        name: str,
        params: dict[str, object] | BaseModel,
        *,
        ctx: Context | None = None,
    ) -> str:
        """Synchronous wrapper for execute().
        
        For sync callers that need middleware support. Uses asyncio.run()
        internally, so cannot be called from within an async context.
        Returns structured error string on failure.
        """
        try:
            return asyncio.run(self.execute(name, params, ctx=ctx))
        except Exception as e:
            return ToolError.from_exception(name, e, "Sync execution failed").render()
    
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
        """Remove all registered tools and middleware."""
        self._tools.clear()
        self._middleware.clear()
        self._chain = None


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
