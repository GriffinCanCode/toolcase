"""Tool discovery - meta-tool for listing available tools.

This built-in tool allows agents to discover what capabilities are available,
helping them decide which tool to use for a given task.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from ..core import BaseTool, ToolMetadata
from ..monads import Ok, ToolResult
from ..registry import get_registry


class DiscoveryParams(BaseModel):
    """Parameters for tool discovery."""
    
    category: str | None = Field(
        default=None,
        description="Filter by category (e.g., 'search', 'memory'). Omit to list all.",
    )
    format: Literal["brief", "detailed"] = Field(
        default="brief",
        description="Output format: 'brief' for names only, 'detailed' for full info",
    )


class DiscoveryTool(BaseTool[DiscoveryParams]):
    """Meta-tool that lists all available tools in the registry.
    
    Helps agents understand what capabilities are available and make
    informed decisions about which tool to use.
    """
    
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="discover_tools",
        description=(
            "List available tools and their capabilities. "
            "Use to discover what tools can help accomplish a task. "
            "Can filter by category."
        ),
        category="meta",
        requires_api_key=False,
        enabled=True,
        streaming=False,
    )
    params_schema: ClassVar[type[DiscoveryParams]] = DiscoveryParams
    
    # Don't cache discovery - always show current state
    cache_enabled: ClassVar[bool] = False
    
    def _run_result(self, params: DiscoveryParams) -> ToolResult:
        """Result-based implementation."""
        registry = get_registry()
        
        # Get tools, optionally filtered by category
        tools = registry.list_by_category(params.category) if params.category else registry.list_tools()
        
        if not tools:
            msg = f"No tools found in category '{params.category}'. Try without a filter." if params.category else "No tools are currently available."
            return Ok(msg)
        
        formatted = self._format_brief(tools, params.category) if params.format == "brief" else self._format_detailed(tools, params.category)
        return Ok(formatted)
    
    def _run(self, params: DiscoveryParams) -> str:
        """String-based fallback."""
        from ..monads.tool import result_to_string
        return result_to_string(self._run_result(params), self.metadata.name)
    
    def _format_brief(self, tools: list[ToolMetadata], category: str | None) -> str:
        """Brief format: grouped by category with one-line descriptions."""
        cat_suffix = f" in '{category}'" if category else ""
        header = f"**Available Tools{cat_suffix}**\n"
        
        # Group by category
        by_cat: dict[str, list[ToolMetadata]] = {}
        for tool in tools:
            by_cat.setdefault(tool.category, []).append(tool)
        
        lines = [header]
        for cat, cat_tools in sorted(by_cat.items()):
            lines.append(f"**{cat.title()}:**")
            for t in cat_tools:
                flags = ""
                if t.requires_api_key:
                    flags += " ⚡"
                if t.streaming:
                    flags += " 📡"
                # Truncate description to 80 chars
                desc = t.description[:77] + "..." if len(t.description) > 80 else t.description
                lines.append(f"- `{t.name}`{flags}: {desc}")
            lines.append("")
        
        lines.append("_⚡ = requires API key | 📡 = supports streaming_")
        return "\n".join(lines)
    
    def _format_detailed(self, tools: list[ToolMetadata], category: str | None) -> str:
        """Detailed format: full information for each tool."""
        cat_suffix = f" in '{category}'" if category else ""
        header = f"**Available Tools{cat_suffix}**\n"
        lines = [header]
        
        for t in sorted(tools, key=lambda x: (x.category, x.name)):
            lines.append(f"### {t.name}")
            lines.append(f"**Category:** {t.category}")
            lines.append(f"**Description:** {t.description}")
            if t.requires_api_key:
                lines.append("**Note:** Requires API key")
            if t.streaming:
                lines.append("**Feature:** Supports progress streaming")
            lines.append("")
        
        return "\n".join(lines)
