"""Tool discovery - meta-tool for listing available tools.

This built-in tool allows agents to discover what capabilities are available,
helping them decide which tool to use for a given task.
"""

from __future__ import annotations

from collections import defaultdict
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from toolcase.foundation.core import BaseTool, ToolMetadata
from toolcase.foundation.errors import Ok, ToolResult
from toolcase.foundation.registry import get_registry


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
        tools = registry.list_by_category(params.category) if params.category else registry.list_tools()
        
        if not tools:
            return Ok(f"No tools found in category '{params.category}'. Try without a filter." if params.category else "No tools are currently available.")
        
        formatter = self._format_brief if params.format == "brief" else self._format_detailed
        return Ok(formatter(tools, params.category))
    
    def _run(self, params: DiscoveryParams) -> str:
        """String-based fallback."""
        from toolcase.foundation.errors import result_to_string
        return result_to_string(self._run_result(params), self.metadata.name)
    
    def _format_brief(self, tools: list[ToolMetadata], category: str | None) -> str:
        """Brief format: grouped by category with one-line descriptions."""
        by_cat: dict[str, list[ToolMetadata]] = defaultdict(list)
        for tool in tools:
            by_cat[tool.category].append(tool)
        
        suffix = f" in '{category}'" if category else ""
        lines = [f"**Available Tools{suffix}**\n"]
        for cat, cat_tools in sorted(by_cat.items()):
            lines.append(f"**{cat.title()}:**")
            for t in cat_tools:
                flags = (" ⚡" * t.requires_api_key) + (" 📡" * t.streaming)
                desc = f"{t.description[:77]}..." if len(t.description) > 80 else t.description
                lines.append(f"- `{t.name}`{flags}: {desc}")
            lines.append("")
        
        lines.append("_⚡ = requires API key | 📡 = supports streaming_")
        return "\n".join(lines)
    
    def _format_detailed(self, tools: list[ToolMetadata], category: str | None) -> str:
        """Detailed format: full information for each tool."""
        suffix = f" in '{category}'" if category else ""
        lines = [f"**Available Tools{suffix}**\n"]
        
        for t in sorted(tools, key=lambda x: (x.category, x.name)):
            lines += [f"### {t.name}", f"**Category:** {t.category}", f"**Description:** {t.description}"]
            if t.requires_api_key:
                lines.append("**Note:** Requires API key")
            if t.streaming:
                lines.append("**Feature:** Supports progress streaming")
            lines.append("")
        
        return "\n".join(lines)
