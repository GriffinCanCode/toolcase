"""Tool discovery - meta-tool for listing and searching available tools.

Extended discovery capabilities:
- Category filtering (original)
- Capability-based search via tags
- Schema-based matching (find tools accepting specific parameter types)
- Usage statistics display

Example:
    >>> # Find tools with "search" capability
    >>> result = await registry.execute("discover_tools", {
    ...     "capabilities": ["search"],
    ...     "format": "detailed"
    ... })
    >>> 
    >>> # Find tools accepting a "query" string parameter
    >>> result = await registry.execute("discover_tools", {
    ...     "param_name": "query",
    ...     "param_type": "string"
    ... })
    >>> 
    >>> # Get usage statistics
    >>> result = await registry.execute("discover_tools", {"include_stats": True})
"""

from __future__ import annotations

from collections import defaultdict
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from toolcase.foundation.core import BaseTool, ToolMetadata
from toolcase.foundation.registry import get_registry

from toolcase.tools.core.query import MatchMode, QueryResult, SchemaPattern, ToolQuery
from toolcase.tools.core.stats import format_stats, get_stats


class DiscoveryParams(BaseModel):
    """Parameters for tool discovery.
    
    Supports multiple discovery modes that can be combined:
    - Category filtering (original)
    - Capability-based search via tags
    - Schema-based parameter matching
    - Usage statistics
    """
    
    # Original parameters
    category: str | None = Field(
        default=None,
        description="Filter by category (e.g., 'search', 'memory'). Omit to list all.",
    )
    format: Literal["brief", "detailed"] = Field(
        default="brief",
        description="Output format: 'brief' for names only, 'detailed' for full info",
    )
    
    # Capability-based search
    capabilities: list[str] | None = Field(
        default=None,
        description="Filter by capability tags (e.g., ['search', 'web']). Tools must have at least one matching tag.",
    )
    capability_match: Literal["any", "all"] = Field(
        default="any",
        description="How to match capabilities: 'any' (OR) or 'all' (AND)",
    )
    
    # Schema-based matching
    param_name: str | None = Field(
        default=None,
        description="Find tools with a parameter matching this name (e.g., 'query')",
    )
    param_type: Literal["string", "integer", "number", "boolean", "array", "object"] | None = Field(
        default=None,
        description="Find tools with a parameter of this type",
    )
    param_required: bool | None = Field(
        default=None,
        description="Filter to required (True) or optional (False) parameters",
    )
    
    # Statistics
    include_stats: bool = Field(
        default=False,
        description="Include usage statistics in output",
    )
    stats_only: bool = Field(
        default=False,
        description="Return only usage statistics (ignores other filters)",
    )


class DiscoveryTool(BaseTool[DiscoveryParams]):
    """Meta-tool that lists and searches available tools in the registry.
    
    Extended capabilities:
    - Category filtering: Filter by tool category
    - Capability search: Find tools by tags/capabilities
    - Schema matching: Find tools that accept specific parameter types
    - Usage statistics: View call counts, success rates, latencies
    
    Helps agents understand what capabilities are available and make
    informed decisions about which tool to use.
    """
    
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="discover_tools",
        description=(
            "List and search available tools. "
            "Filter by category, capabilities, or parameter types. "
            "Use to discover what tools can help accomplish a task."
        ),
        category="meta",
        requires_api_key=False,
        enabled=True,
        streaming=False,
        tags=frozenset({"discovery", "meta", "search"}),
    )
    params_schema: ClassVar[type[DiscoveryParams]] = DiscoveryParams
    cache_enabled: ClassVar[bool] = False  # Always show current state
    
    async def _async_run(self, params: DiscoveryParams) -> str:
        """Execute discovery with optional query filters."""
        # Stats-only mode
        if params.stats_only:
            return format_stats(get_stats(), top_n=10)
        
        registry = get_registry()
        
        # Build query if advanced filters specified
        if self._has_advanced_filters(params):
            results = self._execute_query(params)
            output = self._format_query_results(results, params)
        else:
            # Original behavior: simple category filter
            tools = registry.list_by_category(params.category) if params.category else registry.list_tools()
            if not tools:
                return self._no_tools_message(params.category)
            formatter = self._format_brief if params.format == "brief" else self._format_detailed
            output = formatter(tools, params.category)
        
        # Append stats if requested
        if params.include_stats:
            stats_output = format_stats(get_stats(), top_n=5, include_tools=True)
            output = f"{output}\n\n---\n\n{stats_output}"
        
        return output
    
    def _has_advanced_filters(self, params: DiscoveryParams) -> bool:
        """Check if advanced query filters are specified."""
        return bool(params.capabilities or params.param_name or params.param_type)
    
    def _execute_query(self, params: DiscoveryParams) -> list[QueryResult]:
        """Execute query with advanced filters."""
        registry = get_registry()
        query = ToolQuery(registry)
        
        # Category filter
        if params.category:
            query.by_category(params.category)
        
        # Capability filter
        if params.capabilities:
            mode = MatchMode.ALL if params.capability_match == "all" else MatchMode.ANY
            query.by_capability(params.capabilities, mode)
        
        # Schema filter
        if params.param_name or params.param_type:
            query.by_schema(SchemaPattern(
                name=params.param_name,
                type=params.param_type,
                required=params.param_required,
            ))
        
        return query.execute()
    
    def _format_query_results(self, results: list[QueryResult], params: DiscoveryParams) -> str:
        """Format query results."""
        if not results:
            return self._no_matches_message(params)
        
        if params.format == "brief":
            return self._format_results_brief(results, params)
        return self._format_results_detailed(results, params)
    
    def _format_results_brief(self, results: list[QueryResult], params: DiscoveryParams) -> str:
        """Brief format for query results."""
        by_cat: dict[str, list[QueryResult]] = defaultdict(list)
        for r in results:
            by_cat[r.metadata.category].append(r)
        
        filters = self._describe_filters(params)
        lines = [f"**Matching Tools** ({filters})\n"]
        
        for cat, cat_results in sorted(by_cat.items()):
            lines.append(f"**{cat.title()}:**")
            for r in cat_results:
                m = r.metadata
                flags = (" ⚡" * m.requires_api_key) + (" 📡" * m.streaming)
                score_indicator = f" ({r.score:.0%})" if r.score < 1.0 else ""
                desc = f"{m.description[:70]}..." if len(m.description) > 73 else m.description
                lines.append(f"- `{m.name}`{flags}{score_indicator}: {desc}")
            lines.append("")
        
        lines.append(f"_Found {len(results)} matching tool(s)_")
        return "\n".join(lines)
    
    def _format_results_detailed(self, results: list[QueryResult], params: DiscoveryParams) -> str:
        """Detailed format for query results."""
        filters = self._describe_filters(params)
        lines = [f"**Matching Tools** ({filters})\n"]
        
        for r in sorted(results, key=lambda x: (x.metadata.category, x.metadata.name)):
            m = r.metadata
            lines += [f"### {m.name}", f"**Category:** {m.category}", f"**Description:** {m.description}"]
            
            if r.score < 1.0:
                lines.append(f"**Match Score:** {r.score:.0%}")
            if r.matched_fields:
                lines.append(f"**Matched Parameters:** {', '.join(r.matched_fields)}")
            if r.matched_tags:
                lines.append(f"**Matched Capabilities:** {', '.join(r.matched_tags)}")
            if m.tags:
                lines.append(f"**Tags:** {', '.join(sorted(m.tags))}")
            if m.requires_api_key:
                lines.append("**Note:** Requires API key")
            if m.streaming:
                lines.append("**Feature:** Supports streaming")
            lines.append("")
        
        lines.append(f"_Found {len(results)} matching tool(s)_")
        return "\n".join(lines)
    
    def _describe_filters(self, params: DiscoveryParams) -> str:
        """Describe active filters for output header."""
        parts: list[str] = []
        if params.category:
            parts.append(f"category={params.category}")
        if params.capabilities:
            join = " AND " if params.capability_match == "all" else " OR "
            parts.append(f"capabilities={join.join(params.capabilities)}")
        if params.param_name:
            parts.append(f"param={params.param_name}")
        if params.param_type:
            parts.append(f"type={params.param_type}")
        return ", ".join(parts) if parts else "all"
    
    def _no_matches_message(self, params: DiscoveryParams) -> str:
        """Message when no tools match query."""
        parts: list[str] = []
        if params.capabilities:
            parts.append(f"capabilities={params.capabilities}")
        if params.param_name:
            parts.append(f"param_name='{params.param_name}'")
        if params.param_type:
            parts.append(f"param_type='{params.param_type}'")
        filters = ", ".join(parts) if parts else "specified criteria"
        return f"No tools found matching {filters}. Try broadening your search."
    
    def _no_tools_message(self, category: str | None) -> str:
        """Message when no tools found."""
        if category:
            return f"No tools found in category '{category}'. Try without a filter."
        return "No tools are currently available."
    
    # ─────────────────────────────────────────────────────────────────
    # Original Formatters
    # ─────────────────────────────────────────────────────────────────
    
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
            if t.tags:
                lines.append(f"**Tags:** {', '.join(sorted(t.tags))}")
            if t.requires_api_key:
                lines.append("**Note:** Requires API key")
            if t.streaming:
                lines.append("**Feature:** Supports progress streaming")
            lines.append("")
        
        return "\n".join(lines)
