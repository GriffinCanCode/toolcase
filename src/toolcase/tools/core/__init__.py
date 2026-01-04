"""Core tool infrastructure.

Provides:
- ConfigurableTool: Tools with runtime-configurable behavior
- ToolConfig: Base configuration class
- Query: Schema and capability-based tool matching
- Stats: Usage statistics collection
"""

from .base import ConfigurableTool, ToolConfig
from .query import (
    CapabilityFilter,
    MatchMode,
    QueryResult,
    SchemaPattern,
    ToolQuery,
    find_by_input_type,
    find_by_max_concurrent,
    find_by_param,
    find_by_tags,
    find_cacheable,
    find_streamable,
)
from .stats import (
    StatsMiddleware,
    ToolStats,
    UsageStats,
    format_stats,
    get_stats,
    reset_stats,
    set_stats,
)

__all__ = [
    # Base classes
    "ConfigurableTool",
    "ToolConfig",
    # Query engine
    "ToolQuery",
    "SchemaPattern",
    "CapabilityFilter",
    "QueryResult",
    "MatchMode",
    "find_by_param",
    "find_by_tags",
    "find_by_input_type",
    "find_streamable",
    "find_cacheable",
    "find_by_max_concurrent",
    # Statistics
    "UsageStats",
    "ToolStats",
    "StatsMiddleware",
    "get_stats",
    "set_stats",
    "reset_stats",
    "format_stats",
]
