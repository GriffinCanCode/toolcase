"""Tool query engine for schema-based and capability-based discovery.

Enables semantic tool matching via:
- Schema matching: Find tools accepting specific parameter types
- Capability search: Filter by tags, categories, or metadata patterns
- Capability filtering: Filter by advertised capabilities (streaming, caching, etc.)
- Fuzzy name matching: Find tools by partial name match

Example:
    >>> from toolcase.tools.core.query import ToolQuery, SchemaPattern, CapabilityFilter
    >>> 
    >>> # Find tools with a 'query' string parameter
    >>> matches = ToolQuery(registry).by_schema(SchemaPattern(name="query", type="string"))
    >>> 
    >>> # Find tools with capability tags
    >>> matches = ToolQuery(registry).by_capability(["search", "web"])
    >>> 
    >>> # Find tools supporting streaming with max 5 concurrent executions
    >>> matches = ToolQuery(registry).by_capabilities(
    ...     CapabilityFilter(supports_streaming=True, max_concurrent_lte=5)
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool, ToolMetadata
    from toolcase.foundation.registry import ToolRegistry


class MatchMode(str, Enum):
    """How to combine multiple criteria."""
    ALL = "all"  # AND - all criteria must match
    ANY = "any"  # OR - at least one criterion must match


@dataclass(frozen=True, slots=True)
class SchemaPattern:
    """Pattern for matching tool parameter schemas.
    
    Attributes:
        name: Field name to match (None = any field)
        type: JSON schema type ('string', 'integer', 'boolean', 'array', 'object')
        required: If True, only match required fields
        description_contains: Substring match on field description
    
    Example:
        >>> SchemaPattern(name="query", type="string")  # Exact field match
        >>> SchemaPattern(type="string", required=True)  # Any required string
        >>> SchemaPattern(description_contains="search")  # Semantic match
    """
    name: str | None = None
    type: str | None = None
    required: bool | None = None
    description_contains: str | None = None


@dataclass(frozen=True, slots=True)
class CapabilityFilter:
    """Filter for matching tool capabilities.
    
    Supports exact matches and range queries for concurrency/latency.
    None values are ignored (no filter applied).
    
    Attributes:
        supports_caching: Filter by caching support
        supports_streaming: Filter by streaming support
        idempotent: Filter by idempotency
        requires_confirmation: Filter by confirmation requirement
        max_concurrent_gte: Min max_concurrent (inclusive)
        max_concurrent_lte: Max max_concurrent (inclusive)
        estimated_latency_lte: Max estimated latency (inclusive)
    
    Example:
        >>> CapabilityFilter(supports_streaming=True)  # Streaming tools only
        >>> CapabilityFilter(max_concurrent_lte=5)  # Rate-limited tools
        >>> CapabilityFilter(idempotent=True, supports_caching=True)  # Safe to cache/retry
    """
    supports_caching: bool | None = None
    supports_streaming: bool | None = None
    idempotent: bool | None = None
    requires_confirmation: bool | None = None
    max_concurrent_gte: int | None = None
    max_concurrent_lte: int | None = None
    estimated_latency_lte: int | None = None


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Single tool match with relevance info.
    
    Attributes:
        tool: Matched tool instance
        metadata: Tool metadata
        score: Match relevance (0.0-1.0, higher = better match)
        matched_fields: Fields that matched schema patterns
        matched_tags: Tags that matched capability query
    """
    tool: BaseTool[BaseModel]
    metadata: ToolMetadata
    score: float = 1.0
    matched_fields: tuple[str, ...] = ()
    matched_tags: tuple[str, ...] = ()


@dataclass(slots=True)
class ToolQuery:
    """Fluent query builder for tool discovery.
    
    Chainable methods for building complex queries:
    
    Example:
        >>> results = (ToolQuery(registry)
        ...     .by_capability(["search"])
        ...     .by_schema(SchemaPattern(type="string", required=True))
        ...     .by_category("search")
        ...     .by_capabilities(CapabilityFilter(supports_streaming=True))
        ...     .enabled_only()
        ...     .execute())
    """
    
    _registry: ToolRegistry
    _schema_patterns: list[SchemaPattern] = field(default_factory=list)
    _capabilities: list[str] = field(default_factory=list)
    _capability_filters: list[CapabilityFilter] = field(default_factory=list)
    _categories: list[str] = field(default_factory=list)
    _name_pattern: str | None = None
    _enabled_only: bool = True
    _match_mode: MatchMode = MatchMode.ALL
    _limit: int | None = None
    
    def by_schema(self, *patterns: SchemaPattern) -> ToolQuery:
        """Add schema patterns to match."""
        self._schema_patterns.extend(patterns)
        return self
    
    def by_capability(self, tags: list[str], mode: MatchMode = MatchMode.ANY) -> ToolQuery:
        """Filter by capability tags (OR by default)."""
        self._capabilities.extend(tags)
        self._match_mode = mode
        return self
    
    def by_capabilities(self, *filters: CapabilityFilter) -> ToolQuery:
        """Filter by advertised tool capabilities.
        
        Example:
            >>> query.by_capabilities(CapabilityFilter(supports_streaming=True))
            >>> query.by_capabilities(CapabilityFilter(max_concurrent_lte=5, idempotent=True))
        """
        self._capability_filters.extend(filters)
        return self
    
    def streamable(self) -> ToolQuery:
        """Shortcut: filter to tools supporting streaming."""
        return self.by_capabilities(CapabilityFilter(supports_streaming=True))
    
    def cacheable(self) -> ToolQuery:
        """Shortcut: filter to tools supporting caching."""
        return self.by_capabilities(CapabilityFilter(supports_caching=True))
    
    def idempotent(self) -> ToolQuery:
        """Shortcut: filter to idempotent tools (safe to retry)."""
        return self.by_capabilities(CapabilityFilter(idempotent=True))
    
    def rate_limited(self, max_concurrent: int | None = None) -> ToolQuery:
        """Shortcut: filter to rate-limited tools (with max_concurrent set)."""
        return self.by_capabilities(CapabilityFilter(
            max_concurrent_lte=max_concurrent if max_concurrent else 1000,  # Any limited
        ))
    
    def by_category(self, *categories: str) -> ToolQuery:
        """Filter by one or more categories."""
        self._categories.extend(categories)
        return self
    
    def by_name(self, pattern: str) -> ToolQuery:
        """Filter by name substring match."""
        self._name_pattern = pattern.lower()
        return self
    
    def enabled_only(self, enabled: bool = True) -> ToolQuery:
        """Include only enabled tools (default: True)."""
        self._enabled_only = enabled
        return self
    
    def limit(self, n: int) -> ToolQuery:
        """Limit number of results."""
        self._limit = n
        return self
    
    def execute(self) -> list[QueryResult]:
        """Execute query and return matched tools with scores."""
        results: list[QueryResult] = []
        
        for tool in self._registry:
            meta = tool.metadata
            
            # Skip disabled if requested
            if self._enabled_only and not meta.enabled:
                continue
            
            # Category filter
            if self._categories and meta.category not in self._categories:
                continue
            
            # Name filter
            if self._name_pattern and self._name_pattern not in meta.name.lower():
                continue
            
            # Capability tag matching
            matched_tags: tuple[str, ...] = ()
            if self._capabilities:
                matched = tuple(t for t in self._capabilities if t in meta.tags)
                if self._match_mode == MatchMode.ALL and len(matched) != len(self._capabilities):
                    continue
                if self._match_mode == MatchMode.ANY and not matched:
                    continue
                matched_tags = matched
            
            # Capability filter matching
            if self._capability_filters and not _match_capabilities(meta.capabilities, self._capability_filters):
                continue
            
            # Schema matching
            matched_fields: tuple[str, ...] = ()
            if self._schema_patterns:
                schema = tool.params_schema.model_json_schema()
                matched_fields = _match_schema(schema, self._schema_patterns)
                if not matched_fields:
                    continue
            
            # Calculate relevance score
            score = _calculate_score(meta, matched_fields, matched_tags, self._schema_patterns, self._capabilities)
            results.append(QueryResult(tool, meta, score, matched_fields, matched_tags))
        
        # Sort by score descending, then by name
        results.sort(key=lambda r: (-r.score, r.metadata.name))
        return results[:self._limit] if self._limit else results


def _match_schema(schema: dict, patterns: list[SchemaPattern]) -> tuple[str, ...]:
    """Match schema against patterns, return matched field names."""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    matched: list[str] = []
    
    for pattern in patterns:
        for name, field_schema in props.items():
            if _field_matches(name, field_schema, pattern, name in required):
                matched.append(name)
    
    return tuple(set(matched))  # Deduplicate


def _match_capabilities(caps: object, filters: list[CapabilityFilter]) -> bool:
    """Check if capabilities match all filters. Returns True if all filters pass."""
    from toolcase.foundation.core import ToolCapabilities
    if not isinstance(caps, ToolCapabilities):
        return False
    
    for f in filters:
        # Boolean exact matches
        if f.supports_caching is not None and caps.supports_caching != f.supports_caching:
            return False
        if f.supports_streaming is not None and caps.supports_streaming != f.supports_streaming:
            return False
        if f.idempotent is not None and caps.idempotent != f.idempotent:
            return False
        if f.requires_confirmation is not None and caps.requires_confirmation != f.requires_confirmation:
            return False
        
        # Range queries for max_concurrent
        if f.max_concurrent_gte is not None:
            if caps.max_concurrent is None or caps.max_concurrent < f.max_concurrent_gte:
                return False
        if f.max_concurrent_lte is not None:
            if caps.max_concurrent is None:
                continue  # None = unlimited, doesn't match "limited" filter
            if caps.max_concurrent > f.max_concurrent_lte:
                return False
        
        # Range query for latency
        if f.estimated_latency_lte is not None:
            if caps.estimated_latency_ms is None or caps.estimated_latency_ms > f.estimated_latency_lte:
                return False
    
    return True


def _field_matches(name: str, schema: dict, pattern: SchemaPattern, is_required: bool) -> bool:
    """Check if a field matches a schema pattern."""
    # Name match (exact or None for wildcard)
    if pattern.name and pattern.name != name:
        return False
    
    # Type match
    if pattern.type and schema.get("type") != pattern.type:
        return False
    
    # Required match
    if pattern.required is not None and pattern.required != is_required:
        return False
    
    # Description match
    if pattern.description_contains:
        desc = schema.get("description", "").lower()
        if pattern.description_contains.lower() not in desc:
            return False
    
    return True


def _calculate_score(
    meta: ToolMetadata,
    matched_fields: tuple[str, ...],
    matched_tags: tuple[str, ...],
    schema_patterns: list[SchemaPattern],
    capabilities: list[str],
) -> float:
    """Calculate relevance score (0.0-1.0)."""
    if not schema_patterns and not capabilities:
        return 1.0
    
    scores: list[float] = []
    
    # Schema match ratio
    if schema_patterns:
        scores.append(len(matched_fields) / len(schema_patterns))
    
    # Tag match ratio
    if capabilities:
        scores.append(len(matched_tags) / len(capabilities))
    
    return sum(scores) / len(scores) if scores else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────

def find_by_param(registry: ToolRegistry, name: str, type_: str | None = None) -> list[QueryResult]:
    """Shortcut to find tools accepting a parameter by name and optional type.
    
    Example:
        >>> tools = find_by_param(registry, "query", "string")
        >>> tools = find_by_param(registry, "limit")  # Any type
    """
    return ToolQuery(registry).by_schema(SchemaPattern(name=name, type=type_)).execute()


def find_by_tags(registry: ToolRegistry, *tags: str, mode: MatchMode = MatchMode.ANY) -> list[QueryResult]:
    """Shortcut to find tools by capability tags.
    
    Example:
        >>> tools = find_by_tags(registry, "search", "web")  # OR
        >>> tools = find_by_tags(registry, "search", "summarize", mode=MatchMode.ALL)
    """
    return ToolQuery(registry).by_capability(list(tags), mode).execute()


def find_by_input_type(registry: ToolRegistry, type_: str, *, required_only: bool = False) -> list[QueryResult]:
    """Find tools accepting a specific input type.
    
    Example:
        >>> tools = find_by_input_type(registry, "string")  # Any string param
        >>> tools = find_by_input_type(registry, "integer", required_only=True)
    """
    return ToolQuery(registry).by_schema(
        SchemaPattern(type=type_, required=required_only if required_only else None)
    ).execute()


def find_streamable(registry: ToolRegistry) -> list[QueryResult]:
    """Find tools that support streaming output.
    
    Example:
        >>> streaming_tools = find_streamable(registry)
    """
    return ToolQuery(registry).streamable().execute()


def find_cacheable(registry: ToolRegistry) -> list[QueryResult]:
    """Find tools that support result caching.
    
    Example:
        >>> cacheable_tools = find_cacheable(registry)
    """
    return ToolQuery(registry).cacheable().execute()


def find_by_max_concurrent(registry: ToolRegistry, max_value: int) -> list[QueryResult]:
    """Find tools with max_concurrent <= specified value (rate-limited).
    
    Example:
        >>> rate_limited = find_by_max_concurrent(registry, 5)
    """
    return ToolQuery(registry).by_capabilities(
        CapabilityFilter(max_concurrent_lte=max_value)
    ).execute()
