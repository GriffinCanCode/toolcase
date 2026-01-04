"""Router primitive for conditional tool dispatch.

Routes inputs to different tools based on predicates. Useful for:
- Content-based routing (keywords, patterns)
- Type-based dispatch
- Load balancing across providers
- A/B testing tool variants

Example:
    >>> search = router(
    ...     when=lambda p: "news" in p.get("query", ""), use=NewsTool(),
    ...     when=lambda p: p.get("source") == "academic", use=AcademicTool(),
    ...     default=WebSearchTool(),
    ... )
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from pydantic import BaseModel, Field, ValidationError

from toolcase.foundation.core.base import BaseTool, ToolMetadata
from toolcase.foundation.errors import Err, ErrorCode, ErrorTrace, JsonDict, Ok, ToolResult, format_validation_error

if TYPE_CHECKING:
    pass

# Type alias for condition predicate
Predicate = Callable[[JsonDict], bool]


@dataclass(frozen=True, slots=True)
class Route:
    """A routing rule: condition → tool.
    
    Attributes:
        condition: Predicate that receives input dict, returns True to route
        tool: Tool to execute when condition matches
        name: Optional name for debugging/logging
    """
    
    condition: Predicate
    tool: BaseTool[BaseModel]
    name: str = ""
    
    def matches(self, input_dict: JsonDict) -> bool:
        """Check if this route's condition matches the input."""
        try:
            return self.condition(input_dict)
        except Exception:
            return False


class RouterParams(BaseModel):
    """Parameters for router execution."""
    
    input: JsonDict = Field(
        default_factory=dict,
        description="Input parameters to route and pass to selected tool",
    )


# Rebuild model to resolve recursive JsonValue type
RouterParams.model_rebuild()


class RouterTool(BaseTool[RouterParams]):
    """Conditional tool router based on input predicates.
    
    Evaluates routes in order, executing the first matching tool.
    Falls back to default tool if no routes match.
    
    Uses railway-oriented programming - errors propagate through.
    
    Example:
        >>> router = RouterTool(
        ...     routes=[
        ...         Route(lambda p: "code" in p.get("query", ""), CodeSearchTool()),
        ...         Route(lambda p: "docs" in p.get("query", ""), DocsSearchTool()),
        ...     ],
        ...     default=WebSearchTool(),
        ... )
    """
    
    __slots__ = ("_routes", "_default", "_meta")
    
    params_schema = RouterParams
    cache_enabled = False  # Routes delegate caching to inner tools
    
    def __init__(
        self,
        routes: list[Route],
        default: BaseTool[BaseModel],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self._routes = routes
        self._default = default
        
        # Derive metadata
        route_names = [r.tool.metadata.name for r in routes]
        all_names = route_names + [default.metadata.name]
        derived_name = name or f"router_{'_'.join(all_names[:3])}"
        derived_desc = description or f"Routes to: {', '.join(all_names)}"
        
        self._meta = ToolMetadata(
            name=derived_name,
            description=derived_desc,
            category="agents",
            streaming=any(r.tool.metadata.streaming for r in routes) or default.metadata.streaming,
        )
    
    @property
    def metadata(self) -> ToolMetadata:
        return self._meta
    
    @property
    def routes(self) -> list[Route]:
        return self._routes
    
    @property
    def default(self) -> BaseTool[BaseModel]:
        return self._default
    
    def _select_tool(self, input_dict: JsonDict) -> tuple[BaseTool[BaseModel], str]:
        """Select tool based on input, returns (tool, route_name)."""
        for route in self._routes:
            if route.matches(input_dict):
                return route.tool, route.name or route.tool.metadata.name
        return self._default, f"default:{self._default.metadata.name}"
    
    def _run(self, params: RouterParams) -> str:
        return self._run_async_sync(self._async_run(params))
    
    async def _async_run(self, params: RouterParams) -> str:
        result = await self._async_run_result(params)
        if result.is_ok():
            return result.unwrap()
        return result.unwrap_err().message
    
    async def _async_run_result(self, params: RouterParams) -> ToolResult:
        """Route and execute with Result-based handling."""
        input_dict = params.input
        tool, route_name = self._select_tool(input_dict)
        
        # Build params for selected tool
        try:
            tool_params = tool.params_schema(**input_dict)
        except ValidationError as e:
            trace = ErrorTrace(
                message=format_validation_error(e, tool_name=tool.metadata.name),
                error_code=ErrorCode.INVALID_PARAMS.value,
                recoverable=False,
            ).with_operation(f"router:{self._meta.name}")
            return Err(trace)
        
        # Execute and tag with route info
        result = await tool.arun_result(tool_params)
        return result.map_err(
            lambda e: e.with_operation(f"router:{self._meta.name}", route=route_name)
        )


def router(
    *conditions: tuple[Predicate, BaseTool[BaseModel]],
    default: BaseTool[BaseModel],
    name: str | None = None,
    description: str | None = None,
    **kwargs: BaseTool[BaseModel],  # Alternative: when_keyword=tool
) -> RouterTool:
    """Create a router from conditions and tools.
    
    Supports multiple call styles:
    
    1. Tuple-based (explicit):
        >>> r = router(
        ...     (lambda p: "news" in p.get("q", ""), NewsTool()),
        ...     (lambda p: "code" in p.get("q", ""), CodeTool()),
        ...     default=WebTool(),
        ... )
    
    2. Keyword-based (simple keyword matching):
        >>> r = router(
        ...     default=WebTool(),
        ...     news=NewsTool(),      # Matches if "news" in query
        ...     academic=AcademicTool(),
        ... )
    
    Args:
        *conditions: Tuples of (predicate, tool)
        default: Fallback tool when no route matches
        name: Optional router name
        description: Optional description
        **kwargs: Keyword routes (key becomes keyword to match in 'query')
    
    Returns:
        RouterTool instance
    """
    routes: list[Route] = []
    
    # Add explicit condition routes
    for predicate, tool in conditions:
        routes.append(Route(condition=predicate, tool=tool))
    
    # Add keyword-based routes (convenience)
    for keyword, tool in kwargs.items():
        if keyword in ("name", "description"):
            continue
        # Create predicate that checks for keyword in common fields
        def make_keyword_predicate(kw: str) -> Predicate:
            def predicate(p: JsonDict) -> bool:
                # Check common field names
                for key in ("query", "q", "input", "text", "content"):
                    val = p.get(key, "")
                    if isinstance(val, str) and kw.lower() in val.lower():
                        return True
                return False
            return predicate
        
        routes.append(Route(
            condition=make_keyword_predicate(keyword),
            tool=tool,
            name=f"keyword:{keyword}",
        ))
    
    return RouterTool(routes, default, name=name, description=description)
