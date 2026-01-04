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

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from toolcase.foundation.core.base import BaseTool, ToolMetadata
from toolcase.foundation.errors import Err, ErrorCode, ErrorTrace, JsonDict, ToolResult, format_validation_error

# Type alias for condition predicate
Predicate = Callable[[JsonDict], bool]


@dataclass(frozen=True, slots=True)
class Route:
    """A routing rule: condition → tool. Attrs: condition (predicate), tool, name (optional for debugging)"""
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
    input: JsonDict = Field(default_factory=dict, description="Input parameters to route and pass to selected tool")


RouterParams.model_rebuild()  # Resolve recursive JsonValue type


class RouterTool(BaseTool[RouterParams]):
    """Conditional tool router. Evaluates routes in order, executes first match, falls back to default.
    
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
        self._routes, self._default = routes, default
        all_names = [r.tool.metadata.name for r in routes] + [default.metadata.name]
        self._meta = ToolMetadata(
            name=name or f"router_{'_'.join(all_names[:3])}",
            description=description or f"Routes to: {', '.join(all_names)}",
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
        if r := next((r for r in self._routes if r.matches(input_dict)), None):
            return r.tool, r.name or r.tool.metadata.name
        return self._default, f"default:{self._default.metadata.name}"
    
    def _run(self, params: RouterParams) -> str:
        return self._run_async_sync(self._async_run(params))
    
    async def _async_run(self, params: RouterParams) -> str:
        r = await self._async_run_result(params)
        return r.unwrap() if r.is_ok() else r.unwrap_err().message
    
    async def _async_run_result(self, params: RouterParams) -> ToolResult:
        """Route and execute with Result-based handling."""
        tool, route_name = self._select_tool(params.input)
        
        try:
            tool_params = tool.params_schema(**params.input)
        except ValidationError as e:
            return Err(ErrorTrace(
                message=format_validation_error(e, tool_name=tool.metadata.name),
                error_code=ErrorCode.INVALID_PARAMS.value, recoverable=False,
            ).with_operation(f"router:{self._meta.name}"))
        
        return (await tool.arun_result(tool_params)).map_err(
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
    routes = [Route(condition=pred, tool=tool) for pred, tool in conditions]
    
    # Create predicate that checks for keyword in common fields
    def make_keyword_predicate(kw: str) -> Predicate:
        return lambda p: any(kw.lower() in str(p.get(k, "")).lower() for k in ("query", "q", "input", "text", "content"))
    
    routes += [
        Route(condition=make_keyword_predicate(kw), tool=tool, name=f"keyword:{kw}")
        for kw, tool in kwargs.items() if kw not in ("name", "description")
    ]
    return RouterTool(routes, default, name=name, description=description)
