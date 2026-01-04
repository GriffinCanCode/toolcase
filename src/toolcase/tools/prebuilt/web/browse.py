"""Web Search Tool - Search the web using multiple backends.

A production-ready web search tool with:
- Multiple backend support (Tavily, Perplexity, DuckDuckGo)
- Environment-aware API key loading
- Optional AI-generated answers
- Configurable result limits and timeouts

Example:
    >>> from toolcase.tools.prebuilt.web_search import WebSearchTool, WebSearchConfig
    >>> 
    >>> # Free search (DuckDuckGo, no API key)
    >>> search = WebSearchTool()
    >>> result = await search.acall(query="python async patterns")
    >>> 
    >>> # With Tavily (requires TAVILY_API_KEY)
    >>> search = WebSearchTool(WebSearchConfig(backend="tavily"))
    >>> result = await search.acall(query="latest AI news", include_answer=True)
    >>> 
    >>> # With Perplexity
    >>> search = WebSearchTool(WebSearchConfig(backend="perplexity"))
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from toolcase.foundation.core import ToolMetadata
from toolcase.foundation.errors import Err, ErrorCode, ErrorTrace, Ok, ToolResult

from ...core.base import ConfigurableTool, ToolConfig
from .backends import (
    DuckDuckGoBackend,
    PerplexityBackend,
    SearchBackend,
    SearchBackendType,
    SearchResponse,
    TavilyBackend,
)


class WebSearchConfig(ToolConfig):
    """Configuration for WebSearchTool.
    
    Attributes:
        backend: Search backend to use (tavily, perplexity, duckduckgo)
        max_results: Maximum results per search
        include_answer: Request AI-generated answer by default
        tavily_api_key: Explicit Tavily API key (or use TAVILY_API_KEY env var)
        perplexity_api_key: Explicit Perplexity API key (or use PERPLEXITY_API_KEY env var)
        perplexity_model: Perplexity model tier (sonar or sonar-pro)
    """
    
    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_default=True,
        json_schema_extra={"title": "Web Search Configuration"},
    )
    
    backend: SearchBackendType = Field(default="duckduckgo", description="Search backend (tavily, perplexity, duckduckgo)")
    max_results: int = Field(default=10, ge=1, le=25, description="Max results per search")
    include_answer: bool = Field(default=False, description="Request AI-generated answer by default")
    
    # Backend-specific config
    tavily_api_key: str | None = Field(default=None, description="Tavily API key (or set TAVILY_API_KEY)")
    perplexity_api_key: str | None = Field(default=None, description="Perplexity API key (or set PERPLEXITY_API_KEY)")
    perplexity_model: Literal["sonar", "sonar-pro"] = Field(default="sonar", description="Perplexity model tier")
    
    def __hash__(self) -> int:
        return hash((self.backend, self.max_results, self.include_answer))


class WebSearchParams(BaseModel):
    """Parameters for web search.
    
    Attributes:
        query: Search query string
        max_results: Override max results for this search
        include_answer: Request AI-generated answer (if backend supports it)
    """
    
    model_config = ConfigDict(
        str_strip_whitespace=True, extra="forbid",
        json_schema_extra={
            "title": "Web Search Parameters",
            "examples": [
                {"query": "latest developments in AI"},
                {"query": "how does async/await work", "include_answer": True},
            ],
        },
    )
    
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    max_results: int | None = Field(default=None, ge=1, le=25, description="Override max results")
    include_answer: bool | None = Field(default=None, description="Request AI answer (if supported)")


def _err(msg: str, code: ErrorCode, op: str, recoverable: bool = False, **kw: str) -> ToolResult:
    """Create error result."""
    return Err(ErrorTrace(message=msg, error_code=code.value, recoverable=recoverable, **kw).with_operation(op))


class WebSearchTool(ConfigurableTool[WebSearchParams, WebSearchConfig]):
    """Web search tool with multiple backend support.
    
    Backends:
    - **duckduckgo** (default): Free, no API key required
    - **tavily**: AI-optimized search with answer generation (requires API key)
    - **perplexity**: AI-powered search with citations (requires API key)
    
    Example:
        >>> # Free search (default)
        >>> search = WebSearchTool()
        >>> result = await search.acall(query="python web frameworks")
        >>> 
        >>> # With Tavily for AI answers
        >>> search = WebSearchTool(WebSearchConfig(backend="tavily"))
        >>> result = await search.acall(query="explain transformers", include_answer=True)
    """
    
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="web_search",
        description=(
            "Search the web for information. Returns relevant results with titles, URLs, and snippets. "
            "Can optionally provide AI-generated answers (with Tavily or Perplexity backends). "
            "Default backend (DuckDuckGo) requires no API key."
        ),
        category="search",
        requires_api_key=False,  # Default backend is free
        streaming=False,
        tags=frozenset({"search", "web", "information"}),
    )
    params_schema: ClassVar[type[WebSearchParams]] = WebSearchParams
    config_class: ClassVar[type[WebSearchConfig]] = WebSearchConfig
    cache_enabled: ClassVar[bool] = True  # Search results can be cached
    
    def __init__(self, config: WebSearchConfig | None = None) -> None:
        super().__init__(config)
        self._backend: SearchBackend | None = None
    
    def _get_backend(self) -> SearchBackend:
        """Get or create backend instance."""
        if self._backend is None:
            match self.config.backend:
                case "tavily":
                    self._backend = TavilyBackend(api_key=self.config.tavily_api_key)
                case "perplexity":
                    self._backend = PerplexityBackend(
                        api_key=self.config.perplexity_api_key,
                        model=self.config.perplexity_model,
                    )
                case "duckduckgo":
                    self._backend = DuckDuckGoBackend()
        return self._backend
    
    async def _async_run_result(self, params: WebSearchParams) -> ToolResult:
        """Execute search with Result-based error handling."""
        backend = self._get_backend()
        
        # Validate backend config
        if err := backend.validate_config():
            return _err(err, ErrorCode.API_KEY_MISSING, "backend_validation")
        
        max_results = params.max_results or self.config.max_results
        include_answer = params.include_answer if params.include_answer is not None else self.config.include_answer
        
        # Skip answer request if backend doesn't support it
        if include_answer and not backend.supports_answer:
            include_answer = False
        
        try:
            response = await backend.search(
                query=params.query,
                max_results=max_results,
                include_answer=include_answer,
                timeout=self.config.timeout,
            )
            return Ok(response.format(include_answer=include_answer))
        except ValueError as e:
            return _err(str(e), ErrorCode.INVALID_PARAMS, "search")
        except Exception as e:
            return _err(f"Search failed: {e}", ErrorCode.EXTERNAL_SERVICE_ERROR, "search", recoverable=True)
    
    async def _async_run(self, params: WebSearchParams) -> str:
        """Execute search."""
        from toolcase.foundation.errors import result_to_string
        return result_to_string(await self._async_run_result(params), self.metadata.name)


# Convenience factory functions
def tavily_search(api_key: str | None = None) -> WebSearchTool:
    """Create WebSearchTool with Tavily backend.
    
    Args:
        api_key: Explicit API key (or set TAVILY_API_KEY env var)
    """
    return WebSearchTool(WebSearchConfig(backend="tavily", tavily_api_key=api_key))


def perplexity_search(api_key: str | None = None, model: Literal["sonar", "sonar-pro"] = "sonar") -> WebSearchTool:
    """Create WebSearchTool with Perplexity backend.
    
    Args:
        api_key: Explicit API key (or set PERPLEXITY_API_KEY env var)
        model: Model tier (sonar or sonar-pro)
    """
    return WebSearchTool(WebSearchConfig(backend="perplexity", perplexity_api_key=api_key, perplexity_model=model))


def free_search() -> WebSearchTool:
    """Create WebSearchTool with DuckDuckGo backend (no API key required)."""
    return WebSearchTool(WebSearchConfig(backend="duckduckgo"))
