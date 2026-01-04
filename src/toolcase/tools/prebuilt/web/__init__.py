"""Web tools for search, fetch, parse, and extract operations.

Tools:
- WebSearchTool: Search the web via DuckDuckGo, Tavily, or Perplexity
- UrlFetchTool: Fetch URL content with smart extraction
- HtmlParseTool: Parse HTML and extract structured content
- RegexExtractTool: Extract patterns from text using regex
- JsonExtractTool: Extract and query JSON from text

Example:
    >>> from toolcase.tools.prebuilt.web import WebSearchTool, UrlFetchTool, HtmlParseTool
    >>> 
    >>> # Search -> Fetch -> Parse pipeline
    >>> search = WebSearchTool()
    >>> fetch = UrlFetchTool()
    >>> parse = HtmlParseTool()
    >>> 
    >>> results = await search.acall(query="python async patterns")
    >>> html = await fetch.acall(url=results["links"][0])
    >>> text = await parse.acall(html=html["content"], extract="text")
"""

from .backends import (
    DuckDuckGoBackend,
    PerplexityBackend,
    PerplexityModel,
    SearchBackend,
    SearchBackendType,
    SearchResponse,
    SearchResult,
    TavilyBackend,
)
from .browse import (
    WebSearchConfig,
    WebSearchParams,
    WebSearchTool,
    free_search,
    perplexity_search,
    tavily_search,
)
from .extract import (
    CommonPattern,
    JsonExtractTool,
    RegexExtractConfig,
    RegexExtractParams,
    RegexExtractTool,
)
from .fetch import (
    ContentMode,
    UrlFetchConfig,
    UrlFetchParams,
    UrlFetchTool,
)
from .parse import (
    ExtractMode,
    HtmlParseConfig,
    HtmlParseParams,
    HtmlParseTool,
)

__all__ = [
    # Web Search
    "WebSearchTool",
    "WebSearchConfig",
    "WebSearchParams",
    "free_search",
    "tavily_search",
    "perplexity_search",
    # Backends
    "SearchBackend",
    "SearchBackendType",
    "SearchResult",
    "SearchResponse",
    "TavilyBackend",
    "PerplexityBackend",
    "PerplexityModel",
    "DuckDuckGoBackend",
    # URL Fetch
    "UrlFetchTool",
    "UrlFetchConfig",
    "UrlFetchParams",
    "ContentMode",
    # HTML Parse
    "HtmlParseTool",
    "HtmlParseConfig",
    "HtmlParseParams",
    "ExtractMode",
    # Extraction
    "RegexExtractTool",
    "RegexExtractConfig",
    "RegexExtractParams",
    "CommonPattern",
    "JsonExtractTool",
]
