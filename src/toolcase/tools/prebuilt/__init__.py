"""Prebuilt tools ready for use.

Includes:
- HttpTool: HTTP request tool with auth strategies (explicit and env-based)
- DiscoveryTool: Meta-tool for tool discovery and search
- WebSearchTool: Web search with Tavily, Perplexity, and DuckDuckGo backends
- UrlFetchTool: Fetch URL content with smart extraction
- HtmlParseTool: Parse HTML and extract structured content
- RegexExtractTool: Extract patterns from text using regex
- JsonExtractTool: Extract and query JSON from text
"""

from .discovery import DiscoveryParams, DiscoveryTool
from .http import (
    ApiKeyAuth,
    BasicAuth,
    BearerAuth,
    CustomAuth,
    EnvApiKeyAuth,
    EnvBasicAuth,
    EnvBearerAuth,
    HttpConfig,
    HttpParams,
    HttpResponse,
    HttpTool,
    NoAuth,
    api_key_from_env,
    basic_from_env,
    bearer_from_env,
    get_no_auth,
)
from .web import (
    # Web Search
    DuckDuckGoBackend,
    PerplexityBackend,
    SearchBackend,
    SearchResponse,
    SearchResult,
    TavilyBackend,
    WebSearchConfig,
    WebSearchParams,
    WebSearchTool,
    free_search,
    perplexity_search,
    tavily_search,
    # URL Fetch
    UrlFetchTool,
    UrlFetchConfig,
    UrlFetchParams,
    # HTML Parse
    HtmlParseTool,
    HtmlParseConfig,
    HtmlParseParams,
    # Extraction
    RegexExtractTool,
    RegexExtractConfig,
    RegexExtractParams,
    JsonExtractTool,
)

__all__ = [
    # HTTP Tool
    "HttpTool",
    "HttpConfig",
    "HttpParams",
    "HttpResponse",
    # Auth (explicit secrets)
    "NoAuth",
    "BearerAuth",
    "BasicAuth",
    "ApiKeyAuth",
    "CustomAuth",
    "get_no_auth",
    # Auth (environment-based)
    "EnvBearerAuth",
    "EnvApiKeyAuth",
    "EnvBasicAuth",
    "bearer_from_env",
    "api_key_from_env",
    "basic_from_env",
    # Discovery
    "DiscoveryTool",
    "DiscoveryParams",
    # Web Search
    "WebSearchTool",
    "WebSearchConfig",
    "WebSearchParams",
    "free_search",
    "tavily_search",
    "perplexity_search",
    "SearchBackend",
    "SearchResult",
    "SearchResponse",
    "TavilyBackend",
    "PerplexityBackend",
    "DuckDuckGoBackend",
    # URL Fetch
    "UrlFetchTool",
    "UrlFetchConfig",
    "UrlFetchParams",
    # HTML Parse
    "HtmlParseTool",
    "HtmlParseConfig",
    "HtmlParseParams",
    # Extraction
    "RegexExtractTool",
    "RegexExtractConfig",
    "RegexExtractParams",
    "JsonExtractTool",
]
