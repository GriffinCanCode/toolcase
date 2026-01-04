"""Web search backends.

Available backends:
- TavilyBackend: AI-optimized search with answer generation (requires API key)
- PerplexityBackend: AI-powered search with citations (requires API key)
- DuckDuckGoBackend: Free search, no API key required
"""

from .base import SearchBackend, SearchBackendType, SearchResponse, SearchResult
from .duckduckgo import DuckDuckGoBackend
from .perplexity import PerplexityBackend, PerplexityModel
from .tavily import TavilyBackend

__all__ = [
    # Base
    "SearchBackend",
    "SearchBackendType",
    "SearchResult",
    "SearchResponse",
    # Backends
    "TavilyBackend",
    "PerplexityBackend",
    "PerplexityModel",
    "DuckDuckGoBackend",
]
