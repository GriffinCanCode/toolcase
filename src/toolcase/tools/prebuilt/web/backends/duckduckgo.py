"""DuckDuckGo search backend (free, no API key required).

Uses the ddgs library for free web search.
No API key needed - works out of the box.

Example:
    >>> backend = DuckDuckGoBackend()
    >>> response = await backend.search("python async tutorial")
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from .base import SearchBackend, SearchResponse, SearchResult


class DuckDuckGoBackend(SearchBackend):
    """DuckDuckGo search backend (free).
    
    Features:
    - No API key required
    - Privacy-focused search
    - Rate-limited (be respectful)
    
    Uses ddgs library. Install with: pip install ddgs
    """
    
    name = "duckduckgo"
    requires_api_key = False
    supports_answer = False
    
    def __init__(self, region: str = "wt-wt", safesearch: str = "moderate") -> None:
        """Initialize DuckDuckGo backend.
        
        Args:
            region: Search region (wt-wt = no region, us-en, uk-en, etc.)
            safesearch: Safe search level (on, moderate, off)
        """
        self._region = region
        self._safesearch = safesearch
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    def validate_config(self) -> str | None:
        try:
            from ddgs import DDGS  # noqa: F401
            return None
        except ImportError:
            return "ddgs not installed. Run: pip install ddgs"
    
    def _sync_search(self, query: str, max_results: int) -> list[dict[str, str]]:
        """Synchronous search (runs in thread pool)."""
        from ddgs import DDGS
        
        with DDGS() as ddgs:
            return list(ddgs.text(query, region=self._region, safesearch=self._safesearch, max_results=max_results))
    
    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        include_answer: bool = False,
        timeout: float = 30.0,
    ) -> SearchResponse:
        """Execute DuckDuckGo search.
        
        Args:
            query: Search query
            max_results: Max results (1-25 recommended)
            include_answer: Ignored (DuckDuckGo doesn't provide answers)
            timeout: Search timeout
            
        Returns:
            SearchResponse with search results
        """
        if err := self.validate_config():
            raise ValueError(err)
        
        start = time.perf_counter()
        loop = asyncio.get_event_loop()
        
        try:
            raw_results = await asyncio.wait_for(
                loop.run_in_executor(self._executor, partial(self._sync_search, query, min(max_results, 25))),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raw_results = []
        
        results = [
            SearchResult(
                title=r.get("title", "Untitled"),
                url=r.get("href", r.get("link", "")),
                snippet=r.get("body", r.get("snippet", "")),
                score=None,
            )
            for r in raw_results
        ]
        
        return SearchResponse(
            query=query,
            results=results,
            backend=self.name,
            answer=None,
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
    
    def close(self) -> None:
        """Shutdown executor."""
        self._executor.shutdown(wait=False)
