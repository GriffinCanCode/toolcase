"""Tavily search backend.

Tavily is a search API optimized for AI applications with built-in answer generation.
Requires TAVILY_API_KEY environment variable.

Example:
    >>> backend = TavilyBackend()
    >>> response = await backend.search("latest AI news", include_answer=True)
"""

from __future__ import annotations

import os
import time

import httpx

from .base import SearchBackend, SearchResponse, SearchResult


class TavilyBackend(SearchBackend):
    """Tavily search backend.
    
    Features:
    - AI-optimized search results
    - Built-in answer generation
    - Source relevance scoring
    
    Requires TAVILY_API_KEY environment variable or explicit api_key.
    """
    
    name = "tavily"
    requires_api_key = True
    supports_answer = True
    
    _BASE_URL = "https://api.tavily.com"
    
    def __init__(self, api_key: str | None = None, env_var: str = "TAVILY_API_KEY") -> None:
        """Initialize Tavily backend.
        
        Args:
            api_key: Explicit API key (takes precedence)
            env_var: Environment variable name for API key
        """
        self._api_key = api_key
        self._env_var = env_var
        self._client: httpx.AsyncClient | None = None
    
    def _get_api_key(self) -> str | None:
        return self._api_key or os.environ.get(self._env_var)
    
    def validate_config(self) -> str | None:
        return None if self._get_api_key() else f"Missing API key. Set {self._env_var} environment variable or provide api_key."
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._BASE_URL, timeout=60.0)
        return self._client
    
    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        include_answer: bool = False,
        timeout: float = 30.0,
    ) -> SearchResponse:
        """Execute Tavily search.
        
        Args:
            query: Search query
            max_results: Max results (1-20)
            include_answer: Include AI-generated answer
            timeout: Request timeout
            
        Returns:
            SearchResponse with results and optional answer
        """
        if err := self.validate_config():
            raise ValueError(err)
        
        start = time.perf_counter()
        client = await self._get_client()
        
        payload = {
            "api_key": self._get_api_key(),
            "query": query,
            "max_results": min(max_results, 20),
            "include_answer": include_answer,
            "search_depth": "advanced" if include_answer else "basic",
        }
        
        response = await client.post("/search", json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        results = [
            SearchResult(
                title=r.get("title", "Untitled"),
                url=r.get("url", ""),
                snippet=r.get("content", r.get("snippet", "")),
                score=r.get("score"),
            )
            for r in data.get("results", [])
        ]
        
        return SearchResponse(
            query=query,
            results=results,
            backend=self.name,
            answer=data.get("answer") if include_answer else None,
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
