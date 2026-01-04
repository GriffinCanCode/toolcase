"""Perplexity search backend.

Perplexity is an AI-powered search engine with strong answer generation capabilities.
Requires PERPLEXITY_API_KEY environment variable.

Example:
    >>> backend = PerplexityBackend()
    >>> response = await backend.search("explain quantum computing", include_answer=True)
"""

from __future__ import annotations

import os
import time
from typing import Literal

import httpx

from .base import SearchBackend, SearchResponse, SearchResult


PerplexityModel = Literal["sonar", "sonar-pro"]


class PerplexityBackend(SearchBackend):
    """Perplexity search backend.
    
    Features:
    - AI-powered search with citations
    - Strong answer generation
    - Multiple model tiers
    
    Requires PERPLEXITY_API_KEY environment variable or explicit api_key.
    """
    
    name = "perplexity"
    requires_api_key = True
    supports_answer = True
    
    _BASE_URL = "https://api.perplexity.ai"
    
    def __init__(
        self,
        api_key: str | None = None,
        env_var: str = "PERPLEXITY_API_KEY",
        model: PerplexityModel = "sonar",
    ) -> None:
        """Initialize Perplexity backend.
        
        Args:
            api_key: Explicit API key (takes precedence)
            env_var: Environment variable name for API key
            model: Model to use (sonar or sonar-pro)
        """
        self._api_key = api_key
        self._env_var = env_var
        self._model = model
        self._client: httpx.AsyncClient | None = None
    
    def _get_api_key(self) -> str | None:
        return self._api_key or os.environ.get(self._env_var)
    
    def validate_config(self) -> str | None:
        return None if self._get_api_key() else f"Missing API key. Set {self._env_var} environment variable or provide api_key."
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._BASE_URL,
                headers={"Authorization": f"Bearer {self._get_api_key()}"},
                timeout=60.0,
            )
        return self._client
    
    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        include_answer: bool = False,
        timeout: float = 30.0,
    ) -> SearchResponse:
        """Execute Perplexity search.
        
        Perplexity uses chat completions API with search grounding.
        Results come from citations in the response.
        
        Args:
            query: Search query
            max_results: Max citations to extract
            include_answer: Always True for Perplexity (answer-first)
            timeout: Request timeout
            
        Returns:
            SearchResponse with answer and source citations
        """
        if err := self.validate_config():
            raise ValueError(err)
        
        start = time.perf_counter()
        client = await self._get_client()
        
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": query}],
            "return_citations": True,
        }
        
        response = await client.post("/chat/completions", json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        # Extract answer from response
        answer = ""
        if choices := data.get("choices", []):
            if msg := choices[0].get("message", {}):
                answer = msg.get("content", "")
        
        # Extract citations as results
        results: list[SearchResult] = []
        if citations := data.get("citations", []):
            for i, url in enumerate(citations[:max_results]):
                results.append(SearchResult(
                    title=f"Source {i + 1}",
                    url=url if isinstance(url, str) else str(url),
                    snippet="Referenced in answer",
                    score=None,
                ))
        
        return SearchResponse(
            query=query,
            results=results,
            backend=self.name,
            answer=answer if (include_answer or True) else None,  # Perplexity always provides answer
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
