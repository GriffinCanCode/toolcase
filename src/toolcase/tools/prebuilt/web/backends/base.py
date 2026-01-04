"""Base protocol for web search backends.

Defines the interface that all search backends must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SearchResult(BaseModel):
    """Individual search result from any backend."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    title: str = Field(description="Page title")
    url: str = Field(description="Result URL")
    snippet: str = Field(description="Text snippet/excerpt")
    score: float | None = Field(default=None, ge=0.0, le=1.0, description="Relevance score if available")
    
    def format(self) -> str:
        """Format as markdown."""
        score_str = f" ({self.score:.0%})" if self.score is not None else ""
        return f"**[{self.title}]({self.url})**{score_str}\n{self.snippet}"


class SearchResponse(BaseModel):
    """Response from a search backend."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    query: str = Field(description="Original search query")
    results: list[SearchResult] = Field(default_factory=list, description="Search results")
    backend: str = Field(description="Backend that produced these results")
    answer: str | None = Field(default=None, description="AI-generated answer (if supported)")
    elapsed_ms: float = Field(ge=0, description="Search duration in milliseconds")
    
    def format(self, include_answer: bool = True) -> str:
        """Format response as markdown."""
        lines = [f"**Web Search:** `{self.query}`", f"_Backend: {self.backend} | {self.elapsed_ms:.0f}ms | {len(self.results)} results_\n"]
        if include_answer and self.answer:
            lines.extend(["**Summary:**", self.answer, ""])
        if self.results:
            lines.append("**Results:**")
            lines.extend(f"{i}. {r.format()}\n" for i, r in enumerate(self.results, 1))
        else:
            lines.append("_No results found._")
        return "\n".join(lines)


SearchBackendType = Literal["tavily", "perplexity", "duckduckgo"]


class SearchBackend(ABC):
    """Abstract base for search backends."""
    
    name: str  # Backend identifier
    requires_api_key: bool  # Whether this backend needs an API key
    supports_answer: bool  # Whether backend provides AI-generated answers
    
    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        include_answer: bool = False,
        timeout: float = 30.0,
    ) -> SearchResponse:
        """Execute search query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            include_answer: Request AI-generated answer (if supported)
            timeout: Request timeout in seconds
            
        Returns:
            SearchResponse with results and optional answer
        """
        ...
    
    @abstractmethod
    def validate_config(self) -> str | None:
        """Validate backend configuration.
        
        Returns:
            None if valid, error message if invalid
        """
        ...
