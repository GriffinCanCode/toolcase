"""URL Fetch Tool - Download web page content with smart extraction.

A robust URL fetcher that:
- Handles common HTTP headers (User-Agent, Accept)
- Follows redirects
- Extracts text content from HTML
- Supports timeouts and retries

Example:
    >>> from toolcase.tools.prebuilt.web import UrlFetchTool
    >>> fetch = UrlFetchTool()
    >>> result = await fetch.acall(url="https://example.com")
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from toolcase.foundation.core import ToolMetadata
from toolcase.foundation.errors import ErrorCode, Ok, ToolResult, tool_err

from ...core.base import ConfigurableTool, ToolConfig

ContentMode = Literal["html", "text", "markdown"]


class UrlFetchConfig(ToolConfig):
    """Configuration for UrlFetchTool."""
    
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)
    
    user_agent: str = Field(
        default="Mozilla/5.0 (compatible; ToolcaseBot/1.0)",
        description="User-Agent header for requests",
    )
    max_content_length: int = Field(
        default=1_000_000, ge=1024, le=10_000_000,
        description="Max response size in bytes",
    )
    default_mode: ContentMode = Field(default="text", description="Default content extraction mode")
    follow_redirects: bool = Field(default=True, description="Follow HTTP redirects")


class UrlFetchParams(BaseModel):
    """Parameters for URL fetching."""
    
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    
    url: str = Field(..., description="URL to fetch")
    mode: ContentMode | None = Field(default=None, description="Content mode: html, text, or markdown")
    headers: dict[str, str] | None = Field(default=None, description="Additional HTTP headers")


class UrlFetchTool(ConfigurableTool[UrlFetchParams, UrlFetchConfig]):
    """Fetch web page content with smart extraction.
    
    Modes:
    - **html**: Raw HTML content
    - **text**: Extracted text (default)
    - **markdown**: Convert to markdown format
    """
    
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="url_fetch",
        description="Fetch and extract content from a URL. Returns raw HTML, extracted text, or markdown.",
        category="web",
        requires_api_key=False,
        streaming=False,
        tags=frozenset({"web", "fetch", "download", "scrape"}),
    )
    params_schema: ClassVar[type[UrlFetchParams]] = UrlFetchParams
    config_class: ClassVar[type[UrlFetchConfig]] = UrlFetchConfig
    
    async def _async_run_result(self, params: UrlFetchParams) -> ToolResult:
        try:
            import httpx
        except ImportError:
            return tool_err(self.metadata.name, "httpx required: pip install httpx", ErrorCode.INVALID_PARAMS)
        
        mode = params.mode or self.config.default_mode
        headers = {"User-Agent": self.config.user_agent, **(params.headers or {})}
        
        try:
            async with httpx.AsyncClient(
                follow_redirects=self.config.follow_redirects,
                timeout=self.config.timeout,
            ) as client:
                resp = await client.get(params.url, headers=headers)
                resp.raise_for_status()
                
                if len(resp.content) > self.config.max_content_length:
                    return tool_err(self.metadata.name, f"Content exceeds {self.config.max_content_length} bytes", ErrorCode.RATE_LIMITED)
                
                return Ok(self._format_result(str(resp.url), resp.status_code, mode, self._extract_content(resp.text, mode)))
        except httpx.TimeoutException:
            return tool_err(self.metadata.name, f"Request timed out after {self.config.timeout}s", ErrorCode.TIMEOUT, recoverable=True)
        except httpx.HTTPStatusError as e:
            return tool_err(self.metadata.name, f"HTTP {e.response.status_code}: {e.response.reason_phrase}", ErrorCode.EXTERNAL_SERVICE_ERROR)
        except Exception as e:
            return tool_err(self.metadata.name, f"Fetch failed: {e}", ErrorCode.EXTERNAL_SERVICE_ERROR, recoverable=True)
    
    def _format_result(self, url: str, status: int, mode: ContentMode, content: str) -> str:
        """Format result as readable string."""
        header = f"**URL Fetch:** `{url}`\n_Status: {status} | Mode: {mode} | {len(content)} chars_\n"
        return f"{header}\n{content}"
    
    def _extract_content(self, html: str, mode: ContentMode) -> str:
        if mode == "html":
            return html
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return f"[beautifulsoup4 required for {mode} mode]\n{html[:2000]}"
        
        soup = BeautifulSoup(html, "html.parser")
        [tag.decompose() for tag in soup(["script", "style", "noscript", "header", "footer", "nav"])]
        return soup.get_text(separator="\n", strip=True) if mode == "text" else self._html_to_markdown(soup)
    
    def _html_to_markdown(self, soup) -> str:
        """Convert BeautifulSoup element to markdown."""
        lines = [f"# {title.get_text(strip=True)}", ""] if (title := soup.find("title")) else []
        main = soup.find("main") or soup.find("article") or soup.find("body") or soup
        
        tag_fmt = {"h1": "# {}", "h2": "## {}", "h3": "### {}", "h4": "#### {}", "p": "{}", "li": "- {}", "pre": "```\n{}\n```", "code": "```\n{}\n```"}
        for elem in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "a", "pre", "code"]):
            if not (text := elem.get_text(strip=True)):
                continue
            if elem.name == "a":
                if (href := elem.get("href", "")) and not href.startswith("#"):
                    lines.append(f"[{text}]({href})")
            elif fmt := tag_fmt.get(elem.name):
                lines.append(fmt.format(text))
        return "\n\n".join(lines)
    
    async def _async_run(self, params: UrlFetchParams) -> str:
        from toolcase.foundation.errors import result_to_string
        return result_to_string(await self._async_run_result(params), self.metadata.name)
