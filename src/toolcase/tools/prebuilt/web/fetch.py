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

import asyncio
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from toolcase.foundation.core import ToolMetadata
from toolcase.foundation.errors import Err, ErrorCode, ErrorTrace, Ok, ToolResult

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


def _err(msg: str, code: ErrorCode, op: str, recoverable: bool = False) -> ToolResult:
    return Err(ErrorTrace(message=msg, error_code=code.value, recoverable=recoverable).with_operation(op))


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
            return _err("httpx required: pip install httpx", ErrorCode.INVALID_PARAMS, "import")
        
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
                    return _err(f"Content exceeds {self.config.max_content_length} bytes", ErrorCode.RATE_LIMITED, "fetch")
                
                html = resp.text
                content = self._extract_content(html, mode)
                
                return Ok(self._format_result(str(resp.url), resp.status_code, mode, content))
        except httpx.TimeoutException:
            return _err(f"Request timed out after {self.config.timeout}s", ErrorCode.TIMEOUT, "fetch", recoverable=True)
        except httpx.HTTPStatusError as e:
            return _err(f"HTTP {e.response.status_code}: {e.response.reason_phrase}", ErrorCode.EXTERNAL_SERVICE_ERROR, "fetch")
        except Exception as e:
            return _err(f"Fetch failed: {e}", ErrorCode.EXTERNAL_SERVICE_ERROR, "fetch", recoverable=True)
    
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
            return html if mode == "html" else f"[beautifulsoup4 required for {mode} mode]\n{html[:2000]}"
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script/style elements
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        
        if mode == "text":
            return soup.get_text(separator="\n", strip=True)
        
        # Markdown mode
        return self._html_to_markdown(soup)
    
    def _html_to_markdown(self, soup) -> str:
        """Convert BeautifulSoup element to markdown."""
        lines = []
        
        # Title
        if title := soup.find("title"):
            lines.extend([f"# {title.get_text(strip=True)}", ""])
        
        # Process main content
        main = soup.find("main") or soup.find("article") or soup.find("body") or soup
        
        for elem in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "a", "pre", "code"]):
            text = elem.get_text(strip=True)
            if not text:
                continue
            
            match elem.name:
                case "h1": lines.append(f"# {text}")
                case "h2": lines.append(f"## {text}")
                case "h3": lines.append(f"### {text}")
                case "h4": lines.append(f"#### {text}")
                case "p": lines.append(text)
                case "li": lines.append(f"- {text}")
                case "a": 
                    href = elem.get("href", "")
                    if href and not href.startswith("#"):
                        lines.append(f"[{text}]({href})")
                case "pre" | "code": lines.append(f"```\n{text}\n```")
        
        return "\n\n".join(lines)
    
    async def _async_run(self, params: UrlFetchParams) -> str:
        from toolcase.foundation.errors import result_to_string
        return result_to_string(await self._async_run_result(params), self.metadata.name)
