"""HTML Parse Tool - Extract structured content from HTML.

Tools for parsing HTML and extracting specific elements:
- Text extraction
- Link extraction  
- Table extraction
- Metadata extraction
- CSS selector queries

Example:
    >>> from toolcase.tools.prebuilt.web import HtmlParseTool
    >>> parse = HtmlParseTool()
    >>> result = await parse.acall(html="<html>...", extract="links")
"""

from __future__ import annotations

from typing import ClassVar, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field

from toolcase.foundation.core import ToolMetadata
from toolcase.foundation.errors import ErrorCode, Ok, ToolResult, tool_err

from ...core.base import ConfigurableTool, ToolConfig

ExtractMode = Literal["text", "links", "tables", "metadata", "selector"]


class HtmlParseConfig(ToolConfig):
    """Configuration for HtmlParseTool."""
    
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)
    
    max_links: int = Field(default=100, ge=1, le=1000, description="Max links to extract")
    max_tables: int = Field(default=20, ge=1, le=100, description="Max tables to extract")
    strip_whitespace: bool = Field(default=True, description="Strip excess whitespace from text")


class HtmlParseParams(BaseModel):
    """Parameters for HTML parsing."""
    
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    
    html: str = Field(..., description="HTML content to parse")
    extract: ExtractMode = Field(default="text", description="What to extract")
    selector: str | None = Field(default=None, description="CSS selector (required for 'selector' mode)")
    base_url: str | None = Field(default=None, description="Base URL for resolving relative links")


class HtmlParseTool(ConfigurableTool[HtmlParseParams, HtmlParseConfig]):
    """Parse HTML and extract structured content.
    
    Modes:
    - **text**: Extract all text content
    - **links**: Extract all links with text and href
    - **tables**: Extract tables as structured data
    - **metadata**: Extract title, description, OpenGraph, etc.
    - **selector**: Query specific elements with CSS selector
    """
    
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="html_parse",
        description="Parse HTML and extract text, links, tables, or metadata. Supports CSS selectors.",
        category="web",
        requires_api_key=False,
        streaming=False,
        tags=frozenset({"html", "parse", "extract", "scrape"}),
    )
    params_schema: ClassVar[type[HtmlParseParams]] = HtmlParseParams
    config_class: ClassVar[type[HtmlParseConfig]] = HtmlParseConfig
    
    async def _async_run_result(self, params: HtmlParseParams) -> ToolResult:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return tool_err(self.metadata.name, "beautifulsoup4 required: pip install beautifulsoup4", ErrorCode.INVALID_PARAMS)
        
        soup = BeautifulSoup(params.html, "html.parser")
        
        match params.extract:
            case "text":
                return Ok(self._format_text(self._extract_text(soup)))
            case "links":
                return Ok(self._format_links(self._extract_links(soup, params.base_url)))
            case "tables":
                return Ok(self._format_tables(self._extract_tables(soup)))
            case "metadata":
                return Ok(self._format_metadata(self._extract_metadata(soup)))
            case "selector":
                if not params.selector:
                    return tool_err(self.metadata.name, "selector required for 'selector' mode", ErrorCode.INVALID_PARAMS)
                return Ok(self._format_selector(self._extract_selector(soup, params.selector)))
    
    def _extract_text(self, soup) -> dict:
        # Remove noise
        [tag.decompose() for tag in soup(["script", "style", "noscript"])]
        text = soup.get_text(separator="\n", strip=self.config.strip_whitespace)
        lines = [l.strip() for l in text.split("\n") if l.strip()] if self.config.strip_whitespace else text.split("\n")
        return {"text": "\n".join(lines), "line_count": len(lines), "char_count": len(text)}
    
    def _extract_links(self, soup, base_url: str | None) -> dict:
        links = [{"text": a.get_text(strip=True) or "[no text]", 
                  "href": urljoin(base_url, a["href"]) if base_url else a["href"], 
                  "title": a.get("title")} 
                 for a in soup.find_all("a", href=True)[:self.config.max_links]]
        external_prefixes = ("http://", "https://", "//")
        internal = sum(1 for l in links if not l["href"].startswith(external_prefixes))
        return {"links": links, "total": len(links), "internal": internal, "external": len(links) - internal}
    
    def _extract_tables(self, soup) -> dict:
        tables = []
        for table in soup.find_all("table")[:self.config.max_tables]:
            thead = table.find("thead")
            fr = table.find("tr")
            headers = ([th.get_text(strip=True) for th in thead.find_all(["th", "td"])] if thead
                       else [th.get_text(strip=True) for th in fr.find_all("th")] if fr and fr.find("th") else [])
            rows = [dict(zip(headers, cells)) if headers else cells
                    for tr in table.find_all("tr")
                    if (cells := [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]) and cells != headers]
            if rows:
                tables.append({"headers": headers, "rows": rows, "row_count": len(rows)})
        return {"tables": tables, "total": len(tables)}
    
    def _extract_metadata(self, soup) -> dict:
        meta = {"title": soup.find("title").get_text(strip=True) if soup.find("title") else None,
                "description": None, "og": {}, "twitter": {}, "meta_tags": []}
        for tag in soup.find_all("meta"):
            name, prop, content = tag.get("name", "").lower(), tag.get("property", "").lower(), tag.get("content", "")
            if name == "description": meta["description"] = content
            elif prop.startswith("og:"): meta["og"][prop[3:]] = content
            elif name.startswith("twitter:"): meta["twitter"][name[8:]] = content
            elif name or prop: meta["meta_tags"].append({"name": name or prop, "content": content})
        return meta
    
    def _extract_selector(self, soup, selector: str) -> dict:
        try:
            elements = soup.select(selector)
        except Exception as e:
            return {"error": f"Invalid selector: {e}", "matches": []}
        matches = [{"tag": el.name, "text": el.get_text(strip=True), "attrs": dict(el.attrs), "html": str(el)[:500]}
                   for el in elements[:50]]
        return {"selector": selector, "matches": matches, "total": len(elements)}
    
    # Format methods for string output
    def _format_text(self, data: dict) -> str:
        return f"**Extracted Text** ({data['line_count']} lines, {data['char_count']} chars)\n\n{data['text']}"
    
    def _format_links(self, data: dict) -> str:
        header = f"**Links:** {data['total']} total ({data['internal']} internal, {data['external']} external)\n"
        return "\n".join([header] + [f"{i}. [{l['text']}]({l['href']})" for i, l in enumerate(data["links"][:50], 1)])
    
    def _format_tables(self, data: dict) -> str:
        if not data["tables"]:
            return "**Tables:** None found"
        lines = [f"**Tables:** {data['total']} found\n"]
        for i, t in enumerate(data["tables"], 1):
            lines.extend([f"\n### Table {i} ({t['row_count']} rows)"] + 
                         ([f"Headers: {', '.join(t['headers'])}"] if t["headers"] else []) +
                         [f"  - {row}" for row in t["rows"][:5]])
        return "\n".join(lines)
    
    def _format_metadata(self, data: dict) -> str:
        lines = ["**Page Metadata:**"]
        if data["title"]: lines.append(f"- Title: {data['title']}")
        if data["description"]: lines.append(f"- Description: {data['description']}")
        if data["og"]: lines.append(f"- OpenGraph: {data['og']}")
        if data["twitter"]: lines.append(f"- Twitter: {data['twitter']}")
        return "\n".join(lines)
    
    def _format_selector(self, data: dict) -> str:
        header = f"**CSS Selector:** `{data['selector']}` - {data['total']} matches\n"
        return "\n".join([header] + [f"{i}. <{m['tag']}> {m['text'][:100]}" for i, m in enumerate(data["matches"][:20], 1)])
    
    async def _async_run(self, params: HtmlParseParams) -> str:
        from toolcase.foundation.errors import result_to_string
        return result_to_string(await self._async_run_result(params), self.metadata.name)
