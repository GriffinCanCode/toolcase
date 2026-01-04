"""Text Extraction Tools - Regex and pattern-based text extraction.

Tools for extracting structured data from text:
- Regex pattern matching
- Common pattern extraction (emails, URLs, phones, etc.)
- JSON extraction from text

Example:
    >>> from toolcase.tools.prebuilt.web import RegexExtractTool
    >>> extract = RegexExtractTool()
    >>> result = await extract.acall(text="...", pattern=r"\\d+")
"""

from __future__ import annotations

import json
import re
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from toolcase.foundation.core import ToolMetadata
from toolcase.foundation.errors import ErrorCode, Ok, ToolResult, tool_err

from ...core.base import ConfigurableTool, ToolConfig

CommonPattern = Literal["emails", "urls", "phones", "dates", "numbers", "json", "markdown_links"]

COMMON_PATTERNS: dict[CommonPattern, str] = {
    "emails": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "urls": r"https?://[^\s<>\"']+",
    "phones": r"(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}",
    "dates": r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\w+\s+\d{1,2},?\s+\d{4}",
    "numbers": r"-?\d+(?:,\d{3})*(?:\.\d+)?",
    "json": r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]",
    "markdown_links": r"\[([^\]]+)\]\(([^)]+)\)",
}


class RegexExtractConfig(ToolConfig):
    """Configuration for RegexExtractTool."""
    
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)
    
    max_matches: int = Field(default=1000, ge=1, le=10000, description="Max matches to return")
    case_sensitive: bool = Field(default=True, description="Case-sensitive matching by default")


class RegexExtractParams(BaseModel):
    """Parameters for regex extraction."""
    
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    
    text: str = Field(..., description="Text to search in")
    pattern: str | None = Field(default=None, description="Regex pattern (or use common_pattern)")
    common_pattern: CommonPattern | None = Field(default=None, description="Use a built-in pattern")
    case_sensitive: bool | None = Field(default=None, description="Override case sensitivity")
    with_context: bool = Field(default=False, description="Include surrounding context for matches")


class RegexExtractTool(ConfigurableTool[RegexExtractParams, RegexExtractConfig]):
    """Extract patterns from text using regex or common patterns.
    
    Common patterns:
    - **emails**: Email addresses
    - **urls**: HTTP/HTTPS URLs
    - **phones**: Phone numbers (US format)
    - **dates**: Common date formats
    - **numbers**: Numeric values
    - **json**: JSON objects/arrays embedded in text
    - **markdown_links**: Markdown-style links [text](url)
    """
    
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="regex_extract",
        description="Extract text patterns using regex or common patterns (emails, URLs, dates, etc.)",
        category="text",
        requires_api_key=False,
        streaming=False,
        tags=frozenset({"regex", "extract", "text", "pattern"}),
    )
    params_schema: ClassVar[type[RegexExtractParams]] = RegexExtractParams
    config_class: ClassVar[type[RegexExtractConfig]] = RegexExtractConfig
    
    async def _async_run_result(self, params: RegexExtractParams) -> ToolResult:
        # Determine pattern
        if not (pattern := params.pattern or (COMMON_PATTERNS.get(params.common_pattern) if params.common_pattern else None)):
            return tool_err(self.metadata.name, "Either 'pattern' or 'common_pattern' required", ErrorCode.INVALID_PARAMS)
        
        # Compile regex
        flags = 0 if (params.case_sensitive if params.case_sensitive is not None else self.config.case_sensitive) else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return tool_err(self.metadata.name, f"Invalid regex: {e}", ErrorCode.INVALID_PARAMS)
        
        # Extract matches
        matches = []
        for i, m in enumerate(regex.finditer(params.text)):
            if i >= self.config.max_matches:
                break
            result = {"match": m.group(), "start": m.start(), "end": m.end()}
            if m.groups():
                result["groups"] = m.groups()
            if named := {k: v for k, v in m.groupdict().items() if v is not None}:
                result["named_groups"] = named
            if params.with_context:
                result["context"] = params.text[max(0, m.start() - 50):min(len(params.text), m.end() + 50)]
            matches.append(result)
        
        # For JSON pattern, try to parse matches
        if params.common_pattern == "json":
            for m in matches:
                try:
                    m["parsed"] = json.loads(m["match"])
                except json.JSONDecodeError:
                    pass
        
        return Ok(self._format_result(pattern, params.common_pattern, matches))
    
    def _format_result(self, pattern: str, pattern_name: str | None, matches: list[dict]) -> str:
        """Format extraction result as string."""
        unique = len({m["match"] for m in matches})
        header = f"**Regex Extract:** `{pattern_name or pattern}`\n_Found {len(matches)} matches ({unique} unique)_\n"
        if not matches:
            return f"{header}\nNo matches found."
        lines = [header] + [f"{i}. `{m['match'][:100]}`" for i, m in enumerate(matches[:30], 1)]
        return "\n".join(lines + ([f"... and {len(matches) - 30} more"] if len(matches) > 30 else []))
    
    async def _async_run(self, params: RegexExtractParams) -> str:
        from toolcase.foundation.errors import result_to_string
        return result_to_string(await self._async_run_result(params), self.metadata.name)


class JsonExtractTool(ConfigurableTool[BaseModel, ToolConfig]):
    """Extract and parse JSON from text content."""
    
    class Params(BaseModel):
        model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
        text: str = Field(..., description="Text containing JSON")
        path: str | None = Field(default=None, description="JSONPath-like query (e.g., 'data.items[0].name')")
    
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="json_extract",
        description="Extract and parse JSON from text. Optionally query with JSONPath.",
        category="text",
        requires_api_key=False,
        streaming=False,
        tags=frozenset({"json", "extract", "parse"}),
    )
    params_schema: ClassVar[type[Params]] = Params
    config_class: ClassVar[type[ToolConfig]] = ToolConfig
    
    async def _async_run_result(self, params: Params) -> ToolResult:
        # Find JSON in text
        matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]", params.text)
        parsed = [obj for m in matches if (obj := self._try_parse(m)) is not None]
        
        if not parsed:
            return tool_err(self.metadata.name, "No valid JSON found in text", ErrorCode.INVALID_PARAMS)
        
        result = parsed[0] if len(parsed) == 1 else parsed
        if params.path:
            try:
                result = self._query_path(result, params.path)
            except (KeyError, IndexError, TypeError) as e:
                return tool_err(self.metadata.name, f"Path query failed: {e}", ErrorCode.INVALID_PARAMS)
        
        return Ok(f"**JSON Extract:** Found {len(parsed)} JSON object(s)\n\n```json\n{json.dumps(result, indent=2)[:2000]}\n```")
    
    @staticmethod
    def _try_parse(text: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    
    def _query_path(self, data, path: str):
        """Simple JSONPath-like query."""
        for part in (p for p in re.split(r"\.|\[|\]", path) if p):
            data = data[int(part)] if part.isdigit() else data[part]
        return data
    
    async def _async_run(self, params: Params) -> str:
        from toolcase.foundation.errors import result_to_string
        return result_to_string(await self._async_run_result(params), self.metadata.name)
