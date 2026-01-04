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
from toolcase.foundation.errors import Err, ErrorCode, ErrorTrace, Ok, ToolResult

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


def _err(msg: str, code: ErrorCode, op: str) -> ToolResult:
    return Err(ErrorTrace(message=msg, error_code=code.value).with_operation(op))


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
        if params.pattern:
            pattern = params.pattern
        elif params.common_pattern:
            pattern = COMMON_PATTERNS[params.common_pattern]
        else:
            return _err("Either 'pattern' or 'common_pattern' required", ErrorCode.INVALID_PARAMS, "extract")
        
        # Compile regex
        flags = 0 if (params.case_sensitive if params.case_sensitive is not None else self.config.case_sensitive) else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return _err(f"Invalid regex: {e}", ErrorCode.INVALID_PARAMS, "compile")
        
        # Extract matches
        matches = []
        for i, match in enumerate(regex.finditer(params.text)):
            if i >= self.config.max_matches:
                break
            
            result = {
                "match": match.group(),
                "start": match.start(),
                "end": match.end(),
            }
            
            # Include named/numbered groups if present
            if match.groups():
                result["groups"] = match.groups()
            if match.groupdict():
                result["named_groups"] = {k: v for k, v in match.groupdict().items() if v is not None}
            
            if params.with_context:
                ctx_start = max(0, match.start() - 50)
                ctx_end = min(len(params.text), match.end() + 50)
                result["context"] = params.text[ctx_start:ctx_end]
            
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
        lines = [header]
        for i, m in enumerate(matches[:30], 1):
            lines.append(f"{i}. `{m['match'][:100]}`")
        if len(matches) > 30:
            lines.append(f"... and {len(matches) - 30} more")
        return "\n".join(lines)
    
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
        json_pattern = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]")
        matches = json_pattern.findall(params.text)
        
        parsed = []
        for m in matches:
            try:
                data = json.loads(m)
                parsed.append(data)
            except json.JSONDecodeError:
                continue
        
        if not parsed:
            return _err("No valid JSON found in text", ErrorCode.INVALID_PARAMS, "extract")
        
        result = parsed[0] if len(parsed) == 1 else parsed
        
        # Apply path query if provided
        if params.path:
            try:
                result = self._query_path(result, params.path)
            except (KeyError, IndexError, TypeError) as e:
                return _err(f"Path query failed: {e}", ErrorCode.INVALID_PARAMS, "query")
        
        return Ok(f"**JSON Extract:** Found {len(parsed)} JSON object(s)\n\n```json\n{json.dumps(result, indent=2)[:2000]}\n```")
    
    def _query_path(self, data, path: str):
        """Simple JSONPath-like query."""
        parts = re.split(r"\.|\[|\]", path)
        for part in parts:
            if not part:
                continue
            if part.isdigit():
                data = data[int(part)]
            else:
                data = data[part]
        return data
    
    async def _async_run(self, params: Params) -> str:
        from toolcase.foundation.errors import result_to_string
        return result_to_string(await self._async_run_result(params), self.metadata.name)
