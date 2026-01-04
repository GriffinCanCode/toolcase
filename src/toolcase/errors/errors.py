"""Standardized error handling for tools.

Provides error codes and structured error responses that tools can return
to give agents actionable feedback about failures.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field


class ErrorCode(StrEnum):
    """Standard error codes for tool failures.
    
    Using StrEnum allows these to serialize cleanly and be pattern-matched.
    """
    API_KEY_MISSING = "API_KEY_MISSING"
    API_KEY_INVALID = "API_KEY_INVALID"
    RATE_LIMITED = "RATE_LIMITED"
    NETWORK_ERROR = "NETWORK_ERROR"
    TIMEOUT = "TIMEOUT"
    INVALID_PARAMS = "INVALID_PARAMS"
    NO_RESULTS = "NO_RESULTS"
    PARSE_ERROR = "PARSE_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    NOT_FOUND = "NOT_FOUND"
    UNKNOWN = "UNKNOWN"


# Exception type -> ErrorCode mapping for automatic classification
_EXCEPTION_PATTERNS: dict[str, ErrorCode] = {
    "timeout": ErrorCode.TIMEOUT,
    "connection": ErrorCode.NETWORK_ERROR,
    "network": ErrorCode.NETWORK_ERROR,
    "rate": ErrorCode.RATE_LIMITED,
    "limit": ErrorCode.RATE_LIMITED,
    "auth": ErrorCode.API_KEY_INVALID,
    "permission": ErrorCode.PERMISSION_DENIED,
    "forbidden": ErrorCode.PERMISSION_DENIED,
    "parse": ErrorCode.PARSE_ERROR,
    "json": ErrorCode.PARSE_ERROR,
    "decode": ErrorCode.PARSE_ERROR,
    "validation": ErrorCode.INVALID_PARAMS,
    "value": ErrorCode.INVALID_PARAMS,
    "notfound": ErrorCode.NOT_FOUND,
}


def classify_exception(exc: Exception) -> ErrorCode:
    """Map exception type to appropriate error code."""
    exc_name = type(exc).__name__.lower()
    exc_msg = str(exc).lower()
    
    for pattern, code in _EXCEPTION_PATTERNS.items():
        if pattern in exc_name or pattern in exc_msg:
            return code
    
    return ErrorCode.EXTERNAL_SERVICE_ERROR


class ToolError(BaseModel):
    """Structured error response for tool failures.
    
    Provides consistent error information for LLM consumption and programmatic handling.
    The `recoverable` flag guides retry/fallback behavior.
    
    Example:
        >>> error = ToolError.create("web_search", "API rate limited", ErrorCode.RATE_LIMITED)
        >>> print(error.render())
        **Tool Error (web_search):** API rate limited
        _This error may be recoverable - consider retrying or trying an alternative approach._
    """
    
    tool_name: str = Field(..., description="Tool that encountered the error")
    message: str = Field(..., description="Human-readable error message")
    code: ErrorCode = Field(default=ErrorCode.UNKNOWN, description="Machine-readable error code")
    recoverable: bool = Field(default=True, description="Whether retry/fallback may succeed")
    details: str | None = Field(default=None, description="Stack trace or additional context")
    
    @classmethod
    def create(
        cls,
        tool_name: str,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        *,
        recoverable: bool = True,
        details: str | None = None,
    ) -> Self:
        """Factory method for cleaner construction."""
        return cls(
            tool_name=tool_name,
            message=message,
            code=code,
            recoverable=recoverable,
            details=details,
        )
    
    @classmethod
    def from_exception(
        cls,
        tool_name: str,
        exc: Exception,
        context: str = "",
        *,
        recoverable: bool = True,
        include_trace: bool = True,
    ) -> Self:
        """Create error from caught exception with automatic code classification."""
        import traceback
        
        msg = f"{context}: {exc}" if context else str(exc)
        details = traceback.format_exc() if include_trace else None
        
        return cls(
            tool_name=tool_name,
            message=msg,
            code=classify_exception(exc),
            recoverable=recoverable,
            details=details,
        )
    
    def render(self) -> str:
        """Format error for LLM consumption."""
        lines = [f"**Tool Error ({self.tool_name}):** {self.message}"]
        
        if self.recoverable:
            lines.append(
                "_This error may be recoverable - consider retrying or trying an alternative approach._"
            )
        
        if self.details:
            lines.append(f"\nDetails:\n```\n{self.details}\n```")
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return self.render()


class ToolException(Exception):
    """Exception that wraps a ToolError for raising."""
    
    __slots__ = ("error",)
    
    def __init__(self, error: ToolError) -> None:
        self.error = error
        super().__init__(error.message)
    
    @classmethod
    def create(
        cls,
        tool_name: str,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        *,
        recoverable: bool = True,
    ) -> Self:
        """Create and raise a tool exception."""
        return cls(ToolError.create(tool_name, message, code, recoverable=recoverable))
