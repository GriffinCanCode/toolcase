"""Standardized error handling for tools.

Provides error codes and structured error responses for agent feedback.
Uses Pydantic for validation and serialization with enhanced features.
"""

from __future__ import annotations

import traceback
from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    field_validator,
)


class ErrorCode(StrEnum):
    """Standard error codes for tool failures.
    
    Used for programmatic error handling and retry decisions.
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


# Flattened pattern -> code mapping for O(1) dict lookup after pattern extraction
_PATTERN_CODES: dict[str, ErrorCode] = {
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
_PATTERN_KEYS = tuple(_PATTERN_CODES.keys())  # Ordered for priority


@lru_cache(maxsize=256)
def _classify_cached(exc_key: str) -> ErrorCode:
    """Cached classification by exception signature."""
    haystack = exc_key.lower()
    for pattern in _PATTERN_KEYS:
        if pattern in haystack:
            return _PATTERN_CODES[pattern]
    return ErrorCode.EXTERNAL_SERVICE_ERROR


def classify_exception(exc: Exception) -> ErrorCode:
    """Map exception to error code via pattern matching on name/message."""
    return _classify_cached(f"{type(exc).__name__} {exc}")


class ToolError(BaseModel):
    """Structured error response for tool failures.
    
    Attributes:
        tool_name: Name of the tool that failed
        message: Human-readable error message
        code: Machine-readable error code for programmatic handling
        recoverable: Whether the error might succeed on retry
        details: Optional detailed information (e.g., stack trace)
    """

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
        json_schema_extra={
            "title": "Tool Error",
            "description": "Structured error from tool execution",
            "examples": [{
                "tool_name": "web_search",
                "message": "Rate limit exceeded",
                "code": "RATE_LIMITED",
                "recoverable": True,
            }],
        },
    )

    tool_name: Annotated[str, Field(
        min_length=1,
        description="Name of the tool that produced the error",
    )]
    message: Annotated[str, Field(
        min_length=1,
        description="Human-readable error message",
    )]
    code: ErrorCode = Field(
        default=ErrorCode.UNKNOWN,
        description="Machine-readable error classification",
    )
    recoverable: bool = Field(
        default=True,
        description="Whether retry might succeed",
    )
    details: str | None = Field(
        default=None,
        description="Optional detailed error info (e.g., stack trace)",
    )
    
    @field_validator("message", mode="before")
    @classmethod
    def _ensure_message(cls, v: str | Exception) -> str:
        """Accept Exception objects and extract message."""
        return str(v) if isinstance(v, Exception) else v
    
    @computed_field
    @property
    def is_retryable(self) -> bool:
        """Whether this error is typically retryable (rate limits, timeouts, network)."""
        return self.code in _RETRYABLE_CODES
    
    @computed_field
    @property
    def is_auth_error(self) -> bool:
        """Whether this is an authentication/authorization error."""
        return self.code in (ErrorCode.API_KEY_MISSING, ErrorCode.API_KEY_INVALID, ErrorCode.PERMISSION_DENIED)
    
    @computed_field
    @property
    def severity(self) -> str:
        """Error severity level for logging/display."""
        if self.code in (ErrorCode.RATE_LIMITED, ErrorCode.TIMEOUT):
            return "warning"
        if self.code in (ErrorCode.API_KEY_MISSING, ErrorCode.API_KEY_INVALID, ErrorCode.PERMISSION_DENIED):
            return "critical"
        return "error"

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
        """Factory method for construction."""
        return cls(tool_name=tool_name, message=message, code=code, recoverable=recoverable, details=details)

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
        """Create from exception with auto-classification."""
        return cls(
            tool_name=tool_name,
            message=f"{context}: {exc}" if context else str(exc),
            code=classify_exception(exc),
            recoverable=recoverable,
            details=traceback.format_exc() if include_trace else None,
        )

    def render(self) -> str:
        """Format error for LLM consumption."""
        parts = [f"**Tool Error ({self.tool_name}):** {self.message}"]
        if self.recoverable:
            parts.append("\n_This error may be recoverable - consider retrying or trying an alternative approach._")
        if self.details:
            parts.append(f"\n\nDetails:\n```\n{self.details}\n```")
        return "".join(parts)

    __str__ = render


# Pre-computed retryable codes set for O(1) lookup
_RETRYABLE_CODES: frozenset[ErrorCode] = frozenset({
    ErrorCode.RATE_LIMITED,
    ErrorCode.TIMEOUT,
    ErrorCode.NETWORK_ERROR,
})


class ToolException(Exception):
    """Exception wrapping a ToolError for raising."""

    __slots__ = ("error",)

    def __init__(self, error: ToolError) -> None:
        self.error = error
        super().__init__(error.message)

    @classmethod
    def create(cls, tool_name: str, message: str, code: ErrorCode = ErrorCode.UNKNOWN, *, recoverable: bool = True) -> Self:
        """Create tool exception."""
        return cls(ToolError(tool_name=tool_name, message=message, code=code, recoverable=recoverable))

    @classmethod
    def from_exc(cls, tool_name: str, exc: Exception, context: str = "") -> Self:
        """Fast path: create from exception without trace."""
        return cls(ToolError(
            tool_name=tool_name,
            message=f"{context}: {exc}" if context else str(exc),
            code=classify_exception(exc),
        ))
