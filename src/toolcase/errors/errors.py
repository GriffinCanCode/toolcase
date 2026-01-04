"""Standardized error handling for tools.

Provides error codes and structured error responses for agent feedback.
"""

from __future__ import annotations

import traceback
from enum import StrEnum
from functools import lru_cache
from typing import Self

from pydantic import BaseModel


class ErrorCode(StrEnum):
    """Standard error codes for tool failures."""
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
    """Structured error response for tool failures."""

    model_config = {"frozen": True}

    tool_name: str
    message: str
    code: ErrorCode = ErrorCode.UNKNOWN
    recoverable: bool = True
    details: str | None = None

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

    @property
    def is_retryable(self) -> bool:
        """Whether this error is typically retryable (rate limits, timeouts, network)."""
        return self.code in _RETRYABLE_CODES


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
