"""Standardized error handling for tools.

Provides error codes and structured error responses that tools can return
to give agents actionable feedback about failures.
"""

from .errors import ErrorCode, ToolError, ToolException, classify_exception

__all__ = [
    "ErrorCode",
    "ToolError",
    "ToolException",
    "classify_exception",
]
