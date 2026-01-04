"""Standardized error handling for tools.

Provides error codes and structured error responses for agent feedback.
Uses Pydantic for validation and serialization with enhanced features.
"""

from __future__ import annotations

import traceback
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    field_validator,
)

if TYPE_CHECKING:
    from pydantic import ValidationError


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


# ═══════════════════════════════════════════════════════════════════════════════
# Validation Error Formatting
# ═══════════════════════════════════════════════════════════════════════════════

# Pydantic error type -> (natural language description, fix suggestion)
_ERROR_TYPE_MESSAGES: dict[str, tuple[str, str]] = {
    # Missing/required
    "missing": ("is required but was not provided", "Please provide a value for this field"),
    # Type errors
    "string_type": ("must be a text string", "Provide a string value instead"),
    "int_type": ("must be a whole number", "Provide an integer without decimals"),
    "int_parsing": ("must be a whole number", "Provide an integer without decimals"),
    "float_type": ("must be a number", "Provide a numeric value"),
    "float_parsing": ("must be a number", "Provide a valid numeric value"),
    "bool_type": ("must be true or false", "Provide a boolean value"),
    "bool_parsing": ("must be true or false", "Provide 'true' or 'false'"),
    "list_type": ("must be a list/array", "Provide a list of values"),
    "dict_type": ("must be an object/dictionary", "Provide a key-value object"),
    "none_not_allowed": ("cannot be null/None", "Provide a non-null value"),
    # String constraints
    "string_too_short": ("is too short", "Provide a longer string"),
    "string_too_long": ("is too long", "Shorten the string"),
    "string_pattern_mismatch": ("does not match the expected format", "Check the format requirements"),
    # Numeric constraints
    "greater_than": ("is too small", "Provide a larger value"),
    "greater_than_equal": ("is too small", "Provide a value at or above the minimum"),
    "less_than": ("is too large", "Provide a smaller value"),
    "less_than_equal": ("is too large", "Provide a value at or below the maximum"),
    # Enum/literal
    "enum": ("is not one of the allowed values", "Use one of the allowed options"),
    "literal_error": ("is not one of the allowed values", "Use one of the allowed options"),
    # URL/JSON
    "url_parsing": ("is not a valid URL", "Provide a properly formatted URL"),
    "url_scheme": ("has an invalid URL scheme", "Use a valid scheme (e.g., https://)"),
    "json_invalid": ("is not valid JSON", "Ensure the JSON syntax is correct"),
    "json_type": ("must be valid JSON", "Provide properly formatted JSON"),
    # Generic
    "value_error": ("has an invalid value", "Check the value constraints"),
    "type_error": ("has the wrong type", "Check the expected type for this field"),
    "extra_forbidden": ("is not a recognized parameter", "Remove this field or check spelling"),
    # UUID
    "uuid_parsing": ("is not a valid UUID", "Provide a valid UUID format"),
    # Date/time
    "datetime_parsing": ("is not a valid datetime", "Provide a valid datetime format"),
    "date_parsing": ("is not a valid date", "Provide a valid date format"),
    "time_parsing": ("is not a valid time", "Provide a valid time format"),
}


def format_validation_error(exc: "ValidationError", *, tool_name: str | None = None) -> str:
    """Format Pydantic ValidationError into LLM-friendly natural language.
    
    Converts raw validation errors into clear, actionable descriptions.
    
    Args:
        exc: Pydantic ValidationError to format
        tool_name: Optional tool name for context
        
    Returns:
        Natural language error description with fix suggestions
        
    Example:
        >>> try:
        ...     Model(count="not a number")
        ... except ValidationError as e:
        ...     print(format_validation_error(e))
        Parameter issue: 'count' must be a whole number (you provided: 'not a number').
        → Provide an integer without decimals.
    """
    if not (errors := exc.errors()):
        return str(exc)
    
    prefix = f"[{tool_name}] " if tool_name else ""
    lines = [f"{prefix}{'Parameter issue:' if len(errors) == 1 else f'{len(errors)} parameter issues:'}"]
    
    for err in errors:
        field = ".".join(str(loc) for loc in err.get("loc", ("unknown",)))
        desc, suggestion = _ERROR_TYPE_MESSAGES.get(err.get("type", "unknown"), ("is invalid", "Check the parameter value"))
        input_val, constraint = err.get("input"), _format_constraint(err.get("type", ""), err.get("ctx", {}))
        
        lines.append(f"  • '{field}' {desc}{f' (you provided: {_format_input(input_val)})' if input_val is not None else ''}{constraint}")
        lines.append(f"    → {suggestion}")
    
    return "\n".join(lines)


def _format_input(val: object, max_len: int = 50) -> str:
    """Format input value for display, truncating if needed."""
    if val is None:
        return "None"
    s = repr(val) if isinstance(val, str) else str(val)
    return s[:max_len] + "..." if len(s) > max_len else s


_CONSTRAINT_FORMATTERS: dict[str, str] = {
    "gt": " — must be > {}", "ge": " — must be ≥ {}", "lt": " — must be < {}", "le": " — must be ≤ {}",
    "min_length": " — minimum length is {}", "max_length": " — maximum length is {}", "pattern": " — must match pattern: {}",
}

def _format_constraint(err_type: str, ctx: dict[str, object]) -> str:
    """Format constraint info from error context."""
    for key, fmt in _CONSTRAINT_FORMATTERS.items():
        if key in ctx:
            return fmt.format(ctx[key])
    if exp := ctx.get("expected"):
        # For enums, show allowed values more clearly
        return f" — allowed values: {exp}" if isinstance(exp, str) and "'" in exp else f" — expected: {exp}"
    return ""


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
        return self.code in _AUTH_CODES
    
    @computed_field
    @property
    def severity(self) -> str:
        """Error severity level for logging/display."""
        return "warning" if self.code in _WARNING_CODES else "critical" if self.code in _AUTH_CODES else "error"

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
        base = f"**Tool Error ({self.tool_name}):** {self.message}"
        recover = "\n_This error may be recoverable - consider retrying or trying an alternative approach._" if self.recoverable else ""
        details = f"\n\nDetails:\n```\n{self.details}\n```" if self.details else ""
        return base + recover + details

    __str__ = render


# Pre-computed code sets for O(1) lookup
_RETRYABLE_CODES: frozenset[ErrorCode] = frozenset({ErrorCode.RATE_LIMITED, ErrorCode.TIMEOUT, ErrorCode.NETWORK_ERROR})
_AUTH_CODES: frozenset[ErrorCode] = frozenset({ErrorCode.API_KEY_MISSING, ErrorCode.API_KEY_INVALID, ErrorCode.PERMISSION_DENIED})
_WARNING_CODES: frozenset[ErrorCode] = frozenset({ErrorCode.RATE_LIMITED, ErrorCode.TIMEOUT})


class ToolException(Exception):
    """Base exception wrapping a ToolError for raising.
    
    Use subclasses for more specific error handling:
    - RetryableToolException: For transient errors (rate limits, timeouts, network)
    - AuthToolException: For authentication/authorization failures
    - ValidationToolException: For parameter validation errors
    """

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
    
    @classmethod
    def from_error(cls, error: ToolError) -> "ToolException":
        """Create appropriate exception subclass based on error code. Returns most specific type."""
        if error.code in _RETRYABLE_CODES:
            return RetryableToolException(error)
        if error.code in _AUTH_CODES:
            return AuthToolException(error)
        return ValidationToolException(error) if error.code == ErrorCode.INVALID_PARAMS else cls(error)


class RetryableToolException(ToolException):
    """Exception for transient errors that may succeed on retry.
    
    Includes: RATE_LIMITED, TIMEOUT, NETWORK_ERROR
    
    Example:
        >>> try:
        ...     result = await tool.arun(params)
        ... except RetryableToolException:
        ...     await asyncio.sleep(1)
        ...     result = await tool.arun(params)  # Retry
    """
    
    __slots__ = ()


class AuthToolException(ToolException):
    """Exception for authentication/authorization failures.
    
    Includes: API_KEY_MISSING, API_KEY_INVALID, PERMISSION_DENIED
    
    These errors typically require user intervention (providing credentials,
    requesting access) rather than automatic retry.
    
    Example:
        >>> try:
        ...     result = await tool.arun(params)
        ... except AuthToolException as e:
        ...     log.error(f"Auth failed: {e.error.message}")
        ...     raise  # Don't retry, escalate to user
    """
    
    __slots__ = ()


class ValidationToolException(ToolException):
    """Exception for parameter validation failures.
    
    Includes: INVALID_PARAMS
    
    These errors indicate the request was malformed and should not be
    retried without fixing the parameters.
    
    Example:
        >>> try:
        ...     result = await tool.arun(params)
        ... except ValidationToolException as e:
        ...     # Fix params based on error details
        ...     fixed_params = fix_params(params, e.error)
        ...     result = await tool.arun(fixed_params)
    """
    
    __slots__ = ()