"""Type aliases and error context tracking for monadic error handling.

Provides:
- Type aliases for common Result patterns
- Error context stacking for provenance tracking
- Error trace accumulation through call chains
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .result import Result

# ═════════════════════════════════════════════════════════════════════════════
# Type Aliases
# ═════════════════════════════════════════════════════════════════════════════

# Common Result patterns for tool operations
ResultT: TypeAlias = "Result[str, str]"  # Tool result: success string or error string


# ═════════════════════════════════════════════════════════════════════════════
# Error Context & Provenance Tracking
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class ErrorContext:
    """Context information for an error at a specific call site.
    
    Tracks where and why an error occurred, enabling error provenance
    tracing through nested function calls.
    
    Attributes:
        operation: Description of the operation that failed
        location: Module/function name where error occurred
        metadata: Additional context-specific information
    
    Example:
        >>> ctx = ErrorContext(
        ...     operation="parse_user_input",
        ...     location="auth.validators",
        ...     metadata={"input_type": "email"}
        ... )
    """
    
    operation: str
    location: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Format as human-readable context."""
        parts = [self.operation]
        if self.location:
            parts.append(f"at {self.location}")
        if self.metadata:
            meta_str = ", ".join(f"{k}={v}" for k, v in self.metadata.items())
            parts.append(f"({meta_str})")
        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class ErrorTrace:
    """Stack of error contexts forming a trace through the call chain.
    
    Accumulates context as errors propagate up, providing a breadcrumb
    trail of what operations were attempted.
    
    This enables railway-oriented error handling with full provenance tracking.
    
    Attributes:
        message: Primary error message
        contexts: Stack of contexts from innermost to outermost
        error_code: Optional machine-readable error code
        recoverable: Whether the error might be transient
    
    Example:
        >>> trace = ErrorTrace(
        ...     message="Invalid API key",
        ...     contexts=[
        ...         ErrorContext("validate_credentials", "auth.service"),
        ...         ErrorContext("initialize_client", "api.client"),
        ...     ],
        ...     error_code="AUTH_INVALID",
        ...     recoverable=False,
        ... )
        >>> print(trace.format())
        Invalid API key
        
        Context trace:
        - validate_credentials at auth.service
        - initialize_client at api.client
    """
    
    message: str
    contexts: list[ErrorContext] = field(default_factory=list)
    error_code: str | None = None
    recoverable: bool = True
    details: str | None = None
    
    def with_context(self, context: ErrorContext) -> ErrorTrace:
        """Add context to trace (returns new trace).
        
        Creates a new trace with the context appended, preserving immutability.
        """
        return ErrorTrace(
            message=self.message,
            contexts=[*self.contexts, context],
            error_code=self.error_code,
            recoverable=self.recoverable,
            details=self.details,
        )
    
    def with_operation(
        self,
        operation: str,
        location: str = "",
        **metadata: object,
    ) -> ErrorTrace:
        """Convenience method to add context with operation info."""
        return self.with_context(
            ErrorContext(operation=operation, location=location, metadata=metadata)
        )
    
    def format(self, *, include_details: bool = False) -> str:
        """Format trace as human-readable string.
        
        Args:
            include_details: Whether to include full error details
        
        Returns:
            Formatted error trace with context stack
        """
        lines = [self.message]
        
        if self.error_code:
            lines.append(f"[{self.error_code}]")
        
        if self.contexts:
            lines.append("\nContext trace:")
            for ctx in self.contexts:
                lines.append(f"  - {ctx}")
        
        if self.recoverable:
            lines.append("\n(This error may be recoverable)")
        
        if include_details and self.details:
            lines.append(f"\nDetails:\n{self.details}")
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        """String representation."""
        return self.format()


# ═════════════════════════════════════════════════════════════════════════════
# Error Context Helpers
# ═════════════════════════════════════════════════════════════════════════════


def context(
    operation: str,
    location: str = "",
    **metadata: object,
) -> ErrorContext:
    """Create ErrorContext with concise syntax.
    
    Example:
        >>> ctx = context("parse_config", location="config.loader", format="yaml")
    """
    return ErrorContext(operation=operation, location=location, metadata=metadata)


def trace(
    message: str,
    *,
    code: str | None = None,
    recoverable: bool = True,
    details: str | None = None,
) -> ErrorTrace:
    """Create ErrorTrace with concise syntax.
    
    Example:
        >>> err = trace("Connection failed", code="NETWORK_ERROR", recoverable=True)
    """
    return ErrorTrace(
        message=message,
        error_code=code,
        recoverable=recoverable,
        details=details,
    )
