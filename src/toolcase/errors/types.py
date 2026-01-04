"""Type aliases and error context tracking for monadic error handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .result import Result

# ═══════════════════════════════════════════════════════════════════════════════
# Type Aliases
# ═══════════════════════════════════════════════════════════════════════════════

ResultT: TypeAlias = "Result[str, str]"

# ═══════════════════════════════════════════════════════════════════════════════
# Error Context & Provenance
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class ErrorContext:
    """Context for error at a call site. Tracks operation, location, metadata."""
    
    operation: str
    location: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
    
    def __str__(self) -> str:
        parts = [self.operation]
        if self.location:
            parts.append(f"at {self.location}")
        if self.metadata:
            parts.append(f"({', '.join(f'{k}={v}' for k, v in self.metadata.items())})")
        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class ErrorTrace:
    """Stack of error contexts forming call chain trace for provenance tracking."""
    
    message: str
    contexts: tuple[ErrorContext, ...] = ()  # Tuple for immutability + memory efficiency
    error_code: str | None = None
    recoverable: bool = True
    details: str | None = None
    
    def with_context(self, context: ErrorContext) -> ErrorTrace:
        """Add context to trace (returns new trace, preserves immutability)."""
        return ErrorTrace(self.message, (*self.contexts, context), self.error_code, self.recoverable, self.details)
    
    def with_operation(self, operation: str, location: str = "", **metadata: object) -> ErrorTrace:
        """Add context with operation info."""
        return self.with_context(ErrorContext(operation, location, metadata or {}))
    
    def format(self, *, include_details: bool = False) -> str:
        """Format trace as human-readable string."""
        lines = [self.message]
        if self.error_code:
            lines.append(f"[{self.error_code}]")
        if self.contexts:
            lines.append("\nContext trace:")
            lines.extend(f"  - {ctx}" for ctx in self.contexts)
        if self.recoverable:
            lines.append("\n(This error may be recoverable)")
        if include_details and self.details:
            lines.append(f"\nDetails:\n{self.details}")
        return "\n".join(lines)
    
    __str__ = format


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def context(operation: str, location: str = "", **metadata: object) -> ErrorContext:
    """Create ErrorContext concisely."""
    return ErrorContext(operation, location, metadata or {})


def trace(message: str, *, code: str | None = None, recoverable: bool = True, details: str | None = None) -> ErrorTrace:
    """Create ErrorTrace concisely."""
    return ErrorTrace(message, (), code, recoverable, details)
