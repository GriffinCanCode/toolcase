"""Progress streaming for long-running tool operations.

Enables tools to emit real-time progress updates during execution,
allowing UIs to display meaningful feedback to users.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping


class ProgressKind(StrEnum):
    """Types of progress events a tool can emit."""
    STATUS = "status"           # General status message
    STEP = "step"               # Discrete step completed
    SOURCE_FOUND = "source"     # Found a data source (e.g., search result)
    DATA = "data"               # Intermediate data available
    COMPLETE = "complete"       # Tool finished successfully
    ERROR = "error"             # Tool encountered an error


@dataclass(slots=True, kw_only=True)
class ToolProgress:
    """Progress event emitted during tool execution.
    
    These events provide real-time updates on long-running operations.
    
    Attributes:
        kind: Type of progress event
        message: Human-readable status message
        step: Current step number (1-indexed)
        total_steps: Total number of steps (if known)
        percentage: Completion percentage (0-100)
        data: Arbitrary payload for this event
    
    Example:
        >>> progress = ToolProgress(kind=ProgressKind.STEP, message="Fetching page 2", step=2, total_steps=5)
        >>> progress.percentage  # Auto-calculated
        40.0
    """
    kind: ProgressKind
    message: str = ""
    step: int | None = None
    total_steps: int | None = None
    percentage: float | None = None
    data: dict[str, object] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        # Auto-calculate percentage from step/total if not provided
        if self.percentage is None and self.step and self.total_steps:
            self.percentage = (self.step / self.total_steps) * 100
    
    def to_dict(self) -> dict[str, object]:
        """Serialize for SSE/JSON transmission."""
        result: dict[str, object] = {"kind": self.kind, "message": self.message}
        
        if self.step is not None:
            result["step"] = self.step
        if self.total_steps is not None:
            result["total_steps"] = self.total_steps
        if self.percentage is not None:
            result["percentage"] = self.percentage
        if self.data:
            result["data"] = self.data
        
        return result


# Factory functions for common progress events
def status(message: str, **data: object) -> ToolProgress:
    """Create a status progress event."""
    return ToolProgress(kind=ProgressKind.STATUS, message=message, data=dict(data))


def step(message: str, current: int, total: int, **data: object) -> ToolProgress:
    """Create a step progress event."""
    return ToolProgress(
        kind=ProgressKind.STEP,
        message=message,
        step=current,
        total_steps=total,
        data=dict(data),
    )


def source_found(message: str, source: Mapping[str, object]) -> ToolProgress:
    """Create a source-found progress event."""
    return ToolProgress(
        kind=ProgressKind.SOURCE_FOUND,
        message=message,
        data=dict(source),
    )


def complete(result: str, message: str = "Complete") -> ToolProgress:
    """Create a completion progress event."""
    return ToolProgress(
        kind=ProgressKind.COMPLETE,
        message=message,
        percentage=100.0,
        data={"result": result},
    )


def error(message: str, **data: object) -> ToolProgress:
    """Create an error progress event."""
    return ToolProgress(kind=ProgressKind.ERROR, message=message, data=dict(data))


@runtime_checkable
class ProgressCallback(Protocol):
    """Protocol for progress event handlers."""
    
    def __call__(self, progress: ToolProgress) -> None: ...
