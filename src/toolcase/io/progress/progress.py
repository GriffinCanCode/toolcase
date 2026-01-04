"""Progress streaming for long-running tool operations.

Enables tools to emit real-time progress updates during execution,
allowing UIs to display meaningful feedback to users.

Uses Pydantic for validation and serialization.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    computed_field,
    field_validator,
    model_validator,
)

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


class ToolProgress(BaseModel):
    """Progress event emitted during tool execution.
    
    These events provide real-time updates on long-running operations.
    Uses Pydantic for automatic validation and JSON serialization.
    
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
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "title": "Tool Progress Event",
            "examples": [
                {"kind": "step", "message": "Processing item 2 of 5", "step": 2, "total_steps": 5},
                {"kind": "complete", "message": "Done", "percentage": 100.0},
            ],
        },
    )
    
    kind: ProgressKind
    message: str = ""
    step: Annotated[int, Field(ge=1)] | None = None
    total_steps: Annotated[int, Field(ge=1)] | None = None
    percentage: Annotated[float, Field(ge=0.0, le=100.0)] | None = None
    data: dict[str, object] = Field(default_factory=dict)
    
    @model_validator(mode="after")
    def _auto_calculate_percentage(self) -> "ToolProgress":
        """Auto-calculate percentage from step/total if not provided."""
        if self.percentage is None and self.step and self.total_steps:
            # Need to use object.__setattr__ because model is frozen
            object.__setattr__(self, "percentage", (self.step / self.total_steps) * 100)
        return self
    
    @computed_field
    @property
    def is_terminal(self) -> bool:
        """Whether this is a terminal event (complete or error)."""
        return self.kind in (ProgressKind.COMPLETE, ProgressKind.ERROR)
    
    @computed_field
    @property
    def is_success(self) -> bool:
        """Whether this is a successful completion."""
        return self.kind == ProgressKind.COMPLETE
    
    def to_dict(self) -> dict[str, object]:
        """Serialize for SSE/JSON transmission."""
        return self.model_dump(exclude_none=True, exclude_defaults=False)


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
