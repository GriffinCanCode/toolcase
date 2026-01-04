"""Progress streaming for long-running tool operations.

Enables tools to emit real-time progress updates during execution,
allowing UIs to display meaningful feedback to users.

Optimizations:
- Frozen models for immutability and hashability
- TypeAdapter for fast validation
- model_construct for hot path creation
- Pre-computed terminal states
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    computed_field,
    model_validator,
)

from toolcase.foundation.errors import JsonDict, JsonValue

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


# Pre-computed terminal kinds for fast lookup
_TERMINAL_KINDS = frozenset((ProgressKind.COMPLETE, ProgressKind.ERROR))

# Empty dict singleton to avoid allocations
_EMPTY_DATA: JsonDict = {}


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
        extra="forbid",
        revalidate_instances="never",
        use_enum_values=True,  # Store string values directly for JSON
        json_schema_extra={
            "title": "Tool Progress Event",
            "examples": [
                {"kind": "step", "message": "Processing item 2 of 5", "step": 2, "total_steps": 5},
                {"kind": "complete", "message": "Done", "percentage": 100.0},
            ],
        },
    )
    
    __slots__ = ()
    
    kind: ProgressKind
    message: str = ""
    step: Annotated[int, Field(ge=1)] | None = None
    total_steps: Annotated[int, Field(ge=1)] | None = None
    percentage: Annotated[float, Field(ge=0.0, le=100.0)] | None = None
    data: JsonDict = Field(default_factory=dict, repr=False)
    
    @model_validator(mode="after")
    def _auto_calculate_percentage(self) -> "ToolProgress":
        """Auto-calculate percentage from step/total if not provided."""
        if self.percentage is None and self.step and self.total_steps:
            object.__setattr__(self, "percentage", self.step / self.total_steps * 100)
        return self
    
    @computed_field
    @property
    def is_terminal(self) -> bool:
        """Whether this is a terminal event (complete or error)."""
        return self.kind in _TERMINAL_KINDS
    
    @computed_field
    @property
    def is_success(self) -> bool:
        """Whether this is a successful completion."""
        return self.kind == ProgressKind.COMPLETE
    
    def __hash__(self) -> int:
        """Hash for frozen model."""
        return hash((self.kind, self.message, self.step, self.percentage))
    
    def to_dict(self) -> JsonDict:
        """Serialize for SSE/JSON transmission."""
        return self.model_dump(exclude_none=True, exclude_defaults=False)


# TypeAdapter for fast validation from dicts
_ToolProgressAdapter: TypeAdapter[ToolProgress] = TypeAdapter(ToolProgress)


# Factory functions using model_construct for hot paths (bypasses validation)
_construct = ToolProgress.model_construct


def status(message: str, **data: JsonValue) -> ToolProgress:
    """Create a status progress event (fast path)."""
    return _construct(kind=ProgressKind.STATUS, message=message, step=None, total_steps=None, percentage=None, data={**data} or _EMPTY_DATA)


def step(message: str, current: int, total: int, **data: JsonValue) -> ToolProgress:
    """Create a step progress event with auto-calculated percentage (fast path)."""
    return _construct(kind=ProgressKind.STEP, message=message, step=current, total_steps=total, percentage=current / total * 100 if total else None, data={**data} or _EMPTY_DATA)


def source_found(message: str, source: Mapping[str, object]) -> ToolProgress:
    """Create a source-found progress event (fast path)."""
    return _construct(kind=ProgressKind.SOURCE_FOUND, message=message, step=None, total_steps=None, percentage=None, data=dict(source))


def complete(result: str, message: str = "Complete") -> ToolProgress:
    """Create a completion progress event (fast path)."""
    return _construct(kind=ProgressKind.COMPLETE, message=message, step=None, total_steps=None, percentage=100.0, data={"result": result})


def error(message: str, **data: JsonValue) -> ToolProgress:
    """Create an error progress event (fast path)."""
    return _construct(kind=ProgressKind.ERROR, message=message, step=None, total_steps=None, percentage=None, data={**data} or _EMPTY_DATA)


def validate_progress(data: JsonDict) -> ToolProgress:
    """Validate a dict as ToolProgress (use when validation is needed)."""
    return _ToolProgressAdapter.validate_python(data)


@runtime_checkable
class ProgressCallback(Protocol):
    """Protocol for progress event handlers."""
    
    def __call__(self, progress: ToolProgress) -> None: ...
