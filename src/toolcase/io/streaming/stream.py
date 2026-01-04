"""Core streaming types for incremental result delivery.

Provides typed events for streaming tool outputs with metadata for
state tracking, error handling, and transport serialization.

Uses Pydantic models for validation and serialization where appropriate.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Generic, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    computed_field,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

T = TypeVar("T")


class StreamEventKind(StrEnum):
    """Types of streaming events."""
    START = "start"        # Stream initialized
    CHUNK = "chunk"        # Content chunk
    COMPLETE = "complete"  # Stream finished successfully
    ERROR = "error"        # Stream encountered error


class StreamState(StrEnum):
    """Stream lifecycle states."""
    PENDING = "pending"
    STREAMING = "streaming"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass(slots=True, frozen=True)
class StreamChunk:
    """A single chunk of streamed content.
    
    Attributes:
        content: The actual string content
        index: Chunk sequence number (0-indexed)
        timestamp: When chunk was generated (epoch ms)
        metadata: Optional per-chunk data (e.g., token counts)
    """
    content: str
    index: int = 0
    timestamp: float = field(default_factory=lambda: time.time() * 1000)
    metadata: dict[str, object] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.content)
    
    def to_dict(self) -> dict[str, object]:
        """Serialize for JSON transport."""
        return {
            "content": self.content,
            "index": self.index,
            "timestamp": self.timestamp,
            **({"metadata": self.metadata} if self.metadata else {}),
        }


@dataclass(slots=True)
class StreamEvent:
    """Wrapper event for stream lifecycle management.
    
    Attributes:
        kind: Event type (start, chunk, complete, error)
        tool_name: Name of the streaming tool
        data: Event payload - StreamChunk for chunks, error info for errors
        accumulated: Total content streamed so far (only on complete)
        error: Error message if kind=error
    """
    kind: StreamEventKind
    tool_name: str
    data: StreamChunk | None = None
    accumulated: str | None = None
    error: str | None = None
    timestamp: float = field(default_factory=lambda: time.time() * 1000)
    
    def to_dict(self) -> dict[str, object]:
        """Serialize for JSON transport."""
        result: dict[str, object] = {
            "kind": self.kind,
            "tool": self.tool_name,
            "timestamp": self.timestamp,
        }
        if self.data:
            result["data"] = self.data.to_dict()
        if self.accumulated is not None:
            result["accumulated"] = self.accumulated
        if self.error:
            result["error"] = self.error
        return result
    
    def to_json(self) -> str:
        """JSON string for transport."""
        return json.dumps(self.to_dict())


@dataclass(slots=True)
class StreamResult(Generic[T]):
    """Final result of a streaming operation.
    
    Captures the full accumulated output plus streaming metadata.
    
    Attributes:
        value: Complete accumulated result
        chunks: Number of chunks streamed
        duration_ms: Total streaming duration
        tool_name: Source tool
    """
    value: T
    chunks: int
    duration_ms: float
    tool_name: str
    
    @property
    def success(self) -> bool:
        return self.value is not None


# ─────────────────────────────────────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────────────────────────────────────

def chunk(content: str, index: int = 0, **metadata: object) -> StreamChunk:
    """Create a content chunk."""
    return StreamChunk(content=content, index=index, metadata=dict(metadata))


def stream_start(tool_name: str) -> StreamEvent:
    """Create a stream start event."""
    return StreamEvent(kind=StreamEventKind.START, tool_name=tool_name)


def stream_complete(tool_name: str, accumulated: str) -> StreamEvent:
    """Create a stream complete event with final accumulated content."""
    return StreamEvent(
        kind=StreamEventKind.COMPLETE,
        tool_name=tool_name,
        accumulated=accumulated,
    )


def stream_error(tool_name: str, error: str) -> StreamEvent:
    """Create a stream error event."""
    return StreamEvent(kind=StreamEventKind.ERROR, tool_name=tool_name, error=error)


# ─────────────────────────────────────────────────────────────────────────────
# Stream Collector
# ─────────────────────────────────────────────────────────────────────────────

async def collect_stream(
    stream: AsyncIterator[str | StreamChunk],
    tool_name: str,
) -> StreamResult[str]:
    """Collect all chunks from a stream into a final result.
    
    Useful for consuming a streaming tool when you want the full output.
    
    Args:
        stream: Async iterator yielding strings or StreamChunks
        tool_name: Name of the tool (for result metadata)
    
    Returns:
        StreamResult with accumulated content and stats
    """
    start = time.time()
    parts: list[str] = []
    count = 0
    
    async for item in stream:
        content = item.content if isinstance(item, StreamChunk) else item
        parts.append(content)
        count += 1
    
    return StreamResult(
        value="".join(parts),
        chunks=count,
        duration_ms=(time.time() - start) * 1000,
        tool_name=tool_name,
    )
