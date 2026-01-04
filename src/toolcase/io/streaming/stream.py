"""Core streaming types for incremental result delivery.

Provides typed events for streaming tool outputs with metadata for
state tracking, error handling, and transport serialization.

Uses high-performance codecs (orjson/msgpack) for serialization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Generic, TypeVar

from toolcase.foundation.errors import JsonDict, JsonValue, StreamChunkDict, StreamEventDict

from .codec import Codec, fast_decode, fast_encode, get_codec

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
    """A single chunk of streamed content with sequence number and optional metadata."""
    content: str
    index: int = 0
    timestamp: float = field(default_factory=lambda: time.time() * 1000)
    metadata: JsonDict = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.content)
    
    def to_dict(self) -> StreamChunkDict:
        """Serialize for JSON transport."""
        d: StreamChunkDict = {"content": self.content, "index": self.index, "timestamp": self.timestamp}
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass(slots=True)
class StreamEvent:
    """Wrapper event for stream lifecycle management (start, chunk, complete, error)."""
    kind: StreamEventKind
    tool_name: str
    data: StreamChunk | None = None
    accumulated: str | None = None
    error: str | None = None
    timestamp: float = field(default_factory=lambda: time.time() * 1000)
    
    def to_dict(self) -> StreamEventDict:
        """Serialize for JSON transport."""
        d: StreamEventDict = {"kind": self.kind.value, "tool": self.tool_name, "timestamp": self.timestamp}
        if self.data:
            d["data"] = self.data.to_dict()
        if self.accumulated is not None:
            d["accumulated"] = self.accumulated
        if self.error:
            d["error"] = self.error
        return d
    
    def to_json(self) -> str:
        """JSON string for transport (uses orjson if available)."""
        return fast_encode(self.to_dict()).decode()
    
    def to_bytes(self, codec: Codec | None = None) -> bytes:
        """Serialize to bytes using specified codec (default: orjson/json)."""
        return (codec or get_codec()).encode(self.to_dict())


@dataclass(slots=True)
class StreamResult(Generic[T]):
    """Final result of a streaming operation with accumulated output and metadata."""
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

def chunk(content: str, index: int = 0, **metadata: JsonValue) -> StreamChunk:
    """Create a content chunk."""
    return StreamChunk(content=content, index=index, metadata=dict(metadata))

def stream_start(tool_name: str) -> StreamEvent:
    """Create a stream start event."""
    return StreamEvent(kind=StreamEventKind.START, tool_name=tool_name)

def stream_complete(tool_name: str, accumulated: str) -> StreamEvent:
    """Create a stream complete event with final accumulated content."""
    return StreamEvent(kind=StreamEventKind.COMPLETE, tool_name=tool_name, accumulated=accumulated)

def stream_error(tool_name: str, error: str) -> StreamEvent:
    """Create a stream error event."""
    return StreamEvent(kind=StreamEventKind.ERROR, tool_name=tool_name, error=error)


# ─────────────────────────────────────────────────────────────────────────────
# Stream Collector
# ─────────────────────────────────────────────────────────────────────────────

async def collect_stream(stream: AsyncIterator[str | StreamChunk], tool_name: str) -> StreamResult[str]:
    """Collect all chunks from a stream into a final result with accumulated content and stats."""
    t0 = time.time()
    parts = [item.content if isinstance(item, StreamChunk) else item async for item in stream]
    return StreamResult(value="".join(parts), chunks=len(parts), duration_ms=(time.time() - t0) * 1000, tool_name=tool_name)
