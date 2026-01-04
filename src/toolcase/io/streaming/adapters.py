"""Transport adapters for streaming tool results.

Provides SSE, WebSocket, and JSON Lines adapters for delivering
streaming results to frontends via different protocols.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from toolcase.foundation.errors import JsonDict, JsonValue

from .stream import StreamChunk, StreamEvent, StreamEventKind

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@runtime_checkable
class StreamAdapter(Protocol):
    """Protocol for stream transport adapters."""
    
    def format_event(self, event: StreamEvent) -> str:
        """Format a stream event for transport."""
        ...
    
    def format_chunk(self, chunk: StreamChunk, tool_name: str) -> str:
        """Format a content chunk for transport."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# SSE (Server-Sent Events) Adapter
# ─────────────────────────────────────────────────────────────────────────────

class SSEAdapter:
    """Format streams as Server-Sent Events. SSE format: event: <type>\ndata: <json>\n\n"""
    
    __slots__ = ()
    
    def format_event(self, event: StreamEvent) -> str:
        """Format as SSE event."""
        return f"event: {event.kind}\ndata: {json.dumps(event.to_dict())}\n\n"
    
    def format_chunk(self, chunk: StreamChunk, tool_name: str) -> str:
        """Format chunk as SSE data event."""
        return self.format_event(StreamEvent(kind=StreamEventKind.CHUNK, tool_name=tool_name, data=chunk))
    
    def format_start(self, tool_name: str) -> str:
        """Format stream start event."""
        return self.format_event(StreamEvent(kind=StreamEventKind.START, tool_name=tool_name))
    
    def format_complete(self, tool_name: str, accumulated: str) -> str:
        """Format stream complete event."""
        return self.format_event(StreamEvent(kind=StreamEventKind.COMPLETE, tool_name=tool_name, accumulated=accumulated))
    
    def format_error(self, tool_name: str, error: str) -> str:
        """Format error event."""
        return self.format_event(StreamEvent(kind=StreamEventKind.ERROR, tool_name=tool_name, error=error))


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Adapter
# ─────────────────────────────────────────────────────────────────────────────

class WebSocketAdapter:
    """Format streams for WebSocket delivery. Uses JSON messages with explicit type field for client routing."""
    
    __slots__ = ()
    
    def format_event(self, event: StreamEvent) -> str:
        """Format as JSON message."""
        return event.to_json()
    
    def format_chunk(self, chunk: StreamChunk, tool_name: str) -> str:
        """Format chunk as JSON message."""
        return StreamEvent(kind=StreamEventKind.CHUNK, tool_name=tool_name, data=chunk).to_json()
    
    def format_message(self, tool_name: str, kind: StreamEventKind, **kwargs: JsonValue) -> str:
        """Create a formatted WebSocket message."""
        return json.dumps({"kind": kind.value, "tool": tool_name, **kwargs})


# ─────────────────────────────────────────────────────────────────────────────
# JSON Lines Adapter
# ─────────────────────────────────────────────────────────────────────────────

class JSONLinesAdapter:
    """Format streams as newline-delimited JSON (NDJSON). Each line is a complete JSON object."""
    
    __slots__ = ()
    
    def format_event(self, event: StreamEvent) -> str:
        """Format as JSON line."""
        return f"{event.to_json()}\n"
    
    def format_chunk(self, chunk: StreamChunk, tool_name: str) -> str:
        """Format chunk as JSON line."""
        return self.format_event(StreamEvent(kind=StreamEventKind.CHUNK, tool_name=tool_name, data=chunk))


# ─────────────────────────────────────────────────────────────────────────────
# Singleton Instances
# ─────────────────────────────────────────────────────────────────────────────

sse_adapter = SSEAdapter()
ws_adapter = WebSocketAdapter()
json_lines_adapter = JSONLinesAdapter()


# ─────────────────────────────────────────────────────────────────────────────
# Stream Transform Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def adapt_stream(
    stream: AsyncIterator[str | StreamChunk],
    tool_name: str,
    adapter: SSEAdapter | WebSocketAdapter | JSONLinesAdapter,
) -> AsyncIterator[str]:
    """Transform a raw stream into formatted transport events with start/chunk/complete sequence."""
    yield adapter.format_event(StreamEvent(kind=StreamEventKind.START, tool_name=tool_name))
    accumulated, idx = [], 0
    try:
        async for item in stream:
            chunk = StreamChunk(content=item, index=idx) if isinstance(item, str) else item
            accumulated.append(chunk.content)
            yield adapter.format_chunk(chunk, tool_name)
            idx += 1
        yield adapter.format_event(StreamEvent(kind=StreamEventKind.COMPLETE, tool_name=tool_name, accumulated="".join(accumulated)))
    except Exception as e:
        yield adapter.format_event(StreamEvent(kind=StreamEventKind.ERROR, tool_name=tool_name, error=str(e)))
        raise
