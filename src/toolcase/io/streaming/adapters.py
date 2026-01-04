"""Transport adapters for streaming tool results.

Provides SSE, WebSocket, and JSON Lines adapters for delivering
streaming results to frontends via different protocols.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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
    """Format streams as Server-Sent Events.
    
    SSE format:
        event: <event_type>
        data: <json_payload>
        
    Perfect for browser EventSource consumption.
    """
    
    __slots__ = ()
    
    def format_event(self, event: StreamEvent) -> str:
        """Format as SSE event."""
        data = json.dumps(event.to_dict())
        return f"event: {event.kind}\ndata: {data}\n\n"
    
    def format_chunk(self, chunk: StreamChunk, tool_name: str) -> str:
        """Format chunk as SSE data event."""
        event = StreamEvent(
            kind=StreamEventKind.CHUNK,
            tool_name=tool_name,
            data=chunk,
        )
        return self.format_event(event)
    
    def format_start(self, tool_name: str) -> str:
        """Format stream start event."""
        event = StreamEvent(kind=StreamEventKind.START, tool_name=tool_name)
        return self.format_event(event)
    
    def format_complete(self, tool_name: str, accumulated: str) -> str:
        """Format stream complete event."""
        event = StreamEvent(
            kind=StreamEventKind.COMPLETE,
            tool_name=tool_name,
            accumulated=accumulated,
        )
        return self.format_event(event)
    
    def format_error(self, tool_name: str, error: str) -> str:
        """Format error event."""
        event = StreamEvent(
            kind=StreamEventKind.ERROR,
            tool_name=tool_name,
            error=error,
        )
        return self.format_event(event)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Adapter
# ─────────────────────────────────────────────────────────────────────────────

class WebSocketAdapter:
    """Format streams for WebSocket delivery.
    
    Uses JSON messages with explicit type field for client routing.
    Compatible with any WebSocket library (websockets, FastAPI, etc.)
    """
    
    __slots__ = ()
    
    def format_event(self, event: StreamEvent) -> str:
        """Format as JSON message."""
        return event.to_json()
    
    def format_chunk(self, chunk: StreamChunk, tool_name: str) -> str:
        """Format chunk as JSON message."""
        event = StreamEvent(
            kind=StreamEventKind.CHUNK,
            tool_name=tool_name,
            data=chunk,
        )
        return event.to_json()
    
    def format_message(self, tool_name: str, kind: StreamEventKind, **kwargs: object) -> str:
        """Create a formatted WebSocket message."""
        msg: dict[str, object] = {"kind": kind, "tool": tool_name, **kwargs}
        return json.dumps(msg)


# ─────────────────────────────────────────────────────────────────────────────
# JSON Lines Adapter
# ─────────────────────────────────────────────────────────────────────────────

class JSONLinesAdapter:
    """Format streams as newline-delimited JSON (NDJSON).
    
    Each line is a complete JSON object - simple to parse and
    works well with standard streaming HTTP responses.
    """
    
    __slots__ = ()
    
    def format_event(self, event: StreamEvent) -> str:
        """Format as JSON line."""
        return event.to_json() + "\n"
    
    def format_chunk(self, chunk: StreamChunk, tool_name: str) -> str:
        """Format chunk as JSON line."""
        event = StreamEvent(
            kind=StreamEventKind.CHUNK,
            tool_name=tool_name,
            data=chunk,
        )
        return event.to_json() + "\n"


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
    """Transform a raw stream into formatted transport events.
    
    Wraps chunks in proper start/chunk/complete event sequence
    with the specified adapter format.
    
    Args:
        stream: Raw async stream yielding strings or StreamChunks
        tool_name: Name of the streaming tool
        adapter: Transport adapter to use for formatting
    
    Yields:
        Formatted strings ready for transport
    
    Example:
        >>> async for msg in adapt_stream(tool_stream, "llm_gen", sse_adapter):
        ...     await response.write(msg)
    """
    # Emit start
    yield adapter.format_event(
        StreamEvent(kind=StreamEventKind.START, tool_name=tool_name)
    )
    
    accumulated: list[str] = []
    index = 0
    
    try:
        async for item in stream:
            # Normalize to StreamChunk
            if isinstance(item, str):
                item = StreamChunk(content=item, index=index)
            
            accumulated.append(item.content)
            yield adapter.format_chunk(item, tool_name)
            index += 1
        
        # Emit complete
        yield adapter.format_event(
            StreamEvent(
                kind=StreamEventKind.COMPLETE,
                tool_name=tool_name,
                accumulated="".join(accumulated),
            )
        )
    except Exception as e:
        # Emit error
        yield adapter.format_event(
            StreamEvent(
                kind=StreamEventKind.ERROR,
                tool_name=tool_name,
                error=str(e),
            )
        )
        raise
