"""Transport adapters for streaming tool results.

Provides SSE, WebSocket, JSON Lines, and binary adapters for delivering
streaming results via different protocols with high-performance codecs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from toolcase.foundation.errors import JsonDict, JsonValue, StreamEventDict

from .codec import Codec, fast_encode, get_codec
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
    """Format streams as Server-Sent Events. Uses orjson when available."""
    
    __slots__ = ()
    
    def format_event(self, event: StreamEvent) -> str:
        """Format as SSE event (event: <type>\ndata: <json>\n\n)."""
        return f"event: {event.kind}\ndata: {event.to_json()}\n\n"
    
    def format_chunk(self, chunk: StreamChunk, tool_name: str) -> str:
        """Format chunk as SSE data event."""
        return self.format_event(StreamEvent(kind=StreamEventKind.CHUNK, tool_name=tool_name, data=chunk))
    
    def format_start(self, tool_name: str) -> str:
        return self.format_event(StreamEvent(kind=StreamEventKind.START, tool_name=tool_name))
    
    def format_complete(self, tool_name: str, accumulated: str) -> str:
        return self.format_event(StreamEvent(kind=StreamEventKind.COMPLETE, tool_name=tool_name, accumulated=accumulated))
    
    def format_error(self, tool_name: str, error: str) -> str:
        return self.format_event(StreamEvent(kind=StreamEventKind.ERROR, tool_name=tool_name, error=error))


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Adapter
# ─────────────────────────────────────────────────────────────────────────────

class WebSocketAdapter:
    """Format streams for WebSocket delivery. Uses orjson when available."""
    
    __slots__ = ()
    
    def format_event(self, event: StreamEvent) -> str:
        """Format as JSON string."""
        return event.to_json()
    
    def format_chunk(self, chunk: StreamChunk, tool_name: str) -> str:
        return StreamEvent(kind=StreamEventKind.CHUNK, tool_name=tool_name, data=chunk).to_json()
    
    def format_message(self, tool_name: str, kind: StreamEventKind, **kwargs: JsonValue) -> str:
        return fast_encode({"kind": kind.value, "tool": tool_name, **kwargs}).decode()


# ─────────────────────────────────────────────────────────────────────────────
# JSON Lines Adapter
# ─────────────────────────────────────────────────────────────────────────────

class JSONLinesAdapter:
    """Format streams as newline-delimited JSON (NDJSON). Uses orjson when available."""
    
    __slots__ = ()
    
    def format_event(self, event: StreamEvent) -> str:
        return f"{event.to_json()}\n"
    
    def format_chunk(self, chunk: StreamChunk, tool_name: str) -> str:
        return self.format_event(StreamEvent(kind=StreamEventKind.CHUNK, tool_name=tool_name, data=chunk))


class BinaryAdapter:
    """Binary adapter for high-throughput scenarios (msgpack/protobuf).
    
    Returns bytes instead of strings. Ideal for WebSocket binary frames
    or custom binary protocols.
    
    Example:
        >>> adapter = BinaryAdapter("msgpack")
        >>> data = adapter.format_event(event)  # bytes
        >>> ws.send_bytes(data)
    """
    
    __slots__ = ("_codec",)
    
    def __init__(self, codec: str | Codec = "msgpack") -> None:
        self._codec = get_codec(codec) if isinstance(codec, str) else codec
    
    @property
    def content_type(self) -> str:
        return self._codec.content_type
    
    def format_event(self, event: StreamEvent) -> bytes:
        return event.to_bytes(self._codec)
    
    def format_chunk(self, chunk: StreamChunk, tool_name: str) -> bytes:
        return StreamEvent(kind=StreamEventKind.CHUNK, tool_name=tool_name, data=chunk).to_bytes(self._codec)
    
    def decode_event(self, data: bytes) -> StreamEventDict:
        """Decode received binary data."""
        return self._codec.decode(data)  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Singleton Instances
# ─────────────────────────────────────────────────────────────────────────────

sse_adapter = SSEAdapter()
ws_adapter = WebSocketAdapter()
json_lines_adapter = JSONLinesAdapter()

# Binary adapter factory (not singleton - codec may vary)
def binary_adapter(codec: str = "msgpack") -> BinaryAdapter:
    """Create binary adapter with specified codec."""
    return BinaryAdapter(codec)


# ─────────────────────────────────────────────────────────────────────────────
# Stream Transform Helpers
# ─────────────────────────────────────────────────────────────────────────────

TextAdapter = SSEAdapter | WebSocketAdapter | JSONLinesAdapter


async def adapt_stream(
    stream: AsyncIterator[str | StreamChunk],
    tool_name: str,
    adapter: TextAdapter,
) -> AsyncIterator[str]:
    """Transform raw stream into formatted transport events (text adapters)."""
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


async def adapt_stream_binary(
    stream: AsyncIterator[str | StreamChunk],
    tool_name: str,
    adapter: BinaryAdapter | None = None,
) -> AsyncIterator[bytes]:
    """Transform raw stream into binary transport events (msgpack/protobuf).
    
    Ideal for high-throughput scenarios with binary WebSocket frames.
    
    Args:
        stream: Raw stream of content
        tool_name: Tool identifier
        adapter: Binary adapter (default: msgpack)
    
    Yields:
        Encoded bytes for each event
    """
    adapter = adapter or BinaryAdapter("msgpack")
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
