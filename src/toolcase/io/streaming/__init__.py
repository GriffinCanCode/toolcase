"""True result streaming for tools producing incremental output.

Unlike progress streaming (status updates), result streaming delivers
actual output chunks as they're generated - perfect for LLM-powered tools.

Example:
    >>> @tool(description="Generate a report", streaming=True)
    ... async def generate_report(topic: str) -> AsyncIterator[str]:
    ...     async for chunk in llm.stream(f"Report on {topic}"):
    ...         yield chunk
    >>>
    >>> # Consumer sees incremental results
    >>> async for chunk in registry.stream_execute("generate_report", {"topic": "AI"}):
    ...     print(chunk, end="", flush=True)
"""

from .stream import (
    StreamChunk,
    StreamEvent,
    StreamEventKind,
    StreamState,
    StreamResult,
    chunk,
    stream_complete,
    stream_error,
    stream_start,
)
from .adapters import (
    sse_adapter,
    ws_adapter,
    json_lines_adapter,
    StreamAdapter,
)

__all__ = [
    # Core types
    "StreamChunk",
    "StreamEvent",
    "StreamEventKind",
    "StreamState",
    "StreamResult",
    # Factory functions
    "chunk",
    "stream_start",
    "stream_complete",
    "stream_error",
    # Adapters for transport
    "StreamAdapter",
    "sse_adapter",
    "ws_adapter",
    "json_lines_adapter",
]
