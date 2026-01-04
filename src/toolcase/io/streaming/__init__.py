"""True result streaming for tools producing incremental output.

Unlike progress streaming (status updates), result streaming delivers
actual output chunks as they're generated - perfect for LLM-powered tools.

Features high-performance serialization with orjson (JSON) and msgpack (binary).

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

from .codec import (
    Codec,
    CodecType,
    MsgpackCodec,
    OrjsonCodec,
    decode,
    encode,
    encode_str,
    get_codec,
    pack,
    register_codec,
    unpack,
)
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
    BinaryAdapter,
    StreamAdapter,
    adapt_stream,
    adapt_stream_binary,
    binary_adapter,
    json_lines_adapter,
    sse_adapter,
    ws_adapter,
)
from .result import (
    ResultStream,
    ChunkResult,
    StreamCollectResult,
    ok_chunk,
    err_chunk,
    err_chunk_from_exc,
    result_stream,
    result_stream_resilient,
    unwrap_stream,
    filter_ok,
    filter_err,
    collect_result_stream,
    collect_or_first_error,
    map_ok,
    map_err,
    tap_ok,
    tap_err,
    recover,
)

__all__ = [
    # Codecs (high-performance serialization)
    "Codec", "CodecType", "OrjsonCodec", "MsgpackCodec",
    "get_codec", "register_codec",
    "encode", "decode", "encode_str",  # orjson
    "pack", "unpack",  # msgpack
    # Core types
    "StreamChunk", "StreamEvent", "StreamEventKind", "StreamState", "StreamResult",
    # Factory functions
    "chunk", "stream_start", "stream_complete", "stream_error",
    # Adapters for transport
    "StreamAdapter", "BinaryAdapter",
    "sse_adapter", "ws_adapter", "json_lines_adapter", "binary_adapter",
    "adapt_stream", "adapt_stream_binary",
    # Result-based streaming
    "ResultStream", "ChunkResult", "StreamCollectResult",
    "ok_chunk", "err_chunk", "err_chunk_from_exc",
    "result_stream", "result_stream_resilient", "unwrap_stream",
    "filter_ok", "filter_err", "collect_result_stream", "collect_or_first_error",
    "map_ok", "map_err", "tap_ok", "tap_err", "recover",
]
