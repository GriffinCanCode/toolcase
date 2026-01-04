STREAMING = """
TOPIC: streaming
================

Progress and result streaming for long-running operations.

PROGRESS STREAMING (Status Updates):
    from toolcase import ToolProgress, status, step, complete
    
    class LongRunningTool(BaseTool[Params]):
        metadata = ToolMetadata(..., streaming=True)
        
        async def stream_run(self, params) -> AsyncIterator[ToolProgress]:
            yield status("Starting...")
            
            for i, item in enumerate(params.items, 1):
                result = await process(item)
                yield step(
                    f"Processed {item}",
                    current=i,
                    total=len(params.items)
                )
            
            yield complete("Done!")

PROGRESS KINDS:
    status(msg)                Status update
    step(msg, current, total)  Progress step with counts
    source_found(url, desc)    Found a data source
    complete(msg)              Successful completion
    error(msg, code)           Error occurred

RESULT STREAMING (Incremental Output):
    from toolcase import tool, StreamChunk
    
    @tool(description="Generate report", streaming=True)
    async def generate(topic: str) -> AsyncIterator[str]:
        async for token in llm.stream(f"Report on {topic}"):
            yield token
    
    # Consumer
    async for chunk in registry.stream_execute("generate", {"topic": "AI"}):
        print(chunk, end="", flush=True)

STREAM EVENTS:
    from toolcase import StreamEvent, StreamEventKind, stream_start, stream_complete
    
    async for event in tool.stream_result_events(params):
        match event.kind:
            case StreamEventKind.START:
                print(f"Starting {event.tool_name}")
            case StreamEventKind.CHUNK:
                print(event.data.content, end="")
            case StreamEventKind.COMPLETE:
                print(f"\\nDone: {event.data}")
            case StreamEventKind.ERROR:
                print(f"Error: {event.data}")

STREAM ADAPTERS (Transport):
    from toolcase.io.streaming import (
        sse_adapter, ws_adapter, json_lines_adapter, binary_adapter
    )
    
    # Server-Sent Events format
    async for event in sse_adapter(stream):
        print(event)  # "data: chunk\\n\\n"
    
    # WebSocket format
    async for msg in ws_adapter(stream):
        await websocket.send(msg)
    
    # JSON Lines format
    async for line in json_lines_adapter(stream):
        print(line)  # {"chunk": "content"}
    
    # Binary (msgpack) format - efficient for high-throughput
    async for data in binary_adapter(stream):
        send_binary(data)
    
    # Adapt any stream to a transport format
    from toolcase.io.streaming import adapt_stream, adapt_stream_binary
    adapted = adapt_stream(stream, "sse")       # or "ws", "jsonl"
    adapted = adapt_stream_binary(stream)       # msgpack encoding

RESULT-BASED STREAMING (Error Handling):
    from toolcase.io.streaming import (
        result_stream, ok_chunk, err_chunk,
        filter_ok, filter_err, collect_or_first_error,
        map_ok, tap_ok, recover
    )
    
    # Wrap stream with Result semantics
    async def resilient_stream(source):
        async for item in source:
            try:
                yield ok_chunk(process(item))
            except Exception as e:
                yield err_chunk(str(e))
    
    # Filter only successful chunks
    async for chunk in filter_ok(resilient_stream(source)):
        print(chunk)
    
    # Collect all or fail on first error
    result = await collect_or_first_error(stream)
    if result.is_ok():
        print(f"All chunks: {result.unwrap()}")
    else:
        print(f"Failed: {result.unwrap_err()}")
    
    # Map over successful chunks
    async for chunk in map_ok(stream, lambda x: x.upper()):
        print(chunk)
    
    # Recover from errors
    async for chunk in recover(stream, lambda e: f"[RECOVERED: {e}]"):
        print(chunk)

STREAMING WITH BACKPRESSURE:
    @tool(description="Generate", streaming=True, backpressure_buffer=10)
    async def generate(topic: str) -> AsyncIterator[str]:
        async for chunk in llm.stream(topic):
            yield chunk  # Pauses when buffer full
    
    # Consumer controls pace - producer won't overwhelm memory
    async for chunk in registry.stream_execute("generate", params):
        await slow_process(chunk)

HIGH-PERFORMANCE CODECS:
    from toolcase.io.streaming import encode, decode, pack, unpack
    
    # orjson (JSON, ~3x faster than stdlib)
    data = encode({"key": "value"})   # bytes
    obj = decode(data)                 # dict
    
    # msgpack (binary, ~40% smaller)
    data = pack({"key": "value"})      # bytes
    obj = unpack(data)                 # dict

STREAM RESULT COLLECTION:
    from toolcase import StreamResult
    
    # Collect stream with metadata
    result: StreamResult[str] = await tool.stream_result_collected(params)
    print(f"Value: {result.value}")
    print(f"Chunks: {result.chunks}")
    print(f"Duration: {result.duration_ms:.0f}ms")
    print(f"Tool: {result.tool_name}")

RELATED TOPICS:
    toolcase help tool       Tool creation
    toolcase help pipeline   Streaming pipelines
    toolcase help middleware Streaming middleware
"""
