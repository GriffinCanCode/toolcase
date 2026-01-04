STREAMING = """
TOPIC: streaming
================

Progress and result streaming for long-running operations.

PROGRESS STREAMING:
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
    status(msg)              Status update
    step(msg, current, total) Progress step with counts
    source_found(url, desc)  Found a data source
    complete(msg)            Successful completion
    error(msg, code)         Error occurred

RESULT STREAMING:
    from toolcase import StreamEvent, StreamChunk
    
    class StreamingTool(BaseTool[Params]):
        async def stream_result(self, params) -> AsyncIterator[StreamChunk]:
            for token in llm_stream(params.prompt):
                yield StreamChunk(content=token, index=i)

STREAM ADAPTERS:
    from toolcase import sse_adapter, ws_adapter, json_lines_adapter
    
    # Server-Sent Events format
    async for event in sse_adapter(stream):
        print(event)
    
    # WebSocket format
    async for msg in ws_adapter(stream):
        await websocket.send(msg)
    
    # JSON Lines format
    async for line in json_lines_adapter(stream):
        print(line)

RELATED TOPICS:
    toolcase help tool       Tool creation
    toolcase help pipeline   Streaming pipelines
"""
