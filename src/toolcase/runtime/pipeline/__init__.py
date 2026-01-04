"""Tool composition via pipelines.

Sequential: tool1 >> tool2 >> tool3
Parallel: parallel(tool1, tool2, tool3)
Streaming: streaming_pipeline(tool1, tool2)  # async generator propagation

Example:
    >>> from toolcase import tool
    >>> from toolcase.pipeline import pipeline, parallel, streaming_pipeline
    >>>
    >>> @tool(description="Search the web for information")
    ... def search(query: str) -> str:
    ...     return f"Results for: {query}"
    >>>
    >>> @tool(description="Summarize text content", streaming=True)
    ... async def summarize(input: str) -> AsyncIterator[str]:
    ...     for word in f"Summary of: {input}".split():
    ...         yield word + " "
    >>>
    >>> # Sequential composition
    >>> pipe = search >> summarize
    >>> # Or explicit:
    >>> pipe = pipeline(search, summarize)
    >>>
    >>> # Parallel execution
    >>> multi = parallel(search, search, merge=lambda rs: "\\n".join(rs))
    >>>
    >>> # Streaming pipeline (propagates async generators)
    >>> stream_pipe = streaming_pipeline(search, summarize)
    >>> async for chunk in stream_pipe.stream_result(params):
    ...     print(chunk, end="", flush=True)
"""

from .pipe import (
    # Transform types
    ChunkTransform,
    Merge,
    StreamMerge,
    StreamTransform,
    Transform,
    # Step types
    Step,
    StreamStep,
    # Tool classes
    ParallelTool,
    PipelineTool,
    StreamingParallelTool,
    StreamingPipelineTool,
    # Param schemas
    ParallelParams,
    PipelineParams,
    # Factory functions
    parallel,
    pipeline,
    streaming_parallel,
    streaming_pipeline,
    # Utilities
    concat_merge,
    identity_chunk,
    identity_dict,
    interleave_streams,
)

__all__ = [
    # Transform types
    "Transform",
    "ChunkTransform",
    "StreamTransform",
    "Merge",
    "StreamMerge",
    # Step types
    "Step",
    "StreamStep",
    # Tool classes
    "PipelineTool",
    "ParallelTool",
    "StreamingPipelineTool",
    "StreamingParallelTool",
    # Param schemas
    "PipelineParams",
    "ParallelParams",
    # Factory functions
    "pipeline",
    "parallel",
    "streaming_pipeline",
    "streaming_parallel",
    # Utilities
    "identity_dict",
    "identity_chunk",
    "concat_merge",
    "interleave_streams",
]
