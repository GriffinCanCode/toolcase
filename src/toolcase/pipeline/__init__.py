"""Tool composition via pipelines.

Sequential: tool1 >> tool2 >> tool3
Parallel: parallel(tool1, tool2, tool3)

Example:
    >>> from toolcase import tool
    >>> from toolcase.pipeline import pipeline, parallel
    >>>
    >>> @tool(description="Search the web for information")
    ... def search(query: str) -> str:
    ...     return f"Results for: {query}"
    >>>
    >>> @tool(description="Summarize text content")
    ... def summarize(input: str) -> str:
    ...     return f"Summary of: {input}"
    >>>
    >>> # Sequential composition
    >>> pipe = search >> summarize
    >>> # Or explicit:
    >>> pipe = pipeline(search, summarize)
    >>>
    >>> # Parallel execution
    >>> multi = parallel(search, search, merge=lambda rs: "\\n".join(rs))
"""

from .pipe import (
    Merge,
    ParallelParams,
    ParallelTool,
    PipelineParams,
    PipelineTool,
    Step,
    Transform,
    concat_merge,
    identity_dict,
    parallel,
    pipeline,
)

__all__ = [
    # Types
    "Transform",
    "Merge",
    "Step",
    # Tools
    "PipelineTool",
    "ParallelTool",
    # Params
    "PipelineParams",
    "ParallelParams",
    # Factories
    "pipeline",
    "parallel",
    # Utilities
    "identity_dict",
    "concat_merge",
]
