TOOL = """
TOPIC: tool
===========

How to create tools in toolcase.

TWO APPROACHES:

1. DECORATOR (Simple tools):
    from toolcase import tool
    
    @tool(description="Add two numbers", category="math")
    def add(a: int, b: int) -> str:
        '''Add two integers.
        
        Args:
            a: First number
            b: Second number
        '''
        return str(a + b)

2. CLASS-BASED (Complex tools - async-first design):
    from toolcase import BaseTool, ToolMetadata
    from pydantic import BaseModel, Field
    
    class SearchParams(BaseModel):
        query: str = Field(..., description="Search query")
        limit: int = Field(default=5, ge=1, le=20)
    
    class SearchTool(BaseTool[SearchParams]):
        metadata = ToolMetadata(
            name="search",
            description="Search the web",
            category="search",
        )
        params_schema = SearchParams
        
        async def _async_run(self, params: SearchParams) -> str:
            return f"Found {params.limit} results for: {params.query}"

KEY ATTRIBUTES:
    metadata        ToolMetadata with name, description, category, tags
    params_schema   Pydantic model for parameter validation
    cache_enabled   Enable/disable caching (default: True)
    cache_ttl       Cache time-to-live in seconds (default: 300)
    retry_policy    Optional RetryPolicy for automatic retries

KEY METHODS (Async-First Design):
    _async_run(params)         Primary execution method (implement this)
    _run(params)               Sync wrapper (auto-calls _async_run)
    arun(params, timeout)      Async with caching and timeout
    run(params)                Sync with caching
    arun_result(params)        Async returning Result type
    run_result(params)         Sync returning Result type
    batch_run(params_list)     Execute multiple params concurrently
    stream_run(params)         Progress streaming (AsyncIterator)
    stream_result(params)      Result streaming for incremental output

BATCH EXECUTION:
    # Run tool against multiple parameter sets concurrently
    params = [SearchParams(query=q) for q in ["python", "rust", "go"]]
    results = await search_tool.batch_run(params, BatchConfig(concurrency=3))
    print(f"Success rate: {results.success_rate:.0%}")

RELATED TOPICS:
    toolcase help result      Error handling with Result types
    toolcase help streaming   Progress streaming
    toolcase help cache       Caching configuration
"""
