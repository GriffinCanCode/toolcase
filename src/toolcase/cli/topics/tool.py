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

2. CLASS-BASED (Complex tools):
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
        
        def _run(self, params: SearchParams) -> str:
            return f"Found {params.limit} results for: {params.query}"

KEY ATTRIBUTES:
    metadata        ToolMetadata with name, description, category
    params_schema   Pydantic model for parameter validation
    cache_enabled   Enable/disable caching (default: True)
    cache_ttl       Cache time-to-live in seconds (default: 300)

KEY METHODS:
    _run(params)           Sync execution (required)
    _async_run(params)     Async execution (optional)
    stream_run(params)     Streaming execution (optional)
    _run_result(params)    Result-based execution (recommended)

RELATED TOPICS:
    toolcase help result      Error handling with Result types
    toolcase help streaming   Progress streaming
    toolcase help cache       Caching configuration
"""
