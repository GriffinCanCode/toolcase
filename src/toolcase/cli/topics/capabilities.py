CAPABILITIES = """
TOPIC: capabilities
===================

Tool capabilities for intelligent scheduling and execution.

OVERVIEW:
    Tools advertise capabilities that the registry/scheduler uses for:
    - Caching decisions (skip cache for non-cacheable tools)
    - Concurrency limits (respect max_concurrent for rate-limited APIs)
    - Streaming support (route to streaming pipeline when supported)
    - Idempotency hints (safe to retry without side effects)
    - Confirmation requirements (human-in-the-loop)

TOOL CAPABILITIES:
    from toolcase import ToolCapabilities, ToolMetadata
    
    class MyTool(BaseTool[MyParams]):
        metadata = ToolMetadata(
            name="my_tool",
            description="Does something useful",
            capabilities=ToolCapabilities(
                supports_caching=True,      # Results can be cached
                supports_streaming=True,    # Can stream incremental results
                max_concurrent=5,           # Max parallel executions
                idempotent=True,            # Safe to retry
                estimated_latency_ms=500,   # Typical execution time
                requires_confirmation=False, # Needs user approval
            ),
        )

CAPABILITY PRESETS:
    from toolcase import ToolCapabilities
    
    # Default (standard tools)
    caps = ToolCapabilities.default()
    
    # Streaming-capable
    caps = ToolCapabilities.streaming(max_concurrent=10)
    
    # Non-cacheable (time-sensitive)
    caps = ToolCapabilities.non_cacheable(idempotent=False)
    
    # Rate-limited external APIs
    caps = ToolCapabilities.rate_limited(max_concurrent=5, estimated_latency_ms=200)

DECORATOR SHORTHAND:
    from toolcase import tool
    
    @tool(
        description="Search with rate limiting",
        max_concurrent=5,       # Sets capabilities.max_concurrent
        streaming=True,         # Sets capabilities.supports_streaming
    )
    async def search(query: str) -> str:
        ...

REGISTRY CAPABILITY FILTERING:
    from toolcase import get_registry
    
    registry = get_registry()
    
    # Find tools by capability
    streamable = registry.tools_by_capability(supports_streaming=True)
    cacheable = registry.tools_by_capability(supports_caching=True)
    rate_limited = registry.tools_by_capability(max_concurrent_lte=10)
    idempotent = registry.tools_by_capability(idempotent=True)

AUTOMATIC RATE LIMITING:
    # Registry automatically creates CapacityLimiter for tools with max_concurrent
    # This ensures the tool never exceeds its declared limit
    
    @tool(description="External API call", max_concurrent=3)
    async def call_api(endpoint: str) -> str:
        # Registry ensures max 3 concurrent executions
        ...
    
    # Check limiter stats
    stats = registry.limiter_stats()
    print(stats)  # {"call_api": {"borrowed": 1, "total": 3}}

METADATA ACCESSORS:
    # ToolMetadata provides convenience accessors
    tool.metadata.supports_caching       # -> bool
    tool.metadata.supports_result_streaming  # -> bool
    tool.metadata.max_concurrent         # -> int | None
    tool.metadata.is_idempotent          # -> bool
    tool.metadata.requires_confirmation  # -> bool

RELATED TOPICS:
    toolcase help tool       Tool creation
    toolcase help registry   Tool registration
    toolcase help discovery  Finding tools by capability
"""
