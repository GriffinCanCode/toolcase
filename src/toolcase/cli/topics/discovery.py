DISCOVERY = """
TOPIC: discovery
================

Tool discovery and search for AI agents.

DISCOVERY TOOL:
    from toolcase import DiscoveryTool, init_tools
    
    # init_tools automatically registers DiscoveryTool
    registry = init_tools(MyTool(), AnotherTool())
    
    # List all tools
    result = await registry.execute("discover_tools", {})

DISCOVERY PARAMS:
    category         Filter by category (optional)
    format           "brief" or "detailed"
    capabilities     List of tags to match (e.g., ["search", "web"])
    capability_match "any" (OR) or "all" (AND)
    param_name       Find tools with this parameter name
    param_type       Find tools accepting this type (string, integer, etc.)
    include_stats    Include usage statistics in output
    stats_only       Return only usage statistics

CAPABILITY-BASED SEARCH:
    # Find tools with "search" capability
    result = await registry.execute("discover_tools", {
        "capabilities": ["search", "web"],
        "capability_match": "any"  # OR logic
    })

SCHEMA-BASED MATCHING:
    # Find tools accepting a "query" string parameter
    result = await registry.execute("discover_tools", {
        "param_name": "query",
        "param_type": "string"
    })

USAGE STATISTICS:
    result = await registry.execute("discover_tools", {
        "include_stats": True
    })
    
    # Stats only
    result = await registry.execute("discover_tools", {
        "stats_only": True
    })

PROGRAMMATIC QUERY API:
    from toolcase import ToolQuery, SchemaPattern, find_by_param
    
    # Simple: find by parameter
    results = find_by_param(registry, "query", "string")
    
    # Advanced: fluent query builder
    results = (ToolQuery(registry)
        .by_capability(["search"])
        .by_schema(SchemaPattern(type="string", required=True))
        .by_category("search")
        .execute())
    
    for r in results:
        print(f"{r.metadata.name}: score={r.score:.0%}")

STATISTICS MIDDLEWARE:
    from toolcase import StatsMiddleware, get_stats
    
    registry.use(StatsMiddleware())
    await registry.execute("search", {"query": "python"})
    
    stats = get_stats()
    print(stats.get("search").success_rate)  # 1.0
    print(stats.get("search").avg_duration_ms)

RELATED TOPICS:
    toolcase help registry   Tool registration
    toolcase help formats    Format converters
    toolcase help middleware Middleware system
"""
