REGISTRY = """
TOPIC: registry
===============

Tool registration, discovery, and management.

GLOBAL REGISTRY:
    from toolcase import get_registry, set_registry, reset_registry
    
    registry = get_registry()  # Get global singleton
    reset_registry()           # Clear and reset

REGISTERING TOOLS:
    from toolcase import tool, get_registry, BaseTool
    
    @tool(description="My tool")
    def my_tool(x: str) -> str:
        return x
    
    registry = get_registry()
    registry.register(my_tool)
    
    # Or class-based
    registry.register(MyToolClass())

USING TOOLS:
    # By name
    result = registry["my_tool"](x="hello")
    
    # Get tool instance
    tool = registry.get("my_tool")

EXECUTING TOOLS:
    # Direct execution (string result)
    result = await registry.execute("search", {"query": "python"})
    
    # With Result type for error handling
    from toolcase import Ok, Err
    result = await registry.execute_result("search", {"query": "python"})
    match result:
        case Ok(value): print(value)
        case Err(error): print(f"Error: {error}")
    
    # Streaming execution
    async for chunk in registry.stream_execute("generate", {"topic": "AI"}):
        print(chunk, end="")

DISCOVERY:
    # List all tools
    tools = registry.list_tools()
    
    # Filter by category
    search_tools = registry.list_by_category("search")
    
    # Get unique categories
    categories = registry.categories()

INIT HELPER:
    from toolcase import init_tools
    
    # Registers DiscoveryTool plus your tools
    registry = init_tools(MyTool(), AnotherTool())

MIDDLEWARE:
    from toolcase import LoggingMiddleware, TimeoutMiddleware
    
    # Add middleware to all executions
    registry.use(LoggingMiddleware())
    registry.use(TimeoutMiddleware(30.0))
    
    # Enable validation with custom rules
    validation = registry.use_validation()
    validation.add_rule("search", "query", min_len(3), "query too short")

EVENT SIGNALS:
    # Hook into registration lifecycle
    registry.on_register += lambda tool: print(f"Registered: {tool.metadata.name}")
    registry.on_unregister += lambda name: print(f"Unregistered: {name}")
    registry.on_execute += lambda name, params, result: log_execution(name)

RELATED TOPICS:
    toolcase help tool       Creating tools
    toolcase help events     Event signals and lifecycle hooks
    toolcase help formats    Exporting to OpenAI/Anthropic/Google
    toolcase help middleware Middleware composition
    toolcase help quickstart Complete agent setup guide
"""
