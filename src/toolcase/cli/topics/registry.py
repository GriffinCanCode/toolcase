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

RELATED TOPICS:
    toolcase help tool       Creating tools
    toolcase help formats    Exporting to OpenAI/Anthropic/Google
"""
