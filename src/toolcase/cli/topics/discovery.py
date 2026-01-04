DISCOVERY = """
TOPIC: discovery
================

Tool discovery for AI agents.

DISCOVERY TOOL:
    from toolcase import DiscoveryTool, init_tools
    
    # init_tools automatically registers DiscoveryTool
    registry = init_tools(MyTool(), AnotherTool())
    
    # List all tools
    result = registry["discover"](category=None, format="brief")

DISCOVERY PARAMS:
    category    Filter by category (optional)
    format      "brief" or "detailed"

OUTPUT (BRIEF):
    Available tools:
    - search: Search the web for information
    - summarize: Summarize text content
    - translate: Translate between languages

OUTPUT (DETAILED):
    Tool: search
    Description: Search the web for information
    Category: search
    Parameters:
      - query (str, required): Search query string
      - limit (int, optional): Max results (default: 5)

MANUAL DISCOVERY:
    registry = get_registry()
    
    # List tools
    tools = registry.list_tools()
    tools = registry.list_tools(category="search")
    
    # Get schemas
    schemas = registry.get_schemas()

RELATED TOPICS:
    toolcase help registry   Tool registration
    toolcase help formats    Format converters
"""
