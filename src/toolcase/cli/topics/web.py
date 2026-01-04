WEB = """
TOPIC: web
==========

Prebuilt web tools for search, fetch, parse, and extraction.

OVERVIEW:
    Toolcase includes ready-to-use web tools for common agent tasks:
    - WebSearchTool: Search the web via Tavily, Perplexity, or DuckDuckGo
    - UrlFetchTool: Fetch and extract content from URLs
    - HtmlParseTool: Parse HTML and extract structured content
    - RegexExtractTool: Extract patterns from text using regex
    - JsonExtractTool: Extract and query JSON from text

WEB SEARCH TOOL:
    from toolcase.tools import WebSearchTool, WebSearchConfig
    
    # Free search (DuckDuckGo - no API key needed)
    search = WebSearchTool()
    result = await search.acall(query="python async programming", max_results=5)
    
    # Tavily search (requires TAVILY_API_KEY)
    from toolcase.tools import tavily_search
    search = tavily_search()
    
    # Perplexity search (requires PERPLEXITY_API_KEY)
    from toolcase.tools import perplexity_search
    search = perplexity_search()
    
    # Or configure explicitly
    config = WebSearchConfig(
        backend="tavily",              # "duckduckgo" | "tavily" | "perplexity"
        api_key_env="MY_API_KEY",      # Env var for API key
        default_max_results=10,
        timeout=30.0,
    )
    search = WebSearchTool(config)

SEARCH BACKENDS:
    from toolcase.tools import TavilyBackend, PerplexityBackend, DuckDuckGoBackend
    
    # Custom backend configuration
    backend = TavilyBackend(
        api_key="...",
        search_depth="advanced",      # "basic" or "advanced"
        include_answer=True,          # Include AI-generated answer
    )
    
    backend = PerplexityBackend(
        api_key="...",
        model="llama-3.1-sonar-small-128k-online",
    )
    
    backend = DuckDuckGoBackend()     # No API key needed

SEARCH RESPONSE:
    from toolcase.tools import SearchResult, SearchResponse
    
    response = await search.search(query="python")
    
    for result in response.results:
        print(f"Title: {result.title}")
        print(f"URL: {result.url}")
        print(f"Snippet: {result.snippet}")
        print(f"Score: {result.score}")  # Relevance score (if available)
    
    # Tavily/Perplexity may include AI answer
    if response.answer:
        print(f"AI Answer: {response.answer}")

URL FETCH TOOL:
    from toolcase.tools import UrlFetchTool, UrlFetchConfig
    
    fetch = UrlFetchTool()
    content = await fetch.acall(url="https://example.com/article")
    
    # Configure extraction
    config = UrlFetchConfig(
        timeout=30.0,
        max_content_length=100000,    # Limit content size
        extract_mode="auto",          # "auto" | "text" | "html" | "markdown"
        follow_redirects=True,
        user_agent="Mozilla/5.0...",
    )
    fetch = UrlFetchTool(config)

HTML PARSE TOOL:
    from toolcase.tools import HtmlParseTool, HtmlParseConfig
    
    parser = HtmlParseTool()
    
    # Extract text content
    text = await parser.acall(
        html="<div>Content</div>",
        selector="div",
        output="text"
    )
    
    # Extract structured data
    config = HtmlParseConfig(
        default_selector="article",
        strip_scripts=True,
        strip_styles=True,
    )
    parser = HtmlParseTool(config)
    
    # CSS selector extraction
    data = await parser.acall(
        html=html_content,
        selector="div.product",
        output="json"  # Extract as structured JSON
    )

REGEX EXTRACT TOOL:
    from toolcase.tools import RegexExtractTool, RegexExtractConfig
    
    extractor = RegexExtractTool()
    
    # Extract patterns
    matches = await extractor.acall(
        text="Contact: john@example.com, jane@test.org",
        pattern=r"[\\w.-]+@[\\w.-]+\\.\\w+",
        group=0,           # Capture group (0 = full match)
        find_all=True,     # Find all matches
    )
    # ["john@example.com", "jane@test.org"]
    
    # Named groups
    matches = await extractor.acall(
        text="Price: $19.99",
        pattern=r"\\$(?P<dollars>\\d+)\\.(?P<cents>\\d{2})",
        output="dict"      # Return as dict with group names
    )
    # {"dollars": "19", "cents": "99"}

JSON EXTRACT TOOL:
    from toolcase.tools import JsonExtractTool
    
    extractor = JsonExtractTool()
    
    # Extract JSON from text
    data = await extractor.acall(
        text='Some text {"key": "value"} more text',
        query="$.key",     # JSONPath query (optional)
    )
    
    # Extract nested values
    data = await extractor.acall(
        text=api_response,
        query="$.data.items[*].name"  # Get all item names
    )

COMBINING WEB TOOLS:
    from toolcase import pipeline
    from toolcase.tools import WebSearchTool, UrlFetchTool, HtmlParseTool
    
    # Search → Fetch → Parse pipeline
    research = pipeline(
        WebSearchTool(),
        UrlFetchTool(),
        HtmlParseTool(),
    )
    
    # Or use individually for more control
    search = WebSearchTool()
    fetch = UrlFetchTool()
    parse = HtmlParseTool()
    
    results = await search.acall(query="python tutorials")
    for result in results:
        content = await fetch.acall(url=result.url)
        text = await parse.acall(html=content, selector="article")

QUICK START (Free Search):
    from toolcase import init_tools
    from toolcase.tools import free_search
    
    # No API key needed - uses DuckDuckGo
    registry = init_tools(free_search())
    
    result = await registry.execute("web_search", {
        "query": "python async await tutorial",
        "max_results": 5
    })

ENVIRONMENT VARIABLES:
    TAVILY_API_KEY        API key for Tavily search
    PERPLEXITY_API_KEY    API key for Perplexity search

RELATED TOPICS:
    toolcase help http       HTTP tool for custom requests
    toolcase help pipeline   Combining tools
    toolcase help registry   Tool registration
"""
