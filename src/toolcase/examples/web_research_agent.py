#!/usr/bin/env python3
"""Web Research Agent - Full internal tooling demonstration.

Uses ALL internal toolcase utilities:
- load_env/get_env for environment detection
- to_anthropic for tool formatting
- WebSearchTool with Tavily backend
- UrlFetchTool, HtmlParseTool, RegexExtractTool
- ToolRegistry for tool management
- Result types for error handling

Usage:
    # Run agent test with Claude
    python -m toolcase.examples.web_research_agent --test "python async patterns"
    
    # As MCP server
    python -m toolcase.examples.web_research_agent --mcp
    
    # As HTTP server
    python -m toolcase.examples.web_research_agent --http

Requirements:
    ANTHROPIC_API_KEY - for Claude
    TAVILY_API_KEY - for search (optional, falls back to DuckDuckGo)
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import TypedDict

# Internal toolcase imports
from toolcase import (
    ToolRegistry,
    get_registry,
    reset_registry,
)
from toolcase.foundation.config import get_env, load_env, require_env
from toolcase.ext.integrations.frontiers import to_anthropic
from toolcase.tools.prebuilt.web import (
    HtmlParseTool,
    JsonExtractTool,
    RegexExtractTool,
    UrlFetchTool,
    WebSearchTool,
    WebSearchConfig,
    tavily_search,
    free_search,
)


class ToolCall(TypedDict):
    id: str
    name: str
    input: dict


def create_web_registry() -> ToolRegistry:
    """Create registry with all web tools."""
    reset_registry()  # Clean slate
    registry = get_registry()
    
    # Detect best search backend using internal env detection
    tavily_key = get_env("TAVILY_API_KEY")
    
    if tavily_key:
        print(f"  ✓ TAVILY_API_KEY detected ({tavily_key[:8]}...)")
        search = tavily_search(api_key=tavily_key)
    else:
        print("  ⚠ No TAVILY_API_KEY, using free DuckDuckGo backend")
        search = free_search()
    
    # Register tools
    registry.register(search)
    registry.register(UrlFetchTool())
    registry.register(HtmlParseTool())
    registry.register(RegexExtractTool())
    registry.register(JsonExtractTool())
    
    return registry


async def execute_tool_call(registry: ToolRegistry, tool_call: ToolCall) -> str:
    """Execute a tool call using registry."""
    tool_name = tool_call["name"]
    tool_input = tool_call["input"]
    
    print(f"    🔧 Executing: {tool_name}({json.dumps(tool_input)[:100]}...)")
    
    # Use registry execution (includes middleware if configured)
    result = await registry.execute(tool_name, tool_input)
    
    # Truncate for display
    preview = result[:500] + "..." if len(result) > 500 else result
    print(f"    ✓ Result: {preview[:200]}")
    
    return result


async def run_agent_loop(
    query: str,
    registry: ToolRegistry,
    max_iterations: int = 10,
) -> str:
    """Run agent loop with Claude + tools.
    
    Uses internal to_anthropic formatter for tool definitions.
    """
    try:
        import anthropic
    except ImportError:
        return "❌ anthropic package required: pip install anthropic"
    
    # Get API key using internal env detection
    api_key = get_env("ANTHROPIC_API_KEY")
    if not api_key:
        return "❌ ANTHROPIC_API_KEY not set"
    
    print(f"  ✓ ANTHROPIC_API_KEY detected ({api_key[:12]}...)")
    
    # Convert tools using internal formatter
    tools = to_anthropic(registry)
    print(f"  ✓ Formatted {len(tools)} tools for Anthropic")
    for t in tools:
        print(f"    - {t['name']}: {t['description'][:60]}...")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    messages = [{"role": "user", "content": query}]
    
    system = """You are a web research assistant. Use the available tools to:
1. Search for information using web_search
2. Fetch page content with url_fetch if you need more details
3. Parse HTML content with html_parse to extract structured data
4. Use regex_extract to find patterns (emails, urls, etc.)

Be thorough but efficient. Summarize your findings clearly."""
    
    print(f"\n{'─'*60}")
    print("Agent Loop Starting")
    print(f"{'─'*60}")
    
    for i in range(max_iterations):
        print(f"\n📍 Iteration {i + 1}/{max_iterations}")
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system,
            tools=tools,
            messages=messages,
        )
        
        print(f"  Stop reason: {response.stop_reason}")
        
        # Process response content
        assistant_content = []
        tool_results = []
        
        for block in response.content:
            if block.type == "text":
                print(f"  💬 Claude: {block.text[:200]}...")
                assistant_content.append(block)
            
            elif block.type == "tool_use":
                assistant_content.append(block)
                
                # Execute tool
                result = await execute_tool_call(registry, {
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        
        # Add assistant message
        messages.append({"role": "assistant", "content": assistant_content})
        
        # If there were tool calls, add results and continue
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # If Claude is done (no more tool calls)
        if response.stop_reason == "end_turn":
            # Extract final text response
            for block in response.content:
                if block.type == "text":
                    return block.text
            return "Agent completed but no text response"
    
    return "Max iterations reached"


async def run_test(query: str) -> None:
    """Run full agent test."""
    print(f"\n{'='*60}")
    print("Web Research Agent - Internal Tooling Test")
    print(f"{'='*60}")
    
    # Load env files using internal utility
    print("\n📁 Loading environment...")
    
    # Try loading from project root and src/toolcase
    for path in [Path.cwd(), Path(__file__).parent.parent]:
        loaded = load_env(base_path=path)
        if loaded:
            print(f"  ✓ Loaded from {path}: {list(loaded.keys())}")
    
    # Create registry
    print("\n🔧 Creating tool registry...")
    registry = create_web_registry()
    
    # Run agent
    print(f"\n🤖 Query: {query}")
    result = await run_agent_loop(
        f"Research this topic and give me a comprehensive summary: {query}",
        registry,
    )
    
    print(f"\n{'='*60}")
    print("Final Result")
    print(f"{'='*60}")
    print(result)


def run_mcp_server(port: int = 8080) -> None:
    """Start MCP server."""
    from toolcase.ext.mcp import serve_mcp
    
    # Load env
    load_env(base_path=Path(__file__).parent.parent)
    
    registry = create_web_registry()
    print(f"\n🚀 Starting MCP server on port {port}")
    serve_mcp(registry, name="web-research", transport="sse", port=port)


def run_http_server(port: int = 8000) -> None:
    """Start HTTP server."""
    from toolcase.ext.mcp import serve_http
    
    # Load env
    load_env(base_path=Path(__file__).parent.parent)
    
    registry = create_web_registry()
    print(f"\n🌐 Starting HTTP server on port {port}")
    print(f"  List tools:  curl http://localhost:{port}/tools")
    serve_http(registry, name="web-research", port=port)


def main():
    parser = argparse.ArgumentParser(description="Web Research Agent")
    parser.add_argument("--mcp", action="store_true", help="Run as MCP server")
    parser.add_argument("--http", action="store_true", help="Run as HTTP server")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--test", type=str, help="Run agent test with query")
    
    args = parser.parse_args()
    
    if args.test:
        asyncio.run(run_test(args.test))
    elif args.mcp:
        run_mcp_server(args.port)
    elif args.http:
        run_http_server(args.port if args.port != 8080 else 8000)
    else:
        # Default: run a demo test
        asyncio.run(run_test("What are the best practices for Python async programming?"))


if __name__ == "__main__":
    main()
