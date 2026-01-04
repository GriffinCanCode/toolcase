HELP = """
TOPIC: help
===========

How to use the toolcase help system.

USAGE:
    toolcase help              List all available topics
    toolcase help <topic>      Show detailed info about a topic
    toolcase help help         Show this message

GETTING STARTED:
    toolcase help quickstart   Complete guide to building an AI agent
    toolcase help overview     What is toolcase and why use it

CORE TOPICS:
    toolcase help tool         How to create tools (async-first design)
    toolcase help result       Monadic error handling with Result types
    toolcase help middleware   Request/response middleware
    toolcase help pipeline     Tool composition patterns
    toolcase help validation   Composable validation DSL

EXECUTION:
    toolcase help batch        Batch execution for multiple params
    toolcase help concurrency  Async primitives and structured concurrency
    toolcase help agents       Agentic composition (router, fallback, race)

CONFIGURATION:
    toolcase help settings     Environment variables and .env files
    toolcase help capabilities Tool capabilities for scheduling
    toolcase help http         HTTP tool with auth strategies
    toolcase help effects      Effect system for side-effect tracking

BUILT-IN TOOLS:
    toolcase help web          Web search, URL fetch, HTML parse, extraction

INTEGRATION:
    toolcase help formats      Multi-framework format converters
    toolcase help mcp          Model Context Protocol server
    toolcase help discovery    Tool discovery and search

OBSERVABILITY:
    toolcase help logging      Structured logging with trace correlation
    toolcase help tracing      Distributed tracing
    toolcase help testing      Testing utilities and mocks

This help system is designed for AI assistants. All output is plain text
with consistent structure. No menus, no interactive prompts.
"""
