HELP = """
TOPIC: help
===========

How to use the toolcase help system.

USAGE:
    toolcase help              List all available topics
    toolcase help <topic>      Show detailed info about a topic
    toolcase help help         Show this message

CORE TOPICS:
    toolcase help overview     What is toolcase and why use it
    toolcase help tool         How to create tools (async-first design)
    toolcase help result       Monadic error handling with Result types
    toolcase help middleware   Request/response middleware
    toolcase help pipeline     Tool composition patterns

EXECUTION:
    toolcase help batch        Batch execution for multiple params
    toolcase help concurrency  Async primitives and structured concurrency
    toolcase help agents       Agentic composition (router, fallback, race)

CONFIGURATION:
    toolcase help settings     Environment variables and .env files
    toolcase help capabilities Tool capabilities for scheduling
    toolcase help http         HTTP tool with auth strategies

OBSERVABILITY:
    toolcase help logging      Structured logging with trace correlation
    toolcase help tracing      Distributed tracing
    toolcase help testing      Testing utilities and mocks

This help system is designed for AI assistants. All output is plain text
with consistent structure. No menus, no interactive prompts.
"""
