"""Toolcase CLI - AI-readable help system.

Usage:
    toolcase help              # List all available topics
    toolcase help <topic>      # Get detailed info on a topic
    toolcase help help         # How to use this help system

Example:
    toolcase help result       # Learn about monadic Result types
    toolcase help middleware   # Learn about middleware composition
"""

from .help import main

__all__ = ["main"]
