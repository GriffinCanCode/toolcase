"""AI-readable help system for toolcase.

Plain text documentation accessible via command line.
No interactive elements - just type commands and read output.
"""

from __future__ import annotations

import sys

from .topics import TOPICS

TOPIC_LIST = sorted(TOPICS.keys())

CATEGORIES = {
    "Getting Started": ["help", "overview", "imports", "architecture"],
    "Core Concepts": ["tool", "result", "errors", "registry"],
    "Execution": ["middleware", "retry", "pipeline", "agents", "concurrency"],
    "Data Flow": ["cache", "streaming"],
    "Configuration": ["settings", "di"],
    "Observability": ["tracing", "testing"],
    "Integrations": ["formats", "http", "discovery"],
}


def print_topics() -> None:
    """Print all available topics."""
    print("TOOLCASE HELP")
    print("=============")
    print()
    print("Type 'toolcase help <topic>' for detailed information.")
    print()
    print("AVAILABLE TOPICS:")
    print()
    
    for category, topics in CATEGORIES.items():
        print(f"  {category}:")
        for topic in topics:
            if topic in TOPICS:
                lines = TOPICS[topic].strip().split("\n")
                desc = ""
                for line in lines[3:8]:
                    line = line.strip()
                    if line and not line.startswith("=") and not line.startswith("-"):
                        desc = line[:50] + "..." if len(line) > 50 else line
                        break
                print(f"    {topic:<16} {desc}")
        print()


def print_topic(topic: str) -> None:
    """Print a specific topic."""
    if topic in TOPICS:
        print(TOPICS[topic].strip())
    else:
        print(f"Unknown topic: {topic}")
        print()
        print("Available topics:")
        for t in TOPIC_LIST:
            print(f"  - {t}")
        print()
        print("Usage: toolcase help <topic>")


def main() -> None:
    """CLI entry point."""
    args = sys.argv[1:]
    
    if not args or (len(args) == 1 and args[0] in ("help", "--help", "-h")):
        print_topics()
        return
    
    if args[0] in ("help", "--help", "-h") and len(args) > 1:
        print_topic(args[1].lower())
        return
    
    if args[0].lower() in TOPICS:
        print_topic(args[0].lower())
        return
    
    print(f"Unknown command: {' '.join(args)}")
    print()
    print("Usage:")
    print("  toolcase help              List all topics")
    print("  toolcase help <topic>      Show topic details")
    print()
    print("Examples:")
    print("  toolcase help overview")
    print("  toolcase help result")
    print("  toolcase help middleware")


if __name__ == "__main__":
    main()
