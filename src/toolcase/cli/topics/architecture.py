ARCHITECTURE = """
TOPIC: architecture
==================

Toolcase is organized into layered modules with clear responsibilities.

MODULE STRUCTURE:
    toolcase/
    ├── foundation/        # Core abstractions (no dependencies)
    │   ├── core/          # BaseTool, ToolMetadata, decorator
    │   ├── errors/        # Result types, ErrorTrace, ToolError
    │   ├── registry/      # Tool registration and discovery
    │   ├── config/        # Settings and configuration
    │   ├── di/            # Dependency injection container
    │   ├── formats/       # Multi-framework format converters
    │   └── testing/       # Test utilities and fixtures
    │
    ├── io/                # I/O operations
    │   ├── cache/         # Result caching with TTL
    │   ├── progress/      # Progress events for long operations
    │   └── streaming/     # Result streaming for incremental output
    │
    ├── runtime/           # Execution layer
    │   ├── concurrency/   # UNIFIED ASYNC LAYER (all async ops)
    │   ├── middleware/    # Request/response interceptors
    │   ├── retry/         # Retry policies and backoff
    │   ├── pipeline/      # Tool composition (sequential, parallel)
    │   ├── agents/        # Agentic primitives (router, fallback, race)
    │   └── observability/ # Tracing and correlation
    │
    ├── tools/             # Built-in tools
    │   ├── core/          # Base tool utilities
    │   └── prebuilt/      # HttpTool, DiscoveryTool, etc.
    │
    └── ext/               # Extensions
        ├── mcp/           # Model Context Protocol server
        └── integrations/  # LangChain, external frameworks

CONCURRENCY LAYER (runtime/concurrency/):
    The unified concurrency layer handles ALL async operations:
    
    from toolcase import Concurrency
    
    # All tool execution uses these internally:
    # - BaseTool.arun() uses cancel_scope for timeouts
    # - BaseTool._async_run() uses to_thread for sync ops
    # - ParallelTool uses Concurrency.gather()
    # - RaceTool uses Concurrency.first_success()
    
    # Structure:
    concurrency/
    ├── facade.py         # Concurrency class (primary import)
    ├── primitives/       # TaskGroup, Lock, Semaphore, etc.
    ├── execution/        # race, gather, map_async, pools
    ├── streams/          # merge, buffer, throttle, batch
    └── interop/          # run_sync, to_thread, adapters

KEY DESIGN PRINCIPLES:
    1. Structured Concurrency: Tasks don't outlive their scope
    2. Railway-Oriented: Errors flow through Result types
    3. Zero External Deps: Pure asyncio (Python 3.11+)
    4. Type-Safe: Full generics support throughout

IMPORT HIERARCHY:
    # Most common (recommended)
    from toolcase import Concurrency, BaseTool, tool, Result
    
    # Direct imports when needed
    from toolcase.runtime.concurrency import TaskGroup, race, gather
    from toolcase.foundation.errors import Ok, Err, ErrorTrace

RELATED TOPICS:
    toolcase help overview     Framework introduction
    toolcase help imports      Import patterns
    toolcase help concurrency  Async primitives
    toolcase help pipeline     Tool composition
"""
