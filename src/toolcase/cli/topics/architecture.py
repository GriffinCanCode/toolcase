ARCHITECTURE = """
TOPIC: architecture
===================

Toolcase module structure and design.

DIRECTORY LAYOUT:
    toolcase/
    ├── foundation/     Core building blocks
    │   ├── core/       BaseTool, @tool decorator
    │   ├── errors/     Result monad, ErrorCode
    │   ├── di/         Dependency injection
    │   ├── registry/   Tool registration
    │   ├── testing/    Test utilities
    │   ├── formats/    OpenAI/Anthropic/Google converters
    │   └── config/     Settings management
    │
    ├── io/             Data input/output
    │   ├── cache/      Result caching
    │   ├── progress/   Progress streaming
    │   └── streaming/  Result streaming
    │
    ├── runtime/        Execution and control
    │   ├── agents/     router, fallback, race, gate
    │   ├── middleware/ Request/response middleware
    │   ├── pipeline/   Tool composition
    │   ├── retry/      Retry policies
    │   ├── concurrency/ Async primitives
    │   └── observability/ Tracing
    │
    ├── ext/            External integrations
    │   ├── integrations/ LangChain adapters
    │   └── mcp/        MCP protocol server
    │
    └── tools/          Built-in tools
        ├── core/       ConfigurableTool, DiscoveryTool
        └── prebuilt/   HttpTool

DESIGN PRINCIPLES:
    - Type-safe: Full typing with generics
    - Composable: Small pieces that combine
    - Extensible: Override and customize
    - Zero magic: Explicit is better than implicit
    - Framework agnostic: Works with any AI framework

DEPENDENCIES:
    Core: pydantic, pydantic-settings
    Optional: langchain-core, fastmcp, redis, opentelemetry

RELATED TOPICS:
    toolcase help imports    Import patterns
    toolcase help overview   What is toolcase
"""
