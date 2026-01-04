"""Core tool abstractions: BaseTool, ToolMetadata, and parameter types.

This module provides the foundation for building type-safe, extensible tools
that AI agents can invoke. Tools are defined by subclassing BaseTool with
a typed parameter schema.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    ClassVar,
    Generic,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field

from .cache import DEFAULT_TTL, get_cache
from .errors import ErrorCode, ToolError
from .progress import ProgressCallback, ProgressKind, ToolProgress, complete

if TYPE_CHECKING:
    from collections.abc import Coroutine


class ToolMetadata(BaseModel):
    """Metadata describing a tool's capabilities and requirements.
    
    This information is used for:
    - Tool discovery by agents
    - LLM tool selection prompts
    - UI display and categorization
    - API key validation
    
    Attributes:
        name: Unique identifier (snake_case, e.g., "web_search")
        description: What the tool does (shown to LLM for selection)
        category: Grouping category (e.g., "search", "memory", "external")
        requires_api_key: Whether tool needs external API credentials
        enabled: Whether tool is currently active
        streaming: Whether tool supports progress streaming
    """
    
    model_config = ConfigDict(frozen=True)
    
    name: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(..., min_length=10)
    category: str = Field(default="general")
    requires_api_key: bool = Field(default=False)
    enabled: bool = Field(default=True)
    streaming: bool = Field(default=False)


class EmptyParams(BaseModel):
    """Default parameter schema for tools with no required inputs."""
    pass


# Type variable for tool parameter schemas
TParams = TypeVar("TParams", bound=BaseModel)


class BaseTool(ABC, Generic[TParams]):
    """Abstract base class for all tools.
    
    Subclasses must:
    - Define `metadata` class variable with ToolMetadata
    - Define `params_schema` class variable with the Pydantic model type
    - Implement `_run(params)` returning a string result
    
    Optional overrides:
    - `_async_run(params)` for native async implementation
    - `stream_run(params)` for progress streaming
    - `cache_enabled` / `cache_ttl` for caching behavior
    
    Example:
        >>> class SearchParams(BaseModel):
        ...     query: str = Field(..., description="Search query")
        ...     limit: int = Field(default=5, ge=1, le=20)
        ...
        >>> class SearchTool(BaseTool[SearchParams]):
        ...     metadata = ToolMetadata(
        ...         name="web_search",
        ...         description="Search the web for information",
        ...         category="search",
        ...     )
        ...     params_schema = SearchParams
        ...
        ...     def _run(self, params: SearchParams) -> str:
        ...         return f"Results for: {params.query}"
    """
    
    # Class variables - must be defined in subclasses
    metadata: ClassVar[ToolMetadata]
    params_schema: ClassVar[type[BaseModel]]
    
    # Caching configuration
    cache_enabled: ClassVar[bool] = True
    cache_ttl: ClassVar[float] = DEFAULT_TTL
    
    # ─────────────────────────────────────────────────────────────────
    # Error Handling
    # ─────────────────────────────────────────────────────────────────
    
    def _error(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        *,
        recoverable: bool = True,
        include_trace: bool = False,
    ) -> str:
        """Create a standardized error response string.
        
        Args:
            message: Human-readable error description
            code: Machine-readable error code
            recoverable: Whether caller should retry/fallback
            include_trace: Whether to include stack trace
        
        Returns:
            Formatted error string for LLM consumption
        """
        import traceback
        
        details = traceback.format_exc() if include_trace else None
        return ToolError.create(
            self.metadata.name,
            message,
            code,
            recoverable=recoverable,
            details=details,
        ).render()
    
    def _error_from_exception(
        self,
        exc: Exception,
        context: str = "",
        *,
        recoverable: bool = True,
    ) -> str:
        """Create error response from caught exception."""
        return ToolError.from_exception(
            self.metadata.name, exc, context, recoverable=recoverable
        ).render()
    
    # ─────────────────────────────────────────────────────────────────
    # Async/Sync Interop
    # ─────────────────────────────────────────────────────────────────
    
    def _run_async_sync(self, coro: Coroutine[None, None, str]) -> str:
        """Run async coroutine from sync context.
        
        Handles edge cases:
        - Running inside an existing event loop (e.g., FastAPI)
        - Running in a sync context with no loop
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - just run it
            return asyncio.run(coro)
        
        # Inside a running loop - use thread pool to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    
    # ─────────────────────────────────────────────────────────────────
    # Core Execution
    # ─────────────────────────────────────────────────────────────────
    
    @abstractmethod
    def _run(self, params: TParams) -> str:
        """Execute the tool synchronously.
        
        This is the primary method to implement. Return a string result
        formatted for LLM consumption.
        
        For async implementations, implement `_async_run` and call:
            return self._run_async_sync(self._async_run(params))
        """
        ...
    
    async def _async_run(self, params: TParams) -> str:
        """Execute the tool asynchronously.
        
        Default implementation wraps `_run` in a thread. Override for
        native async implementations (e.g., httpx calls).
        """
        return await asyncio.to_thread(self._run, params)
    
    def run(self, params: TParams) -> str:
        """Execute with caching support.
        
        Checks cache first, executes if miss, caches successful results.
        Error responses (starting with "**Tool Error") are not cached.
        """
        if not self.cache_enabled:
            return self._run(params)
        
        cache = get_cache()
        tool_name = self.metadata.name
        
        # Cache hit?
        if (cached := cache.get(tool_name, params)) is not None:
            return cached
        
        # Execute
        result = self._run(params)
        
        # Cache successful results only
        if not result.startswith("**Tool Error"):
            cache.set(tool_name, params, result, self.cache_ttl)
        
        return result
    
    async def arun(self, params: TParams, timeout: float = 30.0) -> str:
        """Execute asynchronously with caching and timeout."""
        if not self.cache_enabled:
            return await asyncio.wait_for(self._async_run(params), timeout=timeout)
        
        cache = get_cache()
        tool_name = self.metadata.name
        
        if (cached := cache.get(tool_name, params)) is not None:
            return cached
        
        result = await asyncio.wait_for(self._async_run(params), timeout=timeout)
        
        if not result.startswith("**Tool Error"):
            cache.set(tool_name, params, result, self.cache_ttl)
        
        return result
    
    # ─────────────────────────────────────────────────────────────────
    # Streaming Progress
    # ─────────────────────────────────────────────────────────────────
    
    @property
    def supports_streaming(self) -> bool:
        """Whether this tool supports progress streaming."""
        return self.metadata.streaming
    
    async def stream_run(self, params: TParams) -> AsyncIterator[ToolProgress]:
        """Stream progress events during execution.
        
        Override in tools that support streaming for real-time updates.
        Default implementation just runs and yields completion.
        
        Yields:
            ToolProgress events for status, steps, and completion
        """
        yield ToolProgress(kind=ProgressKind.STATUS, message="Starting...")
        try:
            result = await self._async_run(params)
            yield complete(result)
        except Exception as e:
            yield ToolProgress(
                kind=ProgressKind.ERROR,
                message=str(e),
                data={"error": str(e), "type": type(e).__name__},
            )
    
    async def arun_with_progress(
        self,
        params: TParams,
        on_progress: ProgressCallback | None = None,
        timeout: float = 60.0,
    ) -> str:
        """Execute with progress callbacks.
        
        For streaming tools, collects all events and calls callback for each.
        Returns the final result.
        """
        result = ""
        
        async def execute() -> str:
            nonlocal result
            async for progress in self.stream_run(params):
                if on_progress:
                    on_progress(progress)
                if progress.kind == ProgressKind.COMPLETE and progress.data:
                    result = str(progress.data.get("result", ""))
                elif progress.kind == ProgressKind.ERROR:
                    raise RuntimeError(progress.message)
            return result
        
        return await asyncio.wait_for(execute(), timeout=timeout)
    
    # ─────────────────────────────────────────────────────────────────
    # Invocation (kwargs interface)
    # ─────────────────────────────────────────────────────────────────
    
    def __call__(self, **kwargs: object) -> str:
        """Invoke tool with keyword arguments.
        
        Convenience method that constructs params from kwargs.
        """
        params = self.params_schema(**kwargs)  # type: ignore[call-arg]
        return self.run(params)  # type: ignore[arg-type]
    
    async def acall(self, **kwargs: object) -> str:
        """Async invoke with keyword arguments."""
        params = self.params_schema(**kwargs)  # type: ignore[call-arg]
        return await self.arun(params)  # type: ignore[arg-type]
