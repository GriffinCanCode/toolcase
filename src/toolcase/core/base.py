"""Core tool abstractions: BaseTool, ToolMetadata, and parameter types.

This module provides the foundation for building type-safe, extensible tools
that AI agents can invoke. Tools are defined by subclassing BaseTool with
a typed parameter schema.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Callable,
    ClassVar,
    Generic,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field

from ..cache import DEFAULT_TTL, get_cache
from ..errors import (
    ErrorCode,
    ErrorTrace,
    Result,
    ToolError,
    ToolResult,
    classify_exception,
    result_to_string,
    string_to_result,
)
from ..progress import ProgressCallback, ProgressKind, ToolProgress, complete
from ..retry import RetryPolicy, execute_with_retry, execute_with_retry_sync
from ..streaming import (
    StreamChunk,
    StreamEvent,
    StreamEventKind,
    StreamResult,
    stream_complete,
    stream_error,
    stream_start,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine

# Internal constants for fast Result construction
from ..errors.result import _ERR, _OK


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
    
    # Retry configuration (None = no retries)
    retry_policy: ClassVar[RetryPolicy | None] = None
    
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
        return ToolError(
            tool_name=self.metadata.name,
            message=message,
            code=code,
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
        return ToolError.from_exception(self.metadata.name, exc, context, recoverable=recoverable).render()
    
    # ─────────────────────────────────────────────────────────────────
    # Result Helpers
    # ─────────────────────────────────────────────────────────────────
    
    def _ok(self, value: str) -> ToolResult:
        """Create Ok result (convenience method)."""
        return Result(value, _OK)
    
    def _err(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        *,
        recoverable: bool = True,
    ) -> ToolResult:
        """Create Err result from error parameters."""
        from ..errors.types import _EMPTY_CONTEXTS, ErrorContext, _EMPTY_META
        ctx = (ErrorContext(f"tool:{self.metadata.name}", "", _EMPTY_META),)
        return Result(ErrorTrace(message, ctx, code.value, recoverable, None), _ERR)
    
    def _try(
        self,
        operation: Callable[[], str],
        *,
        context: str = "",
    ) -> ToolResult:
        """Execute operation with automatic exception handling."""
        try:
            return Result(operation(), _OK)
        except Exception as e:
            return self._err_from_exc(e, context)
    
    async def _try_async(
        self,
        operation: Callable[[], str],
        *,
        context: str = "",
    ) -> ToolResult:
        """Execute async operation with automatic exception handling."""
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation()  # type: ignore[misc]
            else:
                result = await asyncio.to_thread(operation)
            return Result(result, _OK)
        except Exception as e:
            return self._err_from_exc(e, context)
    
    def _err_from_exc(self, exc: Exception, context: str = "") -> ToolResult:
        """Create Err result from exception (internal helper)."""
        import traceback
        from ..errors.types import _EMPTY_META, ErrorContext
        msg = f"{context}: {exc}" if context else str(exc)
        tool_ctx = ErrorContext(f"tool:{self.metadata.name}", "", _EMPTY_META)
        contexts = (tool_ctx, ErrorContext(context, "", _EMPTY_META)) if context else (tool_ctx,)
        return Result(ErrorTrace(msg, contexts, classify_exception(exc).value, True, traceback.format_exc()), _ERR)
    
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
    
    def _run_result(self, params: TParams) -> ToolResult:
        """Execute tool with Result-based error handling (optional override).
        
        Override this method to use monadic error handling with Result types.
        This provides type-safe error propagation with railway-oriented programming.
        
        If not overridden, falls back to _run() with string-based error handling.
        Catches exceptions and converts them to Err results.
        
        Example:
            >>> def _run_result(self, params: MyParams) -> ToolResult:
            ...     return (
            ...         self._validate_params(params)
            ...         .flat_map(lambda p: self._fetch_data(p))
            ...         .map(lambda data: self._format_result(data))
            ...     )
        
        Returns:
            ToolResult (Result[str, ErrorTrace]) with success or error
        """
        try:
            return Result(self._run(params), _OK)
        except Exception as e:
            return self._err_from_exc(e, "execution")
    
    @abstractmethod
    def _run(self, params: TParams) -> str:
        """Execute the tool synchronously.
        
        This is the primary method to implement. Return a string result
        formatted for LLM consumption.
        
        For Result-based error handling, override _run_result() instead.
        
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
    
    async def _async_run_result(self, params: TParams) -> ToolResult:
        """Execute tool asynchronously with Result-based error handling.
        
        Override for native async implementations using Result types.
        Catches exceptions and converts them to Err results.
        """
        try:
            result = await self._async_run(params)
            return string_to_result(result, self.metadata.name)
        except Exception as e:
            return self._err_from_exc(e, "async execution")
    
    def run(self, params: TParams) -> str:
        """Execute with caching support.
        
        Checks cache first, executes if miss, caches successful results.
        Error responses (starting with "**Tool Error") are not cached.
        """
        if not self.cache_enabled:
            result = self._run_result(params)
            return result_to_string(result, self.metadata.name)
        
        cache = get_cache()
        tool_name = self.metadata.name
        
        # Cache hit?
        if (cached := cache.get(tool_name, params)) is not None:
            return cached
        
        # Execute with Result-based handling
        result = self._run_result(params)
        output = result_to_string(result, tool_name)
        
        # Cache successful results only (Ok variants)
        if result.is_ok():
            cache.set(tool_name, params, output, self.cache_ttl)
        
        return output
    
    def run_result(self, params: TParams) -> ToolResult:
        """Execute with caching and retry, returning Result type.
        
        Type-safe alternative to run() that returns Result[str, ErrorTrace]
        instead of string. Enables monadic error handling in tool compositions.
        
        Respects retry_policy class variable for automatic retries on
        retryable error codes (RATE_LIMITED, TIMEOUT, NETWORK_ERROR by default).
        
        Example:
            >>> result = tool.run_result(params)
            >>> processed = result.map(lambda s: s.upper())
            >>> output = processed.unwrap_or("default")
        
        Returns:
            ToolResult with success string or ErrorTrace
        """
        cache = get_cache() if self.cache_enabled else None
        tool_name = self.metadata.name
        
        # Cache hit?
        if cache and (cached := cache.get(tool_name, params)) is not None:
            return string_to_result(cached, tool_name)
        
        # Execute with optional retry
        if self.retry_policy:
            result = execute_with_retry_sync(
                lambda: self._run_result(params),
                self.retry_policy,
                tool_name,
            )
        else:
            result = self._run_result(params)
        
        # Cache successful results only
        if cache and result.is_ok():
            output = result_to_string(result, tool_name)
            cache.set(tool_name, params, output, self.cache_ttl)
        
        return result
    
    async def arun(self, params: TParams, timeout: float = 30.0) -> str:
        """Execute asynchronously with caching and timeout."""
        if not self.cache_enabled:
            result = await asyncio.wait_for(self._async_run_result(params), timeout=timeout)
            return result_to_string(result, self.metadata.name)
        
        cache = get_cache()
        tool_name = self.metadata.name
        
        if (cached := cache.get(tool_name, params)) is not None:
            return cached
        
        result = await asyncio.wait_for(self._async_run_result(params), timeout=timeout)
        output = result_to_string(result, tool_name)
        
        if result.is_ok():
            cache.set(tool_name, params, output, self.cache_ttl)
        
        return output
    
    async def arun_result(self, params: TParams, timeout: float = 30.0) -> ToolResult:
        """Execute asynchronously with Result type, caching, retry, and timeout."""
        cache = get_cache() if self.cache_enabled else None
        tool_name = self.metadata.name
        
        # Cache hit?
        if cache and (cached := cache.get(tool_name, params)) is not None:
            return string_to_result(cached, tool_name)
        
        # Execute with optional retry (timeout wraps entire retry sequence)
        async def execute() -> ToolResult:
            if self.retry_policy:
                return await execute_with_retry(
                    lambda: self._async_run_result(params),
                    self.retry_policy,
                    tool_name,
                )
            return await self._async_run_result(params)
        
        result = await asyncio.wait_for(execute(), timeout=timeout)
        
        # Cache successful results only
        if cache and result.is_ok():
            output = result_to_string(result, tool_name)
            cache.set(tool_name, params, output, self.cache_ttl)
        
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
        
        result = await self._async_run_result(params)
        
        if result.is_ok():
            yield complete(result.unwrap())
        else:
            trace = result.unwrap_err()
            yield ToolProgress(
                kind=ProgressKind.ERROR,
                message=trace.message,
                data={"error": trace.message, "code": trace.error_code},
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
    # Result Streaming (Incremental Output)
    # ─────────────────────────────────────────────────────────────────
    
    @property
    def supports_result_streaming(self) -> bool:
        """Whether this tool supports incremental result streaming.
        
        Override in subclasses that implement stream_result().
        """
        return False
    
    async def stream_result(self, params: TParams) -> AsyncIterator[str]:
        """Stream result chunks during execution.
        
        Override in tools that produce incremental output (e.g., LLM tools).
        Default implementation runs normally and yields complete result.
        
        Yields:
            String chunks of the result as they become available
        
        Example:
            >>> async for chunk in tool.stream_result(params):
            ...     print(chunk, end="", flush=True)
        """
        result = await self._async_run(params)
        yield result
    
    async def stream_result_events(self, params: TParams) -> AsyncIterator[StreamEvent]:
        """Stream result as typed events with metadata.
        
        Wraps stream_result() with start/chunk/complete/error event lifecycle.
        Useful for WebSocket/SSE delivery with full state tracking.
        
        Yields:
            StreamEvent objects for transport serialization
        """
        tool_name = self.metadata.name
        yield stream_start(tool_name)
        
        accumulated: list[str] = []
        index = 0
        start_time = time.time()
        
        try:
            async for content in self.stream_result(params):
                accumulated.append(content)
                chunk = StreamChunk(content=content, index=index)
                yield StreamEvent(
                    kind=StreamEventKind.CHUNK,
                    tool_name=tool_name,
                    data=chunk,
                )
                index += 1
            
            yield stream_complete(tool_name, "".join(accumulated))
        except Exception as e:
            yield stream_error(tool_name, str(e))
            raise
    
    async def stream_result_collected(
        self,
        params: TParams,
        timeout: float = 60.0,
    ) -> StreamResult[str]:
        """Stream and collect full result with metadata.
        
        Useful when you want streaming behavior but also need final stats.
        
        Returns:
            StreamResult with accumulated content and timing metadata
        """
        start = time.time()
        parts: list[str] = []
        chunk_count = 0
        
        async def collect() -> StreamResult[str]:
            nonlocal parts, chunk_count
            async for content in self.stream_result(params):
                parts.append(content)
                chunk_count += 1
            return StreamResult(
                value="".join(parts),
                chunks=chunk_count,
                duration_ms=(time.time() - start) * 1000,
                tool_name=self.metadata.name,
            )
        
        return await asyncio.wait_for(collect(), timeout=timeout)
    
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
    
    # ─────────────────────────────────────────────────────────────────
    # Composition
    # ─────────────────────────────────────────────────────────────────
    
    def __rshift__(self, other: BaseTool[BaseModel]) -> BaseTool[BaseModel]:
        """Chain tools: self >> other creates a sequential pipeline.
        
        Example:
            >>> search = SearchTool()
            >>> summarize = SummarizeTool()
            >>> pipeline = search >> summarize
        """
        from ..pipeline import PipelineTool, Step
        return PipelineTool(steps=[Step(self), Step(other)])