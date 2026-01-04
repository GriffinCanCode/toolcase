"""Core tool abstractions: BaseTool, ToolMetadata, and parameter types.

This module provides the foundation for building type-safe, extensible tools
that AI agents can invoke. Tools are defined by subclassing BaseTool with
a typed parameter schema.
"""

from __future__ import annotations

import asyncio
import re
import time
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Annotated,
    AsyncIterator,
    Callable,
    ClassVar,
    Generic,
    TypeVar,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    computed_field,
    field_serializer,
    field_validator,
)

from toolcase.io.cache import DEFAULT_TTL, get_cache
from toolcase.foundation.errors import (
    ErrorCode,
    ErrorTrace,
    Result,
    ToolError,
    ToolResult,
    classify_exception,
    result_to_string,
    string_to_result,
)
from toolcase.io.progress import ProgressCallback, ProgressKind, ToolProgress, complete
from toolcase.runtime.retry import RetryPolicy, execute_with_retry, execute_with_retry_sync
from toolcase.runtime.concurrency import run_sync, to_thread
from toolcase.io.streaming import (
    StreamChunk,
    StreamEvent,
    StreamEventKind,
    StreamResult,
    stream_complete,
    stream_error,
    stream_start,
)

if TYPE_CHECKING:
    from toolcase.runtime.batch import BatchConfig, BatchResult

# Internal constants for fast Result construction
from toolcase.foundation.errors.result import _ERR, _OK


class ToolCapabilities(BaseModel):
    """Advertised tool capabilities for intelligent scheduling and execution.
    
    Tools declare capabilities that the registry/scheduler can use for:
    - Caching decisions (skip cache lookup for non-cacheable tools)
    - Concurrency limits (respect max_concurrent for rate-limited APIs)
    - Streaming support (route to streaming pipeline when supported)
    - Idempotency hints (safe to retry without side effects)
    
    Example:
        >>> caps = ToolCapabilities(
        ...     supports_caching=True,
        ...     supports_streaming=True,
        ...     max_concurrent=5,
        ...     idempotent=True,
        ... )
    """
    
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    supports_caching: bool = Field(
        default=True,
        description="Whether results can be cached. False for time-sensitive or stateful tools.",
    )
    supports_streaming: bool = Field(
        default=False,
        description="Whether tool can stream incremental results (LLM output, progress).",
    )
    max_concurrent: int | None = Field(
        default=None,
        ge=1,
        description="Max concurrent executions (None=unlimited). For rate-limited APIs.",
    )
    idempotent: bool = Field(
        default=True,
        description="Whether repeated calls with same params produce same result safely.",
    )
    estimated_latency_ms: int | None = Field(
        default=None,
        ge=0,
        description="Typical execution time hint for scheduling (None=unknown).",
    )
    requires_confirmation: bool = Field(
        default=False,
        description="Whether tool should require user confirmation before execution.",
    )
    
    @classmethod
    def default(cls) -> "ToolCapabilities":
        """Default capabilities for standard tools."""
        return cls()
    
    @classmethod
    def streaming(cls, max_concurrent: int | None = None) -> "ToolCapabilities":
        """Preset for streaming-capable tools."""
        return cls(supports_streaming=True, max_concurrent=max_concurrent)
    
    @classmethod
    def non_cacheable(cls, idempotent: bool = False) -> "ToolCapabilities":
        """Preset for tools with non-cacheable results."""
        return cls(supports_caching=False, idempotent=idempotent)
    
    @classmethod
    def rate_limited(cls, max_concurrent: int, estimated_latency_ms: int | None = None) -> "ToolCapabilities":
        """Preset for rate-limited external APIs."""
        return cls(max_concurrent=max_concurrent, estimated_latency_ms=estimated_latency_ms)


class ToolMetadata(BaseModel):
    """Metadata for tool discovery, LLM selection, UI display, and API key validation."""
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
        revalidate_instances="never",  # Performance: skip revalidation when passing instances
        extra="forbid",  # Catch typos in field names
        json_schema_extra={
            "title": "Tool Metadata",
            "description": "Describes a tool's identity and capabilities for AI agents",
        },
    )
    
    # Pre-compiled regex patterns for performance (class-level)
    _NAME_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")
    _CAMEL_TO_SNAKE_1: ClassVar[re.Pattern[str]] = re.compile(r"(.)([A-Z][a-z]+)")
    _CAMEL_TO_SNAKE_2: ClassVar[re.Pattern[str]] = re.compile(r"([a-z0-9])([A-Z])")
    
    name: Annotated[str, Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        json_schema_extra={"examples": ["web_search", "code_interpreter"]},
    )]
    description: Annotated[str, Field(
        min_length=10,
        json_schema_extra={"examples": ["Search the web for information and return relevant results"]},
    )]
    category: Annotated[str, Field(
        default="general",
        pattern=r"^[a-z][a-z0-9_-]*$",
    )]
    requires_api_key: bool = False
    enabled: bool = True
    streaming: bool = False
    tags: frozenset[str] = Field(default_factory=frozenset, repr=False)  # Exclude from repr (can be verbose)
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    capabilities: ToolCapabilities = Field(default_factory=ToolCapabilities)
    
    @field_validator("capabilities", mode="before")
    @classmethod
    def _coerce_capabilities(cls, v: ToolCapabilities | dict | None) -> ToolCapabilities:
        """Accept dict or ToolCapabilities, return ToolCapabilities."""
        if v is None:
            return ToolCapabilities()
        return v if isinstance(v, ToolCapabilities) else ToolCapabilities(**v)
    
    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, v: str) -> str:
        """Normalize name to snake_case using pre-compiled patterns."""
        return cls._CAMEL_TO_SNAKE_2.sub(r"\1_\2", cls._CAMEL_TO_SNAKE_1.sub(r"\1_\2", v)).lower() if isinstance(v, str) else v
    
    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, v: frozenset[str] | set[str] | list[str] | tuple[str, ...] | None) -> frozenset[str]:
        """Accept various iterables for tags, normalize to frozenset."""
        return v if isinstance(v, frozenset) else frozenset(v or ())
    
    @field_serializer("tags")
    def _serialize_tags(self, v: frozenset[str]) -> list[str]:
        """Serialize frozenset as sorted list for consistent JSON output."""
        return sorted(v)
    
    @computed_field
    @property
    def display_name(self) -> str:
        """Human-readable name derived from snake_case name."""
        return self.name.replace("_", " ").title()
    
    @computed_field
    @property
    def short_description(self) -> str:
        """First sentence of description for compact display."""
        return self.description.split(". ", 1)[0].rstrip(".")
    
    # ─────────────────────────────────────────────────────────────────
    # Capability Accessors (convenience)
    # ─────────────────────────────────────────────────────────────────
    
    @property
    def supports_caching(self) -> bool:
        """Whether tool results can be cached."""
        return self.capabilities.supports_caching
    
    @property
    def supports_result_streaming(self) -> bool:
        """Whether tool can stream incremental results."""
        return self.capabilities.supports_streaming or self.streaming
    
    @property
    def max_concurrent(self) -> int | None:
        """Max concurrent executions (None=unlimited)."""
        return self.capabilities.max_concurrent
    
    @property
    def is_idempotent(self) -> bool:
        """Whether repeated calls are safe."""
        return self.capabilities.idempotent
    
    @property
    def requires_confirmation(self) -> bool:
        """Whether tool should require user confirmation."""
        return self.capabilities.requires_confirmation
    
    def __hash__(self) -> int:
        """Explicit hash for frozen model (enables set/dict membership)."""
        return hash((self.name, self.version))


class EmptyParams(BaseModel):
    """Default parameter schema for tools with no required inputs."""
    
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,  # Immutable params for caching
        revalidate_instances="never",
    )
    
    def __hash__(self) -> int:
        return hash(())  # Empty params always hash the same


# TypeAdapter for high-frequency dict->params validation (bypasses model overhead)
_EmptyParamsAdapter: TypeAdapter[EmptyParams] = TypeAdapter(EmptyParams)


# Type variable for tool parameter schemas
TParams = TypeVar("TParams", bound=BaseModel)


class BaseTool(ABC, Generic[TParams]):
    """Abstract base for tools. Subclasses define metadata, params_schema, and implement _run(params) -> str."""
    
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
        """Create a standardized error response string."""
        import traceback
        return ToolError(
            tool_name=self.metadata.name, message=message, code=code,
            recoverable=recoverable, details=traceback.format_exc() if include_trace else None,
        ).render()
    
    def _error_from_exception(self, exc: Exception, context: str = "", *, recoverable: bool = True) -> str:
        """Create error response from caught exception."""
        return ToolError.from_exception(self.metadata.name, exc, context, recoverable=recoverable).render()
    
    # ─────────────────────────────────────────────────────────────────
    # Result Helpers
    # ─────────────────────────────────────────────────────────────────
    
    def _ok(self, value: str) -> ToolResult:
        """Create Ok result (convenience method)."""
        return Result(value, _OK)
    
    def _err(self, message: str, code: ErrorCode = ErrorCode.UNKNOWN, *, recoverable: bool = True) -> ToolResult:
        """Create Err result from error parameters."""
        from toolcase.foundation.errors.types import ErrorContext, _EMPTY_META
        return Result(ErrorTrace(
            message=message, error_code=code.value, recoverable=recoverable, details=None,
            contexts=(ErrorContext(operation=f"tool:{self.metadata.name}", location="", metadata=_EMPTY_META),),
        ), _ERR)
    
    def _try(self, operation: Callable[[], str], *, context: str = "") -> ToolResult:
        """Execute operation with automatic exception handling."""
        try:
            return Result(operation(), _OK)
        except Exception as e:
            return self._err_from_exc(e, context)
    
    async def _try_async(self, operation: Callable[[], str], *, context: str = "") -> ToolResult:
        """Execute async operation with automatic exception handling."""
        try:
            result = await operation() if asyncio.iscoroutinefunction(operation) else await to_thread(operation)  # type: ignore[misc]
            return Result(result, _OK)
        except Exception as e:
            return self._err_from_exc(e, context)
    
    def _err_from_exc(self, exc: Exception, context: str = "") -> ToolResult:
        """Create Err result from exception (internal helper)."""
        import traceback
        from toolcase.foundation.errors.types import _EMPTY_META, ErrorContext
        tool_ctx = ErrorContext(operation=f"tool:{self.metadata.name}", location="", metadata=_EMPTY_META)
        return Result(ErrorTrace(
            message=f"{context}: {exc}" if context else str(exc),
            contexts=(tool_ctx, ErrorContext(operation=context, location="", metadata=_EMPTY_META)) if context else (tool_ctx,),
            error_code=classify_exception(exc).value, recoverable=True, details=traceback.format_exc(),
        ), _ERR)
    
    # ─────────────────────────────────────────────────────────────────
    # Core Execution (Async-First Design)
    # ─────────────────────────────────────────────────────────────────
    
    @abstractmethod
    async def _async_run(self, params: TParams) -> str:
        """Primary execution method - implement async logic here.
        
        For sync-only operations, return directly (no await needed):
            async def _async_run(self, params): return do_sync_work()
        """
        ...
    
    def _run(self, params: TParams) -> str:
        """Sync wrapper - calls _async_run via run_sync. Override only if needed."""
        return run_sync(self._async_run(params))
    
    async def _async_run_result(self, params: TParams) -> ToolResult:
        """Execute asynchronously with Result-based error handling; catches exceptions as Err."""
        try:
            return string_to_result(await self._async_run(params), self.metadata.name)
        except Exception as e:
            return self._err_from_exc(e, "async execution")
    
    def _run_result(self, params: TParams) -> ToolResult:
        """Sync wrapper for Result-based execution."""
        return run_sync(self._async_run_result(params))
    
    def run(self, params: TParams) -> str:
        """Execute with caching support. Checks cache first, caches successful results only."""
        if not self.cache_enabled:
            return result_to_string(self._run_result(params), self.metadata.name)
        
        cache, tool_name = get_cache(), self.metadata.name
        if (cached := cache.get(tool_name, params)) is not None:
            return cached
        
        result = self._run_result(params)
        output = result_to_string(result, tool_name)
        if result.is_ok():
            cache.set(tool_name, params, output, self.cache_ttl)
        return output
    
    def run_result(self, params: TParams) -> ToolResult:
        """Execute with caching and retry, returning Result type. Enables monadic error handling."""
        cache, tool_name = (get_cache() if self.cache_enabled else None), self.metadata.name
        
        if cache and (cached := cache.get(tool_name, params)) is not None:
            return string_to_result(cached, tool_name)
        
        result = (execute_with_retry_sync(lambda: self._run_result(params), self.retry_policy, tool_name)
                  if self.retry_policy else self._run_result(params))
        
        if cache and result.is_ok():
            cache.set(tool_name, params, result_to_string(result, tool_name), self.cache_ttl)
        return result
    
    async def arun(self, params: TParams, timeout: float = 30.0) -> str:
        """Execute asynchronously with caching and timeout."""
        if not self.cache_enabled:
            return result_to_string(await asyncio.wait_for(self._async_run_result(params), timeout=timeout), self.metadata.name)
        
        cache, tool_name = get_cache(), self.metadata.name
        if (cached := cache.get(tool_name, params)) is not None:
            return cached
        
        result = await asyncio.wait_for(self._async_run_result(params), timeout=timeout)
        output = result_to_string(result, tool_name)
        if result.is_ok():
            cache.set(tool_name, params, output, self.cache_ttl)
        return output
    
    async def arun_result(self, params: TParams, timeout: float = 30.0) -> ToolResult:
        """Execute asynchronously with Result type, caching, retry, and timeout."""
        cache, tool_name = (get_cache() if self.cache_enabled else None), self.metadata.name
        
        if cache and (cached := cache.get(tool_name, params)) is not None:
            return string_to_result(cached, tool_name)
        
        # Execute with optional retry (timeout wraps entire retry sequence)
        coro = (execute_with_retry(lambda: self._async_run_result(params), self.retry_policy, tool_name)
                if self.retry_policy else self._async_run_result(params))
        result = await asyncio.wait_for(coro, timeout=timeout)
        
        if cache and result.is_ok():
            cache.set(tool_name, params, result_to_string(result, tool_name), self.cache_ttl)
        return result
    
    # ─────────────────────────────────────────────────────────────────
    # Streaming Progress
    # ─────────────────────────────────────────────────────────────────
    
    @property
    def supports_streaming(self) -> bool:
        """Whether this tool supports progress streaming."""
        return self.metadata.streaming
    
    async def stream_run(self, params: TParams) -> AsyncIterator[ToolProgress]:
        """Stream progress events during execution. Override for real-time updates."""
        yield ToolProgress(kind=ProgressKind.STATUS, message="Starting...")
        result = await self._async_run_result(params)
        if result.is_ok():
            yield complete(result.unwrap())
        else:
            trace = result.unwrap_err()
            yield ToolProgress(kind=ProgressKind.ERROR, message=trace.message, data={"error": trace.message, "code": trace.error_code})
    
    async def arun_with_progress(self, params: TParams, on_progress: ProgressCallback | None = None, timeout: float = 60.0) -> str:
        """Execute with progress callbacks. Collects events and calls callback for each."""
        async def execute() -> str:
            result = ""
            async for p in self.stream_run(params):
                on_progress and on_progress(p)
                if p.kind == ProgressKind.COMPLETE and p.data:
                    result = str(p.data.get("result", ""))
                elif p.kind == ProgressKind.ERROR:
                    raise RuntimeError(p.message)
            return result
        return await asyncio.wait_for(execute(), timeout=timeout)
    
    # ─────────────────────────────────────────────────────────────────
    # Result Streaming (Incremental Output)
    # ─────────────────────────────────────────────────────────────────
    
    @property
    def supports_result_streaming(self) -> bool:
        """Whether this tool supports incremental result streaming. Override in subclasses that implement stream_result()."""
        return False
    
    async def stream_result(self, params: TParams) -> AsyncIterator[str]:
        """Stream result chunks during execution. Override for incremental output (e.g., LLM tools)."""
        yield await self._async_run(params)
    
    async def stream_result_events(self, params: TParams) -> AsyncIterator[StreamEvent]:
        """Stream result as typed events with metadata. Wraps stream_result() with start/chunk/complete/error lifecycle."""
        tool_name = self.metadata.name
        yield stream_start(tool_name)
        accumulated: list[str] = []
        try:
            async for content in self.stream_result(params):
                accumulated.append(content)
                yield StreamEvent(kind=StreamEventKind.CHUNK, tool_name=tool_name, data=StreamChunk(content=content, index=len(accumulated) - 1))
            yield stream_complete(tool_name, "".join(accumulated))
        except Exception as e:
            yield stream_error(tool_name, str(e))
            raise
    
    async def stream_result_collected(self, params: TParams, timeout: float = 60.0) -> StreamResult[str]:
        """Stream and collect full result with metadata. Returns StreamResult with accumulated content and timing."""
        start, parts = time.time(), [c async for c in self.stream_result(params)]
        return StreamResult(value="".join(parts), chunks=len(parts), duration_ms=(time.time() - start) * 1000, tool_name=self.metadata.name)
    
    # ─────────────────────────────────────────────────────────────────
    # Invocation (kwargs interface)
    # ─────────────────────────────────────────────────────────────────
    
    def __call__(self, **kwargs: object) -> str:
        """Invoke tool with keyword arguments. Convenience method that constructs params from kwargs."""
        return self.run(self.params_schema(**kwargs))  # type: ignore[call-arg, arg-type]
    
    async def acall(self, **kwargs: object) -> str:
        """Async invoke with keyword arguments."""
        return await self.arun(self.params_schema(**kwargs))  # type: ignore[call-arg, arg-type]
    
    # ─────────────────────────────────────────────────────────────────
    # Composition
    # ─────────────────────────────────────────────────────────────────
    
    def __rshift__(self, other: BaseTool[BaseModel]) -> BaseTool[BaseModel]:
        """Chain tools: self >> other creates a sequential pipeline."""
        from toolcase.runtime.pipeline import PipelineTool, Step
        return PipelineTool(steps=[Step(self), Step(other)])
    
    # ─────────────────────────────────────────────────────────────────
    # Batch Execution
    # ─────────────────────────────────────────────────────────────────
    
    async def batch_run(
        self,
        params_list: list[TParams],
        config: BatchConfig | None = None,
    ) -> BatchResult[TParams]:
        """Execute tool against multiple parameter sets concurrently.
        
        Provides intelligent batching with configurable concurrency,
        partial failure handling, and result aggregation.
        
        Args:
            params_list: List of parameter objects to execute
            config: Batch configuration (concurrency, fail_fast, etc.)
        
        Returns:
            BatchResult with all outcomes, metrics, and convenience accessors
        
        Example:
            >>> params = [SearchParams(q=q) for q in ["python", "rust", "go"]]
            >>> results = await search_tool.batch_run(params, BatchConfig(concurrency=3))
            >>> print(f"Success: {results.success_rate:.0%}")
            >>> for item in results.successes:
            ...     print(f"[{item.index}] {item.value[:50]}...")
        """
        from toolcase.runtime.batch import batch_execute
        return await batch_execute(self, params_list, config)
    
    def batch_run_sync(
        self,
        params_list: list[TParams],
        config: BatchConfig | None = None,
    ) -> BatchResult[TParams]:
        """Synchronous batch execution. Wraps batch_run for sync contexts."""
        return run_sync(self.batch_run(params_list, config))