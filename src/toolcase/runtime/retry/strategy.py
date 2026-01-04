"""Composable retry strategies combining retry, fallback, and escalation.

Enables declarative resilience patterns like:
"Retry 3x with exponential backoff → fallback to alternatives → escalate to human"

Example:
    >>> strategy = (
    ...     RetryStrategy()
    ...     .with_retry(max_retries=3, backoff=ExponentialBackoff())
    ...     .with_fallback([BackupAPI(), CacheAPI()], timeout=5.0)
    ...     .with_escalation(QueueEscalation("approvals"))
    ... )
    >>> resilient = strategy.wrap(PrimaryAPI())
    
    >>> # Or use the helper function
    >>> resilient = resilient_tool(
    ...     PrimaryAPI(),
    ...     retry=RetryPolicy(max_retries=3),
    ...     fallback=[BackupAPI()],
    ...     escalate="approval_queue",
    ... )
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError

from toolcase.foundation.errors import Err, ErrorCode, ErrorTrace, JsonDict, ToolResult, format_validation_error
from toolcase.runtime.concurrency import CancelScope, checkpoint

from .backoff import Backoff, ExponentialBackoff
from .policy import DEFAULT_RETRYABLE, RetryPolicy

if TYPE_CHECKING:
    from toolcase.foundation.core.base import BaseTool, ToolMetadata
    from toolcase.runtime.agents.escalation import EscalationHandler, EscalationRequest


logger = logging.getLogger("toolcase.retry.strategy")


@runtime_checkable
class _EscalationHandlerProtocol(Protocol):
    async def escalate(self, request: EscalationRequest) -> object: ...  # Avoids circular import


@dataclass(frozen=True, slots=True)
class RetryStage:
    policy: RetryPolicy


DEFAULT_FALLBACK_CODES: frozenset[ErrorCode] = frozenset({
    ErrorCode.RATE_LIMITED, ErrorCode.TIMEOUT, ErrorCode.NETWORK_ERROR,
    ErrorCode.EXTERNAL_SERVICE_ERROR, ErrorCode.UNKNOWN,
})


@dataclass(frozen=True, slots=True)
class FallbackStage:
    tools: tuple[BaseModel, ...]  # Actually BaseTool, typed loosely for circular import
    timeout: float = 30.0
    fallback_on: frozenset[ErrorCode] = field(default_factory=lambda: DEFAULT_FALLBACK_CODES)


@dataclass(frozen=True, slots=True)
class EscalateStage:
    handler: _EscalationHandlerProtocol


Stage = RetryStage | FallbackStage | EscalateStage


class RetryStrategy:
    """Composable retry strategy builder.
    
    Combines retry policies, fallback chains, and escalation into a single
    declarative strategy that can wrap any tool.
    
    Stages execute in order:
    1. Retry: Retry the primary tool with backoff
    2. Fallback: Try alternative tools in sequence
    3. Escalate: Request human intervention
    
    Each stage only triggers if the previous stage's result is an error.
    
    Example:
        >>> strategy = (
        ...     RetryStrategy()
        ...     .with_retry(max_retries=3, backoff=ExponentialBackoff(base=1.0))
        ...     .with_fallback([BackupAPI()], timeout=5.0)
        ...     .with_escalation(SlackEscalation("#approvals"))
        ... )
        >>> 
        >>> # Wrap primary tool
        >>> search = strategy.wrap(GoogleAPI())
        >>> 
        >>> # Now search will:
        >>> # 1. Try GoogleAPI up to 3 times with exponential backoff
        >>> # 2. On failure, try BackupAPI
        >>> # 3. On failure, escalate to Slack
    """
    
    __slots__ = ("_stages",)
    
    def __init__(self, stages: tuple[Stage, ...] = ()) -> None:
        self._stages = stages
    
    def with_retry(
        self, max_retries: int = 3, backoff: Backoff | None = None,
        retryable_codes: frozenset[ErrorCode] | None = None,
        on_retry: RetryPolicy.__annotations__.get("on_retry") = None,  # type: ignore[name-defined]
    ) -> RetryStrategy:
        """Add retry stage. Args: max_retries, backoff (default Exponential), retryable_codes, on_retry callback."""
        return self.with_policy(RetryPolicy(
            max_retries=max_retries, backoff=backoff or ExponentialBackoff(),
            retryable_codes=retryable_codes or DEFAULT_RETRYABLE, on_retry=on_retry,
        ))
    
    def with_policy(self, policy: RetryPolicy) -> RetryStrategy:
        return RetryStrategy((*self._stages, RetryStage(policy)))  # Add retry stage with existing policy
    
    def with_fallback(
        self, tools: list[BaseModel], *, timeout: float = 30.0, fallback_on: frozenset[ErrorCode] | None = None,
    ) -> RetryStrategy:
        """Add fallback stage. Args: tools (alternatives in order), timeout per-tool, fallback_on codes."""
        return RetryStrategy((*self._stages, FallbackStage(tuple(tools), timeout, fallback_on or DEFAULT_FALLBACK_CODES)))
    
    def with_escalation(self, handler: _EscalationHandlerProtocol | str) -> RetryStrategy:
        """Add escalation stage. Args: EscalationHandler or queue name string."""
        from toolcase.runtime.agents.escalation import QueueEscalation
        return RetryStrategy((*self._stages, EscalateStage(QueueEscalation(handler) if isinstance(handler, str) else handler)))
    
    @property
    def stages(self) -> tuple[Stage, ...]:
        return self._stages
    
    def wrap(self, tool: BaseModel, *, name: str | None = None, description: str | None = None) -> ResilientTool:
        """Wrap tool with this strategy, returning ResilientTool that executes stages."""
        return ResilientTool(tool, self, name=name, description=description)
    
    def __repr__(self) -> str:
        names = [type(s).__name__.replace("Stage", "").lower() for s in self._stages]
        return f"RetryStrategy([{' → '.join(names) or 'empty'}])"


class ResilientParams(BaseModel):
    """Parameters for resilient tool execution."""
    input: JsonDict = Field(default_factory=dict, description="Input parameters for the tool")


ResilientParams.model_rebuild()


def _lazy_imports():
    """Lazy import to avoid circular import."""
    from toolcase.foundation.core.base import BaseTool, ToolMetadata
    return BaseTool, ToolMetadata


class ResilientTool:
    """Tool wrapped with a composable retry strategy.
    
    Executes strategy stages in order until success or all stages fail.
    Created via RetryStrategy.wrap() or resilient_tool().
    Implements tool protocol without inheriting BaseTool to avoid circular imports.
    """
    
    __slots__ = ("_tool", "_strategy", "_meta", "_params_schema")
    
    def __init__(self, tool: BaseModel, strategy: RetryStrategy, *, name: str | None = None, description: str | None = None) -> None:
        self._tool, self._strategy, self._params_schema = tool, strategy, ResilientParams
        _, ToolMetadata = _lazy_imports()
        stage_desc = " → ".join(type(s).__name__.replace("Stage", "").lower() for s in strategy.stages)
        self._meta = ToolMetadata(
            name=name or f"resilient_{tool.metadata.name}",
            description=description or f"{tool.metadata.name} with {stage_desc or 'no'} resilience",
            category="agents", streaming=tool.metadata.streaming,
        )
    
    @property
    def metadata(self) -> ToolMetadata:
        return self._meta
    
    @property
    def params_schema(self) -> type[ResilientParams]:
        return self._params_schema
    
    @property
    def tool(self) -> BaseTool[BaseModel]:
        return self._tool  # Wrapped primary tool
    
    @property
    def strategy(self) -> RetryStrategy:
        return self._strategy
    
    @property
    def cache_enabled(self) -> bool:
        return False
    
    def _coerce_params(self, params: ResilientParams | JsonDict) -> ResilientParams:
        return params if isinstance(params, ResilientParams) else ResilientParams(**params)
    
    async def arun(self, params: ResilientParams | JsonDict) -> str:
        """Async execution with string result."""
        r = await self.arun_result(self._coerce_params(params))
        return r.unwrap() if r.is_ok() else r.unwrap_err().message
    
    async def arun_result(self, params: ResilientParams | JsonDict) -> ToolResult:
        """Execute strategy stages in order until success."""
        params = self._coerce_params(params)
        
        # Validate params for primary tool
        try:
            tool_params = self._tool.params_schema(**params.input)
        except ValidationError as e:
            return Err(ErrorTrace(format_validation_error(e, tool_name=self._tool.metadata.name), ErrorCode.INVALID_PARAMS.value, False))
        
        if (result := await self._tool.arun_result(tool_params)).is_ok():
            return result
        last_error = result.unwrap_err()
        
        # Execute each stage
        for stage in self._strategy.stages:
            await checkpoint()
            match stage:
                case RetryStage(policy=p):
                    result, last_error = await self._execute_retry(tool_params, p, last_error)
                case FallbackStage(tools=t, timeout=to, fallback_on=c):
                    result, last_error = await self._execute_fallback(params.input, t, to, c, last_error)
                case EscalateStage(handler=h):
                    result, last_error = await self._execute_escalate(params.input, h, last_error)
            if result.is_ok():
                return result
        
        return Err(last_error.with_operation(f"resilient:{self._meta.name}"))
    
    def run(self, params: ResilientParams | JsonDict) -> str:
        return asyncio.get_event_loop().run_until_complete(self.arun(params))  # Sync string result
    
    def run_result(self, params: ResilientParams | JsonDict) -> ToolResult:
        return asyncio.get_event_loop().run_until_complete(self.arun_result(params))  # Sync Result
    
    async def _execute_retry(self, tool_params: BaseModel, policy: RetryPolicy, last_error: ErrorTrace) -> tuple[ToolResult, ErrorTrace]:
        """Execute retry stage."""
        if policy.is_disabled:
            return Err(last_error), last_error
        
        for attempt in range(policy.max_retries):
            code = last_error.error_code or ErrorCode.UNKNOWN.value
            if not policy.should_retry(code, attempt):
                break
            delay = policy.get_delay(attempt)
            logger.info(f"[{self._meta.name}] Retry {attempt + 1}/{policy.max_retries} after {delay:.1f}s (code: {code})")
            if policy.on_retry:
                policy.on_retry(attempt, ErrorCode(code), delay)
            await asyncio.sleep(delay)
            await checkpoint()
            if (result := await self._tool.arun_result(tool_params)).is_ok():
                return result, last_error
            last_error = result.unwrap_err()
        
        return Err(last_error), last_error
    
    async def _execute_fallback(
        self, input_dict: JsonDict, tools: tuple[BaseModel, ...], timeout: float, fallback_on: frozenset[ErrorCode], last_error: ErrorTrace,
    ) -> tuple[ToolResult, ErrorTrace]:
        """Execute fallback stage."""
        for tool in tools:
            try:
                tool_params = tool.params_schema(**input_dict)
            except ValidationError:
                continue  # Skip tools that can't accept these params
            
            async with CancelScope(timeout=timeout) as scope:
                result = await tool.arun_result(tool_params)
            
            if scope.cancel_called:
                last_error = ErrorTrace(f"Fallback {tool.metadata.name} timed out after {timeout}s", ErrorCode.TIMEOUT.value, True)
                continue
            if result.is_ok():
                return result, last_error
            last_error = result.unwrap_err()
            # Check if error triggers next fallback
            try:
                if (ErrorCode(last_error.error_code) if last_error.error_code else ErrorCode.UNKNOWN) not in fallback_on:
                    break  # Non-fallback error, stop trying
            except ValueError:
                pass  # Unknown code, continue to next fallback
        return Err(last_error), last_error
    
    async def _execute_escalate(
        self,
        input_dict: JsonDict,
        handler: _EscalationHandlerProtocol,
        last_error: ErrorTrace,
    ) -> tuple[ToolResult, ErrorTrace]:
        """Execute escalation stage."""
        from toolcase.runtime.agents.escalation import EscalationRequest, EscalationStatus
        from toolcase.foundation.errors import Ok
        from datetime import datetime
        
        request = EscalationRequest(
            tool_name=self._tool.metadata.name,
            params=input_dict,
            error=last_error,
            attempt=0,
            timestamp=datetime.utcnow(),
        )
        
        logger.info(f"[{self._meta.name}] Escalating to human: {last_error.message}")
        
        esc = await handler.escalate(request)
        if esc.should_proceed:
            return Ok(esc.value or f"Approved by {esc.reviewer or 'human'}"), last_error
        
        return Err(ErrorTrace(
            message=f"Escalation {esc.status.value}: {esc.reason or 'no reason'}",
            error_code=ErrorCode.PERMISSION_DENIED.value if esc.status == EscalationStatus.REJECTED else ErrorCode.TIMEOUT.value,
            recoverable=False,
        )), last_error
    
    def __repr__(self) -> str:
        return f"ResilientTool({self._tool.metadata.name}, {self._strategy})"


def resilient_tool(
    tool: BaseModel,  # Actually BaseTool
    *,
    retry: RetryPolicy | int | None = None,
    fallback: list[BaseModel] | None = None,  # Actually list[BaseTool]
    fallback_timeout: float = 30.0,
    escalate: _EscalationHandlerProtocol | str | None = None,
    name: str | None = None,
    description: str | None = None,
) -> ResilientTool:
    """Create a resilient tool with composed retry, fallback, and escalation.
    
    Convenience function for building common resilience patterns.
    
    Args:
        tool: Primary tool to wrap
        retry: RetryPolicy or max_retries int (default: 3 retries)
        fallback: Alternative tools to try on failure
        fallback_timeout: Per-tool timeout for fallbacks
        escalate: EscalationHandler or queue name for human escalation
        name: Optional tool name
        description: Optional description
    
    Returns:
        ResilientTool with configured strategy
    
    Example:
        >>> # Simple retry
        >>> search = resilient_tool(GoogleAPI(), retry=3)
        >>> 
        >>> # Full strategy
        >>> search = resilient_tool(
        ...     GoogleAPI(),
        ...     retry=RetryPolicy(max_retries=3, backoff=ExponentialBackoff()),
        ...     fallback=[BingAPI(), CacheAPI()],
        ...     escalate="search_approvals",
        ... )
    """
    strategy = RetryStrategy()
    
    if retry is not None:
        policy = retry if isinstance(retry, RetryPolicy) else RetryPolicy(max_retries=retry)
        strategy = strategy.with_policy(policy)
    
    if fallback:
        strategy = strategy.with_fallback(fallback, timeout=fallback_timeout)
    
    if escalate is not None:
        strategy = strategy.with_escalation(escalate)
    
    return strategy.wrap(tool, name=name, description=description)
