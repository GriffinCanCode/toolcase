"""Escalation primitive for human-in-the-loop patterns.

Retries automated execution, then escalates to humans when automation fails.
Useful for:
- High-stakes operations requiring approval
- Edge cases automation can't handle
- Audit trails for sensitive actions
- Confidence-based human review

Example:
    >>> safe_delete = retry_with_escalation(
    ...     DeleteTool(),
    ...     max_retries=2,
    ...     escalate_to=QueueEscalation("approval_queue"),
    ... )
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError

from toolcase.foundation.core.base import BaseTool, ToolMetadata
from toolcase.foundation.errors import Err, ErrorCode, ErrorTrace, JsonDict, Ok, ToolResult, format_validation_error
from toolcase.runtime.concurrency import to_thread, checkpoint

if TYPE_CHECKING:
    from collections.abc import Awaitable


logger = logging.getLogger("toolcase.agents.escalation")


class EscalationStatus(str, Enum):
    """Status of an escalation request."""
    PENDING = "pending"      # Awaiting human review
    APPROVED = "approved"    # Human approved, proceed
    REJECTED = "rejected"    # Human rejected
    TIMEOUT = "timeout"      # Human didn't respond in time
    OVERRIDE = "override"    # Manual value provided


@dataclass(frozen=True, slots=True)
class EscalationResult:
    """Result from escalation handler. Attrs: status, value (override if APPROVED/OVERRIDE), reason, reviewer"""
    status: EscalationStatus
    value: str | None = None
    reason: str | None = None
    reviewer: str | None = None
    
    @property
    def should_proceed(self) -> bool:
        """Whether execution should proceed with the result."""
        return self.status in {EscalationStatus.APPROVED, EscalationStatus.OVERRIDE}


@dataclass(slots=True)
class EscalationRequest:
    """Request sent to escalation handler. Contains all context needed for human review."""
    tool_name: str
    params: JsonDict
    error: ErrorTrace
    attempt: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: JsonDict = field(default_factory=dict)


@runtime_checkable
class EscalationHandler(Protocol):
    """Protocol for escalation handlers. Implement for queue/webhook/DB/CLI integrations."""
    
    async def escalate(self, request: EscalationRequest) -> EscalationResult:
        """Submit escalation and await resolution."""
        ...


class QueueEscalation:
    """Queue-based escalation handler (async polling). Override publish/poll for your queue system.
    
    Example:
        >>> class RedisEscalation(QueueEscalation):
        ...     async def publish(self, request):
        ...         await redis.lpush(self.queue_name, request.json())
        ...     async def poll(self, request_id):
        ...         return await redis.brpop(f"response:{request_id}", timeout=60)
    """
    
    def __init__(self, queue_name: str, *, timeout: float = 300.0, poll_interval: float = 1.0) -> None:
        self.queue_name, self.timeout, self.poll_interval = queue_name, timeout, poll_interval
        self._pending: dict[str, EscalationResult | None] = {}
    
    def _request_id(self, request: EscalationRequest) -> str:
        """Generate unique request ID."""
        from hashlib import sha256
        return sha256(f"{request.tool_name}:{request.timestamp.isoformat()}:{id(request)}".encode()).hexdigest()[:16]
    
    async def publish(self, request: EscalationRequest, request_id: str) -> None:
        """Publish request to queue. Override for your system."""
        logger.info(f"[{self.queue_name}] Escalation {request_id}: {request.tool_name}")
        self._pending[request_id] = None  # Default: just log (for testing)
    
    async def poll(self, request_id: str) -> EscalationResult | None:
        """Poll for response. Override for your system."""
        return self._pending.get(request_id)  # Default: check in-memory dict (for testing)
    
    def resolve(self, request_id: str, result: EscalationResult) -> None:
        """Manually resolve a pending escalation (for testing)."""
        self._pending[request_id] = result
    
    async def escalate(self, request: EscalationRequest) -> EscalationResult:
        """Submit and poll for resolution."""
        request_id = self._request_id(request)
        await self.publish(request, request_id)
        
        for _ in range(int(self.timeout / self.poll_interval)):
            if (result := await self.poll(request_id)) is not None:
                return result
            await checkpoint()
            await asyncio.sleep(self.poll_interval)
        
        return EscalationResult(status=EscalationStatus.TIMEOUT, reason=f"No response within {self.timeout}s")


class CallbackEscalation:
    """Callback-based escalation for sync workflows (CLI, notebooks, simple approval flows).
    
    Example:
        >>> def cli_approve(request):
        ...     print(f"Approve {request.tool_name}? [y/n]")
        ...     return input().lower() == "y"
        >>> escalation = CallbackEscalation(cli_approve)
    """
    
    def __init__(self, callback: Callable[[EscalationRequest], bool | str | EscalationResult]) -> None:
        self.callback = callback
    
    async def escalate(self, request: EscalationRequest) -> EscalationResult:
        """Call callback and interpret result."""
        match await to_thread(self.callback, request):
            case EscalationResult() as r: return r
            case bool() as b: return EscalationResult(status=EscalationStatus.APPROVED if b else EscalationStatus.REJECTED)
            case str() as s: return EscalationResult(status=EscalationStatus.OVERRIDE, value=s)
            case _: return EscalationResult(status=EscalationStatus.REJECTED, reason="Invalid callback response")


class EscalationParams(BaseModel):
    """Parameters for escalation tool execution."""
    input: JsonDict = Field(default_factory=dict, description="Input parameters for the underlying tool")


EscalationParams.model_rebuild()  # Resolve recursive JsonValue type


class EscalationTool(BaseTool[EscalationParams]):
    """Retry with human escalation on failure. Attempts automated execution up to max_retries, then escalates.
    
    Example:
        >>> delete = EscalationTool(
        ...     tool=DeleteRecordTool(),
        ...     max_retries=2,
        ...     handler=QueueEscalation("delete_approvals"),
        ... )
    """
    
    __slots__ = ("_tool", "_max_retries", "_handler", "_retry_codes", "_meta")
    params_schema = EscalationParams
    cache_enabled = False
    
    def __init__(
        self,
        tool: BaseTool[BaseModel],
        handler: EscalationHandler,
        *,
        max_retries: int = 2,
        retry_codes: frozenset[ErrorCode] | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self._tool, self._handler, self._max_retries = tool, handler, max_retries
        self._retry_codes = retry_codes or frozenset({ErrorCode.RATE_LIMITED, ErrorCode.TIMEOUT, ErrorCode.NETWORK_ERROR})
        self._meta = ToolMetadata(
            name=name or f"escalation_{tool.metadata.name}",
            description=description or f"Retry {tool.metadata.name} with human escalation",
            category="agents", streaming=tool.metadata.streaming,
        )
    
    @property
    def metadata(self) -> ToolMetadata:
        return self._meta
    
    @property
    def tool(self) -> BaseTool[BaseModel]:
        return self._tool
    
    def _should_retry(self, trace: ErrorTrace, attempt: int) -> bool:
        """Check if we should retry based on error code and attempt count."""
        if attempt >= self._max_retries:
            return False
        if not trace.error_code:
            return True
        try:
            return ErrorCode(trace.error_code) in self._retry_codes
        except ValueError:
            return True
    
    def _run(self, params: EscalationParams) -> str:
        return self._run_async_sync(self._async_run(params))
    
    async def _async_run(self, params: EscalationParams) -> str:
        r = await self._async_run_result(params)
        return r.unwrap() if r.is_ok() else r.unwrap_err().message
    
    async def _async_run_result(self, params: EscalationParams) -> ToolResult:
        """Execute with retry and escalation."""
        try:
            tool_params = self._tool.params_schema(**params.input)
        except ValidationError as e:
            return Err(ErrorTrace(
                message=format_validation_error(e, tool_name=self._tool.metadata.name),
                error_code=ErrorCode.INVALID_PARAMS.value, recoverable=False,
            ))
        
        attempt, last_error = 0, None
        while True:
            if (result := await self._tool.arun_result(tool_params)).is_ok():
                return result
            last_error = result.unwrap_err()
            if not self._should_retry(last_error, attempt):
                break
            attempt += 1
            logger.info(f"[{self._meta.name}] Retry {attempt}/{self._max_retries}")
            await checkpoint()
            await asyncio.sleep(0.5 * attempt)  # Simple backoff
        
        # Exhausted retries - escalate to human
        logger.info(f"[{self._meta.name}] Escalating after {attempt} retries")
        request = EscalationRequest(
            tool_name=self._tool.metadata.name, params=params.input,
            error=last_error or ErrorTrace(message="Unknown error"), attempt=attempt,
        )
        
        if (esc := await self._handler.escalate(request)).should_proceed:
            return Ok(esc.value or f"Approved by {esc.reviewer or 'human'}")
        
        return Err(ErrorTrace(
            message=f"Escalation {esc.status.value}: {esc.reason or 'no reason'}",
            error_code=ErrorCode.PERMISSION_DENIED.value if esc.status == EscalationStatus.REJECTED else ErrorCode.TIMEOUT.value,
            recoverable=False,
        ).with_operation(f"escalation:{self._meta.name}"))


def retry_with_escalation(
    tool: BaseTool[BaseModel],
    *,
    max_retries: int = 2,
    escalate_to: EscalationHandler | str,
    retry_codes: frozenset[ErrorCode] | None = None,
    name: str | None = None,
    description: str | None = None,
) -> EscalationTool:
    """Create tool with retry and human escalation.
    
    Retries automated execution up to max_retries times, then escalates to the specified handler for human review.
    
    Args:
        tool: Underlying tool to wrap
        max_retries: Retry attempts before escalation
        escalate_to: EscalationHandler instance or queue name string
        retry_codes: Error codes that trigger retry (default: transient)
        name: Optional tool name
        description: Optional description
    
    Returns:
        EscalationTool instance
    
    Example:
        >>> # With queue name
        >>> safe_delete = retry_with_escalation(
        ...     DeleteTool(),
        ...     max_retries=2,
        ...     escalate_to="delete_approval_queue",
        ... )
        >>>
        >>> # With custom handler
        >>> safe_delete = retry_with_escalation(
        ...     DeleteTool(),
        ...     escalate_to=SlackEscalation("#approvals"),
        ... )
    """
    handler = QueueEscalation(escalate_to) if isinstance(escalate_to, str) else escalate_to
    return EscalationTool(tool, handler, max_retries=max_retries, retry_codes=retry_codes, name=name, description=description)
