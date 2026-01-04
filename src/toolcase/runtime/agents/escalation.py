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
from toolcase.foundation.errors import Err, ErrorCode, ErrorTrace, Ok, ToolResult
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
    """Result from an escalation handler.
    
    Attributes:
        status: Resolution status
        value: Override value if status is APPROVED/OVERRIDE
        reason: Human-provided reason for decision
        reviewer: Identifier of the human reviewer
    """
    
    status: EscalationStatus
    value: str | None = None
    reason: str | None = None
    reviewer: str | None = None
    
    @property
    def should_proceed(self) -> bool:
        """Whether execution should proceed with the result."""
        return self.status in (EscalationStatus.APPROVED, EscalationStatus.OVERRIDE)


@dataclass(slots=True)
class EscalationRequest:
    """Request sent to escalation handler.
    
    Contains all context needed for human review.
    """
    
    tool_name: str
    params: dict[str, object]
    error: ErrorTrace
    attempt: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class EscalationHandler(Protocol):
    """Protocol for escalation handlers.
    
    Implement this to integrate with your approval system:
    - Queue-based (Redis, RabbitMQ, SQS)
    - Webhook-based (Slack, Teams, PagerDuty)
    - Database-based (polling table)
    - Sync blocking (CLI prompt)
    """
    
    async def escalate(self, request: EscalationRequest) -> EscalationResult:
        """Submit escalation and await resolution.
        
        Args:
            request: Full context for human review
        
        Returns:
            EscalationResult with decision
        """
        ...


class QueueEscalation:
    """Queue-based escalation handler (async polling).
    
    Publishes to a named queue and polls for response.
    Override `publish` and `poll` for your queue system.
    
    Example:
        >>> class RedisEscalation(QueueEscalation):
        ...     async def publish(self, request):
        ...         await redis.lpush(self.queue_name, request.json())
        ...     
        ...     async def poll(self, request_id):
        ...         return await redis.brpop(f"response:{request_id}", timeout=60)
    """
    
    def __init__(
        self,
        queue_name: str,
        *,
        timeout: float = 300.0,  # 5 min default
        poll_interval: float = 1.0,
    ) -> None:
        self.queue_name = queue_name
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._pending: dict[str, EscalationResult | None] = {}
    
    def _request_id(self, request: EscalationRequest) -> str:
        """Generate unique request ID."""
        import hashlib
        data = f"{request.tool_name}:{request.timestamp.isoformat()}:{id(request)}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    async def publish(self, request: EscalationRequest, request_id: str) -> None:
        """Publish request to queue. Override for your system."""
        logger.info(f"[{self.queue_name}] Escalation {request_id}: {request.tool_name}")
        # Default: just log (for testing)
        self._pending[request_id] = None
    
    async def poll(self, request_id: str) -> EscalationResult | None:
        """Poll for response. Override for your system."""
        # Default: check in-memory dict (for testing)
        return self._pending.get(request_id)
    
    def resolve(self, request_id: str, result: EscalationResult) -> None:
        """Manually resolve a pending escalation (for testing)."""
        self._pending[request_id] = result
    
    async def escalate(self, request: EscalationRequest) -> EscalationResult:
        """Submit and poll for resolution."""
        request_id = self._request_id(request)
        
        await self.publish(request, request_id)
        
        elapsed = 0.0
        while elapsed < self.timeout:
            result = await self.poll(request_id)
            if result is not None:
                return result
            
            await checkpoint()
            await asyncio.sleep(self.poll_interval)
            elapsed += self.poll_interval
        
        return EscalationResult(
            status=EscalationStatus.TIMEOUT,
            reason=f"No response within {self.timeout}s",
        )


class CallbackEscalation:
    """Callback-based escalation for sync workflows.
    
    Calls a sync function and blocks until it returns.
    Useful for CLI tools, notebooks, or simple approval flows.
    
    Example:
        >>> def cli_approve(request):
        ...     print(f"Approve {request.tool_name}? [y/n]")
        ...     return input().lower() == "y"
        ...
        >>> escalation = CallbackEscalation(cli_approve)
    """
    
    def __init__(
        self,
        callback: Callable[[EscalationRequest], bool | str | EscalationResult],
    ) -> None:
        self.callback = callback
    
    async def escalate(self, request: EscalationRequest) -> EscalationResult:
        """Call callback and interpret result."""
        result = await to_thread(self.callback, request)
        
        if isinstance(result, EscalationResult):
            return result
        elif isinstance(result, bool):
            return EscalationResult(
                status=EscalationStatus.APPROVED if result else EscalationStatus.REJECTED,
            )
        elif isinstance(result, str):
            return EscalationResult(
                status=EscalationStatus.OVERRIDE,
                value=result,
            )
        else:
            return EscalationResult(status=EscalationStatus.REJECTED, reason="Invalid callback response")


class EscalationParams(BaseModel):
    """Parameters for escalation tool execution."""
    
    input: dict[str, object] = Field(
        default_factory=dict,
        description="Input parameters for the underlying tool",
    )


class EscalationTool(BaseTool[EscalationParams]):
    """Retry with human escalation on failure.
    
    Attempts automated execution up to max_retries, then escalates
    to human review. If human approves, returns approval result.
    
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
        self._tool = tool
        self._handler = handler
        self._max_retries = max_retries
        self._retry_codes = retry_codes or frozenset({
            ErrorCode.RATE_LIMITED,
            ErrorCode.TIMEOUT,
            ErrorCode.NETWORK_ERROR,
        })
        
        derived_name = name or f"escalation_{tool.metadata.name}"
        derived_desc = description or f"Retry {tool.metadata.name} with human escalation"
        
        self._meta = ToolMetadata(
            name=derived_name,
            description=derived_desc,
            category="agents",
            streaming=tool.metadata.streaming,
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
            code = ErrorCode(trace.error_code)
            return code in self._retry_codes
        except ValueError:
            return True
    
    def _run(self, params: EscalationParams) -> str:
        return self._run_async_sync(self._async_run(params))
    
    async def _async_run(self, params: EscalationParams) -> str:
        result = await self._async_run_result(params)
        if result.is_ok():
            return result.unwrap()
        return result.unwrap_err().message
    
    async def _async_run_result(self, params: EscalationParams) -> ToolResult:
        """Execute with retry and escalation."""
        # Build params for underlying tool
        try:
            tool_params = self._tool.params_schema(**params.input)
        except ValidationError as e:
            trace = ErrorTrace(
                message=f"Invalid params for {self._tool.metadata.name}: {e}",
                error_code=ErrorCode.INVALID_PARAMS.value,
                recoverable=False,
            )
            return Err(trace)
        
        # Retry loop
        attempt = 0
        last_error: ErrorTrace | None = None
        
        while True:
            result = await self._tool.arun_result(tool_params)
            
            if result.is_ok():
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
            tool_name=self._tool.metadata.name,
            params=params.input,
            error=last_error or ErrorTrace(message="Unknown error"),
            attempt=attempt,
        )
        
        escalation_result = await self._handler.escalate(request)
        
        if escalation_result.should_proceed:
            # Human approved - return their override or success marker
            value = escalation_result.value or f"Approved by {escalation_result.reviewer or 'human'}"
            return Ok(value)
        
        # Human rejected or timeout
        trace = ErrorTrace(
            message=f"Escalation {escalation_result.status.value}: {escalation_result.reason or 'no reason'}",
            error_code=ErrorCode.PERMISSION_DENIED.value if escalation_result.status == EscalationStatus.REJECTED else ErrorCode.TIMEOUT.value,
            recoverable=False,
        ).with_operation(f"escalation:{self._meta.name}")
        
        return Err(trace)


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
    
    Retries automated execution up to max_retries times, then
    escalates to the specified handler for human review.
    
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
    # Convert string to queue handler
    handler: EscalationHandler
    if isinstance(escalate_to, str):
        handler = QueueEscalation(escalate_to)
    else:
        handler = escalate_to
    
    return EscalationTool(
        tool,
        handler,
        max_retries=max_retries,
        retry_codes=retry_codes,
        name=name,
        description=description,
    )
