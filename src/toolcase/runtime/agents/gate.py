"""Gate primitive for pre/post condition checks.

Guards tool execution with validation. Useful for:
- Authorization checks before sensitive operations
- Input validation beyond schema
- Output sanitization/validation
- Confirmation requirements
- Rate limiting at the semantic level

Example:
    >>> safe_delete = gate(
    ...     DeleteTool(),
    ...     pre=lambda p: p.get("confirmed") == True,
    ...     on_block="Deletion requires confirmation",
    ... )
"""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from toolcase.foundation.core.base import BaseTool, ToolMetadata
from toolcase.foundation.errors import Err, ErrorCode, ErrorTrace, JsonDict, Ok, Result, ToolResult, format_validation_error


# Type aliases for gate functions
PreCheck = Callable[[JsonDict], bool | str | ToolResult]
PostCheck = Callable[[str], bool | str | ToolResult]
ParamsTransform = Callable[[JsonDict], JsonDict]


class GateParams(BaseModel):
    """Parameters for gate execution."""
    input: JsonDict = Field(default_factory=dict, description="Input parameters to validate and pass through")


GateParams.model_rebuild()  # Resolve recursive JsonValue type


class GateTool(BaseTool[GateParams]):
    """Pre/post condition gate. Validates inputs before execution and/or outputs after.
    
    Gate functions can return: True/False (pass/block), str (block msg), or ToolResult (full control).
    
    Example:
        >>> gated = GateTool(
        ...     tool=AdminTool(),
        ...     pre_check=lambda p: p.get("is_admin") == True,
        ...     post_check=lambda r: "password" not in r.lower(),
        ...     on_block="Admin access required",
        ... )
    """
    
    __slots__ = ("_tool", "_pre", "_post", "_transform", "_block_msg", "_meta")
    params_schema = GateParams
    cache_enabled = False
    
    def __init__(
        self,
        tool: BaseTool[BaseModel],
        *,
        pre_check: PreCheck | None = None,
        post_check: PostCheck | None = None,
        transform: ParamsTransform | None = None,
        on_block: str = "Gate check failed",
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self._tool, self._pre, self._post = tool, pre_check, post_check
        self._transform, self._block_msg = transform, on_block
        self._meta = ToolMetadata(
            name=name or f"gate_{tool.metadata.name}",
            description=description or f"Gated: {tool.metadata.description}",
            category="agents", streaming=tool.metadata.streaming,
        )
    
    @property
    def metadata(self) -> ToolMetadata:
        return self._meta
    
    @property
    def tool(self) -> BaseTool[BaseModel]:
        return self._tool
    
    def _check_result(self, check_output: bool | str | ToolResult, phase: str) -> ToolResult | None:
        """Interpret check function output. Returns None to proceed, Err to block."""
        match check_output:
            case True: return None
            case False:
                return Err(ErrorTrace(message=self._block_msg, error_code=ErrorCode.PERMISSION_DENIED.value, recoverable=False).with_operation(f"gate:{self._meta.name}", phase=phase))
            case str() as msg:
                return Err(ErrorTrace(message=msg, error_code=ErrorCode.PERMISSION_DENIED.value, recoverable=False).with_operation(f"gate:{self._meta.name}", phase=phase))
            case Result() as r if r.is_err(): return r
            case _: return None
    
    def _run(self, params: GateParams) -> str:
        return self._run_async_sync(self._async_run(params))
    
    async def _async_run(self, params: GateParams) -> str:
        r = await self._async_run_result(params)
        return r.unwrap() if r.is_ok() else r.unwrap_err().message
    
    async def _async_run_result(self, params: GateParams) -> ToolResult:
        """Execute with pre/post gate checks."""
        input_dict = params.input
        
        # Pre-check
        if self._pre:
            try:
                pre_result = self._pre(input_dict)
            except Exception as e:
                return Err(ErrorTrace(
                    message=f"Pre-check failed: {e}", error_code=ErrorCode.INVALID_PARAMS.value, recoverable=False,
                ).with_operation(f"gate:{self._meta.name}", phase="pre"))
            if (blocked := self._check_result(pre_result, "pre")) is not None:
                return blocked
        
        # Transform params if specified
        if self._transform:
            try:
                input_dict = self._transform(input_dict)
            except Exception as e:
                return Err(ErrorTrace(
                    message=f"Transform failed: {e}", error_code=ErrorCode.PARSE_ERROR.value, recoverable=False,
                ).with_operation(f"gate:{self._meta.name}", phase="transform"))
        
        # Build params for underlying tool
        try:
            tool_params = self._tool.params_schema(**input_dict)
        except ValidationError as e:
            return Err(ErrorTrace(
                message=format_validation_error(e, tool_name=self._tool.metadata.name),
                error_code=ErrorCode.INVALID_PARAMS.value, recoverable=False,
            ))
        
        # Execute tool
        if (result := await self._tool.arun_result(tool_params)).is_err():
            return result
        
        output = result.unwrap()
        
        # Post-check
        if self._post:
            try:
                post_result = self._post(output)
            except Exception as e:
                return Err(ErrorTrace(
                    message=f"Post-check failed: {e}", error_code=ErrorCode.INVALID_PARAMS.value, recoverable=False,
                ).with_operation(f"gate:{self._meta.name}", phase="post"))
            
            if (blocked := self._check_result(post_result, "post")) is not None:
                return blocked
            
            # Post-check can transform output (if it returned Result with value)
            if isinstance(post_result, Result) and post_result.is_ok():
                output = post_result.unwrap()
        
        return Ok(output)


def gate(
    tool: BaseTool[BaseModel],
    *,
    pre: PreCheck | None = None,
    post: PostCheck | None = None,
    transform: ParamsTransform | None = None,
    on_block: str = "Gate check failed",
    name: str | None = None,
    description: str | None = None,
) -> GateTool:
    """Create a gated tool with pre/post checks.
    
    Gate functions can return:
    - True: Pass through
    - False: Block with default message
    - str: Block with custom message
    - ToolResult: Full control (Err blocks, Ok passes)
    
    Args:
        tool: Underlying tool to wrap
        pre: Pre-execution check (receives input dict)
        post: Post-execution check (receives output string)
        transform: Optional params transformation
        on_block: Default message when blocked
        name: Optional tool name
        description: Optional description
    
    Returns:
        GateTool instance
    
    Example:
        >>> # Simple confirmation gate
        >>> safe_delete = gate(
        ...     DeleteTool(),
        ...     pre=lambda p: p.get("confirmed") == True,
        ...     on_block="Deletion requires confirmation=True",
        ... )
        >>>
        >>> # Authorization gate
        >>> admin_tool = gate(
        ...     AdminTool(),
        ...     pre=lambda p: check_admin_role(p.get("user_id")),
        ... )
        >>>
        >>> # Output sanitization
        >>> safe_query = gate(
        ...     DatabaseTool(),
        ...     post=lambda r: sanitize_pii(r),  # Returns sanitized string
        ... )
        >>>
        >>> # Combined with transform
        >>> normalized = gate(
        ...     SearchTool(),
        ...     transform=lambda p: {**p, "query": p.get("query", "").lower()},
        ...     post=lambda r: len(r) < 10000,  # Limit response size
        ... )
    """
    return GateTool(tool, pre_check=pre, post_check=post, transform=transform, on_block=on_block, name=name, description=description)
