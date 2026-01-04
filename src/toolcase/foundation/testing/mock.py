"""Mock utilities for tool testing.

Provides mock_tool context manager and MockTool class for:
- Replacing tool behavior with controlled responses
- Simulating errors and edge cases
- Recording invocations for verification
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

from pydantic import BaseModel

from toolcase.foundation.core import BaseTool
from toolcase.foundation.errors import Err, ErrorCode, ErrorTrace, JsonDict, Ok, ToolResult, classify_exception

if TYPE_CHECKING:
    from collections.abc import Generator

T = TypeVar("T", bound=BaseTool[BaseModel])


@dataclass(slots=True)
class Invocation:
    """Record of a single tool invocation."""
    params: JsonDict
    result: ToolResult
    exception: Exception | None = None


@dataclass
class MockTool(Generic[T]):
    """Mock replacement for a tool with invocation recording."""
    original: type[T] | T
    invocations: list[Invocation] = field(default_factory=list)
    return_value: str | None = None
    raises: type[Exception] | Exception | None = None
    side_effect: Callable[[JsonDict], str] | None = None
    error_code: ErrorCode | None = None
    
    @property
    def call_count(self) -> int:
        return len(self.invocations)
    
    @property
    def called(self) -> bool:
        return self.call_count > 0
    
    @property
    def last_call(self) -> Invocation | None:
        return self.invocations[-1] if self.invocations else None
    
    def assert_called(self) -> None:
        if not self.called:
            raise AssertionError("Expected tool to be called")
    
    def assert_not_called(self) -> None:
        if self.called:
            raise AssertionError(f"Tool called {self.call_count} times")
    
    def assert_called_with(self, **kwargs: object) -> None:
        if not self.called:
            raise AssertionError("Expected tool to be called")
        last = self.last_call
        assert last is not None
        for key, expected in kwargs.items():
            if key not in last.params:
                raise AssertionError(f"Parameter '{key}' not in call")
            if last.params[key] != expected:
                raise AssertionError(
                    f"'{key}': expected {expected!r}, got {last.params[key]!r}"
                )
    
    def _get_tool_name(self) -> str:
        if isinstance(self.original, type):
            meta = getattr(self.original, 'metadata', None)
            return meta.name if meta else 'mock_tool'
        return self.original.metadata.name
    
    def _execute(self, params: JsonDict) -> ToolResult:
        result: ToolResult
        exc: Exception | None = None
        tool_name = self._get_tool_name()
        
        try:
            if self.raises is not None:
                exc_to_raise = self.raises() if isinstance(self.raises, type) else self.raises
                raise exc_to_raise
            
            if self.side_effect is not None:
                result = Ok(self.side_effect(params))
            elif self.return_value is not None:
                result = Ok(self.return_value)
            elif self.error_code is not None:
                trace = ErrorTrace(
                    message=f"Mock error: {self.error_code}",
                    error_code=self.error_code.value,
                    recoverable=True,
                ).with_operation(f"tool:{tool_name}")
                result = Err(trace)
            else:
                result = Ok("mock result")
        except Exception as e:
            exc = e
            code = classify_exception(e)
            trace = ErrorTrace(
                message=str(e),
                error_code=code.value,
                recoverable=True,
            ).with_operation(f"tool:{tool_name}")
            result = Err(trace)
        
        self.invocations.append(Invocation(params=params, result=result, exception=exc))
        return result


_active_mocks: dict[str, MockTool[BaseTool[BaseModel]]] = {}


@contextmanager
def mock_tool(
    tool: type[T] | T,
    *,
    return_value: str | None = None,
    raises: type[Exception] | Exception | None = None,
    side_effect: Callable[[JsonDict], str] | None = None,
    error_code: ErrorCode | None = None,
) -> Generator[MockTool[T], None, None]:
    """Context manager for mocking tool behavior.
    
    Replaces tool execution with controlled responses for testing.
    Records all invocations for verification.
    """
    mock = MockTool(
        original=tool,
        return_value=return_value,
        raises=raises,
        side_effect=side_effect,
        error_code=error_code,
    )
    
    tool_name = mock._get_tool_name()
    tool_cls = type(tool) if not isinstance(tool, type) else tool
    
    # Store original methods
    orig_run = tool_cls._run_result
    orig_async = tool_cls._async_run_result
    
    def patched_run(self: BaseTool[BaseModel], params: BaseModel) -> ToolResult:
        m = _active_mocks.get(self.metadata.name)
        return m._execute(params.model_dump()) if m else orig_run(self, params)
    
    async def patched_async(self: BaseTool[BaseModel], params: BaseModel) -> ToolResult:
        m = _active_mocks.get(self.metadata.name)
        return m._execute(params.model_dump()) if m else await orig_async(self, params)
    
    tool_cls._run_result = patched_run  # type: ignore[method-assign]
    tool_cls._async_run_result = patched_async  # type: ignore[method-assign]
    _active_mocks[tool_name] = mock  # type: ignore[assignment]
    
    try:
        yield mock
    finally:
        _active_mocks.pop(tool_name, None)
        tool_cls._run_result = orig_run  # type: ignore[method-assign]
        tool_cls._async_run_result = orig_async  # type: ignore[method-assign]
