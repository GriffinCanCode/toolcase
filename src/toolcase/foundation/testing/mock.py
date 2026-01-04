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
from toolcase.foundation.errors import ErrorCode, JsonDict, Ok, ToolResult, classify_exception, tool_err

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
        return bool(self.invocations)
    
    @property
    def last_call(self) -> Invocation | None:
        return self.invocations[-1] if self.invocations else None
    
    def assert_called(self) -> None:
        if not self.invocations:
            raise AssertionError("Expected tool to be called")
    
    def assert_not_called(self) -> None:
        if self.invocations:
            raise AssertionError(f"Tool called {len(self.invocations)} times")
    
    def assert_called_with(self, **kwargs: object) -> None:
        if not (last := self.last_call):
            raise AssertionError("Expected tool to be called")
        for k, v in kwargs.items():
            if k not in last.params:
                raise AssertionError(f"Parameter '{k}' not in call")
            if last.params[k] != v:
                raise AssertionError(f"'{k}': expected {v!r}, got {last.params[k]!r}")
    
    def _get_tool_name(self) -> str:
        if isinstance(self.original, type):
            return meta.name if (meta := getattr(self.original, 'metadata', None)) else 'mock_tool'
        return self.original.metadata.name
    
    def _make_err(self, message: str, code: ErrorCode) -> ToolResult:
        """Create error result with trace."""
        return tool_err(self._get_tool_name(), message, code, recoverable=True)
    
    def _execute(self, params: JsonDict) -> ToolResult:
        exc: Exception | None = None
        try:
            if self.raises is not None:
                raise self.raises() if isinstance(self.raises, type) else self.raises
            result = (
                Ok(self.side_effect(params)) if self.side_effect is not None
                else Ok(self.return_value) if self.return_value is not None
                else self._make_err(f"Mock error: {self.error_code}", self.error_code) if self.error_code is not None
                else Ok("mock result")
            )
        except Exception as e:
            exc, result = e, self._make_err(str(e), classify_exception(e))
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
    """Context manager for mocking tool behavior. Replaces tool execution with controlled responses for testing and records all invocations for verification."""
    mock: MockTool[T] = MockTool(original=tool, return_value=return_value, raises=raises, side_effect=side_effect, error_code=error_code)
    tool_cls, name = (tool if isinstance(tool, type) else type(tool)), mock._get_tool_name()
    orig_run, orig_async = tool_cls._run_result, tool_cls._async_run_result
    
    async def patched_async(s: BaseTool[BaseModel], p: BaseModel) -> ToolResult:
        return m._execute(p.model_dump()) if (m := _active_mocks.get(s.metadata.name)) else await orig_async(s, p)
    
    tool_cls._run_result = lambda s, p: m._execute(p.model_dump()) if (m := _active_mocks.get(s.metadata.name)) else orig_run(s, p)  # type: ignore[method-assign]
    tool_cls._async_run_result = patched_async  # type: ignore[method-assign]
    _active_mocks[name] = mock  # type: ignore[assignment]
    
    try:
        yield mock
    finally:
        _active_mocks.pop(name, None)
        tool_cls._run_result, tool_cls._async_run_result = orig_run, orig_async  # type: ignore[method-assign]
