"""Test case base class for tool testing with Result-based assertions.

Provides ToolTestCase for async-first testing with monadic Result support:
- assert_ok / assert_err for Result variant checking
- assert_contains for content validation
- invoke helper for type-safe tool execution
"""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, TypeVar, overload

from toolcase.io.cache import reset_cache
from toolcase.foundation.errors import ErrorCode, ErrorTrace, Result

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool
    from pydantic import BaseModel

T = TypeVar("T")


class ToolTestCase(unittest.IsolatedAsyncioTestCase):
    """Base class for testing tools with Result-based assertions.
    
    Extends IsolatedAsyncioTestCase for native async test support.
    Provides assertion helpers that work with ToolResult monads.
    Automatically resets cache between tests to ensure test isolation.
    
    Example:
        >>> class TestSearchTool(ToolTestCase):
        ...     async def test_search_returns_results(self):
        ...         result = await self.invoke(SearchTool(), query="python")
        ...         self.assert_ok(result)
        ...         self.assert_contains(result, "python")
        ...
        ...     async def test_handles_invalid_query(self):
        ...         result = await self.invoke(SearchTool(), query="")
        ...         self.assert_err(result, code=ErrorCode.INVALID_PARAMS)
    """
    
    def setUp(self) -> None:
        """Reset cache before each test for isolation."""
        super().setUp(); reset_cache()
    
    # ─────────────────────────────────────────────────────────────────
    # Invocation Helpers
    # ─────────────────────────────────────────────────────────────────
    
    async def invoke(self, tool: BaseTool[BaseModel], **kwargs: object) -> Result[str, ErrorTrace]:
        """Execute tool and return Result type.
        
        Wraps tool execution in Result for type-safe error handling.
        Works with both sync and async tools.
        
        Args:
            tool: Tool instance to execute
            **kwargs: Parameters to pass to tool
        
        Returns:
            ToolResult with success string or ErrorTrace
        """
        return await tool.arun_result(tool.params_schema(**kwargs))  # type: ignore[call-arg, arg-type]
    
    def invoke_sync(self, tool: BaseTool[BaseModel], **kwargs: object) -> Result[str, ErrorTrace]:
        """Execute tool synchronously. For tools that must run synchronously in tests."""
        return tool.run_result(tool.params_schema(**kwargs))  # type: ignore[call-arg, arg-type]
    
    # ─────────────────────────────────────────────────────────────────
    # Result Assertions
    # ─────────────────────────────────────────────────────────────────
    
    def assert_ok(self, result: Result[T, ErrorTrace], msg: str | None = None) -> T:
        """Assert Result is Ok variant and return value.
        
        Args:
            result: Result to check
            msg: Optional message on failure
        
        Returns:
            Unwrapped Ok value for further assertions
        
        Raises:
            AssertionError: If Result is Err
        """
        if result.is_ok():
            return result.unwrap()
        t = result.unwrap_err()
        base = msg or f"Expected Ok, got Err: {t.message}"
        self.fail(f"{base}\nDetails: {t.details}" if t.details else base)
    
    @overload
    def assert_err(self, result: Result[T, ErrorTrace], *, code: ErrorCode, msg: str | None = None) -> ErrorTrace: ...
    @overload
    def assert_err(self, result: Result[T, ErrorTrace], *, contains: str, msg: str | None = None) -> ErrorTrace: ...
    @overload
    def assert_err(self, result: Result[T, ErrorTrace], *, msg: str | None = None) -> ErrorTrace: ...
    
    def assert_err(
        self,
        result: Result[T, ErrorTrace],
        *,
        code: ErrorCode | None = None,
        contains: str | None = None,
        msg: str | None = None,
    ) -> ErrorTrace:
        """Assert Result is Err variant with optional checks.
        
        Args:
            result: Result to check
            code: Expected error code (optional)
            contains: String that error message should contain (optional)
            msg: Custom failure message (optional)
        
        Returns:
            ErrorTrace for further inspection
        
        Raises:
            AssertionError: If Result is Ok or error doesn't match criteria
        """
        if result.is_ok():
            self.fail(msg or f"Expected Err, got Ok: {result.unwrap()}")
        t = result.unwrap_err()
        if code and t.error_code != code.value:
            self.fail(msg or f"Expected error code {code}, got {t.error_code}")
        if contains and contains not in t.message:
            self.fail(msg or f"Expected error message to contain '{contains}', got: {t.message}")
        return t
    
    def assert_contains(self, result: Result[str, ErrorTrace], substring: str, msg: str | None = None) -> None:
        """Assert Ok result contains substring.
        
        Args:
            result: Result to check (must be Ok)
            substring: Text that should be present
            msg: Custom failure message
        
        Raises:
            AssertionError: If Result is Err or doesn't contain substring
        """
        if substring not in (value := self.assert_ok(result)):
            self.fail(msg or f"Expected result to contain '{substring}', got: {value[:200]}{'...' if len(value) > 200 else ''}")
    
    def assert_not_contains(self, result: Result[str, ErrorTrace], substring: str, msg: str | None = None) -> None:
        """Assert Ok result does NOT contain substring."""
        if substring in self.assert_ok(result):
            self.fail(msg or f"Result should not contain '{substring}'")
    
    # ─────────────────────────────────────────────────────────────────
    # Value Assertions
    # ─────────────────────────────────────────────────────────────────
    
    def assert_result_equals(self, result: Result[T, ErrorTrace], expected: T, msg: str | None = None) -> None:
        """Assert Ok result equals expected value."""
        self.assertEqual(self.assert_ok(result), expected, msg)
    
    def assert_recoverable(self, result: Result[T, ErrorTrace], msg: str | None = None) -> None:
        """Assert error is marked as recoverable."""
        if not self.assert_err(result).recoverable:
            self.fail(msg or "Expected recoverable error")
    
    def assert_not_recoverable(self, result: Result[T, ErrorTrace], msg: str | None = None) -> None:
        """Assert error is NOT recoverable."""
        if self.assert_err(result).recoverable:
            self.fail(msg or "Expected non-recoverable error")
