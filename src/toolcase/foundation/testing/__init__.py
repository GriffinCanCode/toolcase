"""First-class testing utilities for toolcase tools.

Provides batteries-included testing infrastructure:
- ToolTestCase: Async test case with Result-based assertions
- mock_tool: Context manager for mocking tool behavior
- fixture: Decorator for pytest fixture integration
- MockAPI: Simulated API backend for testing integrations

Quick Start:
    >>> from toolcase.testing import ToolTestCase, mock_tool, fixture, MockAPI
    >>> 
    >>> class TestSearchTool(ToolTestCase):
    ...     async def test_search_returns_results(self):
    ...         result = await self.invoke(SearchTool(), query="python")
    ...         self.assert_ok(result)
    ...         self.assert_contains(result, "python")
    ...
    ...     async def test_handles_timeout(self):
    ...         with mock_tool(SearchTool, raises=TimeoutError):
    ...             result = await self.invoke(SearchTool(), query="test")
    ...             self.assert_err(result, code=ErrorCode.TIMEOUT)
    
    >>> @fixture
    ... def api() -> MockAPI:
    ...     return MockAPI(responses={"search": "mocked results"})
"""

from .case import ToolTestCase
from .fixture import (
    MockAPI,
    MockResponse,
    fixture,
    mock_api,
    mock_api_slow,
    mock_api_with_errors,
)
from .mock import Invocation, MockTool, mock_tool

__all__ = [
    # Test case
    "ToolTestCase",
    # Mocking
    "mock_tool",
    "MockTool",
    "Invocation",
    # Fixtures
    "fixture",
    "MockAPI",
    "MockResponse",
    "mock_api",
    "mock_api_with_errors",
    "mock_api_slow",
]
