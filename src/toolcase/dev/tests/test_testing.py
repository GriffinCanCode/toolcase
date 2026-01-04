"""Tests for the testing utilities module."""

import pytest

from toolcase import ErrorCode, tool
from toolcase.foundation.testing import (
    Invocation,
    MockAPI,
    MockResponse,
    MockTool,
    ToolTestCase,
    fixture,
    mock_tool,
)


# ═════════════════════════════════════════════════════════════════════════════
# Sample Tools for Testing
# ═════════════════════════════════════════════════════════════════════════════


def make_search_tool():
    """Factory to create fresh search tool for each test."""
    @tool(description="Search for information in a database")
    def search_tool(query: str, limit: int = 5) -> str:
        return f"Results for '{query}' (limit={limit})"
    return search_tool


def make_async_fetch_tool():
    """Factory to create fresh async fetch tool for each test."""
    @tool(description="Fetch user data from API")
    async def async_fetch_tool(user_id: str) -> str:
        return f"User data for {user_id}"
    return async_fetch_tool


# Module-level tools for simple tests
search_tool = make_search_tool()
async_fetch_tool = make_async_fetch_tool()


# ═════════════════════════════════════════════════════════════════════════════
# ToolTestCase Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestToolTestCase(ToolTestCase):
    """Test the ToolTestCase base class."""
    
    async def test_invoke_returns_ok_result(self):
        """Test that invoke returns Ok result for successful execution."""
        result = await self.invoke(search_tool, query="python")
        self.assert_ok(result)
    
    async def test_assert_ok_returns_value(self):
        """Test assert_ok returns unwrapped value."""
        result = await self.invoke(search_tool, query="test")
        value = self.assert_ok(result)
        assert "test" in value
    
    async def test_assert_contains(self):
        """Test assert_contains validates substring presence."""
        result = await self.invoke(search_tool, query="python")
        self.assert_contains(result, "python")
        self.assert_contains(result, "Results")
    
    async def test_assert_not_contains(self):
        """Test assert_not_contains validates substring absence."""
        result = await self.invoke(search_tool, query="python")
        self.assert_not_contains(result, "javascript")
    
    async def test_assert_result_equals(self):
        """Test exact result equality."""
        result = await self.invoke(search_tool, query="test", limit=10)
        self.assert_result_equals(result, "Results for 'test' (limit=10)")
    
    async def test_invoke_sync(self):
        """Test synchronous invocation."""
        result = self.invoke_sync(search_tool, query="sync_test")
        self.assert_ok(result)
        self.assert_contains(result, "sync_test")
    
    async def test_async_tool_invoke(self):
        """Test invoking async tools."""
        result = await self.invoke(async_fetch_tool, user_id="123")
        self.assert_ok(result)
        self.assert_contains(result, "123")


# ═════════════════════════════════════════════════════════════════════════════
# mock_tool Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestMockTool(ToolTestCase):
    """Test mock_tool context manager."""
    
    async def test_mock_with_return_value(self):
        """Test mocking with fixed return value."""
        test_tool = make_search_tool()
        with mock_tool(test_tool, return_value="mocked result") as mock:
            result = await self.invoke(test_tool, query="test")
            self.assert_ok(result)
            self.assert_result_equals(result, "mocked result")
            mock.assert_called()
    
    async def test_mock_records_invocations(self):
        """Test that mock records invocations."""
        test_tool = make_search_tool()
        with mock_tool(test_tool, return_value="ok") as mock:
            await self.invoke(test_tool, query="first")
            await self.invoke(test_tool, query="second")
            
            assert mock.call_count == 2
            assert mock.called
            assert mock.last_call is not None
            assert mock.last_call.params["query"] == "second"
    
    async def test_mock_assert_called_with(self):
        """Test assert_called_with verification."""
        test_tool = make_search_tool()
        with mock_tool(test_tool, return_value="ok") as mock:
            await self.invoke(test_tool, query="test", limit=10)
            mock.assert_called_with(query="test", limit=10)
    
    async def test_mock_with_raises(self):
        """Test mocking with exception."""
        test_tool = make_search_tool()
        with mock_tool(test_tool, raises=TimeoutError("timed out")):
            result = await self.invoke(test_tool, query="test")
            self.assert_err(result, code=ErrorCode.TIMEOUT)
    
    async def test_mock_with_error_code(self):
        """Test mocking with error code."""
        test_tool = make_search_tool()
        with mock_tool(test_tool, error_code=ErrorCode.RATE_LIMITED):
            result = await self.invoke(test_tool, query="test")
            self.assert_err(result, code=ErrorCode.RATE_LIMITED)
    
    async def test_mock_with_side_effect(self):
        """Test mocking with side effect callable."""
        test_tool = make_search_tool()
        
        def handler(params: dict) -> str:
            return f"Custom: {params['query'].upper()}"
        
        with mock_tool(test_tool, side_effect=handler):
            result = await self.invoke(test_tool, query="test")
            self.assert_ok(result)
            self.assert_result_equals(result, "Custom: TEST")
    
    async def test_mock_assert_not_called(self):
        """Test assert_not_called verification."""
        test_tool = make_search_tool()
        with mock_tool(test_tool, return_value="ok") as mock:
            mock.assert_not_called()
            
            await self.invoke(test_tool, query="test")
            
            with pytest.raises(AssertionError):
                mock.assert_not_called()


# ═════════════════════════════════════════════════════════════════════════════
# MockAPI Tests
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_mock_api_get():
    """Test MockAPI GET requests."""
    api = MockAPI(responses={
        "search": "search results",
        "users/123": {"id": "123", "name": "Test"},
    })
    
    response = await api.get("search", query="python")
    assert response.ok
    assert response.text == "search results"
    assert api.request_count == 1


@pytest.mark.asyncio
async def test_mock_api_json_response():
    """Test MockAPI JSON responses."""
    api = MockAPI(responses={
        "users/123": {"id": "123", "name": "Test User"},
    })
    
    response = await api.get("users/123")
    assert response.ok
    assert response.json() == {"id": "123", "name": "Test User"}


@pytest.mark.asyncio
async def test_mock_api_post():
    """Test MockAPI POST requests."""
    api = MockAPI(responses={"users": "created"})
    
    response = await api.post("users", data={"name": "New User"})
    assert response.ok
    assert api.last_request is not None
    assert api.last_request["method"] == "POST"
    assert api.last_request["data"] == {"name": "New User"}


@pytest.mark.asyncio
async def test_mock_api_error():
    """Test MockAPI error responses."""
    api = MockAPI()
    api.set_error("fail", status=500, message="Server error")
    
    response = await api.get("fail")
    assert not response.ok
    assert response.status == 500


@pytest.mark.asyncio
async def test_mock_api_pattern_matching():
    """Test MockAPI wildcard pattern matching."""
    api = MockAPI(responses={
        "users/*": {"status": "found"},
    })
    
    response = await api.get("users/123")
    assert response.ok
    assert response.json() == {"status": "found"}


@pytest.mark.asyncio
async def test_mock_api_assert_endpoint_called():
    """Test endpoint call assertions."""
    api = MockAPI()
    await api.get("users/123")
    
    api.assert_called()
    api.assert_endpoint_called("users/123")
    
    with pytest.raises(AssertionError):
        api.assert_endpoint_called("users/456")


# ═════════════════════════════════════════════════════════════════════════════
# Fixture Decorator Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_fixture_decorator_integration():
    """Test @fixture decorator integrates with pytest.
    
    When pytest is available, @fixture becomes a pytest.fixture.
    The fixture decorator works, but calling fixtures directly is deprecated.
    """
    # Test that fixture decorator doesn't break the function
    @fixture
    def my_fixture() -> str:
        return "fixture value"
    
    # In pytest, fixtures are accessed via dependency injection, not direct calls
    # Just verify the decorator applied without error
    assert callable(my_fixture)


def test_fixture_with_scope():
    """Test @fixture with scope parameter."""
    @fixture(scope="module")
    def module_fixture() -> int:
        return 42
    
    # Verify decorator applied correctly
    assert callable(module_fixture)


# ═════════════════════════════════════════════════════════════════════════════
# MockResponse Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_mock_response_ok():
    """Test MockResponse ok property."""
    assert MockResponse(status=200).ok
    assert MockResponse(status=201).ok
    assert not MockResponse(status=400).ok
    assert not MockResponse(status=500).ok


def test_mock_response_text():
    """Test MockResponse text property."""
    assert MockResponse(data="hello").text == "hello"
    assert MockResponse(data={"key": "value"}).text == '{"key":"value"}'  # orjson compact format
    assert MockResponse(data=None).text == ""


def test_mock_response_json():
    """Test MockResponse json method."""
    response = MockResponse(data={"key": "value"})
    assert response.json() == {"key": "value"}
    
    with pytest.raises(ValueError):
        MockResponse(data="not json").json()


# ═════════════════════════════════════════════════════════════════════════════
# Invocation Record Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_invocation_dataclass():
    """Test Invocation dataclass."""
    from toolcase.foundation.errors import Ok
    
    invocation = Invocation(
        params={"query": "test"},
        result=Ok("result"),
        exception=None,
    )
    
    assert invocation.params == {"query": "test"}
    assert invocation.result.is_ok()
    assert invocation.exception is None
