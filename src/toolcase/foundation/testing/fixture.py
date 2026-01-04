"""Test fixtures for common tool testing scenarios.

Provides:
- @fixture decorator for pytest fixture integration
- MockAPI for simulating external API responses
- Pre-built fixtures for common patterns
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Callable, ParamSpec, TypeVar, overload

from toolcase.foundation.errors import JsonDict, JsonValue

if TYPE_CHECKING:
    from collections.abc import Awaitable

T = TypeVar("T")
P = ParamSpec("P")


# ═════════════════════════════════════════════════════════════════════════════
# Fixture Decorator
# ═════════════════════════════════════════════════════════════════════════════


@overload
def fixture(func: Callable[P, T]) -> Callable[P, T]: ...
@overload
def fixture(*, scope: str = "function", autouse: bool = False) -> Callable[[Callable[P, T]], Callable[P, T]]: ...

def fixture(
    func: Callable[P, T] | None = None,
    *,
    scope: str = "function",
    autouse: bool = False,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator marking a function as a test fixture.
    
    Integrates with pytest when available, otherwise provides standalone behavior.
    Supports both sync and async fixtures.
    
    Args:
        func: Fixture function (when used without parentheses)
        scope: Fixture scope ("function", "class", "module", "session")
        autouse: Whether to automatically use this fixture
    
    Returns:
        Decorated fixture function
    
    Example:
        >>> @fixture
        ... def mock_api() -> MockAPI:
        ...     return MockAPI(responses={"search": "mocked results"})
        
        >>> @fixture(scope="module")
        ... async def db_connection() -> AsyncIterator[Connection]:
        ...     conn = await create_connection()
        ...     yield conn
        ...     await conn.close()
    """
    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        # Try to use pytest's fixture decorator if available
        try:
            import pytest
            return pytest.fixture(scope=scope, autouse=autouse)(fn)  # type: ignore[return-value]
        except ImportError:
            pass
        
        # Fallback: just mark and return function
        fn._fixture_scope = scope  # type: ignore[attr-defined]
        fn._fixture_autouse = autouse  # type: ignore[attr-defined]
        return wraps(fn)(lambda *a, **kw: fn(*a, **kw))  # type: ignore[return-value]
    
    return decorator(func) if func else decorator


# ═════════════════════════════════════════════════════════════════════════════
# MockAPI - Simulated API Responses
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class MockResponse:
    """Simulated HTTP response."""
    status: int = 200
    data: str | JsonDict | None = None
    headers: dict[str, str] = field(default_factory=dict)
    delay_ms: float = 0
    
    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300
    
    def json(self) -> JsonDict:
        """Return data as dict (mimics httpx/requests API)."""
        if isinstance(self.data, dict):
            return self.data
        raise ValueError("Response data is not JSON")
    
    @property
    def text(self) -> str:
        """Return data as string."""
        return self.data if isinstance(self.data, str) else json.dumps(self.data) if self.data else ""


@dataclass
class MockAPI:
    """Simulated API backend for testing external integrations.
    
    Provides configurable responses for endpoints, including:
    - Success responses with custom data
    - Error responses with status codes
    - Simulated delays
    - Request recording for verification
    
    Example:
        >>> api = MockAPI(responses={
        ...     "search": "search results",
        ...     "users/123": {"id": "123", "name": "Test"},
        ... })
        >>> 
        >>> response = await api.get("search", query="python")
        >>> assert response.text == "search results"
        >>> assert api.request_count == 1
    """
    
    responses: dict[str, str | JsonDict | MockResponse] = field(default_factory=dict)
    default_response: str = "mock response"
    default_status: int = 200
    requests: list[JsonDict] = field(default_factory=list)
    
    @property
    def request_count(self) -> int:
        """Number of requests made."""
        return len(self.requests)
    
    @property
    def last_request(self) -> JsonDict | None:
        """Most recent request, if any."""
        return self.requests[-1] if self.requests else None
    
    def _record(self, method: str, endpoint: str, **kwargs: JsonValue) -> None:
        """Record a request."""
        self.requests.append({"method": method, "endpoint": endpoint, **kwargs})
    
    def _wrap_response(self, resp: str | JsonDict | MockResponse) -> MockResponse:
        """Wrap raw response in MockResponse if needed."""
        return resp if isinstance(resp, MockResponse) else MockResponse(status=self.default_status, data=resp)
    
    def _get_response(self, endpoint: str) -> MockResponse:
        """Get response for endpoint."""
        if endpoint in self.responses:
            return self._wrap_response(self.responses[endpoint])
        for pat, resp in self.responses.items():
            if '*' in pat and endpoint.startswith(pat.split('*')[0]):
                return self._wrap_response(resp)
        return MockResponse(status=self.default_status, data=self.default_response)
    
    async def _request(self, method: str, endpoint: str, **kwargs: JsonValue) -> MockResponse:
        """Execute a simulated request with recording and delay."""
        self._record(method, endpoint, **kwargs)
        resp = self._get_response(endpoint)
        if resp.delay_ms > 0:
            import asyncio
            await asyncio.sleep(resp.delay_ms / 1000)
        return resp
    
    async def get(self, endpoint: str, **params: JsonValue) -> MockResponse:
        """Simulate GET request."""
        return await self._request("GET", endpoint, params=dict(params))
    
    async def post(self, endpoint: str, data: JsonDict | None = None, **kwargs: JsonValue) -> MockResponse:
        """Simulate POST request."""
        return await self._request("POST", endpoint, data=data, **kwargs)
    
    async def put(self, endpoint: str, data: JsonDict | None = None, **kwargs: JsonValue) -> MockResponse:
        """Simulate PUT request."""
        return await self._request("PUT", endpoint, data=data, **kwargs)
    
    async def delete(self, endpoint: str, **kwargs: JsonValue) -> MockResponse:
        """Simulate DELETE request."""
        return await self._request("DELETE", endpoint, **kwargs)
    
    def clear(self) -> None:
        """Clear recorded requests."""
        self.requests.clear()
    
    def set_response(self, endpoint: str, response: str | JsonDict | MockResponse) -> None:
        """Set response for endpoint."""
        self.responses[endpoint] = response
    
    def set_error(self, endpoint: str, status: int = 500, message: str = "Internal Server Error") -> None:
        """Configure endpoint to return error."""
        self.responses[endpoint] = MockResponse(status=status, data={"error": message})
    
    def assert_called(self) -> None:
        """Assert API was called at least once."""
        if not self.requests:
            raise AssertionError("Expected API to be called")
    
    def assert_endpoint_called(self, endpoint: str) -> None:
        """Assert specific endpoint was called."""
        if not any(req["endpoint"] == endpoint for req in self.requests):
            raise AssertionError(f"Endpoint '{endpoint}' was not called")


# ═════════════════════════════════════════════════════════════════════════════
# Pre-built Fixtures
# ═════════════════════════════════════════════════════════════════════════════


@fixture
def mock_api() -> MockAPI:
    """Default MockAPI fixture."""
    return MockAPI()


@fixture
def mock_api_with_errors() -> MockAPI:
    """MockAPI configured to return errors."""
    api = MockAPI(default_status=500)
    api.set_error("*", status=500, message="Service unavailable")
    return api


@fixture
def mock_api_slow() -> MockAPI:
    """MockAPI with simulated latency."""
    return MockAPI(
        responses={
            "*": MockResponse(data="slow response", delay_ms=100),
        }
    )
