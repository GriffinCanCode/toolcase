"""Test fixtures for common tool testing scenarios.

Provides:
- @fixture decorator for pytest fixture integration
- MockAPI for simulating external API responses
- Pre-built fixtures for common patterns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    ParamSpec,
    TypeVar,
    overload,
)

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
def fixture(
    *,
    scope: str = "function",
    autouse: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


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
        
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return fn(*args, **kwargs)
        
        return wrapper  # type: ignore[return-value]
    
    if func is not None:
        return decorator(func)
    return decorator


# ═════════════════════════════════════════════════════════════════════════════
# MockAPI - Simulated API Responses
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class MockResponse:
    """Simulated HTTP response."""
    status: int = 200
    data: str | dict[str, object] | None = None
    headers: dict[str, str] = field(default_factory=dict)
    delay_ms: float = 0
    
    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300
    
    def json(self) -> dict[str, object]:
        """Return data as dict (mimics httpx/requests API)."""
        if isinstance(self.data, dict):
            return self.data
        raise ValueError("Response data is not JSON")
    
    @property
    def text(self) -> str:
        """Return data as string."""
        if isinstance(self.data, str):
            return self.data
        if self.data is not None:
            import json
            return json.dumps(self.data)
        return ""


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
    
    responses: dict[str, str | dict[str, object] | MockResponse] = field(
        default_factory=dict
    )
    default_response: str = "mock response"
    default_status: int = 200
    requests: list[dict[str, object]] = field(default_factory=list)
    
    @property
    def request_count(self) -> int:
        """Number of requests made."""
        return len(self.requests)
    
    @property
    def last_request(self) -> dict[str, object] | None:
        """Most recent request, if any."""
        return self.requests[-1] if self.requests else None
    
    def _record(self, method: str, endpoint: str, **kwargs: object) -> None:
        """Record a request."""
        self.requests.append({
            "method": method,
            "endpoint": endpoint,
            **kwargs,
        })
    
    def _get_response(self, endpoint: str) -> MockResponse:
        """Get response for endpoint."""
        if endpoint in self.responses:
            resp = self.responses[endpoint]
            if isinstance(resp, MockResponse):
                return resp
            return MockResponse(status=self.default_status, data=resp)
        
        # Check for pattern matches (simple glob)
        for pattern, resp in self.responses.items():
            if '*' in pattern:
                prefix = pattern.split('*')[0]
                if endpoint.startswith(prefix):
                    if isinstance(resp, MockResponse):
                        return resp
                    return MockResponse(status=self.default_status, data=resp)
        
        return MockResponse(status=self.default_status, data=self.default_response)
    
    async def _delay(self, response: MockResponse) -> None:
        """Apply simulated delay."""
        if response.delay_ms > 0:
            import asyncio
            await asyncio.sleep(response.delay_ms / 1000)
    
    async def get(
        self,
        endpoint: str,
        **params: object,
    ) -> MockResponse:
        """Simulate GET request."""
        self._record("GET", endpoint, params=params)
        response = self._get_response(endpoint)
        await self._delay(response)
        return response
    
    async def post(
        self,
        endpoint: str,
        data: dict[str, object] | None = None,
        **kwargs: object,
    ) -> MockResponse:
        """Simulate POST request."""
        self._record("POST", endpoint, data=data, **kwargs)
        response = self._get_response(endpoint)
        await self._delay(response)
        return response
    
    async def put(
        self,
        endpoint: str,
        data: dict[str, object] | None = None,
        **kwargs: object,
    ) -> MockResponse:
        """Simulate PUT request."""
        self._record("PUT", endpoint, data=data, **kwargs)
        response = self._get_response(endpoint)
        await self._delay(response)
        return response
    
    async def delete(
        self,
        endpoint: str,
        **kwargs: object,
    ) -> MockResponse:
        """Simulate DELETE request."""
        self._record("DELETE", endpoint, **kwargs)
        response = self._get_response(endpoint)
        await self._delay(response)
        return response
    
    def clear(self) -> None:
        """Clear recorded requests."""
        self.requests.clear()
    
    def set_response(
        self,
        endpoint: str,
        response: str | dict[str, object] | MockResponse,
    ) -> None:
        """Set response for endpoint."""
        self.responses[endpoint] = response
    
    def set_error(
        self,
        endpoint: str,
        status: int = 500,
        message: str = "Internal Server Error",
    ) -> None:
        """Configure endpoint to return error."""
        self.responses[endpoint] = MockResponse(
            status=status,
            data={"error": message},
        )
    
    def assert_called(self) -> None:
        """Assert API was called at least once."""
        if not self.requests:
            raise AssertionError("Expected API to be called")
    
    def assert_endpoint_called(self, endpoint: str) -> None:
        """Assert specific endpoint was called."""
        for req in self.requests:
            if req["endpoint"] == endpoint:
                return
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
