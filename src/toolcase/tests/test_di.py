"""Tests for dependency injection system."""

import pytest

from toolcase import Scope, get_registry, reset_registry, tool


class MockDatabase:
    """Mock database for testing."""
    
    def __init__(self, dsn: str = "mock://localhost"):
        self.dsn = dsn
        self.connected = True
        self.queries: list[str] = []
    
    async def fetch(self, query: str) -> list[dict[str, str]]:
        self.queries.append(query)
        return [{"id": "1", "name": "Test User"}]
    
    async def close(self) -> None:
        self.connected = False


class MockHttpClient:
    """Mock HTTP client for testing."""
    
    def __init__(self):
        self.requests: list[str] = []
    
    async def get(self, url: str) -> str:
        self.requests.append(url)
        return f"Response from {url}"
    
    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset registry before each test."""
    reset_registry()
    yield
    reset_registry()


@pytest.mark.asyncio
async def test_singleton_injection():
    """Test that singleton dependencies are shared across invocations."""
    registry = get_registry()
    
    # Register singleton database
    db_instance = MockDatabase()
    registry.provide("db", lambda: db_instance, Scope.SINGLETON)
    
    @tool(description="Fetch user from database", inject=["db"])
    async def fetch_user(user_id: str, db: MockDatabase) -> str:
        result = await db.fetch(f"SELECT * FROM users WHERE id = {user_id}")
        return f"Found: {result[0]['name']}"
    
    registry.register(fetch_user)
    
    # Execute twice
    result1 = await registry.execute("fetch_user", {"user_id": "1"})
    result2 = await registry.execute("fetch_user", {"user_id": "2"})
    
    assert "Found: Test User" in result1
    assert "Found: Test User" in result2
    assert len(db_instance.queries) == 2


@pytest.mark.asyncio
async def test_scoped_injection():
    """Test that scoped dependencies are new per execution."""
    registry = get_registry()
    
    call_count = 0
    
    def create_client():
        nonlocal call_count
        call_count += 1
        return MockHttpClient()
    
    registry.provide("http", create_client, Scope.SCOPED)
    
    @tool(description="Make HTTP request", inject=["http"])
    async def make_request(url: str, http: MockHttpClient) -> str:
        return await http.get(url)
    
    registry.register(make_request)
    
    # Each execution gets a new scoped client
    await registry.execute("make_request", {"url": "http://a.com"})
    await registry.execute("make_request", {"url": "http://b.com"})
    
    assert call_count == 2


@pytest.mark.asyncio
async def test_multiple_dependencies():
    """Test injecting multiple dependencies."""
    registry = get_registry()
    
    registry.provide("db", MockDatabase, Scope.SINGLETON)
    registry.provide("http", MockHttpClient, Scope.SCOPED)
    
    @tool(description="Fetch and enrich user data", inject=["db", "http"])
    async def fetch_enriched_user(
        user_id: str,
        db: MockDatabase,
        http: MockHttpClient,
    ) -> str:
        user = await db.fetch(f"SELECT * FROM users WHERE id = {user_id}")
        extra = await http.get(f"http://api/users/{user_id}/details")
        return f"User: {user[0]['name']}, Extra: {extra}"
    
    registry.register(fetch_enriched_user)
    
    result = await registry.execute("fetch_enriched_user", {"user_id": "123"})
    
    assert "User: Test User" in result
    assert "Extra: Response from" in result


@pytest.mark.asyncio
async def test_missing_dependency_error():
    """Test that missing dependencies produce clear error."""
    registry = get_registry()
    
    @tool(description="Tool with missing dep", inject=["nonexistent"])
    async def broken_tool(query: str, nonexistent: object) -> str:
        return query
    
    registry.register(broken_tool)
    
    result = await registry.execute("broken_tool", {"query": "test"})
    
    assert "Missing dependency" in result or "nonexistent" in result


@pytest.mark.asyncio
async def test_container_dispose():
    """Test that dispose cleans up singletons."""
    registry = get_registry()
    
    db = MockDatabase()
    registry.provide("db", lambda: db, Scope.SINGLETON)
    
    # Resolve to create instance
    async with registry.container.scope() as ctx:
        await registry.container.resolve("db", ctx)
    
    assert db.connected
    
    await registry.dispose()
    assert not db.connected


def test_provide_fluent_api():
    """Test that provide returns None for chaining."""
    registry = get_registry()
    
    # Should not raise
    registry.provide("a", lambda: 1)
    registry.provide("b", lambda: 2, Scope.SCOPED)
    registry.provide("c", lambda: 3, Scope.TRANSIENT)
    
    assert registry.container.has("a")
    assert registry.container.has("b")
    assert registry.container.has("c")
