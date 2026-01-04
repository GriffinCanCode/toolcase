"""Dependency injection for tool resources.

Enables clean lifecycle management for shared resources like database
connections, HTTP clients, and caches.

Example:
    >>> from toolcase import get_registry
    >>> from toolcase.di import Scope
    >>>
    >>> registry = get_registry()
    >>> registry.provide("db", lambda: AsyncpgPool(), Scope.SINGLETON)
    >>> registry.provide("http", lambda: httpx.AsyncClient(), Scope.SCOPED)
    >>>
    >>> @tool(description="Fetch user data", inject=["db", "http"])
    ... async def fetch_user(user_id: str, db: Database, http: HttpClient) -> str:
    ...     user = await db.fetch_one("SELECT * FROM users WHERE id = $1", user_id)
    ...     return f"User: {user['name']}"
"""

from .container import Container, Disposable, Factory, Provider, Scope, ScopedContext

__all__ = [
    "Container",
    "Disposable",
    "Factory",
    "Provider",
    "Scope",
    "ScopedContext",
]
