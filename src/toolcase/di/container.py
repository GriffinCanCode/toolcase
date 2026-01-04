"""Dependency injection container with scoped lifecycle management.

Provides clean resource management for tools:
- Singleton: One instance per container (app-level)
- Scoped: One instance per request context
- Transient: New instance per injection

Example:
    >>> container = Container()
    >>> container.provide("db", lambda: AsyncpgPool(...), Scope.SINGLETON)
    >>> container.provide("http", lambda: httpx.AsyncClient(), Scope.SCOPED)
    >>>
    >>> async with container.scope() as ctx:
    ...     db = await container.resolve("db", ctx)
    ...     http = await container.resolve("http", ctx)
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from ..errors import Err, ErrorCode, ErrorTrace, Ok, Result

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


T = TypeVar("T")

# Type alias for DI resolution results
DIResult = Result[object, ErrorTrace]


class Scope(Enum):
    """Dependency lifecycle scope."""
    
    SINGLETON = auto()  # One instance per container
    SCOPED = auto()     # One instance per request/context
    TRANSIENT = auto()  # New instance every resolution


@runtime_checkable
class Disposable(Protocol):
    """Protocol for resources that need cleanup."""
    
    async def close(self) -> None: ...


Factory = Callable[[], T | Awaitable[T]]


@dataclass(slots=True)
class Provider:
    """Registered provider with factory and scope."""
    
    factory: Factory[object]
    scope: Scope
    instance: object | None = None  # For singletons


@dataclass(slots=True)
class ScopedContext:
    """Request-scoped container context for scoped instances."""
    
    instances: dict[str, object] = field(default_factory=dict)
    _disposed: bool = False
    
    async def dispose(self) -> None:
        """Clean up all scoped resources."""
        if self._disposed:
            return
        self._disposed = True
        
        # Dispose in reverse order (LIFO)
        for resource in reversed(list(self.instances.values())):
            if isinstance(resource, Disposable):
                try:
                    await resource.close()
                except Exception:
                    pass  # Best-effort cleanup
            elif hasattr(resource, "aclose"):
                try:
                    await resource.aclose()
                except Exception:
                    pass


class Container:
    """Dependency injection container.
    
    Manages provider registration and instance lifecycle.
    Thread-safe for singleton resolution.
    
    Example:
        >>> container = Container()
        >>> container.provide("cache", lambda: RedisCache(), Scope.SINGLETON)
        >>> container.provide("session", SessionFactory, Scope.SCOPED)
        >>>
        >>> # In request handler:
        >>> async with container.scope() as ctx:
        ...     cache = await container.resolve("cache", ctx)
        ...     session = await container.resolve("session", ctx)
    """
    
    __slots__ = ("_providers", "_lock")
    
    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}
        self._lock = asyncio.Lock()
    
    def provide(
        self,
        name: str,
        factory: Factory[T],
        scope: Scope = Scope.SINGLETON,
    ) -> None:
        """Register a dependency provider.
        
        Args:
            name: Dependency identifier (e.g., "db", "http_client")
            factory: Callable returning instance (sync or async)
            scope: Lifecycle scope for instances
        """
        self._providers[name] = Provider(factory=factory, scope=scope)
    
    def has(self, name: str) -> bool:
        """Check if provider exists."""
        return name in self._providers
    
    async def resolve(self, name: str, ctx: ScopedContext | None = None) -> object:
        """Resolve a dependency by name.
        
        Args:
            name: Provider name to resolve
            ctx: Scoped context (required for SCOPED dependencies)
        
        Returns:
            Resolved instance
        
        Raises:
            KeyError: Unknown dependency
            ValueError: Scoped dependency without context
        """
        if name not in self._providers:
            raise KeyError(f"Unknown dependency: {name}")
        
        provider = self._providers[name]
        
        match provider.scope:
            case Scope.SINGLETON:
                return await self._resolve_singleton(provider)
            case Scope.SCOPED:
                return await self._resolve_scoped(name, provider, ctx)
            case Scope.TRANSIENT:
                return await self._create_instance(provider.factory)
    
    async def resolve_result(self, name: str, ctx: ScopedContext | None = None) -> DIResult:
        """Resolve dependency with Result-based error handling.
        
        Type-safe alternative to resolve() that returns Result instead of raising.
        Enables monadic error handling in DI-heavy code paths.
        
        Args:
            name: Provider name to resolve
            ctx: Scoped context (required for SCOPED dependencies)
        
        Returns:
            Result[object, ErrorTrace] with resolved instance or error
        
        Example:
            >>> result = await container.resolve_result("db", ctx)
            >>> db = result.unwrap_or_else(lambda e: fallback_db())
        """
        if name not in self._providers:
            return Err(ErrorTrace(
                message=f"Unknown dependency: {name}",
                error_code=ErrorCode.NOT_FOUND.value,
                recoverable=False,
            ).with_operation("di:resolve", dependency=name))
        
        provider = self._providers[name]
        
        try:
            match provider.scope:
                case Scope.SINGLETON:
                    return Ok(await self._resolve_singleton(provider))
                case Scope.SCOPED:
                    if ctx is None:
                        return Err(ErrorTrace(
                            message=f"Scoped dependency '{name}' requires context",
                            error_code=ErrorCode.INVALID_PARAMS.value,
                            recoverable=False,
                        ).with_operation("di:resolve", dependency=name))
                    return Ok(await self._resolve_scoped(name, provider, ctx))
                case Scope.TRANSIENT:
                    return Ok(await self._create_instance(provider.factory))
        except Exception as e:
            return Err(ErrorTrace(
                message=f"Failed to resolve '{name}': {e}",
                error_code=ErrorCode.EXTERNAL_SERVICE_ERROR.value,
                recoverable=True,
            ).with_operation("di:resolve", dependency=name))
        
        # Unreachable but satisfies type checker
        return Err(ErrorTrace(message=f"Unknown scope for {name}"))  # pragma: no cover
    
    async def resolve_many(
        self,
        names: list[str],
        ctx: ScopedContext | None = None,
    ) -> dict[str, object]:
        """Resolve multiple dependencies at once."""
        return {name: await self.resolve(name, ctx) for name in names}
    
    async def resolve_many_result(
        self,
        names: list[str],
        ctx: ScopedContext | None = None,
    ) -> Result[dict[str, object], ErrorTrace]:
        """Resolve multiple dependencies with Result-based error handling.
        
        Fails fast on first resolution error.
        
        Returns:
            Result with dict of resolved instances or first error
        """
        resolved: dict[str, object] = {}
        for name in names:
            result = await self.resolve_result(name, ctx)
            if result.is_err():
                return result.map_err(
                    lambda e: e.with_operation("di:resolve_many", failed_at=name)
                )
            resolved[name] = result.unwrap()
        return Ok(resolved)
    
    async def _resolve_singleton(self, provider: Provider) -> object:
        """Get or create singleton instance (thread-safe)."""
        if provider.instance is not None:
            return provider.instance
        
        async with self._lock:
            # Double-check after acquiring lock
            if provider.instance is None:
                provider.instance = await self._create_instance(provider.factory)
            return provider.instance
    
    async def _resolve_scoped(
        self,
        name: str,
        provider: Provider,
        ctx: ScopedContext | None,
    ) -> object:
        """Get or create scoped instance."""
        if ctx is None:
            raise ValueError(f"Scoped dependency '{name}' requires context")
        
        if name not in ctx.instances:
            ctx.instances[name] = await self._create_instance(provider.factory)
        return ctx.instances[name]
    
    @staticmethod
    async def _create_instance(factory: Factory[object]) -> object:
        """Invoke factory, handling sync/async."""
        result = factory()
        if asyncio.iscoroutine(result):
            return await result
        return result
    
    @asynccontextmanager
    async def scope(self) -> AsyncIterator[ScopedContext]:
        """Create a scoped context with automatic cleanup.
        
        Example:
            >>> async with container.scope() as ctx:
            ...     db = await container.resolve("db", ctx)
            ...     # db available here
            ... # Scoped resources cleaned up
        """
        ctx = ScopedContext()
        try:
            yield ctx
        finally:
            await ctx.dispose()
    
    async def dispose(self) -> None:
        """Dispose all singleton resources."""
        async with self._lock:
            for provider in self._providers.values():
                if provider.instance is not None:
                    if isinstance(provider.instance, Disposable):
                        await provider.instance.close()
                    elif hasattr(provider.instance, "aclose"):
                        await provider.instance.aclose()
                    provider.instance = None
    
    def clear(self) -> None:
        """Remove all providers (does not dispose)."""
        self._providers.clear()
