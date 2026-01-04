"""Effect System - Declare and track side effects for compile-time verification and pure testing.

Enables tools to explicitly declare their side effects (db, http, file_system, etc.),
allowing the framework to verify effect requirements at registration and enable
testing without mocks through effect handlers.

Example:
    >>> @tool(description="Fetch user from database")
    ... @effects("db", "cache")
    ... async def fetch_user(user_id: str, db: Database) -> str:
    ...     return await db.fetch_one("SELECT * FROM users WHERE id = $1", user_id)
    ...
    >>> # In tests - use pure handlers instead of mocks
    >>> async with test_effects(db=InMemoryDB(), cache=NoOpCache()):
    ...     result = await fetch_user(user_id="123")
    
Effect Verification:
    >>> registry.require_effects()  # Enable compile-time-style verification
    >>> registry.register(fetch_user)  # Verifies declared effects have handlers
"""

from .effects import (
    # Core types
    Effect,
    EffectSet,
    EffectHandler,
    EffectContext,
    # Standard effects
    STANDARD_EFFECTS,
    DB,
    HTTP,
    FILE,
    CACHE,
    ENV,
    TIME,
    RANDOM,
    LOG,
    # Decorator - use alias to avoid module name conflict
    effects as declare_effects,
    get_effects,
    has_effects,
    get_handler,
    # Handlers
    EffectHandlerRegistry,
    EffectScope,
    effect_scope,
    test_effects,
    # Pure handlers for testing
    PureHandler,
    InMemoryDB,
    RecordingHTTP,
    InMemoryFS,
    NoOpCache,
    FrozenTime,
    SeededRandom,
    CollectingLogger,
    # Verification
    EffectViolation,
    verify_effects,
    MissingEffectHandler,
    UndeclaredEffect,
)

# Also export as 'effects' for backwards compatibility
effects = declare_effects

__all__ = [
    # Core types
    "Effect",
    "EffectSet",
    "EffectHandler",
    "EffectContext",
    # Standard effects
    "STANDARD_EFFECTS",
    "DB",
    "HTTP",
    "FILE",
    "CACHE",
    "ENV",
    "TIME",
    "RANDOM",
    "LOG",
    # Decorator
    "effects",
    "declare_effects",
    "get_effects",
    "has_effects",
    "get_handler",
    # Handlers
    "EffectHandlerRegistry",
    "EffectScope",
    "effect_scope",
    "test_effects",
    # Pure handlers
    "PureHandler",
    "InMemoryDB",
    "RecordingHTTP",
    "InMemoryFS",
    "NoOpCache",
    "FrozenTime",
    "SeededRandom",
    "CollectingLogger",
    # Verification
    "EffectViolation",
    "verify_effects",
    "MissingEffectHandler",
    "UndeclaredEffect",
]
