"""Effect system implementation for side-effect tracking and pure testing.

Provides:
- Effect declarations via @effects decorator
- Effect handlers for dependency substitution
- Pure handlers for testing without mocks
- Verification for compile-time-style safety
"""

from __future__ import annotations

import random as stdlib_random
import time as stdlib_time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    ParamSpec,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Generator

P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")


# ═══════════════════════════════════════════════════════════════════════════════
# Effect Type System
# ═══════════════════════════════════════════════════════════════════════════════


class Effect(str, Enum):
    """Standard effect types for common side effects.
    
    Effects categorize the types of side effects a tool may perform:
    - DB: Database operations (read/write)
    - HTTP: Network requests to external services
    - FILE: File system operations
    - CACHE: Caching layer interactions
    - ENV: Environment variable access
    - TIME: Time-dependent operations
    - RANDOM: Non-deterministic random operations
    - LOG: Logging side effects
    """
    
    DB = "db"
    HTTP = "http"
    FILE = "file"
    CACHE = "cache"
    ENV = "env"
    TIME = "time"
    RANDOM = "random"
    LOG = "log"
    
    @classmethod
    def from_str(cls, value: str) -> "Effect":
        """Parse effect from string, case-insensitive."""
        return cls(value.lower())
    
    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if string is a valid effect name."""
        try:
            cls.from_str(value)
            return True
        except ValueError:
            return False


# Convenience aliases
DB = Effect.DB
HTTP = Effect.HTTP
FILE = Effect.FILE
CACHE = Effect.CACHE
ENV = Effect.ENV
TIME = Effect.TIME
RANDOM = Effect.RANDOM
LOG = Effect.LOG

STANDARD_EFFECTS = frozenset(Effect)

# Type alias for effect sets
EffectSet = frozenset[Effect | str]


# ═══════════════════════════════════════════════════════════════════════════════
# Effect Declaration
# ═══════════════════════════════════════════════════════════════════════════════

# Attribute name for storing effects on functions/tools
_EFFECTS_ATTR = "__toolcase_effects__"


def _normalize_effect(e: Effect | str) -> str:
    """Normalize effect to string form."""
    return e.value if isinstance(e, Effect) else e.lower()


def _normalize_effects(*args: Effect | str) -> frozenset[str]:
    """Normalize effects to frozenset of strings."""
    return frozenset(_normalize_effect(e) for e in args)


@overload
def effects(*effect_types: Effect | str) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

@overload
def effects(func: Callable[P, R]) -> Callable[P, R]: ...


def effects(*args: Effect | str | Callable[P, R]) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    """Declare effects that a function/tool performs.
    
    Use as decorator to declare side effects, enabling:
    - Static verification of effect handling
    - Testing without mocks via effect substitution
    - Documentation of tool behavior
    
    Args:
        *effect_types: Effect enum values or string names
    
    Returns:
        Decorated function with effects metadata
    
    Example:
        >>> @effects("db", "http")
        ... async def fetch_data(query: str) -> str:
        ...     # db and http side effects here
        ...     ...
        
        >>> @effects(Effect.DB, Effect.CACHE)
        ... async def cached_query(q: str) -> str:
        ...     ...
    """
    # Called without arguments or with function directly
    if len(args) == 1 and callable(args[0]) and not isinstance(args[0], (Effect, str)):
        func = args[0]
        setattr(func, _EFFECTS_ATTR, frozenset())
        return func
    
    # Called with effect types
    effect_set = _normalize_effects(*args)  # type: ignore[arg-type]
    
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Merge with existing effects if any
        existing = getattr(func, _EFFECTS_ATTR, frozenset())
        setattr(func, _EFFECTS_ATTR, existing | effect_set)
        return func
    
    return decorator


def get_effects(obj: object) -> frozenset[str]:
    """Get declared effects from a function, tool, or class.
    
    Searches for effects on:
    - The object directly (from @effects decorator)
    - _effects attribute (for FunctionTool with effects= param)
    - declared_effects property (for FunctionTool)
    - _func/_original_func attributes (for wrapped tools)
    - metadata.effects (for BaseTool with effect metadata)
    """
    # Direct attribute from @effects decorator
    if (fx := getattr(obj, _EFFECTS_ATTR, None)) is not None:
        return fx
    
    # FunctionTool _effects attribute (from effects= param in @tool)
    if (fx := getattr(obj, "_effects", None)) is not None and fx:
        return fx
    
    # FunctionTool declared_effects property
    if (fx := getattr(obj, "declared_effects", None)) is not None and fx:
        return fx
    
    # FunctionTool wrapping - check the wrapped function
    if (func := getattr(obj, "_func", None)) is not None:
        if (fx := getattr(func, _EFFECTS_ATTR, None)) is not None:
            return fx
    
    if (func := getattr(obj, "_original_func", None)) is not None:
        if (fx := getattr(func, _EFFECTS_ATTR, None)) is not None:
            return fx
    
    # BaseTool with effects in metadata extension
    if (meta := getattr(obj, "metadata", None)) is not None:
        if (fx := getattr(meta, "effects", None)) is not None:
            return fx
    
    return frozenset()


def has_effects(obj: object, *required: Effect | str) -> bool:
    """Check if object has all required effects declared."""
    declared = get_effects(obj)
    return all(_normalize_effect(e) in declared for e in required)


# ═══════════════════════════════════════════════════════════════════════════════
# Effect Handler Protocol
# ═══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class EffectHandler(Protocol):
    """Protocol for effect handlers.
    
    Effect handlers provide substitutable implementations for side effects,
    enabling testing without mocks by swapping real implementations with
    pure/deterministic ones.
    """
    
    @property
    def effect_type(self) -> str:
        """The effect type this handler provides."""
        ...


class PureHandler(ABC):
    """Base class for pure (side-effect-free) handlers for testing.
    
    Pure handlers record operations without executing real side effects,
    enabling deterministic testing and verification.
    """
    
    effect_type: str
    
    @abstractmethod
    def reset(self) -> None:
        """Reset handler state for fresh test runs."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# Built-in Pure Handlers for Testing
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DBRecord:
    """Record of a DB operation."""
    operation: str  # "query", "execute", "fetch_one", etc.
    query: str
    params: tuple[object, ...] = ()
    result: object = None


@dataclass
class InMemoryDB(PureHandler):
    """Pure in-memory database handler for testing.
    
    Records all operations and returns configurable responses.
    
    Example:
        >>> db = InMemoryDB()
        >>> db.set_response("SELECT * FROM users", [{"id": 1, "name": "Alice"}])
        >>> async with test_effects(db=db):
        ...     result = await fetch_users()
        >>> assert db.queries == ["SELECT * FROM users"]
    """
    
    effect_type: str = "db"
    _records: list[DBRecord] = field(default_factory=list)
    _responses: dict[str, object] = field(default_factory=dict)
    _default_response: object = None
    
    def reset(self) -> None:
        self._records.clear()
    
    def set_response(self, query_pattern: str, response: object) -> None:
        """Set response for queries matching pattern."""
        self._responses[query_pattern] = response
    
    def set_default(self, response: object) -> None:
        """Set default response for unmatched queries."""
        self._default_response = response
    
    @property
    def queries(self) -> list[str]:
        """List of executed queries."""
        return [r.query for r in self._records]
    
    @property
    def records(self) -> list[DBRecord]:
        """All recorded operations."""
        return list(self._records)
    
    async def fetch_one(self, query: str, *params: object) -> object:
        """Simulate fetch_one - returns first result or None."""
        result = self._get_response(query)
        self._records.append(DBRecord("fetch_one", query, params, result))
        return result[0] if isinstance(result, list) and result else result
    
    async def fetch_all(self, query: str, *params: object) -> list[object]:
        """Simulate fetch_all - returns list result."""
        result = self._get_response(query)
        self._records.append(DBRecord("fetch_all", query, params, result))
        return result if isinstance(result, list) else [result] if result else []
    
    async def execute(self, query: str, *params: object) -> int:
        """Simulate execute - returns affected rows."""
        self._records.append(DBRecord("execute", query, params, 1))
        return 1
    
    def _get_response(self, query: str) -> object:
        """Get response for query, checking patterns."""
        for pattern, response in self._responses.items():
            if pattern in query:
                return response
        return self._default_response


@dataclass
class HTTPRequest:
    """Record of an HTTP request."""
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes | str | None = None
    response_status: int = 200
    response_body: bytes | str | None = None


@dataclass
class RecordingHTTP(PureHandler):
    """Pure HTTP handler that records requests and returns mocked responses.
    
    Example:
        >>> http = RecordingHTTP()
        >>> http.mock_response("https://api.example.com/users", {"users": []})
        >>> async with test_effects(http=http):
        ...     result = await fetch_users()
        >>> assert http.requests[0].url == "https://api.example.com/users"
    """
    
    effect_type: str = "http"
    _requests: list[HTTPRequest] = field(default_factory=list)
    _responses: dict[str, tuple[int, object]] = field(default_factory=dict)
    _default_response: tuple[int, object] = (200, {})
    
    def reset(self) -> None:
        self._requests.clear()
    
    def mock_response(self, url_pattern: str, body: object, status: int = 200) -> None:
        """Set response for URLs matching pattern."""
        self._responses[url_pattern] = (status, body)
    
    def mock_error(self, url_pattern: str, status: int = 500, body: object = None) -> None:
        """Set error response for URLs matching pattern."""
        self._responses[url_pattern] = (status, body or {"error": "Internal Server Error"})
    
    @property
    def requests(self) -> list[HTTPRequest]:
        """All recorded requests."""
        return list(self._requests)
    
    @property
    def urls(self) -> list[str]:
        """List of requested URLs."""
        return [r.url for r in self._requests]
    
    async def request(self, method: str, url: str, *, headers: dict[str, str] | None = None, body: bytes | str | None = None) -> tuple[int, object]:
        """Simulate HTTP request."""
        status, response = self._get_response(url)
        req = HTTPRequest(method, url, headers or {}, body, status, response)
        self._requests.append(req)
        return status, response
    
    async def get(self, url: str, **kwargs: object) -> tuple[int, object]:
        return await self.request("GET", url, **kwargs)  # type: ignore[arg-type]
    
    async def post(self, url: str, **kwargs: object) -> tuple[int, object]:
        return await self.request("POST", url, **kwargs)  # type: ignore[arg-type]
    
    def _get_response(self, url: str) -> tuple[int, object]:
        for pattern, response in self._responses.items():
            if pattern in url:
                return response
        return self._default_response


@dataclass
class FSOperation:
    """Record of a file system operation."""
    operation: str  # "read", "write", "delete", "exists", etc.
    path: str
    data: bytes | str | None = None


@dataclass
class InMemoryFS(PureHandler):
    """Pure in-memory file system for testing.
    
    Example:
        >>> fs = InMemoryFS()
        >>> fs.write("/config.json", '{"key": "value"}')
        >>> async with test_effects(file=fs):
        ...     config = await read_config()
        >>> assert fs.operations[0].path == "/config.json"
    """
    
    effect_type: str = "file"
    _files: dict[str, bytes] = field(default_factory=dict)
    _operations: list[FSOperation] = field(default_factory=list)
    
    def reset(self) -> None:
        self._operations.clear()
    
    def seed(self, path: str, content: str | bytes) -> None:
        """Pre-populate a file for testing."""
        self._files[path] = content.encode() if isinstance(content, str) else content
    
    @property
    def operations(self) -> list[FSOperation]:
        return list(self._operations)
    
    @property
    def files(self) -> dict[str, bytes]:
        return dict(self._files)
    
    async def read(self, path: str) -> bytes:
        self._operations.append(FSOperation("read", path))
        if path not in self._files:
            raise FileNotFoundError(path)
        return self._files[path]
    
    async def read_text(self, path: str, encoding: str = "utf-8") -> str:
        return (await self.read(path)).decode(encoding)
    
    async def write(self, path: str, data: str | bytes) -> None:
        content = data.encode() if isinstance(data, str) else data
        self._operations.append(FSOperation("write", path, content))
        self._files[path] = content
    
    async def exists(self, path: str) -> bool:
        self._operations.append(FSOperation("exists", path))
        return path in self._files
    
    async def delete(self, path: str) -> None:
        self._operations.append(FSOperation("delete", path))
        self._files.pop(path, None)


@dataclass
class NoOpCache(PureHandler):
    """Cache handler that records but doesn't persist.
    
    Useful for testing cache-dependent code without actual caching.
    """
    
    effect_type: str = "cache"
    _gets: list[str] = field(default_factory=list)
    _sets: list[tuple[str, object, float | None]] = field(default_factory=list)
    _data: dict[str, object] = field(default_factory=dict)
    _enabled: bool = True
    
    def reset(self) -> None:
        self._gets.clear()
        self._sets.clear()
        self._data.clear()
    
    def enable(self) -> None:
        self._enabled = True
    
    def disable(self) -> None:
        """Disable cache (always miss)."""
        self._enabled = False
    
    @property
    def gets(self) -> list[str]:
        return list(self._gets)
    
    @property
    def sets(self) -> list[tuple[str, object, float | None]]:
        return list(self._sets)
    
    def get(self, key: str) -> object | None:
        self._gets.append(key)
        return self._data.get(key) if self._enabled else None
    
    def set(self, key: str, value: object, ttl: float | None = None) -> None:
        self._sets.append((key, value, ttl))
        if self._enabled:
            self._data[key] = value
    
    def delete(self, key: str) -> None:
        self._data.pop(key, None)


@dataclass
class FrozenTime(PureHandler):
    """Time handler that returns controlled values.
    
    Essential for testing time-dependent logic deterministically.
    
    Example:
        >>> frozen = FrozenTime(datetime(2024, 1, 1, 12, 0, 0))
        >>> async with test_effects(time=frozen):
        ...     timestamp = get_current_time()
        >>> assert timestamp == "2024-01-01T12:00:00"
    """
    
    effect_type: str = "time"
    _frozen_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _advances: list[float] = field(default_factory=list)
    _offset: float = 0.0
    
    def reset(self) -> None:
        self._advances.clear()
        self._offset = 0.0
    
    def freeze(self, at: datetime) -> None:
        """Freeze time at specific moment."""
        self._frozen_at = at
        self._offset = 0.0
    
    def advance(self, seconds: float) -> None:
        """Advance frozen time."""
        self._advances.append(seconds)
        self._offset += seconds
    
    def now(self) -> datetime:
        """Get current (frozen) time."""
        from datetime import timedelta
        return self._frozen_at + timedelta(seconds=self._offset)
    
    def time(self) -> float:
        """Get current (frozen) timestamp."""
        return self._frozen_at.timestamp() + self._offset
    
    def sleep(self, seconds: float) -> None:
        """Record sleep (doesn't actually sleep)."""
        self.advance(seconds)


@dataclass
class SeededRandom(PureHandler):
    """Random handler with reproducible sequences.
    
    Example:
        >>> rng = SeededRandom(seed=42)
        >>> async with test_effects(random=rng):
        ...     value = get_random_value()
        >>> # Same seed produces same sequence
    """
    
    effect_type: str = "random"
    _seed: int = 42
    _rng: stdlib_random.Random = field(default_factory=lambda: stdlib_random.Random(42))
    _calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        self._rng = stdlib_random.Random(self._seed)
    
    def reset(self) -> None:
        self._rng.seed(self._seed)
        self._calls.clear()
    
    def reseed(self, seed: int) -> None:
        """Reset with new seed."""
        self._seed = seed
        self._rng.seed(seed)
        self._calls.clear()
    
    @property
    def calls(self) -> list[tuple[str, tuple[object, ...]]]:
        return list(self._calls)
    
    def random(self) -> float:
        self._calls.append(("random", ()))
        return self._rng.random()
    
    def randint(self, a: int, b: int) -> int:
        self._calls.append(("randint", (a, b)))
        return self._rng.randint(a, b)
    
    def choice(self, seq: list[T]) -> T:
        self._calls.append(("choice", (seq,)))
        return self._rng.choice(seq)
    
    def shuffle(self, seq: list[T]) -> None:
        self._calls.append(("shuffle", (list(seq),)))
        self._rng.shuffle(seq)


@dataclass
class LogEntry:
    """A log entry."""
    level: str
    message: str
    kwargs: dict[str, object] = field(default_factory=dict)


@dataclass
class CollectingLogger(PureHandler):
    """Logger that collects entries for verification.
    
    Example:
        >>> logger = CollectingLogger()
        >>> async with test_effects(log=logger):
        ...     await process_data()
        >>> assert any("processed" in e.message for e in logger.entries)
    """
    
    effect_type: str = "log"
    _entries: list[LogEntry] = field(default_factory=list)
    
    def reset(self) -> None:
        self._entries.clear()
    
    @property
    def entries(self) -> list[LogEntry]:
        return list(self._entries)
    
    @property
    def messages(self) -> list[str]:
        return [e.message for e in self._entries]
    
    def debug(self, msg: str, **kwargs: object) -> None:
        self._entries.append(LogEntry("DEBUG", msg, kwargs))
    
    def info(self, msg: str, **kwargs: object) -> None:
        self._entries.append(LogEntry("INFO", msg, kwargs))
    
    def warning(self, msg: str, **kwargs: object) -> None:
        self._entries.append(LogEntry("WARNING", msg, kwargs))
    
    def error(self, msg: str, **kwargs: object) -> None:
        self._entries.append(LogEntry("ERROR", msg, kwargs))
    
    def exception(self, msg: str, **kwargs: object) -> None:
        self._entries.append(LogEntry("EXCEPTION", msg, kwargs))


# ═══════════════════════════════════════════════════════════════════════════════
# Effect Context & Handler Registry
# ═══════════════════════════════════════════════════════════════════════════════

# Context variable for effect handlers
_effect_handlers: ContextVar[dict[str, EffectHandler | PureHandler]] = ContextVar("effect_handlers")


@dataclass
class EffectContext:
    """Context holding active effect handlers.
    
    Provides dict-like access to handlers and tracks which effects are active.
    """
    
    handlers: dict[str, EffectHandler | PureHandler] = field(default_factory=dict)
    
    def __getitem__(self, effect: Effect | str) -> EffectHandler | PureHandler:
        key = _normalize_effect(effect)
        if key not in self.handlers:
            raise KeyError(f"No handler for effect '{key}'")
        return self.handlers[key]
    
    def __contains__(self, effect: Effect | str) -> bool:
        return _normalize_effect(effect) in self.handlers
    
    def get(self, effect: Effect | str, default: T = None) -> EffectHandler | PureHandler | T:  # type: ignore[assignment]
        return self.handlers.get(_normalize_effect(effect), default)
    
    @property
    def active_effects(self) -> frozenset[str]:
        return frozenset(self.handlers.keys())


def get_effect_context() -> EffectContext:
    """Get current effect context."""
    return EffectContext(_effect_handlers.get({}))


def get_handler(effect: Effect | str) -> EffectHandler | PureHandler | None:
    """Get handler for an effect in current context."""
    handlers = _effect_handlers.get({})
    return handlers.get(_normalize_effect(effect))


@dataclass
class EffectHandlerRegistry:
    """Registry for effect handlers.
    
    Manages handler registration and provides scoped contexts for execution.
    """
    
    _handlers: dict[str, EffectHandler | PureHandler] = field(default_factory=dict)
    
    def register(self, handler: EffectHandler | PureHandler) -> None:
        """Register an effect handler."""
        key = handler.effect_type
        self._handlers[key] = handler
    
    def unregister(self, effect: Effect | str) -> None:
        """Remove handler for an effect."""
        self._handlers.pop(_normalize_effect(effect), None)
    
    def get(self, effect: Effect | str) -> EffectHandler | PureHandler | None:
        return self._handlers.get(_normalize_effect(effect))
    
    def has(self, effect: Effect | str) -> bool:
        return _normalize_effect(effect) in self._handlers
    
    @property
    def registered_effects(self) -> frozenset[str]:
        return frozenset(self._handlers.keys())
    
    @asynccontextmanager
    async def scope(self) -> AsyncIterator[EffectContext]:
        """Create async scope with registered handlers."""
        token = _effect_handlers.set(dict(self._handlers))
        try:
            yield EffectContext(dict(self._handlers))
        finally:
            _effect_handlers.reset(token)
    
    @contextmanager
    def scope_sync(self) -> Generator[EffectContext, None, None]:
        """Create sync scope with registered handlers."""
        token = _effect_handlers.set(dict(self._handlers))
        try:
            yield EffectContext(dict(self._handlers))
        finally:
            _effect_handlers.reset(token)


class EffectScope:
    """Scoped effect handler context manager.
    
    Temporarily installs handlers for the duration of the scope, restoring
    previous handlers on exit. Supports nesting.
    
    Example:
        >>> async with EffectScope(db=InMemoryDB(), http=RecordingHTTP()):
        ...     await run_tool_with_effects()
    """
    
    __slots__ = ("_handlers", "_token")
    
    def __init__(self, **handlers: EffectHandler | PureHandler) -> None:
        self._handlers = {_normalize_effect(k): v for k, v in handlers.items()}
        self._token: object | None = None
    
    async def __aenter__(self) -> EffectContext:
        existing = _effect_handlers.get({})
        merged = {**existing, **self._handlers}
        self._token = _effect_handlers.set(merged)
        return EffectContext(merged)
    
    async def __aexit__(self, *exc: object) -> None:
        if self._token is not None:
            _effect_handlers.reset(self._token)
    
    def __enter__(self) -> EffectContext:
        existing = _effect_handlers.get({})
        merged = {**existing, **self._handlers}
        self._token = _effect_handlers.set(merged)
        return EffectContext(merged)
    
    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _effect_handlers.reset(self._token)


def effect_scope(**handlers: EffectHandler | PureHandler) -> EffectScope:
    """Create an effect scope with handlers.
    
    Convenience function for EffectScope.
    """
    return EffectScope(**handlers)


def test_effects(**handlers: EffectHandler | PureHandler) -> EffectScope:
    """Create a testing scope with pure handlers.
    
    Alias for effect_scope, semantically indicating testing context.
    Resets all handlers before use for clean test isolation.
    
    Example:
        >>> async with test_effects(db=InMemoryDB(), time=FrozenTime()):
        ...     result = await my_tool(query="test")
        ...     assert db.queries == ["SELECT ..."]
    """
    # Reset handlers for clean test state
    for h in handlers.values():
        if isinstance(h, PureHandler):
            h.reset()
    return EffectScope(**handlers)


# ═══════════════════════════════════════════════════════════════════════════════
# Effect Verification
# ═══════════════════════════════════════════════════════════════════════════════


class EffectViolation(Exception):
    """Base exception for effect system violations."""
    pass


class MissingEffectHandler(EffectViolation):
    """Raised when a declared effect has no handler."""
    
    def __init__(self, effect: str, tool_name: str) -> None:
        self.effect = effect
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' declares effect '{effect}' but no handler is registered")


class UndeclaredEffect(EffectViolation):
    """Raised when an effect is used without declaration."""
    
    def __init__(self, effect: str, tool_name: str) -> None:
        self.effect = effect
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' uses effect '{effect}' without declaring it")


def verify_effects(
    tool: object,
    handlers: dict[str, EffectHandler | PureHandler] | EffectHandlerRegistry | None = None,
    *,
    strict: bool = False,
) -> list[EffectViolation]:
    """Verify that a tool's declared effects have handlers.
    
    Args:
        tool: Tool or function to verify
        handlers: Available handlers (None = use context)
        strict: If True, raise on first violation instead of collecting
    
    Returns:
        List of violations found (empty if valid)
    
    Raises:
        EffectViolation: In strict mode, on first violation
    """
    declared = get_effects(tool)
    if not declared:
        return []
    
    # Get available handlers
    if handlers is None:
        available = frozenset(_effect_handlers.get({}).keys())
    elif isinstance(handlers, EffectHandlerRegistry):
        available = handlers.registered_effects
    else:
        available = frozenset(handlers.keys())
    
    tool_name = getattr(getattr(tool, "metadata", None), "name", None) or getattr(tool, "__name__", str(tool))
    violations: list[EffectViolation] = []
    
    for effect in declared:
        if effect not in available:
            violation = MissingEffectHandler(effect, tool_name)
            if strict:
                raise violation
            violations.append(violation)
    
    return violations
