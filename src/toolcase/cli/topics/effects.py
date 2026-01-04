"""Effects topic - Effect system documentation."""

from .help_topic import HelpTopic

topic = HelpTopic(
    name="effects",
    summary="Effect system for side-effect tracking and pure testing",
    content="""
# Effect System

The effect system enables tools to explicitly declare their side effects
(db, http, file_system, etc.), providing:

1. **Compile-time-style verification**: Ensure handlers exist before execution
2. **Testing without mocks**: Swap real implementations with pure handlers
3. **Explicit documentation**: Make side effects visible in tool signatures

## Declaring Effects

### Via @tool decorator:

```python
from toolcase.foundation import tool

@tool(description="Fetch user from database", effects=["db", "cache"])
async def fetch_user(user_id: str, db: Database) -> str:
    user = await db.fetch_one("SELECT * FROM users WHERE id = $1", user_id)
    return f"User: {user['name']}"
```

### Via @effects decorator:

```python
from toolcase.foundation import effects

@effects("db", "http")
async def fetch_and_enrich(user_id: str) -> str:
    # Function with db and http side effects
    ...
```

## Standard Effects

| Effect   | Description                          |
|----------|--------------------------------------|
| `db`     | Database operations (read/write)     |
| `http`   | Network requests to external services|
| `file`   | File system operations               |
| `cache`  | Caching layer interactions           |
| `env`    | Environment variable access          |
| `time`   | Time-dependent operations            |
| `random` | Non-deterministic random operations  |
| `log`    | Logging side effects                 |

## Pure Handlers for Testing

Pure handlers record operations without real side effects:

```python
from toolcase.foundation import (
    InMemoryDB, RecordingHTTP, InMemoryFS, NoOpCache,
    FrozenTime, SeededRandom, CollectingLogger,
    test_effects
)

# Configure handlers
db = InMemoryDB()
db.set_response("SELECT * FROM users", [{"id": 1, "name": "Alice"}])

# Test without real database
async with test_effects(db=db):
    result = await fetch_user(user_id="1")
    assert "Alice" in result
    assert db.queries == ["SELECT * FROM users WHERE id = $1"]
```

## Available Pure Handlers

### InMemoryDB
```python
db = InMemoryDB()
db.set_response("SELECT", [{"id": 1}])  # Pattern-based responses
db.set_default(None)                     # Default for unmatched

await db.fetch_one("SELECT * FROM users")
await db.fetch_all("SELECT * FROM items")
await db.execute("DELETE FROM logs")

# Inspect recorded operations
db.queries      # List of executed queries
db.records      # Full DBRecord objects
```

### RecordingHTTP
```python
http = RecordingHTTP()
http.mock_response("api.example.com/users", {"users": []})
http.mock_error("api.example.com/fail", 500)

status, body = await http.get("https://api.example.com/users")

# Inspect recorded requests
http.urls       # List of requested URLs
http.requests   # Full HTTPRequest objects
```

### FrozenTime
```python
from datetime import datetime, timezone

frozen = FrozenTime()
frozen.freeze(datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))

t1 = frozen.now()   # Always returns frozen time
frozen.advance(3600) # Advance 1 hour
t2 = frozen.now()   # Returns 13:00:00
```

### SeededRandom
```python
rng = SeededRandom(_seed=42)
rng.random()        # Reproducible float
rng.randint(1, 10)  # Reproducible int
rng.choice([1,2,3]) # Reproducible choice
rng.reseed(42)      # Reset sequence
```

### CollectingLogger
```python
logger = CollectingLogger()
logger.info("Processing", item_id=123)
logger.error("Failed")

logger.messages  # ["Processing", "Failed"]
logger.entries   # Full LogEntry objects with kwargs
```

## Effect Verification

Enable compile-time-style checking at registration:

```python
from toolcase.foundation import ToolRegistry, InMemoryDB, MissingEffectHandler

registry = ToolRegistry()
registry.require_effects()  # Enable verification

# Register handlers BEFORE tools
registry.provide_effect("db", InMemoryDB())

@tool(effects=["db"])
async def fetch(): ...

registry.register(fetch)  # OK - db handler exists

@tool(effects=["unknown"])
async def broken(): ...

registry.register(broken)  # Raises MissingEffectHandler!
```

## Querying Tools by Effect

```python
# Find all tools that use db effects
db_tools = registry.tools_by_effect("db")

# Find tools that use both db and http
api_tools = registry.tools_by_effect("db", "http")

# Verify all tools have handlers
missing = registry.verify_all_effects()
# Returns: [("tool_name", "missing_effect"), ...]
```

## Effect Scope Management

```python
from toolcase.foundation import EffectScope, effect_scope

# Explicit scope
async with EffectScope(db=db, http=http):
    await run_tools()

# Convenience function
async with effect_scope(db=db):
    await run_tools()

# Testing (resets handlers first)
async with test_effects(db=db):
    await run_tools()  # Clean state guaranteed
```

## Best Practices

1. **Declare all effects**: Make side effects explicit for documentation
2. **Use pure handlers in tests**: Avoid flaky tests and external dependencies
3. **Enable verification in CI**: Catch missing handlers early
4. **Use FrozenTime for dates**: Eliminate time-based test flakiness
5. **Use SeededRandom for randomness**: Make tests reproducible
""",
    keywords=["effects", "side effects", "testing", "mocks", "pure", "handlers", "di"],
)
