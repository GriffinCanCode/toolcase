EFFECTS = """
TOPIC: effects
==============

Effect system for side-effect tracking and pure testing.

CONCEPT:
    The effect system enables tools to explicitly declare their side effects
    (db, http, file, etc.), providing:
    - Compile-time-style verification (ensure handlers exist before execution)
    - Testing without mocks (swap real implementations with pure handlers)
    - Explicit documentation (make side effects visible in tool signatures)

DECLARING EFFECTS:

Via @tool decorator:
    from toolcase import tool
    
    @tool(description="Fetch user from database", effects=["db", "cache"])
    async def fetch_user(user_id: str, db: Database) -> str:
        user = await db.fetch_one("SELECT * FROM users WHERE id = $1", user_id)
        return f"User: {user['name']}"

Via @effects decorator:
    from toolcase.foundation.effects import effects
    
    @effects("db", "http")
    async def fetch_and_enrich(user_id: str) -> str:
        # Function with db and http side effects
        ...

STANDARD EFFECTS:
    db       Database operations (read/write)
    http     Network requests to external services
    file     File system operations
    cache    Caching layer interactions
    env      Environment variable access
    time     Time-dependent operations
    random   Non-deterministic random operations
    log      Logging side effects

PURE HANDLERS FOR TESTING:
    from toolcase.foundation.effects import (
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

InMemoryDB:
    db = InMemoryDB()
    db.set_response("SELECT", [{"id": 1}])   # Pattern-based responses
    db.set_default(None)                      # Default for unmatched
    
    await db.fetch_one("SELECT * FROM users")
    await db.fetch_all("SELECT * FROM items")
    await db.execute("DELETE FROM logs")
    
    db.queries       # List of executed queries
    db.records       # Full DBRecord objects

RecordingHTTP:
    http = RecordingHTTP()
    http.mock_response("api.example.com/users", {"users": []})
    http.mock_error("api.example.com/fail", 500)
    
    status, body = await http.get("https://api.example.com/users")
    
    http.urls        # List of requested URLs
    http.requests    # Full HTTPRequest objects

InMemoryFS:
    fs = InMemoryFS()
    fs.seed("/config.json", '{"key": "value"}')
    
    content = await fs.read_text("/config.json")
    await fs.write("/output.txt", "result")
    exists = await fs.exists("/config.json")
    
    fs.files         # Dict of path -> bytes
    fs.operations    # List of FSOperation records

FrozenTime:
    from datetime import datetime, timezone
    
    frozen = FrozenTime()
    frozen.freeze(datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    
    t1 = frozen.now()      # Always returns frozen time
    frozen.advance(3600)   # Advance 1 hour
    t2 = frozen.now()      # Returns 13:00:00

SeededRandom:
    rng = SeededRandom(_seed=42)
    rng.random()           # Reproducible float
    rng.randint(1, 10)     # Reproducible int
    rng.choice([1,2,3])    # Reproducible choice
    rng.reseed(42)         # Reset sequence

CollectingLogger:
    logger = CollectingLogger()
    logger.info("Processing", item_id=123)
    logger.error("Failed")
    
    logger.messages  # ["Processing", "Failed"]
    logger.entries   # Full LogEntry objects with kwargs

EFFECT VERIFICATION:
    from toolcase.foundation.effects import verify_effects, MissingEffectHandler
    
    # Verify at registration time
    violations = verify_effects(my_tool, handlers)
    if violations:
        for v in violations:
            print(f"Missing: {v.effect} for {v.tool_name}")
    
    # Strict mode raises on first violation
    verify_effects(my_tool, handlers, strict=True)

EFFECT SCOPE MANAGEMENT:
    from toolcase.foundation.effects import EffectScope, effect_scope
    
    # Explicit scope
    async with EffectScope(db=db, http=http):
        await run_tools()
    
    # Convenience function
    async with effect_scope(db=db):
        await run_tools()
    
    # Testing (resets handlers first)
    async with test_effects(db=db):
        await run_tools()  # Clean state guaranteed

QUERYING EFFECTS:
    from toolcase.foundation.effects import get_effects, has_effects
    
    # Get declared effects from tool
    effects = get_effects(my_tool)  # frozenset["db", "http"]
    
    # Check if tool has specific effects
    if has_effects(my_tool, "db", "http"):
        print("Tool uses db and http")

BEST PRACTICES:
    1. Declare all effects: Make side effects explicit for documentation
    2. Use pure handlers in tests: Avoid flaky tests and external dependencies
    3. Enable verification in CI: Catch missing handlers early
    4. Use FrozenTime for dates: Eliminate time-based test flakiness
    5. Use SeededRandom for randomness: Make tests reproducible

RELATED TOPICS:
    toolcase help testing    Testing utilities
    toolcase help di         Dependency injection
    toolcase help tool       Tool creation with effects
"""
