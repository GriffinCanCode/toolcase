VALIDATION = """
TOPIC: validation
=================

Composable validation DSL for type-safe parameter constraints.

CONCEPT:
    The Rule DSL provides algebraic validation primitives that compose
    via combinators (&, |, ~) to build complex, reusable constraints.
    
    Design principles:
    - Rules are monoids (associative, have identity)
    - Parser combinator style composition
    - Railway-oriented error accumulation
    - Cross-field and conditional validation

ATOMIC RULES:
    from toolcase.runtime.middleware.plugins import (
        required, optional, is_type, is_str, is_int, is_float, is_bool,
        rule_in_range, min_len, max_len, length, rule_matches, rule_one_of,
        url, https, email, predicate
    )
    
    # Presence
    required()              # Value must exist and be non-empty
    optional()              # Documentation rule (always passes)
    
    # Type checking
    is_type(int, float)     # Value is one of types
    is_str(), is_int()      # Shorthand for common types
    
    # Constraints
    rule_in_range(0, 100)   # Numeric range [low, high]
    min_len(3)              # Minimum length
    max_len(50)             # Maximum length
    length(10)              # Exact length
    
    # Patterns
    rule_matches(r"^[a-z]+$", "lowercase only")
    rule_one_of("A", "B", "C")
    url(), https(), email()
    
    # Custom
    predicate(lambda v: v > 0, "must be positive")

COMBINATORS:
    # All must pass (AND) - accumulates violations
    required() & is_str() & min_len(3)
    all_of(rule1, rule2, rule3)
    
    # At least one passes (OR) - short-circuits
    is_int() | is_str()
    any_of(rule1, rule2, message="must match one")
    
    # Exactly one passes (XOR)
    xor_of(is_str(), is_int())
    
    # Negation (NOT)
    ~rule_one_of("admin", "root")    # Operator syntax
    rule_not(rule, message="...")     # Factory syntax

CONDITIONAL VALIDATION:
    from toolcase.runtime.middleware.plugins import (
        when, unless, when_eq, when_present, when_absent
    )
    
    # Rule applies only when condition is True
    when(when_eq("method", "POST"), required())
    
    # Rule applies only when condition is False
    unless(when_present("email"), required())
    
    # Fluent method syntax
    required().when(when_eq("method", "POST"))
    required().unless(when_present("email"))
    
    # Condition factories
    when_eq("field", "value")    # field == value
    when_present("field")        # field is not None
    when_absent("field")         # field is None

CROSS-FIELD CONSTRAINTS:
    from toolcase.runtime.middleware.plugins import (
        less_than, less_than_or_eq, equals, mutex, together,
        at_least_one, depends_on, cross
    )
    
    # Comparison
    less_than("start", "end")         # start < end
    less_than_or_eq("min", "max")     # min <= max
    equals("password", "confirm")      # password == confirm
    
    # Exclusivity
    mutex("sync", "async_mode")        # Only one can be set
    together("start", "end")           # All present or all absent
    at_least_one("email", "phone")     # At least one required
    depends_on("body", "method")       # body requires method
    
    # Custom
    cross(
        lambda p: p.start < p.end or "invalid range",
        "start", "end",
        message="date validation"
    )

SCHEMA BUILDER:
    from toolcase.runtime.middleware.plugins import Schema
    
    schema = (
        Schema("http_request")
        .field("url", required() & url() & https())
        .field("method", required() & rule_one_of("GET", "POST", "PUT"))
        .field("timeout", optional() & rule_in_range(1, 60))
        .field("body", when(when_eq("method", "POST"), required()))
        .cross(mutex("sync", "async_mode"))
    )
    
    # Validate params
    result = schema.validate(params)
    if not result.is_valid:
        for violation in result:
            print(f"  • {violation}")

MIDDLEWARE INTEGRATION:
    from toolcase import get_registry
    from toolcase.runtime.middleware.plugins import (
        ValidationMiddleware, Schema, required, https, url
    )
    
    # Method 1: Add pre-built schema
    middleware = ValidationMiddleware()
    schema = Schema("my_tool").field("url", required() & https())
    middleware.add_schema(schema)
    
    # Method 2: Build in-place
    middleware.schema("my_tool").field("url", required() & url())
    
    # Register with registry
    registry = get_registry()
    registry.use(middleware)

MESSAGE OVERRIDES:
    # Override error message
    rule = required().with_message("this field is mandatory")
    
    # Override error code
    rule = required().with_code("MISSING_REQUIRED")

VALIDATION RESULT:
    result = schema.validate(params)
    
    result.is_valid              # bool
    bool(result)                 # Same as is_valid
    result.violations            # tuple[Violation, ...]
    result.format()              # Human-readable string
    
    # Concatenation (monoid)
    combined = result1 + result2  # Accumulates violations

EXAMPLE - API REQUEST VALIDATION:
    schema = (
        Schema("api_call")
        .field("endpoint", required() & url() & https())
        .field("method", required() & rule_one_of("GET", "POST", "PUT", "DELETE"))
        .field("headers", optional() & is_dict())
        .field("body", when(when_eq("method", "POST"), required() & is_dict()))
        .field("timeout", optional() & rule_in_range(1, 300))
        .field("retry_count", optional() & is_int() & rule_in_range(0, 10))
        .cross(mutex("auth_token", "api_key"))
        .cross(depends_on("auth_header", "auth_token"))
    )

LEGACY API:
    The add_rule() and add_constraint() methods still work:
    
    validation.add_rule("search", "query", lambda q: len(q) >= 3, "too short")
    validation.add_constraint("report", lambda p: p.start <= p.end or "bad range")
    
    Both APIs can be combined; schemas run before legacy rules.

RELATED TOPICS:
    toolcase help middleware  Middleware chain composition
    toolcase help errors      Error handling and Result type
    toolcase help testing     Testing tools with mock validation
"""
