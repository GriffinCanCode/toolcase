## Monadic Error Handling

Type-safe error propagation with Haskell-grade rigor for toolcase. Implements Result/Either types with full monadic operations for railway-oriented programming.

### Design Principles

1. **Type Safety First**: Errors are values, checked at compile time
2. **Railway-Oriented Programming**: Automatic error propagation through chains
3. **Zero Runtime Overhead**: Uses `__slots__` and immutable structures
4. **Context Preservation**: Error traces accumulate through call stacks
5. **Backwards Compatible**: Integrates seamlessly with existing `ToolError` system

### Core Types

#### `Result[T, E]`

Discriminated union representing success (`Ok`) or failure (`Err`).

```python
from toolcase import Result, Ok, Err

def divide(a: int, b: int) -> Result[float, str]:
    if b == 0:
        return Err("division by zero")
    return Ok(a / b)

result = divide(10, 2)
if result.is_ok():
    print(f"Success: {result.unwrap()}")
else:
    print(f"Error: {result.unwrap_err()}")
```

#### `ToolResult`

Type alias for tool operations: `Result[str, ErrorTrace]`

```python
from toolcase import ToolResult, Ok, tool_result, ErrorCode

def my_tool_operation() -> ToolResult:
    if success:
        return Ok("Operation succeeded")
    return tool_result(
        "my_tool",
        "Operation failed",
        code=ErrorCode.NETWORK_ERROR,
        recoverable=True
    )
```

#### `ErrorTrace`

Error context with provenance tracking through call chains.

```python
from toolcase import ErrorTrace, ErrorContext

trace = ErrorTrace(
    message="Connection failed",
    error_code="NETWORK_ERROR",
    recoverable=True
)

# Add context as error propagates
trace = trace.with_operation("fetch_data", location="api.client")
trace = trace.with_operation("process_request", location="handlers")

# Format shows full trace
print(trace.format())
# Output:
# Connection failed
# [NETWORK_ERROR]
# 
# Context trace:
#   - fetch_data at api.client
#   - process_request at handlers
```

### Monadic Operations

#### Functor: `map`, `map_err`

Transform values inside Result without unwrapping:

```python
result = Ok(5).map(lambda x: x * 2)  # Ok(10)
result = Err("fail").map(lambda x: x * 2)  # Err("fail") - unchanged
```

#### Monad: `flat_map` (bind)

Chain operations that can fail (railway-oriented programming):

```python
def parse_int(s: str) -> Result[int, str]:
    try:
        return Ok(int(s))
    except ValueError:
        return Err(f"invalid: {s}")

def validate_positive(n: int) -> Result[int, str]:
    return Ok(n) if n > 0 else Err("must be positive")

result = (
    Ok("42")
    .flat_map(parse_int)           # Parse string
    .flat_map(validate_positive)   # Validate value
    .map(lambda x: x * 2)           # Transform if valid
)
# result: Ok(84)
```

Error automatically propagates:

```python
result = (
    Ok("bad")
    .flat_map(parse_int)           # Fails here
    .flat_map(validate_positive)   # Skipped
    .map(lambda x: x * 2)           # Skipped
)
# result: Err("invalid: bad")
```

#### Bifunctor: `bimap`

Map both Ok and Err values:

```python
result = Ok(5).bimap(
    ok_fn=lambda x: x * 2,
    err_fn=lambda e: f"Error: {e}"
)
# Ok(10)
```

#### Pattern Matching: `match`

Exhaustive case analysis:

```python
output = result.match(
    ok=lambda value: f"Success: {value}",
    err=lambda error: f"Failed: {error}"
)
```

### Collection Operations

#### `sequence`: Convert `[Result[T, E]]` → `Result[[T], E]`

```python
from toolcase import sequence

results = [Ok(1), Ok(2), Ok(3)]
sequence(results)  # Ok([1, 2, 3])

results = [Ok(1), Err("fail"), Ok(3)]
sequence(results)  # Err("fail") - fail fast
```

#### `traverse`: Map + sequence

```python
from toolcase import traverse

def parse_int(s: str) -> Result[int, str]:
    try:
        return Ok(int(s))
    except ValueError:
        return Err(f"invalid: {s}")

traverse(["1", "2", "3"], parse_int)  # Ok([1, 2, 3])
traverse(["1", "bad", "3"], parse_int)  # Err("invalid: bad")
```

#### `collect_results`: Accumulate all errors

```python
from toolcase import collect_results

results = [Ok(1), Err("e1"), Ok(3), Err("e2")]
collect_results(results)  # Err(["e1", "e2"])
```

### Tool Integration

#### Method 1: Override `_run_result`

```python
from toolcase import BaseTool, ToolResult, Ok, tool_result

class MyTool(BaseTool[MyParams]):
    def _run_result(self, params: MyParams) -> ToolResult:
        """Type-safe implementation using Result."""
        return (
            self._validate(params)
            .flat_map(lambda p: self._fetch_data(p))
            .map(lambda data: self._format(data))
        )
    
    def _run(self, params: MyParams) -> str:
        """Required but can delegate to _run_result."""
        from toolcase.monads.tool import result_to_string
        result = self._run_result(params)
        return result_to_string(result, self.metadata.name)
```

#### Method 2: Use `try_tool_operation`

Automatic exception handling with Result:

```python
from toolcase import try_tool_operation

def _run_result(self, params: MyParams) -> ToolResult:
    def risky_operation() -> str:
        # Might raise exceptions
        return external_api_call(params)
    
    return try_tool_operation(
        self.metadata.name,
        risky_operation,
        context="fetching data"
    )
```

#### Method 3: Use `run_result` method

Call tools with Result return type:

```python
from toolcase import get_registry

registry = get_registry()
tool = registry["my_tool"]

# Returns Result[str, ErrorTrace]
result = tool.run_result(params)

# Compose with other operations
output = (
    result
    .map(lambda s: s.upper())
    .map(lambda s: f"Result: {s}")
    .unwrap_or("default")
)
```

### Railway-Oriented Programming Pattern

Classic example - validate and process user input:

```python
from toolcase import Result, Ok, Err, ToolResult, tool_result

def validate_email(email: str) -> Result[str, str]:
    if "@" not in email:
        return Err("Invalid email format")
    return Ok(email)

def normalize_email(email: str) -> Result[str, str]:
    return Ok(email.lower().strip())

def check_blacklist(email: str) -> Result[str, str]:
    blacklist = ["spam@example.com"]
    if email in blacklist:
        return Err(f"Email {email} is blacklisted")
    return Ok(email)

def send_verification(email: str) -> Result[str, str]:
    # Simulate sending
    return Ok(f"Verification sent to {email}")

# Compose the pipeline
def process_signup(email: str) -> Result[str, str]:
    return (
        validate_email(email)
        .flat_map(normalize_email)
        .flat_map(check_blacklist)
        .flat_map(send_verification)
    )

# Success path
result = process_signup("User@Example.com")
# Ok("Verification sent to user@example.com")

# Error path - stops at first failure
result = process_signup("invalid-email")
# Err("Invalid email format")
```

### Error Context Stacking

Track error provenance through nested calls:

```python
from toolcase import ErrorTrace, tool_result

def inner_operation() -> Result[str, ErrorTrace]:
    trace = ErrorTrace(message="Database connection failed")
    return Err(trace)

def middle_operation() -> Result[str, ErrorTrace]:
    return inner_operation().map_err(
        lambda trace: trace.with_operation("fetch_user", location="db.users")
    )

def outer_operation() -> Result[str, ErrorTrace]:
    return middle_operation().map_err(
        lambda trace: trace.with_operation("process_request", location="api.handlers")
    )

result = outer_operation()
# Error trace shows:
# Database connection failed
#
# Context trace:
#   - fetch_user at db.users
#   - process_request at api.handlers
```

### Advanced Patterns

#### Parallel Validation with Applicative

```python
from toolcase import Result, Ok, Err

def validate_name(name: str) -> Result[str, str]:
    return Ok(name) if name else Err("Name required")

def validate_age(age: int) -> Result[int, str]:
    return Ok(age) if 0 <= age <= 120 else Err("Invalid age")

# Both validations run
name_result = validate_name("")
age_result = validate_age(150)

# Accumulate errors (would need custom combinator for full applicative)
```

#### Recovery with `or_else`

```python
def fetch_from_primary() -> Result[str, str]:
    return Err("Primary unavailable")

def fetch_from_backup() -> Result[str, str]:
    return Ok("Data from backup")

result = fetch_from_primary().or_else(lambda _: fetch_from_backup())
# Ok("Data from backup")
```

#### Inspection for Logging

```python
result = (
    fetch_data()
    .inspect(lambda data: logger.info(f"Fetched: {data}"))
    .inspect_err(lambda err: logger.error(f"Failed: {err}"))
    .map(process_data)
)
```

### Migration from String-Based Errors

#### Before (String-Based)

```python
def _run(self, params: MyParams) -> str:
    try:
        data = fetch_data(params)
        return format_result(data)
    except Exception as e:
        return self._error_from_exception(e, "Failed to fetch")
```

#### After (Result-Based)

```python
def _run_result(self, params: MyParams) -> ToolResult:
    return try_tool_operation(
        self.metadata.name,
        lambda: format_result(fetch_data(params)),
        context="fetching data"
    )
```

Or with explicit pipeline:

```python
def _run_result(self, params: MyParams) -> ToolResult:
    return (
        self._fetch_data(params)
        .flat_map(lambda data: self._validate_data(data))
        .map(lambda data: self._format_result(data))
    )
```

### Type Signatures (Haskell-Style)

For reference, here are the type signatures in Haskell notation:

```haskell
-- Constructors
Ok    :: a -> Result a e
Err   :: e -> Result a e

-- Functor
map   :: Result a e -> (a -> b) -> Result b e
fmap  = map

-- Applicative
pure  :: a -> Result a e
apply :: Result (a -> b) e -> Result a e -> Result b e

-- Monad
return  :: a -> Result a e
flatMap :: Result a e -> (a -> Result b e) -> Result b e
(>>=)   = flatMap

-- Bifunctor
bimap :: Result a e -> (a -> b) -> (e -> f) -> Result b f

-- Collection
sequence :: [Result a e] -> Result [a] e
traverse :: [a] -> (a -> Result b e) -> Result [b] e
```

### Performance Notes

- **Zero Overhead**: Uses `__slots__` for memory efficiency
- **Immutable**: All operations return new Results (no mutation)
- **Lazy Evaluation**: Short-circuits on first error
- **Stack Safe**: No recursion in core operations

### Integration with Existing Code

The monadic system integrates seamlessly:

1. **Tools can use either pattern**: Override `_run` (string) or `_run_result` (Result)
2. **Automatic conversion**: `result_to_string` and `string_to_result` bridge the gap
3. **Registry unchanged**: Tools work with existing registry system
4. **Cache compatible**: Result-based execution respects caching
5. **LangChain works**: Conversion happens transparently

### Best Practices

1. **Use `flat_map` for chaining**: Don't nest Results manually
2. **Add context at boundaries**: Track error provenance at module boundaries
3. **Fail fast with `sequence`**: Use for independent validations
4. **Accumulate with `collect_results`**: Use when you need all errors
5. **Prefer `try_tool_operation`**: For automatic exception handling
6. **Use `match` for exhaustive handling**: Forces handling both cases

### See Also

- [Railway-Oriented Programming](https://fsharpforfunandprofit.com/rop/) by Scott Wlaschin
- [Rust Result Documentation](https://doc.rust-lang.org/std/result/)
- [Haskell Either](https://hackage.haskell.org/package/base/docs/Data-Either.html)
