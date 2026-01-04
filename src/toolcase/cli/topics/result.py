RESULT = """
TOPIC: result
=============

Monadic error handling with Result types (railway-oriented programming).

WHY USE RESULT:
    - Type-safe: compiler knows when operations can fail
    - Automatic error propagation through chains
    - No manual error string checking
    - Error context stacking for debugging

BASIC USAGE:
    from toolcase import Result, Ok, Err
    
    def divide(a: int, b: int) -> Result[float, str]:
        if b == 0:
            return Err("division by zero")
        return Ok(a / b)
    
    result = divide(10, 2)
    if result.is_ok():
        print(result.unwrap())  # 5.0

RAILWAY PATTERN (Chaining):
    from toolcase import Ok
    
    result = (
        Ok("42")
        .flat_map(parse_int)         # Parse
        .flat_map(validate_positive) # Validate
        .map(lambda x: x * 2)        # Transform
    )

TOOL INTEGRATION:
    from toolcase import BaseTool, ToolResult, Ok, tool_result
    
    class MyTool(BaseTool[MyParams]):
        def _run_result(self, params: MyParams) -> ToolResult:
            return (
                self._validate(params)
                .flat_map(lambda p: self._fetch(p))
                .map(lambda d: self._format(d))
            )

KEY OPERATIONS:
    map(fn)           Transform Ok value
    flat_map(fn)      Chain operations that return Result
    map_err(fn)       Transform Err value
    bimap(ok_fn, err_fn)  Transform both variants
    unwrap()          Get Ok value (raises on Err)
    unwrap_or(val)    Get Ok value or default
    unwrap_or_else(fn)  Get Ok or compute from Err
    or_else(fn)       Try alternative on Err
    and_then(fn)      Alias for flat_map
    match(ok, err)    Pattern match both cases
    inspect(fn)       Side effects on Ok, return self
    inspect_err(fn)   Side effects on Err, return self

COLLECTION OPERATIONS:
    from toolcase import sequence, traverse, collect_results
    
    # Sequence: fail-fast, first Err stops
    results = [Ok(1), Ok(2), Ok(3)]
    combined = sequence(results)  # Ok([1, 2, 3])
    
    # Traverse: map + sequence
    combined = traverse(items, lambda x: validate(x))
    
    # Collect: accumulate ALL errors (not fail-fast)
    combined = collect_results(results)  # Collects all errors

EXCEPTION HANDLING:
    from toolcase import try_tool_operation
    
    result = try_tool_operation(
        "my_tool",
        lambda: risky_api_call(),
        context="calling external API"
    )

RELATED TOPICS:
    toolcase help tool         Tool creation
    toolcase help errors       Error codes and ToolError
"""
