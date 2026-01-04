"""Examples of monadic error handling in toolcase.

Demonstrates:
- Railway-oriented programming patterns
- Error context stacking
- Composition of fallible operations
- Integration with existing ToolError system
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from ..core import BaseTool, ToolMetadata
from ..errors import ErrorCode
from .result import Err, Ok, Result, traverse
from .tool import ToolResult, ok_result, tool_result, try_tool_operation
from .types import ErrorContext, ErrorTrace


# ═════════════════════════════════════════════════════════════════════════════
# Example 1: Basic Result Usage
# ═════════════════════════════════════════════════════════════════════════════


def parse_int(s: str) -> Result[int, str]:
    """Parse string to integer with Result."""
    try:
        return Ok(int(s))
    except ValueError:
        return Err(f"Invalid integer: {s}")


def validate_positive(n: int) -> Result[int, str]:
    """Validate number is positive."""
    return Ok(n) if n > 0 else Err(f"Must be positive, got {n}")


def example_basic_railway() -> None:
    """Demonstrate railway-oriented programming."""
    # Success path
    result = (
        parse_int("42")
        .flat_map(validate_positive)
        .map(lambda x: x * 2)
    )
    assert result.unwrap() == 84
    
    # Error path - parse fails
    result = (
        parse_int("not_a_number")
        .flat_map(validate_positive)
        .map(lambda x: x * 2)
    )
    assert result.is_err()
    assert "Invalid integer" in result.unwrap_err()
    
    # Error path - validation fails
    result = (
        parse_int("-5")
        .flat_map(validate_positive)
        .map(lambda x: x * 2)
    )
    assert result.is_err()
    assert "Must be positive" in result.unwrap_err()


# ═════════════════════════════════════════════════════════════════════════════
# Example 2: Error Context Stacking
# ═════════════════════════════════════════════════════════════════════════════


def fetch_user(user_id: int) -> Result[dict, ErrorTrace]:
    """Fetch user from database."""
    if user_id < 1:
        trace = ErrorTrace(
            message="Invalid user ID",
            error_code=ErrorCode.INVALID_PARAMS.value,
            recoverable=False,
        )
        return Err(trace)
    
    # Simulate not found
    if user_id == 999:
        trace = ErrorTrace(
            message=f"User {user_id} not found",
            error_code=ErrorCode.NOT_FOUND.value,
            recoverable=False,
        )
        return Err(trace)
    
    return Ok({"id": user_id, "name": f"User{user_id}"})


def get_user_name(user_id: int) -> Result[str, ErrorTrace]:
    """Get user name with context stacking."""
    return (
        fetch_user(user_id)
        .map_err(lambda err: err.with_operation("fetch_user", location="user.service"))
        .map(lambda user: user["name"])
        .map_err(lambda err: err.with_operation("get_user_name", location="user.api"))
    )


def example_error_context() -> None:
    """Demonstrate error context stacking."""
    # Error case
    result = get_user_name(999)
    assert result.is_err()
    
    trace = result.unwrap_err()
    assert len(trace.contexts) == 2
    assert trace.contexts[0].operation == "fetch_user"
    assert trace.contexts[1].operation == "get_user_name"
    
    # Format shows full trace
    formatted = trace.format()
    assert "fetch_user" in formatted
    assert "get_user_name" in formatted


# ═════════════════════════════════════════════════════════════════════════════
# Example 3: Collection Operations
# ═════════════════════════════════════════════════════════════════════════════


def example_traverse() -> None:
    """Demonstrate traverse for batch operations."""
    # Success case - all parse
    inputs = ["1", "2", "3", "4", "5"]
    result = traverse(inputs, parse_int)
    assert result.unwrap() == [1, 2, 3, 4, 5]
    
    # Failure case - one fails (fail fast)
    inputs = ["1", "bad", "3"]
    result = traverse(inputs, parse_int)
    assert result.is_err()
    assert "Invalid integer: bad" in result.unwrap_err()


# ═════════════════════════════════════════════════════════════════════════════
# Example 4: Tool Integration
# ═════════════════════════════════════════════════════════════════════════════


class ValidatorParams(BaseModel):
    """Parameters for validation tool."""
    numbers: list[str] = Field(..., description="Numbers to validate")


class ValidationTool(BaseTool[ValidatorParams]):
    """Example tool using Result-based error handling."""
    
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="validate_numbers",
        description="Validate and process numbers with monadic error handling",
        category="validation",
    )
    params_schema: ClassVar[type[ValidatorParams]] = ValidatorParams
    
    def _run_result(self, params: ValidatorParams) -> ToolResult:
        """Implementation using Result monad.
        
        This demonstrates railway-oriented programming for tools.
        Errors automatically propagate without manual checking.
        """
        # Parse all numbers
        parse_results = [parse_int(s) for s in params.numbers]
        
        # Sequence into single Result
        from .result import sequence
        parsed = sequence(parse_results)
        
        # Map error to ToolResult format
        if parsed.is_err():
            return tool_result(
                self.metadata.name,
                parsed.unwrap_err(),
                code=ErrorCode.INVALID_PARAMS,
            )
        
        numbers = parsed.unwrap()
        
        # Validate all positive
        validation_results = [validate_positive(n) for n in numbers]
        validated = sequence(validation_results)
        
        if validated.is_err():
            return tool_result(
                self.metadata.name,
                validated.unwrap_err(),
                code=ErrorCode.INVALID_PARAMS,
            )
        
        # Format result
        valid_numbers = validated.unwrap()
        result = f"Validated {len(valid_numbers)} numbers: {valid_numbers}"
        return ok_result(result)
    
    def _run(self, params: ValidatorParams) -> str:
        """Fallback string-based implementation (required by ABC)."""
        # This is now optional - _run_result takes precedence
        result = self._run_result(params)
        from .tool import result_to_string
        return result_to_string(result, self.metadata.name)


# ═════════════════════════════════════════════════════════════════════════════
# Example 5: Railway-Oriented Tool Composition
# ═════════════════════════════════════════════════════════════════════════════


class ProcessorParams(BaseModel):
    """Parameters for processor tool."""
    input_data: str = Field(..., description="Data to process")


class DataProcessorTool(BaseTool[ProcessorParams]):
    """Complex tool demonstrating full railway-oriented patterns."""
    
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="process_data",
        description="Process data through multiple validation steps",
        category="processing",
    )
    params_schema: ClassVar[type[ProcessorParams]] = ProcessorParams
    
    def _validate_input(self, data: str) -> Result[str, ErrorTrace]:
        """Validation step."""
        if not data:
            trace = ErrorTrace(
                message="Input cannot be empty",
                error_code=ErrorCode.INVALID_PARAMS.value,
            )
            return Err(trace.with_operation("validate_input"))
        return Ok(data)
    
    def _normalize(self, data: str) -> Result[str, ErrorTrace]:
        """Normalization step."""
        normalized = data.strip().lower()
        return Ok(normalized)
    
    def _enrich(self, data: str) -> Result[dict, ErrorTrace]:
        """Enrichment step."""
        return Ok({
            "original": data,
            "length": len(data),
            "words": len(data.split()),
        })
    
    def _format_output(self, data: dict) -> str:
        """Format final output."""
        return f"Processed: {data['original']} ({data['words']} words)"
    
    def _run_result(self, params: ProcessorParams) -> ToolResult:
        """Railway-oriented pipeline.
        
        Each step can fail, errors automatically propagate.
        No manual error checking needed!
        """
        return (
            self._validate_input(params.input_data)
            .flat_map(self._normalize)
            .flat_map(self._enrich)
            .map(self._format_output)
            .map_err(lambda trace: trace.with_operation(
                f"tool:{self.metadata.name}",
                location="toolcase.examples"
            ))
        )
    
    def _run(self, params: ProcessorParams) -> str:
        """String-based fallback."""
        result = self._run_result(params)
        from .tool import result_to_string
        return result_to_string(result, self.metadata.name)


# ═════════════════════════════════════════════════════════════════════════════
# Example 6: try_tool_operation Helper
# ═════════════════════════════════════════════════════════════════════════════


class RiskyParams(BaseModel):
    """Parameters for risky operation."""
    value: int = Field(..., description="Input value")


class RiskyTool(BaseTool[RiskyParams]):
    """Tool using try_tool_operation for exception handling."""
    
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="risky_operation",
        description="Operation that might raise exceptions",
        category="demo",
    )
    params_schema: ClassVar[type[RiskyParams]] = RiskyParams
    
    def _run_result(self, params: RiskyParams) -> ToolResult:
        """Use try_tool_operation to handle exceptions."""
        def risky_operation() -> str:
            if params.value < 0:
                raise ValueError("Value must be non-negative")
            if params.value == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            result = 100 / params.value
            return f"Result: {result}"
        
        return try_tool_operation(
            self.metadata.name,
            risky_operation,
            context="processing value"
        )
    
    def _run(self, params: RiskyParams) -> str:
        """String-based fallback."""
        result = self._run_result(params)
        from .tool import result_to_string
        return result_to_string(result, self.metadata.name)


# ═════════════════════════════════════════════════════════════════════════════
# Running Examples
# ═════════════════════════════════════════════════════════════════════════════


def run_all_examples() -> None:
    """Run all examples to demonstrate patterns."""
    print("Example 1: Basic Railway")
    example_basic_railway()
    print("✓ Passed\n")
    
    print("Example 2: Error Context Stacking")
    example_error_context()
    print("✓ Passed\n")
    
    print("Example 3: Collection Operations")
    example_traverse()
    print("✓ Passed\n")
    
    print("Example 4: Tool Integration")
    tool = ValidationTool()
    result = tool.run_result(ValidatorParams(numbers=["1", "2", "3"]))
    assert result.is_ok()
    print(f"Success: {result.unwrap()}")
    print("✓ Passed\n")
    
    print("Example 5: Railway-Oriented Tool")
    processor = DataProcessorTool()
    result = processor.run_result(ProcessorParams(input_data="Hello World"))
    assert result.is_ok()
    print(f"Success: {result.unwrap()}")
    print("✓ Passed\n")
    
    print("Example 6: Exception Handling")
    risky = RiskyTool()
    result = risky.run_result(RiskyParams(value=10))
    assert result.is_ok()
    print(f"Success: {result.unwrap()}")
    print("✓ Passed\n")
    
    print("All examples passed! ✓")


if __name__ == "__main__":
    run_all_examples()
