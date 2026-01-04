"""Monadic error handling with Haskell-grade rigor.

Provides Result/Either types for type-safe error propagation with:
- Railway-oriented programming patterns
- Functor/Applicative/Monad instances
- Error context stacking and provenance tracking
- Zero-cost abstractions via slots and frozen structures
- Full integration with existing ToolError system

Example:
    >>> from toolcase.monads import Result, Ok, Err
    >>>
    >>> def divide(a: int, b: int) -> Result[float, str]:
    ...     if b == 0:
    ...         return Err("division by zero")
    ...     return Ok(a / b)
    >>>
    >>> result = (
    ...     divide(10, 2)
    ...     .map(lambda x: x * 2)
    ...     .flat_map(lambda x: Ok(x + 1))
    ... )
    >>> assert result.unwrap() == 11.0
"""

from .result import (
    Err,
    Ok,
    Result,
    collect_results,
    sequence,
    traverse,
)
from .tool import ToolResult, batch_results, tool_result, try_tool_operation, try_tool_operation_async
from .types import (
    ErrorContext,
    ErrorTrace,
    ResultT,
)

__all__ = [
    # Core types
    "Result",
    "Ok",
    "Err",
    # Type aliases
    "ResultT",
    # Tool integration
    "ToolResult",
    "tool_result",
    "try_tool_operation",
    "try_tool_operation_async",
    "batch_results",
    # Error context
    "ErrorContext",
    "ErrorTrace",
    # Collection operations
    "sequence",
    "traverse",
    "collect_results",
]
