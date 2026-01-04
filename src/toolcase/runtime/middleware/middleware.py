"""Core middleware types and chain composition.

Middleware follows continuation-passing style: each middleware receives
the tool, params, context, and a `next` function to call downstream.
"""

from __future__ import annotations

from collections.abc import Coroutine, Sequence
from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Callable, Protocol, TypeVar, cast, overload, runtime_checkable

from pydantic import BaseModel

from toolcase.foundation.core.decorator import clear_injected_deps, set_injected_deps
from toolcase.foundation.errors import JsonDict, ToolError, ToolException

if TYPE_CHECKING:
    from typing import Any
    from toolcase.foundation.core import BaseTool

# TypeVar for Context.get() default value
_T = TypeVar("_T")
# Value types typically stored in context
ContextValue = str | int | float | bool | dict[str, "ContextValue"] | list["ContextValue"] | None


@dataclass(slots=True)
class Context:
    """Execution context passed through the middleware chain.
    
    Carries request-scoped state between middleware. Use for:
    - Timing data (start_time)
    - Request IDs for correlation
    - User/auth info
    - Custom middleware state
    
    Example:
        >>> ctx = Context()
        >>> ctx["request_id"] = "abc123"
        >>> ctx.get("request_id")
        'abc123'
    """
    
    data: dict[str, ContextValue] = field(default_factory=dict)
    
    def __getitem__(self, key: str) -> ContextValue: return self.data[key]
    def __setitem__(self, key: str, value: ContextValue) -> None: self.data[key] = value
    def __contains__(self, key: str) -> bool: return key in self.data
    
    @overload
    def get(self, key: str) -> ContextValue: ...
    @overload
    def get(self, key: str, default: _T) -> ContextValue | _T: ...
    def get(self, key: str, default: _T | None = None) -> ContextValue | _T | None: return self.data.get(key, default)


# Type alias for the continuation function
Next = Callable[["BaseTool[BaseModel]", BaseModel, Context], "Coroutine[Any, Any, str]"]


@runtime_checkable
class Middleware(Protocol):
    """Protocol for tool middleware.
    
    Middleware intercepts tool execution for cross-cutting concerns.
    Implement `__call__` to wrap execution with custom logic.
    
    Example:
        >>> class TimingMiddleware:
        ...     async def __call__(self, tool, params, ctx, next):
        ...         start = time.time()
        ...         result = await next(tool, params, ctx)
        ...         ctx["duration"] = time.time() - start
        ...         return result
    """
    
    async def __call__(
        self,
        tool: BaseTool[BaseModel],
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        """Execute middleware logic.
        
        Args:
            tool: The tool being executed
            params: Validated parameters
            ctx: Request-scoped context for sharing state
            next: Continuation to call downstream chain
        
        Returns:
            Tool result (possibly modified)
        """
        ...


def compose(middleware: Sequence[Middleware]) -> Next:
    """Compose middleware into a single execution function.
    
    Uses functional composition via iteration. The resulting function
    wraps each middleware around the base executor with error handling.
    
    Handles dependency injection by setting context-var deps from context["injected"]
    before tool execution.
    
    Args:
        middleware: Ordered list of middleware (first = outermost)
    
    Returns:
        Composed async function: (tool, params, ctx) -> result
    """
    async def base(tool: BaseTool[BaseModel], params: BaseModel, ctx: Context) -> str:
        try:
            # Set injected deps from context if present (cast: injected can be any object)
            if (injected := ctx.get("injected")) and isinstance(injected, dict):
                set_injected_deps(cast("JsonDict", injected))
            try:
                return await tool.arun(params)
            finally:
                clear_injected_deps()
        except ToolException as e:
            return e.error.render()
        except Exception as e:
            return ToolError.from_exception(tool.metadata.name, e, "Execution failed").render()
    
    # Build chain: wrap each middleware around the current chain (innermost to outermost)
    def wrap(nxt: Next, mw: Middleware) -> Next:
        async def wrapped(tool: BaseTool[BaseModel], params: BaseModel, ctx: Context) -> str:
            return await mw(tool, params, ctx, nxt)
        return wrapped
    
    return reduce(wrap, reversed(middleware), base)
