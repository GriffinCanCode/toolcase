"""Core middleware types and chain composition.

Middleware follows continuation-passing style: each middleware receives
the tool, params, context, and a `next` function to call downstream.
"""

from __future__ import annotations

from collections.abc import Coroutine, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from typing import Any
    from ..core import BaseTool


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
    
    data: dict[str, object] = field(default_factory=dict)
    
    def __getitem__(self, key: str) -> object:
        return self.data[key]
    
    def __setitem__(self, key: str, value: object) -> None:
        self.data[key] = value
    
    def __contains__(self, key: str) -> bool:
        return key in self.data
    
    def get(self, key: str, default: object = None) -> object:
        return self.data.get(key, default)


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
    wraps each middleware around the base executor.
    
    Args:
        middleware: Ordered list of middleware (first = outermost)
    
    Returns:
        Composed async function: (tool, params, ctx) -> result
    """
    async def base(tool: BaseTool[BaseModel], params: BaseModel, ctx: Context) -> str:
        return await tool.arun(params)
    
    # Build chain by wrapping from innermost to outermost
    chain: Next = base
    for mw in reversed(middleware):
        # Capture mw and current chain in closure
        def make_wrapper(m: Middleware, nxt: Next) -> Next:
            async def wrapped(
                tool: BaseTool[BaseModel],
                params: BaseModel,
                ctx: Context,
            ) -> str:
                return await m(tool, params, ctx, nxt)
            return wrapped
        chain = make_wrapper(mw, chain)
    
    return chain
