"""Type-safe signal/event system for observer pattern.

Provides a lightweight, generic event emitter supporting both sync and async
handlers with optional weak references and priority ordering.

Example:
    >>> from toolcase.foundation.events import Signal
    >>> 
    >>> on_data: Signal[Callable[[str, int], None]] = Signal()
    >>> 
    >>> def handler(name: str, value: int) -> None:
    ...     print(f"{name}={value}")
    >>> 
    >>> on_data += handler  # Subscribe
    >>> on_data.fire("count", 42)  # Fires: count=42
    >>> on_data -= handler  # Unsubscribe

Async Support:
    >>> async def async_handler(data: dict) -> None:
    ...     await process(data)
    >>> 
    >>> on_event: Signal[Callable[[dict], Awaitable[None] | None]] = Signal()
    >>> on_event += async_handler
    >>> await on_event.fire_async({"key": "value"})

Priority Ordering:
    >>> signal = Signal()
    >>> signal.subscribe(first_handler, priority=10)  # Runs first
    >>> signal.subscribe(second_handler, priority=0)  # Runs second
"""

from __future__ import annotations

import asyncio
import inspect
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Generic, ParamSpec, TypeVar, overload

__all__ = [
    "Signal",
    "SignalHandler",
    "one_shot",
]

P = ParamSpec("P")
T = TypeVar("T")
HandlerT = TypeVar("HandlerT", bound=Callable[..., object])


@dataclass(slots=True, order=True)
class _HandlerEntry(Generic[HandlerT]):
    """Internal handler entry with priority for ordering."""
    priority: int
    handler: HandlerT | weakref.ref[HandlerT] = field(compare=False)
    once: bool = field(default=False, compare=False)
    weak: bool = field(default=False, compare=False)
    
    def resolve(self) -> HandlerT | None:
        """Resolve handler, returning None if weak ref expired."""
        if self.weak and isinstance(self.handler, weakref.ref):
            return self.handler()
        return self.handler  # type: ignore[return-value]


class Signal(Generic[HandlerT]):
    """Type-safe signal/event emitter for observer pattern.
    
    Supports both sync and async handlers with optional weak references
    to prevent memory leaks and priority ordering for execution order.
    
    Type Parameters:
        HandlerT: Callable signature for handlers (e.g., Callable[[str], None])
    
    Example:
        >>> on_register: Signal[Callable[[AnyTool], None]] = Signal()
        >>> 
        >>> def log_registration(tool: AnyTool) -> None:
        ...     print(f"Registered: {tool.metadata.name}")
        >>> 
        >>> on_register += log_registration
        >>> on_register.fire(my_tool)  # Prints: Registered: my_tool
    """
    
    __slots__ = ("_entries", "_firing")
    
    def __init__(self) -> None:
        self._entries: list[_HandlerEntry[HandlerT]] = []
        self._firing = False  # Guard against modification during fire
    
    @property
    def handler_count(self) -> int:
        """Number of subscribed handlers (excluding expired weak refs)."""
        return sum(1 for e in self._entries if e.resolve() is not None)
    
    @property
    def is_empty(self) -> bool:
        """Whether signal has no handlers."""
        return self.handler_count == 0
    
    def subscribe(
        self,
        handler: HandlerT,
        *,
        priority: int = 0,
        once: bool = False,
        weak: bool = False,
    ) -> HandlerT:
        """Subscribe a handler to this signal.
        
        Args:
            handler: Callable to invoke when signal fires
            priority: Higher values fire first (default: 0)
            once: Remove handler after first fire
            weak: Store as weak reference (auto-cleanup when handler gc'd)
        
        Returns:
            The handler (for decorator use)
        
        Example:
            >>> @signal.subscribe
            ... def handler(data): ...
            >>> 
            >>> signal.subscribe(handler, priority=10, once=True)
        """
        ref = weakref.ref(handler) if weak else handler
        entry = _HandlerEntry(priority=-priority, handler=ref, once=once, weak=weak)  # Negate for descending sort
        self._entries.append(entry)
        self._entries.sort()  # Maintain priority order
        return handler
    
    def unsubscribe(self, handler: HandlerT) -> bool:
        """Remove a handler. Returns True if found and removed."""
        for i, entry in enumerate(self._entries):
            resolved = entry.resolve()
            if resolved is handler or (entry.weak and resolved is None):
                del self._entries[i]
                return resolved is handler
        return False
    
    def once(self, handler: HandlerT, *, priority: int = 0) -> HandlerT:
        """Subscribe handler for a single fire only.
        
        Convenience method equivalent to subscribe(..., once=True).
        """
        return self.subscribe(handler, priority=priority, once=True)
    
    def clear(self) -> int:
        """Remove all handlers. Returns count of handlers removed."""
        count = len(self._entries)
        self._entries.clear()
        return count
    
    def fire(self, *args: P.args, **kwargs: P.kwargs) -> None:  # type: ignore[name-defined]
        """Fire signal synchronously, calling all handlers.
        
        Async handlers are scheduled but not awaited. Use fire_async()
        to properly await async handlers.
        
        Handlers are called in priority order (highest first).
        One-shot handlers are removed after calling.
        """
        self._firing = True
        to_remove: list[int] = []
        
        try:
            for i, entry in enumerate(self._entries):
                if (handler := entry.resolve()) is None:
                    to_remove.append(i)
                    continue
                
                result = handler(*args, **kwargs)
                
                # Schedule async handlers (don't await)
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        # No running loop - close the coroutine to avoid warning
                        result.close()
                
                if entry.once:
                    to_remove.append(i)
        finally:
            self._firing = False
            # Remove in reverse to preserve indices
            for i in reversed(to_remove):
                del self._entries[i]
    
    async def fire_async(self, *args: P.args, **kwargs: P.kwargs) -> None:  # type: ignore[name-defined]
        """Fire signal async, awaiting all handlers concurrently.
        
        Sync handlers are called directly. Async handlers are gathered
        and awaited together for efficiency.
        
        Handlers are called in priority order (highest first).
        One-shot handlers are removed after calling.
        """
        self._firing = True
        to_remove: list[int] = []
        awaitables: list[Awaitable[object]] = []
        
        try:
            for i, entry in enumerate(self._entries):
                if (handler := entry.resolve()) is None:
                    to_remove.append(i)
                    continue
                
                result = handler(*args, **kwargs)
                
                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    awaitables.append(result)
                elif inspect.isawaitable(result):
                    awaitables.append(result)
                
                if entry.once:
                    to_remove.append(i)
            
            # Await all async handlers concurrently
            if awaitables:
                await asyncio.gather(*awaitables, return_exceptions=True)
        finally:
            self._firing = False
            for i in reversed(to_remove):
                del self._entries[i]
    
    def __iadd__(self, handler: HandlerT) -> Signal[HandlerT]:
        """Subscribe via += operator."""
        self.subscribe(handler)
        return self
    
    def __isub__(self, handler: HandlerT) -> Signal[HandlerT]:
        """Unsubscribe via -= operator."""
        self.unsubscribe(handler)
        return self
    
    def __len__(self) -> int:
        """Number of handlers (including potentially expired weak refs)."""
        return len(self._entries)
    
    def __bool__(self) -> bool:
        """True if signal has any handlers."""
        return not self.is_empty
    
    def __contains__(self, handler: HandlerT) -> bool:
        """Check if handler is subscribed."""
        return any(e.resolve() is handler for e in self._entries)
    
    def __repr__(self) -> str:
        return f"Signal(handlers={self.handler_count})"


# Convenience type alias
SignalHandler = Callable[P, T]  # type: ignore[valid-type]


def one_shot(signal: Signal[HandlerT]) -> Callable[[HandlerT], HandlerT]:
    """Decorator to subscribe a handler for single use.
    
    Example:
        >>> @one_shot(on_ready)
        ... def handle_ready(data: str) -> None:
        ...     print(f"Ready: {data}")
    """
    def decorator(handler: HandlerT) -> HandlerT:
        signal.once(handler)
        return handler
    return decorator
