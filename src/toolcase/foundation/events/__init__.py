"""Event System - Type-safe signals for observer pattern.

Provides a lightweight event emitter with:
- Generic typing for handler signatures
- Sync and async handler support
- Priority ordering
- Weak reference support
- One-shot subscriptions

Example:
    >>> from toolcase.foundation.events import Signal
    >>> from typing import Callable
    >>> 
    >>> on_change: Signal[Callable[[str, int], None]] = Signal()
    >>> on_change += lambda name, val: print(f"{name}={val}")
    >>> on_change.fire("count", 42)
"""

from .events import Signal, SignalHandler, one_shot

__all__ = [
    "Signal",
    "SignalHandler",
    "one_shot",
]
