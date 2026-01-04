"""Core concurrency primitives: task management and synchronization.

Primitives are the foundational building blocks for concurrent programming:
- Task management: TaskGroup, TaskHandle, CancelScope
- Synchronization: Lock, Semaphore, Event, Barrier, Condition
"""

from .task import (
    TaskGroup,
    TaskHandle,
    TaskState,
    CancelScope,
    shield,
    checkpoint,
    shielded_checkpoint,
    current_task,
    spawn,
    cancellable,
)

from .sync import (
    Lock,
    RLock,
    Semaphore,
    BoundedSemaphore,
    Event,
    Condition,
    Barrier,
    CapacityLimiter,
)

__all__ = [
    # Task management
    "TaskGroup",
    "TaskHandle",
    "TaskState",
    "CancelScope",
    "shield",
    "checkpoint",
    "shielded_checkpoint",
    "current_task",
    "spawn",
    "cancellable",
    # Synchronization
    "Lock",
    "RLock",
    "Semaphore",
    "BoundedSemaphore",
    "Event",
    "Condition",
    "Barrier",
    "CapacityLimiter",
]
