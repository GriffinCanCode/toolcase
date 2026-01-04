"""Agentic composition primitives for complex tool orchestration.

Provides higher-level patterns for agent workflows beyond simple pipelines:

- **Router**: Conditional routing based on input
- **Fallback**: Graceful degradation chains
- **Escalation**: Human-in-the-loop when automation fails
- **Race**: Parallel execution, first success wins
- **Gate**: Pre/post condition checks for safety

All primitives are tools themselves, enabling recursive composition.

Example:
    >>> from toolcase.agents import router, fallback, race, gate
    >>>
    >>> # Route based on input content
    >>> search = router(
    ...     when=lambda p: "news" in p.get("query", ""), use=NewsTool(),
    ...     when=lambda p: "academic" in p.get("query", ""), use=AcademicTool(),
    ...     default=WebSearchTool(),
    ... )
    >>>
    >>> # Automatic fallback chain
    >>> resilient = fallback(PrimaryTool(), BackupTool(), timeout=10.0)
    >>>
    >>> # Race multiple providers
    >>> fastest = race(ProviderA(), ProviderB(), ProviderC())
    >>>
    >>> # Gate with pre-condition
    >>> safe_delete = gate(
    ...     DeleteTool(),
    ...     pre=lambda p: p.get("confirmed") == True,
    ...     on_block="Deletion requires confirmation",
    ... )
"""

from .router import Route, RouterTool, router
from .fallback import FallbackTool, fallback
from .escalation import (
    EscalationHandler,
    EscalationResult,
    EscalationStatus,
    EscalationTool,
    QueueEscalation,
    retry_with_escalation,
)
from .race import RaceTool, race
from .gate import GateTool, gate

__all__ = [
    # Router
    "Route",
    "RouterTool",
    "router",
    # Fallback
    "FallbackTool",
    "fallback",
    # Escalation
    "EscalationHandler",
    "EscalationResult",
    "EscalationStatus",
    "EscalationTool",
    "QueueEscalation",
    "retry_with_escalation",
    # Race
    "RaceTool",
    "race",
    # Gate
    "GateTool",
    "gate",
]
