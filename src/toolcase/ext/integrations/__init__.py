"""Framework integrations for toolcase.

These are optional adapters for popular frameworks like LangChain.
Import only what you need to avoid unnecessary dependencies.
"""

from __future__ import annotations

__all__: list[str] = []

# Lazy imports - only load when accessed
def __getattr__(name: str):
    if name == "to_langchain":
        from .langchain import to_langchain
        return to_langchain
    if name == "to_langchain_tools":
        from .langchain import to_langchain_tools
        return to_langchain_tools
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
