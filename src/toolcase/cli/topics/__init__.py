"""Topic definitions for toolcase CLI help system."""

from .help_topic import HELP
from .overview import OVERVIEW
from .tool import TOOL
from .result import RESULT
from .errors import ERRORS
from .registry import REGISTRY
from .middleware import MIDDLEWARE
from .retry import RETRY
from .pipeline import PIPELINE
from .agents import AGENTS
from .concurrency import CONCURRENCY
from .cache import CACHE
from .streaming import STREAMING
from .formats import FORMATS
from .di import DI
from .settings import SETTINGS
from .tracing import TRACING
from .testing import TESTING
from .http import HTTP
from .discovery import DISCOVERY
from .imports import IMPORTS

TOPICS: dict[str, str] = {
    "help": HELP,
    "overview": OVERVIEW,
    "tool": TOOL,
    "result": RESULT,
    "errors": ERRORS,
    "registry": REGISTRY,
    "middleware": MIDDLEWARE,
    "retry": RETRY,
    "pipeline": PIPELINE,
    "agents": AGENTS,
    "concurrency": CONCURRENCY,
    "cache": CACHE,
    "streaming": STREAMING,
    "formats": FORMATS,
    "di": DI,
    "settings": SETTINGS,
    "tracing": TRACING,
    "testing": TESTING,
    "http": HTTP,
    "discovery": DISCOVERY,
    "imports": IMPORTS,
}

__all__ = ["TOPICS"]
