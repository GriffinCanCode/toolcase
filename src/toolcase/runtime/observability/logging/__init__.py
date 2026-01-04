"""Structured logging module: context-aware logging with trace correlation.

Built on structlog for maximum flexibility and readability.
"""

from .logger import (
    BoundLogger,
    LogScope,
    TracedLogger,
    configure_logging,
    configure_observability,
    get_logger,
    log_context,
    span_logger,
    timed,
)

__all__ = [
    "BoundLogger",
    "LogScope",
    "TracedLogger",
    "configure_logging",
    "configure_observability",
    "get_logger",
    "log_context",
    "span_logger",
    "timed",
]
