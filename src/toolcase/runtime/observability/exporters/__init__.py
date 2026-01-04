"""Span exporters for observability backends.

Organized into:
- core/: Protocol, base exporters, batching, filtering
- clients/: Protocol-specific clients (OTLP)
- vendors/: Platform-specific exporters (Datadog, Honeycomb, Zipkin)
"""

from .core import (
    AsyncBatchExporter,
    BatchExporter,
    CompositeExporter,
    ConsoleExporter,
    Exporter,
    FilteredExporter,
    JsonExporter,
    NoOpExporter,
    SampledExporter,
    SpanPredicate,
    errors_only,
    slow_spans,
)
from .clients import (
    OTLPBridge,
    OTLPHttpBridge,
    create_otlp_exporter,
)
from .vendors import (
    DatadogExporter,
    HoneycombExporter,
    ZipkinExporter,
    datadog,
    honeycomb,
    zipkin,
)

__all__ = [
    # Core - Protocol
    "Exporter",
    # Core - Simple
    "NoOpExporter",
    "ConsoleExporter",
    "JsonExporter",
    # Core - Batching
    "BatchExporter",
    "AsyncBatchExporter",
    "CompositeExporter",
    # Core - Filtering
    "SampledExporter",
    "FilteredExporter",
    "SpanPredicate",
    "errors_only",
    "slow_spans",
    # Clients - OTLP
    "OTLPBridge",
    "OTLPHttpBridge",
    "create_otlp_exporter",
    # Vendors
    "DatadogExporter",
    "HoneycombExporter",
    "ZipkinExporter",
    "datadog",
    "honeycomb",
    "zipkin",
]
