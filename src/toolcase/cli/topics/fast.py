"""CLI help topic for fast msgspec-based validation."""

TOPIC = "fast"
TITLE = "Fast Validation (msgspec)"
SHORT = "10-100x faster validation via msgspec for hot paths"

CONTENT = """
# Fast Validation with msgspec

Toolcase provides an alternative validation path using msgspec that is
10-100x faster than Pydantic for hot paths. Use this when validation
latency is a bottleneck.

## When to Use Fast Validation

✓ High-throughput HTTP endpoints
✓ Batch processing with simple parameters  
✓ Streaming event serialization
✓ Any path where validation is a bottleneck

## When to Use Pydantic (Default)

✓ Complex validation rules (cross-field, custom validators)
✓ Computed fields and model_dump customization
✓ Rich error messages for LLM feedback
✓ IDE autocomplete and type inference

## Quick Start

```python
from toolcase.foundation.fast import FastStruct, validate, encode

# Define a fast struct (immutable by default)
class Event(FastStruct):
    name: str
    value: int = 0

# Validate dict → struct (10-100x faster than Pydantic)
event = validate(Event, {"name": "test", "value": 42})

# Encode to JSON (5x faster than orjson)
json_bytes = encode(event)  # b'{"name":"test","value":42}'
```

## Using Decorators

```python
from toolcase.foundation.fast import fast, fast_frozen

# Mutable struct via decorator
@fast
class Config:
    host: str
    port: int = 8080

# Frozen (immutable) struct via decorator  
@fast_frozen
class Event:
    name: str
    timestamp: float
```

## FastValidator for Reusable Validation

```python
from toolcase.foundation.fast import FastValidator

validator = FastValidator(Event)

# Single item
event = validator({"name": "test"})

# Returns None on failure (no exception)
event_or_none = validator.or_none({"invalid": True})

# Batch validation
events = validator.many([{"name": "a"}, {"name": "b"}])
```

## Pydantic Interoperability

```python
from toolcase.foundation.fast import (
    to_pydantic, from_pydantic, pydantic_to_fast
)

# Fast struct → Pydantic model
pydantic_model = to_pydantic(fast_struct, PydanticModel)

# Pydantic model → Fast struct  
fast_struct = from_pydantic(pydantic_model, FastStruct)

# Generate fast struct from Pydantic schema
FastUser = pydantic_to_fast(PydanticUser)
```

## Fast Validation Middleware

For high-throughput endpoints, use FastValidation middleware:

```python
from toolcase.runtime.middleware.plugins import FastValidation

# 10-100x faster than ValidationMiddleware
registry.use(FastValidation())
```

Note: FastValidation does not support custom validators,
cross-field constraints, or the Schema DSL. Use ValidationMiddleware
when you need those features.

## Streaming Codec

Use msgspec codec for fastest JSON serialization in streams:

```python
from toolcase.io.streaming import get_codec

codec = get_codec("msgspec")  # 5x faster than orjson
encoded = codec.encode(data)
decoded = codec.decode(encoded)
```

## Performance Comparison

| Operation           | Pydantic | msgspec | Speedup |
|---------------------|----------|---------|---------|
| Dict validation     | 1x       | 10-100x | ⚡      |
| JSON encoding       | 1x       | 50x     | ⚡      |
| JSON decoding       | 1x       | 25x     | ⚡      |
| Struct creation     | 1x       | 5-10x   | ⚡      |

## API Reference

### Base Classes
- `FastStruct` - Immutable base struct (kw_only, frozen, gc=False)
- `MutableStruct` - Mutable variant for builder patterns

### Decorators  
- `@fast` - Create mutable struct from class
- `@fast_frozen` - Create frozen struct from class

### Validation
- `validate(type, data)` - Validate and convert to type
- `validate_or_none(type, data)` - Returns None on failure
- `validate_many(type, items)` - Batch validation
- `FastValidator` - Reusable cached validator

### Encoding
- `encode(obj)` - To JSON bytes (5x faster than orjson)
- `encode_str(obj)` - To JSON string
- `decode(data)` - From JSON bytes/string
- `decode(data, type=T)` - Decode with validation

### Conversion
- `to_dict(struct)` - Struct to dict
- `from_dict(type, dict)` - Dict to struct

### Pydantic Interop
- `to_pydantic(struct, model)` - To Pydantic model
- `from_pydantic(model, type)` - From Pydantic model
- `pydantic_to_fast(model)` - Generate fast struct from Pydantic schema
"""
