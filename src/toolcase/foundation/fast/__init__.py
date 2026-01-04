"""High-performance validation and serialization via msgspec.

Provides 10-100x faster validation for hot paths while maintaining
Pydantic interoperability. Use for:
- High-throughput streaming event serialization
- Request validation in HTTP middleware
- Batch processing validation
- Any path where validation is a bottleneck

Usage:
    >>> from toolcase.foundation.fast import FastStruct, fast, validate, encode, decode
    >>>
    >>> @fast
    ... class Event:
    ...     name: str
    ...     value: int = 0
    >>>
    >>> # 10-100x faster than Pydantic
    >>> event = validate(Event, {"name": "test", "value": 42})
    >>> encoded = encode(event)  # ~5x faster than orjson
    >>> decoded = decode(encoded, type=Event)

Interop with Pydantic:
    >>> from pydantic import BaseModel
    >>> from toolcase.foundation.fast import to_pydantic, from_pydantic
    >>>
    >>> fast_obj = validate(FastModel, data)
    >>> pydantic_obj = to_pydantic(fast_obj, PydanticModel)
    >>> back_to_fast = from_pydantic(pydantic_obj, FastModel)
"""

from .spec import (
    # Base classes
    FastStruct,
    # Decorators
    fast,
    fast_frozen,
    # Core validation
    validate,
    validate_or_none,
    validate_many,
    FastValidator,
    ValidationError,
    # Encoding (5x faster than orjson)
    encode,
    encode_str,
    decode,
    decode_lines,
    # Type conversion
    to_dict,
    from_dict,
    # Pydantic interop
    to_pydantic,
    from_pydantic,
    pydantic_to_fast,
    # Codec
    MsgspecCodec,
    get_encoder,
    get_decoder,
)

__all__ = [
    # Base
    "FastStruct",
    # Decorators
    "fast",
    "fast_frozen",
    # Validation
    "validate",
    "validate_or_none",
    "validate_many",
    "FastValidator",
    "ValidationError",
    # Encoding
    "encode",
    "encode_str",
    "decode",
    "decode_lines",
    # Conversion
    "to_dict",
    "from_dict",
    # Pydantic interop
    "to_pydantic",
    "from_pydantic",
    "pydantic_to_fast",
    # Codec
    "MsgspecCodec",
    "get_encoder",
    "get_decoder",
]
