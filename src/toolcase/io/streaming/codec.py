"""High-performance serialization codecs for streaming transport.

Provides orjson (fast JSON) and msgpack (binary) codecs.
Both are core dependencies - no fallback to stdlib json.

Usage:
    >>> from toolcase.io.streaming import get_codec, encode, decode
    >>> encoded = encode({"key": "value"})  # orjson by default
    >>> decoded = decode(encoded)
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import msgpack
import orjson

if TYPE_CHECKING:
    from toolcase.foundation.errors import JsonValue


class CodecType(StrEnum):
    """Supported codec types."""
    ORJSON = "orjson"
    MSGPACK = "msgpack"


@runtime_checkable
class Codec(Protocol):
    """Protocol for serialization codecs."""
    
    name: str
    content_type: str
    
    def encode(self, data: JsonValue) -> bytes: ...
    def decode(self, data: bytes) -> JsonValue: ...


# ═══════════════════════════════════════════════════════════════════════════════
# Codec Implementations
# ═══════════════════════════════════════════════════════════════════════════════

class OrjsonCodec:
    """orjson codec - 3-10x faster than stdlib json.
    
    Features: native datetime/uuid support, numpy arrays, dataclasses.
    """
    
    __slots__ = ()
    name = "orjson"
    content_type = "application/json"
    
    def encode(self, data: JsonValue) -> bytes:
        return orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_UTC_Z)
    
    def decode(self, data: bytes) -> JsonValue:
        return orjson.loads(data)


class MsgpackCodec:
    """MessagePack codec - binary protocol, ~2x faster than orjson.
    
    Produces smaller payloads than JSON. Ideal for high-throughput streaming.
    """
    
    __slots__ = ()
    name = "msgpack"
    content_type = "application/msgpack"
    
    def encode(self, data: JsonValue) -> bytes:
        return msgpack.packb(data, use_bin_type=True, strict_types=False)
    
    def decode(self, data: bytes) -> JsonValue:
        return msgpack.unpackb(data, raw=False, strict_map_key=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton Instances (hot path)
# ═══════════════════════════════════════════════════════════════════════════════

_orjson = OrjsonCodec()
_msgpack = MsgpackCodec()

_CODECS: dict[str, Codec] = {"orjson": _orjson, "msgpack": _msgpack}


def get_codec(name: str | CodecType | None = None) -> Codec:
    """Get codec by name (default: orjson)."""
    return _CODECS.get(str(name) if name else "orjson", _orjson)


def register_codec(name: str, codec: Codec) -> None:
    """Register custom codec implementation."""
    _CODECS[name] = codec


# ═══════════════════════════════════════════════════════════════════════════════
# Direct Functions (hot path - no indirection)
# ═══════════════════════════════════════════════════════════════════════════════

def encode(data: JsonValue) -> bytes:
    """Encode to JSON bytes (orjson)."""
    return orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_UTC_Z)


def decode(data: bytes | str) -> JsonValue:
    """Decode from JSON bytes/str (orjson)."""
    return orjson.loads(data)


def encode_str(data: JsonValue) -> str:
    """Encode to JSON string (orjson)."""
    return orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_UTC_Z).decode()


def pack(data: JsonValue) -> bytes:
    """Encode to msgpack bytes."""
    return msgpack.packb(data, use_bin_type=True, strict_types=False)


def unpack(data: bytes) -> JsonValue:
    """Decode from msgpack bytes."""
    return msgpack.unpackb(data, raw=False, strict_map_key=False)


# Backwards compat aliases
fast_encode = encode
fast_decode = decode
msgpack_encode = pack
msgpack_decode = unpack
