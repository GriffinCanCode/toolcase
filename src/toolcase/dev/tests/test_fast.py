"""Tests for fast msgspec-based validation.

Validates the 10-100x faster validation path using msgspec.
"""

import pytest
from msgspec import Struct

from toolcase.foundation.fast import (
    FastStruct,
    FastValidator,
    MsgspecCodec,
    decode,
    encode,
    encode_str,
    fast,
    fast_frozen,
    from_dict,
    from_pydantic,
    pydantic_to_fast,
    to_dict,
    to_pydantic,
    validate,
    validate_many,
    validate_or_none,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleEvent(FastStruct):
    """Simple event for testing."""
    name: str
    value: int = 0


@fast_frozen
class FrozenConfig:
    """Frozen config created via decorator."""
    host: str
    port: int


@fast
class MutableConfig:
    """Mutable config via decorator."""
    host: str
    port: int


# ═══════════════════════════════════════════════════════════════════════════════
# FastStruct Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFastStruct:
    """Test FastStruct base class."""
    
    def test_create_struct(self):
        """Create struct with required fields."""
        event = SimpleEvent(name="test", value=42)
        assert event.name == "test"
        assert event.value == 42
    
    def test_default_values(self):
        """Default values work correctly."""
        event = SimpleEvent(name="test")
        assert event.value == 0
    
    def test_frozen(self):
        """FastStruct is immutable."""
        event = SimpleEvent(name="test")
        with pytest.raises(AttributeError):
            event.name = "changed"  # type: ignore[misc]
    
    def test_hashable(self):
        """Frozen structs are hashable."""
        e1 = SimpleEvent(name="test", value=1)
        e2 = SimpleEvent(name="test", value=1)
        assert hash(e1) == hash(e2)
        assert {e1, e2} == {e1}


# ═══════════════════════════════════════════════════════════════════════════════
# Decorator Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDecorators:
    """Test @fast and @fast_frozen decorators."""
    
    def test_fast_frozen_decorator(self):
        """@fast_frozen creates frozen struct."""
        config = FrozenConfig(host="localhost", port=8080)
        assert config.host == "localhost"
        assert config.port == 8080
        with pytest.raises(AttributeError):
            config.host = "changed"  # type: ignore[misc]
    
    def test_fast_decorator(self):
        """@fast creates mutable struct."""
        config = MutableConfig(host="localhost", port=8080)
        assert config.host == "localhost"
        config.host = "newhost"
        assert config.host == "newhost"


# ═══════════════════════════════════════════════════════════════════════════════
# Validation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidation:
    """Test validation functions."""
    
    def test_validate_from_dict(self):
        """Validate dict to struct."""
        event = validate(SimpleEvent, {"name": "test", "value": 42})
        assert isinstance(event, SimpleEvent)
        assert event.name == "test"
        assert event.value == 42
    
    def test_validate_from_json_bytes(self):
        """Validate JSON bytes to struct."""
        event = validate(SimpleEvent, b'{"name": "test", "value": 42}')
        assert event.name == "test"
    
    def test_validate_from_json_string(self):
        """Validate JSON string to struct."""
        event = validate(SimpleEvent, '{"name": "test", "value": 42}')
        assert event.name == "test"
    
    def test_validate_invalid_raises(self):
        """Invalid data raises ValidationError."""
        from msgspec import ValidationError
        with pytest.raises(ValidationError):
            validate(SimpleEvent, {"name": 123})  # name must be str
    
    def test_validate_or_none_success(self):
        """validate_or_none returns value on success."""
        event = validate_or_none(SimpleEvent, {"name": "test"})
        assert event is not None
        assert event.name == "test"
    
    def test_validate_or_none_failure(self):
        """validate_or_none returns None on failure."""
        event = validate_or_none(SimpleEvent, {"name": 123})
        assert event is None
    
    def test_validate_many(self):
        """Validate multiple items."""
        events = validate_many(SimpleEvent, [
            {"name": "a", "value": 1},
            {"name": "b", "value": 2},
            {"name": "c", "value": 3},
        ])
        assert len(events) == 3
        assert all(isinstance(e, SimpleEvent) for e in events)
        assert [e.name for e in events] == ["a", "b", "c"]


# ═══════════════════════════════════════════════════════════════════════════════
# FastValidator Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFastValidator:
    """Test FastValidator class."""
    
    def test_validator_callable(self):
        """Validator is callable."""
        validator = FastValidator(SimpleEvent)
        event = validator({"name": "test"})
        assert event.name == "test"
    
    def test_validator_or_none(self):
        """Validator.or_none method."""
        validator = FastValidator(SimpleEvent)
        assert validator.or_none({"name": "test"}) is not None
        assert validator.or_none({"invalid": True}) is None
    
    def test_validator_many(self):
        """Validator.many method."""
        validator = FastValidator(SimpleEvent)
        events = validator.many([{"name": "a"}, {"name": "b"}])
        assert len(events) == 2
    
    def test_validator_type_property(self):
        """Validator exposes type property."""
        validator = FastValidator(SimpleEvent)
        assert validator.type is SimpleEvent


# ═══════════════════════════════════════════════════════════════════════════════
# Encoding Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEncoding:
    """Test encoding/decoding functions."""
    
    def test_encode_struct(self):
        """Encode struct to JSON bytes."""
        event = SimpleEvent(name="test", value=42)
        result = encode(event)
        assert isinstance(result, bytes)
        assert b'"name":"test"' in result
        assert b'"value":42' in result
    
    def test_encode_str(self):
        """Encode struct to JSON string."""
        event = SimpleEvent(name="test")
        result = encode_str(event)
        assert isinstance(result, str)
        assert '"name":"test"' in result
    
    def test_encode_dict(self):
        """Encode dict to JSON bytes."""
        result = encode({"key": "value"})
        assert result == b'{"key":"value"}'
    
    def test_decode_bytes(self):
        """Decode JSON bytes."""
        result = decode(b'{"key": "value"}')
        assert result == {"key": "value"}
    
    def test_decode_string(self):
        """Decode JSON string."""
        result = decode('{"key": "value"}')
        assert result == {"key": "value"}
    
    def test_decode_with_type(self):
        """Decode JSON with target type."""
        result = decode(b'{"name": "test", "value": 1}', type=SimpleEvent)
        assert isinstance(result, SimpleEvent)
        assert result.name == "test"


# ═══════════════════════════════════════════════════════════════════════════════
# Dict Conversion Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDictConversion:
    """Test to_dict/from_dict."""
    
    def test_to_dict(self):
        """Convert struct to dict."""
        event = SimpleEvent(name="test", value=42)
        d = to_dict(event)
        assert d == {"name": "test", "value": 42}
    
    def test_from_dict(self):
        """Create struct from dict."""
        event = from_dict(SimpleEvent, {"name": "test"})
        assert event.name == "test"


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic Interop Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPydanticInterop:
    """Test Pydantic interoperability."""
    
    def test_to_pydantic(self):
        """Convert msgspec struct to Pydantic model."""
        from pydantic import BaseModel
        
        class PydanticEvent(BaseModel):
            name: str
            value: int = 0
        
        fast_event = SimpleEvent(name="test", value=42)
        pydantic_event = to_pydantic(fast_event, PydanticEvent)
        
        assert isinstance(pydantic_event, PydanticEvent)
        assert pydantic_event.name == "test"
        assert pydantic_event.value == 42
    
    def test_from_pydantic(self):
        """Convert Pydantic model to msgspec struct."""
        from pydantic import BaseModel
        
        class PydanticEvent(BaseModel):
            name: str
            value: int = 0
        
        pydantic_event = PydanticEvent(name="test", value=42)
        fast_event = from_pydantic(pydantic_event, SimpleEvent)
        
        assert isinstance(fast_event, SimpleEvent)
        assert fast_event.name == "test"
        assert fast_event.value == 42
    
    def test_pydantic_to_fast(self):
        """Generate fast struct from Pydantic model schema."""
        from pydantic import BaseModel
        
        class PydanticUser(BaseModel):
            name: str
            age: int
        
        FastUser = pydantic_to_fast(PydanticUser)
        user = validate(FastUser, {"name": "alice", "age": 30})
        
        assert user.name == "alice"  # type: ignore[attr-defined]
        assert user.age == 30  # type: ignore[attr-defined]


# ═══════════════════════════════════════════════════════════════════════════════
# MsgspecCodec Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMsgspecCodec:
    """Test MsgspecCodec for streaming integration."""
    
    def test_codec_properties(self):
        """Codec has correct properties."""
        codec = MsgspecCodec()
        assert codec.name == "msgspec"
        assert codec.content_type == "application/json"
    
    def test_codec_encode(self):
        """Codec encodes to bytes."""
        codec = MsgspecCodec()
        result = codec.encode({"key": "value"})
        assert result == b'{"key":"value"}'
    
    def test_codec_decode(self):
        """Codec decodes from bytes."""
        codec = MsgspecCodec()
        result = codec.decode(b'{"key": "value"}')
        assert result == {"key": "value"}


# ═══════════════════════════════════════════════════════════════════════════════
# Omit Defaults Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOmitDefaults:
    """Test that default values are omitted from JSON output."""
    
    def test_omit_default_value(self):
        """Default values are omitted from JSON."""
        event = SimpleEvent(name="test")  # value defaults to 0
        encoded = encode_str(event)
        assert "value" not in encoded  # default omitted
    
    def test_include_non_default_value(self):
        """Non-default values are included."""
        event = SimpleEvent(name="test", value=42)
        encoded = encode_str(event)
        assert '"value":42' in encoded
