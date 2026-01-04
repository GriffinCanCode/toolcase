"""Tests for Result-based streaming (typed error propagation mid-stream)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from toolcase import Err, ErrorTrace, Ok, Result
from toolcase.foundation.errors import trace
from toolcase.io.streaming import (
    ChunkResult,
    ResultStream,
    StreamCollectResult,
    collect_or_first_error,
    collect_result_stream,
    err_chunk,
    err_chunk_from_exc,
    filter_err,
    filter_ok,
    map_err,
    map_ok,
    ok_chunk,
    recover,
    result_stream,
    result_stream_resilient,
    tap_err,
    tap_ok,
    unwrap_stream,
)


# ─────────────────────────────────────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────────────────────────────────────

class TestFactories:
    """Test ok_chunk, err_chunk, err_chunk_from_exc factories."""
    
    def test_ok_chunk_creates_success_result(self) -> None:
        result = ok_chunk("content")
        assert result.is_ok()
        assert result.value == "content"
    
    def test_err_chunk_creates_error_with_trace(self) -> None:
        result = err_chunk("failed", code="NETWORK_ERROR")
        assert result.is_err()
        assert result.error.message == "failed"
        assert result.error.error_code == "NETWORK_ERROR"
        assert result.error.recoverable is True
    
    def test_err_chunk_unrecoverable(self) -> None:
        result = err_chunk("fatal", recoverable=False)
        assert result.is_err()
        assert result.error.recoverable is False
    
    def test_err_chunk_from_exc(self) -> None:
        exc = ValueError("bad value")
        result = err_chunk_from_exc(exc, operation="parse", code="VALIDATION")
        assert result.is_err()
        assert "bad value" in result.error.message
        assert result.error.error_code == "VALIDATION"


# ─────────────────────────────────────────────────────────────────────────────
# Stream Converters
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamConverters:
    """Test result_stream and related conversion utilities."""
    
    @pytest.mark.asyncio
    async def test_result_stream_wraps_success(self) -> None:
        """Successful chunks wrapped as Ok."""
        async def source() -> AsyncIterator[str]:
            yield "a"
            yield "b"
        
        results = [r async for r in result_stream(source())]
        assert all(r.is_ok() for r in results)
        assert [r.value for r in results] == ["a", "b"]
    
    @pytest.mark.asyncio
    async def test_result_stream_catches_exception(self) -> None:
        """Exception becomes Err, stream terminates."""
        async def source() -> AsyncIterator[str]:
            yield "ok"
            raise RuntimeError("boom")
        
        results = [r async for r in result_stream(source(), operation="test")]
        assert len(results) == 2
        assert results[0].is_ok()
        assert results[1].is_err()
        assert "boom" in results[1].error.message
    
    @pytest.mark.asyncio
    async def test_result_stream_resilient_continues(self) -> None:
        """Resilient stream can continue after error."""
        call_count = 0
        
        async def flaky() -> AsyncIterator[str]:
            nonlocal call_count
            for i in range(3):
                call_count += 1
                if i == 1:
                    raise ValueError("flaky fail")
                yield f"chunk{i}"
        
        # Note: resilient can only continue if the source supports it
        # In this case, the generator dies after raising
        results = [r async for r in result_stream_resilient(flaky())]
        
        # First chunk ok, then error
        assert results[0].is_ok()
        assert results[0].value == "chunk0"
        assert results[1].is_err()
    
    @pytest.mark.asyncio
    async def test_unwrap_stream_yields_ok_values(self) -> None:
        """unwrap_stream extracts Ok values."""
        async def source() -> ResultStream:
            yield Ok("a")
            yield Ok("b")
        
        unwrapped = [s async for s in unwrap_stream(source())]
        assert unwrapped == ["a", "b"]
    
    @pytest.mark.asyncio
    async def test_unwrap_stream_raises_on_err(self) -> None:
        """unwrap_stream raises on Err by default."""
        async def source() -> ResultStream:
            yield Ok("a")
            yield Err(trace("error"))
        
        with pytest.raises(RuntimeError, match="error"):
            _ = [s async for s in unwrap_stream(source())]
    
    @pytest.mark.asyncio
    async def test_unwrap_stream_skip_errors(self) -> None:
        """unwrap_stream with raise_on_err=False skips Err."""
        async def source() -> ResultStream:
            yield Ok("a")
            yield Err(trace("skip me"))
            yield Ok("b")
        
        unwrapped = [s async for s in unwrap_stream(source(), raise_on_err=False)]
        assert unwrapped == ["a", "b"]
    
    @pytest.mark.asyncio
    async def test_filter_ok(self) -> None:
        """filter_ok yields only Ok values."""
        async def source() -> ResultStream:
            yield Ok("yes")
            yield Err(trace("no"))
            yield Ok("also yes")
        
        values = [v async for v in filter_ok(source())]
        assert values == ["yes", "also yes"]
    
    @pytest.mark.asyncio
    async def test_filter_err(self) -> None:
        """filter_err yields only Err traces."""
        async def source() -> ResultStream:
            yield Ok("skip")
            yield Err(trace("error1"))
            yield Err(trace("error2"))
        
        errors = [e async for e in filter_err(source())]
        assert len(errors) == 2
        assert errors[0].message == "error1"


# ─────────────────────────────────────────────────────────────────────────────
# Collectors
# ─────────────────────────────────────────────────────────────────────────────

class TestCollectors:
    """Test stream collectors."""
    
    @pytest.mark.asyncio
    async def test_collect_result_stream_all_ok(self) -> None:
        """Collect with all Ok chunks."""
        async def source() -> ResultStream:
            yield Ok("a")
            yield Ok("b")
            yield Ok("c")
        
        result = await collect_result_stream(source())
        assert result.content == "abc"
        assert result.errors == ()
        assert result.chunk_count == 3
        assert result.success is True
        assert result.has_errors is False
    
    @pytest.mark.asyncio
    async def test_collect_result_stream_with_errors(self) -> None:
        """Collect preserves both content and errors."""
        async def source() -> ResultStream:
            yield Ok("partial")
            yield Err(trace("err1"))
            yield Ok("more")
            yield Err(trace("err2"))
        
        result = await collect_result_stream(source())
        assert result.content == "partialmore"
        assert len(result.errors) == 2
        assert result.chunk_count == 4
        assert result.has_errors is True
        assert result.success is False
    
    @pytest.mark.asyncio
    async def test_stream_collect_result_to_result(self) -> None:
        """StreamCollectResult.to_result() conversion."""
        async def ok_source() -> ResultStream:
            yield Ok("x")
        
        async def err_source() -> ResultStream:
            yield Err(trace("fail"))
        
        ok_result = await collect_result_stream(ok_source())
        assert ok_result.to_result().is_ok()
        
        err_result = await collect_result_stream(err_source())
        assert err_result.to_result().is_err()
    
    @pytest.mark.asyncio
    async def test_collect_or_first_error_success(self) -> None:
        """collect_or_first_error returns Ok on success."""
        async def source() -> ResultStream:
            yield Ok("a")
            yield Ok("b")
        
        result = await collect_or_first_error(source())
        assert result.is_ok()
        assert result.value == "ab"
    
    @pytest.mark.asyncio
    async def test_collect_or_first_error_fails_fast(self) -> None:
        """collect_or_first_error returns first Err immediately."""
        collected = []
        
        async def source() -> ResultStream:
            yield Ok("a")
            collected.append("a")
            yield Err(trace("stop here"))
            collected.append("should not reach")
            yield Ok("b")
        
        result = await collect_or_first_error(source())
        assert result.is_err()
        assert result.error.message == "stop here"
        # Generator continues past Err but we don't consume further
        assert "should not reach" not in collected


# ─────────────────────────────────────────────────────────────────────────────
# Combinators
# ─────────────────────────────────────────────────────────────────────────────

class TestCombinators:
    """Test stream combinators (map, tap, recover)."""
    
    @pytest.mark.asyncio
    async def test_map_ok_transforms_values(self) -> None:
        """map_ok transforms Ok values."""
        async def source() -> ResultStream:
            yield Ok("hello")
            yield Err(trace("err"))
            yield Ok("world")
        
        results = [r async for r in map_ok(source(), str.upper)]
        assert results[0].value == "HELLO"
        assert results[1].is_err()  # Unchanged
        assert results[2].value == "WORLD"
    
    @pytest.mark.asyncio
    async def test_map_err_transforms_errors(self) -> None:
        """map_err transforms Err values."""
        async def source() -> ResultStream:
            yield Ok("ok")
            yield Err(trace("original"))
        
        def add_prefix(t: ErrorTrace) -> ErrorTrace:
            return t.with_code("MAPPED")
        
        results = [r async for r in map_err(source(), add_prefix)]
        assert results[0].is_ok()  # Unchanged
        assert results[1].error.error_code == "MAPPED"
    
    @pytest.mark.asyncio
    async def test_tap_ok_side_effect(self) -> None:
        """tap_ok executes side effect on Ok values."""
        seen: list[str] = []
        
        async def source() -> ResultStream:
            yield Ok("a")
            yield Err(trace("skip"))
            yield Ok("b")
        
        results = [r async for r in tap_ok(source(), seen.append)]
        assert seen == ["a", "b"]
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_tap_err_side_effect(self) -> None:
        """tap_err executes side effect on Err values."""
        errors_seen: list[str] = []
        
        async def source() -> ResultStream:
            yield Ok("skip")
            yield Err(trace("error1"))
            yield Err(trace("error2"))
        
        results = [r async for r in tap_err(source(), lambda e: errors_seen.append(e.message))]
        assert errors_seen == ["error1", "error2"]
    
    @pytest.mark.asyncio
    async def test_recover_converts_recoverable_errors(self) -> None:
        """recover converts recoverable Err to Ok."""
        async def source() -> ResultStream:
            yield Ok("a")
            yield Err(trace("recoverable", recoverable=True))
            yield Err(trace("fatal", recoverable=False))
            yield Ok("b")
        
        def handler(err: ErrorTrace) -> str | None:
            return "[recovered]" if err.recoverable else None
        
        results = [r async for r in recover(source(), handler)]
        
        assert results[0].value == "a"
        assert results[1].value == "[recovered]"  # Recovered
        assert results[2].is_err()  # Not recovered
        assert results[3].value == "b"


# ─────────────────────────────────────────────────────────────────────────────
# Integration: Real-world Patterns
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationPatterns:
    """Test realistic usage patterns."""
    
    @pytest.mark.asyncio
    async def test_llm_streaming_with_timeout_errors(self) -> None:
        """Simulate LLM streaming with intermittent timeout errors."""
        async def flaky_llm_stream() -> ResultStream:
            yield ok_chunk("The answer is ")
            yield err_chunk("timeout", code="TIMEOUT", recoverable=True)
            yield ok_chunk("42.")
        
        # Collect with error aggregation
        collected = await collect_result_stream(flaky_llm_stream())
        
        assert "The answer is " in collected.content
        assert "42." in collected.content
        assert collected.has_errors
        assert collected.errors[0].error_code == "TIMEOUT"
    
    @pytest.mark.asyncio
    async def test_pipeline_with_error_recovery(self) -> None:
        """Chain transformations with error recovery."""
        async def source() -> ResultStream:
            yield ok_chunk("good")
            yield err_chunk("bad chunk", recoverable=True)
            yield ok_chunk("also good")
        
        def recovery(e: ErrorTrace) -> str | None:
            return "[REDACTED]" if e.recoverable else None
        
        # Transform: uppercase ok values, recover errors
        stream = map_ok(source(), str.upper)
        stream = recover(stream, recovery)
        
        collected = await collect_result_stream(stream)
        assert collected.content == "GOOD[REDACTED]ALSO GOOD"
        assert not collected.has_errors  # All recovered
    
    @pytest.mark.asyncio
    async def test_error_logging_during_stream(self) -> None:
        """Log errors as they occur while preserving stream flow."""
        logged_errors: list[str] = []
        
        async def source() -> ResultStream:
            yield ok_chunk("chunk1")
            yield err_chunk("network error")
            yield ok_chunk("chunk2")
            yield err_chunk("parse error")
        
        # Tap errors for logging, then filter ok
        stream = tap_err(source(), lambda e: logged_errors.append(e.message))
        content = [c async for c in filter_ok(stream)]
        
        assert content == ["chunk1", "chunk2"]
        assert logged_errors == ["network error", "parse error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
