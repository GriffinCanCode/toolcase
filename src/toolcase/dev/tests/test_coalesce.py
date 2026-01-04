"""Tests for CoalesceMiddleware (request deduplication/singleflight)."""

import asyncio

import pytest
from pydantic import BaseModel

from toolcase.runtime.middleware import Context, compose
from toolcase.runtime.middleware.plugins import CoalesceMiddleware


class MockParams(BaseModel):
    """Test parameter model."""
    query: str


class MockMetadata:
    """Minimal tool metadata."""
    __slots__ = ("name",)
    def __init__(self, name: str = "test_tool") -> None:
        self.name = name


class MockTool:
    """Minimal tool for middleware testing."""
    __slots__ = ("metadata", "_call_count", "_delay")
    
    def __init__(self, name: str = "test_tool", delay: float = 0.0) -> None:
        self.metadata = MockMetadata(name)
        self._call_count = 0
        self._delay = delay
    
    @property
    def call_count(self) -> int:
        return self._call_count
    
    async def arun(self, params: BaseModel) -> str:
        self._call_count += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        return f"result:{params.model_dump_json()}"


class TestCoalesceMiddleware:
    """Tests for CoalesceMiddleware."""
    
    @pytest.mark.asyncio
    async def test_single_request_passthrough(self) -> None:
        """Single request executes normally."""
        mw = CoalesceMiddleware()
        tool = MockTool()
        params = MockParams(query="test")
        ctx = Context()
        
        chain = compose([mw])
        result = await chain(tool, params, ctx)  # type: ignore[arg-type]
        
        assert "result:" in result
        assert tool.call_count == 1
        assert ctx.get("coalesced") is False
    
    @pytest.mark.asyncio
    async def test_concurrent_identical_requests_coalesced(self) -> None:
        """Concurrent identical requests share single execution."""
        mw = CoalesceMiddleware()
        tool = MockTool(delay=0.1)  # Slow enough to overlap
        params = MockParams(query="same")
        
        chain = compose([mw])
        
        async def make_request() -> tuple[str, bool]:
            ctx = Context()
            result = await chain(tool, params, ctx)  # type: ignore[arg-type]
            return result, ctx.get("coalesced", False)  # type: ignore[return-value]
        
        # Fire 5 concurrent requests
        tasks = [asyncio.create_task(make_request()) for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All should get same result
        assert all(r[0] == results[0][0] for r in results)
        # Tool should only execute once
        assert tool.call_count == 1
        # First request not coalesced, rest were
        coalesced_count = sum(1 for _, c in results if c)
        assert coalesced_count == 4
    
    @pytest.mark.asyncio
    async def test_different_params_not_coalesced(self) -> None:
        """Requests with different params execute separately."""
        mw = CoalesceMiddleware()
        tool = MockTool(delay=0.05)
        
        chain = compose([mw])
        
        async def make_request(query: str) -> str:
            ctx = Context()
            params = MockParams(query=query)
            return await chain(tool, params, ctx)  # type: ignore[arg-type]
        
        # Fire concurrent requests with different params
        tasks = [
            asyncio.create_task(make_request("a")),
            asyncio.create_task(make_request("b")),
            asyncio.create_task(make_request("c")),
        ]
        results = await asyncio.gather(*tasks)
        
        # All different results
        assert len(set(results)) == 3
        # All executed separately
        assert tool.call_count == 3
    
    @pytest.mark.asyncio
    async def test_sequential_requests_not_coalesced(self) -> None:
        """Sequential requests for same params execute separately."""
        mw = CoalesceMiddleware()
        tool = MockTool()
        params = MockParams(query="test")
        
        chain = compose([mw])
        
        # Execute sequentially
        for _ in range(3):
            ctx = Context()
            await chain(tool, params, ctx)  # type: ignore[arg-type]
        
        # Each should execute
        assert tool.call_count == 3
    
    @pytest.mark.asyncio
    async def test_error_propagates_to_waiters(self) -> None:
        """Errors from first request propagate to all waiters."""
        mw = CoalesceMiddleware()
        
        class FailingTool:
            metadata = MockMetadata("failing_tool")
            call_count = 0
            
            async def arun(self, params: BaseModel) -> str:
                self.call_count += 1
                await asyncio.sleep(0.05)
                raise ValueError("intentional error")
        
        tool = FailingTool()
        params = MockParams(query="fail")
        
        chain = compose([mw])
        
        async def make_request() -> str:
            ctx = Context()
            return await chain(tool, params, ctx)  # type: ignore[arg-type]
        
        # Fire concurrent requests
        tasks = [asyncio.create_task(make_request()) for _ in range(3)]
        results = await asyncio.gather(*tasks)
        
        # All should receive same error result (compose converts exceptions to error strings)
        assert all("**Tool Error" in r for r in results)
        assert all(results[0] == r for r in results)
        # Tool should only execute once
        assert tool.call_count == 1
    
    @pytest.mark.asyncio
    async def test_stats_tracking(self) -> None:
        """Statistics are tracked correctly."""
        mw = CoalesceMiddleware()
        tool = MockTool(delay=0.05)
        
        chain = compose([mw])
        
        # Fire concurrent requests
        params = MockParams(query="stats")
        tasks = [
            asyncio.create_task(chain(tool, params, Context()))  # type: ignore[arg-type]
            for _ in range(5)
        ]
        await asyncio.gather(*tasks)
        
        stats = mw.stats
        assert stats["total_requests"] == 5
        assert stats["coalesced_requests"] == 4
        assert stats["in_flight"] == 0
        assert stats["coalesce_ratio"] == 0.8
    
    @pytest.mark.asyncio
    async def test_in_flight_tracking(self) -> None:
        """In-flight count is accurate during execution."""
        mw = CoalesceMiddleware()
        tool = MockTool(delay=0.1)
        params = MockParams(query="inflight")
        
        chain = compose([mw])
        
        # Start requests
        task1 = asyncio.create_task(chain(tool, params, Context()))  # type: ignore[arg-type]
        await asyncio.sleep(0.01)  # Let it start
        
        assert mw.in_flight == 1
        
        # Start another with different params
        params2 = MockParams(query="other")
        task2 = asyncio.create_task(chain(tool, params2, Context()))  # type: ignore[arg-type]
        await asyncio.sleep(0.01)
        
        assert mw.in_flight == 2
        
        await asyncio.gather(task1, task2)
        assert mw.in_flight == 0
    
    @pytest.mark.asyncio
    async def test_reset_stats(self) -> None:
        """Stats can be reset."""
        mw = CoalesceMiddleware()
        tool = MockTool()
        params = MockParams(query="reset")
        
        chain = compose([mw])
        await chain(tool, params, Context())  # type: ignore[arg-type]
        
        assert mw.stats["total_requests"] == 1
        
        mw.reset_stats()
        
        assert mw.stats["total_requests"] == 0
        assert mw.stats["coalesced_requests"] == 0
    
    @pytest.mark.asyncio
    async def test_global_mode(self) -> None:
        """Global mode coalesces across tools."""
        mw = CoalesceMiddleware(per_tool=False)
        tool1 = MockTool(name="tool_a", delay=0.05)
        tool2 = MockTool(name="tool_b", delay=0.05)
        params = MockParams(query="global")
        
        chain = compose([mw])
        
        # Same params, different tools with global mode - should NOT coalesce
        # because key includes param hash which differs per tool name prefix
        # Actually, global mode just removes tool name from key...
        # Let me re-check the implementation
        
        # Fire concurrent requests to different tools
        task1 = asyncio.create_task(chain(tool1, params, Context()))  # type: ignore[arg-type]
        task2 = asyncio.create_task(chain(tool2, params, Context()))  # type: ignore[arg-type]
        
        await asyncio.gather(task1, task2)
        
        # With global=False (per_tool=True), they would execute separately
        # With global=True (per_tool=False), they coalesce by params only
        assert tool1.call_count + tool2.call_count == 1  # One coalesced
    
    @pytest.mark.asyncio
    async def test_context_coalesce_key_set(self) -> None:
        """Coalesce key is set in context for first request."""
        mw = CoalesceMiddleware()
        tool = MockTool()
        params = MockParams(query="keytest")
        ctx = Context()
        
        chain = compose([mw])
        await chain(tool, params, ctx)  # type: ignore[arg-type]
        
        assert ctx.get("coalesce_key") is not None
        assert ctx.get("coalesce_key", "").startswith("test_tool:")  # type: ignore[union-attr]
    
    @pytest.mark.asyncio
    async def test_include_in_ctx_disabled(self) -> None:
        """Context annotation can be disabled."""
        mw = CoalesceMiddleware(include_in_ctx=False)
        tool = MockTool()
        params = MockParams(query="nocontext")
        ctx = Context()
        
        chain = compose([mw])
        await chain(tool, params, ctx)  # type: ignore[arg-type]
        
        assert "coalesced" not in ctx
        assert "coalesce_key" not in ctx
    
    @pytest.mark.asyncio
    async def test_cleanup_on_exception(self) -> None:
        """Flight is cleaned up even on exception."""
        mw = CoalesceMiddleware()
        
        class FailingTool:
            metadata = MockMetadata("failing")
            async def arun(self, params: BaseModel) -> str:
                raise RuntimeError("fail")
        
        tool = FailingTool()
        params = MockParams(query="cleanup")
        
        chain = compose([mw])
        
        # compose() catches exceptions and returns error strings
        result = await chain(tool, params, Context())  # type: ignore[arg-type]
        assert "**Tool Error" in result
        
        # Flight should be cleaned up
        assert mw.in_flight == 0
        
        # New request should work
        tool2 = MockTool()
        await chain(tool2, params, Context())  # type: ignore[arg-type]
        assert tool2.call_count == 1
