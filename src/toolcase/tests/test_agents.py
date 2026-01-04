"""Tests for agentic composition primitives."""

import asyncio
import pytest
from pydantic import BaseModel, Field

from toolcase import BaseTool, ToolMetadata, ErrorCode
from toolcase.agents import (
    # Router
    Route,
    RouterTool,
    router,
    # Fallback
    FallbackTool,
    fallback,
    # Escalation
    EscalationHandler,
    EscalationResult,
    EscalationStatus,
    EscalationTool,
    QueueEscalation,
    retry_with_escalation,
    # Race
    RaceTool,
    race,
    # Gate
    GateTool,
    gate,
)
from toolcase.errors import Err, ErrorTrace


# ═══════════════════════════════════════════════════════════════════════════════
# Test Fixtures - Mock Tools
# ═══════════════════════════════════════════════════════════════════════════════


class QueryParams(BaseModel):
    query: str = Field(..., description="Search query")


class EchoTool(BaseTool[QueryParams]):
    """Simple echo tool for testing."""
    
    metadata = ToolMetadata(name="echo", description="Echoes input query")
    params_schema = QueryParams
    cache_enabled = False
    
    def __init__(self, prefix: str = "echo"):
        self._prefix = prefix
    
    def _run(self, params: QueryParams) -> str:
        return f"{self._prefix}: {params.query}"


class FailingTool(BaseTool[QueryParams]):
    """Tool that always fails."""
    
    metadata = ToolMetadata(name="failing", description="Always fails")
    params_schema = QueryParams
    cache_enabled = False
    
    def __init__(self, error_code: ErrorCode = ErrorCode.UNKNOWN, fail_count: int = -1):
        self._error_code = error_code
        self._fail_count = fail_count
        self._attempts = 0
    
    def _run(self, params: QueryParams) -> str:
        self._attempts += 1
        if self._fail_count == -1 or self._attempts <= self._fail_count:
            raise RuntimeError(f"Intentional failure: {params.query}")
        return f"success after {self._attempts} attempts"


class SlowTool(BaseTool[QueryParams]):
    """Tool with configurable delay."""
    
    metadata = ToolMetadata(name="slow", description="Slow tool with delay")
    params_schema = QueryParams
    cache_enabled = False
    
    def __init__(self, delay: float = 1.0):
        self._delay = delay
    
    async def _async_run(self, params: QueryParams) -> str:
        await asyncio.sleep(self._delay)
        return f"slow: {params.query}"
    
    def _run(self, params: QueryParams) -> str:
        return self._run_async_sync(self._async_run(params))


# ═══════════════════════════════════════════════════════════════════════════════
# Router Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRouter:
    """Tests for RouterTool and router() factory."""
    
    def test_router_basic_routing(self):
        """Router selects tool based on condition."""
        news_tool = EchoTool("news")
        code_tool = EchoTool("code")
        default_tool = EchoTool("default")
        
        r = router(
            (lambda p: "news" in p.get("query", ""), news_tool),
            (lambda p: "code" in p.get("query", ""), code_tool),
            default=default_tool,
        )
        
        assert "news:" in r(input={"query": "latest news"})
        assert "code:" in r(input={"query": "python code"})
        assert "default:" in r(input={"query": "random"})
    
    def test_router_keyword_syntax(self):
        """Router supports keyword-based routing."""
        news_tool = EchoTool("news")
        default_tool = EchoTool("default")
        
        r = router(default=default_tool, news=news_tool)
        
        assert "news:" in r(input={"query": "latest news today"})
        assert "default:" in r(input={"query": "random stuff"})
    
    def test_router_first_match_wins(self):
        """Router uses first matching condition."""
        tool1 = EchoTool("first")
        tool2 = EchoTool("second")
        default = EchoTool("default")
        
        r = router(
            (lambda p: "a" in p.get("query", ""), tool1),
            (lambda p: "a" in p.get("query", ""), tool2),  # Also matches, but shouldn't run
            default=default,
        )
        
        assert "first:" in r(input={"query": "abc"})
    
    def test_router_preserves_metadata(self):
        """Router derives appropriate metadata."""
        r = router(
            (lambda p: True, EchoTool()),
            default=EchoTool("default"),
            name="my_router",
            description="Custom router",
        )
        
        assert r.metadata.name == "my_router"
        assert r.metadata.description == "Custom router"
        assert r.metadata.category == "agents"
    
    @pytest.mark.asyncio
    async def test_router_async(self):
        """Router works asynchronously."""
        r = router(default=EchoTool("async"))
        result = await r.acall(input={"query": "test"})
        assert "async:" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Fallback Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFallback:
    """Tests for FallbackTool and fallback() factory."""
    
    def test_fallback_uses_first_success(self):
        """Fallback returns first successful result."""
        f = fallback(EchoTool("primary"), EchoTool("backup"))
        result = f(input={"query": "test"})
        assert "primary:" in result
    
    def test_fallback_on_failure(self):
        """Fallback tries next tool on failure."""
        f = fallback(
            FailingTool(),
            EchoTool("backup"),
        )
        result = f(input={"query": "test"})
        assert "backup:" in result
    
    def test_fallback_chain_multiple(self):
        """Fallback chains through multiple failures."""
        f = fallback(
            FailingTool(),
            FailingTool(),
            EchoTool("last"),
        )
        result = f(input={"query": "test"})
        assert "last:" in result
    
    @pytest.mark.asyncio
    async def test_fallback_timeout(self):
        """Fallback triggers on timeout."""
        f = fallback(
            SlowTool(delay=5.0),  # Will timeout
            EchoTool("backup"),
            timeout=0.1,
        )
        result = await f.acall(input={"query": "test"})
        assert "backup:" in result
    
    def test_fallback_all_fail(self):
        """Fallback returns error when all tools fail."""
        f = fallback(FailingTool(), FailingTool())
        result = f(input={"query": "test"})
        assert "failed" in result.lower()
    
    def test_fallback_metadata(self):
        """Fallback derives appropriate metadata."""
        f = fallback(
            EchoTool("a"),
            EchoTool("b"),
            name="my_fallback",
        )
        assert f.metadata.name == "my_fallback"
        assert f.metadata.category == "agents"


# ═══════════════════════════════════════════════════════════════════════════════
# Escalation Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEscalation:
    """Tests for EscalationTool and retry_with_escalation() factory."""
    
    @pytest.mark.asyncio
    async def test_escalation_success_no_escalate(self):
        """Escalation doesn't trigger on success."""
        escalated = False
        
        class TestHandler:
            async def escalate(self, request):
                nonlocal escalated
                escalated = True
                return EscalationResult(status=EscalationStatus.APPROVED)
        
        e = retry_with_escalation(
            EchoTool(),
            max_retries=2,
            escalate_to=TestHandler(),
        )
        
        result = await e.acall(input={"query": "test"})
        assert "echo:" in result
        assert not escalated
    
    @pytest.mark.asyncio
    async def test_escalation_triggers_on_exhausted_retries(self):
        """Escalation triggers after retries exhausted."""
        escalated = False
        
        class ApproveHandler:
            async def escalate(self, request):
                nonlocal escalated
                escalated = True
                return EscalationResult(
                    status=EscalationStatus.APPROVED,
                    value="human approved",
                )
        
        e = retry_with_escalation(
            FailingTool(),
            max_retries=1,
            escalate_to=ApproveHandler(),
        )
        
        result = await e.acall(input={"query": "test"})
        assert escalated
        assert "approved" in result.lower()
    
    @pytest.mark.asyncio
    async def test_escalation_rejection(self):
        """Escalation returns error on rejection."""
        class RejectHandler:
            async def escalate(self, request):
                return EscalationResult(
                    status=EscalationStatus.REJECTED,
                    reason="Not allowed",
                )
        
        e = retry_with_escalation(
            FailingTool(),
            max_retries=0,
            escalate_to=RejectHandler(),
        )
        
        result = await e.acall(input={"query": "test"})
        assert "rejected" in result.lower()
    
    @pytest.mark.asyncio
    async def test_queue_escalation_basic(self):
        """QueueEscalation basic workflow."""
        handler = QueueEscalation("test_queue", timeout=1.0, poll_interval=0.1)
        
        # Simulate resolution in background
        async def resolve_later():
            await asyncio.sleep(0.2)
            # Find pending request and resolve it
            for req_id in list(handler._pending.keys()):
                handler.resolve(req_id, EscalationResult(
                    status=EscalationStatus.APPROVED,
                    value="resolved",
                ))
        
        e = retry_with_escalation(
            FailingTool(),
            max_retries=0,
            escalate_to=handler,
        )
        
        # Start resolution task
        resolve_task = asyncio.create_task(resolve_later())
        
        result = await e.acall(input={"query": "test"})
        await resolve_task
        
        assert "resolved" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Race Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRace:
    """Tests for RaceTool and race() factory."""
    
    @pytest.mark.asyncio
    async def test_race_first_wins(self):
        """Race returns first successful result."""
        r = race(
            SlowTool(delay=0.5),
            EchoTool("fast"),  # Should win (instant)
            SlowTool(delay=1.0),
        )
        
        result = await r.acall(input={"query": "test"})
        assert "fast:" in result
    
    @pytest.mark.asyncio
    async def test_race_skips_failures(self):
        """Race continues to next success after failures."""
        r = race(
            FailingTool(),  # Fails fast
            EchoTool("winner"),  # Should win
            SlowTool(delay=1.0),
        )
        
        result = await r.acall(input={"query": "test"})
        assert "winner:" in result
    
    @pytest.mark.asyncio
    async def test_race_all_fail(self):
        """Race returns combined error when all fail."""
        r = race(FailingTool(), FailingTool())
        result = await r.acall(input={"query": "test"})
        assert "failed" in result.lower()
    
    @pytest.mark.asyncio
    async def test_race_timeout(self):
        """Race times out if no tool completes in time."""
        r = race(
            SlowTool(delay=10.0),
            SlowTool(delay=10.0),
            timeout=0.1,
        )
        
        result = await r.acall(input={"query": "test"})
        assert "timed out" in result.lower()
    
    def test_race_metadata(self):
        """Race derives appropriate metadata."""
        r = race(EchoTool("a"), EchoTool("b"), name="my_race")
        assert r.metadata.name == "my_race"
        assert r.metadata.category == "agents"


# ═══════════════════════════════════════════════════════════════════════════════
# Gate Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGate:
    """Tests for GateTool and gate() factory."""
    
    def test_gate_pre_check_pass(self):
        """Gate passes when pre-check returns True."""
        g = gate(
            EchoTool(),
            pre=lambda p: p.get("allowed") == True,
        )
        
        result = g(input={"query": "test", "allowed": True})
        assert "echo:" in result
    
    def test_gate_pre_check_block(self):
        """Gate blocks when pre-check returns False."""
        g = gate(
            EchoTool(),
            pre=lambda p: p.get("allowed") == True,
            on_block="Not allowed",
        )
        
        result = g(input={"query": "test", "allowed": False})
        assert "not allowed" in result.lower()
    
    def test_gate_pre_check_custom_message(self):
        """Gate uses custom message from pre-check."""
        g = gate(
            EchoTool(),
            pre=lambda p: "Custom block reason" if not p.get("ok") else True,
        )
        
        result = g(input={"query": "test", "ok": False})
        assert "custom block reason" in result.lower()
    
    def test_gate_post_check_pass(self):
        """Gate passes when post-check returns True."""
        g = gate(
            EchoTool(),
            post=lambda r: "echo" in r,
        )
        
        result = g(input={"query": "test"})
        assert "echo:" in result
    
    def test_gate_post_check_block(self):
        """Gate blocks when post-check returns False."""
        g = gate(
            EchoTool(),
            post=lambda r: "forbidden" not in r.lower(),
            on_block="Output contained forbidden content",
        )
        
        # Make echo return something with "forbidden"
        class ForbiddenEcho(EchoTool):
            def _run(self, params):
                return f"forbidden: {params.query}"
        
        g2 = gate(ForbiddenEcho(), post=lambda r: "forbidden" not in r.lower())
        result = g2(input={"query": "test"})
        assert "check failed" in result.lower()
    
    def test_gate_transform(self):
        """Gate transforms params before execution."""
        g = gate(
            EchoTool(),
            transform=lambda p: {**p, "query": p.get("query", "").upper()},
        )
        
        result = g(input={"query": "test"})
        assert "TEST" in result
    
    def test_gate_combined(self):
        """Gate with both pre and post checks."""
        g = gate(
            EchoTool(),
            pre=lambda p: p.get("auth") == "valid",
            post=lambda r: len(r) < 100,
            transform=lambda p: {**p, "query": p["query"].strip()},
        )
        
        result = g(input={"query": "  test  ", "auth": "valid"})
        assert "echo: test" in result
    
    def test_gate_metadata(self):
        """Gate derives appropriate metadata."""
        g = gate(EchoTool(), name="my_gate")
        assert g.metadata.name == "my_gate"
        assert g.metadata.category == "agents"


# ═══════════════════════════════════════════════════════════════════════════════
# Composition Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestComposition:
    """Tests for composing multiple agentic primitives."""
    
    @pytest.mark.asyncio
    async def test_gated_router(self):
        """Gate can wrap a router."""
        r = router(default=EchoTool())
        g = gate(r, pre=lambda p: p.get("input", {}).get("auth") is not None)
        
        # Should block without auth
        result = await g.acall(input={"input": {"query": "test"}})
        assert "check failed" in result.lower() or "permission" in result.lower()
        
        # Should pass with auth
        result = await g.acall(input={"input": {"query": "test", "auth": "token"}})
        assert "echo:" in result
    
    @pytest.mark.asyncio
    async def test_fallback_with_gates(self):
        """Fallback can use gated tools."""
        # Note: Gate expects {input: {...}} format, so we pass nested structure
        primary = gate(FailingTool(), pre=lambda p: True)
        backup = gate(EchoTool("backup"), pre=lambda p: True)
        
        f = fallback(primary, backup)
        # Pass input in the format that GateParams expects
        result = await f.acall(input={"input": {"query": "test"}})
        assert "backup:" in result
    
    @pytest.mark.asyncio
    async def test_race_with_routers(self):
        """Race can include routers."""
        r1 = router(default=SlowTool(delay=1.0))
        r2 = router(default=EchoTool("fast"))
        
        raced = race(r1, r2, timeout=5.0)
        result = await raced.acall(input={"input": {"query": "test"}})
        assert "fast:" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge case and error handling tests."""
    
    def test_router_condition_exception(self):
        """Router handles exceptions in conditions gracefully."""
        def bad_condition(p):
            raise ValueError("Condition error")
        
        r = router(
            (bad_condition, EchoTool("bad")),
            default=EchoTool("default"),
        )
        
        # Should fall through to default
        result = r(input={"query": "test"})
        assert "default:" in result
    
    def test_empty_tools_errors(self):
        """Primitives raise on empty tool lists."""
        with pytest.raises(ValueError):
            fallback()
        
        with pytest.raises(ValueError):
            race()
    
    def test_gate_check_exception(self):
        """Gate handles exceptions in checks."""
        def bad_check(p):
            raise ValueError("Check error")
        
        g = gate(EchoTool(), pre=bad_check)
        result = g(input={"query": "test"})
        assert "failed" in result.lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_safety(self):
        """Primitives are safe for concurrent use."""
        f = fallback(EchoTool("a"), EchoTool("b"))
        
        # Run many concurrent requests
        results = await asyncio.gather(*[
            f.acall(input={"query": f"test-{i}"})
            for i in range(10)
        ])
        
        assert all("a:" in r for r in results)
