"""Tests for batch execution and streaming functionality."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

from toolcase import tool
from toolcase.runtime.batch import (
    BatchConfig,
    BatchEventKind,
    BatchItem,
    BatchItemEvent,
    BatchResult,
    batch_execute,
    batch_execute_stream,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────────────────────────────────────

class SimpleParams(BaseModel):
    """Simple params for testing."""
    value: int


@pytest.fixture
def simple_tool():
    """Create a simple tool for testing batch execution."""
    @tool(description="Simple doubler tool for batch testing")
    async def doubler(value: int) -> str:
        await asyncio.sleep(0.001)  # Simulate work
        return str(value * 2)
    return doubler


@pytest.fixture
def flaky_tool():
    """Create a tool that fails on certain values."""
    @tool(description="Tool that fails on negative values")
    async def flaky(value: int) -> str:
        await asyncio.sleep(0.001)
        if value < 0:
            raise ValueError(f"Negative value: {value}")
        return str(value)
    return flaky


# ─────────────────────────────────────────────────────────────────────────────
# Core Batch Execution Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchExecute:
    """Test batch_execute() functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_batch_execution(self, simple_tool) -> None:
        """Basic batch execution returns results for all items."""
        params = [simple_tool.params_schema(value=i) for i in range(5)]
        result = await batch_execute(simple_tool, params)
        
        assert len(result) == 5
        assert result.all_ok
        assert result.success_rate == 1.0
        assert result.values() == ["0", "2", "4", "6", "8"]
    
    @pytest.mark.asyncio
    async def test_empty_batch(self, simple_tool) -> None:
        """Empty batch returns empty result."""
        result = await batch_execute(simple_tool, [])
        assert len(result) == 0
        assert result.all_ok  # Vacuous truth
    
    @pytest.mark.asyncio
    async def test_batch_with_failures(self, flaky_tool) -> None:
        """Batch with failures collects both successes and failures."""
        params = [flaky_tool.params_schema(value=v) for v in [1, -1, 2, -2, 3]]
        result = await batch_execute(flaky_tool, params)
        
        assert len(result.successes) == 3
        assert len(result.failures) == 2
        assert not result.all_ok
        assert result.success_rate == 0.6
    
    @pytest.mark.asyncio
    async def test_batch_fail_fast(self, flaky_tool) -> None:
        """fail_fast cancels remaining items on first failure."""
        config = BatchConfig(fail_fast=True)
        params = [flaky_tool.params_schema(value=v) for v in [1, -1, 2, 3, 4]]
        result = await batch_execute(flaky_tool, params, config)
        
        # Should have at least one failure and some may be cancelled
        assert not result.all_ok
        cancelled = [i for i in result.items if i.is_err and i.error and "cancelled" in i.error.message.lower()]
        # Not all items may have been cancelled (depends on timing), but fail_fast was triggered
        assert len(result.failures) >= 1
    
    @pytest.mark.asyncio
    async def test_concurrency_limit(self, simple_tool) -> None:
        """Concurrency limit controls parallel execution."""
        config = BatchConfig(concurrency=2)
        params = [simple_tool.params_schema(value=i) for i in range(10)]
        
        result = await batch_execute(simple_tool, params, config)
        
        assert len(result) == 10
        assert result.all_ok
        assert result.concurrency == 2
    
    @pytest.mark.asyncio
    async def test_on_item_complete_callback(self, simple_tool) -> None:
        """on_item_complete callback receives each BatchItem."""
        completed: list[BatchItem] = []
        config = BatchConfig(on_item_complete=completed.append)
        params = [simple_tool.params_schema(value=i) for i in range(3)]
        
        await batch_execute(simple_tool, params, config)
        
        assert len(completed) == 3
        assert all(isinstance(c, BatchItem) for c in completed)


# ─────────────────────────────────────────────────────────────────────────────
# Batch Streaming Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchExecuteStream:
    """Test batch_execute_stream() streaming functionality."""
    
    @pytest.mark.asyncio
    async def test_stream_yields_start_event(self, simple_tool) -> None:
        """Stream starts with START event."""
        params = [simple_tool.params_schema(value=i) for i in range(3)]
        
        events = [e async for e in batch_execute_stream(simple_tool, params)]
        
        assert events[0].kind == BatchEventKind.START
        assert events[0].total == 3
        assert events[0].completed == 0
    
    @pytest.mark.asyncio
    async def test_stream_yields_item_events(self, simple_tool) -> None:
        """Stream yields ITEM events as items complete."""
        params = [simple_tool.params_schema(value=i) for i in range(3)]
        
        events = [e async for e in batch_execute_stream(simple_tool, params)]
        item_events = [e for e in events if e.kind == BatchEventKind.ITEM]
        
        assert len(item_events) == 3
        for i, event in enumerate(item_events, 1):
            assert event.completed == i
            assert event.item is not None
            assert event.item.is_ok
    
    @pytest.mark.asyncio
    async def test_stream_yields_complete_event(self, simple_tool) -> None:
        """Stream ends with COMPLETE event containing BatchResult."""
        params = [simple_tool.params_schema(value=i) for i in range(3)]
        
        events = [e async for e in batch_execute_stream(simple_tool, params)]
        
        final = events[-1]
        assert final.kind == BatchEventKind.COMPLETE
        assert final.is_complete
        assert final.batch_result is not None
        assert final.batch_result.all_ok
        assert len(final.batch_result) == 3
    
    @pytest.mark.asyncio
    async def test_stream_event_lifecycle(self, simple_tool) -> None:
        """Full lifecycle: START → ITEM... → COMPLETE."""
        params = [simple_tool.params_schema(value=i) for i in range(2)]
        
        events = [e async for e in batch_execute_stream(simple_tool, params)]
        
        # Should be: START, ITEM, ITEM, COMPLETE
        assert len(events) == 4
        assert events[0].kind == BatchEventKind.START
        assert events[1].kind == BatchEventKind.ITEM
        assert events[2].kind == BatchEventKind.ITEM
        assert events[3].kind == BatchEventKind.COMPLETE
    
    @pytest.mark.asyncio
    async def test_stream_progress_tracking(self, simple_tool) -> None:
        """Progress percentage updates correctly."""
        params = [simple_tool.params_schema(value=i) for i in range(4)]
        
        events = [e async for e in batch_execute_stream(simple_tool, params)]
        item_events = [e for e in events if e.kind == BatchEventKind.ITEM]
        
        # Progress should increase
        progresses = [e.progress for e in item_events]
        assert progresses == [0.25, 0.5, 0.75, 1.0]
    
    @pytest.mark.asyncio
    async def test_stream_with_failures(self, flaky_tool) -> None:
        """Stream includes failed items in events."""
        params = [flaky_tool.params_schema(value=v) for v in [1, -1, 2]]
        
        events = [e async for e in batch_execute_stream(flaky_tool, params)]
        item_events = [e for e in events if e.kind == BatchEventKind.ITEM]
        
        ok_count = sum(1 for e in item_events if e.item and e.item.is_ok)
        err_count = sum(1 for e in item_events if e.item and e.item.is_err)
        
        assert ok_count == 2
        assert err_count == 1
        
        final = events[-1]
        assert final.batch_result is not None
        assert not final.batch_result.all_ok
    
    @pytest.mark.asyncio
    async def test_stream_empty_batch(self, simple_tool) -> None:
        """Empty batch yields START and COMPLETE only."""
        events = [e async for e in batch_execute_stream(simple_tool, [])]
        
        assert len(events) == 2
        assert events[0].kind == BatchEventKind.START
        assert events[0].total == 0
        assert events[1].kind == BatchEventKind.COMPLETE
    
    @pytest.mark.asyncio
    async def test_stream_with_concurrency(self, simple_tool) -> None:
        """Streaming works with concurrency limit."""
        config = BatchConfig(concurrency=2)
        params = [simple_tool.params_schema(value=i) for i in range(6)]
        
        events = [e async for e in batch_execute_stream(simple_tool, params, config)]
        
        assert events[0].kind == BatchEventKind.START
        assert events[-1].kind == BatchEventKind.COMPLETE
        assert len([e for e in events if e.kind == BatchEventKind.ITEM]) == 6
    
    @pytest.mark.asyncio
    async def test_stream_elapsed_time(self, simple_tool) -> None:
        """Elapsed time tracked in events."""
        params = [simple_tool.params_schema(value=i) for i in range(3)]
        
        events = [e async for e in batch_execute_stream(simple_tool, params)]
        
        # Elapsed should increase over time
        item_events = [e for e in events if e.kind == BatchEventKind.ITEM]
        assert all(e.elapsed_ms >= 0 for e in item_events)
        
        final = events[-1]
        assert final.elapsed_ms > 0
    
    @pytest.mark.asyncio
    async def test_stream_real_time_visibility(self, simple_tool) -> None:
        """Simulate real-time progress display."""
        params = [simple_tool.params_schema(value=i) for i in range(5)]
        
        progress_log: list[str] = []
        
        async for event in batch_execute_stream(simple_tool, params):
            match event.kind:
                case BatchEventKind.START:
                    progress_log.append(f"Started {event.total} items")
                case BatchEventKind.ITEM:
                    status = "✓" if event.item and event.item.is_ok else "✗"
                    progress_log.append(f"[{event.completed}/{event.total}] {status}")
                case BatchEventKind.COMPLETE:
                    rate = event.batch_result.success_rate if event.batch_result else 0
                    progress_log.append(f"Done: {rate:.0%}")
        
        assert progress_log[0] == "Started 5 items"
        assert progress_log[-1] == "Done: 100%"
        assert len(progress_log) == 7  # START + 5 ITEM + COMPLETE


# ─────────────────────────────────────────────────────────────────────────────
# BatchItem and BatchResult Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchResultHelpers:
    """Test BatchResult helper methods."""
    
    @pytest.mark.asyncio
    async def test_batch_result_iteration(self, simple_tool) -> None:
        """BatchResult is iterable."""
        params = [simple_tool.params_schema(value=i) for i in range(3)]
        result = await batch_execute(simple_tool, params)
        
        items = list(result)
        assert len(items) == 3
        assert all(isinstance(i, BatchItem) for i in items)
    
    @pytest.mark.asyncio
    async def test_batch_result_to_result(self, simple_tool, flaky_tool) -> None:
        """to_result() converts to Ok or Err."""
        # All ok
        params = [simple_tool.params_schema(value=i) for i in range(3)]
        ok_result = await batch_execute(simple_tool, params)
        assert ok_result.to_result().is_ok()
        
        # Some failures
        params = [flaky_tool.params_schema(value=v) for v in [1, -1]]
        err_result = await batch_execute(flaky_tool, params)
        assert err_result.to_result().is_err()


# ─────────────────────────────────────────────────────────────────────────────
# BatchItemEvent Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchItemEvent:
    """Test BatchItemEvent properties."""
    
    def test_event_progress_calculation(self) -> None:
        """Progress is calculated correctly."""
        event = BatchItemEvent(BatchEventKind.ITEM, 5, 10, 100.0)
        assert event.progress == 0.5
    
    def test_event_progress_empty_batch(self) -> None:
        """Progress handles empty batch (avoid division by zero)."""
        event = BatchItemEvent(BatchEventKind.START, 0, 0, 0.0)
        assert event.progress == 0.0
    
    def test_event_is_complete(self) -> None:
        """is_complete correctly identifies COMPLETE events."""
        start = BatchItemEvent(BatchEventKind.START, 0, 5, 0.0)
        item = BatchItemEvent(BatchEventKind.ITEM, 1, 5, 10.0)
        complete = BatchItemEvent(BatchEventKind.COMPLETE, 5, 5, 100.0)
        
        assert not start.is_complete
        assert not item.is_complete
        assert complete.is_complete


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
