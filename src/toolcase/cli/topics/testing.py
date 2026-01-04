TESTING = """
TOPIC: testing
==============

Testing utilities and effect handlers for pure, deterministic tests.

TEST CASE BASE:
    from toolcase import ToolTestCase
    
    class TestMyTool(ToolTestCase):
        def setUp(self):
            self.tool = MyTool()
        
        def test_success(self):
            result = self.tool.call(query="test")
            self.assertSuccess(result)
        
        def test_error(self):
            result = self.tool.call(query="")
            self.assertError(result, ErrorCode.INVALID_PARAMS)

MOCK TOOL:
    from toolcase import mock_tool, MockTool
    
    # Simple mock
    mock = mock_tool("search", return_value="mocked result")
    
    # With tracking
    mock = MockTool("search")
    mock.set_return("result")
    
    result = mock.call(query="test")
    
    assert mock.call_count == 1
    assert mock.last_call.params == {"query": "test"}

MOCK API:
    from toolcase import mock_api, mock_api_with_errors, mock_api_slow
    
    # Successful responses
    api = mock_api([
        {"status": 200, "body": {"data": "value"}},
    ])
    
    # With errors
    api = mock_api_with_errors(
        success_rate=0.8,
        error_codes=[500, 503],
    )
    
    # Simulate latency
    api = mock_api_slow(delay_ms=500)

FIXTURES:
    from toolcase import fixture
    
    @fixture
    def sample_params():
        return {"query": "test", "limit": 5}

EFFECT HANDLERS (Pure Testing):
    from toolcase.foundation.effects import (
        test_effects, InMemoryDB, RecordingHTTP, InMemoryFS,
        FrozenTime, SeededRandom, CollectingLogger, NoOpCache,
    )
    
    # Test with pure handlers (no real I/O)
    async with test_effects(
        db=InMemoryDB(),
        http=RecordingHTTP(),
        cache=NoOpCache(),
    ):
        result = await my_tool.arun(params)
    
    # Verify DB queries
    db = InMemoryDB()
    db.set_response("SELECT", [{"id": 1, "name": "test"}])
    async with test_effects(db=db):
        await my_tool.arun(params)
    assert "SELECT" in db.queries[0]
    
    # Deterministic time
    frozen = FrozenTime()
    frozen.freeze(datetime(2024, 1, 1, 12, 0, 0))
    async with test_effects(time=frozen):
        await time_sensitive_tool.arun(params)
    
    # Reproducible randomness
    rng = SeededRandom(_seed=42)
    async with test_effects(random=rng):
        await randomized_tool.arun(params)

RESULT ASSERTIONS:
    from toolcase import Result, Ok, Err
    
    # Pattern matching
    result = tool.run_result(params)
    match result:
        case Ok(value):
            assert "expected" in value
        case Err(error):
            pytest.fail(f"Unexpected error: {error}")
    
    # Direct assertions
    assert result.is_ok()
    assert result.unwrap() == expected_value
    
    # Error checking
    assert result.is_err()
    assert "validation" in result.unwrap_err()

ASYNC TESTING:
    import pytest
    
    @pytest.mark.asyncio
    async def test_async_tool():
        tool = MyAsyncTool()
        result = await tool.arun(MyParams(query="test"))
        assert "test" in result

RELATED TOPICS:
    toolcase help tool     Tool creation
    toolcase help result   Result types
    toolcase help effects  Effect system for pure testing
"""
