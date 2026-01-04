TESTING = """
TOPIC: testing
==============

Testing utilities for tools.

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
    from toolcase import mock_api, mock_api_with_errors
    
    # Successful responses
    api = mock_api([
        {"status": 200, "body": {"data": "value"}},
    ])
    
    # With errors
    api = mock_api_with_errors(
        success_rate=0.8,
        error_codes=[500, 503],
    )

FIXTURES:
    from toolcase import fixture
    
    @fixture
    def sample_params():
        return {"query": "test", "limit": 5}

RELATED TOPICS:
    toolcase help tool     Tool creation
    toolcase help result   Result types
"""
