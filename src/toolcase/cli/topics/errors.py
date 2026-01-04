ERRORS = """
TOPIC: errors
=============

Error handling infrastructure in toolcase.

ERROR CODES (ErrorCode enum):
    INVALID_PARAMS     Invalid or missing parameters
    NOT_FOUND          Resource not found
    PERMISSION_DENIED  Authorization failure
    TIMEOUT            Operation timed out
    NETWORK_ERROR      Network/connection failure
    RATE_LIMITED       Rate limit exceeded
    INTERNAL_ERROR     Internal tool error
    EXTERNAL_ERROR     External service error
    NOT_IMPLEMENTED    Feature not implemented

TOOLERROR (Structured error response):
    from toolcase import ToolError, ErrorCode
    
    error = ToolError(
        code=ErrorCode.NETWORK_ERROR,
        message="Connection refused",
        tool_name="http_tool",
        details={"host": "api.example.com"},
        recoverable=True,
    )

STRING-BASED ERROR (Traditional):
    def _run(self, params):
        if not params.query:
            return self._error("Query required", ErrorCode.INVALID_PARAMS)
        try:
            return external_call()
        except Exception as e:
            return self._error_from_exception(e)

RESULT-BASED ERROR (Recommended):
    from toolcase import tool_result, ErrorCode
    
    def _run_result(self, params):
        if not params.query:
            return tool_result(
                "my_tool",
                "Query required",
                code=ErrorCode.INVALID_PARAMS
            )
        return Ok(external_call())

RELATED TOPICS:
    toolcase help result     Monadic Result types
    toolcase help tool       Tool creation
"""
