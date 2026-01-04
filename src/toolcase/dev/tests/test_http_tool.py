"""Tests for HttpTool - built-in HTTP request tool."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError

from toolcase.tools import (
    HttpTool,
    HttpConfig,
    HttpParams,
    HttpResponse,
    BearerAuth,
    BasicAuth,
    ApiKeyAuth,
    CustomAuth,
    NoAuth,
    standard_tools,
)
from toolcase.tools.prebuilt.http import AuthStrategy


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHttpConfig:
    """Test HttpConfig validation and defaults."""
    
    def test_default_config(self) -> None:
        """Default config has sensible security defaults."""
        config = HttpConfig()
        assert config.enabled is True
        assert config.timeout == 30.0
        assert config.verify_ssl is True
        assert config.follow_redirects is True
        assert "localhost" in config.blocked_hosts
        assert "127.0.0.1" in config.blocked_hosts
        assert config.max_response_size_bytes == 10 * 1024 * 1024
    
    def test_allowed_hosts_empty_means_all(self) -> None:
        """Empty allowed_hosts allows all (except blocked)."""
        config = HttpConfig(allowed_hosts=[])
        assert config.allowed_hosts == frozenset()
        assert len(config.allowed_hosts) == 0
    
    def test_allowed_hosts_restricts(self) -> None:
        """Allowed hosts can be set."""
        config = HttpConfig(allowed_hosts=["api.example.com", "*.internal.corp"])
        assert "api.example.com" in config.allowed_hosts
        assert "*.internal.corp" in config.allowed_hosts
    
    def test_method_normalization(self) -> None:
        """Methods are normalized to uppercase."""
        config = HttpConfig(allowed_methods={"get", "post"})  # type: ignore[arg-type]
        assert "GET" in config.allowed_methods
        assert "POST" in config.allowed_methods
    
    def test_timeout_bounds(self) -> None:
        """Timeout must be in valid range."""
        config = HttpConfig(timeout=0.1)
        assert config.timeout == 0.1
        
        config = HttpConfig(timeout=300.0)
        assert config.timeout == 300.0
        
        with pytest.raises(ValidationError):
            HttpConfig(timeout=0.01)
        
        with pytest.raises(ValidationError):
            HttpConfig(timeout=500.0)


# ─────────────────────────────────────────────────────────────────────────────
# Auth Strategy Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthStrategies:
    """Test authentication strategies."""
    
    def test_no_auth(self) -> None:
        """NoAuth doesn't modify headers."""
        auth = NoAuth()
        headers: dict[str, str] = {"Existing": "Header"}
        result = auth.apply(headers)
        assert result == {"Existing": "Header"}
    
    def test_bearer_auth(self) -> None:
        """BearerAuth adds Authorization header."""
        auth = BearerAuth(token="sk-test-token")
        headers: dict[str, str] = {}
        result = auth.apply(headers)
        assert result["Authorization"] == "Bearer sk-test-token"
    
    def test_basic_auth(self) -> None:
        """BasicAuth adds Base64 encoded credentials."""
        auth = BasicAuth(username="user", password="pass")
        headers: dict[str, str] = {}
        result = auth.apply(headers)
        # user:pass base64 encoded
        assert result["Authorization"] == "Basic dXNlcjpwYXNz"
    
    def test_api_key_auth_default_header(self) -> None:
        """ApiKeyAuth uses X-API-Key by default."""
        auth = ApiKeyAuth(key="my-api-key")
        headers: dict[str, str] = {}
        result = auth.apply(headers)
        assert result["X-API-Key"] == "my-api-key"
    
    def test_api_key_auth_custom_header(self) -> None:
        """ApiKeyAuth supports custom header name."""
        auth = ApiKeyAuth(key="my-api-key", header_name="Authorization")
        headers: dict[str, str] = {}
        result = auth.apply(headers)
        assert result["Authorization"] == "my-api-key"
    
    def test_custom_auth(self) -> None:
        """CustomAuth adds arbitrary headers."""
        auth = CustomAuth(headers={"X-Custom": "Value", "X-Another": "Header"})
        headers: dict[str, str] = {"Existing": "Header"}
        result = auth.apply(headers)
        assert result["Existing"] == "Header"
        assert result["X-Custom"] == "Value"
        assert result["X-Another"] == "Header"


# ─────────────────────────────────────────────────────────────────────────────
# URL Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUrlValidation:
    """Test URL security validation."""
    
    def test_valid_https_url(self) -> None:
        """Valid HTTPS URL passes validation."""
        tool = HttpTool(HttpConfig(blocked_hosts=[]))
        result = tool._validate_url("https://api.example.com/data")
        assert result.is_ok()
    
    def test_valid_http_url(self) -> None:
        """Valid HTTP URL passes validation."""
        tool = HttpTool(HttpConfig(blocked_hosts=[]))
        result = tool._validate_url("http://api.example.com/data")
        assert result.is_ok()
    
    def test_invalid_scheme_rejected(self) -> None:
        """Non-HTTP schemes are rejected."""
        tool = HttpTool()
        result = tool._validate_url("ftp://files.example.com/data")
        assert result.is_err()
        assert "scheme" in result.unwrap_err().message.lower()
    
    def test_localhost_blocked_by_default(self) -> None:
        """Localhost is blocked by default (SSRF protection)."""
        tool = HttpTool()
        result = tool._validate_url("http://localhost:8080/api")
        assert result.is_err()
        assert "blocked" in result.unwrap_err().message.lower()
    
    def test_127_0_0_1_blocked_by_default(self) -> None:
        """127.0.0.1 is blocked by default."""
        tool = HttpTool()
        result = tool._validate_url("http://127.0.0.1:8080/api")
        assert result.is_err()
    
    def test_allowed_hosts_restricts(self) -> None:
        """Only allowed hosts pass when configured."""
        tool = HttpTool(HttpConfig(
            allowed_hosts=["api.example.com"],
            blocked_hosts=[],
        ))
        
        # Allowed
        result = tool._validate_url("https://api.example.com/data")
        assert result.is_ok()
        
        # Not allowed
        result = tool._validate_url("https://other.com/data")
        assert result.is_err()
        assert "not in allowed" in result.unwrap_err().message.lower()
    
    def test_glob_patterns_in_allowed_hosts(self) -> None:
        """Glob patterns work for allowed hosts."""
        tool = HttpTool(HttpConfig(
            allowed_hosts=["*.example.com", "api.*.internal"],
            blocked_hosts=[],
        ))
        
        assert tool._validate_url("https://api.example.com/data").is_ok()
        assert tool._validate_url("https://cdn.example.com/images").is_ok()
        assert tool._validate_url("https://other.net/data").is_err()
    
    def test_blocked_takes_precedence(self) -> None:
        """Blocked hosts override allowed hosts."""
        tool = HttpTool(HttpConfig(
            allowed_hosts=["*.example.com"],
            blocked_hosts=["secret.example.com"],
        ))
        
        assert tool._validate_url("https://api.example.com/data").is_ok()
        assert tool._validate_url("https://secret.example.com/data").is_err()


# ─────────────────────────────────────────────────────────────────────────────
# Method Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMethodValidation:
    """Test HTTP method validation."""
    
    def test_all_methods_allowed_by_default(self) -> None:
        """All standard methods allowed by default."""
        tool = HttpTool()
        for method in ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]:
            result = tool._validate_method(method)  # type: ignore[arg-type]
            assert result.is_ok(), f"{method} should be allowed"
    
    def test_restricted_methods(self) -> None:
        """Only specified methods allowed when configured."""
        tool = HttpTool(HttpConfig(allowed_methods={"GET", "POST"}))
        
        assert tool._validate_method("GET").is_ok()
        assert tool._validate_method("POST").is_ok()
        assert tool._validate_method("DELETE").is_err()
        assert tool._validate_method("PUT").is_err()


# ─────────────────────────────────────────────────────────────────────────────
# Parameters Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHttpParams:
    """Test HttpParams validation."""
    
    def test_minimal_params(self) -> None:
        """URL is the only required parameter."""
        params = HttpParams(url="https://api.example.com")
        assert params.method == "GET"
        assert params.headers == {}
        assert params.body is None
    
    def test_method_normalized(self) -> None:
        """Method is normalized to uppercase."""
        params = HttpParams(url="https://api.example.com", method="post")  # type: ignore[arg-type]
        assert params.method == "POST"
    
    def test_json_body(self) -> None:
        """JSON body is accepted."""
        params = HttpParams(
            url="https://api.example.com",
            method="POST",
            json_body={"key": "value"},
        )
        assert params.json_body == {"key": "value"}
    
    def test_timeout_bounds(self) -> None:
        """Timeout must be in valid range."""
        params = HttpParams(url="https://api.example.com", timeout=60.0)
        assert params.timeout == 60.0
        
        with pytest.raises(ValidationError):
            HttpParams(url="https://api.example.com", timeout=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Response Formatting Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHttpResponse:
    """Test HttpResponse formatting."""
    
    def test_to_output_format(self) -> None:
        """Response formats correctly for LLM consumption."""
        response = HttpResponse(
            status_code=200,
            headers={"Content-Type": "application/json", "Content-Length": "42"},
            body='{"success": true}',
            url="https://api.example.com/data",
            elapsed_ms=150.5,
        )
        
        output = response.to_output()
        assert "**HTTP 200**" in output
        assert "150ms" in output
        assert "https://api.example.com/data" in output
        assert "Content-Type: application/json" in output
        assert '{"success": true}' in output


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests (Mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestHttpToolExecution:
    """Test HttpTool execution with mocked httpx."""
    
    @pytest.mark.asyncio
    async def test_successful_get_request(self) -> None:
        """Successful GET request returns formatted response."""
        tool = HttpTool(HttpConfig(blocked_hosts=[]))
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.url = "https://api.example.com/data"
        mock_response.aread = AsyncMock(return_value=b'{"result": "success"}')
        
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        
        with patch.object(tool, "_get_client", return_value=mock_client):
            params = HttpParams(url="https://api.example.com/data")
            result = await tool._async_run_result(params)
            
            assert result.is_ok()
            output = result.unwrap()
            assert "HTTP 200" in output
            assert "success" in output
    
    @pytest.mark.asyncio
    async def test_post_with_json_body(self) -> None:
        """POST with JSON body sets headers correctly."""
        tool = HttpTool(HttpConfig(blocked_hosts=[]))
        
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.url = "https://api.example.com/data"
        mock_response.aread = AsyncMock(return_value=b'{"id": 123}')
        
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        
        with patch.object(tool, "_get_client", return_value=mock_client):
            params = HttpParams(
                url="https://api.example.com/data",
                method="POST",
                json_body={"name": "test"},
            )
            result = await tool._async_run_result(params)
            
            assert result.is_ok()
            # Verify request was made with correct content
            call_kwargs = mock_client.request.call_args.kwargs
            assert call_kwargs["method"] == "POST"
            assert "application/json" in str(call_kwargs.get("headers", {}))
    
    @pytest.mark.asyncio
    async def test_auth_applied_to_request(self) -> None:
        """Auth strategy is applied to request headers."""
        tool = HttpTool(HttpConfig(
            blocked_hosts=[],
            auth=BearerAuth(token="test-token"),
        ))
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.url = "https://api.example.com/data"
        mock_response.aread = AsyncMock(return_value=b"ok")
        
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        
        with patch.object(tool, "_get_client", return_value=mock_client):
            params = HttpParams(url="https://api.example.com/data")
            await tool._async_run_result(params)
            
            call_kwargs = mock_client.request.call_args.kwargs
            assert "Bearer test-token" in str(call_kwargs.get("headers", {}))
    
    @pytest.mark.asyncio
    async def test_blocked_host_rejected(self) -> None:
        """Blocked hosts return error without making request."""
        tool = HttpTool()  # Default blocks localhost
        
        params = HttpParams(url="http://localhost:8080/api")
        result = await tool._async_run_result(params)
        
        assert result.is_err()
        assert "blocked" in result.unwrap_err().message.lower()
    
    @pytest.mark.asyncio
    async def test_response_size_limit(self) -> None:
        """Large responses are rejected."""
        tool = HttpTool(HttpConfig(
            blocked_hosts=[],
            max_response_size=1024,  # Minimum allowed (1KB)
        ))
        
        mock_headers = MagicMock()
        mock_headers.get = MagicMock(return_value="10000")  # 10KB > 1KB limit
        mock_headers.__iter__ = MagicMock(return_value=iter([]))
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = mock_headers
        mock_response.url = "https://api.example.com/large"
        
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        
        with patch.object(tool, "_get_client", return_value=mock_client):
            params = HttpParams(url="https://api.example.com/large")
            result = await tool._async_run_result(params)
            
            assert result.is_err()
            assert "too large" in result.unwrap_err().message.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Mutability Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigurability:
    """Test runtime configuration changes."""
    
    def test_configure_updates_config(self) -> None:
        """configure() updates config in place."""
        tool = HttpTool()
        original_timeout = tool.config.timeout
        
        tool.configure(timeout=60.0)
        
        assert tool.config.timeout == 60.0
        assert tool.config.timeout != original_timeout
    
    def test_with_config_returns_new_instance(self) -> None:
        """with_config() returns new instance."""
        tool = HttpTool()
        
        new_tool = tool.with_config(timeout=60.0)
        
        assert new_tool is not tool
        assert new_tool.config.timeout == 60.0
        assert tool.config.timeout == 30.0  # Original unchanged


# ─────────────────────────────────────────────────────────────────────────────
# Standard Tools Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStandardTools:
    """Test standard_tools() utility."""
    
    def test_returns_list_of_tools(self) -> None:
        """standard_tools() returns list of tool instances."""
        tools = standard_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 2  # At least Discovery and Http
    
    def test_includes_http_tool(self) -> None:
        """HttpTool is included in standard tools."""
        tools = standard_tools()
        names = [t.metadata.name for t in tools]
        assert "http_request" in names
    
    def test_includes_discovery_tool(self) -> None:
        """DiscoveryTool is included in standard tools."""
        tools = standard_tools()
        names = [t.metadata.name for t in tools]
        assert "discover_tools" in names
    
    def test_tools_have_valid_metadata(self) -> None:
        """All tools have valid metadata."""
        tools = standard_tools()
        for tool in tools:
            assert tool.metadata.name
            assert len(tool.metadata.description) >= 10
