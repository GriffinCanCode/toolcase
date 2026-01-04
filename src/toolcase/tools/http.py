"""HTTP Tool - Make HTTP requests to external APIs.

A production-ready HTTP tool with:
- Configurable allowed hosts/methods for security
- Multiple auth strategies (Bearer, Basic, API Key, Custom)
- Streaming support for large responses
- Proper timeout and error handling
- Response size limits

Example:
    >>> from toolcase.tools import HttpTool
    >>> 
    >>> # Basic usage (all hosts/methods allowed)
    >>> http = HttpTool()
    >>> result = await http.acall(url="https://api.example.com/data")
    >>> 
    >>> # Restricted to specific hosts
    >>> http = HttpTool(HttpConfig(
    ...     allowed_hosts=["api.example.com", "*.internal.corp"],
    ...     allowed_methods=["GET", "POST"],
    ...     default_timeout=10.0,
    ... ))
    >>> 
    >>> # With authentication
    >>> http = HttpTool(HttpConfig(
    ...     auth=BearerAuth(token="sk-xxx"),
    ... ))
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
from typing import Annotated, AsyncIterator, ClassVar, Literal, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Discriminator, Field, Tag, field_validator

from ..core import ToolMetadata
from ..errors import ErrorCode
from ..monads import Err, Ok, ToolResult
from ..monads.types import ErrorTrace
from .base import ConfigurableTool, ToolConfig

# Type alias for HTTP methods
HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
ALL_METHODS: frozenset[HttpMethod] = frozenset(["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])


# ─────────────────────────────────────────────────────────────────────────────
# Authentication Strategies (Protocol-based for Pydantic compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class NoAuth(BaseModel):
    """No authentication."""
    
    auth_type: Literal["none"] = "none"
    
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        return headers


class BearerAuth(BaseModel):
    """Bearer token authentication (OAuth2, JWT)."""
    
    auth_type: Literal["bearer"] = "bearer"
    token: str = Field(..., description="Bearer token value")
    
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        headers["Authorization"] = f"Bearer {self.token}"
        return headers


class BasicAuth(BaseModel):
    """HTTP Basic authentication."""
    
    auth_type: Literal["basic"] = "basic"
    username: str
    password: str
    
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        import base64
        credentials = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        headers["Authorization"] = f"Basic {credentials}"
        return headers


class ApiKeyAuth(BaseModel):
    """API key authentication (header or query param)."""
    
    auth_type: Literal["api_key"] = "api_key"
    key: str = Field(..., description="API key value")
    header_name: str = Field(default="X-API-Key", description="Header name for the key")
    
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        headers[self.header_name] = self.key
        return headers


class CustomAuth(BaseModel):
    """Custom header-based authentication."""
    
    auth_type: Literal["custom"] = "custom"
    headers: dict[str, str] = Field(default_factory=dict)
    
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        headers.update(self.headers)
        return headers


def _auth_discriminator(v: dict[str, object] | BaseModel) -> str:
    """Discriminator function for auth strategy union."""
    if isinstance(v, dict):
        return str(v.get("auth_type", "none"))
    return getattr(v, "auth_type", "none")


# Discriminated union for proper serialization/deserialization
AuthStrategy = Annotated[
    Annotated[NoAuth, Tag("none")]
    | Annotated[BearerAuth, Tag("bearer")]
    | Annotated[BasicAuth, Tag("basic")]
    | Annotated[ApiKeyAuth, Tag("api_key")]
    | Annotated[CustomAuth, Tag("custom")],
    Discriminator(_auth_discriminator),
]


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

class HttpConfig(ToolConfig):
    """Configuration for HttpTool.
    
    Security-focused defaults with customization options.
    
    Attributes:
        allowed_hosts: Glob patterns for allowed hosts. Empty = all allowed.
        blocked_hosts: Glob patterns for blocked hosts (takes precedence).
        allowed_methods: HTTP methods to allow. Empty = all allowed.
        max_response_size: Maximum response body size in bytes.
        default_timeout: Default request timeout in seconds.
        follow_redirects: Whether to follow HTTP redirects.
        verify_ssl: Whether to verify SSL certificates.
        auth: Default authentication strategy.
        default_headers: Headers added to every request.
    """
    
    allowed_hosts: list[str] = Field(
        default_factory=list,
        description="Glob patterns for allowed hosts (empty = all)",
    )
    blocked_hosts: list[str] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1", "0.0.0.0", "*.local", "169.254.*.*"],
        description="Glob patterns for blocked hosts (SSRF protection)",
    )
    allowed_methods: set[HttpMethod] = Field(
        default_factory=lambda: set(ALL_METHODS),
        description="Allowed HTTP methods",
    )
    max_response_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024,
        le=100 * 1024 * 1024,
        description="Max response size in bytes",
    )
    default_timeout: float = Field(default=30.0, ge=0.1, le=300.0)
    follow_redirects: bool = Field(default=True)
    verify_ssl: bool = Field(default=True)
    auth: AuthStrategy = Field(default_factory=NoAuth)
    default_headers: dict[str, str] = Field(
        default_factory=lambda: {"User-Agent": "toolcase-http/1.0"},
    )
    
    @field_validator("allowed_methods", mode="before")
    @classmethod
    def _normalize_methods(cls, v: set[str] | list[str]) -> set[HttpMethod]:
        return {m.upper() for m in v}  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

class HttpParams(BaseModel):
    """Parameters for HTTP requests.
    
    Attributes:
        url: The URL to request
        method: HTTP method (GET, POST, etc.)
        headers: Additional request headers
        query_params: URL query parameters
        body: Request body (for POST/PUT/PATCH)
        json_body: JSON body (auto-serialized, sets Content-Type)
        timeout: Request timeout override
    """
    
    url: str = Field(..., description="URL to request")
    method: HttpMethod = Field(default="GET", description="HTTP method")
    headers: dict[str, str] = Field(default_factory=dict, description="Additional headers")
    query_params: dict[str, str] = Field(default_factory=dict, description="Query parameters")
    body: str | None = Field(default=None, description="Request body (string)")
    json_body: dict[str, object] | list[object] | None = Field(
        default=None, description="JSON body (auto-serialized)"
    )
    timeout: float | None = Field(default=None, ge=0.1, le=300.0, description="Timeout override")
    
    @field_validator("method", mode="before")
    @classmethod
    def _upper_method(cls, v: str) -> str:
        return v.upper()


# ─────────────────────────────────────────────────────────────────────────────
# Response Model
# ─────────────────────────────────────────────────────────────────────────────

class HttpResponse(BaseModel):
    """Structured HTTP response for tool output."""
    
    status_code: int
    headers: dict[str, str]
    body: str
    url: str
    elapsed_ms: float
    
    def to_output(self) -> str:
        """Format as tool output string."""
        lines = [
            f"**HTTP {self.status_code}** ({self.elapsed_ms:.0f}ms)",
            f"URL: {self.url}",
            "",
        ]
        
        # Include relevant headers
        for key in ("Content-Type", "Content-Length", "Date", "Server"):
            if key.lower() in {k.lower() for k in self.headers}:
                val = next(v for k, v in self.headers.items() if k.lower() == key.lower())
                lines.append(f"{key}: {val}")
        
        lines.extend(["", "**Response:**", self.body])
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Tool
# ─────────────────────────────────────────────────────────────────────────────

class HttpTool(ConfigurableTool[HttpParams, HttpConfig]):
    """HTTP request tool with security controls and streaming support.
    
    Makes HTTP requests to external APIs with configurable security
    constraints, authentication, and response handling.
    
    Features:
        - Host allowlisting/blocklisting for SSRF protection
        - Multiple authentication strategies
        - Response streaming for large payloads
        - Configurable timeouts and limits
        - Structured response formatting
    
    Example:
        >>> http = HttpTool()
        >>> result = await http.acall(url="https://api.github.com/users/octocat")
        
        >>> # With POST and JSON body
        >>> result = await http.acall(
        ...     url="https://api.example.com/data",
        ...     method="POST",
        ...     json_body={"key": "value"},
        ... )
        
        >>> # Restricted configuration
        >>> http = HttpTool(HttpConfig(
        ...     allowed_hosts=["api.example.com"],
        ...     allowed_methods=["GET"],
        ...     auth=BearerAuth(token="sk-xxx"),
        ... ))
    """
    
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="http_request",
        description=(
            "Make HTTP requests to external APIs. Supports GET, POST, PUT, DELETE, PATCH. "
            "Can send JSON bodies, custom headers, and query parameters. "
            "Returns status code, headers, and response body."
        ),
        category="network",
        requires_api_key=False,
        streaming=True,
    )
    params_schema: ClassVar[type[HttpParams]] = HttpParams
    config_class: ClassVar[type[HttpConfig]] = HttpConfig
    
    cache_enabled: ClassVar[bool] = False  # HTTP requests shouldn't be cached by default
    
    def __init__(self, config: HttpConfig | None = None) -> None:
        super().__init__(config)
        self._client: object | None = None  # Lazy httpx client
    
    # ─────────────────────────────────────────────────────────────────
    # Security Validation
    # ─────────────────────────────────────────────────────────────────
    
    def _validate_url(self, url: str) -> ToolResult:
        """Validate URL against security constraints."""
        try:
            parsed = urlparse(url)
        except Exception as e:
            return Err(ErrorTrace(
                message=f"Invalid URL: {e}",
                error_code=ErrorCode.INVALID_PARAMS.value,
                recoverable=False,
            ).with_operation("url_validation"))
        
        # Require scheme
        if parsed.scheme not in ("http", "https"):
            return Err(ErrorTrace(
                message=f"Invalid scheme '{parsed.scheme}'. Use http or https.",
                error_code=ErrorCode.INVALID_PARAMS.value,
                recoverable=False,
            ).with_operation("url_validation"))
        
        host = parsed.hostname or ""
        
        # Check blocked hosts first (SSRF protection)
        for pattern in self.config.blocked_hosts:
            if fnmatch.fnmatch(host, pattern) or fnmatch.fnmatch(host.lower(), pattern.lower()):
                return Err(ErrorTrace(
                    message=f"Host '{host}' is blocked for security reasons.",
                    error_code=ErrorCode.PERMISSION_DENIED.value,
                    recoverable=False,
                ).with_operation("host_validation"))
        
        # Check allowed hosts (if configured)
        if self.config.allowed_hosts:
            if not any(fnmatch.fnmatch(host, p) or fnmatch.fnmatch(host.lower(), p.lower()) 
                      for p in self.config.allowed_hosts):
                return Err(ErrorTrace(
                    message=f"Host '{host}' not in allowed list.",
                    error_code=ErrorCode.PERMISSION_DENIED.value,
                    recoverable=False,
                ).with_operation("host_validation"))
        
        return Ok(url)
    
    def _validate_method(self, method: HttpMethod) -> ToolResult:
        """Validate HTTP method against allowed list."""
        if method not in self.config.allowed_methods:
            return Err(ErrorTrace(
                message=f"Method '{method}' not allowed. Allowed: {', '.join(sorted(self.config.allowed_methods))}",
                error_code=ErrorCode.PERMISSION_DENIED.value,
                recoverable=False,
            ).with_operation("method_validation"))
        return Ok(method)
    
    # ─────────────────────────────────────────────────────────────────
    # HTTP Client
    # ─────────────────────────────────────────────────────────────────
    
    async def _get_client(self) -> object:
        """Get or create httpx async client."""
        if self._client is None:
            try:
                import httpx
            except ImportError as e:
                raise ImportError(
                    "httpx is required for HttpTool. Install with: pip install httpx"
                ) from e
            
            self._client = httpx.AsyncClient(
                follow_redirects=self.config.follow_redirects,
                verify=self.config.verify_ssl,
                timeout=self.config.default_timeout,
            )
        return self._client
    
    async def _close_client(self) -> None:
        """Close the httpx client."""
        if self._client is not None:
            await self._client.aclose()  # type: ignore[union-attr]
            self._client = None
    
    # ─────────────────────────────────────────────────────────────────
    # Execution
    # ─────────────────────────────────────────────────────────────────
    
    def _run(self, params: HttpParams) -> str:
        """Sync execution via async wrapper."""
        return self._run_async_sync(self._async_run(params))
    
    async def _async_run_result(self, params: HttpParams) -> ToolResult:
        """Execute HTTP request with Result-based error handling."""
        import time
        
        # Validate URL
        url_result = self._validate_url(params.url)
        if url_result.is_err():
            return url_result
        
        # Validate method
        method_result = self._validate_method(params.method)
        if method_result.is_err():
            return method_result
        
        try:
            import httpx
        except ImportError:
            return Err(ErrorTrace(
                message="httpx not installed. Run: pip install httpx",
                error_code=ErrorCode.EXTERNAL_SERVICE_ERROR.value,
                recoverable=False,
            ).with_operation("import"))
        
        # Build headers
        headers = {**self.config.default_headers, **params.headers}
        self.config.auth.apply(headers)
        
        # Build body
        content: str | bytes | None = None
        if params.json_body is not None:
            content = json.dumps(params.json_body)
            headers.setdefault("Content-Type", "application/json")
        elif params.body is not None:
            content = params.body
        
        # Execute request
        start = time.perf_counter()
        try:
            client = await self._get_client()
            timeout = params.timeout or self.config.default_timeout
            
            response = await client.request(  # type: ignore[union-attr]
                method=params.method,
                url=params.url,
                headers=headers,
                params=params.query_params or None,
                content=content,
                timeout=timeout,
            )
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            # Check response size
            content_length = int(response.headers.get("content-length", 0))
            if content_length > self.config.max_response_size:
                return Err(ErrorTrace(
                    message=f"Response too large: {content_length} bytes (max: {self.config.max_response_size})",
                    error_code=ErrorCode.INVALID_PARAMS.value,
                    recoverable=False,
                ).with_operation("response_size_check"))
            
            # Read body with size limit
            body_bytes = await response.aread()
            if len(body_bytes) > self.config.max_response_size:
                return Err(ErrorTrace(
                    message=f"Response body exceeded max size: {len(body_bytes)} bytes",
                    error_code=ErrorCode.INVALID_PARAMS.value,
                    recoverable=False,
                ).with_operation("body_read"))
            
            body = body_bytes.decode("utf-8", errors="replace")
            
            # Build response
            http_response = HttpResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                body=body,
                url=str(response.url),
                elapsed_ms=elapsed_ms,
            )
            
            return Ok(http_response.to_output())
            
        except httpx.TimeoutException:
            return Err(ErrorTrace(
                message=f"Request timed out after {params.timeout or self.config.default_timeout}s",
                error_code=ErrorCode.TIMEOUT.value,
                recoverable=True,
            ).with_operation("request"))
        except httpx.NetworkError as e:
            return Err(ErrorTrace(
                message=f"Network error: {e}",
                error_code=ErrorCode.NETWORK_ERROR.value,
                recoverable=True,
            ).with_operation("request"))
        except Exception as e:
            return Err(ErrorTrace(
                message=f"Request failed: {e}",
                error_code=ErrorCode.EXTERNAL_SERVICE_ERROR.value,
                recoverable=True,
                details=str(type(e).__name__),
            ).with_operation("request"))
    
    async def _async_run(self, params: HttpParams) -> str:
        """Execute HTTP request."""
        from ..monads.tool import result_to_string
        result = await self._async_run_result(params)
        return result_to_string(result, self.metadata.name)
    
    # ─────────────────────────────────────────────────────────────────
    # Streaming
    # ─────────────────────────────────────────────────────────────────
    
    @property
    def supports_result_streaming(self) -> bool:
        return True
    
    async def stream_result(self, params: HttpParams) -> AsyncIterator[str]:
        """Stream HTTP response body in chunks.
        
        Useful for large responses or real-time data feeds.
        """
        import time
        
        # Validate first
        url_result = self._validate_url(params.url)
        if url_result.is_err():
            yield f"**Error:** {url_result.unwrap_err().message}"
            return
        
        method_result = self._validate_method(params.method)
        if method_result.is_err():
            yield f"**Error:** {method_result.unwrap_err().message}"
            return
        
        try:
            import httpx
        except ImportError:
            yield "**Error:** httpx not installed. Run: pip install httpx"
            return
        
        # Build request
        headers = {**self.config.default_headers, **params.headers}
        self.config.auth.apply(headers)
        
        content: str | bytes | None = None
        if params.json_body is not None:
            content = json.dumps(params.json_body)
            headers.setdefault("Content-Type", "application/json")
        elif params.body is not None:
            content = params.body
        
        start = time.perf_counter()
        total_bytes = 0
        
        try:
            client = await self._get_client()
            timeout = params.timeout or self.config.default_timeout
            
            async with client.stream(  # type: ignore[union-attr]
                method=params.method,
                url=params.url,
                headers=headers,
                params=params.query_params or None,
                content=content,
                timeout=timeout,
            ) as response:
                # Yield status line first
                yield f"**HTTP {response.status_code}** - streaming response...\n\n"
                
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    total_bytes += len(chunk)
                    if total_bytes > self.config.max_response_size:
                        yield f"\n\n**Error:** Response exceeded max size ({self.config.max_response_size} bytes)"
                        return
                    yield chunk.decode("utf-8", errors="replace")
            
            elapsed = (time.perf_counter() - start) * 1000
            yield f"\n\n---\n_Received {total_bytes} bytes in {elapsed:.0f}ms_"
            
        except Exception as e:
            yield f"\n\n**Error:** {type(e).__name__}: {e}"
