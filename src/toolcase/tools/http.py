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
from typing import TYPE_CHECKING, Annotated, AsyncIterator, ClassVar, Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ByteSize,
    ConfigDict,
    Discriminator,
    Field,
    HttpUrl,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    Tag,
    TypeAdapter,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

from toolcase.foundation.core import ToolMetadata
from toolcase.foundation.errors import Err, ErrorCode, ErrorTrace, Ok, ToolResult

from .base import ConfigurableTool, ToolConfig

if TYPE_CHECKING:
    import httpx

# Type alias for HTTP methods
HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
ALL_METHODS: frozenset[HttpMethod] = frozenset(["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])


# ─────────────────────────────────────────────────────────────────────────────
# Authentication Strategies (Protocol-based for Pydantic compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class NoAuth(BaseModel):
    """No authentication."""
    
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="never")
    auth_type: Literal["none"] = "none"
    
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        return headers
    
    def __hash__(self) -> int:
        return hash(self.auth_type)


# Singleton NoAuth instance for reuse (most common case)
_NO_AUTH: NoAuth | None = None


def get_no_auth() -> NoAuth:
    """Get singleton NoAuth instance."""
    global _NO_AUTH
    if _NO_AUTH is None:
        _NO_AUTH = NoAuth()
    return _NO_AUTH


class BearerAuth(BaseModel):
    """Bearer token authentication (OAuth2, JWT).
    
    Token is stored as SecretStr to prevent accidental logging/exposure.
    """
    
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="never")
    auth_type: Literal["bearer"] = "bearer"
    token: SecretStr = Field(..., description="Bearer token value (OAuth2/JWT)")
    
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        headers["Authorization"] = f"Bearer {self.token.get_secret_value()}"
        return headers
    
    @field_serializer("token", when_used="json")
    def _mask_token(self, v: SecretStr) -> str:
        """Mask token in JSON serialization for security."""
        secret = v.get_secret_value()
        return f"{secret[:4]}...{secret[-4:]}" if len(secret) > 8 else "***"
    
    def __hash__(self) -> int:
        return hash((self.auth_type, self.token.get_secret_value()))


class BasicAuth(BaseModel):
    """HTTP Basic authentication."""
    
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="never")
    auth_type: Literal["basic"] = "basic"
    username: Annotated[str, Field(min_length=1)]
    password: SecretStr
    
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        import base64
        credentials = base64.b64encode(
            f"{self.username}:{self.password.get_secret_value()}".encode()
        ).decode()
        headers["Authorization"] = f"Basic {credentials}"
        return headers
    
    @field_serializer("password", when_used="json")
    def _mask_password(self, v: SecretStr) -> str:
        """Mask password in JSON serialization."""
        return "***"
    
    def __hash__(self) -> int:
        return hash((self.auth_type, self.username))


class ApiKeyAuth(BaseModel):
    """API key authentication (header or query param)."""
    
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        revalidate_instances="never",
    )
    auth_type: Literal["api_key"] = "api_key"
    key: SecretStr = Field(..., description="API key value")
    header_name: Annotated[str, Field(
        default="X-API-Key",
        pattern=r"^[A-Za-z][A-Za-z0-9-]*$",
        description="HTTP header name for the key",
    )]
    
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        headers[self.header_name] = self.key.get_secret_value()
        return headers
    
    @field_serializer("key", when_used="json")
    def _mask_key(self, v: SecretStr) -> str:
        """Mask API key in JSON serialization."""
        secret = v.get_secret_value()
        return f"{secret[:4]}..." if len(secret) > 4 else "***"
    
    def __hash__(self) -> int:
        return hash((self.auth_type, self.header_name))


class CustomAuth(BaseModel):
    """Custom header-based authentication."""
    
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="never")
    auth_type: Literal["custom"] = "custom"
    headers: dict[str, SecretStr] = Field(default_factory=dict)
    
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        headers.update({k: v.get_secret_value() for k, v in self.headers.items()})
        return headers
    
    @field_serializer("headers", when_used="json")
    def _mask_headers(self, v: dict[str, SecretStr]) -> dict[str, str]:
        """Mask all custom header values in JSON serialization."""
        return {k: "***" for k in v}
    
    def __hash__(self) -> int:
        return hash((self.auth_type, tuple(sorted(self.headers.keys()))))


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

# Default blocked hosts for SSRF protection
_DEFAULT_BLOCKED_HOSTS: frozenset[str] = frozenset({
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
    "*.local", "169.254.*.*", "10.*.*.*",
    "172.16.*.*", "172.17.*.*", "172.18.*.*", "172.19.*.*",
    "172.20.*.*", "172.21.*.*", "172.22.*.*", "172.23.*.*",
    "172.24.*.*", "172.25.*.*", "172.26.*.*", "172.27.*.*",
    "172.28.*.*", "172.29.*.*", "172.30.*.*", "172.31.*.*",
    "192.168.*.*",
})


class HttpConfig(ToolConfig):
    """Configuration for HttpTool.
    
    Security-focused defaults with customization options.
    
    Attributes:
        allowed_hosts: Glob patterns for allowed hosts. Empty = all allowed.
        blocked_hosts: Glob patterns for blocked hosts (takes precedence).
        allowed_methods: HTTP methods to allow. Empty = all allowed.
        max_response_size: Maximum response body size (supports "10MB" format).
        default_timeout: Default request timeout in seconds.
        max_redirects: Maximum number of redirects to follow.
        follow_redirects: Whether to follow HTTP redirects.
        verify_ssl: Whether to verify SSL certificates.
        auth: Default authentication strategy.
        default_headers: Headers added to every request.
    """
    
    model_config = ConfigDict(
        validate_default=True,
        str_strip_whitespace=True,
        extra="forbid",  # Catch config typos
        revalidate_instances="never",
        frozen=True,  # Config should be immutable once created
        json_schema_extra={
            "title": "HTTP Tool Configuration",
            "examples": [{
                "allowed_hosts": ["api.example.com"],
                "allowed_methods": ["GET", "POST"],
                "default_timeout": 30.0,
            }],
        },
    )
    
    allowed_hosts: frozenset[str] = Field(
        default_factory=frozenset,
        description="Glob patterns for allowed hosts (empty = all)",
        repr=False,  # Can be verbose
    )
    blocked_hosts: frozenset[str] = Field(
        default=_DEFAULT_BLOCKED_HOSTS,
        description="Glob patterns for blocked hosts (SSRF protection)",
        repr=False,
    )
    allowed_methods: frozenset[HttpMethod] = Field(
        default_factory=lambda: frozenset(ALL_METHODS),
        description="Allowed HTTP methods",
    )
    max_response_size: ByteSize = Field(
        default=ByteSize(10 * 1024 * 1024),  # 10MB
        ge=1024,
        le=100 * 1024 * 1024,
        description="Max response size (e.g., '10MB', '1GB')",
    )
    default_timeout: Annotated[float, Field(
        default=30.0,
        ge=0.1,
        le=300.0,
        description="Default request timeout in seconds",
    )]
    max_redirects: PositiveInt = Field(
        default=10,
        le=30,
        description="Maximum redirects to follow",
    )
    follow_redirects: bool = True
    verify_ssl: bool = True
    auth: AuthStrategy = Field(default_factory=get_no_auth)
    default_headers: dict[str, str] = Field(
        default_factory=lambda: {"User-Agent": "toolcase-http/1.0"},
    )
    
    @field_validator("allowed_hosts", "blocked_hosts", mode="before")
    @classmethod
    def _normalize_host_sets(cls, v: frozenset[str] | set[str] | list[str] | tuple[str, ...]) -> frozenset[str]:
        """Accept various iterables, normalize to frozenset."""
        if isinstance(v, frozenset):
            return v
        return frozenset(v) if v else frozenset()
    
    @field_validator("allowed_methods", mode="before")
    @classmethod
    def _normalize_methods(cls, v: frozenset[str] | set[str] | list[str] | tuple[str, ...]) -> frozenset[str]:
        """Normalize methods to uppercase frozenset.
        
        Note: Returns frozenset[str] to satisfy Pydantic's validator return type.
        The field annotation (frozenset[HttpMethod]) constrains runtime values.
        """
        # Always normalize to uppercase, even if already frozenset (may be lowercase)
        return frozenset(m.upper() for m in v) if v else frozenset()
    
    @model_validator(mode="after")
    def _validate_host_config(self) -> "HttpConfig":
        """Validate that allowed and blocked hosts don't conflict."""
        overlap = self.allowed_hosts & self.blocked_hosts
        if overlap:
            raise ValueError(f"Hosts cannot be both allowed and blocked: {overlap}")
        return self
    
    @computed_field
    @property
    def max_response_size_bytes(self) -> int:
        """Get max response size as integer bytes."""
        return int(self.max_response_size)
    
    @field_serializer("allowed_hosts", "blocked_hosts", when_used="json")
    def _serialize_host_sets(self, v: frozenset[str]) -> list[str]:
        """Serialize frozensets as sorted lists for consistent JSON."""
        return sorted(v)
    
    @field_serializer("allowed_methods", when_used="json")
    def _serialize_methods(self, v: frozenset[HttpMethod]) -> list[str]:
        """Serialize methods as sorted list."""
        return sorted(v)
    
    def __hash__(self) -> int:
        return hash((self.default_timeout, self.verify_ssl, self.follow_redirects))


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

class HttpParams(BaseModel):
    """Parameters for HTTP requests.
    
    Attributes:
        url: The URL to request (validated as proper URL)
        method: HTTP method (GET, POST, etc.)
        headers: Additional request headers
        query_params: URL query parameters
        body: Request body (for POST/PUT/PATCH)
        json_body: JSON body (auto-serialized, sets Content-Type)
        timeout: Request timeout override
    """
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        populate_by_name=True,  # Accept aliases
        json_schema_extra={
            "title": "HTTP Request Parameters",
            "examples": [{
                "url": "https://api.example.com/data",
                "method": "GET",
            }, {
                "url": "https://api.example.com/users",
                "method": "POST",
                "json_body": {"name": "John", "email": "john@example.com"},
            }],
        },
    )
    
    url: Annotated[str, Field(
        description="URL to request",
        json_schema_extra={"format": "uri", "examples": ["https://api.example.com"]},
    )]
    method: HttpMethod = Field(default="GET", description="HTTP method")
    headers: dict[str, str] = Field(default_factory=dict, description="Additional headers", repr=False)
    query_params: dict[str, str] = Field(default_factory=dict, description="Query parameters", repr=False)
    body: str | None = Field(default=None, description="Request body (string)", repr=False)
    json_body: dict[str, object] | list[object] | None = Field(
        default=None, description="JSON body (auto-serialized)", repr=False
    )
    timeout: Annotated[float, Field(ge=0.1, le=300.0)] | None = Field(
        default=None,
        description="Timeout override in seconds",
    )
    
    @field_validator("url", mode="before")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        """Validate URL has proper scheme."""
        if not isinstance(v, str):
            return v
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v
    
    @field_validator("method", mode="before")
    @classmethod
    def _upper_method(cls, v: str) -> str:
        return v.upper() if isinstance(v, str) else v
    
    @model_validator(mode="after")
    def _validate_body_exclusivity(self) -> "HttpParams":
        """Ensure body and json_body are mutually exclusive."""
        if self.body is not None and self.json_body is not None:
            raise ValueError("Cannot specify both 'body' and 'json_body'")
        return self
    
    @computed_field
    @property
    def has_body(self) -> bool:
        """Whether request has a body."""
        return self.body is not None or self.json_body is not None


# TypeAdapter for fast dict->HttpParams validation
_HttpParamsAdapter: TypeAdapter[HttpParams] = TypeAdapter(HttpParams)


# ─────────────────────────────────────────────────────────────────────────────
# Response Model
# ─────────────────────────────────────────────────────────────────────────────

class HttpResponse(BaseModel):
    """Structured HTTP response for tool output."""
    
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="never",
    )
    
    status_code: Annotated[int, Field(ge=100, le=599)]
    headers: dict[str, str] = Field(repr=False)  # Can be verbose
    body: str = Field(repr=False)  # Often large
    url: str
    elapsed_ms: PositiveFloat
    
    @computed_field
    @property
    def is_success(self) -> bool:
        """Whether response indicates success (2xx)."""
        return 200 <= self.status_code < 300
    
    @computed_field
    @property
    def is_redirect(self) -> bool:
        """Whether response is a redirect (3xx)."""
        return 300 <= self.status_code < 400
    
    @computed_field
    @property
    def is_client_error(self) -> bool:
        """Whether response indicates client error (4xx)."""
        return 400 <= self.status_code < 500
    
    @computed_field
    @property
    def is_server_error(self) -> bool:
        """Whether response indicates server error (5xx)."""
        return 500 <= self.status_code < 600
    
    def _header_value(self, key: str) -> str | None:
        """Get header value case-insensitively (O(n) but headers are small)."""
        key_lower = key.lower()
        return next((v for k, v in self.headers.items() if k.lower() == key_lower), None)
    
    @computed_field
    @property
    def content_type(self) -> str | None:
        """Extract Content-Type header (case-insensitive)."""
        ct = self._header_value("content-type")
        return ct.split(";")[0].strip() if ct else None
    
    @computed_field
    @property
    def content_length(self) -> int | None:
        """Extract Content-Length header."""
        cl = self._header_value("content-length")
        return int(cl) if cl else None
    
    def __hash__(self) -> int:
        return hash((self.status_code, self.url, self.elapsed_ms))
    
    def to_output(self) -> str:
        """Format as tool output string."""
        status_emoji = "✓" if self.is_success else "✗" if self.status_code >= 400 else "→"
        lines = [
            f"**HTTP {self.status_code}** {status_emoji} ({self.elapsed_ms:.0f}ms)",
            f"URL: {self.url}",
            "",
        ]
        
        # Include relevant headers
        for key in ("Content-Type", "Content-Length", "Date", "Server"):
            if (val := self._header_value(key)):
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
        self._client: httpx.AsyncClient | None = None  # Lazy httpx client
    
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
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx async client."""
        if self._client is None:
            try:
                import httpx as httpx_mod
            except ImportError as e:
                raise ImportError(
                    "httpx is required for HttpTool. Install with: pip install httpx"
                ) from e
            
            self._client = httpx_mod.AsyncClient(
                follow_redirects=self.config.follow_redirects,
                verify=self.config.verify_ssl,
                timeout=self.config.default_timeout,
            )
        return self._client
    
    async def _close_client(self) -> None:
        """Close the httpx client."""
        if self._client is not None:
            await self._client.aclose()
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
            
            response = await client.request(
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
            max_size = self.config.max_response_size_bytes
            if content_length > max_size:
                return Err(ErrorTrace(
                    message=f"Response too large: {content_length} bytes (max: {max_size})",
                    error_code=ErrorCode.INVALID_PARAMS.value,
                    recoverable=False,
                ).with_operation("response_size_check"))
            
            # Read body with size limit
            body_bytes = await response.aread()
            if len(body_bytes) > max_size:
                return Err(ErrorTrace(
                    message=f"Response body exceeded max size: {len(body_bytes)} bytes (max: {max_size})",
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
        from toolcase.foundation.errors import result_to_string
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
            
            async with client.stream(
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
                    if total_bytes > self.config.max_response_size_bytes:
                        yield f"\n\n**Error:** Response exceeded max size ({self.config.max_response_size_bytes} bytes)"
                        return
                    yield chunk.decode("utf-8", errors="replace")
            
            elapsed = (time.perf_counter() - start) * 1000
            yield f"\n\n---\n_Received {total_bytes} bytes in {elapsed:.0f}ms_"
            
        except Exception as e:
            yield f"\n\n**Error:** {type(e).__name__}: {e}"
