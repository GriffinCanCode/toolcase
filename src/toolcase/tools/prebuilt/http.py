"""HTTP Tool - Make HTTP requests to external APIs.

A production-ready HTTP tool with:
- Configurable allowed hosts/methods for security
- Multiple auth strategies (Bearer, Basic, API Key, Custom)
- Environment-aware auth (auto-load secrets from env vars)
- Streaming support for large responses
- Proper timeout and error handling
- Response size limits

Example:
    >>> from toolcase.tools import HttpTool, BearerAuth, EnvBearerAuth
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
    >>> # With explicit authentication
    >>> http = HttpTool(HttpConfig(auth=BearerAuth(token="sk-xxx")))
    >>> 
    >>> # With environment-based authentication (recommended)
    >>> http = HttpTool(HttpConfig(auth=EnvBearerAuth(env_var="OPENAI_API_KEY")))
    >>> # Or shorthand:
    >>> http = HttpTool(HttpConfig(auth=bearer_from_env("OPENAI_API_KEY")))
"""

from __future__ import annotations

import base64
import fnmatch
import os
import time
from typing import TYPE_CHECKING, Annotated, AsyncIterator, ClassVar, Literal
from urllib.parse import urlparse

import orjson

from pydantic import (
    BaseModel, ByteSize, ConfigDict, Discriminator, Field, PositiveFloat, PositiveInt,
    SecretStr, Tag, TypeAdapter, computed_field, field_serializer, field_validator, model_validator,
)

from toolcase.foundation.core import ToolMetadata
from toolcase.foundation.errors import Err, ErrorCode, ErrorTrace, JsonDict, Ok, ToolResult

from ..core.base import ConfigurableTool, ToolConfig

if TYPE_CHECKING:
    import httpx

# Type alias for HTTP methods
HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
ALL_METHODS: frozenset[HttpMethod] = frozenset({"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"})

# Shared frozen model config for auth classes
_FROZEN_CONFIG = ConfigDict(frozen=True, extra="forbid", revalidate_instances="never")


# ─────────────────────────────────────────────────────────────────────────────
# Authentication Strategies (Protocol-based for Pydantic compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class NoAuth(BaseModel):
    """No authentication."""
    model_config = _FROZEN_CONFIG
    auth_type: Literal["none"] = "none"
    
    def apply(self, headers: dict[str, str]) -> dict[str, str]: return headers
    def __hash__(self) -> int: return hash(self.auth_type)


# Singleton NoAuth instance for reuse (most common case)
_NO_AUTH: NoAuth | None = None

def get_no_auth() -> NoAuth:
    """Get singleton NoAuth instance."""
    global _NO_AUTH
    return _NO_AUTH or (_NO_AUTH := NoAuth())


class BearerAuth(BaseModel):
    """Bearer token authentication (OAuth2, JWT). Token stored as SecretStr to prevent accidental exposure."""
    model_config = _FROZEN_CONFIG
    auth_type: Literal["bearer"] = "bearer"
    token: SecretStr = Field(..., description="Bearer token value (OAuth2/JWT)")
    
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        return headers | {"Authorization": f"Bearer {self.token.get_secret_value()}"}
    
    @field_serializer("token", when_used="json")
    def _mask_token(self, v: SecretStr) -> str:
        """Mask token in JSON serialization for security."""
        return f"{s[:4]}...{s[-4:]}" if len(s := v.get_secret_value()) > 8 else "***"
    
    def __hash__(self) -> int: return hash((self.auth_type, self.token.get_secret_value()))


class BasicAuth(BaseModel):
    """HTTP Basic authentication."""
    model_config = _FROZEN_CONFIG
    auth_type: Literal["basic"] = "basic"
    username: Annotated[str, Field(min_length=1)]
    password: SecretStr
    
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        return headers | {"Authorization": f"Basic {base64.b64encode(f'{self.username}:{self.password.get_secret_value()}'.encode()).decode()}"}
    
    @field_serializer("password", when_used="json")
    def _mask_password(self, v: SecretStr) -> str: return "***"
    
    def __hash__(self) -> int: return hash((self.auth_type, self.username))


class ApiKeyAuth(BaseModel):
    """API key authentication (header or query param)."""
    model_config = ConfigDict(**_FROZEN_CONFIG, str_strip_whitespace=True)
    auth_type: Literal["api_key"] = "api_key"
    key: SecretStr = Field(..., description="API key value")
    header_name: Annotated[str, Field(default="X-API-Key", pattern=r"^[A-Za-z][A-Za-z0-9-]*$", description="HTTP header name for the key")]
    
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        return headers | {self.header_name: self.key.get_secret_value()}
    
    @field_serializer("key", when_used="json")
    def _mask_key(self, v: SecretStr) -> str:
        """Mask API key in JSON serialization."""
        return f"{(s := v.get_secret_value())[:4]}..." if len(s) > 4 else "***"
    
    def __hash__(self) -> int: return hash((self.auth_type, self.header_name))


class CustomAuth(BaseModel):
    """Custom header-based authentication."""
    model_config = _FROZEN_CONFIG
    auth_type: Literal["custom"] = "custom"
    headers: dict[str, SecretStr] = Field(default_factory=dict)
    
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        return headers | {k: v.get_secret_value() for k, v in self.headers.items()}
    
    @field_serializer("headers", when_used="json")
    def _mask_headers(self, v: dict[str, SecretStr]) -> dict[str, str]: return dict.fromkeys(v, "***")
    
    def __hash__(self) -> int: return hash((self.auth_type, tuple(sorted(self.headers.keys()))))


# ─────────────────────────────────────────────────────────────────────────────
# Environment-Aware Authentication (auto-load from env vars)
# ─────────────────────────────────────────────────────────────────────────────

def _get_env_secret(env_var: str, required: bool = True) -> SecretStr:
    """Load a secret from environment variable."""
    value = os.environ.get(env_var)
    if value is None:
        if required:
            raise EnvironmentError(f"Required environment variable '{env_var}' is not set")
        value = ""
    return SecretStr(value)


class EnvBearerAuth(BaseModel):
    """Bearer token from environment variable. Token loaded lazily from env var on first use."""
    model_config = _FROZEN_CONFIG
    auth_type: Literal["env_bearer"] = "env_bearer"
    env_var: str = Field(..., description="Environment variable name containing the token")
    
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        return headers | {"Authorization": f"Bearer {_get_env_secret(self.env_var).get_secret_value()}"}
    
    def __hash__(self) -> int: return hash((self.auth_type, self.env_var))


class EnvApiKeyAuth(BaseModel):
    """API key from environment variable."""
    model_config = ConfigDict(**_FROZEN_CONFIG, str_strip_whitespace=True)
    auth_type: Literal["env_api_key"] = "env_api_key"
    env_var: str = Field(..., description="Environment variable name containing the API key")
    header_name: str = Field(default="X-API-Key", pattern=r"^[A-Za-z][A-Za-z0-9-]*$")
    
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        return headers | {self.header_name: _get_env_secret(self.env_var).get_secret_value()}
    
    def __hash__(self) -> int: return hash((self.auth_type, self.env_var, self.header_name))


class EnvBasicAuth(BaseModel):
    """Basic auth with credentials from environment variables."""
    model_config = _FROZEN_CONFIG
    auth_type: Literal["env_basic"] = "env_basic"
    username_env: str = Field(..., description="Env var for username")
    password_env: str = Field(..., description="Env var for password")
    
    def apply(self, headers: dict[str, str]) -> dict[str, str]:
        username = os.environ.get(self.username_env, "")
        password = _get_env_secret(self.password_env).get_secret_value()
        creds = base64.b64encode(f"{username}:{password}".encode()).decode()
        return headers | {"Authorization": f"Basic {creds}"}
    
    def __hash__(self) -> int: return hash((self.auth_type, self.username_env, self.password_env))


# Convenience factory functions for env-based auth
def bearer_from_env(env_var: str = "API_TOKEN") -> EnvBearerAuth:
    """Create Bearer auth that loads token from environment variable.
    
    Example:
        >>> auth = bearer_from_env("OPENAI_API_KEY")
        >>> http = HttpTool(HttpConfig(auth=auth))
    """
    return EnvBearerAuth(env_var=env_var)


def api_key_from_env(env_var: str = "API_KEY", header: str = "X-API-Key") -> EnvApiKeyAuth:
    """Create API key auth that loads from environment variable.
    
    Example:
        >>> auth = api_key_from_env("ANTHROPIC_API_KEY", header="x-api-key")
    """
    return EnvApiKeyAuth(env_var=env_var, header_name=header)


def basic_from_env(username_env: str = "API_USERNAME", password_env: str = "API_PASSWORD") -> EnvBasicAuth:
    """Create Basic auth that loads credentials from environment variables."""
    return EnvBasicAuth(username_env=username_env, password_env=password_env)


def _auth_discriminator(v: JsonDict | BaseModel) -> str:
    """Discriminator function for auth strategy union."""
    return str(v.get("auth_type", "none")) if isinstance(v, dict) else getattr(v, "auth_type", "none")


# Discriminated union for proper serialization/deserialization
AuthStrategy = Annotated[
    Annotated[NoAuth, Tag("none")]
    | Annotated[BearerAuth, Tag("bearer")]
    | Annotated[BasicAuth, Tag("basic")]
    | Annotated[ApiKeyAuth, Tag("api_key")]
    | Annotated[CustomAuth, Tag("custom")]
    | Annotated[EnvBearerAuth, Tag("env_bearer")]
    | Annotated[EnvApiKeyAuth, Tag("env_api_key")]
    | Annotated[EnvBasicAuth, Tag("env_basic")],
    Discriminator(_auth_discriminator),
]


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Default blocked hosts for SSRF protection
_DEFAULT_BLOCKED_HOSTS: frozenset[str] = frozenset(
    {"localhost", "127.0.0.1", "0.0.0.0", "::1", "*.local", "169.254.*.*", "10.*.*.*", "192.168.*.*"}
    | {f"172.{i}.*.*" for i in range(16, 32)}
)


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
        validate_default=True, str_strip_whitespace=True, extra="forbid",  # Catch config typos
        revalidate_instances="never", frozen=True,  # Config should be immutable once created
        json_schema_extra={
            "title": "HTTP Tool Configuration",
            "examples": [{"allowed_hosts": ["api.example.com"], "allowed_methods": ["GET", "POST"], "default_timeout": 30.0}],
        },
    )
    
    allowed_hosts: frozenset[str] = Field(default_factory=frozenset, description="Glob patterns for allowed hosts (empty = all)", repr=False)
    blocked_hosts: frozenset[str] = Field(default=_DEFAULT_BLOCKED_HOSTS, description="Glob patterns for blocked hosts (SSRF protection)", repr=False)
    allowed_methods: frozenset[HttpMethod] = Field(default_factory=lambda: frozenset(ALL_METHODS), description="Allowed HTTP methods")
    max_response_size: ByteSize = Field(default=ByteSize(10 * 1024 * 1024), ge=1024, le=100 * 1024 * 1024, description="Max response size (e.g., '10MB', '1GB')")
    default_timeout: Annotated[float, Field(default=30.0, ge=0.1, le=300.0, description="Default request timeout in seconds")]
    max_redirects: PositiveInt = Field(default=10, le=30, description="Maximum redirects to follow")
    follow_redirects: bool = True
    verify_ssl: bool = True
    auth: AuthStrategy = Field(default_factory=get_no_auth)
    default_headers: dict[str, str] = Field(default_factory=lambda: {"User-Agent": "toolcase-http/1.0"})
    
    @field_validator("allowed_hosts", "blocked_hosts", mode="before")
    @classmethod
    def _normalize_host_sets(cls, v: frozenset[str] | set[str] | list[str] | tuple[str, ...]) -> frozenset[str]:
        """Accept various iterables, normalize to frozenset."""
        return v if isinstance(v, frozenset) else frozenset(v or ())
    
    @field_validator("allowed_methods", mode="before")
    @classmethod
    def _normalize_methods(cls, v: frozenset[str] | set[str] | list[str] | tuple[str, ...]) -> frozenset[str]:
        """Normalize methods to uppercase frozenset. Returns frozenset[str] for Pydantic; field annotation constrains runtime values."""
        return frozenset(m.upper() for m in v) if v else frozenset()
    
    @model_validator(mode="after")
    def _validate_host_config(self) -> "HttpConfig":
        """Validate that allowed and blocked hosts don't conflict."""
        if overlap := self.allowed_hosts & self.blocked_hosts:
            raise ValueError(f"Hosts cannot be both allowed and blocked: {overlap}")
        return self
    
    @computed_field
    @property
    def max_response_size_bytes(self) -> int:
        """Get max response size as integer bytes."""
        return int(self.max_response_size)
    
    @field_serializer("allowed_hosts", "blocked_hosts", when_used="json")
    def _serialize_host_sets(self, v: frozenset[str]) -> list[str]: return sorted(v)
    
    @field_serializer("allowed_methods", when_used="json")
    def _serialize_methods(self, v: frozenset[HttpMethod]) -> list[str]: return sorted(v)
    
    def __hash__(self) -> int: return hash((self.default_timeout, self.verify_ssl, self.follow_redirects))


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
        str_strip_whitespace=True, validate_default=True, extra="forbid", populate_by_name=True,  # Accept aliases
        json_schema_extra={
            "title": "HTTP Request Parameters",
            "examples": [
                {"url": "https://api.example.com/data", "method": "GET"},
                {"url": "https://api.example.com/users", "method": "POST", "json_body": {"name": "John", "email": "john@example.com"}},
            ],
        },
    )
    
    url: Annotated[str, Field(description="URL to request", json_schema_extra={"format": "uri", "examples": ["https://api.example.com"]})]
    method: HttpMethod = Field(default="GET", description="HTTP method")
    headers: dict[str, str] = Field(default_factory=dict, description="Additional headers", repr=False)
    query_params: dict[str, str] = Field(default_factory=dict, description="Query parameters", repr=False)
    body: str | None = Field(default=None, description="Request body (string)", repr=False)
    json_body: JsonDict | list[object] | None = Field(default=None, description="JSON body (auto-serialized)", repr=False)
    timeout: Annotated[float, Field(ge=0.1, le=300.0)] | None = Field(default=None, description="Timeout override in seconds")
    
    @field_validator("url", mode="before")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        """Validate URL has proper scheme."""
        if not isinstance(v, str): return v
        if not (v := v.strip()).startswith(("http://", "https://")): raise ValueError("URL must start with http:// or https://")
        return v
    
    @field_validator("method", mode="before")
    @classmethod
    def _upper_method(cls, v: str) -> str: return v.upper() if isinstance(v, str) else v
    
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
    model_config = _FROZEN_CONFIG
    
    status_code: Annotated[int, Field(ge=100, le=599)]
    headers: dict[str, str] = Field(repr=False)  # Can be verbose
    body: str = Field(repr=False)  # Often large
    url: str
    elapsed_ms: PositiveFloat
    
    def _status_in(self, lo: int, hi: int) -> bool: return lo <= self.status_code < hi
    
    @computed_field
    @property
    def is_success(self) -> bool: return self._status_in(200, 300)
    
    @computed_field
    @property
    def is_redirect(self) -> bool: return self._status_in(300, 400)
    
    @computed_field
    @property
    def is_client_error(self) -> bool: return self._status_in(400, 500)
    
    @computed_field
    @property
    def is_server_error(self) -> bool: return self._status_in(500, 600)
    
    def _header_value(self, key: str) -> str | None:
        """Get header value case-insensitively (O(n) but headers are small)."""
        return next((v for k, v in self.headers.items() if k.lower() == key.lower()), None)
    
    @computed_field
    @property
    def content_type(self) -> str | None: return ct.split(";")[0].strip() if (ct := self._header_value("content-type")) else None
    
    @computed_field
    @property
    def content_length(self) -> int | None: return int(cl) if (cl := self._header_value("content-length")) else None
    
    def __hash__(self) -> int: return hash((self.status_code, self.url, self.elapsed_ms))
    
    def to_output(self) -> str:
        """Format as tool output string."""
        emoji = "✓" if self.is_success else ("✗" if self.status_code >= 400 else "→")
        header_lines = [f"{k}: {v}" for k in ("Content-Type", "Content-Length", "Date", "Server") if (v := self._header_value(k))]
        return "\n".join([f"**HTTP {self.status_code}** {emoji} ({self.elapsed_ms:.0f}ms)", f"URL: {self.url}", "", *header_lines, "", "**Response:**", self.body])


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Tool
# ─────────────────────────────────────────────────────────────────────────────

def _err(msg: str, code: ErrorCode, op: str, recoverable: bool = False, **kw: str) -> ToolResult:
    """Helper to create error results."""
    return Err(ErrorTrace(message=msg, error_code=code.value, recoverable=recoverable, **kw).with_operation(op))


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
            return _err(f"Invalid URL: {e}", ErrorCode.INVALID_PARAMS, "url_validation")
        
        if parsed.scheme not in ("http", "https"):
            return _err(f"Invalid scheme '{parsed.scheme}'. Use http or https.", ErrorCode.INVALID_PARAMS, "url_validation")
        
        host, host_lower = parsed.hostname or "", (parsed.hostname or "").lower()
        matches = lambda h, patterns: any(fnmatch.fnmatch(h, p) or fnmatch.fnmatch(host_lower, p.lower()) for p in patterns)
        
        # Check blocked hosts first (SSRF protection)
        if matches(host, self.config.blocked_hosts):
            return _err(f"Host '{host}' is blocked for security reasons.", ErrorCode.PERMISSION_DENIED, "host_validation")
        # Check allowed hosts (if configured)
        if self.config.allowed_hosts and not matches(host, self.config.allowed_hosts):
            return _err(f"Host '{host}' not in allowed list.", ErrorCode.PERMISSION_DENIED, "host_validation")
        return Ok(url)
    
    def _validate_method(self, method: HttpMethod) -> ToolResult:
        """Validate HTTP method against allowed list."""
        if method not in self.config.allowed_methods:
            return _err(f"Method '{method}' not allowed. Allowed: {', '.join(sorted(self.config.allowed_methods))}", ErrorCode.PERMISSION_DENIED, "method_validation")
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
                raise ImportError("httpx is required for HttpTool. Install with: pip install httpx") from e
            self._client = httpx_mod.AsyncClient(
                follow_redirects=self.config.follow_redirects,
                verify=self.config.verify_ssl,
                timeout=self.config.default_timeout,
            )
        return self._client
    
    async def _close_client(self) -> None:
        """Close the httpx client."""
        if self._client: await self._client.aclose(); self._client = None
    
    def _build_request(self, params: HttpParams) -> tuple[dict[str, str], str | bytes | None]:
        """Build headers and content for a request."""
        headers = self.config.auth.apply(self.config.default_headers | params.headers)
        if params.json_body is not None:
            headers.setdefault("Content-Type", "application/json")
            return headers, orjson.dumps(params.json_body).decode()
        return headers, params.body
    
    # ─────────────────────────────────────────────────────────────────
    # Execution
    # ─────────────────────────────────────────────────────────────────
    
    async def _async_run_result(self, params: HttpParams) -> ToolResult:
        """Execute HTTP request with Result-based error handling."""
        if (r := self._validate_url(params.url)).is_err():
            return r
        if (r := self._validate_method(params.method)).is_err():
            return r
        
        try:
            import httpx
        except ImportError:
            return _err("httpx not installed. Run: pip install httpx", ErrorCode.EXTERNAL_SERVICE_ERROR, "import")
        
        headers, content = self._build_request(params)
        max_size, start = self.config.max_response_size_bytes, time.perf_counter()
        
        try:
            response = await (await self._get_client()).request(
                method=params.method, url=params.url, headers=headers,
                params=params.query_params or None, content=content,
                timeout=params.timeout or self.config.default_timeout,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            # Check response size
            if (cl := int(response.headers.get("content-length", 0))) > max_size:
                return _err(f"Response too large: {cl} bytes (max: {max_size})", ErrorCode.INVALID_PARAMS, "response_size_check")
            
            # Read body with size limit
            if len(body_bytes := await response.aread()) > max_size:
                return _err(f"Response body exceeded max size: {len(body_bytes)} bytes (max: {max_size})", ErrorCode.INVALID_PARAMS, "body_read")
            
            return Ok(HttpResponse(
                status_code=response.status_code, headers=dict(response.headers),
                body=body_bytes.decode("utf-8", errors="replace"), url=str(response.url), elapsed_ms=elapsed_ms,
            ).to_output())
        except httpx.TimeoutException:
            return _err(f"Request timed out after {params.timeout or self.config.default_timeout}s", ErrorCode.TIMEOUT, "request", recoverable=True)
        except httpx.NetworkError as e:
            return _err(f"Network error: {e}", ErrorCode.NETWORK_ERROR, "request", recoverable=True)
        except Exception as e:
            return _err(f"Request failed: {e}", ErrorCode.EXTERNAL_SERVICE_ERROR, "request", recoverable=True, details=type(e).__name__)
    
    async def _async_run(self, params: HttpParams) -> str:
        """Execute HTTP request."""
        from toolcase.foundation.errors import result_to_string
        return result_to_string(await self._async_run_result(params), self.metadata.name)
    
    # ─────────────────────────────────────────────────────────────────
    # Streaming
    # ─────────────────────────────────────────────────────────────────
    
    @property
    def supports_result_streaming(self) -> bool: return True
    
    async def stream_result(self, params: HttpParams) -> AsyncIterator[str]:
        """Stream HTTP response body in chunks. Useful for large responses or real-time data feeds."""
        import time
        for r in (self._validate_url(params.url), self._validate_method(params.method)):
            if r.is_err(): yield f"**Error:** {r.unwrap_err().message}"; return
        try:
            import httpx
        except ImportError: yield "**Error:** httpx not installed. Run: pip install httpx"; return
        
        headers, content = self._build_request(params)
        start, total_bytes, max_size = time.perf_counter(), 0, self.config.max_response_size_bytes
        try:
            async with (await self._get_client()).stream(
                method=params.method, url=params.url, headers=headers,
                params=params.query_params or None, content=content,
                timeout=params.timeout or self.config.default_timeout,
            ) as response:
                yield f"**HTTP {response.status_code}** - streaming response...\n\n"
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    if (total_bytes := total_bytes + len(chunk)) > max_size:
                        yield f"\n\n**Error:** Response exceeded max size ({max_size} bytes)"; return
                    yield chunk.decode("utf-8", errors="replace")
            yield f"\n\n---\n_Received {total_bytes} bytes in {(time.perf_counter() - start) * 1000:.0f}ms_"
        except Exception as e:
            yield f"\n\n**Error:** {type(e).__name__}: {e}"
