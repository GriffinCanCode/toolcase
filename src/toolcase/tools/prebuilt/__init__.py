"""Prebuilt tools ready for use.

Includes:
- HttpTool: HTTP request tool with auth strategies (explicit and env-based)
- DiscoveryTool: Meta-tool for tool discovery and search
"""

from .discovery import DiscoveryParams, DiscoveryTool
from .http import (
    ApiKeyAuth,
    BasicAuth,
    BearerAuth,
    CustomAuth,
    EnvApiKeyAuth,
    EnvBasicAuth,
    EnvBearerAuth,
    HttpConfig,
    HttpParams,
    HttpResponse,
    HttpTool,
    NoAuth,
    api_key_from_env,
    basic_from_env,
    bearer_from_env,
    get_no_auth,
)

__all__ = [
    # HTTP Tool
    "HttpTool",
    "HttpConfig",
    "HttpParams",
    "HttpResponse",
    # Auth (explicit secrets)
    "NoAuth",
    "BearerAuth",
    "BasicAuth",
    "ApiKeyAuth",
    "CustomAuth",
    "get_no_auth",
    # Auth (environment-based)
    "EnvBearerAuth",
    "EnvApiKeyAuth",
    "EnvBasicAuth",
    "bearer_from_env",
    "api_key_from_env",
    "basic_from_env",
    # Discovery
    "DiscoveryTool",
    "DiscoveryParams",
]
