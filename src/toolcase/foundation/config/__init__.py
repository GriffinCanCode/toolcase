"""Configuration management using pydantic-settings.

Provides environment-based configuration with type safety and validation.
Supports multiple .env file variants, environment-specific overrides, and
compatibility with python-dotenv, django-environ, and python-decouple.
"""

from .env import (
    dotenv_values,
    env,
    get_env,
    get_env_files_loaded,
    get_env_prefix,
    load_env,
    require_env,
)
from .settings import (
    CacheSettings,
    HttpSettings,
    LoggingSettings,
    RateLimitSettings,
    RetrySettings,
    ToolcaseSettings,
    TracingSettings,
    clear_settings_cache,
    get_settings,
)

__all__ = [
    # Settings classes
    "CacheSettings",
    "HttpSettings",
    "LoggingSettings",
    "RateLimitSettings",
    "RetrySettings",
    "ToolcaseSettings",
    "TracingSettings",
    "clear_settings_cache",
    "get_settings",
    # Env utilities
    "load_env",
    "get_env",
    "require_env",
    "get_env_prefix",
    "get_env_files_loaded",
    "dotenv_values",
    "env",
]
