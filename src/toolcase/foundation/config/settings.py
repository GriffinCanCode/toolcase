"""Environment-based configuration using pydantic-settings.

Provides type-safe, validated configuration from environment variables
with sensible defaults. Supports .env files and nested configuration.

Example:
    >>> from toolcase.foundation.settings import get_settings
    >>> settings = get_settings()
    >>> print(settings.cache.ttl)
    3600.0
    >>> print(settings.logging.level)
    'INFO'
    
    # Or with environment variables:
    # TOOLCASE_CACHE_TTL=7200
    # TOOLCASE_LOG_LEVEL=DEBUG
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import (
    ByteSize,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    computed_field,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class CacheSettings(BaseSettings):
    """Cache-related configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="TOOLCASE_CACHE_",
        extra="ignore",
    )
    
    enabled: bool = True
    ttl: PositiveFloat = Field(default=3600.0, description="Default cache TTL in seconds")
    max_size: PositiveInt = Field(default=1000, description="Max cache entries")
    redis_url: SecretStr | None = Field(default=None, description="Redis URL for distributed cache")
    
    @computed_field
    @property
    def backend(self) -> Literal["memory", "redis"]:
        """Determine cache backend from configuration."""
        return "redis" if self.redis_url else "memory"


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="TOOLCASE_LOG_",
        extra="ignore",
    )
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "text"] = "text"
    include_timestamps: bool = True
    include_correlation_id: bool = True


class RetrySettings(BaseSettings):
    """Default retry configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="TOOLCASE_RETRY_",
        extra="ignore",
    )
    
    max_retries: Annotated[int, Field(ge=0, le=10)] = 3
    base_delay: PositiveFloat = Field(default=1.0, description="Base delay in seconds")
    max_delay: PositiveFloat = Field(default=30.0, description="Maximum delay in seconds")
    exponential_base: PositiveFloat = Field(default=2.0, description="Exponential backoff base")
    jitter: bool = True


class HttpSettings(BaseSettings):
    """HTTP client default configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="TOOLCASE_HTTP_",
        extra="ignore",
    )
    
    timeout: PositiveFloat = Field(default=30.0, description="Default request timeout")
    max_response_size: ByteSize = Field(
        default=ByteSize(10 * 1024 * 1024),
        description="Max response size",
    )
    verify_ssl: bool = True
    follow_redirects: bool = True
    max_redirects: PositiveInt = Field(default=10)
    user_agent: str = "toolcase-http/1.0"


class TracingSettings(BaseSettings):
    """Observability/tracing configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="TOOLCASE_TRACING_",
        extra="ignore",
    )
    
    enabled: bool = False
    service_name: str = "toolcase"
    otlp_endpoint: str | None = Field(
        default=None,
        description="OpenTelemetry collector endpoint",
    )
    sample_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    export_batch_size: PositiveInt = 100
    export_timeout: PositiveFloat = 30.0


class RateLimitSettings(BaseSettings):
    """Rate limiting defaults."""
    
    model_config = SettingsConfigDict(
        env_prefix="TOOLCASE_RATELIMIT_",
        extra="ignore",
    )
    
    enabled: bool = False
    max_calls: PositiveInt = Field(default=100, description="Max calls per window")
    window_seconds: PositiveFloat = Field(default=60.0, description="Time window in seconds")
    strategy: Literal["sliding", "fixed"] = "sliding"


class ToolcaseSettings(BaseSettings):
    """Root settings for Toolcase framework.
    
    Loads configuration from environment variables with TOOLCASE_ prefix.
    Supports nested configuration and .env files.
    
    Example environment variables:
        TOOLCASE_DEBUG=true
        TOOLCASE_CACHE_TTL=7200
        TOOLCASE_LOG_LEVEL=DEBUG
        TOOLCASE_HTTP_TIMEOUT=60
        TOOLCASE_TRACING_ENABLED=true
    """
    
    model_config = SettingsConfigDict(
        env_prefix="TOOLCASE_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        validate_default=True,
    )
    
    # Global settings
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: Literal["development", "staging", "production"] = "development"
    
    # Nested settings (loaded with TOOLCASE_CACHE_, TOOLCASE_LOG_, etc.)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    http: HttpSettings = Field(default_factory=HttpSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    
    @field_validator("environment", mode="before")
    @classmethod
    def _normalize_env(cls, v: str) -> str:
        """Normalize environment name to lowercase."""
        return v.lower() if isinstance(v, str) else v
    
    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"
    
    @computed_field
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"


# Singleton pattern for settings
@lru_cache(maxsize=1)
def get_settings() -> ToolcaseSettings:
    """Get the global settings instance (cached).
    
    Returns:
        Cached ToolcaseSettings instance
    
    Example:
        >>> settings = get_settings()
        >>> settings.debug
        False
    """
    return ToolcaseSettings()


def clear_settings_cache() -> None:
    """Clear the settings cache (useful for testing).
    
    After calling this, the next get_settings() call will
    reload configuration from environment.
    """
    get_settings.cache_clear()
