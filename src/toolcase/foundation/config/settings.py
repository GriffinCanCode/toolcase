"""Environment-based configuration using pydantic-settings.

Provides type-safe, validated configuration from environment variables
with sensible defaults. Supports .env files and nested configuration.

Optimizations:
- Uses frozen models where appropriate for immutability
- AliasChoices for flexible env var naming
- Coercion flags for strict/lenient parsing
- Computed fields for derived values

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
    AliasChoices,
    ByteSize,
    Field,
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
        frozen=True,  # Settings should be immutable
        revalidate_instances="never",
    )
    
    enabled: bool = True
    ttl: PositiveFloat = Field(
        default=3600.0,
        description="Default cache TTL in seconds",
        validation_alias=AliasChoices("ttl", "TTL", "cache_ttl"),  # Accept multiple formats
    )
    max_size: PositiveInt = Field(default=1000, description="Max cache entries")
    redis_url: SecretStr | None = Field(default=None, description="Redis URL for distributed cache")
    
    @computed_field
    @property
    def backend(self) -> Literal["memory", "redis"]:
        """Determine cache backend from configuration."""
        return "redis" if self.redis_url else "memory"
    
    def __hash__(self) -> int:
        return hash((self.enabled, self.ttl, self.max_size))


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="TOOLCASE_LOG_",
        extra="ignore",
        frozen=True,
        revalidate_instances="never",
        use_enum_values=True,  # Store enum values directly
    )
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        validation_alias=AliasChoices("level", "LEVEL", "log_level"),
    )
    format: Literal["json", "text"] = "text"
    include_timestamps: bool = True
    include_correlation_id: bool = True
    
    def __hash__(self) -> int:
        return hash((self.level, self.format))


class RetrySettings(BaseSettings):
    """Default retry configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="TOOLCASE_RETRY_",
        extra="ignore",
        frozen=True,
        revalidate_instances="never",
    )
    
    max_retries: Annotated[int, Field(ge=0, le=10)] = 3
    base_delay: PositiveFloat = Field(default=1.0, description="Base delay in seconds")
    max_delay: PositiveFloat = Field(default=30.0, description="Maximum delay in seconds")
    exponential_base: PositiveFloat = Field(default=2.0, description="Exponential backoff base")
    jitter: bool = True
    
    @computed_field
    @property
    def is_enabled(self) -> bool:
        """Whether retries are enabled."""
        return self.max_retries > 0
    
    def __hash__(self) -> int:
        return hash((self.max_retries, self.base_delay, self.max_delay))


class HttpSettings(BaseSettings):
    """HTTP client default configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="TOOLCASE_HTTP_",
        extra="ignore",
        frozen=True,
        revalidate_instances="never",
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
    
    @computed_field
    @property
    def max_response_bytes(self) -> int:
        """Max response size as int."""
        return int(self.max_response_size)
    
    def __hash__(self) -> int:
        return hash((self.timeout, self.verify_ssl))


class TracingSettings(BaseSettings):
    """Observability/tracing configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="TOOLCASE_TRACING_",
        extra="ignore",
        frozen=True,
        revalidate_instances="never",
    )
    
    enabled: bool = False
    service_name: str = "toolcase"
    otlp_endpoint: str | None = Field(
        default=None,
        description="OpenTelemetry collector endpoint",
        validation_alias=AliasChoices("otlp_endpoint", "OTLP_ENDPOINT", "endpoint"),
    )
    sample_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    export_batch_size: PositiveInt = 100
    export_timeout: PositiveFloat = 30.0
    
    @computed_field
    @property
    def is_configured(self) -> bool:
        """Whether tracing is fully configured."""
        return self.enabled and self.otlp_endpoint is not None
    
    def __hash__(self) -> int:
        return hash((self.enabled, self.service_name))


class RateLimitSettings(BaseSettings):
    """Rate limiting defaults."""
    
    model_config = SettingsConfigDict(
        env_prefix="TOOLCASE_RATELIMIT_",
        extra="ignore",
        frozen=True,
        revalidate_instances="never",
        use_enum_values=True,
    )
    
    enabled: bool = False
    max_calls: PositiveInt = Field(default=100, description="Max calls per window")
    window_seconds: PositiveFloat = Field(default=60.0, description="Time window in seconds")
    strategy: Literal["sliding", "fixed"] = "sliding"
    
    @computed_field
    @property
    def calls_per_second(self) -> float:
        """Compute calls per second rate."""
        return self.max_calls / self.window_seconds  # PositiveFloat guarantees > 0
    
    def __hash__(self) -> int:
        return hash((self.enabled, self.max_calls, self.window_seconds))


class ToolcaseSettings(BaseSettings):
    """Root settings for Toolcase framework. Loads from env vars (TOOLCASE_ prefix), .env files, nested config. Frozen/immutable with cached singleton via get_settings()."""
    
    model_config = SettingsConfigDict(
        env_prefix="TOOLCASE_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        validate_default=True,
        frozen=True,  # Settings are immutable once loaded
        revalidate_instances="never",
        use_enum_values=True,
    )
    
    # Global settings
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
        validation_alias=AliasChoices("debug", "DEBUG"),
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        validation_alias=AliasChoices("environment", "ENV", "ENVIRONMENT"),
    )
    
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
    
    @computed_field
    @property
    def is_staging(self) -> bool:
        """Check if running in staging."""
        return self.environment == "staging"
    
    def __hash__(self) -> int:
        return hash((self.debug, self.environment))


# Singleton pattern for settings
@lru_cache(maxsize=1)
def get_settings() -> ToolcaseSettings:
    """Get the cached global settings instance."""
    return ToolcaseSettings()


def clear_settings_cache() -> None:
    """Clear settings cache; next get_settings() reloads from env."""
    get_settings.cache_clear()
