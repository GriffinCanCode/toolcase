SETTINGS = """
TOPIC: settings
===============

Centralized configuration via environment variables and .env files.

GETTING SETTINGS:
    from toolcase import get_settings, clear_settings_cache
    
    settings = get_settings()
    print(settings.cache.enabled)
    print(settings.retry.max_retries)
    
    clear_settings_cache()  # Force reload

ENV FILE SUPPORT (priority order, later overrides earlier):
    .env                     Base configuration
    .env.{environment}       Environment-specific (.env.development)
    .env.{environment}.local Environment-specific local overrides
    .env.local               Local overrides (typically gitignored)

ENV UTILITIES:
    from toolcase import load_env, get_env, require_env, env
    
    load_env()                           # Load .env files with priority
    api_key = get_env("OPENAI_API_KEY")  # Get with optional default
    secret = require_env("DATABASE_URL") # Raises if missing
    debug = get_env("DEBUG", cast=bool)  # Type casting
    hosts = get_env("HOSTS", cast=list)  # Comma-separated list

SETTINGS CLASSES:
    ToolcaseSettings     Root settings (debug, environment)
    CacheSettings        enabled, ttl, max_size, redis_url
    LoggingSettings      level, format, include_timestamps
    RetrySettings        max_retries, base_delay, max_delay, jitter
    HttpSettings         timeout, max_response_size, verify_ssl
    TracingSettings      enabled, service_name, otlp_endpoint, sample_rate
    RateLimitSettings    enabled, max_calls, window_seconds, strategy

ENVIRONMENT VARIABLES:
    TOOLCASE_DEBUG=true
    TOOLCASE_ENVIRONMENT=production
    TOOLCASE_CACHE_ENABLED=true
    TOOLCASE_CACHE_TTL=3600
    TOOLCASE_LOG_LEVEL=INFO
    TOOLCASE_LOG_FORMAT=json
    TOOLCASE_RETRY_MAX_RETRIES=3
    TOOLCASE_HTTP_TIMEOUT=30
    TOOLCASE_TRACING_ENABLED=true
    TOOLCASE_TRACING_SERVICE_NAME=my-service
    TOOLCASE_TRACING_OTLP_ENDPOINT=http://localhost:4317
    TOOLCASE_RATELIMIT_MAX_CALLS=100
    TOOLCASE_RATELIMIT_WINDOW_SECONDS=60

ENV-BASED AUTH (HTTP Tool):
    from toolcase import bearer_from_env, api_key_from_env, HttpTool, HttpConfig
    
    # Load API keys from env vars (recommended for production)
    http = HttpTool(HttpConfig(auth=bearer_from_env("OPENAI_API_KEY")))
    http = HttpTool(HttpConfig(auth=api_key_from_env("ANTHROPIC_API_KEY")))

DIRECT INSTANTIATION:
    from toolcase import ToolcaseSettings, CacheSettings
    
    settings = ToolcaseSettings(
        cache=CacheSettings(enabled=True, ttl=600),
    )

COMPATIBLE WITH:
    python-dotenv, pydantic-settings, django-environ, python-decouple

RELATED TOPICS:
    toolcase help cache     Caching configuration
    toolcase help tracing   Distributed tracing
    toolcase help http      HTTP tool authentication
"""
