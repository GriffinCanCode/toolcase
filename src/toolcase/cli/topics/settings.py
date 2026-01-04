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

SETTINGS CLASSES:
    ToolcaseSettings     Root settings container
    CacheSettings        cache_enabled, cache_ttl
    LoggingSettings      log_level, log_format
    RetrySettings        max_retries, backoff_base
    HttpSettings         timeout, max_connections
    TracingSettings      enabled, service_name, exporter
    RateLimitSettings    requests_per_second, burst_size

ENVIRONMENT VARIABLES:
    TOOLCASE_CACHE__ENABLED=true
    TOOLCASE_CACHE__TTL=300
    TOOLCASE_RETRY__MAX_RETRIES=3
    TOOLCASE_TRACING__ENABLED=true
    TOOLCASE_TRACING__SERVICE_NAME=my-service

ENV-BASED AUTH (HTTP Tool):
    from toolcase import bearer_from_env, api_key_from_env, HttpTool
    
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
