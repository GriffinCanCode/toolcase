SETTINGS = """
TOPIC: settings
===============

Centralized configuration via environment variables.

GETTING SETTINGS:
    from toolcase import get_settings, clear_settings_cache
    
    settings = get_settings()
    print(settings.cache.enabled)
    print(settings.retry.max_retries)
    
    clear_settings_cache()  # Force reload

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

DIRECT INSTANTIATION:
    from toolcase import ToolcaseSettings, CacheSettings
    
    settings = ToolcaseSettings(
        cache=CacheSettings(enabled=True, ttl=600),
    )

RELATED TOPICS:
    toolcase help cache     Caching configuration
    toolcase help tracing   Distributed tracing
"""
