HTTP = """
TOPIC: http
===========

Built-in HTTP tool with security and authentication.

BASIC USAGE:
    from toolcase import HttpTool
    
    http = HttpTool()
    result = await http.acall(url="https://api.example.com/data")

CONFIGURATION:
    from toolcase import HttpTool, HttpConfig, BearerAuth
    
    http = HttpTool(HttpConfig(
        allowed_hosts=["api.example.com", "*.internal.corp"],
        blocked_hosts=["localhost", "*.local"],  # SSRF protection
        allowed_methods=["GET", "POST"],
        default_timeout=30.0,
        max_response_size="10MB",
        auth=BearerAuth(token=SecretStr("sk-xxx")),
    ))

AUTH STRATEGIES (Direct):
    NoAuth()                                   No auth (default)
    BearerAuth(token=SecretStr("..."))         OAuth2/JWT bearer
    BasicAuth(username="...", password=SecretStr("..."))  HTTP Basic
    ApiKeyAuth(key=SecretStr("..."), header_name="X-API-Key")
    CustomAuth(headers={"X-Custom": SecretStr("...")})

AUTH STRATEGIES (Environment-Based - Recommended):
    from toolcase import bearer_from_env, api_key_from_env, basic_from_env
    
    # Load secrets from env vars (production-safe, no hardcoded tokens)
    http = HttpTool(HttpConfig(auth=bearer_from_env("OPENAI_API_KEY")))
    http = HttpTool(HttpConfig(auth=api_key_from_env("ANTHROPIC_API_KEY", header="x-api-key")))
    http = HttpTool(HttpConfig(auth=basic_from_env("API_USER", "API_PASS")))
    
    # Or explicit env auth classes:
    from toolcase import EnvBearerAuth, EnvApiKeyAuth, EnvBasicAuth
    http = HttpTool(HttpConfig(auth=EnvBearerAuth(env_var="MY_TOKEN")))

PARAMS:
    from toolcase import HttpParams
    
    params = HttpParams(
        url="https://api.example.com/users",
        method="POST",
        headers={"X-Request-ID": "abc123"},
        json_body={"name": "John", "email": "john@example.com"},
        timeout=10.0,
    )

RESPONSE FIELDS:
    response.status_code     HTTP status code (100-599)
    response.headers         Response headers dict
    response.body            Response body string
    response.is_success      True if 2xx
    response.is_client_error True if 4xx
    response.is_server_error True if 5xx
    response.content_type    Parsed Content-Type header
    response.elapsed_ms      Request duration

STREAMING (Large Responses):
    async for chunk in http.stream_result(params):
        print(chunk, end="", flush=True)

RELATED TOPICS:
    toolcase help tool      Tool creation
    toolcase help settings  Environment configuration
    toolcase help retry     Retry configuration
"""
