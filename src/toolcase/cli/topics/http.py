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
        allowed_methods=["GET", "POST"],
        timeout=30.0,
        auth=BearerAuth(token="sk-xxx"),
    ))

AUTH STRATEGIES:
    NoAuth()                        No authentication (default)
    BearerAuth(token="...")         OAuth2/JWT bearer token
    BasicAuth(user="...", pwd="...")  HTTP Basic auth
    ApiKeyAuth(key="...", header="X-API-Key")  API key header
    CustomAuth(headers={...})       Arbitrary headers

PARAMS:
    from toolcase import HttpParams
    
    params = HttpParams(
        url="https://api.example.com/users",
        method="POST",
        headers={"Content-Type": "application/json"},
        body={"name": "John"},
        timeout=10.0,
    )

RESPONSE:
    from toolcase import HttpResponse
    
    # Response fields
    response.status_code    HTTP status code
    response.headers        Response headers dict
    response.body           Response body (parsed JSON or text)

RELATED TOPICS:
    toolcase help tool     Tool creation
    toolcase help retry    Retry configuration
"""
