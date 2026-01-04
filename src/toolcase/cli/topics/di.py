DI = """
TOPIC: di
=========

Dependency injection for tool configuration.

CONTAINER:
    from toolcase import Container, Provider, Scope
    
    container = Container()
    
    # Register services
    container.register(DatabasePool, lambda: create_pool())
    container.register(Cache, lambda: RedisCache(), scope=Scope.SINGLETON)

SCOPES:
    Scope.TRANSIENT    New instance each time (default)
    Scope.SINGLETON    Single shared instance
    Scope.SCOPED       Instance per scope context

INJECTION:
    # Resolve dependencies
    pool = container.resolve(DatabasePool)
    
    # Scoped resolution
    with container.scope() as scope:
        service = scope.resolve(ScopedService)

TOOL INTEGRATION:
    class MyTool(BaseTool[Params]):
        def __init__(self, container: Container):
            self.db = container.resolve(DatabasePool)
        
        def _run(self, params):
            return self.db.query(params.query)

FACTORY PATTERN:
    from toolcase import Factory
    
    container.register(
        HttpClient,
        Factory(lambda config: HttpClient(config.timeout))
    )

RELATED TOPICS:
    toolcase help tool       Tool creation
    toolcase help settings   Configuration
"""
