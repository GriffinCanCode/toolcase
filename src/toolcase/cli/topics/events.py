EVENTS = """
TOPIC: events
=============

Type-safe signals for observer pattern and lifecycle hooks.

BASIC USAGE:
    from toolcase import Signal
    from typing import Callable
    
    # Define a typed signal
    on_data: Signal[Callable[[str, int], None]] = Signal()
    
    # Subscribe handlers
    def handler(name: str, value: int) -> None:
        print(f"{name}={value}")
    
    on_data += handler          # Using += operator
    on_data.subscribe(handler)  # Explicit method
    
    # Fire the signal
    on_data.fire("count", 42)   # Prints: count=42
    
    # Unsubscribe
    on_data -= handler
    on_data.unsubscribe(handler)

REGISTRY EVENTS:
    from toolcase import get_registry
    
    registry = get_registry()
    
    # Tool registration hook
    registry.on_register += lambda tool: print(f"Registered: {tool.metadata.name}")
    
    # Tool unregistration hook
    registry.on_unregister += lambda name: print(f"Unregistered: {name}")
    
    # Execution completion hook
    registry.on_execute += lambda name, params, result: log_execution(name, result)

ASYNC HANDLERS:
    from toolcase import Signal
    
    on_event: Signal[Callable[[dict], Awaitable[None] | None]] = Signal()
    
    async def async_handler(data: dict) -> None:
        await process_async(data)
    
    on_event += async_handler
    
    # fire() schedules but doesn't await async handlers
    on_event.fire({"key": "value"})
    
    # fire_async() properly awaits all handlers concurrently
    await on_event.fire_async({"key": "value"})

ONE-SHOT SUBSCRIPTIONS:
    from toolcase import Signal, one_shot
    
    signal: Signal[Callable[[], None]] = Signal()
    
    # Handler removed after first fire
    signal.once(my_handler)
    
    # Or use decorator
    @one_shot(signal)
    def handle_once() -> None:
        print("Only called once!")
    
    signal.fire()  # Calls handle_once
    signal.fire()  # handle_once already removed

PRIORITY ORDERING:
    signal: Signal[Callable[[str], None]] = Signal()
    
    signal.subscribe(low_priority, priority=0)     # Runs last
    signal.subscribe(high_priority, priority=100)  # Runs first
    signal.subscribe(medium_priority, priority=50) # Runs second

WEAK REFERENCES:
    # Prevent memory leaks by using weak refs
    signal: Signal[Callable[[str], None]] = Signal()
    
    def handler(msg: str) -> None:
        print(msg)
    
    signal.subscribe(handler, weak=True)
    
    # Handler auto-removed when gc'd (no reference kept)
    del handler
    signal.fire("test")  # Handler not called

PRACTICAL PATTERNS:

    Plugin System:
        class PluginManager:
            on_load: Signal[Callable[[Plugin], None]] = Signal()
            on_unload: Signal[Callable[[str], None]] = Signal()
            
            def load(self, plugin: Plugin) -> None:
                self._plugins[plugin.name] = plugin
                self.on_load.fire(plugin)
    
    Audit Logging:
        registry.on_execute += lambda name, _, result: (
            audit_log.info(f"Tool executed: {name}", result=result[:100])
        )
    
    Metrics Collection:
        def track_registration(tool: AnyTool) -> None:
            metrics.increment("tools.registered", tags={"category": tool.metadata.category})
        
        registry.on_register += track_registration

SIGNAL API REFERENCE:
    signal.subscribe(handler, priority=0, once=False, weak=False)
    signal.unsubscribe(handler) -> bool
    signal.once(handler, priority=0) -> handler
    signal.fire(*args, **kwargs)          # Sync fire
    await signal.fire_async(*args, **kwargs)  # Async fire
    signal.clear() -> int                 # Clear all handlers
    signal.handler_count -> int
    signal.is_empty -> bool
    handler in signal -> bool

RELATED TOPICS:
    toolcase help registry   Tool registration
    toolcase help middleware Middleware composition
    toolcase help effects    Effect system
"""
