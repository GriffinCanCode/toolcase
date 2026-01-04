AGENTS = """
TOPIC: agents
=============

Agentic composition primitives for intelligent tool orchestration.

ROUTER (Conditional dispatch):
    from toolcase import router, Route
    
    smart_search = router(
        Route(
            condition=lambda p: "code" in p["query"],
            tool=code_search_tool,
        ),
        Route(
            condition=lambda p: "image" in p["query"],
            tool=image_search_tool,
        ),
        default=general_search_tool,
    )

FALLBACK (Try alternatives on failure):
    from toolcase import fallback
    
    resilient_api = fallback(
        primary_api_tool,
        backup_api_tool,
        cache_tool,
    )

RACE (First to complete wins):
    from toolcase import race
    
    fastest_search = race(
        api_a_tool,
        api_b_tool,
        cache_tool,
        timeout=5.0,
    )

GATE (Conditional execution):
    from toolcase import gate
    
    premium_search = gate(
        condition=lambda p, ctx: ctx.get("user_tier") == "premium",
        tool=premium_search_tool,
        fallback=basic_search_tool,
    )

ESCALATION (Human-in-the-loop):
    from toolcase import retry_with_escalation, QueueEscalation
    
    safe_tool = retry_with_escalation(
        dangerous_tool,
        max_auto_retries=2,
        escalation=QueueEscalation(queue_name="human_review"),
    )

RELATED TOPICS:
    toolcase help pipeline     Sequential/parallel composition
    toolcase help concurrency  Async primitives
"""
