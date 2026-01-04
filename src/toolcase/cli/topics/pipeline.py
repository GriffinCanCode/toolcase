PIPELINE = """
TOPIC: pipeline
===============

Tool composition patterns - sequential and parallel execution.

SEQUENTIAL PIPELINE:
    from toolcase import pipeline
    
    # Chain tools: output of one feeds into next
    search_and_summarize = pipeline(
        search_tool,
        summarize_tool,
        format_tool,
    )
    result = await search_and_summarize(query="python async")

PARALLEL EXECUTION:
    from toolcase import parallel
    
    # Run tools concurrently, collect all results
    multi_search = parallel(
        google_search,
        bing_search,
        ddg_search,
    )
    results = await multi_search(query="python")

STREAMING PIPELINE:
    from toolcase import streaming_pipeline
    
    # Stream chunks through pipeline
    stream_pipe = streaming_pipeline(
        fetch_tool,
        transform_tool,
        output_tool,
    )
    async for chunk in stream_pipe(url="..."):
        print(chunk, end="", flush=True)

CUSTOM STEPS:
    from toolcase import Step
    
    class ValidateStep(Step):
        def transform(self, input_data, params):
            if not input_data:
                raise ValueError("Empty input")
            return input_data

RELATED TOPICS:
    toolcase help agents       Agentic composition (router, fallback)
    toolcase help streaming    Result streaming
"""
