"""Async stream utilities and combinators.

Provides functions for working with async iterators/generators:
merging, buffering, throttling, batching, and timeouts.

Key Operations:
    - merge_streams: Combine multiple streams into one
    - interleave_streams: Round-robin interleaving
    - buffer_stream: Add buffering for smoother consumption
    - throttle_stream: Rate limit stream items
    - batch_stream: Group items into batches
    - timeout_stream: Add timeout to stream consumption

Example:
    >>> # Merge multiple data sources
    >>> async for item in merge_streams(source1, source2, source3):
    ...     process(item)
    
    >>> # Throttle to 10 items/second
    >>> async for item in throttle_stream(fast_source, rate=10.0):
    ...     handle(item)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

T, U = TypeVar("T"), TypeVar("U")

__all__ = [
    "merge_streams",
    "interleave_streams",
    "buffer_stream",
    "throttle_stream",
    "batch_stream",
    "timeout_stream",
    "take_stream",
    "skip_stream",
    "filter_stream",
    "map_stream",
    "flatten_stream",
    "StreamMerger",
    "enumerate_stream",
    "zip_streams",
    "chain_streams",
]


async def merge_streams(*streams: AsyncIterator[T]) -> AsyncIterator[T]:
    """Merge multiple async streams into one.
    
    Items are yielded as they become available from any stream.
    Continues until all streams are exhausted.
    
    Example:
        >>> async def stream_a():
        ...     for i in [1, 3, 5]:
        ...         await asyncio.sleep(0.1)
        ...         yield i
        >>> 
        >>> async def stream_b():
        ...     for i in [2, 4, 6]:
        ...         await asyncio.sleep(0.15)
        ...         yield i
        >>> 
        >>> async for x in merge_streams(stream_a(), stream_b()):
        ...     print(x)  # Interleaved based on timing
    """
    if not streams:
        return
    
    iters = list(streams)
    
    async def get_next(idx: int) -> tuple[int, T | None, bool]:
        try:
            return (idx, await iters[idx].__anext__(), True)
        except StopAsyncIteration:
            return (idx, None, False)
    
    # Start initial fetch for each stream
    pending = {i: asyncio.create_task(get_next(i)) for i in range(len(streams))}
    
    while pending:
        done, _ = await asyncio.wait(pending.values(), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            idx, value, has_more = task.result()
            del pending[idx]
            if has_more:
                yield value  # type: ignore[misc]
                pending[idx] = asyncio.create_task(get_next(idx))  # Re-arm


async def interleave_streams(*streams: AsyncIterator[T]) -> AsyncIterator[T]:
    """Interleave streams in round-robin fashion.
    
    Unlike merge_streams, this yields one item from each stream in order,
    cycling through streams. Faster streams wait for slower ones.
    
    Example:
        >>> async for x in interleave_streams(stream_a, stream_b):
        ...     print(x)  # a1, b1, a2, b2, ...
    """
    if not streams:
        return
    
    iters = list(streams)
    active = list(range(len(streams)))
    
    while active:
        next_active: list[int] = []
        for idx in active:
            try:
                yield await iters[idx].__anext__()
                next_active.append(idx)
            except StopAsyncIteration:
                pass  # Stream exhausted
        active = next_active


async def buffer_stream(stream: AsyncIterator[T], maxsize: int = 10) -> AsyncIterator[T]:
    """Buffer stream items for smoother consumption.
    
    Pre-fetches items from the source stream into a buffer.
    Useful when producer and consumer have variable speeds.
    
    Args:
        stream: Source async iterator
        maxsize: Maximum buffer size
    
    Example:
        >>> # Buffer up to 100 items from slow producer
        >>> async for item in buffer_stream(slow_producer, maxsize=100):
        ...     fast_process(item)
    """
    buf: asyncio.Queue[T | None] = asyncio.Queue(maxsize=maxsize)
    error: BaseException | None = None
    
    async def producer() -> None:
        nonlocal error
        try:
            async for item in stream:
                await buf.put(item)
        except BaseException as e:
            error = e
        finally:
            await buf.put(None)  # Sentinel
    
    task = asyncio.create_task(producer())
    try:
        while (item := await buf.get()) is not None:
            yield item
        if error:
            raise error
    finally:
        task.cancel()
        with asyncio.suppress(asyncio.CancelledError):
            await task


async def throttle_stream(
    stream: AsyncIterator[T], rate: float, *, per: float = 1.0, burst: int = 1,
) -> AsyncIterator[T]:
    """Rate-limit stream consumption.
    
    Ensures items are yielded no faster than the specified rate.
    Useful for API rate limits or resource protection.
    
    Args:
        stream: Source async iterator
        rate: Maximum items per `per` seconds
        per: Time window (default: 1 second)
        burst: Allow burst of this many items before throttling
    
    Example:
        >>> # Max 10 items per second
        >>> async for item in throttle_stream(fast_source, rate=10):
        ...     await api_call(item)
    """
    interval, tokens, last = per / rate, float(burst), time.monotonic()
    
    async for item in stream:
        now = time.monotonic()
        tokens = min(burst, tokens + (now - last) * rate / per)
        last = now
        
        if tokens < 1.0:
            await asyncio.sleep((1.0 - tokens) * interval)
            tokens = 0.0
        else:
            tokens -= 1.0
        yield item


async def batch_stream(
    stream: AsyncIterator[T], size: int, *, timeout: float | None = None,
) -> AsyncIterator[list[T]]:
    """Group stream items into batches.
    
    Collects items until batch is full or timeout expires.
    
    Args:
        stream: Source async iterator
        size: Maximum batch size
        timeout: Optional max time to wait for full batch
    
    Example:
        >>> # Process in batches of 100
        >>> async for batch in batch_stream(items, size=100):
        ...     await bulk_insert(batch)
    """
    batch: list[T] = []
    deadline: float | None = None
    
    while True:
        if not batch and timeout:
            deadline = time.monotonic() + timeout
        try:
            if timeout and deadline and (remaining := deadline - time.monotonic()) <= 0:
                if batch:  # Timeout—flush current batch
                    yield batch
                    batch = []
                deadline = None
                continue
            item = await (asyncio.wait_for(stream.__anext__(), timeout=remaining) 
                         if timeout and deadline else stream.__anext__())
            batch.append(item)
            if len(batch) >= size:
                yield batch
                batch, deadline = [], None
        except asyncio.TimeoutError:
            if batch:
                yield batch
            batch, deadline = [], None
        except StopAsyncIteration:
            break
    if batch:
        yield batch


async def timeout_stream(
    stream: AsyncIterator[T], timeout: float, *, on_timeout: Callable[[], Awaitable[T]] | T | None = None,
) -> AsyncIterator[T]:
    """Add timeout to stream item retrieval.
    
    If fetching the next item takes longer than timeout, either yields a default value or raises TimeoutError.
    
    Args:
        stream: Source async iterator
        timeout: Max seconds to wait for each item
        on_timeout: Value or async callable to yield on timeout (or raise if None)
    
    Example:
        >>> async for item in timeout_stream(slow_source, timeout=5.0):
        ...     process(item)  # Each item must arrive within 5s
    """
    while True:
        try:
            yield await asyncio.wait_for(stream.__anext__(), timeout=timeout)
        except asyncio.TimeoutError:
            if on_timeout is None:
                raise
            yield await on_timeout() if callable(on_timeout) else on_timeout  # type: ignore[misc]
        except StopAsyncIteration:
            break


async def take_stream(stream: AsyncIterator[T], n: int) -> AsyncIterator[T]:
    """Take first n items from stream."""
    async for i, item in enumerate_stream(stream):
        if i >= n:
            break
        yield item


async def skip_stream(stream: AsyncIterator[T], n: int) -> AsyncIterator[T]:
    """Skip first n items from stream."""
    async for i, item in enumerate_stream(stream):
        if i >= n:
            yield item


async def filter_stream(
    stream: AsyncIterator[T], predicate: Callable[[T], bool] | Callable[[T], Awaitable[bool]],
) -> AsyncIterator[T]:
    """Filter stream items by predicate."""
    is_async = asyncio.iscoroutinefunction(predicate)
    async for item in stream:
        if (await predicate(item)) if is_async else predicate(item):  # type: ignore[misc]
            yield item


async def map_stream(
    stream: AsyncIterator[T], func: Callable[[T], U] | Callable[[T], Awaitable[U]],
) -> AsyncIterator[U]:
    """Map function over stream items."""
    is_async = asyncio.iscoroutinefunction(func)
    async for item in stream:
        yield (await func(item)) if is_async else func(item)  # type: ignore[misc]


async def flatten_stream(stream: AsyncIterator[AsyncIterator[T]]) -> AsyncIterator[T]:
    """Flatten nested async iterators."""
    async for inner in stream:
        async for item in inner:
            yield item


@dataclass
class StreamMerger(Generic[T]):
    """Configurable stream merger with advanced options.
    
    Provides more control over stream merging than merge_streams().
    
    Features: Priority ordering, error handling modes, completion callbacks, cancellation support
    
    Example:
        >>> merger = StreamMerger()
        >>> merger.add(stream1, priority=1)
        >>> merger.add(stream2, priority=2)
        >>> 
        >>> async for item in merger:
        ...     process(item)
    """
    
    on_error: str = "propagate"  # 'propagate', 'skip', 'stop'
    _streams: list[tuple[AsyncIterator[T], int]] = field(default_factory=list, repr=False)
    _started: bool = field(default=False, repr=False)
    
    def add(self, stream: AsyncIterator[T], *, priority: int = 0) -> StreamMerger[T]:
        """Add a stream to merge. Args: stream (async iterator), priority (higher=first). Returns self."""
        if self._started:
            raise RuntimeError("Cannot add streams after iteration started")
        self._streams.append((stream, priority))
        return self
    
    def __aiter__(self) -> AsyncIterator[T]:
        self._started = True
        return self._merge([s for s, _ in sorted(self._streams, key=lambda x: -x[1])])
    
    async def _merge(self, streams: list[AsyncIterator[T]]) -> AsyncIterator[T]:
        """Internal merge implementation."""
        if not streams:
            return
        
        async def get_next(idx: int) -> tuple[int, T | None, bool, BaseException | None]:
            try:
                return (idx, await streams[idx].__anext__(), True, None)
            except StopAsyncIteration:
                return (idx, None, False, None)
            except BaseException as e:
                return (idx, None, False, e)
        
        # Start initial fetch
        pending = {i: asyncio.create_task(get_next(i)) for i in range(len(streams))}
        
        while pending:
            done, _ = await asyncio.wait(pending.values(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                idx, value, has_more, error = task.result()
                del pending[idx]
                
                if error:
                    if self.on_error in ("propagate", "stop"):
                        for t in pending.values():
                            t.cancel()
                        if self.on_error == "propagate":
                            raise error
                        return
                    continue  # 'skip' - continue without this stream
                
                if has_more:
                    yield value  # type: ignore[misc]
                    pending[idx] = asyncio.create_task(get_next(idx))


async def enumerate_stream(stream: AsyncIterator[T], start: int = 0) -> AsyncIterator[tuple[int, T]]:
    """Add index to stream items."""
    idx = start
    async for item in stream:
        yield (idx, item)
        idx += 1


async def zip_streams(*streams: AsyncIterator[object]) -> AsyncIterator[tuple[object, ...]]:
    """Zip multiple streams together. Yields tuples of items from each stream. Stops when shortest stream is exhausted."""
    if not streams:
        return
    
    iters = list(streams)
    while True:
        try:
            yield tuple([await it.__anext__() for it in iters])
        except StopAsyncIteration:
            return


async def chain_streams(*streams: AsyncIterator[T]) -> AsyncIterator[T]:
    """Chain multiple streams sequentially."""
    for stream in streams:
        async for item in stream:
            yield item
