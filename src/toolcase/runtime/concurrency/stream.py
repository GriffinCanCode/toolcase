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
from collections import deque
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")

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
    
    pending: dict[int, asyncio.Task[tuple[int, T | None, bool]]] = {}
    active: set[int] = set(range(len(streams)))
    iterators = list(streams)
    
    async def get_next(idx: int) -> tuple[int, T | None, bool]:
        try:
            value = await iterators[idx].__anext__()
            return (idx, value, True)
        except StopAsyncIteration:
            return (idx, None, False)
    
    # Start initial fetch for each stream
    for i in active:
        pending[i] = asyncio.create_task(get_next(i))
    
    while pending:
        done, _ = await asyncio.wait(pending.values(), return_when=asyncio.FIRST_COMPLETED)
        
        for task in done:
            idx, value, has_more = task.result()
            del pending[idx]
            
            if has_more:
                yield value  # type: ignore[misc]
                # Re-arm this stream
                pending[idx] = asyncio.create_task(get_next(idx))
            else:
                active.discard(idx)


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
    
    iterators = list(streams)
    active = list(range(len(streams)))
    
    while active:
        next_active: list[int] = []
        
        for idx in active:
            try:
                value = await iterators[idx].__anext__()
                yield value
                next_active.append(idx)
            except StopAsyncIteration:
                pass  # Stream exhausted
        
        active = next_active


async def buffer_stream(
    stream: AsyncIterator[T],
    maxsize: int = 10,
) -> AsyncIterator[T]:
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
    buffer: asyncio.Queue[T | None] = asyncio.Queue(maxsize=maxsize)
    done = asyncio.Event()
    error: BaseException | None = None
    
    async def producer() -> None:
        nonlocal error
        try:
            async for item in stream:
                await buffer.put(item)
        except BaseException as e:
            error = e
        finally:
            await buffer.put(None)  # Sentinel
            done.set()
    
    task = asyncio.create_task(producer())
    
    try:
        while True:
            item = await buffer.get()
            if item is None:
                break
            yield item
        
        if error:
            raise error
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def throttle_stream(
    stream: AsyncIterator[T],
    rate: float,
    *,
    per: float = 1.0,
    burst: int = 1,
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
    interval = per / rate
    tokens = float(burst)
    last_time = time.monotonic()
    
    async for item in stream:
        now = time.monotonic()
        elapsed = now - last_time
        tokens = min(burst, tokens + elapsed * rate / per)
        last_time = now
        
        if tokens < 1.0:
            wait_time = (1.0 - tokens) * interval
            await asyncio.sleep(wait_time)
            tokens = 0.0
        else:
            tokens -= 1.0
        
        yield item


async def batch_stream(
    stream: AsyncIterator[T],
    size: int,
    *,
    timeout: float | None = None,
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
    exhausted = False
    
    while not exhausted:
        if not batch and timeout:
            deadline = time.monotonic() + timeout
        
        try:
            if timeout and deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Timeout—flush current batch
                    if batch:
                        yield batch
                        batch = []
                    deadline = None
                    continue
                item = await asyncio.wait_for(stream.__anext__(), timeout=remaining)
            else:
                item = await stream.__anext__()
            
            batch.append(item)
            if len(batch) >= size:
                yield batch
                batch = []
                deadline = None
                
        except asyncio.TimeoutError:
            if batch:
                yield batch
                batch = []
            deadline = None
        except StopAsyncIteration:
            exhausted = True
    
    if batch:
        yield batch


async def timeout_stream(
    stream: AsyncIterator[T],
    timeout: float,
    *,
    on_timeout: Callable[[], Awaitable[T]] | T | None = None,
) -> AsyncIterator[T]:
    """Add timeout to stream item retrieval.
    
    If fetching the next item takes longer than timeout, either
    yields a default value or raises TimeoutError.
    
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
            item = await asyncio.wait_for(stream.__anext__(), timeout=timeout)
            yield item
        except asyncio.TimeoutError:
            if on_timeout is None:
                raise
            elif callable(on_timeout):
                yield await on_timeout()  # type: ignore[misc]
            else:
                yield on_timeout
        except StopAsyncIteration:
            break


async def take_stream(stream: AsyncIterator[T], n: int) -> AsyncIterator[T]:
    """Take first n items from stream."""
    count = 0
    async for item in stream:
        if count >= n:
            break
        yield item
        count += 1


async def skip_stream(stream: AsyncIterator[T], n: int) -> AsyncIterator[T]:
    """Skip first n items from stream."""
    count = 0
    async for item in stream:
        if count >= n:
            yield item
        count += 1


async def filter_stream(
    stream: AsyncIterator[T],
    predicate: Callable[[T], bool] | Callable[[T], Awaitable[bool]],
) -> AsyncIterator[T]:
    """Filter stream items by predicate."""
    async for item in stream:
        result = predicate(item)
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            yield item


async def map_stream(
    stream: AsyncIterator[T],
    func: Callable[[T], U] | Callable[[T], Awaitable[U]],
) -> AsyncIterator[U]:
    """Map function over stream items."""
    async for item in stream:
        result = func(item)
        if asyncio.iscoroutine(result):
            result = await result
        yield result  # type: ignore[misc]


async def flatten_stream(stream: AsyncIterator[AsyncIterator[T]]) -> AsyncIterator[T]:
    """Flatten nested async iterators."""
    async for inner in stream:
        async for item in inner:
            yield item


@dataclass
class StreamMerger(Generic[T]):
    """Configurable stream merger with advanced options.
    
    Provides more control over stream merging than merge_streams().
    
    Features:
        - Priority ordering
        - Error handling modes
        - Completion callbacks
        - Cancellation support
    
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
    
    def add(
        self,
        stream: AsyncIterator[T],
        *,
        priority: int = 0,
    ) -> StreamMerger[T]:
        """Add a stream to merge.
        
        Args:
            stream: Async iterator to add
            priority: Higher = checked first (default: 0)
        
        Returns:
            self for chaining
        """
        if self._started:
            raise RuntimeError("Cannot add streams after iteration started")
        self._streams.append((stream, priority))
        return self
    
    def __aiter__(self) -> AsyncIterator[T]:
        self._started = True
        # Sort by priority (higher first)
        sorted_streams = sorted(self._streams, key=lambda x: -x[1])
        streams = [s[0] for s in sorted_streams]
        return self._merge(streams)
    
    async def _merge(self, streams: list[AsyncIterator[T]]) -> AsyncIterator[T]:
        """Internal merge implementation."""
        if not streams:
            return
        
        pending: dict[int, asyncio.Task[tuple[int, T | None, bool, BaseException | None]]] = {}
        active: set[int] = set(range(len(streams)))
        
        async def get_next(idx: int) -> tuple[int, T | None, bool, BaseException | None]:
            try:
                value = await streams[idx].__anext__()
                return (idx, value, True, None)
            except StopAsyncIteration:
                return (idx, None, False, None)
            except BaseException as e:
                return (idx, None, False, e)
        
        # Start initial fetch
        for i in active:
            pending[i] = asyncio.create_task(get_next(i))
        
        while pending:
            done, _ = await asyncio.wait(pending.values(), return_when=asyncio.FIRST_COMPLETED)
            
            for task in done:
                idx, value, has_more, error = task.result()
                del pending[idx]
                
                if error:
                    if self.on_error == "propagate":
                        # Cancel other tasks
                        for t in pending.values():
                            t.cancel()
                        raise error
                    elif self.on_error == "stop":
                        for t in pending.values():
                            t.cancel()
                        return
                    # 'skip' - continue without this stream
                    active.discard(idx)
                    continue
                
                if has_more:
                    yield value  # type: ignore[misc]
                    pending[idx] = asyncio.create_task(get_next(idx))
                else:
                    active.discard(idx)


async def enumerate_stream(
    stream: AsyncIterator[T],
    start: int = 0,
) -> AsyncIterator[tuple[int, T]]:
    """Add index to stream items."""
    idx = start
    async for item in stream:
        yield (idx, item)
        idx += 1


async def zip_streams(*streams: AsyncIterator[object]) -> AsyncIterator[tuple[object, ...]]:
    """Zip multiple streams together.
    
    Yields tuples of items from each stream. Stops when shortest
    stream is exhausted.
    """
    if not streams:
        return
    
    iterators = list(streams)
    
    while True:
        items: list[object] = []
        for it in iterators:
            try:
                items.append(await it.__anext__())
            except StopAsyncIteration:
                return
        yield tuple(items)


async def chain_streams(*streams: AsyncIterator[T]) -> AsyncIterator[T]:
    """Chain multiple streams sequentially."""
    for stream in streams:
        async for item in stream:
            yield item
