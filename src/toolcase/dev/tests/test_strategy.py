"""Tests for advanced cache invalidation strategies."""

import asyncio
import time

import pytest

from toolcase.io.cache import (
    DEFAULT_STALE_MULTIPLIER,
    PatternMatcher,
    SWRCache,
    SWRConfig,
    SWRState,
    TaggedEntry,
    TaggedMemoryCache,
    TagIndex,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TagIndex Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTagIndex:
    """Tests for TagIndex reverse indexing."""
    
    def test_add_and_lookup(self) -> None:
        """Test basic add and tag lookup."""
        idx = TagIndex()
        idx.add("key1", frozenset({"user:123", "product"}))
        idx.add("key2", frozenset({"user:123", "order"}))
        idx.add("key3", frozenset({"product"}))
        
        assert idx.keys_for_tag("user:123") == frozenset({"key1", "key2"})
        assert idx.keys_for_tag("product") == frozenset({"key1", "key3"})
        assert idx.keys_for_tag("order") == frozenset({"key2"})
        assert idx.keys_for_tag("nonexistent") == frozenset()
    
    def test_keys_for_tags_union(self) -> None:
        """Test multi-tag lookup returns union."""
        idx = TagIndex()
        idx.add("key1", frozenset({"a"}))
        idx.add("key2", frozenset({"b"}))
        idx.add("key3", frozenset({"a", "b"}))
        
        assert idx.keys_for_tags(frozenset({"a", "b"})) == frozenset({"key1", "key2", "key3"})
        assert idx.keys_for_tags(frozenset({"a"})) == frozenset({"key1", "key3"})
    
    def test_remove_cleans_up(self) -> None:
        """Test remove cleans both directions."""
        idx = TagIndex()
        idx.add("key1", frozenset({"tag1", "tag2"}))
        
        tags = idx.remove("key1")
        assert tags == frozenset({"tag1", "tag2"})
        assert idx.keys_for_tag("tag1") == frozenset()
        assert idx.tags_for_key("key1") == frozenset()
    
    def test_remove_nonexistent(self) -> None:
        """Test remove of nonexistent key returns empty."""
        idx = TagIndex()
        assert idx.remove("nonexistent") == frozenset()
    
    def test_add_empty_tags(self) -> None:
        """Test adding empty tags is no-op."""
        idx = TagIndex()
        idx.add("key", frozenset())
        assert idx.key_count == 0
        assert idx.tag_count == 0
    
    def test_clear(self) -> None:
        """Test clear removes all mappings."""
        idx = TagIndex()
        idx.add("key1", frozenset({"tag1"}))
        idx.add("key2", frozenset({"tag2"}))
        idx.clear()
        
        assert idx.key_count == 0
        assert idx.tag_count == 0
    
    def test_tag_count_and_key_count(self) -> None:
        """Test count properties."""
        idx = TagIndex()
        idx.add("k1", frozenset({"a", "b"}))
        idx.add("k2", frozenset({"b", "c"}))
        
        assert idx.key_count == 2
        assert idx.tag_count == 3  # a, b, c


# ═══════════════════════════════════════════════════════════════════════════════
# PatternMatcher Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPatternMatcher:
    """Tests for PatternMatcher glob and regex patterns."""
    
    def test_glob_simple(self) -> None:
        """Test simple glob patterns."""
        m = PatternMatcher.glob("user:*")
        assert m.matches("user:123")
        assert m.matches("user:")
        assert not m.matches("order:123")
    
    def test_glob_middle_wildcard(self) -> None:
        """Test glob with wildcard in middle."""
        m = PatternMatcher.glob("tool:*:search")
        assert m.matches("tool:123:search")
        assert m.matches("tool::search")
        assert not m.matches("tool:123:query")
    
    def test_glob_question_mark(self) -> None:
        """Test glob single char wildcard."""
        m = PatternMatcher.glob("v?")
        assert m.matches("v1")
        assert m.matches("va")
        assert not m.matches("v12")
    
    def test_regex_pattern(self) -> None:
        """Test regex pattern matching."""
        m = PatternMatcher.regex(r"tool:\d+:.*")
        assert m.matches("tool:123:search")
        assert m.matches("tool:1:x")
        assert not m.matches("tool:abc:search")
    
    def test_filter_keys(self) -> None:
        """Test filter returns matching subset."""
        m = PatternMatcher.glob("user:*")
        keys = frozenset({"user:1", "user:2", "order:1", "product:1"})
        assert m.filter(keys) == frozenset({"user:1", "user:2"})
    
    def test_repr(self) -> None:
        """Test repr shows pattern type."""
        assert "glob" in repr(PatternMatcher.glob("test"))
        assert "regex" in repr(PatternMatcher.regex("test"))


# ═══════════════════════════════════════════════════════════════════════════════
# TaggedEntry Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTaggedEntry:
    """Tests for TaggedEntry state tracking."""
    
    def test_fresh_entry(self) -> None:
        """Test fresh entry state."""
        now = time.time()
        entry = TaggedEntry("value", expires_at=now + 60, stale_at=now + 120)
        
        assert entry.is_fresh
        assert not entry.expired
        assert not entry.stale
    
    def test_expired_not_stale(self) -> None:
        """Test entry past TTL but within stale window."""
        now = time.time()
        entry = TaggedEntry("value", expires_at=now - 1, stale_at=now + 60)
        
        assert not entry.is_fresh
        assert entry.expired  # past soft expiry
        assert not entry.stale  # within stale window
    
    def test_stale_entry(self) -> None:
        """Test entry past stale window."""
        now = time.time()
        entry = TaggedEntry("value", expires_at=now - 60, stale_at=now - 1)
        
        assert not entry.is_fresh
        assert entry.expired
        assert entry.stale


# ═══════════════════════════════════════════════════════════════════════════════
# SWRState Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSWRState:
    """Tests for SWRState properties."""
    
    def test_fresh_hit(self) -> None:
        """Test fresh cache hit."""
        state = SWRState(value="data", is_fresh=True, is_expired=False)
        assert state.hit
        assert not state.should_revalidate
    
    def test_stale_hit(self) -> None:
        """Test stale hit triggers revalidation."""
        state = SWRState(value="data", is_stale=True, is_expired=False)
        assert state.hit
        assert state.should_revalidate
    
    def test_stale_already_revalidating(self) -> None:
        """Test stale but already revalidating."""
        state = SWRState(value="data", is_stale=True, is_expired=False, revalidating=True)
        assert state.hit
        assert not state.should_revalidate
    
    def test_expired_miss(self) -> None:
        """Test expired is a miss."""
        state = SWRState(value=None, is_expired=True)
        assert not state.hit
        assert not state.should_revalidate


# ═══════════════════════════════════════════════════════════════════════════════
# SWRConfig Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSWRConfig:
    """Tests for SWRConfig."""
    
    def test_default_multiplier(self) -> None:
        """Test default stale multiplier."""
        cfg = SWRConfig()
        assert cfg.stale_multiplier == DEFAULT_STALE_MULTIPLIER
    
    def test_stale_window_calculation(self) -> None:
        """Test stale window from TTL."""
        cfg = SWRConfig(stale_multiplier=3.0)
        assert cfg.stale_window(60.0) == 180.0


# ═══════════════════════════════════════════════════════════════════════════════
# TaggedMemoryCache Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTaggedMemoryCache:
    """Tests for TaggedMemoryCache."""
    
    def test_basic_get_set(self) -> None:
        """Test basic get/set operations."""
        cache = TaggedMemoryCache()
        cache.set("tool", {"q": "test"}, "result")
        assert cache.get("tool", {"q": "test"}) == "result"
        assert cache.get("tool", {"q": "other"}) is None
    
    def test_set_with_tags(self) -> None:
        """Test setting entries with tags."""
        cache = TaggedMemoryCache()
        cache.set("tool", {"q": "a"}, "result_a", tags=frozenset({"user:1", "type:search"}))
        cache.set("tool", {"q": "b"}, "result_b", tags=frozenset({"user:2", "type:search"}))
        
        assert cache.get("tool", {"q": "a"}) == "result_a"
        stats = cache.stats()
        assert stats["tag_count"] == 3  # user:1, user:2, type:search
    
    def test_invalidate_by_tag(self) -> None:
        """Test tag-based invalidation."""
        cache = TaggedMemoryCache()
        cache.set("tool", {"q": "a"}, "result_a", tags=frozenset({"user:1"}))
        cache.set("tool", {"q": "b"}, "result_b", tags=frozenset({"user:1"}))
        cache.set("tool", {"q": "c"}, "result_c", tags=frozenset({"user:2"}))
        
        removed = cache.invalidate_by_tag("user:1")
        assert removed == 2
        assert cache.get("tool", {"q": "a"}) is None
        assert cache.get("tool", {"q": "b"}) is None
        assert cache.get("tool", {"q": "c"}) == "result_c"
    
    def test_invalidate_by_tags_union(self) -> None:
        """Test multi-tag invalidation."""
        cache = TaggedMemoryCache()
        cache.set("tool", {"q": "a"}, "result_a", tags=frozenset({"tag1"}))
        cache.set("tool", {"q": "b"}, "result_b", tags=frozenset({"tag2"}))
        cache.set("tool", {"q": "c"}, "result_c", tags=frozenset({"tag3"}))
        
        removed = cache.invalidate_by_tags(frozenset({"tag1", "tag2"}))
        assert removed == 2
        assert cache.get("tool", {"q": "c"}) == "result_c"
    
    def test_invalidate_by_pattern(self) -> None:
        """Test pattern-based invalidation."""
        cache = TaggedMemoryCache()
        cache.set("search", {"q": "a"}, "result_a")
        cache.set("search", {"q": "b"}, "result_b")
        cache.set("query", {"q": "c"}, "result_c")
        
        removed = cache.invalidate_by_pattern("search:*")
        assert removed == 2
        assert cache.get("search", {"q": "a"}) is None
        assert cache.get("query", {"q": "c"}) == "result_c"
    
    def test_invalidate_tool(self) -> None:
        """Test invalidating all entries for a tool."""
        cache = TaggedMemoryCache()
        cache.set("tool1", {"q": "a"}, "result")
        cache.set("tool1", {"q": "b"}, "result")
        cache.set("tool2", {"q": "a"}, "result")
        
        removed = cache.invalidate_tool("tool1")
        assert removed == 2
        assert cache.get("tool2", {"q": "a"}) == "result"
    
    def test_swr_fresh_state(self) -> None:
        """Test SWR state for fresh entry."""
        cache = TaggedMemoryCache(swr=SWRConfig())
        cache.set("tool", {"q": "test"}, "result")
        
        state = cache.get_swr("tool", {"q": "test"})
        assert state.is_fresh
        assert state.value == "result"
        assert not state.should_revalidate
    
    def test_swr_stale_state(self) -> None:
        """Test SWR state for stale entry."""
        cache = TaggedMemoryCache(default_ttl=0.01, swr=SWRConfig(stale_multiplier=10.0))
        cache.set("tool", {"q": "test"}, "result")
        
        time.sleep(0.02)  # Past TTL but within stale window
        state = cache.get_swr("tool", {"q": "test"})
        assert state.is_stale
        assert state.value == "result"
        assert state.should_revalidate
    
    def test_swr_expired_state(self) -> None:
        """Test SWR state for expired entry."""
        cache = TaggedMemoryCache(default_ttl=0.01, swr=SWRConfig(stale_multiplier=1.0))
        cache.set("tool", {"q": "test"}, "result")
        
        time.sleep(0.02)  # Past stale window
        state = cache.get_swr("tool", {"q": "test"})
        assert state.is_expired
        assert state.value is None
    
    def test_mark_revalidating(self) -> None:
        """Test revalidation tracking."""
        cache = TaggedMemoryCache(swr=SWRConfig())
        cache.set("tool", {"q": "test"}, "result")
        
        assert cache.mark_revalidating("tool", {"q": "test"})
        assert not cache.mark_revalidating("tool", {"q": "test"})  # Already marked
        
        cache.clear_revalidating("tool", {"q": "test"})
        assert cache.mark_revalidating("tool", {"q": "test"})
    
    def test_max_concurrent_revalidations(self) -> None:
        """Test revalidation limit."""
        cache = TaggedMemoryCache(swr=SWRConfig(max_concurrent_revalidations=2))
        
        cache.set("tool", {"q": "1"}, "r")
        cache.set("tool", {"q": "2"}, "r")
        cache.set("tool", {"q": "3"}, "r")
        
        assert cache.mark_revalidating("tool", {"q": "1"})
        assert cache.mark_revalidating("tool", {"q": "2"})
        assert not cache.mark_revalidating("tool", {"q": "3"})  # At limit
    
    def test_eviction_removes_stale_first(self) -> None:
        """Test eviction prioritizes stale entries."""
        cache = TaggedMemoryCache(max_entries=5, default_ttl=0.01, swr=SWRConfig(stale_multiplier=1.0))
        
        # Fill cache with entries that will become stale
        for i in range(3):
            cache.set("tool", {"i": i}, f"result_{i}")
        
        time.sleep(0.02)  # Let them go stale
        
        # Add fresh entries to trigger eviction
        for i in range(5):
            cache.set("fresh", {"i": i}, f"fresh_{i}")
        
        assert cache.size <= 5
    
    def test_stats(self) -> None:
        """Test stats output."""
        cache = TaggedMemoryCache(default_ttl=60, max_entries=100, swr=SWRConfig())
        cache.set("tool", {"q": "a"}, "result", tags=frozenset({"tag1"}))
        
        stats = cache.stats()
        assert stats["backend"] == "memory_tagged"
        assert stats["total_entries"] == 1
        assert stats["tag_count"] == 1
        assert stats["swr_enabled"] is True
    
    def test_clear(self) -> None:
        """Test clear removes everything."""
        cache = TaggedMemoryCache(swr=SWRConfig())
        cache.set("tool", {"q": "a"}, "result", tags=frozenset({"tag1"}))
        cache.mark_revalidating("tool", {"q": "a"})
        
        cache.clear()
        
        assert cache.size == 0
        stats = cache.stats()
        assert stats["tag_count"] == 0
        assert stats["revalidating"] == 0
    
    def test_ping(self) -> None:
        """Test ping always returns True."""
        cache = TaggedMemoryCache()
        assert cache.ping() is True
    
    def test_update_entry_updates_tags(self) -> None:
        """Test updating entry updates tag index."""
        cache = TaggedMemoryCache()
        cache.set("tool", {"q": "a"}, "result1", tags=frozenset({"old_tag"}))
        cache.set("tool", {"q": "a"}, "result2", tags=frozenset({"new_tag"}))
        
        # Old tag should no longer point to key
        assert cache.invalidate_by_tag("old_tag") == 0
        assert cache.invalidate_by_tag("new_tag") == 1


# ═══════════════════════════════════════════════════════════════════════════════
# SWRCache Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSWRCache:
    """Tests for SWRCache wrapper."""
    
    @pytest.mark.asyncio
    async def test_fresh_hit_no_revalidation(self) -> None:
        """Test fresh hit doesn't trigger revalidation."""
        revalidation_count = 0
        
        async def revalidator(tool: str, params: dict) -> str | None:
            nonlocal revalidation_count
            revalidation_count += 1
            return "new_value"
        
        cache = TaggedMemoryCache(swr=SWRConfig())
        cache.set("tool", {"q": "test"}, "cached_value")
        swr = SWRCache(cache, revalidator)
        
        result = await swr.get_or_revalidate("tool", {"q": "test"})
        assert result == "cached_value"
        assert revalidation_count == 0
    
    @pytest.mark.asyncio
    async def test_stale_triggers_revalidation(self) -> None:
        """Test stale hit triggers background revalidation."""
        revalidation_started = asyncio.Event()
        
        async def revalidator(tool: str, params: dict) -> str | None:
            revalidation_started.set()
            return "new_value"
        
        cache = TaggedMemoryCache(default_ttl=0.01, swr=SWRConfig(stale_multiplier=10.0))
        cache.set("tool", {"q": "test"}, "stale_value")
        swr = SWRCache(cache, revalidator)
        
        time.sleep(0.02)  # Let entry go stale
        
        result = await swr.get_or_revalidate("tool", {"q": "test"})
        assert result == "stale_value"  # Returns stale immediately
        
        # Wait for background revalidation
        await asyncio.wait_for(revalidation_started.wait(), timeout=1.0)
        await asyncio.sleep(0.01)  # Let it complete
        
        # New value should be cached
        assert cache.get("tool", {"q": "test"}) == "new_value"
    
    @pytest.mark.asyncio
    async def test_expired_returns_none(self) -> None:
        """Test expired entry returns None."""
        async def revalidator(tool: str, params: dict) -> str | None:
            return "value"
        
        cache = TaggedMemoryCache(default_ttl=0.01, swr=SWRConfig(stale_multiplier=1.0))
        cache.set("tool", {"q": "test"}, "value")
        swr = SWRCache(cache, revalidator)
        
        time.sleep(0.02)  # Past stale window
        
        result = await swr.get_or_revalidate("tool", {"q": "test"})
        assert result is None
    
    @pytest.mark.asyncio
    async def test_revalidation_timeout(self) -> None:
        """Test revalidation respects timeout."""
        async def slow_revalidator(tool: str, params: dict) -> str | None:
            await asyncio.sleep(10)  # Very slow
            return "value"
        
        # Use longer TTL and stale window so entry doesn't expire during test
        cache = TaggedMemoryCache(default_ttl=0.05, swr=SWRConfig(stale_multiplier=100.0))
        cache.set("tool", {"q": "test"}, "stale_value")
        swr = SWRCache(cache, slow_revalidator, SWRConfig(revalidation_timeout=0.1))
        
        time.sleep(0.06)  # Past TTL but within stale window (5s)
        
        result = await swr.get_or_revalidate("tool", {"q": "test"})
        assert result == "stale_value"
        
        await asyncio.sleep(0.2)  # Wait for timeout
        
        # Should still have old value (revalidation timed out, not updated)
        # Value is stale but not expired, so still in cache
        state = cache.get_swr("tool", {"q": "test"})
        assert state.value == "stale_value"
    
    @pytest.mark.asyncio
    async def test_revalidation_error_swallowed(self) -> None:
        """Test revalidation errors don't propagate."""
        async def failing_revalidator(tool: str, params: dict) -> str | None:
            raise ValueError("API error")
        
        cache = TaggedMemoryCache(default_ttl=0.01, swr=SWRConfig(stale_multiplier=10.0))
        cache.set("tool", {"q": "test"}, "stale_value")
        swr = SWRCache(cache, failing_revalidator)
        
        time.sleep(0.02)
        
        result = await swr.get_or_revalidate("tool", {"q": "test"})
        assert result == "stale_value"
        
        await asyncio.sleep(0.1)  # Let revalidation fail
        
        # Revalidation flag should be cleared
        assert swr.pending_revalidations == 0
    
    @pytest.mark.asyncio
    async def test_close_cancels_tasks(self) -> None:
        """Test close cancels pending revalidations."""
        started = asyncio.Event()
        
        async def slow_revalidator(tool: str, params: dict) -> str | None:
            started.set()
            await asyncio.sleep(10)
            return "value"
        
        cache = TaggedMemoryCache(default_ttl=0.01, swr=SWRConfig(stale_multiplier=10.0))
        cache.set("tool", {"q": "test"}, "value")
        swr = SWRCache(cache, slow_revalidator)
        
        time.sleep(0.02)
        
        await swr.get_or_revalidate("tool", {"q": "test"})
        await started.wait()  # Ensure task started
        
        await swr.close()
        
        assert swr.pending_revalidations == 0
    
    @pytest.mark.asyncio
    async def test_with_tags_passed_to_set(self) -> None:
        """Test tags are passed through on revalidation."""
        async def revalidator(tool: str, params: dict) -> str | None:
            return "new_value"
        
        cache = TaggedMemoryCache(default_ttl=0.01, swr=SWRConfig(stale_multiplier=10.0))
        cache.set("tool", {"q": "test"}, "stale_value", tags=frozenset({"user:1"}))
        swr = SWRCache(cache, revalidator)
        
        time.sleep(0.02)
        
        await swr.get_or_revalidate("tool", {"q": "test"}, tags=frozenset({"user:1"}))
        await asyncio.sleep(0.1)  # Let revalidation complete
        
        # Verify tags were preserved
        assert cache.invalidate_by_tag("user:1") == 1
