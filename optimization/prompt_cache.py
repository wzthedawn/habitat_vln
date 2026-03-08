"""Prompt cache for reusing computed prompts."""

from typing import Dict, Any, Optional, List
from collections import OrderedDict
import hashlib
import logging
import time


class PromptCache:
    """
    Prompt Cache for storing and reusing computed prompts.

    Implements LRU caching with TTL support for prompt caching.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize prompt cache.

        Args:
            config: Cache configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger("PromptCache")

        # Cache settings
        self.max_size = self.config.get("max_size", 1000)
        self.ttl_seconds = self.config.get("ttl_seconds", 3600)  # 1 hour default

        # Cache storage (OrderedDict for LRU)
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

    def get(self, key: str) -> Optional[str]:
        """
        Get cached prompt.

        Args:
            key: Cache key

        Returns:
            Cached prompt or None
        """
        if key not in self._cache:
            self._stats["misses"] += 1
            return None

        entry = self._cache[key]

        # Check TTL
        if self._is_expired(entry):
            self._evict(key)
            self._stats["misses"] += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._stats["hits"] += 1

        return entry["prompt"]

    def set(self, key: str, prompt: str, metadata: Dict[str, Any] = None) -> None:
        """
        Store prompt in cache.

        Args:
            key: Cache key
            prompt: Prompt to cache
            metadata: Optional metadata
        """
        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            self._evict_lru()

        # Store entry
        self._cache[key] = {
            "prompt": prompt,
            "metadata": metadata or {},
            "created_at": time.time(),
            "access_count": 0,
        }

        # Move to end
        self._cache.move_to_end(key)

    def get_or_compute(
        self,
        key: str,
        compute_fn: callable,
    ) -> str:
        """
        Get from cache or compute and cache.

        Args:
            key: Cache key
            compute_fn: Function to compute prompt if not cached

        Returns:
            Prompt string
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        # Compute
        prompt = compute_fn()
        self.set(key, prompt)
        return prompt

    def contains(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        return key in self._cache and not self._is_expired(self._cache[key])

    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._stats["evictions"] = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Statistics dictionary
        """
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "hit_rate": hit_rate,
        }

    def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self._cache.items()
            if self._is_expired(entry)
        ]

        for key in expired_keys:
            self._evict(key)

        return len(expired_keys)

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds <= 0:
            return False

        age = time.time() - entry["created_at"]
        return age > self.ttl_seconds

    def _evict(self, key: str) -> None:
        """Evict an entry."""
        if key in self._cache:
            del self._cache[key]
            self._stats["evictions"] += 1

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            self._evict(oldest_key)


class PromptKeyBuilder:
    """Builder for cache keys."""

    @staticmethod
    def build_key(
        template_name: str,
        instruction: str,
        context_hash: str = None,
        **kwargs,
    ) -> str:
        """
        Build cache key.

        Args:
            template_name: Template name
            instruction: Navigation instruction
            context_hash: Optional context hash
            **kwargs: Additional key components

        Returns:
            Cache key string
        """
        # Build base key
        components = [template_name, instruction[:100]]

        if context_hash:
            components.append(context_hash)

        # Add kwargs
        for key, value in sorted(kwargs.items()):
            if value is not None:
                components.append(f"{key}={str(value)[:50]}")

        # Hash if too long
        key_str = "|".join(components)
        if len(key_str) > 200:
            return hashlib.md5(key_str.encode()).hexdigest()

        return key_str

    @staticmethod
    def hash_context(
        position: tuple,
        step: int,
        room_type: str,
    ) -> str:
        """
        Create hash from context elements.

        Args:
            position: Agent position
            step: Current step
            room_type: Current room

        Returns:
            Context hash
        """
        context_str = f"{position}_{step}_{room_type}"
        return hashlib.md5(context_str.encode()).hexdigest()[:8]


class MultiLevelCache:
    """
    Multi-level cache with different TTLs.

    Provides different caching strategies for different
    types of prompts.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize multi-level cache."""
        self.config = config or {}

        # Create different cache levels
        self._caches = {
            "short": PromptCache({
                "max_size": 100,
                "ttl_seconds": 300,  # 5 minutes
            }),
            "medium": PromptCache({
                "max_size": 500,
                "ttl_seconds": 1800,  # 30 minutes
            }),
            "long": PromptCache({
                "max_size": 200,
                "ttl_seconds": 3600,  # 1 hour
            }),
        }

    def get(self, key: str, level: str = "medium") -> Optional[str]:
        """Get from specific cache level."""
        if level in self._caches:
            return self._caches[level].get(key)
        return None

    def set(
        self,
        key: str,
        prompt: str,
        level: str = "medium",
        metadata: Dict[str, Any] = None,
    ) -> None:
        """Store in specific cache level."""
        if level in self._caches:
            self._caches[level].set(key, prompt, metadata)

    def get_from_any(self, key: str) -> Optional[str]:
        """Get from any cache level."""
        for level in ["short", "medium", "long"]:
            result = self._caches[level].get(key)
            if result is not None:
                return result
        return None

    def clear_all(self) -> None:
        """Clear all cache levels."""
        for cache in self._caches.values():
            cache.clear()

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all levels."""
        return {
            level: cache.get_stats()
            for level, cache in self._caches.items()
        }