"""Comprehensive caching system with LRU eviction and advanced features."""

import hashlib
import json
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live only
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int = 1
    ttl_seconds: Optional[int] = None
    tags: Set[str] = field(default_factory=set)
    priority: int = 0  # Higher priority items are less likely to be evicted
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        age = time.time() - self.created_at
        return age > self.ttl_seconds
    
    def get_age(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at
    
    def get_idle_time(self) -> float:
        """Get time since last access in seconds."""
        return time.time() - self.last_accessed


@dataclass
class CacheStatistics:
    """Cache performance statistics."""
    total_hits: int = 0
    total_misses: int = 0
    total_evictions: int = 0
    total_expirations: int = 0
    total_invalidations: int = 0
    bytes_saved: int = 0
    bytes_evicted: int = 0
    avg_hit_time_ms: float = 0.0
    avg_miss_time_ms: float = 0.0
    cache_efficiency: float = 0.0
    memory_efficiency: float = 0.0
    
    def calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "total_evictions": self.total_evictions,
            "total_expirations": self.total_expirations,
            "total_invalidations": self.total_invalidations,
            "hit_rate": self.calculate_hit_rate(),
            "bytes_saved": self.bytes_saved,
            "bytes_evicted": self.bytes_evicted,
            "avg_hit_time_ms": self.avg_hit_time_ms,
            "avg_miss_time_ms": self.avg_miss_time_ms,
            "cache_efficiency": self.cache_efficiency,
            "memory_efficiency": self.memory_efficiency,
        }


class CacheSystem:
    """Advanced caching system with multiple eviction policies and features."""
    
    def __init__(
        self,
        name: str,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        ttl_seconds: int = 3600,
        policy: CachePolicy = CachePolicy.LRU,
        persistent: bool = False,
        auto_cleanup_interval: int = 300,
    ):
        """
        Initialize cache system.
        
        Args:
            name: Cache name for identification
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            ttl_seconds: Default time-to-live for entries
            policy: Eviction policy to use
            persistent: Whether to persist cache to disk
            auto_cleanup_interval: Interval for automatic cleanup in seconds
        """
        self.name = name
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.policy = policy
        self.persistent = persistent
        self.auto_cleanup_interval = auto_cleanup_interval
        
        # Cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = Lock()
        
        # Statistics
        self.stats = CacheStatistics()
        self.hit_times: List[float] = []
        self.miss_times: List[float] = []
        
        # Persistence
        if self.persistent:
            self.cache_file = settings.cache_dir / f"{name}_cache.pkl"
            self._load_from_disk()
        
        # Start cleanup thread
        self.cleanup_thread = Thread(target=self._auto_cleanup, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"Initialized cache '{name}' with policy {policy.value}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        start_time = time.time()
        
        with self.lock:
            entry = self.cache.get(key)
            
            if entry is None:
                self.stats.total_misses += 1
                elapsed = (time.time() - start_time) * 1000
                self.miss_times.append(elapsed)
                self._update_avg_times()
                return default
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self.stats.total_expirations += 1
                self.stats.total_misses += 1
                elapsed = (time.time() - start_time) * 1000
                self.miss_times.append(elapsed)
                self._update_avg_times()
                return default
            
            # Update access metadata
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            # Move to end for LRU
            if self.policy == CachePolicy.LRU:
                self.cache.move_to_end(key)
            
            self.stats.total_hits += 1
            self.stats.bytes_saved += entry.size_bytes
            
            elapsed = (time.time() - start_time) * 1000
            self.hit_times.append(elapsed)
            self._update_avg_times()
            
            return entry.value
    
    def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[Set[str]] = None,
        priority: int = 0,
    ) -> bool:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override
            tags: Optional tags for grouping
            priority: Priority level (higher = less likely to evict)
            
        Returns:
            True if successfully cached
        """
        try:
            # Estimate size
            size_bytes = self._estimate_size(value)
            
            # Check if value is too large
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"Value too large to cache: {size_bytes} bytes")
                return False
            
            with self.lock:
                # Remove existing entry if present
                if key in self.cache:
                    self._remove_entry(key)
                
                # Evict entries if needed
                while len(self.cache) >= self.max_size:
                    self._evict_entry()
                
                while self._get_total_size() + size_bytes > self.max_memory_bytes:
                    if not self.cache:
                        return False
                    self._evict_entry()
                
                # Create new entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    size_bytes=size_bytes,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    ttl_seconds=ttl or self.ttl_seconds,
                    tags=tags or set(),
                    priority=priority,
                )
                
                self.cache[key] = entry
                
                # Persist if enabled
                if self.persistent:
                    self._save_to_disk()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to cache value: {e}")
            return False
    
    def invalidate(self, key: Optional[str] = None, tags: Optional[Set[str]] = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            key: Specific key to invalidate
            tags: Invalidate all entries with these tags
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        with self.lock:
            if key:
                if key in self.cache:
                    self._remove_entry(key)
                    count = 1
                    self.stats.total_invalidations += 1
            
            if tags:
                keys_to_remove = []
                for k, entry in self.cache.items():
                    if tags.intersection(entry.tags):
                        keys_to_remove.append(k)
                
                for k in keys_to_remove:
                    self._remove_entry(k)
                    count += 1
                    self.stats.total_invalidations += 1
            
            if self.persistent and count > 0:
                self._save_to_disk()
        
        if count > 0:
            logger.info(f"Invalidated {count} cache entries")
        
        return count
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            if self.persistent:
                self._save_to_disk()
        logger.info(f"Cache '{self.name}' cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            current_size = len(self.cache)
            memory_usage = self._get_total_size()
            
            # Calculate efficiency metrics
            if current_size > 0:
                avg_entry_size = memory_usage / current_size
                self.stats.memory_efficiency = current_size / self.max_size
            else:
                avg_entry_size = 0
                self.stats.memory_efficiency = 0
            
            # Calculate cache efficiency (hit rate * memory efficiency)
            self.stats.cache_efficiency = (
                self.stats.calculate_hit_rate() * self.stats.memory_efficiency
            )
            
            return {
                "name": self.name,
                "policy": self.policy.value,
                "current_size": current_size,
                "max_size": self.max_size,
                "memory_usage_mb": memory_usage / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "avg_entry_size_kb": avg_entry_size / 1024,
                "persistent": self.persistent,
                "statistics": self.stats.to_dict(),
            }
    
    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a cache entry."""
        with self.lock:
            entry = self.cache.get(key)
            if entry:
                return {
                    "key": entry.key,
                    "size_bytes": entry.size_bytes,
                    "created_at": datetime.fromtimestamp(entry.created_at).isoformat(),
                    "last_accessed": datetime.fromtimestamp(entry.last_accessed).isoformat(),
                    "access_count": entry.access_count,
                    "age_seconds": entry.get_age(),
                    "idle_seconds": entry.get_idle_time(),
                    "ttl_seconds": entry.ttl_seconds,
                    "expired": entry.is_expired(),
                    "tags": list(entry.tags),
                    "priority": entry.priority,
                }
        return None
    
    def _evict_entry(self) -> None:
        """Evict entry based on configured policy."""
        if not self.cache:
            return
        
        key_to_evict = None
        
        if self.policy == CachePolicy.LRU:
            # Evict least recently used (first item)
            key_to_evict = next(iter(self.cache))
            
        elif self.policy == CachePolicy.LFU:
            # Evict least frequently used
            min_count = float('inf')
            for k, entry in self.cache.items():
                if entry.priority <= 0 and entry.access_count < min_count:
                    min_count = entry.access_count
                    key_to_evict = k
            
        elif self.policy == CachePolicy.FIFO:
            # Evict oldest entry (first item in OrderedDict)
            key_to_evict = next(iter(self.cache))
            
        elif self.policy == CachePolicy.TTL:
            # Evict expired or oldest
            for k, entry in self.cache.items():
                if entry.is_expired():
                    key_to_evict = k
                    break
            if not key_to_evict:
                key_to_evict = next(iter(self.cache))
            
        elif self.policy == CachePolicy.ADAPTIVE:
            # Adaptive eviction based on multiple factors
            min_score = float('inf')
            for k, entry in self.cache.items():
                if entry.priority > 0:
                    continue
                    
                # Calculate eviction score (lower = more likely to evict)
                age_factor = entry.get_age() / 3600  # Age in hours
                idle_factor = entry.get_idle_time() / 3600  # Idle time in hours
                freq_factor = 1 / (entry.access_count + 1)
                size_factor = entry.size_bytes / self.max_memory_bytes
                
                score = (
                    freq_factor * 0.4 +
                    idle_factor * 0.3 +
                    age_factor * 0.2 +
                    size_factor * 0.1
                )
                
                if score < min_score:
                    min_score = score
                    key_to_evict = k
        
        if key_to_evict:
            entry = self.cache[key_to_evict]
            self.stats.bytes_evicted += entry.size_bytes
            self._remove_entry(key_to_evict)
            self.stats.total_evictions += 1
            logger.debug(f"Evicted cache entry: {key_to_evict}")
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self.cache:
            del self.cache[key]
    
    def _get_total_size(self) -> int:
        """Get total size of cached data in bytes."""
        return sum(entry.size_bytes for entry in self.cache.values())
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate size of object in bytes."""
        try:
            # Try pickling for accurate size
            return len(pickle.dumps(obj))
        except:
            # Fallback to JSON for estimation
            try:
                return len(json.dumps(obj, default=str).encode())
            except:
                # Last resort: assume 1KB
                return 1024
    
    def _update_avg_times(self) -> None:
        """Update average hit/miss times."""
        if self.hit_times:
            self.stats.avg_hit_time_ms = sum(self.hit_times[-100:]) / len(self.hit_times[-100:])
        if self.miss_times:
            self.stats.avg_miss_time_ms = sum(self.miss_times[-100:]) / len(self.miss_times[-100:])
    
    def _auto_cleanup(self) -> None:
        """Automatically clean up expired entries."""
        while True:
            time.sleep(self.auto_cleanup_interval)
            
            expired_count = 0
            with self.lock:
                keys_to_remove = []
                for key, entry in self.cache.items():
                    if entry.is_expired():
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    self._remove_entry(key)
                    expired_count += 1
                    self.stats.total_expirations += 1
                
                if self.persistent and expired_count > 0:
                    self._save_to_disk()
            
            if expired_count > 0:
                logger.debug(f"Auto-cleanup removed {expired_count} expired entries from '{self.name}'")
    
    def _save_to_disk(self) -> None:
        """Save cache to disk."""
        if not self.persistent:
            return
        
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self.persistent or not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, 'rb') as f:
                loaded_cache = pickle.load(f)
                
                # Filter out expired entries
                for key, entry in loaded_cache.items():
                    if not entry.is_expired():
                        self.cache[key] = entry
                
            logger.info(f"Loaded {len(self.cache)} entries from disk for '{self.name}'")
        except Exception as e:
            logger.error(f"Failed to load cache from disk: {e}")


def cache_key_generator(*args, **kwargs) -> str:
    """
    Generate a cache key from arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Cache key string
    """
    key_parts = []
    
    # Add positional arguments
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            # Hash complex objects
            key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
    
    # Add keyword arguments
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(f"{k}={v}")
        else:
            # Hash complex objects
            key_parts.append(f"{k}={hashlib.md5(str(v).encode()).hexdigest()[:8]}")
    
    return ":".join(key_parts)