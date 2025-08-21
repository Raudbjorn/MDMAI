"""Cache management with LRU eviction policy."""

from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
import time
import sys
from threading import Lock

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class LRUCache:
    """Thread-safe LRU cache with size limits and TTL support."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100, ttl_seconds: int = 3600):
        """
        Initialize LRU cache with size and memory limits.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = Lock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            # Check if key exists
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check TTL
            if self._is_expired(key):
                self._remove(key)
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            
            return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Remove if already exists to update position
            if key in self.cache:
                self._remove(key)
            
            # Check size limit
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Check memory limit
            while self._estimate_memory_usage() >= self.max_memory_bytes:
                if not self.cache:
                    logger.warning("Cannot cache item - exceeds memory limit")
                    return
                self._evict_lru()
            
            # Add to cache
            self.cache[key] = value
            self.timestamps[key] = time.time()
            
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            logger.info(f"Cache cleared. Stats: hits={self.hits}, misses={self.misses}, evictions={self.evictions}")
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def memory_usage_mb(self) -> float:
        """Get estimated memory usage in MB."""
        return self._estimate_memory_usage() / (1024 * 1024)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "memory_usage_mb": self.memory_usage_mb(),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
                "ttl_seconds": self.ttl_seconds,
            }
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.timestamps:
            return True
        
        age = time.time() - self.timestamps[key]
        return age > self.ttl_seconds
    
    def _remove(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self.cache:
            # Pop first item (least recently used)
            key, _ = self.cache.popitem(last=False)
            if key in self.timestamps:
                del self.timestamps[key]
            self.evictions += 1
            logger.debug(f"Evicted cache entry: {key}")
    
    def _estimate_memory_usage(self) -> int:
        """
        Estimate memory usage of cache in bytes.
        
        Returns:
            Estimated memory usage
        """
        # Simple estimation - can be improved with more accurate sizing
        total_size = 0
        
        for key, value in self.cache.items():
            # Estimate key size
            total_size += sys.getsizeof(key)
            
            # Estimate value size
            if isinstance(value, dict):
                total_size += self._estimate_dict_size(value)
            elif isinstance(value, list):
                total_size += self._estimate_list_size(value)
            else:
                total_size += sys.getsizeof(value)
        
        # Add timestamp overhead
        total_size += len(self.timestamps) * sys.getsizeof(0.0)
        
        return total_size
    
    def _estimate_dict_size(self, d: Dict) -> int:
        """Estimate size of dictionary."""
        size = sys.getsizeof(d)
        for k, v in d.items():
            size += sys.getsizeof(k)
            if isinstance(v, dict):
                size += self._estimate_dict_size(v)
            elif isinstance(v, list):
                size += self._estimate_list_size(v)
            else:
                size += sys.getsizeof(v)
        return size
    
    def _estimate_list_size(self, lst: List) -> int:
        """Estimate size of list."""
        size = sys.getsizeof(lst)
        for item in lst:
            if isinstance(item, dict):
                size += self._estimate_dict_size(item)
            elif isinstance(item, list):
                size += self._estimate_list_size(item)
            else:
                size += sys.getsizeof(item)
        return size


class SearchCacheManager:
    """Manages multiple caches for different search types."""
    
    def __init__(self):
        """Initialize cache manager with separate caches."""
        # Allocate memory proportionally from the global limit
        total_memory = settings.cache_max_memory_mb
        
        # Allocate memory as percentages of total
        # Query cache: 30%, Embedding cache: 50%, Cross-ref cache: 20%
        query_memory = int(total_memory * 0.3)
        embedding_memory = int(total_memory * 0.5)
        cross_ref_memory = int(total_memory * 0.2)
        
        # Different caches for different search types
        self.query_cache = LRUCache(
            max_size=settings.search_cache_size,
            max_memory_mb=query_memory,
            ttl_seconds=settings.cache_ttl_seconds
        )
        
        self.embedding_cache = LRUCache(
            max_size=500,
            max_memory_mb=embedding_memory,
            ttl_seconds=3600 * 24  # 24 hours for embeddings
        )
        
        self.cross_ref_cache = LRUCache(
            max_size=200,
            max_memory_mb=cross_ref_memory,
            ttl_seconds=1800  # 30 minutes for cross-references
        )
    
    def get_query_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached query result."""
        return self.query_cache.get(cache_key)
    
    def cache_query_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache query result."""
        self.query_cache.put(cache_key, result)
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding."""
        return self.embedding_cache.get(text)
    
    def cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache embedding."""
        self.embedding_cache.put(text, embedding)
    
    def get_cross_reference(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached cross-reference."""
        return self.cross_ref_cache.get(key)
    
    def cache_cross_reference(self, key: str, refs: Dict[str, Any]) -> None:
        """Cache cross-reference."""
        self.cross_ref_cache.put(key, refs)
    
    def clear_all(self) -> None:
        """Clear all caches."""
        self.query_cache.clear()
        self.embedding_cache.clear()
        self.cross_ref_cache.clear()
        logger.info("All caches cleared")
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all caches.
        
        Returns:
            Dictionary with stats for each cache
        """
        return {
            "query_cache": self.query_cache.get_stats(),
            "embedding_cache": self.embedding_cache.get_stats(),
            "cross_ref_cache": self.cross_ref_cache.get_stats(),
            "total_memory_mb": (
                self.query_cache.memory_usage_mb() +
                self.embedding_cache.memory_usage_mb() +
                self.cross_ref_cache.memory_usage_mb()
            ),
        }
    
    def cleanup_expired(self) -> None:
        """Remove expired entries from all caches."""
        # This is handled automatically on access, but we can force cleanup
        for cache in [self.query_cache, self.embedding_cache, self.cross_ref_cache]:
            expired_keys = []
            with cache.lock:
                for key in list(cache.cache.keys()):
                    if cache._is_expired(key):
                        expired_keys.append(key)
                
                for key in expired_keys:
                    cache._remove(key)
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")