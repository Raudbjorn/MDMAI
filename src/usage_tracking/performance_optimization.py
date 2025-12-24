"""Advanced performance optimization with multi-tier caching and intelligent indexing."""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import hashlib
import json
from collections import defaultdict, OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import weakref

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..ai_providers.models import UsageRecord, ProviderType
from .chroma_extensions import UsageTrackingChromaExtensions
from .json_persistence import JsonPersistenceManager
from config.logging_config import get_logger

logger = get_logger(__name__)


def _safe_json_encoder(obj: Any) -> Any:
    """Custom JSON encoder for cache serialization."""
    if hasattr(obj, 'isoformat'):  # datetime
        return {'__datetime__': obj.isoformat()}
    if hasattr(obj, '__dataclass_fields__'):  # dataclass
        return {'__dataclass__': obj.__class__.__name__, 'data': asdict(obj)}
    if isinstance(obj, Enum):
        return {'__enum__': obj.__class__.__name__, 'value': obj.value}
    if isinstance(obj, set):
        return {'__set__': list(obj)}
    return str(obj)


def _safe_serialize(value: Any) -> str:
    """Safely serialize a value to JSON string (replaces pickle.dumps)."""
    return json.dumps(value, default=_safe_json_encoder)


def _safe_deserialize(data: str) -> Any:
    """Safely deserialize a JSON string (replaces pickle.loads)."""
    return json.loads(data)


class CacheLevel(Enum):
    """Cache levels in the multi-tier caching system."""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_PERSISTENT = "l3_persistent"


class CacheStrategy(Enum):
    """Cache replacement strategies."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class IndexType(Enum):
    """Index types for query optimization."""
    HASH = "hash"
    BTREE = "btree"
    BITMAP = "bitmap"
    BLOOM_FILTER = "bloom_filter"
    COMPOSITE = "composite"
    SPATIAL = "spatial"


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int]
    size_bytes: int
    tags: Set[str]
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    def touch(self) -> None:
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class IndexDefinition:
    """Defines an index for query optimization."""
    index_name: str
    fields: List[str]
    index_type: IndexType
    unique: bool = False
    sparse: bool = False
    partial_filter: Optional[Dict[str, Any]] = None
    ttl_seconds: Optional[int] = None
    background: bool = True


@dataclass
class QueryPattern:
    """Represents a query pattern for optimization."""
    pattern_id: str
    fields_accessed: List[str]
    filter_conditions: Dict[str, Any]
    sort_fields: List[str]
    frequency: int
    avg_response_time_ms: float
    last_seen: datetime
    optimization_score: float


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    avg_query_time_ms: float = 0.0
    index_usage_stats: Dict[str, int] = None
    memory_usage_mb: float = 0.0
    
    def __post_init__(self):
        if self.index_usage_stats is None:
            self.index_usage_stats = {}
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class MultiTierCache:
    """Multi-tier caching system with L1 memory, L2 Redis, L3 persistent storage."""
    
    def __init__(
        self,
        l1_max_size: int = 1000,
        l1_ttl_seconds: int = 300,
        l2_redis_url: Optional[str] = None,
        l2_ttl_seconds: int = 3600,
        l3_persistent_path: Optional[str] = None,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    ):
        # L1 Memory Cache
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l1_max_size = l1_max_size
        self.l1_ttl_seconds = l1_ttl_seconds
        self.l1_lock = threading.RLock()
        
        # L2 Redis Cache
        self.redis_client = None
        self.l2_ttl_seconds = l2_ttl_seconds
        if l2_redis_url and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(l2_redis_url, decode_responses=False)
                self.redis_client.ping()
                logger.info("Redis L2 cache initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis L2 cache: {e}")
                self.redis_client = None
        
        # L3 Persistent Cache
        self.l3_persistent_path = l3_persistent_path
        
        # Cache strategy
        self.strategy = strategy
        
        # Metrics
        self.metrics = PerformanceMetrics()
        self.metrics_lock = threading.Lock()
        
        # Query pattern tracking
        self.query_patterns: Dict[str, QueryPattern] = {}
        
        # Background cleanup
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start(self) -> None:
        """Start the cache system."""
        self.running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_worker())
        logger.info("Multi-tier cache started")
    
    async def stop(self) -> None:
        """Stop the cache system."""
        self.running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Multi-tier cache stopped")
    
    def _generate_cache_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generate a consistent cache key."""
        # Sort parameters for consistent key generation
        sorted_params = json.dumps(params, sort_keys=True, default=str)
        key_hash = hashlib.md5(sorted_params.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def _calculate_entry_size(self, value: Any) -> int:
        """Calculate the approximate size of a cache entry."""
        try:
            return len(_safe_serialize(value).encode())
        except Exception:
            return len(str(value).encode())
    
    async def get(self, key: str, tags: Optional[Set[str]] = None) -> Optional[Any]:
        """Get value from cache, checking all tiers."""
        start_time = time.time()
        
        # Try L1 cache first
        with self.l1_lock:
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                if not entry.is_expired():
                    entry.touch()
                    # Move to end for LRU
                    self.l1_cache.move_to_end(key)
                    
                    with self.metrics_lock:
                        self.metrics.cache_hits += 1
                    
                    self._track_query_time(time.time() - start_time)
                    return entry.value
                else:
                    # Remove expired entry
                    del self.l1_cache[key]
        
        # Try L2 Redis cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    value = _safe_deserialize(cached_data.decode() if isinstance(cached_data, bytes) else cached_data)
                    
                    # Promote to L1 cache
                    await self._set_l1(key, value, tags or set())
                    
                    with self.metrics_lock:
                        self.metrics.cache_hits += 1
                    
                    self._track_query_time(time.time() - start_time)
                    return value
            except Exception as e:
                logger.debug(f"Redis L2 cache error: {e}")
        
        # Try L3 persistent cache
        if self.l3_persistent_path:
            # Implementation for persistent cache would go here
            pass
        
        # Cache miss
        with self.metrics_lock:
            self.metrics.cache_misses += 1
        
        self._track_query_time(time.time() - start_time)
        return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None,
        tags: Optional[Set[str]] = None
    ) -> None:
        """Set value in cache across all appropriate tiers."""
        tags = tags or set()
        
        # Set in L1 cache
        await self._set_l1(key, value, tags, ttl_seconds)
        
        # Set in L2 Redis cache
        if self.redis_client:
            try:
                cached_data = _safe_serialize(value)
                ttl = ttl_seconds or self.l2_ttl_seconds
                self.redis_client.setex(key, ttl, cached_data)
                
                # Store tags separately for invalidation
                if tags:
                    for tag in tags:
                        self.redis_client.sadd(f"tag:{tag}", key)
                        self.redis_client.expire(f"tag:{tag}", ttl)
                        
            except Exception as e:
                logger.debug(f"Redis L2 cache set error: {e}")
        
        # Set in L3 persistent cache
        if self.l3_persistent_path:
            # Implementation for persistent cache would go here
            pass
    
    async def _set_l1(
        self, 
        key: str, 
        value: Any, 
        tags: Set[str], 
        ttl_seconds: Optional[int] = None
    ) -> None:
        """Set value in L1 memory cache."""
        with self.l1_lock:
            # Check if we need to evict entries
            while len(self.l1_cache) >= self.l1_max_size:
                self._evict_l1_entry()
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl_seconds=ttl_seconds or self.l1_ttl_seconds,
                size_bytes=self._calculate_entry_size(value),
                tags=tags
            )
            
            self.l1_cache[key] = entry
    
    def _evict_l1_entry(self) -> None:
        """Evict an entry from L1 cache based on strategy."""
        if not self.l1_cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used (first item in OrderedDict)
            key, _ = self.l1_cache.popitem(last=False)
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            min_access = min(entry.access_count for entry in self.l1_cache.values())
            for key, entry in self.l1_cache.items():
                if entry.access_count == min_access:
                    del self.l1_cache[key]
                    break
        elif self.strategy == CacheStrategy.FIFO:
            # Remove first in, first out
            key, _ = self.l1_cache.popitem(last=False)
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired entries first
            now = datetime.now()
            for key, entry in list(self.l1_cache.items()):
                if entry.is_expired():
                    del self.l1_cache[key]
                    break
            else:
                # No expired entries, fall back to LRU
                key, _ = self.l1_cache.popitem(last=False)
        else:  # ADAPTIVE
            # Adaptive strategy based on access patterns
            self._adaptive_eviction()
        
        with self.metrics_lock:
            self.metrics.cache_evictions += 1
    
    def _adaptive_eviction(self) -> None:
        """Adaptive eviction strategy based on access patterns."""
        if not self.l1_cache:
            return
        
        # Score entries based on multiple factors
        scores = {}
        now = datetime.now()
        
        for key, entry in self.l1_cache.items():
            age_score = (now - entry.created_at).total_seconds() / 3600  # Age in hours
            access_score = entry.access_count / max(1, (now - entry.created_at).total_seconds() / 3600)
            recency_score = (now - entry.last_accessed).total_seconds() / 3600
            
            # Lower score = more likely to evict
            scores[key] = access_score / (age_score + recency_score + 1)
        
        # Remove entry with lowest score
        worst_key = min(scores.keys(), key=lambda k: scores[k])
        del self.l1_cache[worst_key]
    
    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate cache entries by tags."""
        invalidated_count = 0
        
        # Invalidate L1 cache
        with self.l1_lock:
            keys_to_remove = []
            for key, entry in self.l1_cache.items():
                if entry.tags.intersection(tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.l1_cache[key]
                invalidated_count += 1
        
        # Invalidate L2 Redis cache
        if self.redis_client:
            try:
                for tag in tags:
                    tag_key = f"tag:{tag}"
                    keys = self.redis_client.smembers(tag_key)
                    if keys:
                        self.redis_client.delete(*keys)
                        self.redis_client.delete(tag_key)
                        invalidated_count += len(keys)
            except Exception as e:
                logger.debug(f"Redis tag invalidation error: {e}")
        
        logger.debug(f"Invalidated {invalidated_count} cache entries by tags: {tags}")
        return invalidated_count
    
    async def _cleanup_worker(self) -> None:
        """Background worker to clean up expired entries."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Clean L1 cache
                with self.l1_lock:
                    expired_keys = [
                        key for key, entry in self.l1_cache.items() 
                        if entry.is_expired()
                    ]
                    for key in expired_keys:
                        del self.l1_cache[key]
                
                # Update memory usage metric
                with self.metrics_lock:
                    total_size = sum(entry.size_bytes for entry in self.l1_cache.values())
                    self.metrics.memory_usage_mb = total_size / (1024 * 1024)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)
    
    def _track_query_time(self, time_seconds: float) -> None:
        """Track query response time."""
        time_ms = time_seconds * 1000
        with self.metrics_lock:
            # Exponential moving average
            alpha = 0.1
            self.metrics.avg_query_time_ms = (
                alpha * time_ms + (1 - alpha) * self.metrics.avg_query_time_ms
            )
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self.metrics_lock:
            return PerformanceMetrics(
                cache_hits=self.metrics.cache_hits,
                cache_misses=self.metrics.cache_misses,
                cache_evictions=self.metrics.cache_evictions,
                avg_query_time_ms=self.metrics.avg_query_time_ms,
                index_usage_stats=dict(self.metrics.index_usage_stats),
                memory_usage_mb=self.metrics.memory_usage_mb
            )


class IntelligentIndexManager:
    """Manages intelligent indexing for query optimization."""
    
    def __init__(self):
        self.indices: Dict[str, IndexDefinition] = {}
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.index_usage_stats: Dict[str, int] = defaultdict(int)
        
        # Bloom filters for existence checks
        self.bloom_filters: Dict[str, Set] = {}  # Simplified bloom filter
        
        # Background optimization
        self.optimization_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.metrics_lock = threading.Lock()
    
    async def start(self) -> None:
        """Start the index manager."""
        self.running = True
        self.optimization_task = asyncio.create_task(self._optimization_worker())
        logger.info("Intelligent index manager started")
    
    async def stop(self) -> None:
        """Stop the index manager."""
        self.running = False
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        logger.info("Intelligent index manager stopped")
    
    def add_index(self, index_def: IndexDefinition) -> None:
        """Add an index definition."""
        self.indices[index_def.index_name] = index_def
        
        # Initialize bloom filter if needed
        if index_def.index_type == IndexType.BLOOM_FILTER:
            self.bloom_filters[index_def.index_name] = set()
        
        logger.info(f"Index added: {index_def.index_name}")
    
    def track_query_pattern(
        self, 
        fields_accessed: List[str],
        filter_conditions: Dict[str, Any],
        sort_fields: List[str],
        response_time_ms: float
    ) -> None:
        """Track a query pattern for optimization analysis."""
        # Generate pattern ID
        pattern_data = {
            "fields": sorted(fields_accessed),
            "filters": sorted(filter_conditions.keys()),
            "sorts": sorted(sort_fields)
        }
        pattern_id = hashlib.md5(json.dumps(pattern_data, sort_keys=True).encode()).hexdigest()
        
        if pattern_id in self.query_patterns:
            pattern = self.query_patterns[pattern_id]
            pattern.frequency += 1
            pattern.avg_response_time_ms = (
                pattern.avg_response_time_ms * 0.9 + response_time_ms * 0.1
            )
            pattern.last_seen = datetime.now()
        else:
            pattern = QueryPattern(
                pattern_id=pattern_id,
                fields_accessed=fields_accessed,
                filter_conditions=filter_conditions,
                sort_fields=sort_fields,
                frequency=1,
                avg_response_time_ms=response_time_ms,
                last_seen=datetime.now(),
                optimization_score=0.0
            )
            self.query_patterns[pattern_id] = pattern
        
        # Calculate optimization score
        pattern.optimization_score = self._calculate_optimization_score(pattern)
    
    def _calculate_optimization_score(self, pattern: QueryPattern) -> float:
        """Calculate optimization score for a query pattern."""
        # Higher score = higher priority for optimization
        frequency_score = min(pattern.frequency / 100, 1.0) * 40  # Max 40 points
        latency_score = min(pattern.avg_response_time_ms / 1000, 1.0) * 40  # Max 40 points
        recency_score = max(0, 1.0 - (datetime.now() - pattern.last_seen).days / 7) * 20  # Max 20 points
        
        return frequency_score + latency_score + recency_score
    
    def suggest_indices(self, min_score: float = 50.0) -> List[IndexDefinition]:
        """Suggest new indices based on query patterns."""
        suggestions = []
        
        # Analyze high-scoring query patterns
        high_value_patterns = [
            p for p in self.query_patterns.values()
            if p.optimization_score >= min_score
        ]
        
        for pattern in high_value_patterns:
            # Suggest composite index for frequently accessed fields
            if len(pattern.fields_accessed) > 1:
                suggested_index = IndexDefinition(
                    index_name=f"composite_{pattern.pattern_id[:8]}",
                    fields=pattern.fields_accessed,
                    index_type=IndexType.COMPOSITE,
                    background=True
                )
                suggestions.append(suggested_index)
            
            # Suggest individual indices for filter fields
            for field in pattern.filter_conditions.keys():
                if field not in [idx.fields[0] for idx in self.indices.values() if len(idx.fields) == 1]:
                    suggested_index = IndexDefinition(
                        index_name=f"idx_{field}",
                        fields=[field],
                        index_type=IndexType.BTREE,
                        background=True
                    )
                    suggestions.append(suggested_index)
        
        return suggestions
    
    def get_optimal_indices_for_query(
        self, 
        fields: List[str], 
        filters: Dict[str, Any],
        sorts: List[str]
    ) -> List[str]:
        """Get optimal indices for a specific query."""
        relevant_indices = []
        
        for index_name, index_def in self.indices.items():
            # Check if index covers query fields
            if any(field in index_def.fields for field in fields + list(filters.keys()) + sorts):
                relevance_score = self._calculate_index_relevance(
                    index_def, fields, filters, sorts
                )
                relevant_indices.append((index_name, relevance_score))
        
        # Sort by relevance and return index names
        relevant_indices.sort(key=lambda x: x[1], reverse=True)
        return [idx[0] for idx in relevant_indices[:3]]  # Top 3 indices
    
    def _calculate_index_relevance(
        self,
        index_def: IndexDefinition,
        query_fields: List[str],
        query_filters: Dict[str, Any],
        query_sorts: List[str]
    ) -> float:
        """Calculate relevance score of an index for a query."""
        score = 0.0
        
        # Score for field coverage
        field_coverage = len(set(index_def.fields) & set(query_fields + list(query_filters.keys())))
        score += field_coverage * 10
        
        # Score for filter field coverage
        filter_coverage = len(set(index_def.fields) & set(query_filters.keys()))
        score += filter_coverage * 15
        
        # Score for sort field coverage
        sort_coverage = len(set(index_def.fields) & set(query_sorts))
        score += sort_coverage * 20
        
        # Bonus for exact field match
        if set(index_def.fields) == set(query_fields):
            score += 25
        
        return score
    
    def record_index_usage(self, index_name: str) -> None:
        """Record usage of an index."""
        self.index_usage_stats[index_name] += 1
        
        with self.metrics_lock:
            self.metrics.index_usage_stats[index_name] = self.index_usage_stats[index_name]
    
    async def _optimization_worker(self) -> None:
        """Background worker for index optimization."""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Analyze query patterns and suggest optimizations
                suggestions = self.suggest_indices()
                if suggestions:
                    logger.info(f"Index optimization suggestions: {len(suggestions)} indices")
                    # In practice, these suggestions would be evaluated and potentially created
                
                # Clean up old query patterns
                cutoff_date = datetime.now() - timedelta(days=30)
                old_patterns = [
                    pattern_id for pattern_id, pattern in self.query_patterns.items()
                    if pattern.last_seen < cutoff_date
                ]
                
                for pattern_id in old_patterns:
                    del self.query_patterns[pattern_id]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Index optimization error: {e}")
                await asyncio.sleep(3600)


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(
        self,
        chroma_extensions: UsageTrackingChromaExtensions,
        json_persistence: JsonPersistenceManager,
        cache_config: Optional[Dict[str, Any]] = None
    ):
        self.chroma_extensions = chroma_extensions
        self.json_persistence = json_persistence
        
        # Initialize cache system
        cache_config = cache_config or {}
        self.cache = MultiTierCache(
            l1_max_size=cache_config.get("l1_max_size", 1000),
            l1_ttl_seconds=cache_config.get("l1_ttl_seconds", 300),
            l2_redis_url=cache_config.get("redis_url"),
            l2_ttl_seconds=cache_config.get("l2_ttl_seconds", 3600),
            strategy=CacheStrategy(cache_config.get("strategy", "adaptive"))
        )
        
        # Initialize index manager
        self.index_manager = IntelligentIndexManager()
        
        # Query optimization
        self.query_optimizer = QueryOptimizer()
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Background tasks
        self.optimization_tasks: List[asyncio.Task] = []
        self.running = False
    
    async def start(self) -> None:
        """Start the performance optimizer."""
        if self.running:
            return
        
        self.running = True
        
        # Start components
        await self.cache.start()
        await self.index_manager.start()
        
        # Start background optimization tasks
        self.optimization_tasks = [
            asyncio.create_task(self._adaptive_optimization_worker()),
            asyncio.create_task(self._performance_monitoring_worker())
        ]
        
        logger.info("Performance optimizer started")
    
    async def stop(self) -> None:
        """Stop the performance optimizer."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop components
        await self.cache.stop()
        await self.index_manager.stop()
        
        # Cancel background tasks
        for task in self.optimization_tasks:
            task.cancel()
        
        if self.optimization_tasks:
            await asyncio.gather(*self.optimization_tasks, return_exceptions=True)
        
        logger.info("Performance optimizer stopped")
    
    async def optimize_query(
        self, 
        query_func: Callable,
        cache_key_prefix: str,
        query_params: Dict[str, Any],
        cache_ttl: Optional[int] = None,
        cache_tags: Optional[Set[str]] = None
    ) -> Any:
        """Optimize a query with caching and performance tracking."""
        start_time = time.time()
        
        # Generate cache key
        cache_key = self.cache._generate_cache_key(cache_key_prefix, query_params)
        
        # Try to get from cache first
        cached_result = await self.cache.get(cache_key, cache_tags)
        if cached_result is not None:
            execution_time = (time.time() - start_time) * 1000
            self.performance_monitor.record_cache_hit(execution_time)
            return cached_result
        
        # Cache miss - execute query
        try:
            result = await query_func(**query_params)
            
            # Cache the result
            await self.cache.set(cache_key, result, cache_ttl, cache_tags)
            
            # Track performance
            execution_time = (time.time() - start_time) * 1000
            self.performance_monitor.record_query_execution(execution_time)
            
            # Track query pattern for optimization
            self.index_manager.track_query_pattern(
                fields_accessed=query_params.get("fields", []),
                filter_conditions=query_params.get("filters", {}),
                sort_fields=query_params.get("sort", []),
                response_time_ms=execution_time
            )
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.performance_monitor.record_query_error(execution_time)
            raise
    
    async def invalidate_cache_by_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern."""
        # Convert pattern to tags for invalidation
        tags = {pattern}
        return await self.cache.invalidate_by_tags(tags)
    
    async def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get comprehensive optimization recommendations."""
        cache_metrics = self.cache.get_metrics()
        performance_stats = self.performance_monitor.get_stats()
        index_suggestions = self.index_manager.suggest_indices()
        
        recommendations = {
            "cache_optimization": [],
            "index_optimization": [],
            "query_optimization": [],
            "overall_health": "good"
        }
        
        # Cache optimization recommendations
        if cache_metrics.cache_hit_rate < 0.8:
            recommendations["cache_optimization"].append({
                "type": "increase_cache_size",
                "current_hit_rate": cache_metrics.cache_hit_rate,
                "recommendation": "Consider increasing cache size or TTL"
            })
        
        if cache_metrics.cache_evictions > cache_metrics.cache_hits * 0.1:
            recommendations["cache_optimization"].append({
                "type": "high_eviction_rate",
                "eviction_rate": cache_metrics.cache_evictions / max(cache_metrics.cache_hits, 1),
                "recommendation": "Cache size may be too small for workload"
            })
        
        # Index optimization recommendations
        if index_suggestions:
            recommendations["index_optimization"] = [
                {
                    "type": "suggested_index",
                    "index_name": idx.index_name,
                    "fields": idx.fields,
                    "index_type": idx.index_type.value,
                    "reasoning": "Based on query pattern analysis"
                }
                for idx in index_suggestions[:5]  # Top 5 suggestions
            ]
        
        # Query optimization recommendations
        slow_patterns = [
            pattern for pattern in self.index_manager.query_patterns.values()
            if pattern.avg_response_time_ms > 1000  # Slower than 1 second
        ]
        
        if slow_patterns:
            recommendations["query_optimization"] = [
                {
                    "type": "slow_query_pattern",
                    "pattern_id": pattern.pattern_id,
                    "avg_time_ms": pattern.avg_response_time_ms,
                    "frequency": pattern.frequency,
                    "recommendation": "Consider optimizing query or adding indices"
                }
                for pattern in slow_patterns[:3]  # Top 3 slow patterns
            ]
        
        # Overall health assessment
        if (cache_metrics.cache_hit_rate < 0.5 or 
            cache_metrics.avg_query_time_ms > 2000 or
            len(slow_patterns) > 10):
            recommendations["overall_health"] = "needs_attention"
        elif (cache_metrics.cache_hit_rate < 0.7 or
              cache_metrics.avg_query_time_ms > 1000 or
              len(slow_patterns) > 5):
            recommendations["overall_health"] = "fair"
        
        return recommendations
    
    async def _adaptive_optimization_worker(self) -> None:
        """Background worker for adaptive optimization."""
        while self.running:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                # Get recommendations and apply automatic optimizations
                recommendations = await self.get_optimization_recommendations()
                
                # Auto-apply safe optimizations
                for cache_rec in recommendations.get("cache_optimization", []):
                    if cache_rec["type"] == "increase_cache_size":
                        # Could automatically increase cache size within limits
                        pass
                
                # Log recommendations for manual review
                if any(recommendations.values()):
                    logger.info("Performance optimization recommendations available",
                              recommendations=recommendations)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Adaptive optimization error: {e}")
                await asyncio.sleep(1800)
    
    async def _performance_monitoring_worker(self) -> None:
        """Background worker for performance monitoring."""
        while self.running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Collect performance metrics
                cache_metrics = self.cache.get_metrics()
                
                # Log performance alerts
                if cache_metrics.cache_hit_rate < 0.5:
                    logger.warning("Low cache hit rate detected", 
                                 hit_rate=cache_metrics.cache_hit_rate)
                
                if cache_metrics.avg_query_time_ms > 2000:
                    logger.warning("High average query time detected",
                                 avg_time_ms=cache_metrics.avg_query_time_ms)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        cache_metrics = self.cache.get_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cache_metrics": {
                "hit_rate": cache_metrics.cache_hit_rate,
                "total_hits": cache_metrics.cache_hits,
                "total_misses": cache_metrics.cache_misses,
                "evictions": cache_metrics.cache_evictions,
                "memory_usage_mb": cache_metrics.memory_usage_mb
            },
            "query_performance": {
                "avg_query_time_ms": cache_metrics.avg_query_time_ms,
                "query_patterns_tracked": len(self.index_manager.query_patterns),
                "indices_available": len(self.index_manager.indices)
            },
            "optimization_status": await self.get_optimization_recommendations()
        }


class QueryOptimizer:
    """Query optimization utilities."""
    
    def optimize_usage_query(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize usage record query parameters."""
        optimized = query_params.copy()
        
        # Optimize date range queries
        if "date_range" in optimized:
            start_date, end_date = optimized["date_range"]
            # Ensure reasonable date ranges
            max_range = timedelta(days=90)
            if end_date - start_date > max_range:
                optimized["date_range"] = (end_date - max_range, end_date)
        
        # Optimize limit values
        if "limit" in optimized:
            # Cap limit to prevent memory issues
            optimized["limit"] = min(optimized["limit"], 10000)
        
        # Add intelligent defaults
        if "limit" not in optimized:
            optimized["limit"] = 100
        
        return optimized


class PerformanceMonitor:
    """Performance monitoring and alerting."""
    
    def __init__(self):
        self.query_times: deque = deque(maxlen=1000)
        self.cache_hits = 0
        self.cache_misses = 0
        self.query_errors = 0
        
    def record_query_execution(self, time_ms: float) -> None:
        """Record query execution time."""
        self.query_times.append(time_ms)
    
    def record_cache_hit(self, time_ms: float) -> None:
        """Record cache hit."""
        self.cache_hits += 1
        self.query_times.append(time_ms)
    
    def record_query_error(self, time_ms: float) -> None:
        """Record query error."""
        self.query_errors += 1
        self.query_times.append(time_ms)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.query_times:
            return {"avg_time_ms": 0, "queries_tracked": 0}
        
        return {
            "avg_time_ms": sum(self.query_times) / len(self.query_times),
            "min_time_ms": min(self.query_times),
            "max_time_ms": max(self.query_times),
            "p95_time_ms": sorted(self.query_times)[int(len(self.query_times) * 0.95)],
            "queries_tracked": len(self.query_times),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "query_errors": self.query_errors
        }