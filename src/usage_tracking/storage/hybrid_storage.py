"""
Hybrid storage strategy combining ChromaDB and JSON storage.

This module provides:
- Intelligent data placement based on access patterns
- Automatic data migration between storage tiers
- Consistent caching layer with write-through semantics
- Conflict resolution and synchronization
- Performance optimization and monitoring
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor

from .models import (
    UsageRecord, UserProfile, UsagePattern, UsageMetrics,
    HybridStorageConfig, StorageType, UsageEventType, ProviderType,
    TimeAggregation
)
from .chromadb_storage import ChromaDBUsageStorage
from .json_storage import JSONUsageStorage

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cache entry with TTL and access tracking."""
    
    def __init__(self, data: Any, ttl_seconds: int = 3600):
        self.data = data
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 1
        self.ttl_seconds = ttl_seconds
        self.is_dirty = False  # Needs to be written to storage
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return (time.time() - self.created_at) > self.ttl_seconds
    
    def access(self) -> Any:
        """Access cached data and update tracking."""
        self.last_accessed = time.time()
        self.access_count += 1
        return self.data
    
    def update(self, data: Any, mark_dirty: bool = True) -> None:
        """Update cached data."""
        self.data = data
        self.last_accessed = time.time()
        self.access_count += 1
        if mark_dirty:
            self.is_dirty = True


class IntelligentCache:
    """Intelligent cache with LRU eviction and write-through semantics."""
    
    def __init__(self, max_size_mb: int = 500, ttl_seconds: int = 3600):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RWLock()
        
        # Access pattern tracking
        self.access_patterns = defaultdict(list)
        self.hit_rate_stats = {"hits": 0, "misses": 0}
        
        # Background cleanup
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of cached data."""
        if isinstance(data, str):
            return len(data.encode('utf-8'))
        elif isinstance(data, dict):
            return len(str(data).encode('utf-8'))
        elif hasattr(data, '__dict__'):
            return len(str(data.__dict__).encode('utf-8'))
        else:
            return len(str(data).encode('utf-8'))
    
    def _should_cleanup(self) -> bool:
        """Check if cache cleanup should run."""
        return (time.time() - self.last_cleanup) > self.cleanup_interval
    
    def _cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        with self.lock.write_lock():
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            self.last_cleanup = time.time()
            return len(expired_keys)
    
    def _evict_lru(self, target_size: int) -> int:
        """Evict least recently used entries to reach target size."""
        if not self.cache:
            return 0
        
        # Sort by last accessed time (LRU first)
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        evicted_count = 0
        current_size = self.get_current_size()
        
        for key, entry in sorted_entries:
            if current_size <= target_size:
                break
            
            entry_size = self._estimate_size(entry.data)
            del self.cache[key]
            current_size -= entry_size
            evicted_count += 1
        
        return evicted_count
    
    def get_current_size(self) -> int:
        """Get current cache size in bytes."""
        total_size = 0
        for entry in self.cache.values():
            total_size += self._estimate_size(entry.data)
        return total_size
    
    def get(self, key: str) -> Optional[Any]:
        """Get data from cache."""
        with self.lock.read_lock():
            if key not in self.cache:
                self.hit_rate_stats["misses"] += 1
                return None
            
            entry = self.cache[key]
            if entry.is_expired:
                self.hit_rate_stats["misses"] += 1
                return None
            
            self.hit_rate_stats["hits"] += 1
            return entry.access()
    
    def put(self, key: str, data: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Put data into cache."""
        with self.lock.write_lock():
            # Use default TTL if not specified
            ttl = ttl_seconds or self.ttl_seconds
            
            # Check if we need cleanup
            if self._should_cleanup():
                self._cleanup_expired()
            
            # Estimate new entry size
            new_entry_size = self._estimate_size(data)
            current_size = self.get_current_size()
            
            # Evict if necessary
            if current_size + new_entry_size > self.max_size_bytes:
                target_size = self.max_size_bytes - new_entry_size
                self._evict_lru(target_size)
            
            # Add new entry
            self.cache[key] = CacheEntry(data, ttl)
            
            # Track access pattern
            self.access_patterns[key].append(time.time())
            
            return True
    
    def invalidate(self, key: str) -> bool:
        """Remove entry from cache."""
        with self.lock.write_lock():
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def get_dirty_entries(self) -> List[Tuple[str, Any]]:
        """Get entries that need to be written to storage."""
        with self.lock.read_lock():
            dirty_entries = []
            for key, entry in self.cache.items():
                if entry.is_dirty:
                    dirty_entries.append((key, entry.data))
            return dirty_entries
    
    def mark_clean(self, key: str) -> None:
        """Mark cache entry as clean (written to storage)."""
        with self.lock.write_lock():
            if key in self.cache:
                self.cache[key].is_dirty = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock.read_lock():
            total_hits = self.hit_rate_stats["hits"]
            total_requests = total_hits + self.hit_rate_stats["misses"]
            hit_rate = total_hits / max(total_requests, 1)
            
            return {
                "cache_size": len(self.cache),
                "size_bytes": self.get_current_size(),
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "total_hits": total_hits,
                "total_misses": self.hit_rate_stats["misses"]
            }


class DataTierManager:
    """Manages data placement and migration between storage tiers."""
    
    def __init__(self, config: HybridStorageConfig):
        self.config = config
        self.access_tracker = defaultdict(list)  # Track access times by record ID
        self.migration_queue = asyncio.Queue()
        self.migration_stats = {
            "hot_to_warm": 0,
            "warm_to_cold": 0,
            "cold_to_warm": 0,
            "warm_to_hot": 0
        }
    
    def track_access(self, record_id: str) -> None:
        """Track access to a record for tier placement decisions."""
        current_time = time.time()
        self.access_tracker[record_id].append(current_time)
        
        # Keep only recent access history
        cutoff_time = current_time - (self.config.hot_data_days * 24 * 3600)
        self.access_tracker[record_id] = [
            t for t in self.access_tracker[record_id] if t >= cutoff_time
        ]
    
    def determine_optimal_tier(self, record_id: str, record_age_days: float) -> StorageType:
        """Determine optimal storage tier for a record."""
        access_times = self.access_tracker.get(record_id, [])
        recent_accesses = len(access_times)
        
        # Hot tier criteria: recent access or young age
        if (recent_accesses > 0 and record_age_days <= self.config.hot_data_days):
            return self.config.hot_data_storage
        
        # Warm tier criteria: some access or medium age
        if (recent_accesses > 0 or record_age_days <= self.config.warm_data_days):
            return self.config.warm_data_storage
        
        # Cold tier: old and rarely accessed
        return self.config.cold_data_storage
    
    async def queue_migration(
        self, 
        record_id: str, 
        from_tier: StorageType, 
        to_tier: StorageType, 
        data: Any
    ) -> None:
        """Queue a data migration between tiers."""
        if from_tier == to_tier:
            return
        
        migration_task = {
            "record_id": record_id,
            "from_tier": from_tier,
            "to_tier": to_tier,
            "data": data,
            "queued_at": time.time()
        }
        
        await self.migration_queue.put(migration_task)
    
    def get_migration_stats(self) -> Dict[str, Any]:
        """Get data migration statistics."""
        return {
            "migration_stats": self.migration_stats.copy(),
            "queue_size": self.migration_queue.qsize(),
            "tracked_records": len(self.access_tracker)
        }


class HybridUsageStorage:
    """Hybrid storage strategy combining ChromaDB and JSON storage."""
    
    def __init__(
        self,
        config: HybridStorageConfig,
        chromadb_storage: ChromaDBUsageStorage,
        json_storage: JSONUsageStorage
    ):
        self.config = config
        self.chromadb = chromadb_storage
        self.json = json_storage
        
        # Initialize cache and tier manager
        self.cache = IntelligentCache(
            max_size_mb=config.cache_size_mb,
            ttl_seconds=3600
        )
        self.tier_manager = DataTierManager(config)
        
        # Background workers
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="hybrid_storage")
        self.migration_worker_running = False
        self.sync_worker_running = False
        
        # Performance tracking
        self.operation_stats = defaultdict(lambda: {
            "count": 0, "total_time": 0.0, "errors": 0
        })
        
        # Start background workers
        if config.auto_migrate:
            self._start_workers()
        
        logger.info(
            "Hybrid storage initialized",
            hot_storage=config.hot_data_storage.value,
            warm_storage=config.warm_data_storage.value,
            cold_storage=config.cold_data_storage.value,
            cache_size_mb=config.cache_size_mb
        )
    
    def _start_workers(self) -> None:
        """Start background worker tasks."""
        self.migration_worker_running = True
        self.sync_worker_running = True
        
        # Start migration worker
        asyncio.create_task(self._migration_worker())
        
        # Start sync worker
        asyncio.create_task(self._sync_worker())
    
    async def _migration_worker(self) -> None:
        """Background worker for data migration between tiers."""
        logger.info("Migration worker started")
        
        while self.migration_worker_running:
            try:
                # Wait for migration task
                migration_task = await asyncio.wait_for(
                    self.tier_manager.migration_queue.get(),
                    timeout=60.0
                )
                
                await self._perform_migration(migration_task)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Migration worker error: {e}")
                await asyncio.sleep(10)
    
    async def _sync_worker(self) -> None:
        """Background worker for cache synchronization."""
        logger.info("Sync worker started")
        
        while self.sync_worker_running:
            try:
                await asyncio.sleep(self.config.sync_frequency_minutes * 60)
                await self._sync_dirty_cache_entries()
                
            except Exception as e:
                logger.error(f"Sync worker error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_migration(self, migration_task: Dict[str, Any]) -> None:
        """Perform data migration between storage tiers."""
        record_id = migration_task["record_id"]
        from_tier = migration_task["from_tier"]
        to_tier = migration_task["to_tier"]
        data = migration_task["data"]
        
        try:
            # Write to destination tier
            if to_tier == StorageType.CHROMADB:
                if isinstance(data, UsageRecord):
                    await self.chromadb.store_usage_record(data)
                elif isinstance(data, UsagePattern):
                    await self.chromadb.store_usage_pattern(data)
            elif to_tier == StorageType.JSON:
                if isinstance(data, UsageRecord):
                    await self.json.store_usage_record(data)
                elif isinstance(data, UsageMetrics):
                    await self.json.store_usage_metrics(data)
            
            # Update migration stats
            migration_key = f"{from_tier.value}_to_{to_tier.value}"
            self.tier_manager.migration_stats[migration_key] = \
                self.tier_manager.migration_stats.get(migration_key, 0) + 1
            
            logger.debug(
                "Data migrated",
                record_id=record_id,
                from_tier=from_tier.value,
                to_tier=to_tier.value
            )
            
        except Exception as e:
            logger.error(
                f"Failed to migrate {record_id} from {from_tier.value} to {to_tier.value}: {e}"
            )
    
    async def _sync_dirty_cache_entries(self) -> None:
        """Synchronize dirty cache entries to storage."""
        dirty_entries = self.cache.get_dirty_entries()
        if not dirty_entries:
            return
        
        synced_count = 0
        for key, data in dirty_entries:
            try:
                # Determine appropriate storage based on data type and age
                if isinstance(data, UserProfile):
                    await self.json.save_user_profile(data)
                elif isinstance(data, UsageRecord):
                    # Store in appropriate tier based on age
                    record_age = (datetime.utcnow() - data.timestamp).days
                    optimal_tier = self.tier_manager.determine_optimal_tier(
                        data.record_id, record_age
                    )
                    
                    if optimal_tier == StorageType.CHROMADB:
                        await self.chromadb.store_usage_record(data)
                    else:
                        await self.json.store_usage_record(data)
                
                # Mark as clean
                self.cache.mark_clean(key)
                synced_count += 1
                
            except Exception as e:
                logger.error(f"Failed to sync cache entry {key}: {e}")
        
        if synced_count > 0:
            logger.info(f"Synced {synced_count} dirty cache entries")
    
    async def store_usage_record(self, record: UsageRecord) -> str:
        """Store usage record using hybrid strategy."""
        start_time = time.time()
        operation = "store_usage_record"
        
        try:
            # Cache the record
            cache_key = f"usage_record:{record.record_id}"
            self.cache.put(cache_key, record, ttl_seconds=3600)
            
            # Track access for tier management
            self.tier_manager.track_access(record.record_id)
            
            # Determine optimal storage tier
            record_age = 0  # New record
            optimal_tier = self.tier_manager.determine_optimal_tier(
                record.record_id, record_age
            )
            
            # Store in optimal tier
            if optimal_tier == StorageType.CHROMADB:
                result = await self.chromadb.store_usage_record(record)
            else:
                result = await self.json.store_usage_record(record)
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_operation_performance(operation, execution_time, success=True)
            
            logger.debug(
                "Usage record stored",
                record_id=record.record_id,
                storage_tier=optimal_tier.value,
                execution_time=execution_time
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._track_operation_performance(operation, execution_time, success=False)
            logger.error(f"Failed to store usage record: {e}")
            raise
    
    async def get_usage_records(
        self,
        user_id: Optional[str] = None,
        event_types: Optional[List[UsageEventType]] = None,
        providers: Optional[List[ProviderType]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        semantic_query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get usage records using hybrid strategy with intelligent caching."""
        start_time = time.time()
        operation = "get_usage_records"
        
        try:
            # Build cache key for query results
            cache_key = f"query:{user_id}:{event_types}:{providers}:{start_date}:{end_date}:{limit}:{semantic_query}"
            
            # Check cache first
            cached_result = self.cache.get(cache_key)
            if cached_result:
                execution_time = time.time() - start_time
                self._track_operation_performance(operation, execution_time, success=True)
                return cached_result
            
            # Query from both storage tiers and merge results
            results = []
            
            # Query ChromaDB for recent/hot data
            if start_date is None or (datetime.utcnow() - start_date).days <= self.config.hot_data_days:
                try:
                    chromadb_results = await self.chromadb.query_usage_records(
                        user_id=user_id,
                        event_types=event_types,
                        providers=providers,
                        start_date=start_date,
                        end_date=end_date,
                        limit=limit,
                        semantic_query=semantic_query
                    )
                    results.extend(chromadb_results)
                except Exception as e:
                    logger.warning(f"ChromaDB query failed: {e}")
            
            # Query JSON storage for older data if needed
            if len(results) < limit:
                try:
                    json_results = await self.json.get_usage_records(
                        user_id=user_id,
                        start_time=start_date,
                        end_time=end_date,
                        limit=limit - len(results)
                    )
                    # Convert JSON results to match ChromaDB format
                    for record in json_results:
                        results.append({
                            "document": f"Usage record {record.get('record_id', '')}",
                            "metadata": record,
                            "relevance_score": 1.0
                        })
                except Exception as e:
                    logger.warning(f"JSON storage query failed: {e}")
            
            # Sort by timestamp and limit
            results.sort(key=lambda x: x.get("metadata", {}).get("timestamp", ""), reverse=True)
            results = results[:limit]
            
            # Cache results
            self.cache.put(cache_key, results, ttl_seconds=300)  # 5 minute TTL for queries
            
            # Track access for records
            for result in results:
                record_id = result.get("metadata", {}).get("record_id")
                if record_id:
                    self.tier_manager.track_access(record_id)
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_operation_performance(operation, execution_time, success=True)
            
            logger.debug(
                "Usage records retrieved",
                result_count=len(results),
                execution_time=execution_time,
                cached=False
            )
            
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._track_operation_performance(operation, execution_time, success=False)
            logger.error(f"Failed to get usage records: {e}")
            raise
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile with caching."""
        start_time = time.time()
        operation = "get_user_profile"
        
        try:
            # Check cache first
            cache_key = f"user_profile:{user_id}"
            cached_profile = self.cache.get(cache_key)
            if cached_profile:
                execution_time = time.time() - start_time
                self._track_operation_performance(operation, execution_time, success=True)
                return cached_profile
            
            # Load from JSON storage (user profiles are always stored in JSON)
            profile = await self.json.get_user_profile(user_id)
            
            # Cache the profile
            if profile:
                self.cache.put(cache_key, profile, ttl_seconds=1800)  # 30 minute TTL
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_operation_performance(operation, execution_time, success=True)
            
            return profile
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._track_operation_performance(operation, execution_time, success=False)
            logger.error(f"Failed to get user profile: {e}")
            raise
    
    async def save_user_profile(self, profile: UserProfile) -> bool:
        """Save user profile with write-through caching."""
        start_time = time.time()
        operation = "save_user_profile"
        
        try:
            # Save to JSON storage
            success = await self.json.save_user_profile(profile)
            
            if success:
                # Update cache
                cache_key = f"user_profile:{profile.user_id}"
                self.cache.put(cache_key, profile, ttl_seconds=1800)
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_operation_performance(operation, execution_time, success=success)
            
            return success
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._track_operation_performance(operation, execution_time, success=False)
            logger.error(f"Failed to save user profile: {e}")
            raise
    
    async def store_usage_pattern(self, pattern: UsagePattern) -> str:
        """Store usage pattern in ChromaDB (patterns are always hot data)."""
        return await self.chromadb.store_usage_pattern(pattern)
    
    async def store_usage_metrics(self, metrics: UsageMetrics) -> str:
        """Store usage metrics in JSON storage (for aggregated data)."""
        return await self.json.store_usage_metrics(metrics)
    
    def _track_operation_performance(
        self, 
        operation: str, 
        execution_time: float, 
        success: bool = True
    ) -> None:
        """Track operation performance metrics."""
        stats = self.operation_stats[operation]
        stats["count"] += 1
        stats["total_time"] += execution_time
        if not success:
            stats["errors"] += 1
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all storage components."""
        stats = {
            "hybrid_config": {
                "hot_storage": self.config.hot_data_storage.value,
                "warm_storage": self.config.warm_data_storage.value,
                "cold_storage": self.config.cold_data_storage.value,
                "cache_size_mb": self.config.cache_size_mb,
                "auto_migrate": self.config.auto_migrate
            },
            "cache_stats": self.cache.get_stats(),
            "tier_management": self.tier_manager.get_migration_stats(),
            "operation_performance": {}
        }
        
        # Calculate operation performance metrics
        for operation, raw_stats in self.operation_stats.items():
            if raw_stats["count"] > 0:
                stats["operation_performance"][operation] = {
                    "count": raw_stats["count"],
                    "avg_time": raw_stats["total_time"] / raw_stats["count"],
                    "error_rate": raw_stats["errors"] / raw_stats["count"],
                    "total_time": raw_stats["total_time"]
                }
        
        # Get storage-specific stats
        try:
            stats["chromadb_stats"] = await self.chromadb.get_collection_stats()
        except Exception as e:
            stats["chromadb_stats"] = {"error": str(e)}
        
        try:
            stats["json_stats"] = await self.json.get_storage_stats()
        except Exception as e:
            stats["json_stats"] = {"error": str(e)}
        
        return stats
    
    async def cleanup_old_data(self, retention_days: int = 90) -> Dict[str, Any]:
        """Clean up old data from all storage tiers."""
        cleanup_results = {}
        
        # Cleanup ChromaDB
        try:
            chromadb_cleaned = await self.chromadb.cleanup_old_records(retention_days)
            cleanup_results["chromadb_records"] = chromadb_cleaned
        except Exception as e:
            cleanup_results["chromadb_error"] = str(e)
        
        # Cleanup JSON storage
        try:
            json_cleaned = await self.json.cleanup_old_data(retention_days)
            cleanup_results["json_storage"] = json_cleaned
        except Exception as e:
            cleanup_results["json_error"] = str(e)
        
        # Cache cleanup is automatic, but force it
        expired_removed = self.cache._cleanup_expired()
        cleanup_results["cache_expired"] = expired_removed
        
        logger.info("Hybrid storage cleanup completed", **cleanup_results)
        return cleanup_results
    
    async def close(self) -> None:
        """Close hybrid storage and all components."""
        try:
            # Stop background workers
            self.migration_worker_running = False
            self.sync_worker_running = False
            
            # Sync any remaining dirty cache entries
            await self._sync_dirty_cache_entries()
            
            # Close storage components
            await self.chromadb.close()
            await self.json.close()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("Hybrid storage closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing hybrid storage: {e}")