"""Global cache manager for the TTRPG Assistant."""

from typing import Any, Dict, Optional

from config.logging_config import get_logger
from config.settings import settings
from .cache_system import CacheSystem, CachePolicy
from .cache_invalidator import CacheInvalidator
from .cache_config import CacheConfiguration, CacheType
from .result_cache import result_cache

logger = get_logger(__name__)


class GlobalCacheManager:
    """Manages all cache systems in the application."""
    
    def __init__(self):
        """Initialize global cache manager."""
        self.caches: Dict[str, CacheSystem] = {}
        self.invalidator = CacheInvalidator()
        self.config = CacheConfiguration()
        
        # Initialize caches based on configuration
        self._initialize_caches()
        
        # Register caches with invalidator
        for name, cache in self.caches.items():
            self.invalidator.register_cache(name, cache)
        
        logger.info(f"Global cache manager initialized with {len(self.caches)} caches")
    
    def _initialize_caches(self) -> None:
        """Initialize all configured cache systems."""
        # Initialize caches from profiles
        for profile in self.config.profiles.values():
            try:
                # Convert policy string to enum
                policy = CachePolicy[profile.policy.upper()]
                
                cache = CacheSystem(
                    name=profile.name,
                    max_size=profile.max_size,
                    max_memory_mb=profile.max_memory_mb,
                    ttl_seconds=profile.ttl_seconds,
                    policy=policy,
                    persistent=profile.persistent,
                    auto_cleanup_interval=profile.auto_cleanup_interval,
                )
                
                self.caches[profile.name] = cache
                logger.info(f"Initialized cache: {profile.name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize cache {profile.name}: {e}")
    
    def get_cache(self, name: str) -> Optional[CacheSystem]:
        """
        Get a cache by name.
        
        Args:
            name: Cache name
            
        Returns:
            Cache system or None
        """
        return self.caches.get(name)
    
    def get_cache_by_type(self, cache_type: CacheType) -> Optional[CacheSystem]:
        """
        Get the primary cache for a specific type.
        
        Args:
            cache_type: Type of cache
            
        Returns:
            Cache system or None
        """
        profiles = self.config.get_profiles_by_type(cache_type)
        if profiles:
            # Return the first (primary) cache for this type
            return self.caches.get(profiles[0].name)
        return None
    
    def create_cache(
        self,
        name: str,
        cache_type: CacheType,
        max_size: int = 100,
        max_memory_mb: int = 10,
        ttl_seconds: int = 3600,
        policy: CachePolicy = CachePolicy.LRU,
        persistent: bool = False,
    ) -> CacheSystem:
        """
        Create a new cache dynamically.
        
        Args:
            name: Cache name
            cache_type: Type of cache
            max_size: Maximum entries
            max_memory_mb: Maximum memory in MB
            ttl_seconds: Time-to-live
            policy: Eviction policy
            persistent: Whether to persist to disk
            
        Returns:
            Created cache system
        """
        if name in self.caches:
            logger.warning(f"Cache {name} already exists")
            return self.caches[name]
        
        # Create cache
        cache = CacheSystem(
            name=name,
            max_size=max_size,
            max_memory_mb=max_memory_mb,
            ttl_seconds=ttl_seconds,
            policy=policy,
            persistent=persistent,
        )
        
        self.caches[name] = cache
        
        # Register with invalidator
        self.invalidator.register_cache(name, cache)
        
        # Add profile to configuration
        from .cache_config import CacheProfile
        profile = CacheProfile(
            name=name,
            cache_type=cache_type,
            max_size=max_size,
            max_memory_mb=max_memory_mb,
            ttl_seconds=ttl_seconds,
            policy=policy.value,
            persistent=persistent,
        )
        self.config.add_profile(profile)
        
        logger.info(f"Created new cache: {name}")
        return cache
    
    def invalidate_all(self) -> Dict[str, int]:
        """
        Invalidate all caches.
        
        Returns:
            Dictionary mapping cache names to invalidated count
        """
        results = {}
        for name, cache in self.caches.items():
            size_before = len(cache.cache)
            cache.clear()
            results[name] = size_before
        
        logger.info(f"Invalidated all caches: {sum(results.values())} total entries")
        return results
    
    def get_global_stats(self) -> Dict[str, Any]:
        """
        Get global cache statistics.
        
        Returns:
            Global statistics across all caches
        """
        total_hits = 0
        total_misses = 0
        total_evictions = 0
        total_memory = 0
        total_entries = 0
        
        cache_stats = {}
        
        for name, cache in self.caches.items():
            stats = cache.get_stats()
            cache_stats[name] = stats
            
            total_hits += stats["statistics"]["total_hits"]
            total_misses += stats["statistics"]["total_misses"]
            total_evictions += stats["statistics"]["total_evictions"]
            total_memory += stats["memory_usage_mb"]
            total_entries += stats["current_size"]
        
        total_requests = total_hits + total_misses
        global_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        return {
            "total_caches": len(self.caches),
            "total_entries": total_entries,
            "total_memory_mb": total_memory,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_evictions": total_evictions,
            "global_hit_rate": global_hit_rate,
            "cache_stats": cache_stats,
            "invalidation_stats": self.invalidator.get_invalidation_stats(),
            "result_cache_stats": result_cache.get_all_stats(),
        }
    
    def optimize_all(self) -> Dict[str, Any]:
        """
        Optimize all caches based on current usage patterns.
        
        Returns:
            Optimization results
        """
        optimizations = {}
        
        # Get total memory budget
        total_memory = self.config.get_global_setting(
            "max_total_memory_mb",
            settings.cache_max_memory_mb
        )
        
        # Optimize memory allocation
        allocations = self.config.optimize_memory_allocation(total_memory)
        
        # Apply optimizations
        for name, memory_mb in allocations.items():
            if name in self.caches:
                cache = self.caches[name]
                old_memory = cache.max_memory_bytes / (1024 * 1024)
                cache.max_memory_bytes = memory_mb * 1024 * 1024
                
                optimizations[name] = {
                    "old_memory_mb": old_memory,
                    "new_memory_mb": memory_mb,
                    "change": memory_mb - old_memory,
                }
        
        # Process scheduled invalidations
        invalidation_results = self.invalidator.process_scheduled_invalidations()
        
        # Invalidate stale entries
        stale_results = self.invalidator.invalidate_stale(
            max_age_seconds=86400,  # 24 hours
            max_idle_seconds=7200,  # 2 hours idle
        )
        
        return {
            "memory_optimizations": optimizations,
            "scheduled_invalidations": invalidation_results,
            "stale_invalidations": stale_results,
        }
    
    def shutdown(self) -> None:
        """Shutdown cache manager and save persistent caches."""
        logger.info("Shutting down cache manager")
        
        # Save persistent caches
        for name, cache in self.caches.items():
            if cache.persistent:
                cache._save_to_disk()
                logger.info(f"Saved persistent cache: {name}")
        
        # Clear non-persistent caches
        for name, cache in self.caches.items():
            if not cache.persistent:
                cache.clear()
        
        logger.info("Cache manager shutdown complete")