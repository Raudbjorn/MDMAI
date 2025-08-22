"""MCP tools for cache management and performance monitoring."""

from typing import Any, Dict, List, Optional, Set

from config.logging_config import get_logger

logger = get_logger(__name__)

# Global cache manager and invalidator (will be initialized in main.py)
_cache_manager = None
_cache_invalidator = None
_cache_config = None


def initialize_performance_tools(cache_manager, cache_invalidator, cache_config):
    """
    Initialize performance tools with required components.
    
    Args:
        cache_manager: Global cache manager instance
        cache_invalidator: Cache invalidator instance
        cache_config: Cache configuration instance
    """
    global _cache_manager, _cache_invalidator, _cache_config
    _cache_manager = cache_manager
    _cache_invalidator = cache_invalidator
    _cache_config = cache_config
    logger.info("Performance tools initialized")


def register_performance_tools(mcp_server):
    """
    Register performance tools with the MCP server.
    
    Args:
        mcp_server: The MCP server instance to register tools with
    """
    mcp_server.tool()(get_cache_stats)
    mcp_server.tool()(clear_cache)
    mcp_server.tool()(invalidate_cache)
    mcp_server.tool()(configure_cache)
    mcp_server.tool()(optimize_cache_memory)
    mcp_server.tool()(get_cache_recommendations)
    logger.info("Performance tools registered with MCP server")


async def get_cache_stats(
    cache_name: Optional[str] = None,
    detailed: bool = False,
) -> Dict[str, Any]:
    """
    Get cache statistics and performance metrics.
    
    Args:
        cache_name: Specific cache to get stats for (None for all)
        detailed: Whether to include detailed statistics
        
    Returns:
        Cache statistics and metrics
    """
    try:
        if not _cache_manager:
            return {
                "success": False,
                "error": "Cache manager not initialized",
            }
        
        stats = {}
        
        if cache_name:
            # Get stats for specific cache
            if cache_name in _cache_manager.caches:
                cache_stats = _cache_manager.caches[cache_name].get_stats()
                if detailed:
                    # Add more detailed information
                    cache_stats["invalidation_history"] = (
                        _cache_invalidator.get_invalidation_stats()
                        if _cache_invalidator else {}
                    )
                stats[cache_name] = cache_stats
            else:
                return {
                    "success": False,
                    "error": f"Cache '{cache_name}' not found",
                }
        else:
            # Get stats for all caches
            for name, cache in _cache_manager.caches.items():
                stats[name] = cache.get_stats()
            
            # Add global statistics
            if detailed:
                total_hits = sum(
                    c.stats.total_hits for c in _cache_manager.caches.values()
                )
                total_misses = sum(
                    c.stats.total_misses for c in _cache_manager.caches.values()
                )
                total_memory = sum(
                    c._get_total_size() for c in _cache_manager.caches.values()
                )
                
                stats["global"] = {
                    "total_caches": len(_cache_manager.caches),
                    "total_hits": total_hits,
                    "total_misses": total_misses,
                    "global_hit_rate": total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0,
                    "total_memory_mb": total_memory / (1024 * 1024),
                    "invalidation_stats": (
                        _cache_invalidator.get_invalidation_stats()
                        if _cache_invalidator else {}
                    ),
                }
        
        return {
            "success": True,
            "statistics": stats,
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache statistics: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def clear_cache(
    cache_name: Optional[str] = None,
    confirm: bool = False,
) -> Dict[str, Any]:
    """
    Clear cache entries.
    
    Args:
        cache_name: Specific cache to clear (None for all)
        confirm: Confirmation flag to prevent accidental clearing
        
    Returns:
        Operation status
    """
    try:
        if not _cache_manager:
            return {
                "success": False,
                "error": "Cache manager not initialized",
            }
        
        if not confirm:
            return {
                "success": False,
                "error": "Please set confirm=True to clear cache",
            }
        
        cleared = []
        
        if cache_name:
            # Clear specific cache
            if cache_name in _cache_manager.caches:
                _cache_manager.caches[cache_name].clear()
                cleared.append(cache_name)
            else:
                return {
                    "success": False,
                    "error": f"Cache '{cache_name}' not found",
                }
        else:
            # Clear all caches
            for name, cache in _cache_manager.caches.items():
                cache.clear()
                cleared.append(name)
        
        return {
            "success": True,
            "message": f"Cleared {len(cleared)} cache(s)",
            "cleared_caches": cleared,
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def invalidate_cache(
    pattern: Optional[str] = None,
    tags: Optional[List[str]] = None,
    cache_name: Optional[str] = None,
    rule_name: Optional[str] = None,
    cascade: bool = True,
) -> Dict[str, Any]:
    """
    Invalidate cache entries based on criteria.
    
    Args:
        pattern: Key pattern to match (supports wildcards)
        tags: Tags to match for invalidation
        cache_name: Specific cache to target
        rule_name: Apply a predefined invalidation rule
        cascade: Whether to cascade invalidation to related entries
        
    Returns:
        Invalidation results
    """
    try:
        if not _cache_invalidator:
            return {
                "success": False,
                "error": "Cache invalidator not initialized",
            }
        
        results = {}
        
        if pattern:
            # Invalidate by pattern
            results = _cache_invalidator.invalidate_by_pattern(
                pattern=pattern,
                cache_name=cache_name,
            )
        elif tags or rule_name:
            # Invalidate by tags or rule
            tag_set = set(tags) if tags else None
            results = _cache_invalidator.invalidate(
                cache_name=cache_name,
                tags=tag_set,
                rule_name=rule_name,
                cascade=cascade,
            )
        else:
            return {
                "success": False,
                "error": "Must specify pattern, tags, or rule_name",
            }
        
        total_invalidated = sum(results.values())
        
        return {
            "success": True,
            "message": f"Invalidated {total_invalidated} entries",
            "results": results,
        }
        
    except Exception as e:
        logger.error(f"Failed to invalidate cache: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def configure_cache(
    cache_name: str,
    max_size: Optional[int] = None,
    max_memory_mb: Optional[int] = None,
    ttl_seconds: Optional[int] = None,
    policy: Optional[str] = None,
    persistent: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Configure cache parameters.
    
    Args:
        cache_name: Cache to configure
        max_size: Maximum number of entries
        max_memory_mb: Maximum memory in MB
        ttl_seconds: Time-to-live in seconds
        policy: Eviction policy (lru, lfu, fifo, ttl, adaptive)
        persistent: Whether to persist to disk
        
    Returns:
        Configuration status
    """
    try:
        if not _cache_config:
            return {
                "success": False,
                "error": "Cache configuration not initialized",
            }
        
        # Get or create profile
        profile = _cache_config.get_profile(cache_name)
        if not profile:
            return {
                "success": False,
                "error": f"Cache profile '{cache_name}' not found",
            }
        
        # Update configuration
        updated = []
        if max_size is not None:
            profile.max_size = max_size
            updated.append("max_size")
        if max_memory_mb is not None:
            profile.max_memory_mb = max_memory_mb
            updated.append("max_memory_mb")
        if ttl_seconds is not None:
            profile.ttl_seconds = ttl_seconds
            updated.append("ttl_seconds")
        if policy is not None:
            profile.policy = policy
            updated.append("policy")
        if persistent is not None:
            profile.persistent = persistent
            updated.append("persistent")
        
        # Save configuration
        _cache_config.add_profile(profile)
        
        # Apply to existing cache if it exists
        if _cache_manager and cache_name in _cache_manager.caches:
            cache = _cache_manager.caches[cache_name]
            if max_size is not None:
                cache.max_size = max_size
            if max_memory_mb is not None:
                cache.max_memory_bytes = max_memory_mb * 1024 * 1024
            if ttl_seconds is not None:
                cache.ttl_seconds = ttl_seconds
            if persistent is not None:
                cache.persistent = persistent
        
        return {
            "success": True,
            "message": f"Updated {len(updated)} configuration(s)",
            "updated": updated,
            "profile": profile.to_dict(),
        }
        
    except Exception as e:
        logger.error(f"Failed to configure cache: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def optimize_cache_memory(
    total_memory_mb: Optional[int] = None,
    auto_apply: bool = False,
) -> Dict[str, Any]:
    """
    Optimize memory allocation across caches.
    
    Args:
        total_memory_mb: Total memory budget (uses configured default if not specified)
        auto_apply: Whether to automatically apply recommendations
        
    Returns:
        Memory optimization recommendations
    """
    try:
        if not _cache_config:
            return {
                "success": False,
                "error": "Cache configuration not initialized",
            }
        
        # Use configured limit if not specified
        if total_memory_mb is None:
            total_memory_mb = _cache_config.get_global_setting(
                "max_total_memory_mb",
                100  # Default fallback
            )
        
        # Get optimized allocations
        allocations = _cache_config.optimize_memory_allocation(total_memory_mb)
        
        # Apply if requested
        if auto_apply and _cache_manager:
            for name, memory_mb in allocations.items():
                if name in _cache_manager.caches:
                    cache = _cache_manager.caches[name]
                    cache.max_memory_bytes = memory_mb * 1024 * 1024
                    logger.info(f"Applied memory allocation to {name}: {memory_mb}MB")
        
        return {
            "success": True,
            "total_memory_mb": total_memory_mb,
            "allocations": allocations,
            "applied": auto_apply,
        }
        
    except Exception as e:
        logger.error(f"Failed to optimize cache memory: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def get_cache_recommendations() -> Dict[str, Any]:
    """
    Get cache configuration recommendations.
    
    Returns:
        Cache optimization recommendations
    """
    try:
        if not _cache_config:
            return {
                "success": False,
                "error": "Cache configuration not initialized",
            }
        
        recommendations = _cache_config.get_cache_recommendations()
        
        # Add runtime recommendations if cache manager is available
        if _cache_manager:
            runtime_recs = []
            
            for name, cache in _cache_manager.caches.items():
                stats = cache.get_stats()
                
                # Check hit rate
                hit_rate = stats["statistics"]["hit_rate"]
                if hit_rate < 0.3:
                    runtime_recs.append({
                        "cache": name,
                        "issue": "Low hit rate",
                        "recommendation": f"Hit rate is {hit_rate:.1%}. Consider increasing cache size or TTL.",
                    })
                
                # Check memory efficiency
                memory_eff = stats["statistics"]["memory_efficiency"]
                if memory_eff < 0.5:
                    runtime_recs.append({
                        "cache": name,
                        "issue": "Low memory efficiency",
                        "recommendation": f"Memory efficiency is {memory_eff:.1%}. Consider reducing cache size.",
                    })
                
                # Check eviction rate
                evictions = stats["statistics"]["total_evictions"]
                hits = stats["statistics"]["total_hits"]
                if hits > 0 and evictions / hits > 0.5:
                    runtime_recs.append({
                        "cache": name,
                        "issue": "High eviction rate",
                        "recommendation": "Too many evictions. Consider increasing cache size or memory limit.",
                    })
            
            recommendations["runtime"] = runtime_recs
        
        return {
            "success": True,
            "recommendations": recommendations,
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache recommendations: {e}")
        return {
            "success": False,
            "error": str(e),
        }