"""MCP tools for cache management and performance monitoring."""

from typing import Any, Dict, List, Optional, Set

from config.logging_config import get_logger

logger = get_logger(__name__)

# Global cache manager and invalidator (will be initialized in main.py)
_cache_manager = None
_cache_invalidator = None
_cache_config = None
_db = None
_optimizer = None
_monitor = None


def initialize_performance_tools(cache_manager, cache_invalidator, cache_config, db=None):
    """
    Initialize performance tools with required components.
    
    Args:
        cache_manager: Global cache manager instance
        cache_invalidator: Cache invalidator instance
        cache_config: Cache configuration instance
        db: Database manager with optimizer and monitor
    """
    global _cache_manager, _cache_invalidator, _cache_config, _db, _optimizer, _monitor
    _cache_manager = cache_manager
    _cache_invalidator = cache_invalidator
    _cache_config = cache_config
    _db = db
    _optimizer = db.optimizer if db else None
    _monitor = db.monitor if db else None
    logger.info("Performance tools initialized")


def register_performance_tools(mcp_server):
    """
    Register performance tools with the MCP server.
    
    Args:
        mcp_server: The MCP server instance to register tools with
    """
    # Cache management tools
    mcp_server.tool()(get_cache_stats)
    mcp_server.tool()(clear_cache)
    mcp_server.tool()(invalidate_cache)
    mcp_server.tool()(configure_cache)
    mcp_server.tool()(optimize_cache_memory)
    mcp_server.tool()(get_cache_recommendations)
    
    # Database performance tools
    mcp_server.tool()(optimize_database)
    mcp_server.tool()(get_database_performance_report)
    mcp_server.tool()(start_performance_monitoring)
    mcp_server.tool()(stop_performance_monitoring)
    mcp_server.tool()(analyze_query_performance)
    
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
    Configure cache parameters with validation.
    
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
        
        # Validate and update configuration
        updated = []
        if max_size is not None:
            if max_size <= 0:
                return {
                    "success": False,
                    "error": "max_size must be positive"
                }
            profile.max_size = max_size
            updated.append("max_size")
        if max_memory_mb is not None:
            if max_memory_mb <= 0:
                return {
                    "success": False,
                    "error": "max_memory_mb must be positive"
                }
            profile.max_memory_mb = max_memory_mb
            updated.append("max_memory_mb")
        if ttl_seconds is not None:
            if ttl_seconds < 0:
                return {
                    "success": False,
                    "error": "ttl_seconds cannot be negative"
                }
            profile.ttl_seconds = ttl_seconds
            updated.append("ttl_seconds")
        if policy is not None:
            valid_policies = ["LRU", "LFU", "FIFO", "RANDOM"]
            if policy.upper() not in valid_policies:
                return {
                    "success": False,
                    "error": f"Invalid policy. Must be one of: {', '.join(valid_policies)}"
                }
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


async def optimize_database(
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Optimize database indices for improved search performance.
    
    Args:
        collection: Specific collection to optimize, or None for all collections
        
    Returns:
        Optimization results including metrics and recommendations
    """
    if not _db:
        return {
            "success": False,
            "error": "Database not initialized",
        }
    
    try:
        results = await _db.optimize_indices(collection)
        
        return {
            "success": True,
            "message": f"Successfully optimized {len(results.get('optimized', []))} collections",
            "optimized_collections": results.get("optimized", []),
            "metrics": results.get("metrics", {}),
            "total_time": results.get("total_time", 0),
            "errors": results.get("errors", []),
        }
        
    except Exception as e:
        logger.error(f"Database optimization failed: {str(e)}")
        return {
            "success": False,
            "error": f"Optimization failed: {str(e)}",
        }


async def get_database_performance_report() -> Dict[str, Any]:
    """
    Generate a comprehensive performance report for the database and system.
    
    Returns:
        Performance metrics, statistics, and recommendations
    """
    if not _db:
        return {
            "success": False,
            "error": "Database not initialized",
        }
    
    try:
        report = await _db.get_performance_report()
        
        # Extract key metrics
        result = {
            "success": True,
            "database_stats": report.get("database", {}),
            "query_performance": {},
            "system_metrics": {},
            "recommendations": [],
        }
        
        # Process optimizer report
        if "optimizer" in report and report["optimizer"]:
            optimizer_report = report["optimizer"]
            if "query_metrics" in optimizer_report:
                result["query_performance"] = optimizer_report["query_metrics"]
            if "recommendations" in optimizer_report:
                result["recommendations"].extend(optimizer_report["recommendations"])
        
        # Process monitor report
        if "monitor" in report and report["monitor"]:
            monitor_report = report["monitor"]
            if "system" in monitor_report:
                result["system_metrics"] = monitor_report["system"]
            if "operations" in monitor_report:
                result["operation_stats"] = monitor_report["operations"]
            if "recommendations" in monitor_report:
                result["recommendations"].extend(monitor_report["recommendations"])
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate performance report: {str(e)}")
        return {
            "success": False,
            "error": f"Report generation failed: {str(e)}",
        }


async def start_performance_monitoring(
    interval: int = 60,
) -> Dict[str, Any]:
    """
    Start continuous performance monitoring.
    
    Args:
        interval: Monitoring interval in seconds (default: 60)
        
    Returns:
        Monitoring status
    """
    if not _db:
        return {
            "success": False,
            "error": "Database not initialized",
        }
    
    try:
        await _db.start_monitoring(interval)
        
        return {
            "success": True,
            "message": f"Performance monitoring started with {interval}s interval",
            "status": "active",
            "interval": interval,
        }
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {str(e)}")
        return {
            "success": False,
            "error": f"Monitoring start failed: {str(e)}",
        }


async def stop_performance_monitoring() -> Dict[str, Any]:
    """
    Stop performance monitoring.
    
    Returns:
        Monitoring status
    """
    if not _db:
        return {
            "success": False,
            "error": "Database not initialized",
        }
    
    try:
        await _db.stop_monitoring()
        
        return {
            "success": True,
            "message": "Performance monitoring stopped",
            "status": "stopped",
        }
        
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {str(e)}")
        return {
            "success": False,
            "error": f"Monitoring stop failed: {str(e)}",
        }


async def analyze_query_performance(
    query: str,
) -> Dict[str, Any]:
    """
    Analyze and optimize a specific query for better performance.
    
    Args:
        query: The query to analyze
        
    Returns:
        Analysis results and optimization suggestions
    """
    if not _optimizer:
        return {
            "success": False,
            "error": "Query optimizer not available",
        }
    
    try:
        # Analyze the query
        analysis = _optimizer._analyze_query(query)
        
        # Generate optimized version
        optimized = _optimizer._rewrite_query(query, analysis)
        
        return {
            "success": True,
            "original_query": query,
            "optimized_query": optimized,
            "analysis": {
                "length": analysis["length"],
                "word_count": analysis["word_count"],
                "has_special_chars": analysis["has_special_chars"],
            },
            "suggestions": analysis["suggestions"],
            "optimization_applied": query != optimized,
        }
        
    except Exception as e:
        logger.error(f"Query analysis failed: {str(e)}")
        return {
            "success": False,
            "error": f"Analysis failed: {str(e)}",
        }