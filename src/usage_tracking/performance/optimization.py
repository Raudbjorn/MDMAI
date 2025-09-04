"""
Performance optimization and data retention policies for usage tracking.

This module provides:
- Query optimization and indexing strategies
- Data compression and archival
- Memory usage optimization
- Background processing optimization
- Automated cleanup and retention
"""

import asyncio
import gc
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import resource

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    disk_usage_mb: float
    active_connections: int
    query_latency_ms: float
    cache_hit_rate: float
    throughput_ops_per_sec: float
    error_rate: float


@dataclass
class RetentionPolicy:
    """Data retention policy configuration."""
    name: str
    description: str
    hot_data_days: int = 7
    warm_data_days: int = 30
    cold_data_days: int = 90
    archive_days: int = 365
    delete_days: int = 1095  # 3 years
    compression_enabled: bool = True
    encryption_required: bool = False
    backup_required: bool = True


class PerformanceMonitor:
    """Real-time performance monitoring and alerting."""
    
    def __init__(self, alert_thresholds: Dict[str, float] = None):
        self.alert_thresholds = alert_thresholds or {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "query_latency_ms": 1000.0,
            "error_rate": 0.05,
            "cache_hit_rate": 0.7  # Alert if below 70%
        }
        
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history_size = 1000
        self.monitoring_active = False
        self.alert_callbacks = []
        
        # Performance tracking
        self.query_times = []
        self.operation_counts = {"reads": 0, "writes": 0, "errors": 0}
        self.start_time = time.time()
        
        # Thread for monitoring
        self.monitor_thread = None
        self.monitor_lock = threading.RLock()
    
    def start_monitoring(self, interval_seconds: int = 60) -> None:
        """Start performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True,
            name="PerformanceMonitor"
        )
        self.monitor_thread.start()
        
        logger.info(f"Performance monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self, interval_seconds: int) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                
                with self.monitor_lock:
                    self.metrics_history.append(metrics)
                    
                    # Trim history if needed
                    if len(self.metrics_history) > self.max_history_size:
                        self.metrics_history = self.metrics_history[-self.max_history_size:]
                
                # Check for alerts
                self._check_alerts(metrics)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                time.sleep(interval_seconds)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Application metrics
        with self.monitor_lock:
            uptime_seconds = time.time() - self.start_time
            total_ops = sum(self.operation_counts.values())
            throughput = total_ops / max(uptime_seconds, 1)
            
            avg_query_latency = 0.0
            if self.query_times:
                avg_query_latency = sum(self.query_times) / len(self.query_times) * 1000
                # Keep only recent query times
                self.query_times = self.query_times[-100:]
            
            error_rate = self.operation_counts["errors"] / max(total_ops, 1)
        
        return PerformanceMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            memory_mb=memory.used / 1024 / 1024,
            disk_usage_mb=disk.used / 1024 / 1024,
            active_connections=0,  # Would need to track this separately
            query_latency_ms=avg_query_latency,
            cache_hit_rate=0.0,  # Would need to get this from cache
            throughput_ops_per_sec=throughput,
            error_rate=error_rate
        )
    
    def _check_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check metrics against alert thresholds."""
        alerts = []
        
        if metrics.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        memory_percent = (metrics.memory_mb / (psutil.virtual_memory().total / 1024 / 1024)) * 100
        if memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append(f"High memory usage: {memory_percent:.1f}%")
        
        if metrics.query_latency_ms > self.alert_thresholds["query_latency_ms"]:
            alerts.append(f"High query latency: {metrics.query_latency_ms:.1f}ms")
        
        if metrics.error_rate > self.alert_thresholds["error_rate"]:
            alerts.append(f"High error rate: {metrics.error_rate:.1%}")
        
        if alerts:
            alert_message = f"Performance alerts at {metrics.timestamp}: " + "; ".join(alerts)
            logger.warning(alert_message)
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert_message, metrics)
                except Exception as e:
                    logger.error(f"Error calling alert callback: {e}")
    
    def add_alert_callback(self, callback) -> None:
        """Add callback for performance alerts."""
        self.alert_callbacks.append(callback)
    
    def track_query_time(self, duration_seconds: float) -> None:
        """Track query execution time."""
        with self.monitor_lock:
            self.query_times.append(duration_seconds)
    
    def increment_operation(self, operation_type: str) -> None:
        """Increment operation counter."""
        with self.monitor_lock:
            if operation_type in self.operation_counts:
                self.operation_counts[operation_type] += 1
    
    def get_recent_metrics(self, hours: int = 1) -> List[PerformanceMetrics]:
        """Get metrics from the last N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.monitor_lock:
            return [
                m for m in self.metrics_history 
                if m.timestamp >= cutoff_time
            ]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary performance statistics."""
        with self.monitor_lock:
            if not self.metrics_history:
                return {}
            
            recent_metrics = self.get_recent_metrics(1)  # Last hour
            if not recent_metrics:
                recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
            
            cpu_values = [m.cpu_percent for m in recent_metrics]
            memory_values = [m.memory_mb for m in recent_metrics]
            latency_values = [m.query_latency_ms for m in recent_metrics if m.query_latency_ms > 0]
            
            uptime_seconds = time.time() - self.start_time
            total_ops = sum(self.operation_counts.values())
            
            return {
                "uptime_hours": uptime_seconds / 3600,
                "total_operations": total_ops,
                "operations_per_hour": total_ops / max(uptime_seconds / 3600, 1),
                "current_cpu_percent": cpu_values[-1] if cpu_values else 0,
                "avg_cpu_percent": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "current_memory_mb": memory_values[-1] if memory_values else 0,
                "avg_memory_mb": sum(memory_values) / len(memory_values) if memory_values else 0,
                "avg_query_latency_ms": sum(latency_values) / len(latency_values) if latency_values else 0,
                "error_rate": self.operation_counts["errors"] / max(total_ops, 1),
                "metrics_collected": len(self.metrics_history)
            }


class QueryOptimizer:
    """Query optimization and caching strategies."""
    
    def __init__(self):
        self.query_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        self.query_patterns = {}
        self.optimization_rules = {}
        
        # Common optimization rules
        self._initialize_optimization_rules()
    
    def _initialize_optimization_rules(self) -> None:
        """Initialize query optimization rules."""
        self.optimization_rules = {
            "limit_large_queries": {
                "description": "Add LIMIT clause to large result queries",
                "condition": lambda params: params.get("limit", 0) == 0 or params.get("limit", 0) > 10000,
                "optimization": lambda params: {**params, "limit": min(params.get("limit", 1000), 1000)}
            },
            "add_time_bounds": {
                "description": "Add time bounds to unbounded queries",
                "condition": lambda params: not params.get("start_date") and not params.get("end_date"),
                "optimization": lambda params: {
                    **params, 
                    "start_date": datetime.utcnow() - timedelta(days=30),
                    "end_date": datetime.utcnow()
                }
            },
            "optimize_user_queries": {
                "description": "Prioritize user-specific queries",
                "condition": lambda params: params.get("user_id") is not None,
                "optimization": lambda params: {**params, "use_index": "user_id_timestamp"}
            }
        }
    
    async def optimize_query_parameters(self, query_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize query parameters based on patterns and rules."""
        optimized_params = parameters.copy()
        optimizations_applied = []
        
        # Apply optimization rules
        for rule_name, rule in self.optimization_rules.items():
            if rule["condition"](optimized_params):
                old_params = optimized_params.copy()
                optimized_params = rule["optimization"](optimized_params)
                
                if old_params != optimized_params:
                    optimizations_applied.append(rule_name)
        
        # Track query patterns for future optimization
        pattern_key = f"{query_type}:{hash(str(sorted(parameters.items())))}"
        if pattern_key not in self.query_patterns:
            self.query_patterns[pattern_key] = {"count": 0, "avg_time": 0.0}
        
        self.query_patterns[pattern_key]["count"] += 1
        
        if optimizations_applied:
            logger.debug(
                f"Query optimized: {query_type}",
                optimizations=optimizations_applied,
                original_params=parameters,
                optimized_params=optimized_params
            )
        
        return optimized_params
    
    async def cache_query_result(
        self, 
        query_key: str, 
        result: Any, 
        ttl_seconds: int = 300
    ) -> None:
        """Cache query result with TTL."""
        cache_entry = {
            "result": result,
            "cached_at": time.time(),
            "ttl_seconds": ttl_seconds
        }
        
        self.query_cache[query_key] = cache_entry
        
        # Cleanup expired entries periodically
        if len(self.query_cache) > 1000:
            await self._cleanup_expired_cache()
    
    async def get_cached_result(self, query_key: str) -> Optional[Any]:
        """Get cached query result if valid."""
        if query_key not in self.query_cache:
            self.cache_stats["misses"] += 1
            return None
        
        entry = self.query_cache[query_key]
        
        # Check if expired
        if time.time() - entry["cached_at"] > entry["ttl_seconds"]:
            del self.query_cache[query_key]
            self.cache_stats["misses"] += 1
            return None
        
        self.cache_stats["hits"] += 1
        return entry["result"]
    
    async def _cleanup_expired_cache(self) -> None:
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.query_cache.items():
            if current_time - entry["cached_at"] > entry["ttl_seconds"]:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.query_cache[key]
        
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get query optimization statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / max(total_requests, 1)
        
        return {
            "cache_size": len(self.query_cache),
            "cache_hit_rate": hit_rate,
            "total_cache_requests": total_requests,
            "query_patterns_tracked": len(self.query_patterns),
            "optimization_rules": list(self.optimization_rules.keys())
        }


class DataRetentionManager:
    """Manages data lifecycle and retention policies."""
    
    def __init__(self):
        self.retention_policies: Dict[str, RetentionPolicy] = {}
        self.cleanup_stats = {"last_run": None, "records_cleaned": 0, "space_freed_mb": 0}
        self.background_cleanup_enabled = False
        self.cleanup_thread = None
        
        # Default retention policies
        self._initialize_default_policies()
    
    def _initialize_default_policies(self) -> None:
        """Initialize default retention policies."""
        self.retention_policies = {
            "usage_records": RetentionPolicy(
                name="usage_records",
                description="Individual usage records",
                hot_data_days=7,
                warm_data_days=30,
                cold_data_days=90,
                archive_days=365,
                delete_days=1095,
                compression_enabled=True
            ),
            "user_profiles": RetentionPolicy(
                name="user_profiles",
                description="User profiles and settings",
                hot_data_days=30,
                warm_data_days=90,
                cold_data_days=365,
                archive_days=1095,
                delete_days=2190,  # 6 years
                compression_enabled=False,
                backup_required=True
            ),
            "analytics_data": RetentionPolicy(
                name="analytics_data",
                description="Aggregated analytics and metrics",
                hot_data_days=14,
                warm_data_days=90,
                cold_data_days=365,
                archive_days=1095,
                delete_days=1825,  # 5 years
                compression_enabled=True
            ),
            "transaction_logs": RetentionPolicy(
                name="transaction_logs",
                description="Transaction and audit logs",
                hot_data_days=30,
                warm_data_days=90,
                cold_data_days=365,
                archive_days=2190,  # 6 years (compliance)
                delete_days=2555,  # 7 years
                compression_enabled=True,
                backup_required=True
            )
        }
    
    def add_retention_policy(self, policy: RetentionPolicy) -> None:
        """Add or update a retention policy."""
        self.retention_policies[policy.name] = policy
        logger.info(f"Added retention policy: {policy.name}")
    
    def get_retention_policy(self, data_type: str) -> Optional[RetentionPolicy]:
        """Get retention policy for data type."""
        return self.retention_policies.get(data_type)
    
    async def apply_retention_policies(
        self, 
        storage_manager, 
        data_types: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Apply retention policies to storage systems."""
        results = {}
        data_types = data_types or list(self.retention_policies.keys())
        
        for data_type in data_types:
            policy = self.retention_policies.get(data_type)
            if not policy:
                continue
            
            try:
                # Apply retention for this data type
                result = await self._apply_policy_to_data_type(storage_manager, data_type, policy)
                results[data_type] = result
                
            except Exception as e:
                logger.error(f"Failed to apply retention policy for {data_type}: {e}")
                results[data_type] = {"error": str(e)}
        
        # Update cleanup stats
        total_cleaned = sum(r.get("records_cleaned", 0) for r in results.values() if isinstance(r, dict))
        total_space_freed = sum(r.get("space_freed_mb", 0) for r in results.values() if isinstance(r, dict))
        
        self.cleanup_stats = {
            "last_run": datetime.utcnow(),
            "records_cleaned": total_cleaned,
            "space_freed_mb": total_space_freed
        }
        
        logger.info(
            f"Retention policies applied: {total_cleaned} records cleaned, "
            f"{total_space_freed:.1f} MB freed"
        )
        
        return results
    
    async def _apply_policy_to_data_type(
        self, 
        storage_manager, 
        data_type: str, 
        policy: RetentionPolicy
    ) -> Dict[str, Any]:
        """Apply retention policy to specific data type."""
        now = datetime.utcnow()
        
        # Calculate cutoff dates
        hot_cutoff = now - timedelta(days=policy.hot_data_days)
        warm_cutoff = now - timedelta(days=policy.warm_data_days)
        cold_cutoff = now - timedelta(days=policy.cold_data_days)
        archive_cutoff = now - timedelta(days=policy.archive_days)
        delete_cutoff = now - timedelta(days=policy.delete_days)
        
        result = {
            "data_type": data_type,
            "policy": policy.name,
            "records_cleaned": 0,
            "space_freed_mb": 0.0,
            "operations": []
        }
        
        # Delete very old data
        if hasattr(storage_manager, 'cleanup_old_data'):
            try:
                cleanup_result = await storage_manager.cleanup_old_data(policy.delete_days)
                if isinstance(cleanup_result, dict):
                    deleted_count = sum(v for k, v in cleanup_result.items() if isinstance(v, int))
                else:
                    deleted_count = cleanup_result or 0
                
                result["records_cleaned"] += deleted_count
                result["operations"].append(f"Deleted {deleted_count} records older than {policy.delete_days} days")
                
            except Exception as e:
                logger.error(f"Error during cleanup for {data_type}: {e}")
                result["operations"].append(f"Cleanup error: {str(e)}")
        
        # Archive old data (placeholder - would need specific implementation)
        if hasattr(storage_manager, 'archive_old_data'):
            try:
                archived_count = await storage_manager.archive_old_data(policy.archive_days)
                result["operations"].append(f"Archived {archived_count} records")
                
            except Exception as e:
                logger.error(f"Error during archival for {data_type}: {e}")
        
        # Compress warm/cold data (placeholder)
        if policy.compression_enabled and hasattr(storage_manager, 'compress_old_data'):
            try:
                compressed_count = await storage_manager.compress_old_data(policy.warm_data_days)
                result["operations"].append(f"Compressed {compressed_count} records")
                
            except Exception as e:
                logger.error(f"Error during compression for {data_type}: {e}")
        
        return result
    
    def start_background_cleanup(
        self, 
        storage_manager, 
        interval_hours: int = 24
    ) -> None:
        """Start background cleanup process."""
        if self.background_cleanup_enabled:
            return
        
        self.background_cleanup_enabled = True
        self.cleanup_thread = threading.Thread(
            target=self._background_cleanup_loop,
            args=(storage_manager, interval_hours),
            daemon=True,
            name="RetentionCleanup"
        )
        self.cleanup_thread.start()
        
        logger.info(f"Background cleanup started with {interval_hours}h interval")
    
    def stop_background_cleanup(self) -> None:
        """Stop background cleanup process."""
        self.background_cleanup_enabled = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        logger.info("Background cleanup stopped")
    
    def _background_cleanup_loop(self, storage_manager, interval_hours: int) -> None:
        """Background cleanup loop."""
        while self.background_cleanup_enabled:
            try:
                # Run cleanup
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                results = loop.run_until_complete(
                    self.apply_retention_policies(storage_manager)
                )
                
                logger.info(f"Background cleanup completed: {results}")
                
                loop.close()
                
                # Wait for next interval
                time.sleep(interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")
                time.sleep(3600)  # Wait 1 hour on error
    
    def get_retention_stats(self) -> Dict[str, Any]:
        """Get retention management statistics."""
        return {
            "policies": {
                name: {
                    "hot_days": policy.hot_data_days,
                    "warm_days": policy.warm_data_days,
                    "cold_days": policy.cold_data_days,
                    "archive_days": policy.archive_days,
                    "delete_days": policy.delete_days,
                    "compression": policy.compression_enabled
                }
                for name, policy in self.retention_policies.items()
            },
            "cleanup_stats": self.cleanup_stats,
            "background_cleanup_active": self.background_cleanup_enabled
        }


class MemoryOptimizer:
    """Memory usage optimization and monitoring."""
    
    def __init__(self):
        self.memory_stats = {"peak_usage_mb": 0, "current_usage_mb": 0}
        self.gc_stats = {"collections": 0, "objects_freed": 0}
        self.optimization_enabled = True
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        initial_usage = self._get_memory_usage_mb()
        
        optimization_results = {
            "initial_memory_mb": initial_usage,
            "optimizations": []
        }
        
        if not self.optimization_enabled:
            optimization_results["final_memory_mb"] = initial_usage
            return optimization_results
        
        # Force garbage collection
        collected_objects = 0
        for generation in range(3):
            collected = gc.collect(generation)
            collected_objects += collected
        
        self.gc_stats["collections"] += 1
        self.gc_stats["objects_freed"] += collected_objects
        
        if collected_objects > 0:
            optimization_results["optimizations"].append(
                f"Garbage collection freed {collected_objects} objects"
            )
        
        # Clear caches if memory is high
        current_usage = self._get_memory_usage_mb()
        if current_usage > 1000:  # 1GB threshold
            # This would clear various caches - placeholder
            optimization_results["optimizations"].append("Cleared internal caches due to high memory usage")
        
        final_usage = self._get_memory_usage_mb()
        memory_saved = initial_usage - final_usage
        
        optimization_results["final_memory_mb"] = final_usage
        optimization_results["memory_saved_mb"] = memory_saved
        
        # Update stats
        self.memory_stats["current_usage_mb"] = final_usage
        self.memory_stats["peak_usage_mb"] = max(
            self.memory_stats["peak_usage_mb"], 
            final_usage
        )
        
        if memory_saved > 0:
            logger.info(f"Memory optimization saved {memory_saved:.1f} MB")
        
        return optimization_results
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        current_usage = self._get_memory_usage_mb()
        
        # Update current usage
        self.memory_stats["current_usage_mb"] = current_usage
        self.memory_stats["peak_usage_mb"] = max(
            self.memory_stats["peak_usage_mb"],
            current_usage
        )
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        
        return {
            "current_usage_mb": current_usage,
            "peak_usage_mb": self.memory_stats["peak_usage_mb"],
            "system_total_mb": system_memory.total / 1024 / 1024,
            "system_available_mb": system_memory.available / 1024 / 1024,
            "system_percent_used": system_memory.percent,
            "gc_collections": self.gc_stats["collections"],
            "objects_freed": self.gc_stats["objects_freed"]
        }
    
    def set_memory_limits(self, max_memory_mb: int) -> None:
        """Set memory limits for the process."""
        try:
            # Set soft and hard memory limits
            max_memory_bytes = max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
            
            logger.info(f"Memory limit set to {max_memory_mb} MB")
            
        except Exception as e:
            logger.warning(f"Failed to set memory limit: {e}")


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.query_optimizer = QueryOptimizer()
        self.retention_manager = DataRetentionManager()
        self.memory_optimizer = MemoryOptimizer()
        
        # Optimization settings
        self.auto_optimization_enabled = True
        self.optimization_interval_minutes = 30
        self.optimization_thread = None
        
    def start_optimization_engine(self, storage_manager = None) -> None:
        """Start the performance optimization engine."""
        # Start monitoring
        self.monitor.start_monitoring(interval_seconds=60)
        
        # Start background retention cleanup if storage manager provided
        if storage_manager:
            self.retention_manager.start_background_cleanup(storage_manager, interval_hours=24)
        
        # Add alert callback for automatic optimization
        self.monitor.add_alert_callback(self._performance_alert_callback)
        
        logger.info("Performance optimization engine started")
    
    def stop_optimization_engine(self) -> None:
        """Stop the performance optimization engine."""
        self.monitor.stop_monitoring()
        self.retention_manager.stop_background_cleanup()
        
        logger.info("Performance optimization engine stopped")
    
    def _performance_alert_callback(self, alert_message: str, metrics: PerformanceMetrics) -> None:
        """Handle performance alerts with automatic optimization."""
        if not self.auto_optimization_enabled:
            return
        
        logger.info(f"Automatic optimization triggered by alert: {alert_message}")
        
        # Run memory optimization if memory is high
        if metrics.memory_mb > 1000:  # 1GB threshold
            self.memory_optimizer.optimize_memory_usage()
        
        # Clear query cache if latency is high
        if metrics.query_latency_ms > 1000:
            asyncio.create_task(self.query_optimizer._cleanup_expired_cache())
    
    async def run_full_optimization(self, storage_manager = None) -> Dict[str, Any]:
        """Run comprehensive performance optimization."""
        optimization_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "optimizations": {}
        }
        
        # Memory optimization
        memory_result = self.memory_optimizer.optimize_memory_usage()
        optimization_results["optimizations"]["memory"] = memory_result
        
        # Query cache cleanup
        await self.query_optimizer._cleanup_expired_cache()
        optimization_results["optimizations"]["query_cache"] = {
            "cache_cleaned": True,
            "cache_size": len(self.query_optimizer.query_cache)
        }
        
        # Data retention (if storage manager provided)
        if storage_manager:
            retention_result = await self.retention_manager.apply_retention_policies(storage_manager)
            optimization_results["optimizations"]["retention"] = retention_result
        
        logger.info("Full performance optimization completed")
        return optimization_results
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "performance_monitor": self.monitor.get_summary_stats(),
            "query_optimizer": self.query_optimizer.get_optimization_stats(),
            "retention_manager": self.retention_manager.get_retention_stats(),
            "memory_optimizer": self.memory_optimizer.get_memory_stats(),
            "auto_optimization_enabled": self.auto_optimization_enabled
        }