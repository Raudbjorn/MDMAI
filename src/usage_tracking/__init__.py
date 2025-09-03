"""
Comprehensive Usage Tracking and Cost Management Data Persistence System

This module provides a complete data persistence and state management solution
for usage tracking in the MDMAI system, featuring:

1. Multi-tier storage architecture (PostgreSQL + ChromaDB + JSON)
2. Advanced state synchronization with conflict resolution
3. High-performance caching and indexing
4. Comprehensive data retention and cleanup policies
5. Thread-safe concurrent access patterns
6. Automated backup and recovery mechanisms
7. Version migration strategies for schema evolution
8. Performance optimization and monitoring

Key Components:
- ChromaDB Extensions: Vector storage for usage analytics
- JSON Persistence: High-performance local file storage
- State Synchronization: Multi-backend consistency management
- Data Retention: Automated cleanup with compliance support
- Performance Optimization: Multi-tier caching and intelligent indexing
- Concurrent Access: Thread-safe patterns with optimistic locking
- Backup Recovery: Comprehensive backup and point-in-time recovery
- Migration Strategies: Version upgrades and schema evolution
"""

import asyncio
from datetime import datetime

from .chroma_extensions import (
    UsageTrackingChromaExtensions,
    UsageAnalyticsType,
    UsageVectorRecord
)
from .json_persistence import (
    JsonPersistenceManager,
    PersistenceConfig,
    CompressionType,
    PartitionStrategy
)
from .state_synchronization import (
    StateSynchronizationManager,
    SyncConfiguration,
    SyncStatus,
    StorageBackend,
    ConflictResolution
)
from .data_retention import (
    DataRetentionManager,
    RetentionRule,
    RetentionPolicy,
    DataCategory,
    RetentionAction
)
from .performance_optimization import (
    PerformanceOptimizer,
    MultiTierCache,
    IntelligentIndexManager,
    CacheLevel,
    CacheStrategy,
    IndexType
)
from .concurrent_access import (
    ConcurrencyManager,
    ConcurrencyPattern,
    ReadWriteLock,
    AsyncReadWriteLock,
    PartitionedLockManager,
    OptimisticLockManager
)
from .backup_recovery import (
    BackupRecoveryManager,
    BackupEngine,
    RecoveryEngine,
    BackupConfiguration,
    BackupType,
    BackupStrategy,
    RecoveryType
)
from .migration_strategies import (
    MigrationManager,
    MigrationDefinition,
    MigrationScript,
    MigrationType,
    MigrationStrategy,
    MigrationState,
    SchemaUpgradeMigration,
    DataTransformationMigration,
    StorageFormatMigration
)

__version__ = "1.0.0"
__author__ = "MDMAI Development Team"

__all__ = [
    # ChromaDB Extensions
    "UsageTrackingChromaExtensions",
    "UsageAnalyticsType",
    "UsageVectorRecord",
    
    # JSON Persistence
    "JsonPersistenceManager", 
    "PersistenceConfig",
    "CompressionType",
    "PartitionStrategy",
    
    # State Synchronization
    "StateSynchronizationManager",
    "SyncConfiguration", 
    "SyncStatus",
    "StorageBackend",
    "ConflictResolution",
    
    # Data Retention
    "DataRetentionManager",
    "RetentionRule",
    "RetentionPolicy",
    "DataCategory", 
    "RetentionAction",
    
    # Performance Optimization
    "PerformanceOptimizer",
    "MultiTierCache",
    "IntelligentIndexManager",
    "CacheLevel",
    "CacheStrategy",
    "IndexType",
    
    # Concurrent Access
    "ConcurrencyManager",
    "ConcurrencyPattern",
    "ReadWriteLock",
    "AsyncReadWriteLock", 
    "PartitionedLockManager",
    "OptimisticLockManager",
    
    # Backup & Recovery
    "BackupRecoveryManager",
    "BackupEngine",
    "RecoveryEngine",
    "BackupConfiguration",
    "BackupType",
    "BackupStrategy", 
    "RecoveryType",
    
    # Migration Strategies
    "MigrationManager",
    "MigrationDefinition",
    "MigrationScript",
    "MigrationType",
    "MigrationStrategy",
    "MigrationState",
    "SchemaUpgradeMigration",
    "DataTransformationMigration", 
    "StorageFormatMigration"
]


# Integration utilities
async def create_integrated_usage_system(
    postgres_persistence,
    storage_base_path: str = "./data/usage_tracking",
    cache_config: dict = None,
    concurrency_pattern: ConcurrencyPattern = ConcurrencyPattern.PARTITION_BASED,
    enable_all_features: bool = True
):
    """
    Create a fully integrated usage tracking persistence system.
    
    NOTE: This factory function has grown large and should be refactored into 
    separate builder classes in future versions for better maintainability.
    
    Args:
        postgres_persistence: PostgreSQL persistence layer
        storage_base_path: Base path for file storage
        cache_config: Cache configuration dictionary
        concurrency_pattern: Concurrency access pattern
        enable_all_features: Whether to enable all advanced features
        
    Returns:
        Dictionary containing all initialized components
    """
    import asyncio
    from pathlib import Path
    
    # Ensure storage directory exists
    storage_path = Path(storage_base_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize core components
    chroma_extensions = UsageTrackingChromaExtensions()
    
    json_persistence = JsonPersistenceManager(
        PersistenceConfig(
            base_path=str(storage_path / "json_files"),
            compression_type=CompressionType.GZIP,
            partition_strategy=PartitionStrategy.DAILY,
            enable_indexing=True,
            enable_backup=True
        )
    )
    
    # Initialize state synchronization
    sync_config = SyncConfiguration(
        real_time_sync=True,
        enable_conflict_detection=True,
        default_conflict_resolution=ConflictResolution.LATEST_WINS
    )
    
    sync_manager = StateSynchronizationManager(
        postgres_persistence,
        chroma_extensions,
        json_persistence,
        sync_config
    )
    
    # Initialize performance optimization
    performance_optimizer = PerformanceOptimizer(
        chroma_extensions,
        json_persistence,
        cache_config or {
            "l1_max_size": 1000,
            "l1_ttl_seconds": 300,
            "redis_url": None,  # Optional Redis URL
            "strategy": "adaptive"
        }
    )
    
    # Initialize concurrency management
    concurrency_manager = ConcurrencyManager(concurrency_pattern)
    
    components = {
        "chroma_extensions": chroma_extensions,
        "json_persistence": json_persistence,
        "sync_manager": sync_manager,
        "performance_optimizer": performance_optimizer,
        "concurrency_manager": concurrency_manager
    }
    
    if enable_all_features:
        # Initialize data retention
        data_retention_manager = DataRetentionManager(
            postgres_persistence,
            chroma_extensions,
            json_persistence,
            sync_manager
        )
        components["data_retention_manager"] = data_retention_manager
        
        # Initialize backup and recovery
        backup_recovery_manager = BackupRecoveryManager(
            postgres_persistence,
            chroma_extensions, 
            json_persistence,
            sync_manager,
            BackupConfiguration(
                backup_root_path=str(storage_path / "backups"),
                full_backup_interval_days=7,
                incremental_backup_interval_hours=6
            )
        )
        components["backup_recovery_manager"] = backup_recovery_manager
        
        # Initialize migration management
        migration_manager = MigrationManager(
            postgres_persistence,
            chroma_extensions,
            json_persistence, 
            backup_recovery_manager,
            str(storage_path / "migrations")
        )
        components["migration_manager"] = migration_manager
    
    # Start all components
    start_tasks = []
    
    if hasattr(json_persistence, 'start'):
        start_tasks.append(json_persistence.start())
    if hasattr(sync_manager, 'start'):
        start_tasks.append(sync_manager.start())
    if hasattr(performance_optimizer, 'start'):
        start_tasks.append(performance_optimizer.start())
    if hasattr(concurrency_manager, 'start'):
        start_tasks.append(concurrency_manager.start())
    
    if enable_all_features:
        if hasattr(components.get("data_retention_manager"), 'start'):
            start_tasks.append(components["data_retention_manager"].start())
        if hasattr(components.get("backup_recovery_manager"), 'start'):
            start_tasks.append(components["backup_recovery_manager"].start())
    
    # Start all components concurrently
    await asyncio.gather(*start_tasks)
    
    return components


class IntegratedUsageTrackingSystem:
    """
    Integrated usage tracking system that provides a unified interface
    to all persistence and state management components.
    """
    
    def __init__(self, components: dict):
        self.components = components
        self._running = False
    
    async def start(self):
        """Start all system components."""
        if self._running:
            return
        
        start_tasks = []
        for component in self.components.values():
            if hasattr(component, 'start'):
                start_tasks.append(component.start())
        
        await asyncio.gather(*start_tasks, return_exceptions=True)
        self._running = True
    
    async def stop(self):
        """Stop all system components."""
        if not self._running:
            return
        
        stop_tasks = []
        for component in self.components.values():
            if hasattr(component, 'stop'):
                stop_tasks.append(component.stop())
        
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        self._running = False
    
    async def store_usage_record(self, usage_record, user_id: str = None):
        """Store a usage record with full system integration."""
        # Use concurrency management
        concurrency_manager = self.components.get("concurrency_manager")
        if concurrency_manager:
            async with concurrency_manager.write_access(usage_record.request_id):
                # Store in JSON persistence
                json_persistence = self.components.get("json_persistence")
                if json_persistence:
                    await json_persistence.store_usage_record(usage_record)
                
                # Trigger synchronization
                sync_manager = self.components.get("sync_manager")
                if sync_manager:
                    from .state_synchronization import StorageBackend
                    await sync_manager.sync_usage_record(
                        usage_record, 
                        StorageBackend.JSON_FILES
                    )
        else:
            # Fallback without concurrency management
            json_persistence = self.components.get("json_persistence")
            if json_persistence:
                await json_persistence.store_usage_record(usage_record)
    
    async def get_usage_analytics(self, user_id: str, time_range: str = "7d"):
        """Get usage analytics with caching optimization."""
        performance_optimizer = self.components.get("performance_optimizer")
        chroma_extensions = self.components.get("chroma_extensions")
        
        if performance_optimizer and chroma_extensions:
            # Use performance optimization
            return await performance_optimizer.optimize_query(
                chroma_extensions.get_usage_insights,
                "usage_analytics",
                {"user_id": user_id, "time_range": time_range},
                cache_ttl=300,  # 5 minutes
                cache_tags={"analytics", f"user_{user_id}"}
            )
        elif chroma_extensions:
            # Direct query without optimization
            return await chroma_extensions.get_usage_insights(user_id, time_range)
        else:
            return {"error": "Analytics not available"}
    
    async def create_backup(self, description: str = ""):
        """Create a system backup."""
        backup_manager = self.components.get("backup_recovery_manager")
        if backup_manager:
            return await backup_manager.create_backup(
                description=description or f"System backup {datetime.now().isoformat()}"
            )
        else:
            raise ValueError("Backup system not available")
    
    async def execute_retention_policies(self):
        """Execute data retention policies."""
        retention_manager = self.components.get("data_retention_manager")
        if retention_manager:
            return await retention_manager.execute_retention_policies()
        else:
            raise ValueError("Data retention system not available")
    
    async def get_system_health(self):
        """Get comprehensive system health information."""
        health_data = {
            "timestamp": datetime.now().isoformat(),
            "running": self._running,
            "components": {}
        }
        
        for component_name, component in self.components.items():
            try:
                if hasattr(component, 'get_performance_report'):
                    report = await component.get_performance_report()
                    health_data["components"][component_name] = {
                        "status": "healthy",
                        "metrics": report
                    }
                elif hasattr(component, 'get_metrics'):
                    metrics = component.get_metrics()
                    health_data["components"][component_name] = {
                        "status": "healthy", 
                        "metrics": metrics
                    }
                else:
                    health_data["components"][component_name] = {
                        "status": "healthy",
                        "metrics": {}
                    }
            except Exception as e:
                health_data["components"][component_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return health_data
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        return False