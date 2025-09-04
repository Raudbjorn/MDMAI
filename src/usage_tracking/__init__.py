"""
Usage Tracking and Cost Management System

A comprehensive solution for tracking AI usage, managing costs, and optimizing performance
across multiple providers and storage backends.

Key Features:
- Multi-tier storage with ChromaDB and JSON backends
- Real-time usage tracking and cost monitoring
- User budgets and spending limits
- Advanced analytics with trend analysis
- Performance optimization and data retention
- Seamless data migration between storage systems

Quick Start:
    from src.usage_tracking import create_storage_manager, UsageEventType, ProviderType
    from decimal import Decimal
    
    # Initialize storage manager
    manager = create_storage_manager()
    await manager.initialize()
    
    # Track usage
    record_id = await manager.track_usage(
        user_id="user123",
        event_type=UsageEventType.API_CALL,
        provider=ProviderType.OPENAI,
        cost_usd=Decimal("0.002"),
        token_count=100,
        model_name="gpt-4"
    )
    
    # Get analytics
    analytics = await manager.generate_usage_analytics(
        user_id="user123",
        aggregation_type=TimeAggregation.DAILY
    )
    
    # Close when done
    await manager.close()

Components:
- storage_manager: Main orchestration layer
- storage: Data persistence (ChromaDB, JSON, Hybrid)
- analytics: Time-series analysis and insights
- performance: Optimization and monitoring
- migration: Data migration utilities
"""

from .storage_manager import UsageTrackingStorageManager, create_storage_manager
from .storage.models import (
    # Core models
    UsageRecord,
    UserProfile,
    UsagePattern,
    UsageMetrics,
    CostBreakdown,
    SpendingLimit,
    
    # Enums
    UsageEventType,
    ProviderType,
    StorageType,
    TimeAggregation,
    CostType,
    
    # Configuration models
    StorageSchema,
    ChromaDBConfig,
    JSONStorageConfig,
    HybridStorageConfig,
)

from .migration.data_migration import (
    MigrationEngine,
    MigrationType,
    MigrationStatus
)

# Version information
__version__ = "1.0.0"
__author__ = "MDMAI Development Team"
__license__ = "MIT"

# Export main classes and functions
__all__ = [
    # Main manager
    "UsageTrackingStorageManager",
    "create_storage_manager",
    
    # Core models
    "UsageRecord",
    "UserProfile", 
    "UsagePattern",
    "UsageMetrics",
    "CostBreakdown",
    "SpendingLimit",
    
    # Enums
    "UsageEventType",
    "ProviderType", 
    "StorageType",
    "TimeAggregation",
    "CostType",
    
    # Configuration
    "StorageSchema",
    "ChromaDBConfig",
    "JSONStorageConfig",
    "HybridStorageConfig",
    
    # Migration
    "MigrationEngine",
    "MigrationType",
    "MigrationStatus",
    
    # Version info
    "__version__",
]