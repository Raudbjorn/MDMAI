"""
Main storage manager for usage tracking and cost management.

This module provides a unified interface to all storage components:
- Hybrid storage orchestration
- Performance optimization
- Migration management
- Analytics processing
- Data retention policies
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .storage.models import (
    UsageRecord, UserProfile, UsagePattern, UsageMetrics,
    StorageSchema, HybridStorageConfig, ChromaDBConfig, JSONStorageConfig,
    StorageType, UsageEventType, ProviderType, TimeAggregation
)
from .storage.chromadb_storage import ChromaDBUsageStorage
from .storage.json_storage import JSONUsageStorage
from .storage.hybrid_storage import HybridUsageStorage
from .analytics.time_series import TimeSeriesAggregator, TrendAnalyzer, CostOptimizationAnalyzer
from .performance.optimization import PerformanceOptimizer
from .migration.data_migration import MigrationEngine, MigrationType

logger = logging.getLogger(__name__)


class UsageTrackingStorageManager:
    """
    Unified storage manager for usage tracking and cost management.
    
    This class orchestrates all storage, analytics, and optimization components
    to provide a comprehensive solution for usage data management.
    """
    
    def __init__(self, config: StorageSchema):
        self.config = config
        
        # Initialize storage components
        self.chromadb_storage = ChromaDBUsageStorage(config.chromadb)
        self.json_storage = JSONUsageStorage(config.json_storage)
        self.hybrid_storage = HybridUsageStorage(
            config.hybrid,
            self.chromadb_storage,
            self.json_storage
        )
        
        # Initialize analytics components
        self.time_series_aggregator = TimeSeriesAggregator()
        self.trend_analyzer = TrendAnalyzer()
        self.cost_optimizer = CostOptimizationAnalyzer()
        
        # Initialize optimization and migration
        self.performance_optimizer = PerformanceOptimizer()
        self.migration_engine = MigrationEngine()
        
        # Active storage backend
        self.active_storage = self.hybrid_storage
        
        # Performance tracking
        self.operation_metrics = {
            "reads": {"count": 0, "total_time": 0.0, "errors": 0},
            "writes": {"count": 0, "total_time": 0.0, "errors": 0},
            "analytics": {"count": 0, "total_time": 0.0, "errors": 0}
        }
        
        logger.info(
            "Storage manager initialized",
            storage_type="hybrid",
            cache_enabled=config.cache_enabled,
            performance_optimization=True
        )
    
    async def initialize(self) -> None:
        """Initialize all storage components."""
        try:
            # Start performance optimization engine
            self.performance_optimizer.start_optimization_engine(self.active_storage)
            
            logger.info("Storage manager initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize storage manager: {e}")
            raise
    
    async def close(self) -> None:
        """Close all storage components and clean up resources."""
        try:
            # Stop performance optimization
            self.performance_optimizer.stop_optimization_engine()
            
            # Close storage components
            await self.active_storage.close()
            
            logger.info("Storage manager closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing storage manager: {e}")
    
    # Core Usage Tracking Methods
    
    async def track_usage(
        self,
        user_id: str,
        event_type: UsageEventType,
        provider: ProviderType,
        cost_usd: Decimal = Decimal("0.00"),
        token_count: int = 0,
        model_name: Optional[str] = None,
        session_id: Optional[str] = None,
        context_id: Optional[str] = None,
        operation: Optional[str] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Track a usage event.
        
        Args:
            user_id: User identifier
            event_type: Type of usage event
            provider: AI provider used
            cost_usd: Cost in USD
            token_count: Number of tokens used
            model_name: Name of the model used
            session_id: Session identifier
            context_id: Context identifier
            operation: Specific operation performed
            success: Whether the operation was successful
            metadata: Additional metadata
            
        Returns:
            Record ID of the stored usage record
        """
        import time
        start_time = time.time()
        
        try:
            # Create usage record
            record = UsageRecord(
                user_id=user_id,
                event_type=event_type,
                provider=provider,
                cost_usd=cost_usd,
                token_count=token_count,
                model_name=model_name,
                session_id=session_id,
                context_id=context_id,
                operation=operation,
                success=success,
                metadata=metadata or {}
            )
            
            # Store the record
            record_id = await self.active_storage.store_usage_record(record)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.operation_metrics["writes"]["count"] += 1
            self.operation_metrics["writes"]["total_time"] += execution_time
            
            logger.debug(
                "Usage tracked",
                record_id=record_id,
                user_id=user_id,
                event_type=event_type.value,
                cost_usd=float(cost_usd),
                execution_time=execution_time
            )
            
            return record_id
            
        except Exception as e:
            self.operation_metrics["writes"]["errors"] += 1
            logger.error(f"Failed to track usage: {e}")
            raise
    
    async def get_user_usage(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[UsageEventType]] = None,
        providers: Optional[List[ProviderType]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get usage records for a specific user.
        
        Args:
            user_id: User identifier
            start_date: Start date for filtering
            end_date: End date for filtering
            event_types: Event types to filter by
            providers: Providers to filter by
            limit: Maximum number of records to return
            
        Returns:
            List of usage records
        """
        import time
        start_time = time.time()
        
        try:
            records = await self.active_storage.get_usage_records(
                user_id=user_id,
                event_types=event_types,
                providers=providers,
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
            
            # Update metrics
            execution_time = time.time() - start_time
            self.operation_metrics["reads"]["count"] += 1
            self.operation_metrics["reads"]["total_time"] += execution_time
            
            logger.debug(
                "User usage retrieved",
                user_id=user_id,
                record_count=len(records),
                execution_time=execution_time
            )
            
            return records
            
        except Exception as e:
            self.operation_metrics["reads"]["errors"] += 1
            logger.error(f"Failed to get user usage: {e}")
            raise
    
    async def search_usage(
        self,
        query: str,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on usage records.
        
        Args:
            query: Search query
            user_id: Optional user filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum results to return
            
        Returns:
            List of matching usage records with relevance scores
        """
        import time
        start_time = time.time()
        
        try:
            records = await self.active_storage.get_usage_records(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                semantic_query=query
            )
            
            # Update metrics
            execution_time = time.time() - start_time
            self.operation_metrics["reads"]["count"] += 1
            self.operation_metrics["reads"]["total_time"] += execution_time
            
            logger.debug(
                "Usage search completed",
                query=query,
                result_count=len(records),
                execution_time=execution_time
            )
            
            return records
            
        except Exception as e:
            self.operation_metrics["reads"]["errors"] += 1
            logger.error(f"Failed to search usage: {e}")
            raise
    
    # User Profile and Budget Management
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile with usage statistics and spending limits."""
        return await self.active_storage.get_user_profile(user_id)
    
    async def update_user_profile(self, profile: UserProfile) -> bool:
        """Update user profile."""
        return await self.active_storage.save_user_profile(profile)
    
    async def set_spending_limit(
        self,
        user_id: str,
        limit_type: str,  # "daily", "weekly", "monthly"
        amount_usd: Decimal,
        reset_current: bool = False
    ) -> bool:
        """
        Set spending limit for a user.
        
        Args:
            user_id: User identifier
            limit_type: Type of limit (daily, weekly, monthly)
            amount_usd: Limit amount in USD
            reset_current: Whether to reset current spending
            
        Returns:
            True if successful
        """
        try:
            from .storage.models import SpendingLimit
            
            # Create spending limit
            spending_limit = SpendingLimit(
                limit_type=limit_type,
                amount_usd=amount_usd,
                current_spent=Decimal("0.00") if reset_current else None
            )
            
            # Update via JSON storage (user profiles are always in JSON)
            success = await self.json_storage.update_spending_limit(
                user_id, limit_type, spending_limit
            )
            
            logger.info(
                "Spending limit updated",
                user_id=user_id,
                limit_type=limit_type,
                amount_usd=float(amount_usd)
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to set spending limit: {e}")
            return False
    
    async def check_spending_limits(
        self,
        user_id: str,
        additional_cost: Decimal = Decimal("0.00")
    ) -> Dict[str, Dict[str, Any]]:
        """
        Check if user is within spending limits.
        
        Args:
            user_id: User identifier
            additional_cost: Additional cost to check against limits
            
        Returns:
            Dict with limit status for each limit type
        """
        try:
            profile = await self.get_user_profile(user_id)
            if not profile or not profile.spending_limits:
                return {}
            
            results = {}
            for limit_type, limit in profile.spending_limits.items():
                would_exceed = (limit.current_spent + additional_cost) > limit.amount_usd
                
                results[limit_type] = {
                    "limit_amount": float(limit.amount_usd),
                    "current_spent": float(limit.current_spent),
                    "remaining": float(limit.remaining),
                    "percentage_used": limit.percentage_used,
                    "would_exceed": would_exceed,
                    "is_exceeded": limit.is_exceeded
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to check spending limits: {e}")
            return {}
    
    # Analytics and Insights
    
    async def generate_usage_analytics(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        aggregation_type: TimeAggregation = TimeAggregation.DAILY,
        group_by: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive usage analytics.
        
        Args:
            user_id: Optional user filter
            start_date: Analysis start date
            end_date: Analysis end date
            aggregation_type: Time aggregation level
            group_by: Fields to group by
            
        Returns:
            Comprehensive analytics data
        """
        import time
        start_time = time.time()
        
        try:
            # Set default date range if not provided
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            # Get usage records for analysis
            records_data = await self.get_user_usage(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                limit=10000
            )
            
            # Convert to UsageRecord objects
            records = []
            for record_data in records_data:
                try:
                    metadata = record_data.get("metadata", {})
                    record = UsageRecord(
                        record_id=metadata.get("record_id", "unknown"),
                        user_id=metadata["user_id"],
                        timestamp=datetime.fromisoformat(metadata["timestamp"]),
                        event_type=UsageEventType(metadata["event_type"]),
                        provider=ProviderType(metadata["provider"]),
                        cost_usd=Decimal(str(metadata.get("cost_usd", "0.00"))),
                        token_count=metadata.get("token_count", 0),
                        success=metadata.get("success", True)
                    )
                    records.append(record)
                except Exception as e:
                    logger.warning(f"Skipping invalid record: {e}")
                    continue
            
            # Time series aggregation
            aggregated_data = await self.time_series_aggregator.aggregate_usage_records(
                records=records,
                aggregation_type=aggregation_type,
                start_time=start_date,
                end_time=end_date,
                group_by=group_by
            )
            
            # Trend analysis
            analytics_results = {}
            for group_key, buckets in aggregated_data.items():
                trend_analysis = await self.trend_analyzer.detect_trends(
                    buckets=buckets,
                    metric="cost"
                )
                
                rollup_stats = await self.time_series_aggregator.compute_rollups(
                    buckets=buckets
                )
                
                analytics_results[group_key] = {
                    "trend_analysis": trend_analysis,
                    "rollup_statistics": rollup_stats,
                    "bucket_count": len(buckets),
                    "time_range": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat()
                    }
                }
            
            # Cost optimization analysis
            cost_optimization = await self.cost_optimizer.analyze_cost_opportunities(records)
            
            # Compile final results
            final_results = {
                "analysis_metadata": {
                    "user_id": user_id,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "aggregation_type": aggregation_type.value,
                    "records_analyzed": len(records),
                    "execution_time_seconds": time.time() - start_time
                },
                "group_analytics": analytics_results,
                "cost_optimization": cost_optimization,
                "summary": {
                    "total_cost": sum(float(r.cost_usd) for r in records),
                    "total_tokens": sum(r.token_count for r in records),
                    "total_requests": len(records),
                    "success_rate": sum(1 for r in records if r.success) / max(len(records), 1),
                    "unique_providers": len(set(r.provider.value for r in records)),
                    "date_range_days": (end_date - start_date).days
                }
            }
            
            # Update metrics
            execution_time = time.time() - start_time
            self.operation_metrics["analytics"]["count"] += 1
            self.operation_metrics["analytics"]["total_time"] += execution_time
            
            logger.info(
                "Analytics generated",
                user_id=user_id,
                records_analyzed=len(records),
                execution_time=execution_time
            )
            
            return final_results
            
        except Exception as e:
            self.operation_metrics["analytics"]["errors"] += 1
            logger.error(f"Failed to generate analytics: {e}")
            raise
    
    # Data Management
    
    async def cleanup_old_data(self, retention_days: int = 90) -> Dict[str, Any]:
        """Clean up old data according to retention policies."""
        return await self.active_storage.cleanup_old_data(retention_days)
    
    async def create_backup(self) -> str:
        """Create backup of all usage tracking data."""
        return await self.json_storage.create_backup()
    
    async def migrate_data(
        self,
        migration_type: MigrationType,
        source_type: StorageType,
        target_type: StorageType,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Migrate data between storage systems.
        
        Args:
            migration_type: Type of migration
            source_type: Source storage type
            target_type: Target storage type
            filters: Optional filters for migration
            
        Returns:
            Migration plan ID
        """
        try:
            # Determine source and target storage
            source_storage = self._get_storage_by_type(source_type)
            target_storage = self._get_storage_by_type(target_type)
            
            # Create migration plan
            plan = await self.migration_engine.create_migration_plan(
                migration_type=migration_type,
                source_storage=source_storage,
                target_storage=target_storage,
                filters=filters
            )
            
            logger.info(
                f"Migration plan created: {plan.plan_id}",
                migration_type=migration_type.value,
                source_type=source_type.value,
                target_type=target_type.value
            )
            
            return plan.plan_id
            
        except Exception as e:
            logger.error(f"Failed to create migration plan: {e}")
            raise
    
    async def execute_migration(self, plan_id: str) -> bool:
        """Execute a migration plan."""
        try:
            plan = self.migration_engine.migration_history.get(plan_id)
            if not plan:
                raise ValueError(f"Migration plan not found: {plan_id}")
            
            # Determine storage types from plan
            if plan.migration_type == MigrationType.JSON_TO_CHROMADB:
                source_storage = self.json_storage
                target_storage = self.chromadb_storage
            elif plan.migration_type == MigrationType.CHROMADB_TO_JSON:
                source_storage = self.chromadb_storage
                target_storage = self.json_storage
            else:
                source_storage = self.active_storage
                target_storage = self.active_storage
            
            # Execute migration
            success = await self.migration_engine.execute_migration(
                plan_id=plan_id,
                source_storage=source_storage,
                target_storage=target_storage
            )
            
            logger.info(f"Migration execution completed: {plan_id}", success=success)
            return success
            
        except Exception as e:
            logger.error(f"Failed to execute migration: {e}")
            return False
    
    def _get_storage_by_type(self, storage_type: StorageType):
        """Get storage instance by type."""
        if storage_type == StorageType.CHROMADB:
            return self.chromadb_storage
        elif storage_type == StorageType.JSON:
            return self.json_storage
        elif storage_type == StorageType.HYBRID:
            return self.hybrid_storage
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")
    
    # Status and Monitoring
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all storage components."""
        try:
            status = {
                "timestamp": datetime.utcnow().isoformat(),
                "active_storage": "hybrid",
                "operation_metrics": self._calculate_operation_metrics(),
                "storage_stats": await self.active_storage.get_comprehensive_stats(),
                "performance_stats": self.performance_optimizer.get_comprehensive_stats(),
                "migration_stats": self.migration_engine.get_migration_stats()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive status: {e}")
            return {"error": str(e)}
    
    def _calculate_operation_metrics(self) -> Dict[str, Any]:
        """Calculate operation performance metrics."""
        metrics = {}
        
        for operation, stats in self.operation_metrics.items():
            if stats["count"] > 0:
                metrics[operation] = {
                    "total_operations": stats["count"],
                    "total_time_seconds": stats["total_time"],
                    "average_time_seconds": stats["total_time"] / stats["count"],
                    "error_count": stats["errors"],
                    "error_rate": stats["errors"] / stats["count"],
                    "operations_per_second": stats["count"] / max(stats["total_time"], 0.001)
                }
            else:
                metrics[operation] = {
                    "total_operations": 0,
                    "average_time_seconds": 0,
                    "error_rate": 0,
                    "operations_per_second": 0
                }
        
        return metrics
    
    async def run_performance_optimization(self) -> Dict[str, Any]:
        """Run comprehensive performance optimization."""
        return await self.performance_optimizer.run_full_optimization(self.active_storage)


# Factory function for easy initialization

def create_storage_manager(
    config: Optional[StorageSchema] = None,
    base_path: str = "./data/usage_tracking",
    chromadb_path: str = "./data/chromadb"
) -> UsageTrackingStorageManager:
    """
    Create a storage manager with default or custom configuration.
    
    Args:
        config: Optional storage schema configuration
        base_path: Base path for JSON storage
        chromadb_path: Path for ChromaDB persistence
        
    Returns:
        Configured storage manager instance
    """
    if config is None:
        # Create default configuration
        config = StorageSchema(
            chromadb=ChromaDBConfig(persist_directory=chromadb_path),
            json_storage=JSONStorageConfig(base_path=base_path),
            hybrid=HybridStorageConfig()
        )
    
    return UsageTrackingStorageManager(config)