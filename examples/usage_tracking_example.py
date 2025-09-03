"""
Usage Tracking and Cost Management Example

This example demonstrates the complete usage tracking system including:
- Storage initialization and configuration
- Usage tracking for different providers
- User profile and budget management
- Analytics generation and cost optimization
- Data migration between storage backends
- Performance monitoring and optimization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the usage tracking system
from src.usage_tracking import (
    create_storage_manager,
    UsageEventType,
    ProviderType, 
    TimeAggregation,
    MigrationType,
    StorageType,
    StorageSchema,
    ChromaDBConfig,
    JSONStorageConfig,
    HybridStorageConfig
)

logger = logging.getLogger(__name__)


async def basic_usage_example():
    """Demonstrate basic usage tracking functionality."""
    logger.info("=== Basic Usage Tracking Example ===")
    
    # Create storage manager with default configuration
    manager = create_storage_manager(
        base_path="./data/usage_tracking_example",
        chromadb_path="./data/chromadb_example"
    )
    
    try:
        # Initialize the storage manager
        await manager.initialize()
        
        # Track some usage events
        usage_records = []
        
        # Example: OpenAI GPT-4 usage
        record_id = await manager.track_usage(
            user_id="user_alice",
            event_type=UsageEventType.API_CALL,
            provider=ProviderType.OPENAI,
            cost_usd=Decimal("0.006"),  # $0.006 for 200 tokens
            token_count=200,
            input_tokens=150,
            output_tokens=50,
            model_name="gpt-4",
            session_id="session_001",
            operation="text_generation",
            success=True,
            metadata={"temperature": 0.7, "max_tokens": 100}
        )
        usage_records.append(record_id)
        logger.info(f"Tracked OpenAI usage: {record_id}")
        
        # Example: Anthropic Claude usage
        record_id = await manager.track_usage(
            user_id="user_alice",
            event_type=UsageEventType.API_CALL,
            provider=ProviderType.ANTHROPIC,
            cost_usd=Decimal("0.004"),  # $0.004 for 180 tokens
            token_count=180,
            input_tokens=120,
            output_tokens=60,
            model_name="claude-3-sonnet",
            session_id="session_001",
            operation="text_generation",
            success=True
        )
        usage_records.append(record_id)
        logger.info(f"Tracked Anthropic usage: {record_id}")
        
        # Example: Embedding generation
        record_id = await manager.track_usage(
            user_id="user_alice",
            event_type=UsageEventType.EMBEDDING_GENERATION,
            provider=ProviderType.OPENAI,
            cost_usd=Decimal("0.0001"),  # $0.0001 for embedding
            token_count=50,
            model_name="text-embedding-ada-002",
            operation="document_embedding",
            success=True
        )
        usage_records.append(record_id)
        logger.info(f"Tracked embedding usage: {record_id}")
        
        # Example: Failed request
        record_id = await manager.track_usage(
            user_id="user_alice",
            event_type=UsageEventType.API_CALL,
            provider=ProviderType.OPENAI,
            cost_usd=Decimal("0.000"),  # No cost for failed request
            token_count=0,
            model_name="gpt-4",
            operation="text_generation",
            success=False,
            metadata={"error": "rate_limit_exceeded"}
        )
        usage_records.append(record_id)
        logger.info(f"Tracked failed usage: {record_id}")
        
        # Get user usage history
        usage_history = await manager.get_user_usage(
            user_id="user_alice",
            start_date=datetime.utcnow() - timedelta(hours=1)
        )
        logger.info(f"Retrieved {len(usage_history)} usage records for user_alice")
        
        # Perform semantic search
        search_results = await manager.search_usage(
            query="OpenAI text generation",
            user_id="user_alice"
        )
        logger.info(f"Semantic search returned {len(search_results)} results")
        
        logger.info("Basic usage tracking completed successfully")
        
    finally:
        await manager.close()


async def budget_management_example():
    """Demonstrate user budget and spending limit management."""
    logger.info("=== Budget Management Example ===")
    
    manager = create_storage_manager()
    
    try:
        await manager.initialize()
        
        # Set spending limits for a user
        user_id = "user_bob"
        
        # Daily limit: $5.00
        success = await manager.set_spending_limit(
            user_id=user_id,
            limit_type="daily",
            amount_usd=Decimal("5.00")
        )
        logger.info(f"Set daily spending limit: {success}")
        
        # Weekly limit: $25.00  
        success = await manager.set_spending_limit(
            user_id=user_id,
            limit_type="weekly", 
            amount_usd=Decimal("25.00")
        )
        logger.info(f"Set weekly spending limit: {success}")
        
        # Monthly limit: $100.00
        success = await manager.set_spending_limit(
            user_id=user_id,
            limit_type="monthly",
            amount_usd=Decimal("100.00")
        )
        logger.info(f"Set monthly spending limit: {success}")
        
        # Simulate some usage to test limits
        for i in range(5):
            await manager.track_usage(
                user_id=user_id,
                event_type=UsageEventType.API_CALL,
                provider=ProviderType.OPENAI,
                cost_usd=Decimal("0.50"),  # $0.50 per call
                token_count=150,
                model_name="gpt-3.5-turbo"
            )
        
        # Check spending status
        spending_status = await manager.check_spending_limits(
            user_id=user_id,
            additional_cost=Decimal("1.00")  # Proposed additional $1.00
        )
        
        logger.info("Spending limit status:")
        for limit_type, status in spending_status.items():
            logger.info(f"  {limit_type}: ${status['current_spent']:.2f} / ${status['limit_amount']:.2f} "
                       f"({status['percentage_used']:.1f}%) - Would exceed: {status['would_exceed']}")
        
        # Get user profile with spending data
        profile = await manager.get_user_profile(user_id)
        if profile:
            logger.info(f"User profile - Total spent: ${profile.total_spent}, "
                       f"Total requests: {profile.total_requests}")
        
        logger.info("Budget management completed successfully")
        
    finally:
        await manager.close()


async def analytics_example():
    """Demonstrate analytics generation and cost optimization."""
    logger.info("=== Analytics and Optimization Example ===")
    
    manager = create_storage_manager()
    
    try:
        await manager.initialize()
        
        # Generate sample data for multiple users
        users = ["user_charlie", "user_diana", "user_eve"]
        providers = [ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.GOOGLE]
        
        # Create varied usage patterns over the past week
        base_date = datetime.utcnow() - timedelta(days=7)
        
        for day in range(7):
            current_date = base_date + timedelta(days=day)
            
            for user in users:
                for hour in range(0, 24, 3):  # Every 3 hours
                    timestamp = current_date.replace(hour=hour)
                    
                    # Vary cost and usage by user and time
                    cost_multiplier = 1.0 + (hash(user) % 5) * 0.2  # 1.0 to 1.8x
                    peak_multiplier = 1.5 if 9 <= hour <= 17 else 1.0  # Business hours
                    
                    base_cost = Decimal("0.003") * Decimal(str(cost_multiplier * peak_multiplier))
                    
                    await manager.track_usage(
                        user_id=user,
                        event_type=UsageEventType.API_CALL,
                        provider=providers[hash(f"{user}_{hour}") % len(providers)],
                        cost_usd=base_cost,
                        token_count=int(100 * cost_multiplier),
                        model_name="gpt-4" if hour % 2 == 0 else "gpt-3.5-turbo",
                        success=(hour % 13 != 0),  # Occasional failures
                        timestamp=timestamp
                    )
        
        logger.info("Generated sample analytics data")
        
        # Generate comprehensive analytics for a specific user
        analytics = await manager.generate_usage_analytics(
            user_id="user_charlie",
            start_date=base_date,
            end_date=datetime.utcnow(),
            aggregation_type=TimeAggregation.DAILY,
            group_by=["provider", "event_type"]
        )
        
        logger.info("Analytics Results:")
        logger.info(f"  Records analyzed: {analytics['analysis_metadata']['records_analyzed']}")
        logger.info(f"  Total cost: ${analytics['summary']['total_cost']:.4f}")
        logger.info(f"  Total tokens: {analytics['summary']['total_tokens']:,}")
        logger.info(f"  Success rate: {analytics['summary']['success_rate']:.2%}")
        logger.info(f"  Unique providers: {analytics['summary']['unique_providers']}")
        
        # Cost optimization opportunities
        cost_optimization = analytics['cost_optimization']
        logger.info(f"Cost Optimization Opportunities: {len(cost_optimization['opportunities'])}")
        
        for opportunity in cost_optimization['opportunities']:
            logger.info(f"  - {opportunity['type']}: {opportunity['description']}")
            logger.info(f"    Potential savings: ${opportunity['potential_savings']:.4f}")
            logger.info(f"    Confidence: {opportunity['confidence']:.1%}")
        
        # Generate global analytics (all users)
        global_analytics = await manager.generate_usage_analytics(
            start_date=base_date,
            end_date=datetime.utcnow(),
            aggregation_type=TimeAggregation.HOURLY
        )
        
        logger.info(f"Global Analytics - Total cost: ${global_analytics['summary']['total_cost']:.4f}")
        
        logger.info("Analytics generation completed successfully")
        
    finally:
        await manager.close()


async def migration_example():
    """Demonstrate data migration between storage backends."""
    logger.info("=== Data Migration Example ===")
    
    manager = create_storage_manager()
    
    try:
        await manager.initialize()
        
        # Add some test data
        for i in range(10):
            await manager.track_usage(
                user_id="migration_user",
                event_type=UsageEventType.API_CALL,
                provider=ProviderType.OPENAI,
                cost_usd=Decimal("0.001") * i,
                token_count=50 + i * 10,
                model_name="gpt-4"
            )
        
        logger.info("Created test data for migration")
        
        # Create migration plan: JSON to ChromaDB
        plan_id = await manager.migrate_data(
            migration_type=MigrationType.JSON_TO_CHROMADB,
            source_type=StorageType.JSON,
            target_type=StorageType.CHROMADB,
            filters={"user_id": "migration_user"}
        )
        
        logger.info(f"Created migration plan: {plan_id}")
        
        # Check migration plan status before execution
        status = manager.migration_engine.get_migration_status(plan_id)
        if status:
            logger.info(f"Migration plan status: {status['status']}")
            logger.info(f"Steps: {len(status['steps'])}")
            for step in status['steps']:
                logger.info(f"  - {step['step_id']}: {step['description']}")
        
        # Execute the migration
        success = await manager.execute_migration(plan_id)
        logger.info(f"Migration execution result: {success}")
        
        # Check final status
        final_status = manager.migration_engine.get_migration_status(plan_id)
        if final_status:
            logger.info(f"Final migration status: {final_status['status']}")
            logger.info(f"Overall progress: {final_status['overall_progress']:.1f}%")
            
            if final_status['duration_seconds']:
                logger.info(f"Migration duration: {final_status['duration_seconds']:.2f} seconds")
        
        # Get migration history
        history = manager.migration_engine.get_migration_history()
        logger.info(f"Migration history: {len(history)} migrations")
        
        logger.info("Data migration completed successfully")
        
    finally:
        await manager.close()


async def performance_monitoring_example():
    """Demonstrate performance monitoring and optimization."""
    logger.info("=== Performance Monitoring Example ===")
    
    manager = create_storage_manager()
    
    try:
        await manager.initialize()
        
        # Generate load to observe performance
        logger.info("Generating load for performance testing...")
        
        start_time = datetime.utcnow()
        
        # High-frequency operations
        tasks = []
        for i in range(100):
            task = manager.track_usage(
                user_id=f"perf_user_{i % 10}",
                event_type=UsageEventType.API_CALL,
                provider=ProviderType.OPENAI,
                cost_usd=Decimal("0.001"),
                token_count=100
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_operations = sum(1 for r in results if not isinstance(r, Exception))
        
        logger.info(f"Completed {successful_operations}/{len(tasks)} operations")
        
        # Generate analytics (CPU-intensive operation)
        analytics = await manager.generate_usage_analytics(
            start_date=start_time,
            aggregation_type=TimeAggregation.HOURLY
        )
        
        logger.info(f"Generated analytics for {analytics['analysis_metadata']['records_analyzed']} records")
        
        # Get comprehensive system status
        status = await manager.get_comprehensive_status()
        
        logger.info("System Performance Status:")
        
        # Operation metrics
        if "operation_metrics" in status:
            for operation, metrics in status["operation_metrics"].items():
                logger.info(f"  {operation}:")
                logger.info(f"    Operations: {metrics['total_operations']}")
                logger.info(f"    Avg time: {metrics['average_time_seconds']:.4f}s")
                logger.info(f"    Error rate: {metrics['error_rate']:.2%}")
                logger.info(f"    Ops/sec: {metrics['operations_per_second']:.2f}")
        
        # Storage stats
        if "storage_stats" in status:
            cache_stats = status["storage_stats"].get("cache_stats", {})
            logger.info(f"  Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")
            logger.info(f"  Cache size: {cache_stats.get('cache_size', 0)} entries")
        
        # Performance optimization
        if "performance_stats" in status:
            perf_stats = status["performance_stats"]
            if "performance_monitor" in perf_stats:
                monitor_stats = perf_stats["performance_monitor"]
                logger.info(f"  Uptime: {monitor_stats.get('uptime_hours', 0):.2f} hours")
                logger.info(f"  Current CPU: {monitor_stats.get('current_cpu_percent', 0):.1f}%")
                logger.info(f"  Current Memory: {monitor_stats.get('current_memory_mb', 0):.1f} MB")
        
        # Run performance optimization
        optimization_result = await manager.run_performance_optimization()
        logger.info("Performance optimization completed:")
        
        if "optimizations" in optimization_result:
            for opt_type, opt_result in optimization_result["optimizations"].items():
                logger.info(f"  {opt_type}: {opt_result}")
        
        logger.info("Performance monitoring completed successfully")
        
    finally:
        await manager.close()


async def comprehensive_example():
    """Run a comprehensive example showcasing all features."""
    logger.info("=== Comprehensive Usage Tracking Example ===")
    
    # Custom configuration for this example
    config = StorageSchema(
        chromadb=ChromaDBConfig(
            persist_directory="./data/comprehensive_chromadb",
            batch_size=500
        ),
        json_storage=JSONStorageConfig(
            base_path="./data/comprehensive_json",
            compression=True,
            retention_days=60
        ),
        hybrid=HybridStorageConfig(
            hot_data_days=3,
            warm_data_days=14,
            cache_size_mb=100,
            auto_migrate=True
        )
    )
    
    manager = create_storage_manager(config)
    
    try:
        await manager.initialize()
        
        logger.info("Running comprehensive usage tracking demonstration...")
        
        # 1. Multi-user, multi-provider usage simulation
        logger.info("1. Simulating diverse usage patterns...")
        
        users = ["alice", "bob", "charlie", "diana"]
        providers = [ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.GOOGLE]
        operations = ["chat", "completion", "embedding", "analysis", "generation"]
        
        for day in range(5):  # 5 days of data
            current_date = datetime.utcnow() - timedelta(days=4-day)
            
            for user in users:
                # Each user has different usage patterns
                daily_requests = 20 + (hash(user) % 30)  # 20-50 requests per day
                
                for req in range(daily_requests):
                    # Randomize timing within the day
                    hour = (req * 24) // daily_requests
                    minute = (req * 60) % 60
                    
                    timestamp = current_date.replace(hour=hour, minute=minute)
                    
                    # Provider and cost vary by user preferences
                    provider = providers[hash(f"{user}_{req}") % len(providers)]
                    base_cost = {"openai": 0.002, "anthropic": 0.003, "google": 0.001}
                    cost = Decimal(str(base_cost.get(provider.value, 0.002)))
                    
                    await manager.track_usage(
                        user_id=user,
                        event_type=UsageEventType.API_CALL,
                        provider=provider,
                        cost_usd=cost,
                        token_count=100 + (req % 200),
                        model_name=f"model-{provider.value}",
                        operation=operations[req % len(operations)],
                        success=(req % 15 != 0),  # ~93% success rate
                        timestamp=timestamp
                    )
        
        logger.info("Generated comprehensive usage data")
        
        # 2. Set up user budgets and limits
        logger.info("2. Configuring user budgets...")
        
        for user in users:
            await manager.set_spending_limit(user, "daily", Decimal("1.00"))
            await manager.set_spending_limit(user, "weekly", Decimal("5.00"))
            await manager.set_spending_limit(user, "monthly", Decimal("20.00"))
        
        # 3. Generate comprehensive analytics
        logger.info("3. Generating analytics for all users...")
        
        all_analytics = {}
        for user in users:
            analytics = await manager.generate_usage_analytics(
                user_id=user,
                start_date=datetime.utcnow() - timedelta(days=5),
                aggregation_type=TimeAggregation.DAILY,
                group_by=["provider"]
            )
            all_analytics[user] = analytics
            
            logger.info(f"  {user}: ${analytics['summary']['total_cost']:.4f} "
                       f"({analytics['summary']['total_requests']} requests)")
        
        # 4. Global trend analysis
        logger.info("4. Analyzing global trends...")
        
        global_analytics = await manager.generate_usage_analytics(
            start_date=datetime.utcnow() - timedelta(days=5),
            aggregation_type=TimeAggregation.DAILY
        )
        
        logger.info(f"Global metrics:")
        logger.info(f"  Total cost: ${global_analytics['summary']['total_cost']:.4f}")
        logger.info(f"  Total requests: {global_analytics['summary']['total_requests']:,}")
        logger.info(f"  Success rate: {global_analytics['summary']['success_rate']:.2%}")
        
        # Cost optimization opportunities
        opportunities = global_analytics['cost_optimization']['opportunities']
        if opportunities:
            logger.info(f"Cost optimization opportunities:")
            for opp in opportunities[:3]:  # Top 3
                logger.info(f"  - {opp['description']}: ${opp['potential_savings']:.4f} savings")
        
        # 5. Performance monitoring
        logger.info("5. Checking system performance...")
        
        status = await manager.get_comprehensive_status()
        
        # Performance summary
        if "operation_metrics" in status:
            reads = status["operation_metrics"].get("reads", {})
            writes = status["operation_metrics"].get("writes", {})
            analytics_ops = status["operation_metrics"].get("analytics", {})
            
            logger.info(f"Performance summary:")
            logger.info(f"  Read ops: {reads.get('total_operations', 0)} "
                       f"(avg {reads.get('average_time_seconds', 0):.3f}s)")
            logger.info(f"  Write ops: {writes.get('total_operations', 0)} "
                       f"(avg {writes.get('average_time_seconds', 0):.3f}s)")
            logger.info(f"  Analytics: {analytics_ops.get('total_operations', 0)} "
                       f"(avg {analytics_ops.get('average_time_seconds', 0):.3f}s)")
        
        # 6. Data migration demonstration
        logger.info("6. Demonstrating data migration...")
        
        # Create a migration plan
        plan_id = await manager.migrate_data(
            migration_type=MigrationType.STORAGE_CONSOLIDATION,
            source_type=StorageType.HYBRID,
            target_type=StorageType.HYBRID,
            filters={"user_id": "alice"}
        )
        
        # Get migration status
        migration_status = manager.migration_engine.get_migration_status(plan_id)
        if migration_status:
            logger.info(f"Created migration plan with {len(migration_status['steps'])} steps")
        
        # 7. System optimization
        logger.info("7. Running system optimization...")
        
        optimization_result = await manager.run_performance_optimization()
        logger.info("Optimization completed:")
        
        for opt_type, result in optimization_result.get("optimizations", {}).items():
            if isinstance(result, dict) and "memory_saved_mb" in result:
                logger.info(f"  {opt_type}: {result['memory_saved_mb']:.1f} MB saved")
            elif isinstance(result, dict):
                logger.info(f"  {opt_type}: {result}")
        
        # 8. Final status report
        logger.info("8. Final system status:")
        
        final_status = await manager.get_comprehensive_status()
        
        # Storage utilization
        storage_stats = final_status.get("storage_stats", {})
        if "cache_stats" in storage_stats:
            cache = storage_stats["cache_stats"]
            logger.info(f"  Cache: {cache.get('cache_size', 0)} entries, "
                       f"{cache.get('hit_rate', 0):.2%} hit rate")
        
        # Migration history
        migration_stats = final_status.get("migration_stats", {})
        if migration_stats:
            stats = migration_stats.get("migration_stats", {})
            logger.info(f"  Migrations: {stats.get('total_migrations', 0)} total, "
                       f"{stats.get('successful_migrations', 0)} successful")
        
        logger.info("Comprehensive example completed successfully!")
        
        # Save status report
        import json
        with open("usage_tracking_status_report.json", "w") as f:
            json.dump(final_status, f, indent=2, default=str)
        
        logger.info("Status report saved to: usage_tracking_status_report.json")
        
    finally:
        await manager.close()


async def main():
    """Run all examples."""
    try:
        await basic_usage_example()
        await asyncio.sleep(1)
        
        await budget_management_example()
        await asyncio.sleep(1)
        
        await analytics_example()
        await asyncio.sleep(1)
        
        await migration_example()
        await asyncio.sleep(1)
        
        await performance_monitoring_example()
        await asyncio.sleep(1)
        
        await comprehensive_example()
        
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())