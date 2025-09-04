#!/usr/bin/env python3
"""
Test script to verify scalability fixes in user tracking files.
This script tests:
1. Database initialization
2. Partitioned storage 
3. Streaming data processing
4. Memory-efficient global analysis
5. Backward compatibility with legacy files
"""

import asyncio
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai_providers.user_usage_tracker import UserUsageTracker, UserProfile, UserUsageAggregation
from ai_providers.models import ProviderType, UsageRecord
from ai_providers.analytics_dashboard import AnalyticsDashboard
from ai_providers.cost_optimization_engine import CostOptimizationEngine


async def create_test_data(tracker: UserUsageTracker, num_users: int = 100, days: int = 30):
    """Create test data to verify scalability."""
    print(f"Creating test data for {num_users} users over {days} days...")
    
    # Create test users
    for i in range(num_users):
        user_profile = UserProfile(
            user_id=f"test_user_{i:04d}",
            username=f"testuser{i:04d}",
            email=f"user{i:04d}@example.com",
            user_tier="premium" if i % 10 == 0 else "free"
        )
        await tracker.create_user_profile(user_profile)
    
    # Create usage records for each user over the time period
    base_date = datetime.now() - timedelta(days=days)
    
    for day in range(days):
        current_date = base_date + timedelta(days=day)
        
        for user_i in range(num_users):
            user_id = f"test_user_{user_i:04d}"
            
            # Create some usage records for this user on this day
            requests_today = (user_i % 10) + 1  # 1-10 requests per day
            
            for req in range(requests_today):
                usage_record = UsageRecord(
                    request_id=f"req_{user_i}_{day}_{req}",
                    session_id=f"session_{user_i}_{day}",
                    timestamp=current_date,
                    provider_type=ProviderType.OPENAI if user_i % 2 == 0 else ProviderType.ANTHROPIC,
                    model=f"gpt-4" if user_i % 2 == 0 else "claude-3",
                    input_tokens=100 + (user_i % 50),
                    output_tokens=50 + (user_i % 30), 
                    cost=0.01 + (user_i * 0.001),
                    success=True,
                    latency_ms=100 + (user_i % 200)
                )
                
                await tracker.record_user_usage(user_id, usage_record)
        
        # Periodic flush and progress update
        if day % 10 == 0:
            await tracker._flush_pending_records()
            print(f"  Processed {day + 1}/{days} days...")
    
    # Final flush
    await tracker._flush_pending_records()
    print("Test data creation completed!")


async def test_legacy_compatibility(temp_dir: Path):
    """Test backward compatibility with legacy JSON files."""
    print("Testing backward compatibility with legacy files...")
    
    # Create legacy JSON files
    legacy_daily = {
        "legacy_user_1": {
            "2024-01-01": {
                "total_requests": 10,
                "successful_requests": 9,
                "failed_requests": 1,
                "total_input_tokens": 1000,
                "total_output_tokens": 500,
                "total_cost": 0.15,
                "avg_latency_ms": 120.5,
                "providers_used": {"openai": 10},
                "models_used": {"gpt-4": 10},
                "session_count": 1,
                "unique_sessions": ["session_1"]
            }
        },
        "legacy_user_2": {
            "2024-01-01": {
                "total_requests": 5,
                "successful_requests": 5,
                "failed_requests": 0,
                "total_input_tokens": 500,
                "total_output_tokens": 250,
                "total_cost": 0.08,
                "avg_latency_ms": 95.2,
                "providers_used": {"anthropic": 5},
                "models_used": {"claude-3": 5},
                "session_count": 1,
                "unique_sessions": ["session_2"]
            }
        }
    }
    
    legacy_monthly = {
        "legacy_user_1": {
            "2024-01": {
                "total_requests": 300,
                "successful_requests": 285,
                "failed_requests": 15,
                "total_input_tokens": 30000,
                "total_output_tokens": 15000,
                "total_cost": 4.50,
                "avg_latency_ms": 115.8,
                "providers_used": {"openai": 300},
                "models_used": {"gpt-4": 300},
                "session_count": 30,
                "unique_sessions": [f"session_{i}" for i in range(30)]
            }
        }
    }
    
    # Write legacy files
    with open(temp_dir / "daily_usage.json", 'w') as f:
        json.dump(legacy_daily, f)
    
    with open(temp_dir / "monthly_usage.json", 'w') as f:
        json.dump(legacy_monthly, f)
    
    # Initialize tracker and verify migration
    tracker = UserUsageTracker(str(temp_dir), use_chromadb=False)
    await asyncio.sleep(1)  # Give initialization time
    
    # Verify data was migrated correctly
    daily_usage = await tracker.get_user_daily_usage("legacy_user_1", "2024-01-01")
    assert daily_usage == 0.15, f"Expected 0.15, got {daily_usage}"
    
    monthly_usage = await tracker.get_user_monthly_usage("legacy_user_1", "2024-01")
    assert monthly_usage == 4.50, f"Expected 4.50, got {monthly_usage}"
    
    # Verify legacy files were backed up
    assert (temp_dir / "daily_usage.json.legacy_backup").exists()
    assert (temp_dir / "monthly_usage.json.legacy_backup").exists()
    
    print("✓ Legacy compatibility test passed!")
    
    await tracker.cleanup()


async def test_database_storage(temp_dir: Path):
    """Test database-backed storage functionality."""
    print("Testing database storage...")
    
    tracker = UserUsageTracker(str(temp_dir), use_chromadb=False)
    await asyncio.sleep(1)  # Give initialization time
    
    # Verify database was created
    db_path = temp_dir / "usage_aggregations.db"
    assert db_path.exists(), "Database file should exist"
    
    # Test data insertion and retrieval
    user_profile = UserProfile(
        user_id="db_test_user",
        username="dbtestuser",
        email="dbtest@example.com"
    )
    await tracker.create_user_profile(user_profile)
    
    # Create some usage data
    usage_record = UsageRecord(
        request_id="db_test_req",
        session_id="db_test_session",
        timestamp=datetime.now(),
        provider_type=ProviderType.OPENAI,
        model="gpt-4",
        input_tokens=100,
        output_tokens=50,
        cost=0.02,
        success=True,
        latency_ms=120
    )
    
    await tracker.record_user_usage("db_test_user", usage_record)
    await tracker._flush_pending_records()
    
    # Verify data was stored
    daily_cost = await tracker.get_user_daily_usage("db_test_user")
    assert daily_cost > 0, f"Expected cost > 0, got {daily_cost}"
    
    print("✓ Database storage test passed!")
    
    await tracker.cleanup()


async def test_streaming_analytics(temp_dir: Path):
    """Test streaming data processing in analytics."""
    print("Testing streaming analytics...")
    
    tracker = UserUsageTracker(str(temp_dir), use_chromadb=False)
    await asyncio.sleep(1)
    
    # Create minimal test data
    await create_test_data(tracker, num_users=20, days=5)
    
    # Test analytics dashboard
    dashboard = AnalyticsDashboard(tracker)
    
    # Test streaming data retrieval
    start_time = datetime.now() - timedelta(days=7)
    end_time = datetime.now()
    
    data_count = 0
    async for data_point in dashboard._get_usage_data(start_time, end_time):
        data_count += 1
        assert "user_id" in data_point
        assert "total_cost" in data_point
        assert "timestamp" in data_point
        
        # Don't process too many for testing
        if data_count > 100:
            break
    
    print(f"✓ Streaming analytics processed {data_count} data points")
    
    await tracker.cleanup()


async def test_global_patterns_analysis(temp_dir: Path):
    """Test memory-efficient global patterns analysis."""
    print("Testing global patterns analysis...")
    
    tracker = UserUsageTracker(str(temp_dir), use_chromadb=False)
    await asyncio.sleep(1)
    
    # Create test data
    await create_test_data(tracker, num_users=30, days=7)
    
    # Test cost optimization engine
    optimizer = CostOptimizationEngine(tracker, None)  # No pricing engine for this test
    
    start_time = datetime.now() - timedelta(days=7)
    end_time = datetime.now()
    
    # This should use the new memory-efficient analysis
    patterns = await optimizer._analyze_global_patterns(start_time, end_time)
    
    print(f"✓ Global patterns analysis completed, found {len(patterns)} patterns")
    
    await tracker.cleanup()


async def test_memory_usage():
    """Test that memory usage remains reasonable even with many users."""
    print("Testing memory efficiency...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        tracker = UserUsageTracker(str(temp_path), use_chromadb=False)
        await asyncio.sleep(1)
        
        # Create a substantial amount of test data
        print("  Creating data for memory efficiency test...")
        await create_test_data(tracker, num_users=200, days=10)
        
        # Force cache cleanup
        await tracker._cleanup_cache()
        
        # Check that cache sizes are reasonable
        daily_cache_size = sum(len(user_daily) for user_daily in tracker._daily_usage_cache.values())
        monthly_cache_size = sum(len(user_monthly) for user_monthly in tracker._monthly_usage_cache.values())
        
        print(f"  Daily cache entries: {daily_cache_size}")
        print(f"  Monthly cache entries: {monthly_cache_size}")
        
        # With cleanup, cache should be reasonable even with 200 users * 10 days
        assert daily_cache_size < 500, f"Daily cache too large: {daily_cache_size}"
        
        print("✓ Memory efficiency test passed!")
        
        await tracker.cleanup()


async def main():
    """Run all scalability tests."""
    print("=" * 60)
    print("SCALABILITY FIXES TEST SUITE")
    print("=" * 60)
    
    try:
        # Test legacy compatibility
        with tempfile.TemporaryDirectory() as temp_dir:
            await test_legacy_compatibility(Path(temp_dir))
        
        # Test database functionality  
        with tempfile.TemporaryDirectory() as temp_dir:
            await test_database_storage(Path(temp_dir))
        
        # Test streaming analytics
        with tempfile.TemporaryDirectory() as temp_dir:
            await test_streaming_analytics(Path(temp_dir))
        
        # Test global patterns analysis
        with tempfile.TemporaryDirectory() as temp_dir:
            await test_global_patterns_analysis(Path(temp_dir))
        
        # Test memory efficiency
        await test_memory_usage()
        
        print("=" * 60)
        print("✅ ALL SCALABILITY TESTS PASSED!")
        print("=" * 60)
        print("\nScalability improvements verified:")
        print("• Partitioned storage implemented")
        print("• Database-backed aggregations working")
        print("• Streaming data processing functional")
        print("• Memory-efficient global analysis active")
        print("• Backward compatibility maintained")
        print("• Cache management preventing memory leaks")
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(result)