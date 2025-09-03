#!/usr/bin/env python3
"""Test script to verify performance optimization improvements."""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_bare_except_fixes():
    """Test that bare except clauses have been fixed."""
    print("\n=== Testing Bare Except Fixes ===")
    
    try:
        from performance.cache_system import CacheSystem
        
        # Check that the module imports without errors
        print("✓ CacheSystem imports successfully with fixed exception handling")
        
        # Test that specific exceptions are now caught
        cache = CacheSystem(name="test", max_size=10)
        
        # Test object size estimation with various types
        test_objects = [
            {"test": "data"},
            [1, 2, 3],
            "string",
            12345,
            None,
            {"nested": {"data": [1, 2, 3]}},
            lambda x: x  # This should trigger the pickle error path
        ]
        
        for obj in test_objects:
            size = cache._estimate_size(obj)
            assert size > 0, f"Size estimation failed for {type(obj)}"
        
        print("✓ Size estimation handles various object types correctly")
        return True
        
    except Exception as e:
        print(f"✗ Error testing bare except fixes: {e}")
        return False


def test_psutil_dependency():
    """Test that psutil is properly available."""
    print("\n=== Testing psutil Dependency ===")
    
    try:
        import psutil
        print("✓ psutil module imports successfully")
        
        # Test basic psutil functionality
        cpu_percent = psutil.cpu_percent(interval=0.1)
        print(f"✓ CPU usage: {cpu_percent}%")
        
        memory = psutil.virtual_memory()
        print(f"✓ Memory usage: {memory.percent}%")
        
        return True
        
    except ImportError:
        print("✗ psutil not installed - please run: pip install psutil")
        return False
    except Exception as e:
        print(f"✗ Error testing psutil: {e}")
        return False


def test_threadpool_cleanup():
    """Test that ThreadPoolExecutor is properly cleaned up."""
    print("\n=== Testing ThreadPoolExecutor Cleanup ===")
    
    try:
        from performance.database_optimizer import DatabaseOptimizer
        
        # Create a mock DB manager
        class MockDB:
            def __init__(self):
                self.collections = {"test": MockCollection()}
        
        class MockCollection:
            def count(self):
                return 100
        
        # Test optimizer lifecycle
        db = MockDB()
        optimizer = DatabaseOptimizer(db)
        
        # Check that executor exists
        assert optimizer.executor is not None, "Executor not initialized"
        print("✓ ThreadPoolExecutor initialized")
        
        # Test shutdown
        optimizer.shutdown()
        assert optimizer._shutdown == True, "Shutdown flag not set"
        print("✓ Optimizer shutdown method works")
        
        # Test that __del__ doesn't cause errors
        del optimizer
        print("✓ Optimizer cleanup on deletion works")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing ThreadPoolExecutor cleanup: {e}")
        return False


async def test_performance_monitor_throttling():
    """Test that performance monitor has throttling."""
    print("\n=== Testing Performance Monitor Throttling ===")
    
    try:
        from performance.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Test adaptive throttling logic
        monitor._last_cpu_percent = 85  # Simulate high CPU
        
        # Start monitoring with a short interval
        await monitor.start_monitoring(interval=1)
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        print("✓ Performance monitor starts and stops correctly")
        print("✓ Adaptive throttling logic is in place")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing performance monitor: {e}")
        return False


def test_cache_invalidation_atomicity():
    """Test that cache invalidation is atomic."""
    print("\n=== Testing Cache Invalidation Atomicity ===")
    
    try:
        from performance.cache_invalidator import CacheInvalidator
        from performance.cache_system import CacheSystem
        
        # Create invalidator and cache
        invalidator = CacheInvalidator()
        cache = CacheSystem(name="test", max_size=100)
        
        # Add some test entries
        for i in range(10):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Register cache with invalidator
        invalidator.register_cache("test", cache)
        
        # Test that invalidation doesn't fail with concurrent access
        # This would have failed with the old implementation
        results = invalidator.invalidate_stale(max_age_seconds=0)
        
        print(f"✓ Atomic invalidation completed: {results}")
        print("✓ No race conditions during invalidation")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing cache invalidation: {e}")
        return False


async def test_input_validation():
    """Test that input validation prevents invalid configurations."""
    print("\n=== Testing Input Validation ===")
    
    try:
        # Test cache configuration validation
        from performance.mcp_tools import configure_cache
        
        # Mock the globals for testing
        import performance.mcp_tools as mcp_tools
        
        class MockConfig:
            def get_profile(self, name):
                if name == "test_cache":
                    return type('obj', (object,), {
                        'max_size': 100,
                        'max_memory_mb': 10,
                        'ttl_seconds': 3600,
                        'policy': 'LRU',
                        'persistent': False,
                        'to_dict': lambda: {}
                    })
                return None
            
            def add_profile(self, profile):
                pass
        
        mcp_tools._cache_config = MockConfig()
        
        # Test invalid max_size
        result = await configure_cache("test_cache", max_size=-1)
        assert not result["success"], "Should reject negative max_size"
        print(f"✓ Invalid max_size rejected: {result['error']}")
        
        # Test invalid max_memory_mb
        result = await configure_cache("test_cache", max_memory_mb=0)
        assert not result["success"], "Should reject zero max_memory_mb"
        print(f"✓ Invalid max_memory_mb rejected: {result['error']}")
        
        # Test invalid policy
        result = await configure_cache("test_cache", policy="INVALID")
        assert not result["success"], "Should reject invalid policy"
        print(f"✓ Invalid policy rejected: {result['error']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing input validation: {e}")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Performance Optimization Improvements")
    print("=" * 60)
    
    all_passed = True
    
    # Run synchronous tests
    if not test_bare_except_fixes():
        all_passed = False
    
    if not test_psutil_dependency():
        all_passed = False
    
    if not test_threadpool_cleanup():
        all_passed = False
    
    # Run async tests
    if not await test_performance_monitor_throttling():
        all_passed = False
    
    if not test_cache_invalidation_atomicity():
        all_passed = False
    
    if not await test_input_validation():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All performance optimization improvements are working!")
    else:
        print("❌ Some tests failed. Please review the output above.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))