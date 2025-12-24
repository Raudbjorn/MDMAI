#!/usr/bin/env python3
"""Validation script to demonstrate JSON Lines scalability improvements.

This script validates that the persistence layer is using scalable append-only
operations and can handle large files efficiently.
"""

import asyncio
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from usage_tracking.json_persistence import JsonPersistenceManager, PersistenceConfig
from ai_providers.models import UsageRecord, ProviderType
from config.logging_config import get_logger

logger = get_logger(__name__)


async def validate_append_only_behavior():
    """Test that writes are truly append-only (no reading of existing data)."""
    print("🔍 Testing append-only behavior...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = PersistenceConfig(
            base_path=temp_dir,
            write_buffer_size=5,  # Small buffer for testing
            flush_interval_seconds=1
        )
        
        async with JsonPersistenceManager(config) as manager:
            # Create a large file first
            large_batch = []
            for i in range(1000):
                record = UsageRecord(
                    request_id=f"large_batch_{i}",
                    session_id="test_session",
                    provider_type=ProviderType.OPENAI,
                    model="gpt-4",
                    input_tokens=100,
                    output_tokens=50,
                    cost=0.001,
                    latency_ms=100,
                    success=True,
                    timestamp=datetime.now()
                )
                large_batch.append(record)
            
            print("  📝 Writing initial large batch (1000 records)...")
            await manager.store_usage_records_batch(large_batch)
            
            # Now append single records - this should NOT read the existing data
            print("  ➕ Appending individual records...")
            append_times = []
            
            for i in range(10):
                start_time = asyncio.get_event_loop().time()
                
                record = UsageRecord(
                    request_id=f"append_test_{i}",
                    session_id="test_session",
                    provider_type=ProviderType.ANTHROPIC,
                    model="claude-3",
                    input_tokens=200,
                    output_tokens=100,
                    cost=0.002,
                    latency_ms=150,
                    success=True,
                    timestamp=datetime.now()
                )
                
                await manager.store_usage_record(record)
                
                append_time = asyncio.get_event_loop().time() - start_time
                append_times.append(append_time)
            
            # Force final flush
            await manager._flush_all_buffers()
            
            # Check that append times are consistent (not growing with file size)
            avg_append_time = sum(append_times) / len(append_times)
            max_append_time = max(append_times)
            
            print(f"  ⏱️  Average append time: {avg_append_time*1000:.2f}ms")
            print(f"  ⏱️  Maximum append time: {max_append_time*1000:.2f}ms")
            
            # Performance should be consistent (O(1))
            performance_consistent = max_append_time < avg_append_time * 2
            
            if performance_consistent:
                print("  ✅ Append-only behavior confirmed - performance is O(1)")
            else:
                print("  ❌ Performance not consistent - may not be truly append-only")
            
            return performance_consistent


async def validate_json_lines_format():
    """Test that files are written in proper JSON Lines format."""
    print("\n🔍 Testing JSON Lines format compliance...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = PersistenceConfig(
            base_path=temp_dir,
            write_buffer_size=3,  # Small buffer for immediate writes
            compression_type=config.CompressionType.NONE  # No compression for easier validation
        )
        
        async with JsonPersistenceManager(config) as manager:
            # Create test records
            test_records = []
            for i in range(5):
                record = UsageRecord(
                    request_id=f"format_test_{i}",
                    session_id="format_session",
                    provider_type=ProviderType.OPENAI,
                    model="gpt-3.5-turbo",
                    input_tokens=50 + i,
                    output_tokens=25 + i,
                    cost=0.001 * i,
                    latency_ms=100 + i*10,
                    success=True,
                    timestamp=datetime.now()
                )
                test_records.append(record)
            
            print("  📝 Writing test records...")
            await manager.store_usage_records_batch(test_records)
            await manager._flush_all_buffers()
            
            # Find the created file
            partitions_dir = Path(temp_dir) / "partitions"
            if partitions_dir.exists():
                jsonl_files = list(partitions_dir.glob("*.jsonl"))
                if jsonl_files:
                    file_path = jsonl_files[0]
                    print(f"  📄 Validating file: {file_path.name}")
                    
                    # Read file and validate JSON Lines format
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    valid_lines = 0
                    total_lines = len([line for line in lines if line.strip()])
                    
                    for line_num, line in enumerate(lines, 1):
                        if line.strip():
                            try:
                                import json
                                obj = json.loads(line)
                                if isinstance(obj, dict):
                                    valid_lines += 1
                                    # Check for required fields
                                    if "request_id" in obj and "provider_type" in obj:
                                        pass  # Good
                                    else:
                                        print(f"    ⚠️  Line {line_num}: Missing required fields")
                                else:
                                    print(f"    ❌ Line {line_num}: Not a JSON object")
                            except json.JSONDecodeError as e:
                                print(f"    ❌ Line {line_num}: Invalid JSON - {e}")
                    
                    print(f"  📊 Valid JSON Lines: {valid_lines}/{total_lines}")
                    
                    if valid_lines == total_lines and total_lines > 0:
                        print("  ✅ JSON Lines format is valid")
                        return True
                    else:
                        print("  ❌ JSON Lines format validation failed")
                        return False
                else:
                    print("  ❌ No .jsonl files found")
                    return False
            else:
                print("  ❌ No partitions directory found")
                return False


async def validate_file_format_detection():
    """Test that the system can detect and handle legacy JSON format."""
    print("\n🔍 Testing legacy format detection...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a legacy JSON file
        legacy_file = Path(temp_dir) / "legacy_data.json"
        legacy_data = {
            "records": [
                {
                    "request_id": "legacy_1",
                    "provider_type": "openai",
                    "model": "gpt-4",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost": 0.001,
                    "timestamp": "2024-01-01T12:00:00Z"
                }
            ]
        }
        
        import json
        with open(legacy_file, 'w') as f:
            json.dump(legacy_data, f)
        
        print(f"  📄 Created legacy file: {legacy_file.name}")
        
        config = PersistenceConfig(base_path=temp_dir)
        manager = JsonPersistenceManager(config)
        
        # Test format validation
        validation_result = await manager.validate_file_format(str(legacy_file))
        
        print(f"  🔍 Detected format: {validation_result['format']}")
        print(f"  📊 Legacy file: {validation_result['is_legacy']}")
        print(f"  📈 Valid: {validation_result['valid']}")
        
        if validation_result['is_legacy'] and validation_result['format'] == 'legacy_json_array':
            print("  ✅ Legacy format detection working")
            return True
        else:
            print("  ❌ Legacy format detection failed")
            return False


async def run_performance_benchmark():
    """Run performance benchmark to show scalability."""
    print("\n🔍 Running performance benchmark...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = PersistenceConfig(
            base_path=temp_dir,
            write_buffer_size=100,
            compression_type=config.CompressionType.NONE
        )
        
        async with JsonPersistenceManager(config) as manager:
            # Run benchmark with moderate number of records
            results = await manager.benchmark_append_performance(num_records=1000)
            
            print(f"  📊 Records written: {results['test_records']}")
            print(f"  ⚡ Records/second: {results['records_per_second']:.2f}")
            print(f"  💾 MB/second: {results['mb_per_second']:.2f}")
            print(f"  ⏱️  Average write time: {results['avg_write_time_ms']:.2f}ms")
            print(f"  🎯 Consistent performance: {results['consistent_performance']}")
            
            if results['records_per_second'] > 100:  # Should be much higher for append-only
                print("  ✅ Performance benchmark passed")
                return True
            else:
                print("  ❌ Performance benchmark failed")
                return False


async def main():
    """Run all validation tests."""
    print("🚀 JSON Lines Scalability Validation")
    print("="*50)
    
    tests = [
        ("Append-Only Behavior", validate_append_only_behavior),
        ("JSON Lines Format", validate_json_lines_format),
        ("Legacy Format Detection", validate_file_format_detection),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result, None))
        except Exception as e:
            print(f"  ❌ Test failed with error: {e}")
            results.append((test_name, False, str(e)))
    
    # Summary
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if error:
            print(f"      Error: {error}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All scalability validations passed!")
        print("The JSON Lines implementation is working correctly.")
    else:
        print("⚠️  Some validations failed - check implementation.")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n❌ Validation cancelled by user")
        sys.exit(1)