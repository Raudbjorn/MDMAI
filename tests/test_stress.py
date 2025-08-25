"""Stress tests for concurrent operations and load testing in TTRPG Assistant.

This module provides comprehensive stress testing for:
- High concurrent user load
- Database connection pooling limits
- Memory pressure scenarios
- CPU-intensive operations under load
- Network/IO bottleneck testing
- Resource exhaustion handling
- Recovery from system stress
- Cascade failure prevention
"""

import asyncio
import gc
import json
import os
import random
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import psutil
import pytest
import pytest_asyncio

from config.settings import settings
from src.campaign.campaign_manager import CampaignManager
from src.character_generation.character_generator import CharacterGenerator
from src.core.database import ChromaDBManager
from src.pdf_processing.pipeline import PDFProcessingPipeline
from src.performance.cache_manager import GlobalCacheManager
from src.performance.parallel_processor import ParallelProcessor
from src.search.search_service import SearchService
from src.session.session_manager import SessionManager


@dataclass
class StressTestMetrics:
    """Metrics for stress testing."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    start_time: float = 0
    end_time: float = 0
    peak_memory_mb: float = 0
    peak_cpu_percent: float = 0
    peak_threads: int = 0
    errors: List[str] = None
    response_times: List[float] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.response_times is None:
            self.response_times = []
    
    @property
    def duration(self) -> float:
        """Get test duration in seconds."""
        return self.end_time - self.start_time if self.end_time else 0
    
    @property
    def success_rate(self) -> float:
        """Get success rate percentage."""
        if self.total_operations == 0:
            return 0
        return (self.successful_operations / self.total_operations) * 100
    
    @property
    def throughput(self) -> float:
        """Get operations per second."""
        if self.duration == 0:
            return 0
        return self.total_operations / self.duration
    
    @property
    def avg_response_time(self) -> float:
        """Get average response time."""
        if not self.response_times:
            return 0
        return sum(self.response_times) / len(self.response_times)
    
    @property
    def p95_response_time(self) -> float:
        """Get 95th percentile response time."""
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        index = int(len(sorted_times) * 0.95)
        return sorted_times[index] if index < len(sorted_times) else sorted_times[-1]
    
    def report(self) -> str:
        """Generate stress test report."""
        return f"""
Stress Test Report:
==================
Duration: {self.duration:.2f}s
Total Operations: {self.total_operations}
Successful: {self.successful_operations}
Failed: {self.failed_operations}
Success Rate: {self.success_rate:.1f}%
Throughput: {self.throughput:.2f} ops/sec

Response Times:
  Average: {self.avg_response_time*1000:.2f}ms
  P95: {self.p95_response_time*1000:.2f}ms

Resource Usage:
  Peak Memory: {self.peak_memory_mb:.2f} MB
  Peak CPU: {self.peak_cpu_percent:.1f}%
  Peak Threads: {self.peak_threads}

Errors: {len(self.errors)}
"""


class ResourceMonitor:
    """Monitor system resources during stress tests."""
    
    def __init__(self):
        self.monitoring = False
        self.process = psutil.Process()
        self.peak_memory = 0
        self.peak_cpu = 0
        self.peak_threads = 0
        self.samples = []
        
    def start(self):
        """Start monitoring resources."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """Stop monitoring resources."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
            
    def _monitor_loop(self):
        """Monitor loop running in separate thread."""
        while self.monitoring:
            try:
                # Sample resource usage
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()
                num_threads = self.process.num_threads()
                
                # Update peaks
                self.peak_memory = max(self.peak_memory, memory_mb)
                self.peak_cpu = max(self.peak_cpu, cpu_percent)
                self.peak_threads = max(self.peak_threads, num_threads)
                
                # Store sample
                self.samples.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_mb,
                    'cpu_percent': cpu_percent,
                    'threads': num_threads
                })
                
                time.sleep(0.1)  # Sample every 100ms
            except Exception:
                logging.exception("Exception occurred in ResourceMonitor._monitor_loop")


class TestHighConcurrentLoad:
    """Test system under high concurrent load."""
    
    @pytest_asyncio.fixture
    async def stress_db(self, tmp_path):
        """Create database for stress testing."""
        with patch.object(settings, "chroma_db_path", tmp_path / "stress_db"):
            with patch.object(settings, "embedding_model", "all-MiniLM-L6-v2"):
                db = ChromaDBManager()
                
                # Pre-populate with test data
                for i in range(100):
                    db.add_document(
                        "rulebooks",
                        f"stress_doc_{i}",
                        f"Stress test document {i} content",
                        {"index": i}
                    )
                
                yield db
                
                if hasattr(db, "cleanup"):
                    await db.cleanup()
    
    @pytest.mark.asyncio
    async def test_concurrent_searches(self, stress_db):
        """Test system with many concurrent search operations."""
        search_service = SearchService(stress_db)
        metrics = StressTestMetrics()
        monitor = ResourceMonitor()
        
        # Test parameters
        num_concurrent = 100
        num_iterations = 10
        
        monitor.start()
        metrics.start_time = time.time()
        
        async def search_task(task_id: int):
            """Individual search task."""
            try:
                start = time.perf_counter()
                query = f"test query {task_id % 10}"
                result = await search_service.search(
                    query=query,
                    collection_name="rulebooks",
                    max_results=5
                )
                elapsed = time.perf_counter() - start
                metrics.response_times.append(elapsed)
                metrics.successful_operations += 1
                return True
            except Exception as e:
                metrics.failed_operations += 1
                metrics.errors.append(str(e))
                return False
        
        # Run stress test
        for iteration in range(num_iterations):
            tasks = [
                asyncio.create_task(search_task(i))
                for i in range(num_concurrent)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            metrics.total_operations += num_concurrent
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        metrics.end_time = time.time()
        monitor.stop()
        
        # Collect resource metrics
        metrics.peak_memory_mb = monitor.peak_memory
        metrics.peak_cpu_percent = monitor.peak_cpu
        metrics.peak_threads = monitor.peak_threads
        
        # Print report
        print(metrics.report())
        
        # Assertions
        assert metrics.success_rate > 95  # At least 95% success rate
        assert metrics.avg_response_time < 1.0  # Average response under 1 second
        assert metrics.peak_memory_mb < 2000  # Memory usage under 2GB
    
    @pytest.mark.asyncio
    async def test_concurrent_writes(self, stress_db):
        """Test system with many concurrent write operations."""
        metrics = StressTestMetrics()
        monitor = ResourceMonitor()
        
        num_writers = 50
        writes_per_writer = 20
        
        monitor.start()
        metrics.start_time = time.time()
        
        async def write_task(writer_id: int):
            """Individual write task."""
            successes = 0
            for i in range(writes_per_writer):
                try:
                    start = time.perf_counter()
                    doc_id = f"writer_{writer_id}_doc_{i}_{uuid.uuid4()}"
                    stress_db.add_document(
                        "campaigns",
                        doc_id,
                        f"Content from writer {writer_id}",
                        {"writer": writer_id, "index": i}
                    )
                    elapsed = time.perf_counter() - start
                    metrics.response_times.append(elapsed)
                    successes += 1
                except Exception as e:
                    metrics.errors.append(f"Writer {writer_id}: {str(e)}")
            return successes
        
        # Run concurrent writes
        tasks = [
            asyncio.create_task(write_task(i))
            for i in range(num_writers)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.end_time = time.time()
        monitor.stop()
        
        # Calculate metrics
        metrics.total_operations = num_writers * writes_per_writer
        metrics.successful_operations = sum(r for r in results if isinstance(r, int))
        metrics.failed_operations = metrics.total_operations - metrics.successful_operations
        
        metrics.peak_memory_mb = monitor.peak_memory
        metrics.peak_cpu_percent = monitor.peak_cpu
        metrics.peak_threads = monitor.peak_threads
        
        print(metrics.report())
        
        # Assertions
        assert metrics.success_rate > 90  # At least 90% success rate
        assert metrics.throughput > 50  # At least 50 writes per second
    
    @pytest.mark.asyncio
    async def test_mixed_workload(self, stress_db):
        """Test system with mixed read/write workload."""
        search_service = SearchService(stress_db)
        campaign_manager = CampaignManager(stress_db)
        session_manager = SessionManager(stress_db)
        
        metrics = StressTestMetrics()
        monitor = ResourceMonitor()
        
        monitor.start()
        metrics.start_time = time.time()
        
        async def mixed_operation(op_id: int):
            """Perform mixed operations."""
            op_type = op_id % 4
            try:
                start = time.perf_counter()
                
                if op_type == 0:  # Search
                    await search_service.search("test", "rulebooks", max_results=5)
                elif op_type == 1:  # Create campaign
                    await campaign_manager.create_campaign(
                        name=f"Campaign {op_id}",
                        system="D&D 5e",
                        description="Stress test campaign"
                    )
                elif op_type == 2:  # Create session
                    await session_manager.create_session(
                        campaign_id=f"campaign_{op_id % 10}",
                        title=f"Session {op_id}",
                        session_number=op_id
                    )
                else:  # Add document
                    stress_db.add_document(
                        "rulebooks",
                        f"mixed_doc_{op_id}",
                        f"Mixed content {op_id}",
                        {"op": op_id}
                    )
                
                elapsed = time.perf_counter() - start
                metrics.response_times.append(elapsed)
                metrics.successful_operations += 1
                return True
                
            except Exception as e:
                metrics.failed_operations += 1
                metrics.errors.append(f"Op {op_id}: {str(e)}")
                return False
        
        # Run mixed workload
        num_operations = 200
        tasks = [
            asyncio.create_task(mixed_operation(i))
            for i in range(num_operations)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.end_time = time.time()
        monitor.stop()
        
        metrics.total_operations = num_operations
        metrics.peak_memory_mb = monitor.peak_memory
        metrics.peak_cpu_percent = monitor.peak_cpu
        metrics.peak_threads = monitor.peak_threads
        
        print(metrics.report())
        
        # Assertions
        assert metrics.success_rate > 85  # Mixed workload harder, 85% acceptable
        assert metrics.avg_response_time < 2.0  # Average under 2 seconds


class TestResourceExhaustion:
    """Test system behavior under resource exhaustion."""
    
    @pytest.mark.asyncio
    async def test_memory_pressure(self, tmp_path):
        """Test system under memory pressure."""
        metrics = StressTestMetrics()
        monitor = ResourceMonitor()
        
        with patch.object(settings, "chroma_db_path", tmp_path / "memory_stress"):
            db = ChromaDBManager()
            
            monitor.start()
            metrics.start_time = time.time()
            
            # Create large documents to stress memory
            large_content = "x" * (1024 * 1024)  # 1MB per document
            
            try:
                for i in range(100):  # Try to use ~100MB
                    try:
                        db.add_document(
                            "rulebooks",
                            f"large_doc_{i}",
                            large_content,
                            {"index": i}
                        )
                        metrics.successful_operations += 1
                    except Exception as e:
                        metrics.failed_operations += 1
                        metrics.errors.append(str(e))
                        
                        # Force garbage collection
                        gc.collect()
                    
                    metrics.total_operations += 1
                    
                    # Check memory usage
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    if current_memory > 1000:  # If over 1GB, start being careful
                        gc.collect()
                        await asyncio.sleep(0.1)
                
            finally:
                metrics.end_time = time.time()
                monitor.stop()
                
                if hasattr(db, "cleanup"):
                    await db.cleanup()
            
            metrics.peak_memory_mb = monitor.peak_memory
            metrics.peak_cpu_percent = monitor.peak_cpu
            metrics.peak_threads = monitor.peak_threads
            
            print(metrics.report())
            
            # System should handle memory pressure gracefully
            assert metrics.success_rate > 50  # At least half should succeed
            assert metrics.peak_memory_mb < 3000  # Should not exceed 3GB
    
    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self, tmp_path):
        """Test system with connection pool exhaustion."""
        metrics = StressTestMetrics()
        
        with patch.object(settings, "chroma_db_path", tmp_path / "conn_stress"):
            # Simulate limited connection pool
            with patch.object(settings, "max_connections", 10):
                db = ChromaDBManager()
                
                metrics.start_time = time.time()
                
                async def connection_task(task_id: int):
                    """Task that holds a connection."""
                    try:
                        # Simulate holding connection
                        await asyncio.sleep(random.uniform(0.1, 0.5))
                        
                        # Perform operation
                        db.add_document(
                            "sessions",
                            f"conn_doc_{task_id}",
                            f"Connection test {task_id}",
                            {"task": task_id}
                        )
                        metrics.successful_operations += 1
                        return True
                    except Exception as e:
                        metrics.failed_operations += 1
                        metrics.errors.append(f"Connection {task_id}: {str(e)}")
                        return False
                
                # Try to use more connections than available
                num_tasks = 50
                tasks = [
                    asyncio.create_task(connection_task(i))
                    for i in range(num_tasks)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                metrics.end_time = time.time()
                metrics.total_operations = num_tasks
                
                print(metrics.report())
                
                # Should handle connection exhaustion gracefully
                assert metrics.success_rate > 70  # Most should eventually succeed
                
                if hasattr(db, "cleanup"):
                    await db.cleanup()
    
    @pytest.mark.asyncio
    async def test_cpu_intensive_load(self):
        """Test system under CPU-intensive load."""
        processor = ParallelProcessor(max_workers=4)
        metrics = StressTestMetrics()
        monitor = ResourceMonitor()
        
        monitor.start()
        metrics.start_time = time.time()
        
        def cpu_intensive_task(n: int) -> int:
            """CPU-intensive calculation."""
            result = 0
            for i in range(n * 10000):
                result += i ** 2
            return result
        
        # Create many CPU-intensive tasks
        num_tasks = 100
        tasks = [cpu_intensive_task for _ in range(num_tasks)]
        args = [(100,) for _ in range(num_tasks)]
        
        try:
            results = await processor.run_parallel(tasks, args)
            metrics.successful_operations = len([r for r in results if r is not None])
            metrics.failed_operations = num_tasks - metrics.successful_operations
        except Exception as e:
            metrics.errors.append(str(e))
            metrics.failed_operations = num_tasks
        
        metrics.end_time = time.time()
        monitor.stop()
        
        metrics.total_operations = num_tasks
        metrics.peak_memory_mb = monitor.peak_memory
        metrics.peak_cpu_percent = monitor.peak_cpu
        metrics.peak_threads = monitor.peak_threads
        
        print(metrics.report())
        
        # Should handle CPU load without crashing
        assert metrics.success_rate > 80
        assert metrics.duration < 60  # Should complete within 1 minute


class TestCascadeFailurePrevention:
    """Test prevention of cascade failures."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self, tmp_path):
        """Test circuit breaker pattern under failures."""
        metrics = StressTestMetrics()
        
        class CircuitBreaker:
            """Simple circuit breaker implementation."""
            def __init__(self, failure_threshold=5, timeout=1.0):
                self.failure_count = 0
                self.failure_threshold = failure_threshold
                self.timeout = timeout
                self.last_failure_time = 0
                self.is_open = False
                
            async def call(self, func, *args, **kwargs):
                """Call function with circuit breaker protection."""
                # Check if circuit is open
                if self.is_open:
                    if time.time() - self.last_failure_time > self.timeout:
                        self.is_open = False  # Try to close circuit
                        self.failure_count = 0
                    else:
                        raise Exception("Circuit breaker is open")
                
                try:
                    result = await func(*args, **kwargs)
                    self.failure_count = 0  # Reset on success
                    return result
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= self.failure_threshold:
                        self.is_open = True
                    
                    raise e
        
        with patch.object(settings, "chroma_db_path", tmp_path / "circuit_test"):
            db = ChromaDBManager()
            circuit_breaker = CircuitBreaker()
            
            metrics.start_time = time.time()
            
            # Simulate service that fails periodically
            failure_counter = 0
            
            async def unreliable_operation(op_id: int):
                """Operation that fails periodically."""
                nonlocal failure_counter
                failure_counter += 1
                
                # Fail every 3rd call initially
                if failure_counter <= 15 and failure_counter % 3 == 0:
                    raise Exception("Service temporarily unavailable")
                
                # Succeed otherwise
                return db.add_document(
                    "rulebooks",
                    f"circuit_doc_{op_id}",
                    f"Content {op_id}",
                    {"op": op_id}
                )
            
            # Test operations with circuit breaker
            for i in range(50):
                try:
                    await circuit_breaker.call(unreliable_operation, i)
                    metrics.successful_operations += 1
                except Exception as e:
                    metrics.failed_operations += 1
                    metrics.errors.append(str(e))
                
                metrics.total_operations += 1
                
                # Small delay between operations
                await asyncio.sleep(0.05)
            
            metrics.end_time = time.time()
            
            print(metrics.report())
            
            # Circuit breaker should prevent cascade failure
            assert metrics.total_operations == 50
            assert metrics.success_rate > 60  # Should recover after initial failures
            
            if hasattr(db, "cleanup"):
                await db.cleanup()
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, tmp_path):
        """Test graceful degradation under stress."""
        metrics = StressTestMetrics()
        
        with patch.object(settings, "chroma_db_path", tmp_path / "degrade_test"):
            db = ChromaDBManager()
            search_service = SearchService(db)
            
            # Add test data
            for i in range(50):
                db.add_document(
                    "rulebooks",
                    f"degrade_doc_{i}",
                    f"Document content {i}",
                    {"index": i}
                )
            
            metrics.start_time = time.time()
            
            # Simulate degraded service levels
            degradation_levels = {
                "full": {"max_results": 10, "use_cache": True},
                "degraded": {"max_results": 5, "use_cache": True},
                "minimal": {"max_results": 1, "use_cache": False}
            }
            
            current_level = "full"
            consecutive_failures = 0
            
            for i in range(100):
                try:
                    # Adjust service level based on failures
                    config = degradation_levels[current_level]
                    
                    # Simulate random failures under load
                    if random.random() < 0.1:  # 10% failure rate
                        raise Exception("Random failure")
                    
                    # Perform search with current service level
                    await search_service.search(
                        query="test",
                        collection_name="rulebooks",
                        max_results=config["max_results"]
                    )
                    
                    metrics.successful_operations += 1
                    consecutive_failures = 0
                    
                    # Try to upgrade service level
                    if current_level == "minimal" and consecutive_failures == 0:
                        current_level = "degraded"
                    elif current_level == "degraded" and consecutive_failures == 0:
                        current_level = "full"
                        
                except Exception as e:
                    metrics.failed_operations += 1
                    consecutive_failures += 1
                    
                    # Degrade service level
                    if consecutive_failures >= 3:
                        if current_level == "full":
                            current_level = "degraded"
                        elif current_level == "degraded":
                            current_level = "minimal"
                
                metrics.total_operations += 1
            
            metrics.end_time = time.time()
            
            print(metrics.report())
            print(f"Final service level: {current_level}")
            
            # Should maintain service despite failures
            assert metrics.success_rate > 85
            
            if hasattr(db, "cleanup"):
                await db.cleanup()


class TestRecoveryMechanisms:
    """Test system recovery mechanisms."""
    
    @pytest.mark.asyncio
    async def test_automatic_retry_with_backoff(self):
        """Test automatic retry with exponential backoff."""
        metrics = StressTestMetrics()
        
        class RetryManager:
            """Manage retries with exponential backoff."""
            def __init__(self, max_retries=3, base_delay=0.1):
                self.max_retries = max_retries
                self.base_delay = base_delay
                
            async def execute_with_retry(self, func, *args, **kwargs):
                """Execute function with retry logic."""
                last_exception = None
                
                for attempt in range(self.max_retries):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < self.max_retries - 1:
                            delay = self.base_delay * (2 ** attempt)
                            await asyncio.sleep(delay)
                
                raise last_exception
        
        retry_manager = RetryManager()
        failure_count = 0
        
        async def flaky_operation():
            """Operation that fails initially then succeeds."""
            nonlocal failure_count
            failure_count += 1
            
            if failure_count <= 2:
                raise Exception(f"Temporary failure {failure_count}")
            
            return {"status": "success", "attempts": failure_count}
        
        metrics.start_time = time.time()
        
        # Test retry mechanism
        for i in range(20):
            failure_count = 0
            try:
                result = await retry_manager.execute_with_retry(flaky_operation)
                metrics.successful_operations += 1
            except Exception as e:
                metrics.failed_operations += 1
                metrics.errors.append(str(e))
            
            metrics.total_operations += 1
        
        metrics.end_time = time.time()
        
        print(metrics.report())
        
        # Retry mechanism should improve success rate
        assert metrics.success_rate > 90
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_after_stress(self, tmp_path):
        """Test resource cleanup after stress conditions."""
        with patch.object(settings, "chroma_db_path", tmp_path / "cleanup_test"):
            db = ChromaDBManager()
            cache_manager = GlobalCacheManager()
            
            # Create stress conditions
            for i in range(100):
                db.add_document(
                    "sessions",
                    f"cleanup_doc_{i}",
                    f"Content {i}" * 100,
                    {"index": i}
                )
                cache_manager.set(f"key_{i}", f"value_{i}" * 100)
            
            # Check resource usage before cleanup
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Perform cleanup
            cache_manager.clear_all()
            if hasattr(db, "cleanup"):
                await db.cleanup()
            gc.collect()
            
            # Check resource usage after cleanup
            await asyncio.sleep(1)  # Allow cleanup to complete
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            
            memory_freed = memory_before - memory_after
            print(f"Memory freed: {memory_freed:.2f} MB")
            
            # Should free significant memory
            assert memory_freed > 0


class TestLongRunningStress:
    """Test system under long-running stress conditions."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow  # Mark as slow test
    async def test_sustained_load(self, tmp_path):
        """Test system under sustained load for extended period."""
        metrics = StressTestMetrics()
        monitor = ResourceMonitor()
        
        with patch.object(settings, "chroma_db_path", tmp_path / "sustained_test"):
            db = ChromaDBManager()
            search_service = SearchService(db)
            
            # Add initial data
            for i in range(100):
                db.add_document(
                    "rulebooks",
                    f"sustained_doc_{i}",
                    f"Initial content {i}",
                    {"index": i}
                )
            
            monitor.start()
            metrics.start_time = time.time()
            
            # Run sustained operations for 30 seconds
            end_time = time.time() + 30
            operation_count = 0
            
            while time.time() < end_time:
                try:
                    # Mix of operations
                    if operation_count % 3 == 0:
                        # Search
                        await search_service.search("content", "rulebooks", max_results=5)
                    elif operation_count % 3 == 1:
                        # Write
                        db.add_document(
                            "rulebooks",
                            f"sustained_new_{operation_count}",
                            f"New content {operation_count}",
                            {"op": operation_count}
                        )
                    else:
                        # Update
                        db.update_document(
                            "rulebooks",
                            f"sustained_doc_{operation_count % 100}",
                            f"Updated content {operation_count}",
                            {"updated": True, "op": operation_count}
                        )
                    
                    metrics.successful_operations += 1
                    
                except Exception as e:
                    metrics.failed_operations += 1
                    metrics.errors.append(str(e))
                
                metrics.total_operations += 1
                operation_count += 1
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
            
            metrics.end_time = time.time()
            monitor.stop()
            
            metrics.peak_memory_mb = monitor.peak_memory
            metrics.peak_cpu_percent = monitor.peak_cpu
            metrics.peak_threads = monitor.peak_threads
            
            print(metrics.report())
            
            # System should remain stable over time
            assert metrics.success_rate > 95
            assert metrics.peak_memory_mb < 2000  # Memory should not grow unbounded
            
            if hasattr(db, "cleanup"):
                await db.cleanup()


# Test result aggregation
@pytest.fixture(scope="module")
def stress_test_summary():
    """Aggregate stress test results."""
    summary = {
        "concurrent_load": {},
        "resource_exhaustion": {},
        "cascade_prevention": {},
        "recovery": {},
        "sustained": {}
    }
    yield summary
    
    # Print summary
    print("\n" + "="*50)
    print("STRESS TEST SUMMARY")
    print("="*50)
    
    for category, results in summary.items():
        if results:
            print(f"\n{category.upper()}:")
            for metric, value in results.items():
                print(f"  {metric}: {value}")