"""
Performance Testing Framework for Provider Router with Fallback System.

This module provides comprehensive performance testing with benchmarks for
sub-100ms routing decision latency, concurrent request handling (1000+ RPS),
memory usage under load, cache performance and hit rates, and provider failover
speed and recovery time.

Test Coverage:
- Sub-100ms routing decision latency benchmarks
- Concurrent request handling (1000+ RPS) 
- Memory usage profiling under load
- Cache performance and hit rate analysis
- Provider failover speed and recovery time
- Throughput and latency percentile measurements
- Resource utilization monitoring
- Scalability and stress testing
"""

import asyncio
import gc
import json
import psutil
import statistics
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

from src.ai_providers.models import (
    AIRequest, AIResponse, ProviderType, ProviderCapability, ModelSpec
)
from src.ai_providers.abstract_provider import AbstractProvider
from src.context.provider_router_context_manager import (
    ProviderRouterContextManager,
    ProviderHealthState,
    CircuitBreakerState,
    RoutingDecision,
    StateConsistencyLevel,
    InMemoryStateCache
)
from src.context.provider_router_performance_optimization import (
    ProviderRouterPerformanceOptimizer,
    PerformanceTarget,
    benchmark_state_operations
)


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    operation_name: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    
    # Latency metrics (milliseconds)
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    
    # Throughput metrics
    operations_per_second: float
    total_duration_s: float
    
    # Resource metrics
    peak_memory_mb: float
    avg_cpu_percent: float
    
    # Error metrics
    error_rate: float
    timeout_count: int
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "operation_name": self.operation_name,
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "operations_per_second": self.operations_per_second,
            "total_duration_s": self.total_duration_s,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_cpu_percent": self.avg_cpu_percent,
            "error_rate": self.error_rate,
            "timeout_count": self.timeout_count,
            "metadata": self.metadata
        }


class PerformanceTestProvider(AbstractProvider):
    """Optimized provider for performance testing."""
    
    def __init__(self, provider_type: ProviderType, latency_ms: float = 10.0, 
                 error_rate: float = 0.0, enable_metrics: bool = True):
        from src.ai_providers.models import ProviderConfig
        config = ProviderConfig(provider_type=provider_type, api_key="perf-test-key")
        super().__init__(config)
        
        self.base_latency_ms = latency_ms
        self.error_rate = error_rate
        self.enable_metrics = enable_metrics
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.min_latency = float('inf')
        self.max_latency = 0.0
        
        self._models = {
            f"{provider_type.value}-perf": ModelSpec(
                model_id=f"{provider_type.value}-perf",
                provider_type=provider_type,
                display_name=f"{provider_type.value.title()} Performance Model",
                capabilities=[ProviderCapability.TEXT_GENERATION, ProviderCapability.STREAMING],
                cost_per_input_token=0.0001,  # Low cost for performance testing
                cost_per_output_token=0.0002,
                supports_streaming=True,
                supports_tools=True,
                context_length=8192,
                max_output_tokens=4096,
                is_available=True
            )
        }
    
    async def _initialize_client(self):
        await asyncio.sleep(0.001)  # Minimal initialization time
    
    async def _cleanup_client(self):
        pass
    
    async def _load_models(self):
        pass
    
    async def _generate_response_impl(self, request: AIRequest) -> AIResponse:
        start_time = time.perf_counter()
        self.request_count += 1
        
        # Simulate minimal processing latency
        await asyncio.sleep(self.base_latency_ms / 1000.0)
        
        # Simulate errors
        if self.error_rate > 0 and (self.request_count * self.error_rate) >= 1:
            self.error_count += 1
            from src.ai_providers.error_handler import AIProviderError
            raise AIProviderError(f"Performance test error from {self.provider_type.value}")
        
        actual_latency = (time.perf_counter() - start_time) * 1000
        
        if self.enable_metrics:
            self.success_count += 1
            self.total_latency += actual_latency
            self.min_latency = min(self.min_latency, actual_latency)
            self.max_latency = max(self.max_latency, actual_latency)
        
        return AIResponse(
            request_id=request.request_id,
            provider_type=self.provider_type,
            model=request.model,
            content=f"Performance test response from {self.provider_type.value}",
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            cost=0.003,  # Low cost
            latency_ms=actual_latency,
            metadata={"perf_test": True}
        )
    
    async def _stream_response_impl(self, request: AIRequest):
        from src.ai_providers.models import StreamingChunk
        chunks = ["Performance", " test", " streaming"]
        
        for i, chunk in enumerate(chunks):
            await asyncio.sleep(self.base_latency_ms / len(chunks) / 1000.0)
            yield StreamingChunk(
                request_id=request.request_id,
                content=chunk,
                is_complete=(i == len(chunks) - 1),
                finish_reason="stop" if i == len(chunks) - 1 else None
            )
    
    def _get_supported_capabilities(self):
        return [ProviderCapability.TEXT_GENERATION, ProviderCapability.STREAMING]
    
    async def _perform_health_check(self):
        await asyncio.sleep(0.001)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get provider performance metrics."""
        if self.success_count > 0:
            avg_latency = self.total_latency / self.success_count
        else:
            avg_latency = 0.0
        
        return {
            "requests": self.request_count,
            "successes": self.success_count,
            "errors": self.error_count,
            "avg_latency_ms": avg_latency,
            "min_latency_ms": self.min_latency if self.min_latency != float('inf') else 0,
            "max_latency_ms": self.max_latency,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0
        }


class PerformanceBenchmark:
    """Performance benchmark runner."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.operation_times = []
        self.memory_samples = []
        self.cpu_samples = []
        self.successful_operations = 0
        self.failed_operations = 0
        self.timeout_count = 0
        self.process = psutil.Process()
        
    async def __aenter__(self):
        """Start benchmark."""
        self.start_time = time.perf_counter()
        gc.collect()  # Clean up before benchmark
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End benchmark."""
        self.end_time = time.perf_counter()
    
    def record_operation(self, duration_ms: float, success: bool = True, timeout: bool = False):
        """Record a single operation."""
        self.operation_times.append(duration_ms)
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
        if timeout:
            self.timeout_count += 1
    
    def sample_resources(self):
        """Sample current resource usage."""
        try:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()
            self.memory_samples.append(memory_mb)
            self.cpu_samples.append(cpu_percent)
        except Exception:
            pass  # Ignore sampling errors
    
    def get_metrics(self) -> PerformanceMetrics:
        """Calculate performance metrics."""
        total_duration = (self.end_time - self.start_time) if self.end_time else 0
        total_operations = len(self.operation_times)
        
        if total_operations > 0:
            sorted_times = sorted(self.operation_times)
            avg_latency = statistics.mean(self.operation_times)
            p50_latency = statistics.median(self.operation_times)
            p95_latency = sorted_times[int(0.95 * len(sorted_times))] if len(sorted_times) > 1 else sorted_times[0]
            p99_latency = sorted_times[int(0.99 * len(sorted_times))] if len(sorted_times) > 1 else sorted_times[0]
            min_latency = min(self.operation_times)
            max_latency = max(self.operation_times)
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = min_latency = max_latency = 0.0
        
        ops_per_second = total_operations / total_duration if total_duration > 0 else 0
        error_rate = self.failed_operations / total_operations if total_operations > 0 else 0
        peak_memory = max(self.memory_samples) if self.memory_samples else 0
        avg_cpu = statistics.mean(self.cpu_samples) if self.cpu_samples else 0
        
        return PerformanceMetrics(
            operation_name=self.name,
            total_operations=total_operations,
            successful_operations=self.successful_operations,
            failed_operations=self.failed_operations,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            operations_per_second=ops_per_second,
            total_duration_s=total_duration,
            peak_memory_mb=peak_memory,
            avg_cpu_percent=avg_cpu,
            error_rate=error_rate,
            timeout_count=self.timeout_count
        )


@pytest.fixture
async def high_performance_context_manager():
    """High-performance context manager for benchmarking."""
    # Use in-memory mocks for maximum performance
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.setex = AsyncMock(return_value=True)
    mock_redis.mget = AsyncMock(return_value=[])
    mock_redis.keys = AsyncMock(return_value=[])
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.publish = AsyncMock(return_value=1)
    mock_redis.pipeline = AsyncMock()
    mock_redis.close = AsyncMock()
    
    mock_chroma = Mock()
    mock_collection = Mock()
    mock_collection.add = Mock()
    mock_collection.upsert = Mock()
    mock_collection.get = Mock(return_value={"ids": [], "metadatas": []})
    mock_collection.count = Mock(return_value=0)
    mock_chroma.get_or_create_collection = Mock(return_value=mock_collection)
    
    with patch('redis.asyncio.from_url', return_value=mock_redis), \
         patch('chromadb.HttpClient', return_value=mock_chroma):
        
        cm = ProviderRouterContextManager(
            chroma_host="localhost",
            chroma_port=8000,
            redis_url="redis://localhost:6379/0",
            enable_recovery=False,  # Disable for performance
            cache_size=10000  # Large cache for performance
        )
        
        await cm.initialize()
        await cm.start()
        
        yield cm
        
        await cm.stop()


@pytest.fixture
def performance_providers():
    """High-performance test providers."""
    return {
        "fast": PerformanceTestProvider(ProviderType.ANTHROPIC, latency_ms=5.0),
        "medium": PerformanceTestProvider(ProviderType.OPENAI, latency_ms=25.0),
        "slow": PerformanceTestProvider(ProviderType.GOOGLE, latency_ms=75.0),
        "unreliable": PerformanceTestProvider(ProviderType.ANTHROPIC, latency_ms=10.0, error_rate=0.1)
    }


class TestRoutingPerformance:
    """Test routing decision performance benchmarks."""
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_sub_100ms_routing_decision_benchmark(self, high_performance_context_manager):
        """Benchmark routing decisions for sub-100ms requirement."""
        cm = high_performance_context_manager
        
        # Pre-populate with provider health data
        providers = ["anthropic", "openai", "google"]
        for i, provider in enumerate(providers):
            await cm.update_provider_health(
                provider,
                {
                    "provider_type": provider,
                    "is_available": True,
                    "response_time_ms": 50.0 + (i * 20),
                    "error_rate": 0.01,
                    "success_rate": 0.99,
                    "uptime_percentage": 99.9,
                    "consecutive_failures": 0,
                    "circuit_breaker_state": "CLOSED",
                    "metadata": {"benchmark": True}
                }
            )
        
        # Warm up caches
        for _ in range(10):
            for provider in providers:
                await cm.get_provider_health(provider)
        
        async with PerformanceBenchmark("routing_decision") as benchmark:
            num_decisions = 1000
            
            for i in range(num_decisions):
                start = time.perf_counter()
                
                # Simulate routing decision process
                health_states = []
                for provider in providers:
                    health = await cm.get_provider_health(provider)
                    if health:
                        health_states.append((provider, health.response_time_ms))
                
                # Select best provider (simple logic for benchmark)
                if health_states:
                    best_provider = min(health_states, key=lambda x: x[1])[0]
                    
                    # Store routing decision
                    await cm.store_routing_decision(
                        f"benchmark-{i}",
                        {
                            "selected_provider": best_provider,
                            "alternative_providers": [p for p in providers if p != best_provider],
                            "routing_strategy": "benchmark",
                            "decision_factors": {"iteration": i},
                            "estimated_cost": 0.1,
                            "estimated_latency_ms": health_states[0][1],
                            "confidence_score": 0.95,
                            "fallback_chain": providers
                        }
                    )
                
                duration_ms = (time.perf_counter() - start) * 1000
                benchmark.record_operation(duration_ms)
                
                # Sample resources every 100 operations
                if i % 100 == 0:
                    benchmark.sample_resources()
        
        metrics = benchmark.get_metrics()
        
        # Performance assertions for sub-100ms requirement
        assert metrics.p95_latency_ms < 100.0, f"P95 latency {metrics.p95_latency_ms:.2f}ms exceeds 100ms requirement"
        assert metrics.avg_latency_ms < 50.0, f"Average latency {metrics.avg_latency_ms:.2f}ms too high"
        assert metrics.p99_latency_ms < 150.0, f"P99 latency {metrics.p99_latency_ms:.2f}ms exceeds threshold"
        
        # Throughput assertion
        assert metrics.operations_per_second > 100, f"Throughput {metrics.operations_per_second:.0f} ops/sec too low"
        
        print(f"Routing Performance: avg={metrics.avg_latency_ms:.2f}ms, "
              f"p95={metrics.p95_latency_ms:.2f}ms, p99={metrics.p99_latency_ms:.2f}ms, "
              f"throughput={metrics.operations_per_second:.0f} ops/sec")
        
        return metrics
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_concurrent_routing_performance(self, high_performance_context_manager):
        """Test concurrent routing performance."""
        cm = high_performance_context_manager
        
        # Setup providers
        providers = ["anthropic", "openai", "google"]
        for provider in providers:
            await cm.update_provider_health(
                provider,
                {
                    "provider_type": provider,
                    "is_available": True,
                    "response_time_ms": 40.0,
                    "error_rate": 0.0,
                    "success_rate": 1.0,
                    "uptime_percentage": 100.0,
                    "consecutive_failures": 0,
                    "circuit_breaker_state": "CLOSED"
                }
            )
        
        async def routing_worker(worker_id: int, num_requests: int) -> List[float]:
            """Worker that performs routing decisions."""
            times = []
            
            for i in range(num_requests):
                start = time.perf_counter()
                
                # Get provider health
                health = await cm.get_provider_health("anthropic")
                
                # Store routing decision
                await cm.store_routing_decision(
                    f"concurrent-{worker_id}-{i}",
                    {
                        "selected_provider": "anthropic",
                        "alternative_providers": ["openai", "google"],
                        "routing_strategy": "concurrent_test",
                        "decision_factors": {"worker_id": worker_id, "request_index": i},
                        "estimated_cost": 0.1,
                        "estimated_latency_ms": health.response_time_ms if health else 50.0,
                        "confidence_score": 0.9,
                        "fallback_chain": ["openai", "google"]
                    }
                )
                
                duration_ms = (time.perf_counter() - start) * 1000
                times.append(duration_ms)
            
            return times
        
        # Run concurrent workers
        num_workers = 20
        requests_per_worker = 100
        
        start_time = time.perf_counter()
        tasks = [routing_worker(i, requests_per_worker) for i in range(num_workers)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time
        
        # Analyze results
        all_times = [time for worker_times in results for time in worker_times]
        total_requests = len(all_times)
        
        metrics = PerformanceMetrics(
            operation_name="concurrent_routing",
            total_operations=total_requests,
            successful_operations=total_requests,
            failed_operations=0,
            avg_latency_ms=statistics.mean(all_times),
            p50_latency_ms=statistics.median(all_times),
            p95_latency_ms=statistics.quantiles(all_times, n=20)[18] if len(all_times) > 20 else max(all_times),
            p99_latency_ms=statistics.quantiles(all_times, n=100)[98] if len(all_times) > 100 else max(all_times),
            min_latency_ms=min(all_times),
            max_latency_ms=max(all_times),
            operations_per_second=total_requests / total_time,
            total_duration_s=total_time,
            peak_memory_mb=0,  # Not measured in this test
            avg_cpu_percent=0,
            error_rate=0.0,
            timeout_count=0
        )
        
        # Performance assertions
        assert metrics.operations_per_second >= 1000, f"Concurrent throughput {metrics.operations_per_second:.0f} RPS below 1000 requirement"
        assert metrics.p95_latency_ms < 100.0, f"Concurrent P95 latency {metrics.p95_latency_ms:.2f}ms too high"
        assert metrics.avg_latency_ms < 50.0, f"Concurrent average latency {metrics.avg_latency_ms:.2f}ms too high"
        
        print(f"Concurrent Performance: {metrics.operations_per_second:.0f} RPS, "
              f"avg={metrics.avg_latency_ms:.2f}ms, p95={metrics.p95_latency_ms:.2f}ms")
        
        return metrics
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark 
    async def test_cache_performance_benchmark(self, high_performance_context_manager):
        """Benchmark cache performance and hit rates."""
        cm = high_performance_context_manager
        
        # Pre-populate cache
        cache_entries = 1000
        for i in range(cache_entries):
            await cm.update_provider_health(
                f"provider-{i}",
                {
                    "provider_type": "test",
                    "is_available": True,
                    "response_time_ms": 50.0,
                    "error_rate": 0.0,
                    "success_rate": 1.0,
                    "uptime_percentage": 100.0,
                    "consecutive_failures": 0,
                    "circuit_breaker_state": "CLOSED"
                }
            )
        
        # Benchmark cache access patterns
        async with PerformanceBenchmark("cache_access") as benchmark:
            num_accesses = 10000
            cache_hits = 0
            
            for i in range(num_accesses):
                start = time.perf_counter()
                
                # Mix of hot and cold cache accesses
                if i % 10 < 8:  # 80% hot cache access
                    provider_id = f"provider-{i % 100}"  # Access first 100 frequently
                else:  # 20% cold cache access
                    provider_id = f"provider-{i % cache_entries}"
                
                health = await cm.get_provider_health(provider_id)
                
                duration_ms = (time.perf_counter() - start) * 1000
                benchmark.record_operation(duration_ms, success=health is not None)
                
                if health is not None:
                    cache_hits += 1
                
                if i % 1000 == 0:
                    benchmark.sample_resources()
        
        metrics = benchmark.get_metrics()
        cache_hit_rate = cache_hits / num_accesses
        
        # Cache performance assertions
        assert cache_hit_rate >= 0.95, f"Cache hit rate {cache_hit_rate:.2%} too low"
        assert metrics.avg_latency_ms < 5.0, f"Cache access average latency {metrics.avg_latency_ms:.2f}ms too high"
        assert metrics.p95_latency_ms < 20.0, f"Cache access P95 latency {metrics.p95_latency_ms:.2f}ms too high"
        assert metrics.operations_per_second > 5000, f"Cache throughput {metrics.operations_per_second:.0f} ops/sec too low"
        
        # Get cache statistics
        cache_stats = cm.memory_cache.get_stats()
        
        print(f"Cache Performance: hit_rate={cache_hit_rate:.2%}, "
              f"avg_latency={metrics.avg_latency_ms:.2f}ms, "
              f"throughput={metrics.operations_per_second:.0f} ops/sec, "
              f"cache_size={cache_stats['size']}")
        
        metrics.metadata.update({
            "cache_hit_rate": cache_hit_rate,
            "cache_stats": cache_stats
        })
        
        return metrics


class TestThroughputAndLatency:
    """Test system throughput and latency under various conditions."""
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    @pytest.mark.load
    async def test_high_throughput_sustained_load(self, high_performance_context_manager, performance_providers):
        """Test sustained high throughput load."""
        cm = high_performance_context_manager
        provider = performance_providers["fast"]
        await provider.initialize()
        
        # Initialize provider health
        await cm.update_provider_health(
            provider.provider_type.value,
            {
                "provider_type": provider.provider_type.value,
                "is_available": True,
                "response_time_ms": provider.base_latency_ms,
                "error_rate": 0.0,
                "success_rate": 1.0,
                "uptime_percentage": 100.0,
                "consecutive_failures": 0,
                "circuit_breaker_state": "CLOSED"
            }
        )
        
        async def sustained_load_worker(duration_s: int) -> Dict[str, Any]:
            """Worker that generates sustained load."""
            end_time = time.time() + duration_s
            operations = 0
            successful = 0
            times = []
            
            while time.time() < end_time:
                start = time.perf_counter()
                
                try:
                    # Routing decision + provider call simulation
                    health = await cm.get_provider_health(provider.provider_type.value)
                    
                    request = AIRequest(
                        model=f"{provider.provider_type.value}-perf",
                        messages=[{"role": "user", "content": f"Load test {operations}"}],
                        max_tokens=100
                    )
                    
                    response = await provider._generate_response_impl(request)
                    successful += 1
                    
                    # Store routing decision
                    await cm.store_routing_decision(
                        request.request_id,
                        {
                            "selected_provider": provider.provider_type.value,
                            "alternative_providers": [],
                            "routing_strategy": "sustained_load",
                            "decision_factors": {"operation": operations},
                            "estimated_cost": response.cost,
                            "estimated_latency_ms": response.latency_ms,
                            "confidence_score": 1.0,
                            "fallback_chain": []
                        }
                    )
                    
                except Exception as e:
                    pass  # Continue load test despite errors
                
                duration_ms = (time.perf_counter() - start) * 1000
                times.append(duration_ms)
                operations += 1
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.001)
            
            return {
                "operations": operations,
                "successful": successful,
                "times": times,
                "duration": duration_s
            }
        
        # Run sustained load test
        load_duration_s = 10  # 10 second load test
        num_workers = 10
        
        start_time = time.perf_counter()
        tasks = [sustained_load_worker(load_duration_s) for _ in range(num_workers)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.perf_counter() - start_time
        
        # Analyze results
        total_operations = sum(r["operations"] for r in results if not isinstance(r, Exception))
        total_successful = sum(r["successful"] for r in results if not isinstance(r, Exception))
        all_times = []
        for r in results:
            if not isinstance(r, Exception):
                all_times.extend(r["times"])
        
        # Calculate metrics
        sustained_throughput = total_operations / total_time
        success_rate = total_successful / total_operations if total_operations > 0 else 0
        
        if all_times:
            avg_latency = statistics.mean(all_times)
            p95_latency = statistics.quantiles(all_times, n=20)[18] if len(all_times) > 20 else max(all_times)
            p99_latency = statistics.quantiles(all_times, n=100)[98] if len(all_times) > 100 else max(all_times)
        else:
            avg_latency = p95_latency = p99_latency = 0
        
        # Performance assertions for sustained load
        assert sustained_throughput >= 500, f"Sustained throughput {sustained_throughput:.0f} ops/sec too low"
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} too low under sustained load"
        assert avg_latency < 100.0, f"Average latency {avg_latency:.2f}ms too high under load"
        assert p95_latency < 200.0, f"P95 latency {p95_latency:.2f}ms too high under load"
        
        print(f"Sustained Load: {sustained_throughput:.0f} ops/sec, "
              f"success_rate={success_rate:.2%}, "
              f"avg_latency={avg_latency:.2f}ms, p95_latency={p95_latency:.2f}ms")
        
        await provider.cleanup()
        
        return {
            "throughput": sustained_throughput,
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency
        }
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_latency_under_memory_pressure(self, high_performance_context_manager):
        """Test latency performance under memory pressure."""
        cm = high_performance_context_manager
        
        # Create memory pressure
        memory_pressure_data = []
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        target_pressure_mb = 100  # 100MB additional pressure
        
        try:
            # Generate memory pressure
            while len(memory_pressure_data) < target_pressure_mb:
                # Create 1MB chunks of data
                chunk = [0] * (1024 * 256)  # ~1MB of integers
                memory_pressure_data.append(chunk)
            
            # Setup provider
            await cm.update_provider_health(
                "memory_test",
                {
                    "provider_type": "test",
                    "is_available": True,
                    "response_time_ms": 30.0,
                    "error_rate": 0.0,
                    "success_rate": 1.0,
                    "uptime_percentage": 100.0,
                    "consecutive_failures": 0,
                    "circuit_breaker_state": "CLOSED"
                }
            )
            
            # Benchmark under memory pressure
            async with PerformanceBenchmark("memory_pressure_latency") as benchmark:
                num_operations = 1000
                
                for i in range(num_operations):
                    start = time.perf_counter()
                    
                    # Routing operations under memory pressure
                    health = await cm.get_provider_health("memory_test")
                    
                    await cm.store_routing_decision(
                        f"memory-pressure-{i}",
                        {
                            "selected_provider": "memory_test",
                            "alternative_providers": [],
                            "routing_strategy": "memory_pressure_test",
                            "decision_factors": {"memory_pressure": True, "iteration": i},
                            "estimated_cost": 0.1,
                            "estimated_latency_ms": 30.0,
                            "confidence_score": 0.9,
                            "fallback_chain": []
                        }
                    )
                    
                    duration_ms = (time.perf_counter() - start) * 1000
                    benchmark.record_operation(duration_ms)
                    
                    if i % 100 == 0:
                        benchmark.sample_resources()
                        
                        # Trigger occasional garbage collection
                        if i % 500 == 0:
                            gc.collect()
            
            metrics = benchmark.get_metrics()
            
            # Memory pressure should not significantly degrade performance
            assert metrics.avg_latency_ms < 100.0, f"Average latency {metrics.avg_latency_ms:.2f}ms too high under memory pressure"
            assert metrics.p95_latency_ms < 200.0, f"P95 latency {metrics.p95_latency_ms:.2f}ms too high under memory pressure"
            assert metrics.operations_per_second > 50, f"Throughput {metrics.operations_per_second:.0f} ops/sec too low under memory pressure"
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            print(f"Memory Pressure Performance: "
                  f"memory_increase={memory_increase:.1f}MB, "
                  f"avg_latency={metrics.avg_latency_ms:.2f}ms, "
                  f"p95_latency={metrics.p95_latency_ms:.2f}ms, "
                  f"throughput={metrics.operations_per_second:.0f} ops/sec")
            
            return metrics
            
        finally:
            # Clean up memory pressure
            del memory_pressure_data
            gc.collect()


class TestFailoverPerformance:
    """Test provider failover speed and recovery time."""
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_circuit_breaker_failover_speed(self, high_performance_context_manager, performance_providers):
        """Test circuit breaker failover speed."""
        cm = high_performance_context_manager
        
        primary = performance_providers["fast"]
        backup = performance_providers["medium"]
        
        await primary.initialize()
        await backup.initialize()
        
        # Setup initial states
        for provider in [primary, backup]:
            await cm.update_provider_health(
                provider.provider_type.value,
                {
                    "provider_type": provider.provider_type.value,
                    "is_available": True,
                    "response_time_ms": provider.base_latency_ms,
                    "error_rate": 0.0,
                    "success_rate": 1.0,
                    "uptime_percentage": 100.0,
                    "consecutive_failures": 0,
                    "circuit_breaker_state": "CLOSED"
                }
            )
        
        # Initialize circuit breakers
        for provider in [primary, backup]:
            await cm.update_circuit_breaker_state(
                provider.provider_type.value,
                {
                    "state": "CLOSED",
                    "failure_count": 0,
                    "success_count": 0,
                    "last_failure_time": None,
                    "next_attempt_time": None,
                    "failure_threshold": 3,
                    "success_threshold": 2,
                    "timeout_duration_s": 30,
                    "half_open_max_calls": 2,
                    "current_half_open_calls": 0
                }
            )
        
        # Benchmark failover scenario
        failover_times = []
        recovery_times = []
        
        for iteration in range(10):  # Multiple failover cycles
            # Phase 1: Trigger circuit breaker opening
            primary.error_rate = 1.0  # Force failures
            
            failure_start = time.perf_counter()
            failures_to_open = 0
            
            # Generate failures to open circuit breaker
            while failures_to_open < 5:  # Ensure circuit opens
                request = AIRequest(
                    model=f"{primary.provider_type.value}-perf",
                    messages=[{"role": "user", "content": f"Failover test {iteration}-{failures_to_open}"}],
                    max_tokens=100
                )
                
                try:
                    await primary._generate_response_impl(request)
                except Exception:
                    failures_to_open += 1
                    
                    # Update circuit breaker
                    cb_state = await cm.get_circuit_breaker_state(primary.provider_type.value)
                    if cb_state:
                        new_failure_count = cb_state.failure_count + 1
                        new_state = "OPEN" if new_failure_count >= cb_state.failure_threshold else "CLOSED"
                        
                        await cm.update_circuit_breaker_state(
                            primary.provider_type.value,
                            {
                                **cb_state.__dict__,
                                "state": new_state,
                                "failure_count": new_failure_count,
                                "last_failure_time": datetime.now(timezone.utc),
                                "next_attempt_time": datetime.now(timezone.utc) + timedelta(seconds=30) if new_state == "OPEN" else None
                            }
                        )
                        
                        if new_state == "OPEN":
                            break
            
            # Phase 2: Measure failover to backup
            failover_start = time.perf_counter()
            
            # Check circuit breaker and route to backup
            cb_state = await cm.get_circuit_breaker_state(primary.provider_type.value)
            if cb_state and cb_state.state == "OPEN":
                # Use backup provider
                request = AIRequest(
                    model=f"{backup.provider_type.value}-perf",
                    messages=[{"role": "user", "content": f"Failover backup {iteration}"}],
                    max_tokens=100
                )
                
                response = await backup._generate_response_impl(request)
                
                await cm.store_routing_decision(
                    request.request_id,
                    {
                        "selected_provider": backup.provider_type.value,
                        "alternative_providers": [primary.provider_type.value],
                        "routing_strategy": "circuit_breaker_failover",
                        "decision_factors": {"primary_circuit_open": True, "iteration": iteration},
                        "estimated_cost": response.cost,
                        "estimated_latency_ms": response.latency_ms,
                        "confidence_score": 0.8,
                        "fallback_chain": []
                    }
                )
            
            failover_time = (time.perf_counter() - failover_start) * 1000
            failover_times.append(failover_time)
            
            # Phase 3: Simulate recovery
            primary.error_rate = 0.0  # Fix primary
            
            recovery_start = time.perf_counter()
            
            # Wait for circuit breaker timeout (simulate)
            await asyncio.sleep(0.1)  # Simulate brief timeout
            
            # Transition to HALF_OPEN
            await cm.update_circuit_breaker_state(
                primary.provider_type.value,
                {
                    **cb_state.__dict__,
                    "state": "HALF_OPEN",
                    "current_half_open_calls": 0
                }
            )
            
            # Test recovery
            recovery_request = AIRequest(
                model=f"{primary.provider_type.value}-perf",
                messages=[{"role": "user", "content": f"Recovery test {iteration}"}],
                max_tokens=100
            )
            
            recovery_response = await primary._generate_response_impl(recovery_request)
            
            # Close circuit breaker
            await cm.update_circuit_breaker_state(
                primary.provider_type.value,
                {
                    **cb_state.__dict__,
                    "state": "CLOSED",
                    "failure_count": 0,
                    "success_count": 1,
                    "current_half_open_calls": 1
                }
            )
            
            recovery_time = (time.perf_counter() - recovery_start) * 1000
            recovery_times.append(recovery_time)
        
        # Analyze failover performance
        avg_failover_time = statistics.mean(failover_times)
        p95_failover_time = statistics.quantiles(failover_times, n=20)[18] if len(failover_times) > 1 else failover_times[0]
        
        avg_recovery_time = statistics.mean(recovery_times)
        p95_recovery_time = statistics.quantiles(recovery_times, n=20)[18] if len(recovery_times) > 1 else recovery_times[0]
        
        # Performance assertions
        assert avg_failover_time < 50.0, f"Average failover time {avg_failover_time:.2f}ms too slow"
        assert p95_failover_time < 100.0, f"P95 failover time {p95_failover_time:.2f}ms too slow"
        assert avg_recovery_time < 100.0, f"Average recovery time {avg_recovery_time:.2f}ms too slow"
        assert p95_recovery_time < 200.0, f"P95 recovery time {p95_recovery_time:.2f}ms too slow"
        
        print(f"Failover Performance: "
              f"avg_failover={avg_failover_time:.2f}ms, p95_failover={p95_failover_time:.2f}ms, "
              f"avg_recovery={avg_recovery_time:.2f}ms, p95_recovery={p95_recovery_time:.2f}ms")
        
        await primary.cleanup()
        await backup.cleanup()
        
        return {
            "avg_failover_ms": avg_failover_time,
            "p95_failover_ms": p95_failover_time,
            "avg_recovery_ms": avg_recovery_time,
            "p95_recovery_ms": p95_recovery_time
        }
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_end_to_end_performance_benchmark(self, high_performance_context_manager, performance_providers):
        """Comprehensive end-to-end performance benchmark."""
        cm = high_performance_context_manager
        
        # Use the existing benchmark function
        benchmark_results = await benchmark_state_operations(cm, num_operations=2000)
        
        # Verify comprehensive performance requirements
        overall = benchmark_results["overall"]
        
        assert overall["p95_latency_ms"] <= 100.0, f"End-to-end P95 latency {overall['p95_latency_ms']:.2f}ms exceeds requirement"
        assert overall["avg_latency_ms"] <= 50.0, f"End-to-end average latency {overall['avg_latency_ms']:.2f}ms too high"
        assert overall["target_achievement"]["sub_100ms_target"], "Failed to meet sub-100ms target"
        
        # Additional performance checks
        for operation_type, metrics in benchmark_results["operations"].items():
            assert metrics["avg_latency_ms"] < 100.0, f"{operation_type} average latency too high"
            assert metrics["p95_latency_ms"] < 150.0, f"{operation_type} P95 latency too high"
        
        print(f"End-to-End Performance Benchmark:")
        print(f"  Overall: avg={overall['avg_latency_ms']:.2f}ms, p95={overall['p95_latency_ms']:.2f}ms, p99={overall['p99_latency_ms']:.2f}ms")
        for op_type, metrics in benchmark_results["operations"].items():
            print(f"  {op_type}: avg={metrics['avg_latency_ms']:.2f}ms, p95={metrics['p95_latency_ms']:.2f}ms")
        
        return benchmark_results


# Test markers for performance tests
pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.performance
]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "benchmark"])