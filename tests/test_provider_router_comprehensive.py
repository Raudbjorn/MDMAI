"""Comprehensive Testing Suite for Provider Router with Fallback System.

This module provides an advanced testing framework for the MDMAI Provider Router,
covering unit tests, integration tests, performance tests, failure scenarios,
and chaos engineering patterns.

Test Categories:
- Unit Tests: Core component validation
- Integration Tests: End-to-end workflows
- Performance Tests: Latency and throughput benchmarks
- Failure Tests: Resilience and recovery scenarios
- Chaos Tests: System behavior under unpredictable conditions
"""

import asyncio
import json
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import hypothesis.strategies as st
import pytest
import pytest_asyncio
from hypothesis import given, settings, assume, note
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, invariant
from locust import HttpUser, task, between, events
from pytest_benchmark.fixture import BenchmarkFixture

# Import system components (adjust imports based on actual structure)
from src.ai_providers.mcp_provider_router import (
    MCPProviderRouter,
    MCPRequestType,
    MCPMessageType,
    MCPErrorCode,
    RouteRequestParams,
    RouteRequestResponse,
    ProviderHealthUpdate,
    FallbackChainConfig,
    RoutingConfiguration,
    CircuitBreakerState
)
from src.ai_providers.models import (
    AIRequest,
    AIResponse,
    ProviderType,
    ProviderStatus,
    ProviderSelection,
    ProviderHealth,
    CostBudget
)
from src.ai_providers.provider_manager import AIProviderManager
from src.ai_providers.error_handler import (
    AIProviderError,
    NoProviderAvailableError,
    CircuitBreakerOpenError
)


# ============================================================================
# TEST CONFIGURATION AND FIXTURES
# ============================================================================

@dataclass
class TestConfig:
    """Test configuration parameters."""
    max_latency_ms: float = 100.0
    target_rps: int = 1000
    cache_hit_ratio: float = 0.8
    failure_recovery_time: float = 5.0
    chaos_intensity: float = 0.3
    network_partition_probability: float = 0.1
    provider_failure_rate: float = 0.05
    memory_limit_mb: int = 512
    concurrent_requests: int = 100


@pytest.fixture
def test_config() -> TestConfig:
    """Provide test configuration."""
    return TestConfig()


@pytest.fixture
def mock_providers() -> Dict[str, Mock]:
    """Create mock provider instances."""
    providers = {}
    for provider_type in ["openai", "anthropic", "google", "azure", "aws"]:
        mock = AsyncMock()
        mock.name = provider_type
        mock.is_available = True
        mock.health_score = 1.0
        mock.avg_latency_ms = 50.0
        mock.error_rate = 0.01
        mock.cost_per_token = 0.0001
        mock.max_tokens = 100000
        mock.rate_limit = 1000
        mock.current_load = 0
        providers[provider_type] = mock
    return providers


@pytest_asyncio.fixture
async def router(mock_providers) -> MCPProviderRouter:
    """Create configured router instance."""
    router = MCPProviderRouter(
        providers=mock_providers,
        enable_fallback=True,
        enable_caching=True,
        enable_monitoring=True,
        circuit_breaker_threshold=5,
        circuit_breaker_timeout=30.0
    )
    await router.initialize()
    yield router
    await router.shutdown()


@pytest.fixture
def test_request() -> AIRequest:
    """Create test AI request."""
    return AIRequest(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the meaning of life?"}
        ],
        temperature=0.7,
        max_tokens=1000,
        stream=False
    )


@pytest.fixture
def chaos_injector():
    """Chaos injection utilities."""
    class ChaosInjector:
        def __init__(self):
            self.active = False
            self.failure_types = []
            
        def inject_latency(self, min_ms: float = 100, max_ms: float = 1000):
            """Inject random latency."""
            if self.active:
                delay = random.uniform(min_ms, max_ms) / 1000
                asyncio.sleep(delay)
                
        def inject_failure(self, probability: float = 0.3):
            """Inject random failure."""
            if self.active and random.random() < probability:
                raise Exception("Chaos: Injected failure")
                
        def inject_network_partition(self):
            """Simulate network partition."""
            if self.active:
                raise ConnectionError("Chaos: Network partition")
                
        def corrupt_data(self, data: Any, probability: float = 0.1):
            """Corrupt data with given probability."""
            if self.active and random.random() < probability:
                if isinstance(data, dict):
                    key = random.choice(list(data.keys()))
                    data[key] = None
                elif isinstance(data, list) and data:
                    idx = random.randint(0, len(data) - 1)
                    data[idx] = None
            return data
            
    return ChaosInjector()


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestProviderSelection:
    """Unit tests for provider selection algorithms."""
    
    @pytest.mark.asyncio
    async def test_cost_based_selection(self, router, test_request):
        """Test cost-optimized provider selection."""
        router.set_routing_strategy("cost")
        
        # Mock provider costs
        router.providers["openai"].cost_per_token = 0.0002
        router.providers["anthropic"].cost_per_token = 0.0001
        router.providers["google"].cost_per_token = 0.0003
        
        selection = await router.select_provider(test_request)
        
        assert selection.provider == "anthropic"
        assert selection.strategy == "cost"
        assert selection.cost_estimate < 0.2
        
    @pytest.mark.asyncio
    async def test_latency_based_selection(self, router, test_request):
        """Test speed-optimized provider selection."""
        router.set_routing_strategy("speed")
        
        # Mock provider latencies
        router.providers["openai"].avg_latency_ms = 100
        router.providers["anthropic"].avg_latency_ms = 50
        router.providers["google"].avg_latency_ms = 75
        
        selection = await router.select_provider(test_request)
        
        assert selection.provider == "anthropic"
        assert selection.strategy == "speed"
        assert selection.expected_latency_ms == 50
        
    @pytest.mark.asyncio
    async def test_capability_based_selection(self, router, test_request):
        """Test capability-based provider selection."""
        router.set_routing_strategy("capability")
        
        # Set specific model requirement
        test_request.model = "claude-3-opus"
        
        # Mock provider capabilities
        router.providers["openai"].supports_model = lambda m: False
        router.providers["anthropic"].supports_model = lambda m: m.startswith("claude")
        router.providers["google"].supports_model = lambda m: m.startswith("gemini")
        
        selection = await router.select_provider(test_request)
        
        assert selection.provider == "anthropic"
        assert selection.strategy == "capability"
        
    @pytest.mark.asyncio
    async def test_load_balanced_selection(self, router, test_request):
        """Test load-balanced provider selection."""
        router.set_routing_strategy("load_balanced")
        
        # Mock provider loads
        router.providers["openai"].current_load = 80
        router.providers["anthropic"].current_load = 30
        router.providers["google"].current_load = 60
        
        selections = []
        for _ in range(100):
            selection = await router.select_provider(test_request)
            selections.append(selection.provider)
            
        # Verify load distribution
        anthropic_count = selections.count("anthropic")
        assert anthropic_count > 40  # Should get more traffic due to lower load
        
    @pytest.mark.asyncio
    async def test_priority_based_selection(self, router, test_request):
        """Test priority-based provider selection."""
        router.set_routing_strategy("priority")
        router.set_provider_priorities({
            "google": 1,
            "openai": 2,
            "anthropic": 3
        })
        
        selection = await router.select_provider(test_request)
        
        assert selection.provider == "google"
        assert selection.strategy == "priority"


class TestCircuitBreaker:
    """Unit tests for circuit breaker state transitions."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self, router):
        """Test circuit breaker opens after threshold failures."""
        provider = "openai"
        
        # Simulate failures
        for i in range(5):
            await router.record_failure(provider, Exception(f"Error {i}"))
            
        state = router.get_circuit_breaker_state(provider)
        assert state == CircuitBreakerState.OPEN
        
        # Verify provider is excluded
        available = await router.get_available_providers()
        assert provider not in available
        
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_transition(self, router):
        """Test circuit breaker transitions to half-open after timeout."""
        provider = "openai"
        
        # Open circuit
        for i in range(5):
            await router.record_failure(provider, Exception(f"Error {i}"))
            
        # Wait for timeout (mocked)
        original_time = time.time()
        with patch("time.time", side_effect=lambda: original_time + 31):
            state = router.get_circuit_breaker_state(provider)
            assert state == CircuitBreakerState.HALF_OPEN
            
    @pytest.mark.asyncio
    async def test_circuit_breaker_closes_on_success(self, router):
        """Test circuit breaker closes after successful requests in half-open state."""
        provider = "openai"
        
        # Set to half-open
        router.circuit_breakers[provider].state = CircuitBreakerState.HALF_OPEN
        
        # Record successes
        for _ in range(3):
            await router.record_success(provider, latency_ms=50)
            
        state = router.get_circuit_breaker_state(provider)
        assert state == CircuitBreakerState.CLOSED
        
    @pytest.mark.asyncio
    async def test_circuit_breaker_reopens_on_half_open_failure(self, router):
        """Test circuit breaker reopens if failure occurs in half-open state."""
        provider = "openai"
        
        # Set to half-open
        router.circuit_breakers[provider].state = CircuitBreakerState.HALF_OPEN
        
        # Record failure
        await router.record_failure(provider, Exception("Half-open failure"))
        
        state = router.get_circuit_breaker_state(provider)
        assert state == CircuitBreakerState.OPEN


class TestFallbackChain:
    """Unit tests for fallback chain execution."""
    
    @pytest.mark.asyncio
    async def test_fallback_chain_execution(self, router, test_request):
        """Test fallback chain executes in order."""
        chain = ["openai", "anthropic", "google"]
        router.set_fallback_chain(chain)
        
        # Make first provider fail
        router.providers["openai"].process = AsyncMock(
            side_effect=Exception("Provider unavailable")
        )
        
        response = await router.route_with_fallback(test_request)
        
        assert response.provider_used == "anthropic"
        assert response.fallback_attempts == 1
        
    @pytest.mark.asyncio
    async def test_fallback_chain_exhaustion(self, router, test_request):
        """Test behavior when all providers in chain fail."""
        chain = ["openai", "anthropic", "google"]
        router.set_fallback_chain(chain)
        
        # Make all providers fail
        for provider in chain:
            router.providers[provider].process = AsyncMock(
                side_effect=Exception(f"{provider} unavailable")
            )
            
        with pytest.raises(NoProviderAvailableError) as exc:
            await router.route_with_fallback(test_request)
            
        assert "All providers failed" in str(exc.value)
        assert exc.value.attempted_providers == chain
        
    @pytest.mark.asyncio
    async def test_fallback_with_retry(self, router, test_request):
        """Test fallback with retry logic."""
        router.set_fallback_chain(["openai", "anthropic"])
        router.set_retry_config(max_retries=2, backoff_factor=1.0)
        
        # Make openai fail twice then succeed
        call_count = 0
        async def mock_process(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return AIResponse(content="Success", provider="openai")
            
        router.providers["openai"].process = mock_process
        
        response = await router.route_with_fallback(test_request)
        
        assert response.provider_used == "openai"
        assert response.retry_count == 2
        
    @pytest.mark.asyncio
    async def test_conditional_fallback(self, router, test_request):
        """Test conditional fallback based on error type."""
        router.set_fallback_chain(["openai", "anthropic"])
        
        # Configure conditional fallback
        router.set_fallback_conditions({
            "RateLimitError": True,
            "AuthenticationError": False
        })
        
        # Simulate rate limit error (should fallback)
        router.providers["openai"].process = AsyncMock(
            side_effect=RateLimitError("Rate limit exceeded")
        )
        
        response = await router.route_with_fallback(test_request)
        assert response.provider_used == "anthropic"
        
        # Simulate auth error (should not fallback)
        router.providers["openai"].process = AsyncMock(
            side_effect=AuthenticationError("Invalid API key")
        )
        
        with pytest.raises(AuthenticationError):
            await router.route_with_fallback(test_request)


class TestCostOptimization:
    """Unit tests for cost calculation and optimization."""
    
    @pytest.mark.asyncio
    async def test_cost_calculation(self, router, test_request):
        """Test accurate cost calculation."""
        router.providers["openai"].cost_per_token = 0.00002
        test_request.max_tokens = 1000
        
        estimated_cost = await router.calculate_cost(
            provider="openai",
            request=test_request
        )
        
        # Cost = (prompt_tokens + completion_tokens) * cost_per_token
        # Assuming ~100 prompt tokens
        expected_cost = (100 + 1000) * 0.00002
        assert abs(estimated_cost - expected_cost) < 0.001
        
    @pytest.mark.asyncio
    async def test_cost_budget_enforcement(self, router, test_request):
        """Test cost budget enforcement."""
        budget = CostBudget(
            daily_limit=10.0,
            monthly_limit=300.0,
            alert_threshold=0.8
        )
        router.set_cost_budget(budget)
        
        # Simulate spending near limit
        router.current_spending = {
            "daily": 9.5,
            "monthly": 250.0
        }
        
        # Request that would exceed budget
        test_request.max_tokens = 100000
        router.providers["openai"].cost_per_token = 0.0001
        
        with pytest.raises(BudgetExceededError):
            await router.route_with_budget_check(test_request)
            
    @pytest.mark.asyncio
    async def test_cost_optimization_routing(self, router, test_request):
        """Test cost-optimized routing decisions."""
        # Set different costs for providers
        router.providers["openai"].cost_per_token = 0.00003
        router.providers["anthropic"].cost_per_token = 0.00002
        router.providers["google"].cost_per_token = 0.00001
        
        # Set quality scores
        router.providers["openai"].quality_score = 0.95
        router.providers["anthropic"].quality_score = 0.90
        router.providers["google"].quality_score = 0.85
        
        # Test cost-quality tradeoff
        router.set_optimization_mode("balanced")
        selection = await router.select_provider(test_request)
        
        # Should select anthropic (good balance)
        assert selection.provider == "anthropic"
        assert selection.cost_quality_score > 0.8


class TestHealthMonitoring:
    """Unit tests for health metric collection."""
    
    @pytest.mark.asyncio
    async def test_health_metrics_collection(self, router):
        """Test health metrics are collected correctly."""
        provider = "openai"
        
        # Record various metrics
        await router.record_success(provider, latency_ms=50)
        await router.record_success(provider, latency_ms=60)
        await router.record_failure(provider, Exception("Error"))
        await router.record_success(provider, latency_ms=55)
        
        health = await router.get_provider_health(provider)
        
        assert health.success_rate == 0.75  # 3/4 successes
        assert health.avg_latency_ms == 55  # (50+60+55)/3
        assert health.total_requests == 4
        assert health.error_count == 1
        
    @pytest.mark.asyncio
    async def test_health_score_calculation(self, router):
        """Test composite health score calculation."""
        provider = "openai"
        
        # Set health metrics
        router.providers[provider].metrics = {
            "success_rate": 0.95,
            "avg_latency_ms": 100,
            "p99_latency_ms": 500,
            "error_rate": 0.05,
            "availability": 0.99
        }
        
        score = await router.calculate_health_score(provider)
        
        assert 0 <= score <= 1
        assert score > 0.8  # Should be healthy
        
    @pytest.mark.asyncio
    async def test_health_degradation_detection(self, router):
        """Test detection of health degradation."""
        provider = "openai"
        
        # Record increasing failures
        for i in range(10):
            if i < 5:
                await router.record_success(provider, latency_ms=50)
            else:
                await router.record_failure(provider, Exception(f"Error {i}"))
                
        health = await router.get_provider_health(provider)
        assert health.is_degraded
        assert health.degradation_reason == "High error rate"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestMCPProtocolIntegration:
    """Integration tests for MCP protocol communication."""
    
    @pytest.mark.asyncio
    async def test_mcp_request_handling(self, router):
        """Test handling of MCP protocol requests."""
        request = {
            "jsonrpc": "2.0",
            "id": "test-123",
            "method": "route_request",
            "params": {
                "request": {
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 100
                },
                "strategy": "cost",
                "fallback_enabled": True
            }
        }
        
        response = await router.handle_mcp_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-123"
        assert "result" in response
        assert response["result"]["status"] == "success"
        
    @pytest.mark.asyncio
    async def test_mcp_error_response(self, router):
        """Test MCP error response format."""
        request = {
            "jsonrpc": "2.0",
            "id": "test-456",
            "method": "invalid_method",
            "params": {}
        }
        
        response = await router.handle_mcp_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-456"
        assert "error" in response
        assert response["error"]["code"] == -32601  # Method not found
        
    @pytest.mark.asyncio
    async def test_mcp_notification_dispatch(self, router):
        """Test MCP notification dispatch."""
        notifications = []
        
        async def notification_handler(notification):
            notifications.append(notification)
            
        router.on_notification = notification_handler
        
        # Trigger provider health change
        await router.update_provider_health("openai", ProviderHealth(
            is_healthy=False,
            health_score=0.3,
            last_check=datetime.utcnow()
        ))
        
        # Allow time for async notification
        await asyncio.sleep(0.1)
        
        assert len(notifications) == 1
        assert notifications[0]["method"] == "provider_health_change"
        assert notifications[0]["params"]["provider"] == "openai"
        assert not notifications[0]["params"]["is_healthy"]


class TestProviderFailover:
    """Integration tests for provider failover scenarios."""
    
    @pytest.mark.asyncio
    async def test_automatic_failover(self, router, test_request):
        """Test automatic failover on provider failure."""
        # Configure failover chain
        router.set_fallback_chain(["openai", "anthropic", "google"])
        
        # Simulate openai failure
        router.providers["openai"].process = AsyncMock(
            side_effect=ProviderUnavailableError("Service down")
        )
        
        response = await router.route_request(test_request)
        
        assert response.provider_used == "anthropic"
        assert response.failover_occurred
        assert len(response.failover_trace) == 1
        
    @pytest.mark.asyncio
    async def test_cascading_failover(self, router, test_request):
        """Test cascading failover through multiple providers."""
        router.set_fallback_chain(["openai", "anthropic", "google", "azure"])
        
        # Make first two providers fail
        router.providers["openai"].process = AsyncMock(
            side_effect=Exception("Openai down")
        )
        router.providers["anthropic"].process = AsyncMock(
            side_effect=Exception("Anthropic down")
        )
        
        response = await router.route_request(test_request)
        
        assert response.provider_used == "google"
        assert response.failover_count == 2
        assert response.failover_trace == ["openai", "anthropic"]
        
    @pytest.mark.asyncio
    async def test_failover_with_circuit_breaker(self, router, test_request):
        """Test failover interaction with circuit breakers."""
        router.set_fallback_chain(["openai", "anthropic", "google"])
        
        # Open circuit breaker for openai
        for _ in range(5):
            await router.record_failure("openai", Exception("Error"))
            
        # Should skip openai and go directly to anthropic
        response = await router.route_request(test_request)
        
        assert response.provider_used == "anthropic"
        assert response.skipped_providers == ["openai"]
        assert response.skip_reason == "Circuit breaker open"


class TestStateSynchronization:
    """Integration tests for state synchronization across tiers."""
    
    @pytest.mark.asyncio
    async def test_cache_tier_synchronization(self, router):
        """Test cache synchronization across multiple tiers."""
        # Configure multi-tier cache
        router.configure_cache({
            "l1": {"type": "memory", "size": 100, "ttl": 60},
            "l2": {"type": "redis", "size": 1000, "ttl": 600},
            "l3": {"type": "disk", "size": 10000, "ttl": 3600}
        })
        
        # Add item to cache
        cache_key = "test_key"
        cache_value = {"result": "test_value"}
        await router.cache_set(cache_key, cache_value)
        
        # Verify synchronization across tiers
        l1_value = await router.cache_get(cache_key, tier="l1")
        l2_value = await router.cache_get(cache_key, tier="l2")
        l3_value = await router.cache_get(cache_key, tier="l3")
        
        assert l1_value == l2_value == l3_value == cache_value
        
    @pytest.mark.asyncio
    async def test_state_consistency_under_load(self, router):
        """Test state consistency under concurrent updates."""
        state_updates = []
        
        async def update_state(provider: str, metric: str, value: float):
            await router.update_provider_metric(provider, metric, value)
            state_updates.append((provider, metric, value))
            
        # Concurrent updates
        tasks = []
        for i in range(100):
            provider = random.choice(["openai", "anthropic", "google"])
            metric = random.choice(["latency", "success_rate", "cost"])
            value = random.random()
            tasks.append(update_state(provider, metric, value))
            
        await asyncio.gather(*tasks)
        
        # Verify state consistency
        for provider, metric, expected_value in state_updates[-10:]:
            actual_value = await router.get_provider_metric(provider, metric)
            assert actual_value == expected_value


class TestCacheConsistency:
    """Integration tests for cache consistency validation."""
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_propagation(self, router):
        """Test cache invalidation propagates across all tiers."""
        cache_key = "test_key"
        cache_value = {"data": "initial"}
        
        # Set value in cache
        await router.cache_set(cache_key, cache_value)
        
        # Update value
        new_value = {"data": "updated"}
        await router.cache_set(cache_key, new_value)
        
        # Verify all tiers have updated value
        for tier in ["l1", "l2", "l3"]:
            value = await router.cache_get(cache_key, tier=tier)
            assert value == new_value
            
    @pytest.mark.asyncio
    async def test_cache_ttl_consistency(self, router):
        """Test TTL consistency across cache tiers."""
        cache_key = "ttl_test"
        cache_value = {"data": "test"}
        ttl = 60  # 60 seconds
        
        await router.cache_set(cache_key, cache_value, ttl=ttl)
        
        # Check TTL on all tiers
        for tier in ["l1", "l2", "l3"]:
            remaining_ttl = await router.cache_ttl(cache_key, tier=tier)
            assert 55 <= remaining_ttl <= 60  # Allow small variance


class TestEndToEndWorkflows:
    """Integration tests for end-to-end routing workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_request_workflow(self, router, test_request):
        """Test complete request routing workflow."""
        # Set up monitoring
        events = []
        
        async def event_handler(event):
            events.append(event)
            
        router.on_event = event_handler
        
        # Execute request
        response = await router.route_request(test_request)
        
        # Verify workflow events
        event_types = [e["type"] for e in events]
        assert "request_received" in event_types
        assert "provider_selected" in event_types
        assert "request_processed" in event_types
        assert "response_sent" in event_types
        
        # Verify response
        assert response.status == "success"
        assert response.provider_used in ["openai", "anthropic", "google"]
        assert response.latency_ms < 1000
        
    @pytest.mark.asyncio
    async def test_streaming_request_workflow(self, router):
        """Test streaming request workflow."""
        stream_request = AIRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Stream test"}],
            stream=True,
            max_tokens=100
        )
        
        chunks = []
        async for chunk in router.route_stream(stream_request):
            chunks.append(chunk)
            
        assert len(chunks) > 0
        assert all(chunk.delta for chunk in chunks)
        assert chunks[-1].finish_reason == "stop"


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestRoutingPerformance:
    """Performance tests for routing latency."""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_routing_latency(self, router, test_request, benchmark):
        """Test routing decision latency."""
        async def route():
            return await router.select_provider(test_request)
            
        result = await benchmark(route)
        assert result.provider is not None
        
        # Verify sub-100ms latency
        assert benchmark.stats["mean"] < 0.1  # 100ms
        
    @pytest.mark.asyncio
    async def test_routing_throughput(self, router, test_request):
        """Test routing throughput (requests per second)."""
        start_time = time.time()
        request_count = 0
        
        # Run for 10 seconds
        while time.time() - start_time < 10:
            await router.select_provider(test_request)
            request_count += 1
            
        rps = request_count / 10
        assert rps > 1000  # Target: 1000+ RPS
        
    @pytest.mark.asyncio
    async def test_concurrent_routing(self, router, test_request):
        """Test concurrent request handling."""
        async def make_request():
            return await router.route_request(test_request)
            
        # Execute 100 concurrent requests
        start_time = time.time()
        tasks = [make_request() for _ in range(100)]
        responses = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # All requests should succeed
        assert all(r.status == "success" for r in responses)
        
        # Should complete within reasonable time
        assert duration < 5  # 5 seconds for 100 requests


class TestMemoryPerformance:
    """Performance tests for memory usage."""
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, router, test_request):
        """Test memory usage under sustained load."""
        import tracemalloc
        import gc
        
        # Start memory tracking
        tracemalloc.start()
        gc.collect()
        snapshot_start = tracemalloc.take_snapshot()
        
        # Generate load
        for _ in range(1000):
            await router.route_request(test_request)
            
        # Measure memory
        gc.collect()
        snapshot_end = tracemalloc.take_snapshot()
        
        # Calculate difference
        stats = snapshot_end.compare_to(snapshot_start, 'lineno')
        total_memory = sum(stat.size_diff for stat in stats)
        
        # Memory usage should be reasonable
        assert total_memory < 100 * 1024 * 1024  # Less than 100MB
        
        tracemalloc.stop()
        
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, router, test_request):
        """Test for memory leaks during extended operation."""
        import gc
        import sys
        
        initial_objects = len(gc.get_objects())
        
        # Run many iterations
        for _ in range(100):
            await router.route_request(test_request)
            gc.collect()
            
        final_objects = len(gc.get_objects())
        
        # Object count should not grow significantly
        object_growth = final_objects - initial_objects
        assert object_growth < 1000  # Allow some growth but not unbounded


class TestCachePerformance:
    """Performance tests for caching system."""
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate(self, router):
        """Test cache hit rate optimization."""
        # Warm up cache
        for i in range(100):
            key = f"key_{i % 20}"  # 20 unique keys
            await router.cache_set(key, {"value": i})
            
        # Measure hit rate
        hits = 0
        total = 1000
        
        for i in range(total):
            key = f"key_{i % 20}"
            result = await router.cache_get(key)
            if result is not None:
                hits += 1
                
        hit_rate = hits / total
        assert hit_rate > 0.8  # Target: 80% hit rate
        
    @pytest.mark.asyncio
    async def test_cache_latency(self, router, benchmark):
        """Test cache operation latency."""
        key = "test_key"
        value = {"data": "test" * 100}
        
        # Test write latency
        async def cache_write():
            await router.cache_set(key, value)
            
        write_result = await benchmark(cache_write)
        assert benchmark.stats["mean"] < 0.001  # Sub-1ms writes
        
        # Test read latency
        async def cache_read():
            return await router.cache_get(key)
            
        read_result = await benchmark(cache_read)
        assert benchmark.stats["mean"] < 0.0005  # Sub-0.5ms reads


# ============================================================================
# FAILURE SCENARIO TESTS
# ============================================================================

class TestProviderOutages:
    """Tests for provider outage scenarios."""
    
    @pytest.mark.asyncio
    async def test_single_provider_outage(self, router, test_request):
        """Test handling of single provider outage."""
        # Simulate openai outage
        router.providers["openai"].is_available = False
        router.providers["openai"].process = AsyncMock(
            side_effect=ServiceUnavailableError("Provider down")
        )
        
        # Should automatically use alternative
        response = await router.route_request(test_request)
        
        assert response.provider_used != "openai"
        assert response.status == "success"
        
    @pytest.mark.asyncio
    async def test_multiple_provider_outages(self, router, test_request):
        """Test handling of multiple simultaneous outages."""
        # Simulate multiple outages
        outage_providers = ["openai", "anthropic"]
        for provider in outage_providers:
            router.providers[provider].is_available = False
            
        # Should use remaining providers
        response = await router.route_request(test_request)
        
        assert response.provider_used not in outage_providers
        assert response.provider_used in ["google", "azure", "aws"]
        
    @pytest.mark.asyncio
    async def test_rolling_provider_failures(self, router, test_request):
        """Test handling of rolling provider failures."""
        providers = ["openai", "anthropic", "google"]
        
        for i, provider in enumerate(providers):
            # Fail providers one by one
            router.providers[provider].is_available = False
            
            # Should still route successfully if any provider available
            if i < len(providers) - 1:
                response = await router.route_request(test_request)
                assert response.status == "success"
                assert response.provider_used not in providers[:i+1]


class TestRateLimiting:
    """Tests for rate limiting and backpressure."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, router, test_request):
        """Test handling of rate limit errors."""
        # Configure rate limits
        router.providers["openai"].rate_limit = 10  # 10 requests per second
        
        # Send burst of requests
        tasks = []
        for _ in range(20):
            tasks.append(router.route_request(test_request))
            
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Some should be rate limited and fallback
        rate_limited = sum(1 for r in responses 
                          if isinstance(r, dict) and r.get("fallback_reason") == "rate_limit")
        assert rate_limited > 0
        
    @pytest.mark.asyncio
    async def test_adaptive_rate_limiting(self, router):
        """Test adaptive rate limiting based on provider feedback."""
        provider = "openai"
        
        # Simulate rate limit headers
        await router.handle_rate_limit_headers(provider, {
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Remaining": "10",
            "X-RateLimit-Reset": str(int(time.time()) + 60)
        })
        
        # Should reduce request rate
        current_rate = router.get_provider_rate(provider)
        assert current_rate < 100  # Should throttle below limit
        
    @pytest.mark.asyncio
    async def test_backpressure_propagation(self, router, test_request):
        """Test backpressure propagation through system."""
        # Simulate high load
        router.system_load = 0.9  # 90% load
        
        # Should apply backpressure
        response = await router.route_request(test_request)
        
        assert response.backpressure_applied
        assert response.queue_time_ms > 0


class TestNetworkPartitions:
    """Tests for network partition handling."""
    
    @pytest.mark.asyncio
    async def test_network_partition_detection(self, router):
        """Test detection of network partitions."""
        provider = "openai"
        
        # Simulate network errors
        for _ in range(3):
            await router.record_failure(
                provider, 
                ConnectionError("Network unreachable")
            )
            
        # Should detect partition
        partition_detected = await router.is_network_partitioned(provider)
        assert partition_detected
        
    @pytest.mark.asyncio
    async def test_partition_recovery(self, router):
        """Test recovery from network partition."""
        provider = "openai"
        
        # Simulate partition
        router.mark_provider_partitioned(provider)
        
        # Wait and recover
        await asyncio.sleep(1)
        router.providers[provider].process = AsyncMock(
            return_value=AIResponse(content="Success")
        )
        
        # Test probe should succeed
        recovered = await router.probe_provider(provider)
        assert recovered
        assert not router.is_provider_partitioned(provider)


class TestStateCorruption:
    """Tests for state corruption recovery."""
    
    @pytest.mark.asyncio
    async def test_corrupted_cache_recovery(self, router):
        """Test recovery from corrupted cache entries."""
        # Corrupt cache entry
        await router.cache_set("corrupted_key", None)
        
        # Should handle gracefully
        result = await router.cache_get("corrupted_key")
        assert result is None
        
        # Should mark as corrupted and clean
        corrupted_keys = await router.get_corrupted_cache_keys()
        assert "corrupted_key" in corrupted_keys
        
    @pytest.mark.asyncio
    async def test_state_validation_and_repair(self, router):
        """Test state validation and automatic repair."""
        # Corrupt internal state
        router.providers["openai"].health_score = -1  # Invalid
        router.providers["anthropic"].cost_per_token = None  # Missing
        
        # Run validation
        issues = await router.validate_state()
        assert len(issues) == 2
        
        # Run repair
        await router.repair_state()
        
        # State should be valid
        assert 0 <= router.providers["openai"].health_score <= 1
        assert router.providers["anthropic"].cost_per_token > 0


class TestCascadingFailures:
    """Tests for cascading failure prevention."""
    
    @pytest.mark.asyncio
    async def test_cascading_failure_detection(self, router):
        """Test detection of cascading failures."""
        # Simulate failures spreading
        providers = ["openai", "anthropic", "google"]
        
        for i, provider in enumerate(providers):
            await router.record_failure(provider, Exception("Cascading"))
            
            # Check if cascade detected
            if i >= 1:  # After 2 providers fail
                cascade_detected = await router.detect_cascading_failure()
                assert cascade_detected
                
    @pytest.mark.asyncio
    async def test_cascade_mitigation(self, router):
        """Test cascading failure mitigation strategies."""
        # Trigger cascade detection
        router.cascade_detected = True
        
        # Should apply mitigation
        mitigation = await router.get_cascade_mitigation()
        
        assert mitigation["reduced_traffic"]
        assert mitigation["increased_timeouts"]
        assert mitigation["circuit_breakers_triggered"]
        
        # Traffic should be reduced
        assert router.traffic_reduction_factor < 1.0


# ============================================================================
# CHAOS ENGINEERING TESTS
# ============================================================================

class TestRandomFailures:
    """Chaos tests with random provider failures."""
    
    @pytest.mark.asyncio
    async def test_random_provider_failures(self, router, test_request, chaos_injector):
        """Test system behavior with random provider failures."""
        chaos_injector.active = True
        successful_requests = 0
        failed_requests = 0
        
        for _ in range(100):
            # Randomly fail providers
            for provider in router.providers.values():
                if random.random() < 0.2:  # 20% failure rate
                    provider.process = AsyncMock(
                        side_effect=Exception("Chaos failure")
                    )
                else:
                    provider.process = AsyncMock(
                        return_value=AIResponse(content="Success")
                    )
                    
            try:
                response = await router.route_request(test_request)
                if response.status == "success":
                    successful_requests += 1
            except Exception:
                failed_requests += 1
                
        # System should maintain reasonable success rate
        success_rate = successful_requests / (successful_requests + failed_requests)
        assert success_rate > 0.7  # At least 70% success despite chaos
        
    @pytest.mark.asyncio
    async def test_random_latency_spikes(self, router, test_request, chaos_injector):
        """Test system behavior with random latency spikes."""
        chaos_injector.active = True
        
        async def delayed_process(*args, **kwargs):
            await chaos_injector.inject_latency(min_ms=100, max_ms=2000)
            return AIResponse(content="Success")
            
        # Apply random delays
        for provider in router.providers.values():
            provider.process = delayed_process
            
        # System should handle gracefully
        response = await router.route_request(test_request)
        assert response.status == "success"
        
        # Should detect slow providers
        slow_providers = await router.get_slow_providers(threshold_ms=500)
        assert len(slow_providers) > 0


class TestNetworkChaos:
    """Chaos tests for network conditions."""
    
    @pytest.mark.asyncio
    async def test_network_latency_injection(self, router, test_request):
        """Test behavior under varying network latencies."""
        latencies = [10, 50, 100, 500, 1000, 2000]  # ms
        
        for latency in latencies:
            # Inject latency
            router.inject_network_latency(latency)
            
            start = time.time()
            response = await router.route_request(test_request)
            duration = (time.time() - start) * 1000
            
            # Should complete despite latency
            assert response.status == "success"
            
            # Should timeout if too slow
            if latency > 1000:
                assert response.timeout_occurred or response.fallback_used
                
    @pytest.mark.asyncio
    async def test_packet_loss_simulation(self, router, test_request):
        """Test behavior under packet loss conditions."""
        loss_rates = [0.01, 0.05, 0.1, 0.2]  # 1%, 5%, 10%, 20%
        
        for loss_rate in loss_rates:
            router.simulate_packet_loss(loss_rate)
            
            successes = 0
            attempts = 100
            
            for _ in range(attempts):
                try:
                    response = await router.route_request(test_request)
                    if response.status == "success":
                        successes += 1
                except Exception:
                    pass
                    
            success_rate = successes / attempts
            
            # Should degrade gracefully
            assert success_rate > (1 - loss_rate) * 0.8  # Allow 20% additional degradation


class TestResourceExhaustion:
    """Chaos tests for resource exhaustion scenarios."""
    
    @pytest.mark.asyncio
    async def test_memory_exhaustion(self, router, test_request):
        """Test behavior under memory pressure."""
        # Simulate memory pressure
        router.available_memory_mb = 50  # Low memory
        
        # Should adapt behavior
        response = await router.route_request(test_request)
        
        assert response.memory_optimized
        assert response.cache_disabled  # Should disable caching under memory pressure
        
    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self, router, test_request):
        """Test behavior when connection pools are exhausted."""
        # Exhaust connection pools
        router.max_connections = 10
        
        # Create many concurrent requests
        tasks = [router.route_request(test_request) for _ in range(50)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Some should be queued or rejected
        queued = sum(1 for r in responses 
                    if isinstance(r, dict) and r.get("queued"))
        assert queued > 0
        
    @pytest.mark.asyncio
    async def test_cpu_saturation(self, router, test_request):
        """Test behavior under CPU saturation."""
        # Simulate high CPU usage
        router.cpu_usage = 0.95  # 95% CPU
        
        # Should apply throttling
        response = await router.route_request(test_request)
        
        assert response.throttled
        assert response.processing_time_ms > 100  # Should be slower


class TestByzantineFailures:
    """Chaos tests for Byzantine failures."""
    
    @pytest.mark.asyncio
    async def test_inconsistent_provider_responses(self, router, test_request):
        """Test handling of inconsistent provider responses."""
        # Make providers return different results
        router.providers["openai"].process = AsyncMock(
            return_value=AIResponse(content="Response A")
        )
        router.providers["anthropic"].process = AsyncMock(
            return_value=AIResponse(content="Response B")
        )
        
        # Enable response validation
        router.enable_response_validation = True
        
        # Should detect inconsistency
        with pytest.raises(InconsistentResponseError):
            await router.route_with_validation(test_request)
            
    @pytest.mark.asyncio
    async def test_corrupted_provider_data(self, router, test_request, chaos_injector):
        """Test handling of corrupted provider data."""
        chaos_injector.active = True
        
        async def corrupted_process(*args, **kwargs):
            response = AIResponse(content="Test response")
            # Corrupt response
            return chaos_injector.corrupt_data(response.__dict__, probability=0.5)
            
        router.providers["openai"].process = corrupted_process
        
        # Should detect and handle corruption
        response = await router.route_request(test_request)
        
        assert response.corruption_detected or response.fallback_used


class TestSplitBrainScenarios:
    """Chaos tests for split-brain scenarios."""
    
    @pytest.mark.asyncio
    async def test_split_brain_detection(self, router):
        """Test detection of split-brain conditions."""
        # Simulate network partition creating split-brain
        router.simulate_split_brain(["openai", "anthropic"], ["google", "azure"])
        
        # Should detect split-brain
        split_brain_detected = await router.detect_split_brain()
        assert split_brain_detected
        
        # Should have mitigation strategy
        strategy = await router.get_split_brain_strategy()
        assert strategy["quorum_required"]
        assert strategy["consistency_level"] == "strong"
        
    @pytest.mark.asyncio
    async def test_split_brain_resolution(self, router):
        """Test resolution of split-brain conditions."""
        # Create split-brain
        router.split_brain_active = True
        router.partitions = [
            {"providers": ["openai", "anthropic"], "leader": "openai"},
            {"providers": ["google", "azure"], "leader": "google"}
        ]
        
        # Attempt resolution
        resolved = await router.resolve_split_brain()
        
        assert resolved
        assert router.split_brain_active is False
        assert len(router.partitions) == 1  # Unified partition


# ============================================================================
# PROPERTY-BASED TESTS
# ============================================================================

class ProviderRoutingStateMachine(RuleBasedStateMachine):
    """Stateful property testing for provider routing."""
    
    providers = Bundle('providers')
    requests = Bundle('requests')
    
    @rule(target=providers, 
          name=st.sampled_from(["openai", "anthropic", "google"]),
          health=st.floats(min_value=0, max_value=1),
          latency=st.floats(min_value=10, max_value=1000))
    def add_provider(self, name, health, latency):
        """Add a provider to the system."""
        provider = {
            "name": name,
            "health": health,
            "latency": latency,
            "available": health > 0.3
        }
        return provider
        
    @rule(target=requests,
          model=st.sampled_from(["gpt-4", "claude-3", "gemini"]),
          tokens=st.integers(min_value=10, max_value=10000))
    def create_request(self, model, tokens):
        """Create a request."""
        return {
            "model": model,
            "max_tokens": tokens,
            "timestamp": time.time()
        }
        
    @rule(provider=providers, request=requests)
    def route_request(self, provider, request):
        """Route a request to a provider."""
        if provider["available"]:
            # Should succeed
            assert provider["health"] > 0.3
            note(f"Routed to {provider['name']}")
        else:
            # Should fail and trigger fallback
            note(f"Provider {provider['name']} unavailable")
            
    @invariant()
    def at_least_one_provider_available(self):
        """System should maintain at least one available provider."""
        # In real system, check actual router state
        pass


@given(
    providers=st.lists(
        st.tuples(
            st.sampled_from(["openai", "anthropic", "google"]),
            st.floats(min_value=0, max_value=1),  # health
            st.floats(min_value=0.00001, max_value=0.001)  # cost
        ),
        min_size=1,
        max_size=5
    ),
    strategy=st.sampled_from(["cost", "speed", "balanced"])
)
@settings(max_examples=100)
async def test_provider_selection_properties(providers, strategy):
    """Test properties of provider selection."""
    router = MCPProviderRouter()
    
    # Configure providers
    for name, health, cost in providers:
        await router.add_provider(name, health=health, cost_per_token=cost)
        
    router.set_routing_strategy(strategy)
    
    # Select provider
    selection = await router.select_provider(AIRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=100
    ))
    
    # Properties that should always hold
    assert selection.provider in [p[0] for p in providers]
    
    if strategy == "cost":
        # Should select cheapest available provider
        available_providers = [p for p in providers if p[1] > 0.3]
        if available_providers:
            cheapest = min(available_providers, key=lambda p: p[2])
            assert selection.provider == cheapest[0]


# ============================================================================
# LOAD TESTING WITH LOCUST
# ============================================================================

class ProviderRouterUser(HttpUser):
    """Locust user for load testing provider router."""
    
    wait_time = between(0.1, 1)
    
    @task(3)
    def route_request(self):
        """Test request routing endpoint."""
        self.client.post("/route", json={
            "request": {
                "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": "Load test message"}
                ],
                "max_tokens": 100
            },
            "strategy": random.choice(["cost", "speed", "balanced"])
        })
        
    @task(1)
    def check_health(self):
        """Test health check endpoint."""
        self.client.get("/health")
        
    @task(1)
    def get_stats(self):
        """Test statistics endpoint."""
        self.client.get("/stats")
        
    def on_start(self):
        """Initialize user session."""
        # Authenticate or set up session
        pass


# ============================================================================
# TEST UTILITIES AND HELPERS
# ============================================================================

class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def generate_requests(count: int, 
                         models: List[str] = None,
                         token_range: Tuple[int, int] = (10, 1000)) -> List[AIRequest]:
        """Generate test requests."""
        if models is None:
            models = ["gpt-4", "gpt-3.5-turbo", "claude-3", "gemini-pro"]
            
        requests = []
        for i in range(count):
            requests.append(AIRequest(
                model=random.choice(models),
                messages=[
                    {"role": "system", "content": "Test system message"},
                    {"role": "user", "content": f"Test query {i}"}
                ],
                temperature=random.uniform(0, 1),
                max_tokens=random.randint(*token_range),
                stream=random.choice([True, False])
            ))
        return requests
        
    @staticmethod
    def generate_provider_states(providers: List[str]) -> Dict[str, Dict]:
        """Generate random provider states."""
        states = {}
        for provider in providers:
            states[provider] = {
                "health": random.uniform(0.5, 1.0),
                "latency_ms": random.uniform(20, 200),
                "error_rate": random.uniform(0, 0.1),
                "cost_per_token": random.uniform(0.00001, 0.0001),
                "available": random.random() > 0.1
            }
        return states


class MetricsCollector:
    """Collect and analyze test metrics."""
    
    def __init__(self):
        self.metrics = {
            "latencies": [],
            "success_count": 0,
            "failure_count": 0,
            "fallback_count": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
    def record_latency(self, latency_ms: float):
        """Record request latency."""
        self.metrics["latencies"].append(latency_ms)
        
    def record_success(self):
        """Record successful request."""
        self.metrics["success_count"] += 1
        
    def record_failure(self):
        """Record failed request."""
        self.metrics["failure_count"] += 1
        
    def get_statistics(self) -> Dict:
        """Calculate statistics from collected metrics."""
        if not self.metrics["latencies"]:
            return {}
            
        latencies = sorted(self.metrics["latencies"])
        total = self.metrics["success_count"] + self.metrics["failure_count"]
        
        return {
            "success_rate": self.metrics["success_count"] / total if total > 0 else 0,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": latencies[len(latencies) // 2],
            "p95_latency_ms": latencies[int(len(latencies) * 0.95)],
            "p99_latency_ms": latencies[int(len(latencies) * 0.99)],
            "cache_hit_rate": (self.metrics["cache_hits"] / 
                              (self.metrics["cache_hits"] + self.metrics["cache_misses"])
                              if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0
                              else 0)
        }


# ============================================================================
# TEST RUNNER CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    # Run tests with coverage and performance profiling
    pytest.main([
        __file__,
        "-v",
        "--cov=src.ai_providers",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--benchmark-only",
        "--benchmark-autosave",
        "-n", "auto",  # Use pytest-xdist for parallel execution
        "--tb=short"
    ])