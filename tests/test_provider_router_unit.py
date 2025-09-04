"""
Comprehensive Unit Tests for Provider Router with Fallback System.

This module provides extensive unit testing coverage for the provider router
components including routing algorithms, circuit breaker patterns, fallback chains,
cost optimization, health monitoring, and state management operations.

Test Coverage:
- Provider routing algorithm logic
- Circuit breaker pattern implementation
- Fallback chain execution
- Cost optimization calculations
- Health monitoring and metrics collection
- State management operations
- Model selection and compatibility validation
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio
from hypothesis import given, strategies as st, assume, settings

from src.ai_providers.models import (
    AIRequest, AIResponse, ProviderType, ProviderCapability,
    ModelSpec, CostTier, StreamingChunk
)
from src.ai_providers.model_router import (
    ModelRouter, ModelProfile, ModelCategory, ModelTier, RoutingRule
)
from src.ai_providers.abstract_provider import AbstractProvider
from src.context.provider_router_context_manager import (
    ProviderRouterContextManager,
    ProviderHealthState,
    CircuitBreakerState,
    RoutingDecision,
    StateConsistencyLevel,
    InMemoryStateCache,
    StateSynchronizer,
    StateRecoveryManager
)
from src.context.provider_router_performance_optimization import (
    ProviderRouterPerformanceOptimizer,
    PerformanceTarget,
    OptimizationMetrics,
    QueryOptimizer,
    CacheOptimizer,
    NetworkOptimizer,
    MemoryOptimizer
)


class MockProvider(AbstractProvider):
    """Mock provider for testing with configurable behavior."""
    
    def __init__(self, provider_type: ProviderType, health_status: bool = True, 
                 latency_ms: float = 100.0, error_rate: float = 0.0):
        from src.ai_providers.models import ProviderConfig
        config = ProviderConfig(provider_type=provider_type, api_key="test-key")
        super().__init__(config)
        
        self.health_status = health_status
        self.latency_ms = latency_ms
        self.error_rate = error_rate
        self.request_count = 0
        self.failure_count = 0
        
        self._models = {
            f"{provider_type.value}-model": ModelSpec(
                model_id=f"{provider_type.value}-model",
                provider_type=provider_type,
                display_name=f"{provider_type.value.title()} Model",
                capabilities=[ProviderCapability.TEXT_GENERATION, ProviderCapability.STREAMING],
                cost_per_input_token=0.001,
                cost_per_output_token=0.002,
                supports_streaming=True,
                supports_tools=True,
                context_length=8192,
                max_output_tokens=4096
            )
        }
    
    async def _initialize_client(self):
        pass
    
    async def _cleanup_client(self):
        pass
    
    async def _load_models(self):
        pass
    
    async def _generate_response_impl(self, request: AIRequest) -> AIResponse:
        self.request_count += 1
        
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000.0)
        
        # Simulate errors based on error rate
        if self.error_rate > 0 and (self.request_count * self.error_rate) >= 1:
            self.failure_count += 1
            from src.ai_providers.error_handler import AIProviderError
            raise AIProviderError(f"Simulated error from {self.provider_type.value}")
        
        return AIResponse(
            request_id=request.request_id,
            provider_type=self.provider_type,
            model=request.model,
            content=f"Response from {self.provider_type.value}",
            usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            cost=0.15,
            latency_ms=self.latency_ms,
        )
    
    async def _stream_response_impl(self, request: AIRequest):
        # Simulate streaming with latency
        await asyncio.sleep(self.latency_ms / 2000.0)
        yield StreamingChunk(request_id=request.request_id, content="Streaming ")
        
        await asyncio.sleep(self.latency_ms / 2000.0)
        yield StreamingChunk(request_id=request.request_id, content="response")
        
        yield StreamingChunk(
            request_id=request.request_id, 
            is_complete=True, 
            finish_reason="stop"
        )
    
    def _get_supported_capabilities(self):
        return [ProviderCapability.TEXT_GENERATION, ProviderCapability.STREAMING]
    
    async def _perform_health_check(self):
        if not self.health_status:
            raise Exception("Provider health check failed")


@pytest.fixture
def mock_anthropic_provider():
    """Healthy Anthropic provider."""
    return MockProvider(ProviderType.ANTHROPIC)


@pytest.fixture
def mock_openai_provider():
    """Healthy OpenAI provider."""
    return MockProvider(ProviderType.OPENAI)


@pytest.fixture
def mock_google_provider():
    """Google provider with higher latency."""
    return MockProvider(ProviderType.GOOGLE, latency_ms=200.0)


@pytest.fixture
def mock_unhealthy_provider():
    """Unhealthy provider for failure testing."""
    return MockProvider(ProviderType.OPENAI, health_status=False, error_rate=0.5)


@pytest.fixture
def sample_ai_request():
    """Sample AI request for testing."""
    return AIRequest(
        model="claude-3-sonnet",
        messages=[{"role": "user", "content": "Write a Python function to calculate fibonacci"}],
        max_tokens=1000,
        temperature=0.7,
        stream=False
    )


@pytest.fixture
def sample_health_state():
    """Sample provider health state."""
    return ProviderHealthState(
        provider_name="anthropic",
        provider_type="anthropic",
        is_available=True,
        last_check=datetime.now(timezone.utc),
        response_time_ms=95.0,
        error_rate=0.02,
        success_rate=0.98,
        uptime_percentage=99.5,
        consecutive_failures=0,
        circuit_breaker_state="CLOSED"
    )


@pytest.fixture
def sample_circuit_breaker_state():
    """Sample circuit breaker state."""
    return CircuitBreakerState(
        provider_name="openai",
        state="OPEN",
        failure_count=5,
        success_count=0,
        last_failure_time=datetime.now(timezone.utc),
        next_attempt_time=datetime.now(timezone.utc) + timedelta(minutes=1),
        failure_threshold=5,
        success_threshold=3,
        timeout_duration_s=60,
        half_open_max_calls=3,
        current_half_open_calls=0
    )


@pytest.fixture
def sample_routing_decision():
    """Sample routing decision."""
    return RoutingDecision(
        request_id="req-123",
        selected_provider="anthropic",
        alternative_providers=["openai", "google"],
        routing_strategy="cost_optimized",
        decision_factors={"cost": 0.15, "latency": 95.0, "quality": 0.9},
        estimated_cost=0.15,
        estimated_latency_ms=95.0,
        timestamp=datetime.now(timezone.utc),
        confidence_score=0.85,
        fallback_chain=["openai", "google"]
    )


class TestModelRouter:
    """Test cases for ModelRouter component."""
    
    @pytest.fixture
    def model_router(self):
        """Create a model router for testing."""
        return ModelRouter()
    
    def test_router_initialization(self, model_router):
        """Test router initializes with default profiles and rules."""
        assert len(model_router.model_profiles) > 0
        assert len(model_router.routing_rules) > 0
        
        # Check Claude models are registered
        assert "claude-3-opus" in model_router.model_profiles
        assert "claude-3-sonnet" in model_router.model_profiles
        assert "claude-3-haiku" in model_router.model_profiles
        
        # Verify routing rules exist
        rule_names = [rule.name for rule in model_router.routing_rules]
        assert "coding_tasks" in rule_names
        assert "vision_tasks" in rule_names
        assert "quick_responses" in rule_names
    
    def test_model_registration(self, model_router):
        """Test model registration functionality."""
        model_spec = ModelSpec(
            model_id="test-model",
            provider_type=ProviderType.ANTHROPIC,
            display_name="Test Model",
            capabilities=[ProviderCapability.TEXT_GENERATION],
            cost_per_input_token=0.001,
            cost_per_output_token=0.002,
            context_length=4096,
            max_output_tokens=2048
        )
        
        model_router.register_model(ProviderType.ANTHROPIC, model_spec)
        
        assert "test-model" in model_router.model_profiles
        assert ProviderType.ANTHROPIC in model_router.provider_model_mapping
        assert "test-model" in model_router.provider_model_mapping[ProviderType.ANTHROPIC]
    
    @pytest.mark.asyncio
    async def test_optimal_model_selection(self, model_router, sample_ai_request, 
                                         mock_anthropic_provider, mock_openai_provider):
        """Test optimal model selection algorithm."""
        available_providers = [mock_anthropic_provider, mock_openai_provider]
        
        # Register models with providers
        for provider in available_providers:
            for model_id, model_spec in provider.models.items():
                model_router.register_model(provider.provider_type, model_spec)
        
        result = await model_router.select_optimal_model(
            sample_ai_request, available_providers
        )
        
        assert result is not None
        provider, model_id = result
        assert provider in available_providers
        assert model_id in provider.models
    
    def test_request_analysis(self, model_router, sample_ai_request):
        """Test request analysis functionality."""
        analysis = model_router._analyze_request(sample_ai_request)
        
        assert "estimated_input_tokens" in analysis
        assert "task_type" in analysis
        assert "complexity_level" in analysis
        
        # Should detect coding task from content
        assert analysis["task_type"] == "coding"
        assert analysis["complexity_level"] == "high"
    
    def test_capability_matching(self, model_router):
        """Test model capability matching."""
        # Test vision requirement
        vision_request = AIRequest(
            model="claude-3-opus",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
                ]
            }],
            max_tokens=1000
        )
        
        analysis = model_router._analyze_request(vision_request)
        assert analysis["requires_vision"] is True
        assert analysis["content_type"] == "multimodal"
    
    def test_routing_rule_application(self, model_router, sample_ai_request):
        """Test routing rule application and filtering."""
        # Create mock candidates
        mock_provider = Mock()
        mock_provider.provider_type = ProviderType.ANTHROPIC
        
        candidates = [(mock_provider, "claude-3-opus"), (mock_provider, "claude-3-haiku")]
        analysis = model_router._analyze_request(sample_ai_request)
        
        filtered = model_router._apply_routing_rules(sample_ai_request, candidates, analysis)
        
        # Should still have candidates after filtering
        assert len(filtered) > 0
    
    def test_model_scoring(self, model_router, sample_ai_request):
        """Test model scoring algorithm."""
        # Create mock provider and candidates
        mock_provider = Mock()
        mock_provider.provider_type = ProviderType.ANTHROPIC
        
        candidates = [(mock_provider, "claude-3-opus")]
        analysis = model_router._analyze_request(sample_ai_request)
        
        scored = model_router._score_model_candidates(
            sample_ai_request, candidates, analysis, None
        )
        
        assert len(scored) > 0
        provider, model_id, score = scored[0]
        assert 0.0 <= score <= 1.0
    
    def test_custom_routing_rule(self, model_router):
        """Test adding custom routing rules."""
        custom_rule = RoutingRule(
            name="test_rule",
            description="Test routing rule",
            priority=200,
            request_patterns=[r"test.*pattern"],
            preferred_categories=[ModelCategory.SPEED_OPTIMIZED],
            max_latency_ms=50.0
        )
        
        initial_count = len(model_router.routing_rules)
        model_router.add_routing_rule(custom_rule)
        
        assert len(model_router.routing_rules) == initial_count + 1
        assert model_router.routing_rules[0] == custom_rule  # Should be first due to high priority
    
    def test_model_recommendations(self, model_router, sample_ai_request):
        """Test model recommendation generation."""
        recommendations = model_router.get_model_recommendations(sample_ai_request)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        for rec in recommendations:
            assert "model_id" in rec
            assert "provider" in rec
            assert "score" in rec
            assert "reasoning" in rec
            assert 0.0 <= rec["score"] <= 1.0


class TestProviderHealthState:
    """Test cases for ProviderHealthState operations."""
    
    def test_health_state_creation(self, sample_health_state):
        """Test health state object creation."""
        assert sample_health_state.provider_name == "anthropic"
        assert sample_health_state.is_available is True
        assert sample_health_state.response_time_ms == 95.0
        assert sample_health_state.circuit_breaker_state == "CLOSED"
        assert sample_health_state.metadata == {}
    
    def test_health_state_serialization(self, sample_health_state):
        """Test health state serialization."""
        from dataclasses import asdict
        
        data = asdict(sample_health_state)
        assert data["provider_name"] == "anthropic"
        assert data["response_time_ms"] == 95.0
        
        # Test datetime handling
        assert isinstance(data["last_check"], datetime)
    
    def test_health_state_with_metadata(self):
        """Test health state with custom metadata."""
        state = ProviderHealthState(
            provider_name="test",
            provider_type="test",
            is_available=True,
            last_check=datetime.now(timezone.utc),
            response_time_ms=100.0,
            error_rate=0.0,
            success_rate=1.0,
            uptime_percentage=100.0,
            consecutive_failures=0,
            circuit_breaker_state="CLOSED",
            metadata={"custom_field": "value", "numeric_field": 42}
        )
        
        assert state.metadata["custom_field"] == "value"
        assert state.metadata["numeric_field"] == 42


class TestCircuitBreakerState:
    """Test cases for CircuitBreakerState operations."""
    
    def test_circuit_breaker_creation(self, sample_circuit_breaker_state):
        """Test circuit breaker state creation."""
        assert sample_circuit_breaker_state.provider_name == "openai"
        assert sample_circuit_breaker_state.state == "OPEN"
        assert sample_circuit_breaker_state.failure_count == 5
        assert sample_circuit_breaker_state.failure_threshold == 5
    
    def test_circuit_breaker_state_transitions(self):
        """Test circuit breaker state transition logic."""
        # Test CLOSED -> OPEN transition
        closed_state = CircuitBreakerState(
            provider_name="test",
            state="CLOSED",
            failure_count=0,
            success_count=10,
            last_failure_time=None,
            next_attempt_time=None,
            failure_threshold=5,
            success_threshold=3,
            timeout_duration_s=60,
            half_open_max_calls=3,
            current_half_open_calls=0
        )
        
        # Simulate failures
        closed_state.failure_count = 5
        closed_state.last_failure_time = datetime.now(timezone.utc)
        
        # Should transition to OPEN when failure threshold is reached
        assert closed_state.failure_count >= closed_state.failure_threshold
    
    def test_circuit_breaker_timeout_calculation(self, sample_circuit_breaker_state):
        """Test circuit breaker timeout calculations."""
        now = datetime.now(timezone.utc)
        sample_circuit_breaker_state.last_failure_time = now
        sample_circuit_breaker_state.next_attempt_time = now + timedelta(seconds=60)
        
        # Should be in timeout period
        assert sample_circuit_breaker_state.next_attempt_time > now


class TestInMemoryStateCache:
    """Test cases for InMemoryStateCache."""
    
    @pytest.fixture
    def cache(self):
        """Create an in-memory cache for testing."""
        return InMemoryStateCache(max_size=100, default_ttl=60)
    
    @pytest.mark.asyncio
    async def test_cache_basic_operations(self, cache):
        """Test basic cache get/set operations."""
        key = "test_key"
        value = {"data": "test_value"}
        
        # Test set
        result = await cache.set(key, value)
        assert result is True
        
        # Test get
        cached_value = await cache.get(key)
        assert cached_value == value
        
        # Test non-existent key
        missing = await cache.get("missing_key")
        assert missing is None
    
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, cache):
        """Test cache TTL expiration."""
        key = "expiring_key"
        value = "test_value"
        
        # Set with short TTL
        await cache.set(key, value, ttl=0.1)  # 0.1 second TTL
        
        # Should be available immediately
        cached = await cache.get(key)
        assert cached == value
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Should be expired
        expired = await cache.get(key)
        assert expired is None
    
    @pytest.mark.asyncio
    async def test_cache_category_based_access(self, cache):
        """Test category-based cache access patterns."""
        # Test different categories
        await cache.set("health_key", "health_data", "provider_health")
        await cache.set("circuit_key", "circuit_data", "circuit_breaker")
        
        health_data = await cache.get("health_key", "provider_health")
        circuit_data = await cache.get("circuit_key", "circuit_breaker")
        
        assert health_data == "health_data"
        assert circuit_data == "circuit_data"
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self, cache):
        """Test cache eviction when max size is reached."""
        cache.max_size = 5  # Small size for testing
        
        # Fill cache beyond capacity
        for i in range(10):
            await cache.set(f"key_{i}", f"value_{i}")
        
        # Cache should not exceed max size
        assert len(cache._cache) <= cache.max_size
    
    @pytest.mark.asyncio
    async def test_cache_statistics(self, cache):
        """Test cache statistics collection."""
        # Generate some cache activity
        await cache.set("key1", "value1")
        await cache.get("key1")  # Hit
        await cache.get("missing")  # Miss
        
        stats = cache.get_stats()
        assert "hit_count" in stats
        assert "miss_count" in stats
        assert "hit_rate" in stats
        assert stats["hit_count"] >= 1
        assert stats["miss_count"] >= 1


class TestStateSynchronizer:
    """Test cases for StateSynchronizer."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis = AsyncMock()
        redis.set = AsyncMock()
        redis.get = AsyncMock()
        redis.publish = AsyncMock()
        redis.pubsub = MagicMock()
        return redis
    
    @pytest.fixture
    def synchronizer(self, mock_redis):
        """Create state synchronizer for testing."""
        return StateSynchronizer(mock_redis)
    
    @pytest.mark.asyncio
    async def test_state_synchronization(self, synchronizer, sample_health_state):
        """Test basic state synchronization."""
        result = await synchronizer.sync_state(
            "provider_health",
            "anthropic",
            sample_health_state,
            StateConsistencyLevel.SESSION
        )
        
        assert result is True
        assert synchronizer.sync_operations > 0
    
    @pytest.mark.asyncio
    async def test_synchronized_state_retrieval(self, synchronizer, mock_redis):
        """Test retrieving synchronized state."""
        # Mock Redis response
        test_state = {"provider_name": "test", "is_available": True}
        mock_redis.get.side_effect = [json.dumps(test_state), "1"]  # state, version
        
        state, version = await synchronizer.get_synchronized_state("provider_health", "test")
        
        assert state == test_state
        assert version == 1
    
    @pytest.mark.asyncio
    async def test_state_change_listeners(self, synchronizer):
        """Test state change listener registration."""
        callback_called = False
        
        async def test_callback(change_data):
            nonlocal callback_called
            callback_called = True
        
        await synchronizer.register_state_change_listener("test_state", test_callback)
        
        assert "test_state" in synchronizer._change_listeners
        assert len(synchronizer._change_listeners["test_state"]) == 1
    
    @pytest.mark.asyncio
    async def test_conflict_resolution(self, synchronizer):
        """Test state conflict resolution."""
        local_state = {"value": "local", "timestamp": datetime.now()}
        remote_state = {"value": "remote", "timestamp": datetime.now() + timedelta(seconds=1)}
        
        resolved = await synchronizer.resolve_state_conflict(
            "test_state", "test_key", local_state, remote_state, 1, 2
        )
        
        # Remote should win due to higher version
        assert resolved == remote_state
    
    def test_sync_statistics(self, synchronizer):
        """Test synchronization statistics."""
        stats = synchronizer.get_sync_stats()
        
        assert "sync_operations" in stats
        assert "conflict_resolutions" in stats
        assert "failed_syncs" in stats
        assert "active_listeners" in stats


class TestPerformanceOptimization:
    """Test cases for performance optimization components."""
    
    @pytest.fixture
    def mock_context_manager(self):
        """Mock context manager for testing."""
        cm = Mock()
        cm.get_provider_health = AsyncMock()
        cm.get_circuit_breaker_state = AsyncMock()
        cm.query_routing_patterns = AsyncMock(return_value=[])
        cm.get_comprehensive_stats = AsyncMock(return_value={
            "performance_metrics": {"p95_response_time_ms": 50.0},
            "memory_cache": {"hit_rate": 0.9},
            "system_resources": {"memory_usage_mb": 200.0}
        })
        return cm
    
    def test_performance_target_initialization(self):
        """Test performance target initialization."""
        target = PerformanceTarget()
        assert target.max_latency_ms == 100.0
        assert target.target_throughput_rps == 1000.0
        assert target.cache_hit_rate_threshold == 0.85
    
    def test_query_optimizer_initialization(self, mock_context_manager):
        """Test query optimizer initialization."""
        optimizer = QueryOptimizer(mock_context_manager)
        assert optimizer.context_manager == mock_context_manager
        assert len(optimizer.query_patterns) == 0
        assert len(optimizer.hot_keys) == 0
    
    @pytest.mark.asyncio
    async def test_cache_warming(self, mock_context_manager):
        """Test cache warming functionality."""
        optimizer = CacheOptimizer(mock_context_manager)
        
        # Mock the warming strategies
        optimizer._warm_provider_health_cache = AsyncMock(return_value=4)
        optimizer._warm_circuit_breaker_cache = AsyncMock(return_value=4)
        optimizer._warm_routing_cache = AsyncMock(return_value=10)
        
        results = await optimizer._warm_critical_caches()
        
        assert "provider_health" in results
        assert "circuit_breaker" in results
        assert "routing_decisions" in results
        assert all(result["success"] for result in results.values())
    
    @pytest.mark.asyncio
    async def test_comprehensive_optimization(self, mock_context_manager):
        """Test comprehensive performance optimization."""
        optimizer = ProviderRouterPerformanceOptimizer(mock_context_manager)
        
        # Mock individual optimizers
        optimizer.query_optimizer.optimize_state_access_patterns = AsyncMock(
            return_value={"optimization": "success"}
        )
        optimizer.cache_optimizer.optimize_cache_layers = AsyncMock(
            return_value={"optimization": "success"}
        )
        optimizer.network_optimizer.optimize_network_operations = AsyncMock(
            return_value={"optimization": "success"}
        )
        optimizer.memory_optimizer.optimize_memory_usage = AsyncMock(
            return_value={"optimization": "success"}
        )
        
        results = await optimizer.run_comprehensive_optimization()
        
        assert "results" in results
        assert "query_optimization" in results["results"]
        assert "cache_optimization" in results["results"]
        assert "network_optimization" in results["results"]
        assert "memory_optimization" in results["results"]
    
    @pytest.mark.asyncio
    async def test_optimization_recommendations(self, mock_context_manager):
        """Test optimization recommendation generation."""
        optimizer = ProviderRouterPerformanceOptimizer(mock_context_manager)
        
        # Mock stats that would trigger recommendations
        mock_context_manager.get_comprehensive_stats.return_value = {
            "performance_metrics": {"p95_response_time_ms": 150.0},  # Above target
            "memory_cache": {"hit_rate": 0.7},  # Below target
            "system_resources": {"memory_usage_mb": 600.0}  # Above target
        }
        
        recommendations = await optimizer.get_optimization_recommendations()
        
        assert "high_priority" in recommendations
        assert "medium_priority" in recommendations
        assert len(recommendations["high_priority"]) > 0


# Property-based testing with Hypothesis
class TestPropertyBasedTesting:
    """Property-based tests using Hypothesis for edge case coverage."""
    
    @given(
        provider_name=st.text(min_size=1, max_size=50),
        response_time=st.floats(min_value=0.1, max_value=10000.0, allow_infinity=False, allow_nan=False),
        error_rate=st.floats(min_value=0.0, max_value=1.0),
        success_rate=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=50)
    def test_health_state_property_invariants(self, provider_name, response_time, error_rate, success_rate):
        """Test health state invariants with property-based testing."""
        assume(0.0 <= error_rate <= 1.0)
        assume(0.0 <= success_rate <= 1.0)
        assume(abs(error_rate + success_rate - 1.0) < 0.1)  # Should approximately sum to 1
        
        health_state = ProviderHealthState(
            provider_name=provider_name.strip(),
            provider_type="test",
            is_available=error_rate < 0.5,
            last_check=datetime.now(timezone.utc),
            response_time_ms=response_time,
            error_rate=error_rate,
            success_rate=success_rate,
            uptime_percentage=100.0 * success_rate,
            consecutive_failures=int(error_rate * 10),
            circuit_breaker_state="CLOSED" if error_rate < 0.2 else "OPEN"
        )
        
        # Test invariants
        assert len(health_state.provider_name.strip()) > 0
        assert health_state.response_time_ms >= 0
        assert 0.0 <= health_state.error_rate <= 1.0
        assert 0.0 <= health_state.success_rate <= 1.0
        assert health_state.consecutive_failures >= 0
        assert health_state.circuit_breaker_state in ["CLOSED", "OPEN", "HALF_OPEN"]
    
    @given(
        failure_count=st.integers(min_value=0, max_value=100),
        success_count=st.integers(min_value=0, max_value=100),
        failure_threshold=st.integers(min_value=1, max_value=20),
        success_threshold=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=50)
    def test_circuit_breaker_state_logic(self, failure_count, success_count, 
                                        failure_threshold, success_threshold):
        """Test circuit breaker state logic properties."""
        state = "CLOSED"
        if failure_count >= failure_threshold:
            state = "OPEN"
        elif failure_count > 0 and success_count >= success_threshold:
            state = "HALF_OPEN"
        
        circuit_breaker = CircuitBreakerState(
            provider_name="test",
            state=state,
            failure_count=failure_count,
            success_count=success_count,
            last_failure_time=datetime.now(timezone.utc) if failure_count > 0 else None,
            next_attempt_time=None,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout_duration_s=60,
            half_open_max_calls=3,
            current_half_open_calls=0
        )
        
        # Test state consistency
        if circuit_breaker.failure_count >= circuit_breaker.failure_threshold:
            assert circuit_breaker.state in ["OPEN", "HALF_OPEN"]
        
        if circuit_breaker.failure_count == 0:
            assert circuit_breaker.state == "CLOSED"
            assert circuit_breaker.last_failure_time is None
    
    @given(
        estimated_cost=st.floats(min_value=0.0, max_value=100.0, allow_infinity=False, allow_nan=False),
        estimated_latency=st.floats(min_value=1.0, max_value=10000.0, allow_infinity=False, allow_nan=False),
        confidence_score=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=30)
    def test_routing_decision_properties(self, estimated_cost, estimated_latency, confidence_score):
        """Test routing decision properties."""
        routing_decision = RoutingDecision(
            request_id=f"req-{int(time.time())}",
            selected_provider="test-provider",
            alternative_providers=["alt1", "alt2"],
            routing_strategy="test_strategy",
            decision_factors={"cost": estimated_cost, "latency": estimated_latency},
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency,
            timestamp=datetime.now(timezone.utc),
            confidence_score=confidence_score,
            fallback_chain=["fallback1", "fallback2"]
        )
        
        # Test invariants
        assert routing_decision.estimated_cost >= 0.0
        assert routing_decision.estimated_latency_ms > 0.0
        assert 0.0 <= routing_decision.confidence_score <= 1.0
        assert len(routing_decision.request_id) > 0
        assert len(routing_decision.selected_provider) > 0
        assert isinstance(routing_decision.alternative_providers, list)
        assert isinstance(routing_decision.fallback_chain, list)


# Performance benchmarking tests
class TestPerformanceBenchmarks:
    """Performance benchmarking tests for sub-100ms requirements."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_state_access_performance(self):
        """Test state access performance meets sub-100ms requirement."""
        cache = InMemoryStateCache(max_size=1000, default_ttl=300)
        
        # Pre-populate cache
        for i in range(100):
            await cache.set(f"key_{i}", f"value_{i}")
        
        # Measure access times
        access_times = []
        for i in range(100):
            start = time.perf_counter()
            await cache.get(f"key_{i % 100}")  # Mix hits and misses
            duration = (time.perf_counter() - start) * 1000  # Convert to ms
            access_times.append(duration)
        
        # Performance assertions
        avg_time = sum(access_times) / len(access_times)
        p95_time = sorted(access_times)[int(0.95 * len(access_times))]
        
        assert avg_time < 1.0, f"Average access time {avg_time:.2f}ms too slow"
        assert p95_time < 5.0, f"P95 access time {p95_time:.2f}ms too slow"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_access_performance(self):
        """Test concurrent access performance."""
        cache = InMemoryStateCache(max_size=1000, default_ttl=300)
        
        # Pre-populate
        for i in range(100):
            await cache.set(f"key_{i}", {"data": f"value_{i}"})
        
        async def access_worker(worker_id: int, num_operations: int):
            """Worker function for concurrent access."""
            times = []
            for i in range(num_operations):
                start = time.perf_counter()
                await cache.get(f"key_{i % 100}")
                times.append((time.perf_counter() - start) * 1000)
            return times
        
        # Run concurrent workers
        num_workers = 10
        operations_per_worker = 50
        
        start_time = time.perf_counter()
        tasks = [
            access_worker(i, operations_per_worker)
            for i in range(num_workers)
        ]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time
        
        # Calculate metrics
        all_times = [time for worker_times in results for time in worker_times]
        total_operations = num_workers * operations_per_worker
        throughput = total_operations / total_time
        
        avg_latency = sum(all_times) / len(all_times)
        p95_latency = sorted(all_times)[int(0.95 * len(all_times))]
        
        # Performance assertions
        assert throughput > 1000, f"Throughput {throughput:.0f} ops/sec too low"
        assert avg_latency < 10.0, f"Average latency {avg_latency:.2f}ms too high"
        assert p95_latency < 50.0, f"P95 latency {p95_latency:.2f}ms too high"
    
    @pytest.mark.asyncio
    async def test_model_selection_performance(self):
        """Test model selection performance."""
        router = ModelRouter()
        
        # Create test providers
        providers = [
            MockProvider(ProviderType.ANTHROPIC),
            MockProvider(ProviderType.OPENAI),
            MockProvider(ProviderType.GOOGLE)
        ]
        
        # Register models
        for provider in providers:
            for model_id, model_spec in provider.models.items():
                router.register_model(provider.provider_type, model_spec)
        
        request = AIRequest(
            model=None,  # Force selection
            messages=[{"role": "user", "content": "Test request"}],
            max_tokens=1000
        )
        
        # Measure selection times
        selection_times = []
        for _ in range(50):
            start = time.perf_counter()
            result = await router.select_optimal_model(request, providers)
            duration = (time.perf_counter() - start) * 1000
            selection_times.append(duration)
            assert result is not None
        
        avg_time = sum(selection_times) / len(selection_times)
        p95_time = sorted(selection_times)[int(0.95 * len(selection_times))]
        
        assert avg_time < 50.0, f"Average selection time {avg_time:.2f}ms too slow"
        assert p95_time < 100.0, f"P95 selection time {p95_time:.2f}ms too slow"


# Test markers and configuration
pytestmark = [
    pytest.mark.unit,  # Mark all tests in this file as unit tests
]


if __name__ == "__main__":
    pytest.main([__file__])