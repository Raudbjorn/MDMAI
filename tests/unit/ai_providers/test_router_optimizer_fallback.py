"""
Comprehensive tests for intelligent router, cost optimizer, and fallback manager.
Addresses PR #59 review issues: realistic test scenarios, consistent targets, proper imports.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import List, Dict, Any

# Proper imports - no relative imports
from src.ai_providers.models import (
    AIRequest,
    AIResponse,
    ProviderType,
    ProviderCapability,
    ModelSpec,
    CostTier,
)
from src.ai_providers.abstract_provider import AbstractProvider
from src.ai_providers.health_monitor import HealthMonitor, ProviderMetrics, ProviderStatus
from src.ai_providers.intelligent_router import (
    IntelligentRouter,
    SelectionStrategy,
    SelectionCriteria,
    ProviderScore,
)
from src.ai_providers.advanced_cost_optimizer import (
    AdvancedCostOptimizer,
    OptimizationStrategy,
    CostMetrics,
    BudgetAlert,
    AlertSeverity,
)
from src.ai_providers.fallback_manager import (
    FallbackManager,
    FallbackTier,
    CircuitState,
    CircuitBreakerConfig,
    FallbackRule,
)
from src.ai_providers.config.model_config import (
    ModelConfigManager,
    ModelProfile,
    ModelCostConfig,
    NormalizationConfig,
)
from src.ai_providers.utils.cost_utils import (
    ErrorClassification,
    classify_error,
    estimate_input_tokens,
    estimate_request_cost,
)


# Mock provider for testing
class MockProvider(AbstractProvider):
    """Mock provider for testing."""
    
    def __init__(
        self,
        provider_type: ProviderType,
        models: Dict[str, ModelSpec],
        is_available: bool = True,
        should_fail: bool = False,
        latency_ms: float = 1000.0,
    ):
        self.provider_type = provider_type
        self.models = models
        self.is_available = is_available
        self.should_fail = should_fail
        self.latency_ms = latency_ms
        self.request_count = 0
    
    async def generate_response(self, request: AIRequest) -> AIResponse:
        """Generate a mock response."""
        self.request_count += 1
        
        if self.should_fail:
            raise Exception("Provider failed")
        
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000.0)
        
        return AIResponse(
            content="Mock response",
            model=request.model,
            provider=self.provider_type,
            usage={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
            cost=0.01,
            latency_ms=self.latency_ms,
            request_id=request.request_id,
        )


class TestIntelligentRouter:
    """Test intelligent router functionality."""
    
    @pytest.fixture
    def config_manager(self):
        """Create a test config manager."""
        return ModelConfigManager()
    
    @pytest.fixture
    def health_monitor(self):
        """Create a test health monitor."""
        monitor = MagicMock(spec=HealthMonitor)
        
        # Default metrics
        default_metrics = ProviderMetrics(
            provider_type=ProviderType.ANTHROPIC,
            status=ProviderStatus.HEALTHY,
            uptime_percentage=99.5,
            avg_latency_ms=2000.0,
            error_rate=0.01,
            total_requests=1000,
            successful_requests=990,
            failed_requests=10,
        )
        
        monitor.get_metrics.return_value = default_metrics
        return monitor
    
    @pytest.fixture
    def router(self, health_monitor, config_manager):
        """Create a test router."""
        return IntelligentRouter(
            health_monitor=health_monitor,
            config_manager=config_manager,
        )
    
    @pytest.fixture
    def mock_providers(self):
        """Create mock providers for testing."""
        return [
            MockProvider(
                provider_type=ProviderType.ANTHROPIC,
                models={
                    "claude-3-opus": ModelSpec(
                        model_id="claude-3-opus",
                        context_length=200000,
                        max_output_tokens=4096,
                        cost_tier=CostTier.PREMIUM,
                        capabilities=[ProviderCapability.GENERAL, ProviderCapability.CODING],
                        supports_streaming=True,
                        supports_tools=True,
                        supports_vision=True,
                    ),
                },
                latency_ms=3000.0,  # Realistic latency for premium model
            ),
            MockProvider(
                provider_type=ProviderType.OPENAI,
                models={
                    "gpt-3.5-turbo": ModelSpec(
                        model_id="gpt-3.5-turbo",
                        context_length=16385,
                        max_output_tokens=4096,
                        cost_tier=CostTier.LOW,
                        capabilities=[ProviderCapability.GENERAL],
                        supports_streaming=True,
                        supports_tools=True,
                    ),
                },
                latency_ms=1500.0,  # Realistic latency for fast model
            ),
            MockProvider(
                provider_type=ProviderType.GOOGLE,
                models={
                    "gemini-pro": ModelSpec(
                        model_id="gemini-pro",
                        context_length=32768,
                        max_output_tokens=2048,
                        cost_tier=CostTier.LOW,
                        capabilities=[ProviderCapability.GENERAL],
                        supports_streaming=True,
                    ),
                },
                latency_ms=2000.0,  # Realistic latency
            ),
        ]
    
    @pytest.mark.asyncio
    async def test_weighted_composite_selection(self, router, mock_providers):
        """Test weighted composite provider selection."""
        request = AIRequest(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "Test message"}],
            max_tokens=100,
        )
        
        criteria = SelectionCriteria(
            cost_weight=0.3,
            speed_weight=0.3,
            quality_weight=0.2,
            reliability_weight=0.2,
        )
        
        result = await router.select_optimal_provider(
            request=request,
            available_providers=mock_providers[:1],  # Only Anthropic
            strategy=SelectionStrategy.WEIGHTED_COMPOSITE,
            criteria=criteria,
        )
        
        assert result is not None
        assert isinstance(result, ProviderScore)
        assert result.provider_type == ProviderType.ANTHROPIC
        assert result.model_id == "claude-3-opus"
        assert 0.0 <= result.total_score <= 1.0
        assert result.estimated_cost > 0
        assert result.estimated_latency_ms > 0
    
    @pytest.mark.asyncio
    async def test_cost_optimized_selection(self, router, mock_providers):
        """Test cost-optimized provider selection."""
        request = AIRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Simple query"}],
            max_tokens=50,
        )
        
        # Filter to only providers with the model
        available = [p for p in mock_providers if "gpt-3.5-turbo" in p.models]
        
        result = await router.select_optimal_provider(
            request=request,
            available_providers=available,
            strategy=SelectionStrategy.COST_OPTIMIZED,
        )
        
        assert result is not None
        assert result.provider_type == ProviderType.OPENAI
        assert "cost" in result.selection_reason.lower()
    
    @pytest.mark.asyncio
    async def test_speed_optimized_selection(self, router, mock_providers):
        """Test speed-optimized provider selection."""
        request = AIRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Need fast response"}],
            max_tokens=50,
        )
        
        available = [p for p in mock_providers if "gpt-3.5-turbo" in p.models]
        
        result = await router.select_optimal_provider(
            request=request,
            available_providers=available,
            strategy=SelectionStrategy.SPEED_OPTIMIZED,
        )
        
        assert result is not None
        assert "response time" in result.selection_reason.lower() or "fast" in result.selection_reason.lower()
    
    @pytest.mark.asyncio
    async def test_capability_filtering(self, router, mock_providers):
        """Test provider filtering by capabilities."""
        request = AIRequest(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "Code this"}],
            tools=[{"name": "test_tool", "description": "Test"}],
        )
        
        criteria = SelectionCriteria(
            required_capabilities=[ProviderCapability.CODING],
            prefer_tools=True,
        )
        
        result = await router.select_optimal_provider(
            request=request,
            available_providers=mock_providers,
            criteria=criteria,
        )
        
        assert result is not None
        # Should select Anthropic as it has coding capability
        assert result.provider_type == ProviderType.ANTHROPIC
    
    @pytest.mark.asyncio
    async def test_health_based_filtering(self, router, mock_providers, health_monitor):
        """Test filtering unhealthy providers."""
        # Make one provider unhealthy
        unhealthy_metrics = ProviderMetrics(
            provider_type=ProviderType.OPENAI,
            status=ProviderStatus.DEGRADED,
            uptime_percentage=75.0,  # Below 80% threshold
            avg_latency_ms=10000.0,
            error_rate=0.3,
            total_requests=100,
            successful_requests=70,
            failed_requests=30,
        )
        
        def get_metrics_side_effect(provider_type):
            if provider_type == ProviderType.OPENAI:
                return unhealthy_metrics
            return ProviderMetrics(
                provider_type=provider_type,
                status=ProviderStatus.HEALTHY,
                uptime_percentage=99.0,
                avg_latency_ms=2000.0,
                error_rate=0.01,
                total_requests=1000,
                successful_requests=990,
                failed_requests=10,
            )
        
        health_monitor.get_metrics.side_effect = get_metrics_side_effect
        
        request = AIRequest(
            model="gemini-pro",
            messages=[{"role": "user", "content": "Test"}],
        )
        
        available = [p for p in mock_providers if "gemini-pro" in p.models]
        
        result = await router.select_optimal_provider(
            request=request,
            available_providers=available,
        )
        
        assert result is not None
        # Should not select unhealthy provider
        assert result.provider_type != ProviderType.OPENAI


class TestAdvancedCostOptimizer:
    """Test advanced cost optimizer functionality."""
    
    @pytest.fixture
    def config_manager(self):
        """Create a test config manager."""
        return ModelConfigManager()
    
    @pytest.fixture
    def optimizer(self, config_manager):
        """Create a test optimizer."""
        return AdvancedCostOptimizer(
            optimization_strategy=OptimizationStrategy.COST_QUALITY_BALANCE,
            config_manager=config_manager,
        )
    
    @pytest.mark.asyncio
    async def test_cost_tracking(self, optimizer):
        """Test cost tracking functionality."""
        request = AIRequest(
            request_id="test-123",
            model="claude-3-opus",
            messages=[{"role": "user", "content": "Test"}],
        )
        
        response = AIResponse(
            content="Response",
            model="claude-3-opus",
            provider=ProviderType.ANTHROPIC,
            usage={"input_tokens": 100, "output_tokens": 50},
            cost=0.015,  # Realistic cost
            latency_ms=3000.0,
            request_id="test-123",
        )
        
        await optimizer.track_request_cost(
            request=request,
            response=response,
            provider_type=ProviderType.ANTHROPIC,
            model="claude-3-opus",
        )
        
        # Check metrics updated
        assert optimizer.cost_metrics.hourly_cost >= 0.015
        assert len(optimizer.cost_history) > 0
        assert len(optimizer.request_costs) > 0
    
    @pytest.mark.asyncio
    async def test_budget_alerts(self, optimizer):
        """Test budget alert generation."""
        # Set a low budget
        optimizer.set_budget_limit("daily", 1.0)
        
        # Track costs that exceed thresholds
        for i in range(10):
            request = AIRequest(
                request_id=f"test-{i}",
                model="gpt-4",
                messages=[{"role": "user", "content": "Test"}],
            )
            
            response = AIResponse(
                content="Response",
                model="gpt-4",
                provider=ProviderType.OPENAI,
                usage={"input_tokens": 1000, "output_tokens": 500},
                cost=0.15,  # High cost
                latency_ms=5000.0,
                request_id=f"test-{i}",
            )
            
            await optimizer.track_request_cost(
                request=request,
                response=response,
                provider_type=ProviderType.OPENAI,
                model="gpt-4",
            )
        
        # Force metrics update
        await optimizer._update_cost_metrics()
        await optimizer._check_budget_alerts()
        
        # Check alerts generated
        assert len(optimizer.budget_alerts) > 0
        
        # Check alert severity
        high_severity_alerts = [
            a for a in optimizer.budget_alerts 
            if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        ]
        assert len(high_severity_alerts) > 0
    
    @pytest.mark.asyncio
    async def test_optimization_strategies(self, optimizer):
        """Test different optimization strategies."""
        request = AIRequest(
            model="claude-3-opus",
            messages=[
                {"role": "user", "content": "Complex analysis required here with detailed explanation"}
            ],
        )
        
        available_providers = {
            ProviderType.ANTHROPIC: ["claude-3-opus", "claude-3-sonnet"],
            ProviderType.OPENAI: ["gpt-4", "gpt-3.5-turbo"],
            ProviderType.GOOGLE: ["gemini-pro"],
        }
        
        performance_data = {
            "ANTHROPIC:claude-3-opus": {
                "quality_score": 0.95,
                "avg_latency_ms": 3000,
                "success_rate": 0.99,
            },
            "OPENAI:gpt-4": {
                "quality_score": 0.93,
                "avg_latency_ms": 5000,
                "success_rate": 0.98,
            },
        }
        
        # Test cost optimization
        optimizer.optimization_strategy = OptimizationStrategy.MINIMIZE_COST
        result = await optimizer.optimize_request_routing(
            request=request,
            available_providers=available_providers,
            performance_data=performance_data,
        )
        
        assert result is not None
        assert "estimated_cost" in result
        assert result["estimated_cost"] >= 0
        
        # Test budget-aware optimization
        optimizer.optimization_strategy = OptimizationStrategy.BUDGET_AWARE
        optimizer.set_budget_limit("daily", 10.0)
        result = await optimizer.optimize_request_routing(
            request=request,
            available_providers=available_providers,
            performance_data=performance_data,
        )
        
        assert result is not None
        assert "budget" in result["optimization_reason"].lower()
    
    def test_token_estimation(self, optimizer):
        """Test token estimation accuracy."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a Python function to calculate fibonacci."},
        ]
        
        # Test using centralized utility
        tokens = estimate_input_tokens(messages)
        
        # Realistic expectation: ~20-30 tokens for this input
        assert 10 < tokens < 50
    
    def test_cost_calculation(self, optimizer):
        """Test cost calculation accuracy."""
        # Test with known model costs
        cost = estimate_request_cost(
            provider_type=ProviderType.ANTHROPIC,
            model_id="claude-3-opus",
            input_tokens=1000,
            output_tokens=500,
            config_manager=optimizer.config_manager,
        )
        
        # Expected: (1000/1000 * 0.015) + (500/1000 * 0.075) = 0.015 + 0.0375 = 0.0525
        assert 0.05 < cost < 0.06


class TestFallbackManager:
    """Test fallback manager functionality."""
    
    @pytest.fixture
    def config_manager(self):
        """Create a test config manager."""
        return ModelConfigManager()
    
    @pytest.fixture
    def health_monitor(self):
        """Create a test health monitor."""
        return MagicMock(spec=HealthMonitor)
    
    @pytest.fixture
    def router(self, health_monitor, config_manager):
        """Create a test router."""
        return IntelligentRouter(
            health_monitor=health_monitor,
            config_manager=config_manager,
        )
    
    @pytest.fixture
    def fallback_manager(self, router, health_monitor, config_manager):
        """Create a test fallback manager."""
        return FallbackManager(
            intelligent_router=router,
            health_monitor=health_monitor,
            config_manager=config_manager,
        )
    
    @pytest.mark.asyncio
    async def test_fallback_execution(self, fallback_manager):
        """Test fallback execution with provider failures."""
        # Create providers with different failure modes
        providers = [
            MockProvider(
                provider_type=ProviderType.ANTHROPIC,
                models={"claude-3-opus": ModelSpec(
                    model_id="claude-3-opus",
                    context_length=200000,
                    max_output_tokens=4096,
                    cost_tier=CostTier.PREMIUM,
                )},
                should_fail=True,  # Primary will fail
            ),
            MockProvider(
                provider_type=ProviderType.OPENAI,
                models={"claude-3-opus": ModelSpec(
                    model_id="claude-3-opus",
                    context_length=128000,
                    max_output_tokens=4096,
                    cost_tier=CostTier.HIGH,
                )},
                should_fail=False,  # Secondary will succeed
            ),
        ]
        
        request = AIRequest(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "Test"}],
        )
        
        response, attempts = await fallback_manager.execute_with_fallback(
            request=request,
            available_providers=providers,
            max_fallback_attempts=3,
        )
        
        assert response is not None
        assert len(attempts) >= 2  # At least primary failure and secondary success
        assert attempts[-1].success is True
        assert attempts[-1].fallback_provider == ProviderType.OPENAI
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, fallback_manager):
        """Test circuit breaker functionality."""
        provider_type = ProviderType.ANTHROPIC
        
        # Configure circuit breaker
        breaker_config = CircuitBreakerConfig(
            provider_type=provider_type,
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=1,
        )
        fallback_manager.circuit_breakers[provider_type] = breaker_config
        
        # Record failures to trip the breaker
        for i in range(3):
            error = Exception("Connection timeout")
            await fallback_manager._record_circuit_breaker_failure(provider_type, error)
        
        # Check breaker is open
        assert breaker_config.state == CircuitState.OPEN
        assert not fallback_manager._is_circuit_closed(provider_type)
        
        # Wait for timeout
        await asyncio.sleep(1.1)
        
        # Check breaker moves to half-open
        assert fallback_manager._is_circuit_closed(provider_type)
        assert breaker_config.state == CircuitState.HALF_OPEN
        
        # Record successes to close breaker
        for i in range(2):
            await fallback_manager._record_circuit_breaker_success(provider_type)
        
        # Check breaker is closed
        assert breaker_config.state == CircuitState.CLOSED
    
    def test_error_classification(self, fallback_manager):
        """Test error classification system."""
        # Test rate limit error
        error = Exception("Rate limit exceeded")
        error.status_code = 429
        classification = classify_error(error)
        assert classification == ErrorClassification.TOO_MANY_REQUESTS
        
        # Test timeout error
        error = TimeoutError("Request timed out")
        classification = classify_error(error)
        assert classification == ErrorClassification.TIMEOUT
        
        # Test authentication error
        error = Exception("Invalid API key")
        error.status_code = 401
        classification = classify_error(error)
        assert classification == ErrorClassification.AUTHENTICATION_ERROR
        
        # Test service error
        error = Exception("Internal server error")
        error.status_code = 500
        classification = classify_error(error)
        assert classification == ErrorClassification.INTERNAL_SERVER_ERROR
    
    @pytest.mark.asyncio
    async def test_tiered_fallback(self, fallback_manager):
        """Test multi-tier fallback strategy."""
        # Create providers for different tiers
        providers = [
            MockProvider(
                provider_type=ProviderType.ANTHROPIC,
                models={"test-model": ModelSpec(
                    model_id="test-model",
                    context_length=100000,
                    max_output_tokens=4096,
                    cost_tier=CostTier.PREMIUM,
                )},
                should_fail=True,  # Primary tier fails
            ),
            MockProvider(
                provider_type=ProviderType.GOOGLE,
                models={"test-model": ModelSpec(
                    model_id="test-model",
                    context_length=32768,
                    max_output_tokens=2048,
                    cost_tier=CostTier.LOW,
                )},
                should_fail=True,  # Secondary tier fails
            ),
            MockProvider(
                provider_type=ProviderType.OPENAI,
                models={"test-model": ModelSpec(
                    model_id="test-model",
                    context_length=128000,
                    max_output_tokens=4096,
                    cost_tier=CostTier.MEDIUM,
                )},
                should_fail=False,  # Emergency tier succeeds
            ),
        ]
        
        request = AIRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Test"}],
        )
        
        response, attempts = await fallback_manager.execute_with_fallback(
            request=request,
            available_providers=providers,
            max_fallback_attempts=5,
        )
        
        assert response is not None
        assert len(attempts) >= 3  # All tiers attempted
        
        # Check tier progression
        tiers_used = [attempt.fallback_tier for attempt in attempts]
        assert FallbackTier.PRIMARY in tiers_used or FallbackTier.SECONDARY in tiers_used
    
    def test_fallback_statistics(self, fallback_manager):
        """Test fallback statistics generation."""
        # Add some test data
        from src.ai_providers.fallback_manager import FallbackAttempt
        
        fallback_manager.fallback_history = [
            FallbackAttempt(
                request_id="test-1",
                original_provider=ProviderType.ANTHROPIC,
                fallback_provider=ProviderType.OPENAI,
                fallback_tier=FallbackTier.SECONDARY,
                attempt_number=1,
                trigger_reason="Rate limit",
                success=True,
                latency_ms=2000.0,
            ),
            FallbackAttempt(
                request_id="test-2",
                original_provider=ProviderType.ANTHROPIC,
                fallback_provider=ProviderType.GOOGLE,
                fallback_tier=FallbackTier.EMERGENCY,
                attempt_number=2,
                trigger_reason="Service error",
                success=False,
                error_message="Connection failed",
            ),
        ]
        
        stats = fallback_manager.get_fallback_statistics()
        
        assert stats["total_fallback_attempts"] == 2
        assert stats["successful_fallbacks"] == 1
        assert stats["overall_success_rate"] == 0.5
        assert "tier_statistics" in stats
        assert "circuit_breaker_states" in stats


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def complete_system(self):
        """Create a complete system for integration testing."""
        config_manager = ModelConfigManager()
        health_monitor = HealthMonitor()
        
        router = IntelligentRouter(
            health_monitor=health_monitor,
            config_manager=config_manager,
        )
        
        optimizer = AdvancedCostOptimizer(
            config_manager=config_manager,
        )
        
        fallback_manager = FallbackManager(
            intelligent_router=router,
            health_monitor=health_monitor,
            config_manager=config_manager,
        )
        
        return {
            "config_manager": config_manager,
            "health_monitor": health_monitor,
            "router": router,
            "optimizer": optimizer,
            "fallback_manager": fallback_manager,
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_request_flow(self, complete_system):
        """Test complete request flow through the system."""
        router = complete_system["router"]
        optimizer = complete_system["optimizer"]
        fallback_manager = complete_system["fallback_manager"]
        
        # Create realistic providers
        providers = [
            MockProvider(
                provider_type=ProviderType.ANTHROPIC,
                models={"claude-3-haiku": ModelSpec(
                    model_id="claude-3-haiku",
                    context_length=200000,
                    max_output_tokens=4096,
                    cost_tier=CostTier.LOW,
                    cost_per_input_token=0.00025,
                    cost_per_output_token=0.00125,
                )},
                latency_ms=1500.0,
            ),
            MockProvider(
                provider_type=ProviderType.OPENAI,
                models={"gpt-3.5-turbo": ModelSpec(
                    model_id="gpt-3.5-turbo",
                    context_length=16385,
                    max_output_tokens=4096,
                    cost_tier=CostTier.LOW,
                    cost_per_input_token=0.0015,
                    cost_per_output_token=0.002,
                )},
                latency_ms=2000.0,
            ),
        ]
        
        # Create a request
        request = AIRequest(
            model="claude-3-haiku",
            messages=[
                {"role": "user", "content": "Explain quantum computing in simple terms"}
            ],
            max_tokens=200,
        )
        
        # Get optimization recommendation
        available = {
            p.provider_type: list(p.models.keys())
            for p in providers
        }
        performance_data = {
            f"{p.provider_type.value}:{model}": {
                "quality_score": 0.8,
                "avg_latency_ms": p.latency_ms,
                "success_rate": 0.95,
            }
            for p in providers
            for model in p.models.keys()
        }
        
        optimization = await optimizer.optimize_request_routing(
            request=request,
            available_providers=available,
            performance_data=performance_data,
        )
        
        assert optimization is not None
        assert optimization["estimated_cost"] > 0
        assert optimization["recommended_provider"] in [p.provider_type for p in providers]
        
        # Execute with fallback
        response, attempts = await fallback_manager.execute_with_fallback(
            request=request,
            available_providers=providers,
            max_fallback_attempts=3,
        )
        
        assert response is not None
        assert response.content == "Mock response"
        assert len(attempts) > 0
        
        # Track the cost
        await optimizer.track_request_cost(
            request=request,
            response=response,
            provider_type=response.provider,
            model=response.model,
        )
        
        # Verify cost tracking
        analysis = optimizer.get_cost_analysis()
        assert analysis["current_metrics"]["hourly_cost"] >= 0
        assert response.provider.value in str(analysis["provider_breakdown"])
    
    @pytest.mark.asyncio
    async def test_configuration_management(self, complete_system):
        """Test configuration management and updates."""
        config_manager = complete_system["config_manager"]
        
        # Test model profile retrieval
        profile = config_manager.get_model_profile("claude-3-opus")
        assert profile is not None
        assert profile.model_id == "claude-3-opus"
        assert profile.cost_config.input_cost_per_1k_tokens > 0
        
        # Test routing rules
        rules = config_manager.get_routing_rules(enabled_only=True)
        assert len(rules) > 0
        
        # Test fallback tier configuration
        tier = config_manager.get_fallback_tier("primary")
        assert tier is not None
        assert len(tier.providers) > 0
        
        # Test normalization
        norm_config = config_manager.normalization_config
        
        # Test cost normalization
        low_cost_score = norm_config.normalize_cost(0.001)
        high_cost_score = norm_config.normalize_cost(0.5)
        assert low_cost_score > high_cost_score  # Lower cost should have higher score
        
        # Test latency normalization
        low_latency_score = norm_config.normalize_latency(500)
        high_latency_score = norm_config.normalize_latency(10000)
        assert low_latency_score > high_latency_score  # Lower latency should have higher score


# Performance and memory tests with realistic targets
class TestPerformance:
    """Test performance characteristics with realistic targets."""
    
    @pytest.mark.asyncio
    async def test_router_performance(self):
        """Test router selection performance."""
        import time
        
        config_manager = ModelConfigManager()
        health_monitor = MagicMock(spec=HealthMonitor)
        health_monitor.get_metrics.return_value = ProviderMetrics(
            provider_type=ProviderType.ANTHROPIC,
            status=ProviderStatus.HEALTHY,
            uptime_percentage=99.0,
            avg_latency_ms=2000.0,
            error_rate=0.01,
            total_requests=1000,
            successful_requests=990,
            failed_requests=10,
        )
        
        router = IntelligentRouter(
            health_monitor=health_monitor,
            config_manager=config_manager,
        )
        
        providers = [
            MockProvider(
                provider_type=ProviderType.ANTHROPIC,
                models={"test-model": ModelSpec(
                    model_id="test-model",
                    context_length=100000,
                    max_output_tokens=4096,
                    cost_tier=CostTier.MEDIUM,
                )},
            )
            for _ in range(10)  # 10 providers
        ]
        
        request = AIRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Test"}],
        )
        
        # Measure selection time
        start = time.time()
        for _ in range(100):  # 100 selections
            await router.select_optimal_provider(
                request=request,
                available_providers=providers,
            )
        elapsed = time.time() - start
        
        # Realistic target: <10ms per selection (1 second for 100 selections)
        assert elapsed < 1.0, f"Selection too slow: {elapsed:.3f}s for 100 selections"
        
        # Average time per selection
        avg_time = elapsed / 100
        assert avg_time < 0.01, f"Average selection time too high: {avg_time*1000:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage with realistic limits."""
        import sys
        
        config_manager = ModelConfigManager()
        
        # Check config manager memory footprint
        config_size = sys.getsizeof(config_manager.model_profiles) + \
                     sys.getsizeof(config_manager.routing_rules) + \
                     sys.getsizeof(config_manager.fallback_tiers)
        
        # Realistic limit: Config should be under 10MB
        assert config_size < 10 * 1024 * 1024, f"Config too large: {config_size / (1024*1024):.2f}MB"
        
        # Test cost optimizer memory with history
        optimizer = AdvancedCostOptimizer(config_manager=config_manager)
        
        # Add 1000 cost history entries
        for i in range(1000):
            optimizer.cost_history.append((datetime.now(), 0.01))
        
        optimizer_size = sys.getsizeof(optimizer.cost_history) + \
                        sys.getsizeof(optimizer.request_costs) + \
                        sys.getsizeof(optimizer.provider_performance)
        
        # Realistic limit: Optimizer should be under 50MB with history
        assert optimizer_size < 50 * 1024 * 1024, f"Optimizer too large: {optimizer_size / (1024*1024):.2f}MB"