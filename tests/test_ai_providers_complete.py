"""Comprehensive tests for AI provider integration."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.ai_providers.abstract_provider import AbstractProvider
from src.ai_providers.anthropic_provider import AnthropicProvider
from src.ai_providers.config import AIProviderConfigManager, AIProviderSettings
from src.ai_providers.cost_optimizer import CostOptimizer, UsageTracker
from src.ai_providers.error_handler import (
    AIProviderError,
    BudgetExceededError,
    CircuitBreaker,
    ErrorHandler,
    RateLimitError,
    RetryStrategy,
)
from src.ai_providers.models import (
    AIRequest,
    AIResponse,
    CostBudget,
    CostTier,
    MCPTool,
    ModelSpec,
    ProviderCapability,
    ProviderConfig,
    ProviderHealth,
    ProviderSelection,
    ProviderStatus,
    ProviderType,
    StreamingChunk,
    UsageRecord,
)
from src.ai_providers.openai_provider import OpenAIProvider
from src.ai_providers.provider_manager import AIProviderManager
from src.ai_providers.provider_registry import ProviderRegistry
from src.ai_providers.streaming_manager import StreamingManager, StreamingResponse
from src.ai_providers.tool_translator import ToolTranslator


class TestAIProviderModels:
    """Test data models."""
    
    def test_provider_config_creation(self):
        """Test creating provider configuration."""
        config = ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            api_key="test-key",
            enabled=True,
            priority=10,
            rate_limit_rpm=60,
            budget_limit=100.0,
        )
        
        assert config.provider_type == ProviderType.ANTHROPIC
        assert config.api_key == "test-key"
        assert config.enabled is True
        assert config.priority == 10
        assert config.budget_limit == 100.0
    
    def test_ai_request_creation(self):
        """Test creating AI request."""
        request = AIRequest(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=100,
            temperature=0.7,
            stream=False,
        )
        
        assert request.model == "test-model"
        assert len(request.messages) == 1
        assert request.max_tokens == 100
        assert request.stream is False
        assert request.request_id  # Should have auto-generated ID
    
    def test_cost_budget_creation(self):
        """Test creating cost budget."""
        budget = CostBudget(
            name="Test Budget",
            daily_limit=50.0,
            monthly_limit=1500.0,
            alert_thresholds=[0.5, 0.8, 0.95],
        )
        
        assert budget.name == "Test Budget"
        assert budget.daily_limit == 50.0
        assert budget.monthly_limit == 1500.0
        assert len(budget.alert_thresholds) == 3


class TestErrorHandling:
    """Test error handling and retry logic."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        
        # Initially closed
        assert breaker.is_closed()
        assert not breaker.is_open()
        
        # Record failures
        breaker.call_failed()
        breaker.call_failed()
        assert breaker.is_closed()  # Still closed
        
        breaker.call_failed()
        assert breaker.is_open()  # Now open after 3 failures
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        assert breaker.state == "half_open"
        
        # Successful call should close it
        breaker.call_succeeded()
        breaker.call_succeeded()
        breaker.call_succeeded()
        assert breaker.is_closed()
    
    @pytest.mark.asyncio
    async def test_retry_strategy_exponential(self):
        """Test exponential backoff retry strategy."""
        handler = ErrorHandler()
        
        call_count = 0
        
        async def failing_func(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limited", retry_after=0.1)
            return "success"
        
        result = await handler.retry_with_strategy(
            failing_func,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        )
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_strategy_none(self):
        """Test no retry strategy."""
        handler = ErrorHandler()
        
        async def failing_func(**kwargs):
            raise AIProviderError("Test error")
        
        with pytest.raises(AIProviderError):
            await handler.retry_with_strategy(
                failing_func,
                strategy=RetryStrategy.NONE,
            )
    
    def test_error_classification(self):
        """Test error classification."""
        handler = ErrorHandler()
        
        # Rate limit error
        error = Exception("Rate limit exceeded")
        classified = handler._classify_error(error)
        assert isinstance(classified, RateLimitError)
        assert classified.retryable
        
        # Authentication error
        error = Exception("Invalid API key")
        classified = handler._classify_error(error)
        assert classified.retryable is False


class TestCostOptimization:
    """Test cost optimization functionality."""
    
    @pytest.mark.asyncio
    async def test_usage_tracking(self):
        """Test usage tracking."""
        tracker = UsageTracker()
        
        request = AIRequest(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
        )
        
        response = AIResponse(
            request_id=request.request_id,
            provider_type=ProviderType.ANTHROPIC,
            model="test-model",
            content="response",
            usage={"input_tokens": 10, "output_tokens": 20},
            cost=0.001,
            latency_ms=500,
        )
        
        await tracker.record_usage(
            request, response, ProviderType.ANTHROPIC, success=True
        )
        
        # Check daily usage
        daily = tracker.get_daily_usage()
        assert daily == 0.001
        
        # Check provider usage
        provider_usage = tracker.get_provider_usage(ProviderType.ANTHROPIC)
        assert provider_usage == 0.001
        
        # Check stats
        stats = tracker.get_usage_stats()
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 1
        assert stats["total_cost"] == 0.001
    
    @pytest.mark.asyncio
    async def test_budget_enforcement(self):
        """Test budget enforcement."""
        tracker = UsageTracker()
        optimizer = CostOptimizer(tracker)
        
        # Add budget
        budget = CostBudget(
            name="Test Budget",
            daily_limit=1.0,
            monthly_limit=30.0,
        )
        optimizer.add_budget(budget)
        
        # Check budget within limits
        request = AIRequest(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
        )
        
        allowed, violations = await optimizer.check_budget_limits(
            request, 0.5, ProviderType.ANTHROPIC
        )
        assert allowed is True
        assert len(violations) == 0
        
        # Simulate exceeding daily limit
        for _ in range(10):
            await tracker.record_usage(
                request,
                AIResponse(
                    request_id=request.request_id,
                    provider_type=ProviderType.ANTHROPIC,
                    model="test",
                    content="test",
                    cost=0.15,
                ),
                ProviderType.ANTHROPIC,
            )
        
        allowed, violations = await optimizer.check_budget_limits(
            request, 0.5, ProviderType.ANTHROPIC
        )
        assert allowed is False
        assert len(violations) > 0
    
    def test_cheapest_provider_selection(self):
        """Test finding cheapest provider."""
        tracker = UsageTracker()
        optimizer = CostOptimizer(tracker)
        
        # Register provider models
        optimizer.register_provider_models(
            ProviderType.ANTHROPIC,
            {
                "claude-cheap": ModelSpec(
                    model_id="claude-cheap",
                    provider_type=ProviderType.ANTHROPIC,
                    display_name="Cheap Claude",
                    cost_per_input_token=0.001,
                    cost_per_output_token=0.002,
                    cost_tier=CostTier.LOW,
                    is_available=True,
                ),
            },
        )
        
        optimizer.register_provider_models(
            ProviderType.OPENAI,
            {
                "gpt-expensive": ModelSpec(
                    model_id="gpt-expensive",
                    provider_type=ProviderType.OPENAI,
                    display_name="Expensive GPT",
                    cost_per_input_token=0.01,
                    cost_per_output_token=0.02,
                    cost_tier=CostTier.HIGH,
                    is_available=True,
                ),
            },
        )
        
        request = AIRequest(
            model="any",
            messages=[{"role": "user", "content": "test"}],
        )
        
        result = optimizer.find_cheapest_provider(
            request,
            [ProviderType.ANTHROPIC, ProviderType.OPENAI],
        )
        
        assert result is not None
        provider, model, cost = result
        assert provider == ProviderType.ANTHROPIC
        assert model == "claude-cheap"


class TestProviderRegistry:
    """Test provider registry functionality."""
    
    @pytest.mark.asyncio
    async def test_provider_registration(self):
        """Test registering providers."""
        registry = ProviderRegistry()
        
        # Create mock provider
        provider = MagicMock(spec=AbstractProvider)
        provider.provider_type = ProviderType.ANTHROPIC
        provider.is_available = True
        provider._initialized = False
        provider.initialize = AsyncMock()
        
        await registry.register_provider(provider, priority=10)
        
        assert ProviderType.ANTHROPIC in registry._providers
        assert registry._provider_priorities[ProviderType.ANTHROPIC] == 10
        provider.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_provider_selection_strategies(self):
        """Test different provider selection strategies."""
        registry = ProviderRegistry()
        
        # Create mock providers
        providers = []
        for i, ptype in enumerate(ProviderType):
            provider = MagicMock(spec=AbstractProvider)
            provider.provider_type = ptype
            provider.is_available = True
            provider.models = {"test-model": MagicMock()}
            provider.supports_streaming = MagicMock(return_value=True)
            provider.supports_tools = MagicMock(return_value=True)
            provider.get_model_cost = MagicMock(return_value=0.001 * (i + 1))
            providers.append(provider)
            registry._providers[ptype] = provider
            registry._provider_priorities[ptype] = 10 - i
        
        request = AIRequest(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
        )
        
        # Test priority strategy
        selected = registry._priority_strategy(providers, request)
        assert selected.provider_type == ProviderType.ANTHROPIC
        
        # Test round-robin strategy
        selected1 = registry._round_robin_strategy(providers, request)
        selected2 = registry._round_robin_strategy(providers, request)
        assert selected1 != selected2 or len(providers) == 1
        
        # Test cost strategy
        selected = registry._cost_optimized_strategy(providers, request)
        assert selected is not None
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self):
        """Test health monitoring."""
        registry = ProviderRegistry()
        
        # Create mock provider
        provider = MagicMock(spec=AbstractProvider)
        provider.provider_type = ProviderType.ANTHROPIC
        provider.health_check = AsyncMock(
            return_value=ProviderHealth(
                provider_type=ProviderType.ANTHROPIC,
                status=ProviderStatus.AVAILABLE,
                uptime_percentage=99.5,
            )
        )
        
        registry._providers[ProviderType.ANTHROPIC] = provider
        
        # Perform health check
        results = await registry.perform_health_check()
        
        assert ProviderType.ANTHROPIC in results
        assert results[ProviderType.ANTHROPIC].status == ProviderStatus.AVAILABLE
        assert results[ProviderType.ANTHROPIC].uptime_percentage == 99.5


class TestStreamingManager:
    """Test streaming response management."""
    
    @pytest.mark.asyncio
    async def test_streaming_session_creation(self):
        """Test creating streaming sessions."""
        manager = StreamingManager()
        
        session = await manager.create_session(
            "test-request",
            ProviderType.ANTHROPIC,
        )
        
        assert session.request_id == "test-request"
        assert session.provider_type == ProviderType.ANTHROPIC
        assert session.is_active
        assert session.session_id in manager._sessions
    
    @pytest.mark.asyncio
    async def test_streaming_response_aggregation(self):
        """Test streaming response aggregation."""
        
        async def chunk_generator():
            yield StreamingChunk(request_id="test", content="Hello ")
            yield StreamingChunk(request_id="test", content="world!")
            yield StreamingChunk(
                request_id="test",
                is_complete=True,
                finish_reason="stop",
            )
        
        response = StreamingResponse(
            "test",
            ProviderType.ANTHROPIC,
            chunk_generator(),
        )
        
        # Stream and aggregate
        chunks = []
        async for chunk in response.stream():
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert response.aggregated_content == "Hello world!"
        assert response.is_complete
        assert response.finish_reason == "stop"
        
        # Convert to AIResponse
        ai_response = response.to_response()
        assert ai_response.content == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_streaming_error_handling(self):
        """Test streaming error handling."""
        
        async def error_generator():
            yield StreamingChunk(request_id="test", content="Start")
            raise Exception("Stream error")
        
        response = StreamingResponse(
            "test",
            ProviderType.ANTHROPIC,
            error_generator(),
        )
        
        chunks = []
        async for chunk in response.stream():
            chunks.append(chunk)
        
        # Should get error chunk
        assert len(chunks) > 1
        assert "[Error:" in chunks[-1].content


class TestToolTranslation:
    """Test tool format translation."""
    
    def test_mcp_to_anthropic_translation(self):
        """Test MCP to Anthropic tool translation."""
        translator = ToolTranslator()
        
        mcp_tools = [
            MCPTool(
                name="test_tool",
                description="Test tool",
                inputSchema={
                    "type": "object",
                    "properties": {"param": {"type": "string"}},
                },
            )
        ]
        
        anthropic_tools = translator.mcp_to_provider(
            mcp_tools, ProviderType.ANTHROPIC
        )
        
        assert len(anthropic_tools) == 1
        assert anthropic_tools[0]["name"] == "test_tool"
        assert anthropic_tools[0]["description"] == "Test tool"
        assert "input_schema" in anthropic_tools[0]
    
    def test_mcp_to_openai_translation(self):
        """Test MCP to OpenAI tool translation."""
        translator = ToolTranslator()
        
        mcp_tools = [
            MCPTool(
                name="test_tool",
                description="Test tool",
                inputSchema={
                    "type": "object",
                    "properties": {"param": {"type": "string"}},
                },
            )
        ]
        
        openai_tools = translator.mcp_to_provider(
            mcp_tools, ProviderType.OPENAI
        )
        
        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "test_tool"
        assert "parameters" in openai_tools[0]["function"]
    
    def test_mcp_to_google_translation(self):
        """Test MCP to Google tool translation."""
        translator = ToolTranslator()
        
        mcp_tools = [
            MCPTool(
                name="test_tool",
                description="Test tool",
                inputSchema={
                    "type": "object",
                    "properties": {"param": {"type": "string"}},
                },
            )
        ]
        
        google_tools = translator.mcp_to_provider(
            mcp_tools, ProviderType.GOOGLE
        )
        
        assert len(google_tools) == 1
        assert "functionDeclarations" in google_tools[0]
        assert len(google_tools[0]["functionDeclarations"]) == 1


class TestProviderManager:
    """Test AI provider manager integration."""
    
    @pytest.mark.asyncio
    async def test_provider_manager_initialization(self):
        """Test initializing provider manager."""
        manager = AIProviderManager()
        
        # Create test config
        config = ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            api_key="test-key",
            enabled=True,
        )
        
        with patch.object(manager, "_initialize_provider", new_callable=AsyncMock):
            await manager.initialize([config], [])
            
            assert manager._initialized
            manager._initialize_provider.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_request_processing_with_budget_check(self):
        """Test processing request with budget checks."""
        manager = AIProviderManager()
        
        # Mock components
        manager._initialized = True
        manager.registry = MagicMock()
        manager.cost_optimizer = MagicMock()
        manager.error_handler = MagicMock()
        manager.usage_tracker = MagicMock()
        
        # Mock provider
        provider = MagicMock()
        provider.provider_type = ProviderType.ANTHROPIC
        provider.generate_response = AsyncMock(
            return_value=AIResponse(
                request_id="test",
                provider_type=ProviderType.ANTHROPIC,
                model="test",
                content="response",
                cost=0.001,
            )
        )
        
        manager.registry.select_provider = AsyncMock(return_value=provider)
        manager.cost_optimizer.estimate_request_cost = MagicMock(return_value=0.001)
        manager.cost_optimizer.check_budget_limits = AsyncMock(
            return_value=(True, [])
        )
        manager.error_handler.retry_with_strategy = AsyncMock(
            return_value=provider.generate_response.return_value
        )
        manager.usage_tracker.record_usage = AsyncMock()
        
        request = AIRequest(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
        )
        
        response = await manager.process_request(request)
        
        assert response.content == "response"
        assert response.cost == 0.001
        
        # Verify budget was checked
        manager.cost_optimizer.check_budget_limits.assert_called_once()
        manager.usage_tracker.record_usage.assert_called_once()


class TestConfiguration:
    """Test configuration management."""
    
    def test_config_loading_from_environment(self):
        """Test loading configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test-anthropic-key",
                "OPENAI_API_KEY": "test-openai-key",
                "AI_DAILY_BUDGET": "100.0",
                "AI_SELECTION_STRATEGY": "cost",
            },
        ):
            settings = AIProviderSettings()
            
            assert settings.anthropic_api_key.get_secret_value() == "test-anthropic-key"
            assert settings.openai_api_key.get_secret_value() == "test-openai-key"
            assert settings.daily_budget_limit == 100.0
            assert settings.default_selection_strategy == "cost"
    
    def test_config_validation(self):
        """Test configuration validation."""
        manager = AIProviderConfigManager()
        
        # Mock configs
        manager._provider_configs = {
            ProviderType.ANTHROPIC: ProviderConfig(
                provider_type=ProviderType.ANTHROPIC,
                api_key="test-key",
                enabled=True,
                rate_limit_rpm=60,
                timeout=30.0,
            )
        }
        
        validation = manager.validate_configuration()
        
        assert validation["valid"]
        assert validation["providers_configured"] == 1
        assert len(validation["issues"]) == 0
    
    def test_config_summary(self):
        """Test getting configuration summary."""
        manager = AIProviderConfigManager()
        
        # Mock configs
        manager._provider_configs = {
            ProviderType.ANTHROPIC: ProviderConfig(
                provider_type=ProviderType.ANTHROPIC,
                api_key="test-key",
                enabled=True,
                priority=10,
                rate_limit_rpm=60,
            )
        }
        
        summary = manager.get_config_summary()
        
        assert "providers" in summary
        assert "anthropic" in summary["providers"]
        assert summary["providers"]["anthropic"]["enabled"]
        assert summary["providers"]["anthropic"]["priority"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])