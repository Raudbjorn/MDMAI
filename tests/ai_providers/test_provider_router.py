"""
Tests for provider router functionality.

Tests cover provider registration, routing, fallback behavior, and circuit breaker patterns.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from src.ai_providers.base_provider import (
    ProviderType, ProviderConfig, BaseAIProvider, CompletionResponse,
    ProviderError, ProviderAuthenticationError, ProviderRateLimitError,
    ProviderTimeoutError, NoAvailableProvidersError
)
from src.ai_providers.provider_router import ProviderRouter, CircuitState


class MockProvider(BaseAIProvider):
    """Mock provider for testing."""
    
    def __init__(self, config: ProviderConfig, should_fail: bool = False, delay: float = 0.0):
        super().__init__(config)
        self.should_fail = should_fail
        self.delay = delay
        self.call_count = 0
        self._authenticated = True
    
    async def complete(self, messages, tools=None, stream=False):
        """Mock completion method."""
        self.call_count += 1
        
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        if self.should_fail:
            raise ProviderError("Mock provider failure")
        
        if stream:
            async def mock_stream():
                yield "Mock "
                yield "response "
                yield "content"
            return mock_stream()
        else:
            yield "Mock response content"
    
    async def validate_credentials(self):
        """Mock credential validation."""
        return not self.should_fail
    
    def estimate_cost(self, input_tokens, output_tokens):
        """Mock cost estimation."""
        return 0.01
    
    async def health_check(self):
        """Mock health check."""
        return not self.should_fail


class TestProviderRouter:
    """Test cases for ProviderRouter."""
    
    @pytest.fixture
    def router(self):
        """Create ProviderRouter instance for testing."""
        return ProviderRouter()
    
    @pytest.fixture
    def mock_config(self):
        """Create mock provider configuration."""
        return ProviderConfig(
            type=ProviderType.ANTHROPIC,
            api_key="test-key",
            model="test-model"
        )
    
    def test_router_initialization(self, router):
        """Test router initialization."""
        assert len(router.providers) == 0
        assert len(router.user_providers) == 0
        assert len(router.provider_stats) == 0
        assert router.default_fallback_order == [
            ProviderType.ANTHROPIC,
            ProviderType.OPENAI,
            ProviderType.GOOGLE
        ]
    
    def test_register_provider(self, router, mock_config):
        """Test provider registration."""
        user_id = "test_user"
        
        # Mock provider factory
        with patch.object(router.provider_factories, 'get') as mock_factory:
            mock_provider = MockProvider(mock_config)
            mock_factory.return_value = MockProvider
            
            with patch.object(MockProvider, '__init__', return_value=None) as mock_init:
                with patch.object(MockProvider, 'config', mock_config):
                    provider_key = router.register_provider(
                        user_id, ProviderType.ANTHROPIC, mock_config, is_primary=True
                    )
        
        expected_key = f"{user_id}:{ProviderType.ANTHROPIC.value}"
        assert provider_key == expected_key
        assert expected_key in router.providers
        assert expected_key in router.provider_stats
        assert user_id in router.user_providers
        assert ProviderType.ANTHROPIC.value in router.user_providers[user_id]
    
    def test_register_unsupported_provider(self, router, mock_config):
        """Test registering unsupported provider type."""
        user_id = "test_user"
        
        # Remove provider from factory to simulate unsupported
        original_factories = router.provider_factories.copy()
        router.provider_factories.clear()
        
        with pytest.raises(ValueError, match="Unsupported provider type"):
            router.register_provider(user_id, ProviderType.ANTHROPIC, mock_config)
        
        # Restore factories
        router.provider_factories = original_factories
    
    def test_unregister_provider(self, router, mock_config):
        """Test provider unregistration."""
        user_id = "test_user"
        provider_key = f"{user_id}:{ProviderType.ANTHROPIC.value}"
        
        # Register first
        router.providers[provider_key] = MockProvider(mock_config)
        router.provider_stats[provider_key] = Mock()
        router.user_providers[user_id] = {ProviderType.ANTHROPIC.value: provider_key}
        
        # Unregister
        result = router.unregister_provider(user_id, ProviderType.ANTHROPIC)
        
        assert result is True
        assert provider_key not in router.providers
        assert provider_key not in router.provider_stats
        assert ProviderType.ANTHROPIC.value not in router.user_providers[user_id]
    
    def test_unregister_nonexistent_provider(self, router):
        """Test unregistering non-existent provider."""
        result = router.unregister_provider("nonexistent_user", ProviderType.ANTHROPIC)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_completion_success(self, router, mock_config):
        """Test successful completion request."""
        user_id = "test_user"
        provider_key = f"{user_id}:{ProviderType.ANTHROPIC.value}"
        
        # Setup mock provider
        mock_provider = MockProvider(mock_config)
        router.providers[provider_key] = mock_provider
        router.user_providers[user_id] = {ProviderType.ANTHROPIC.value: provider_key}
        
        messages = [{"role": "user", "content": "Hello"}]
        
        result = await router.get_completion(
            user_id, messages, preferred_provider=ProviderType.ANTHROPIC
        )
        
        assert isinstance(result, CompletionResponse)
        assert result.content == "Mock response content"
        assert result.provider == ProviderType.ANTHROPIC
        assert result.is_fallback is False
    
    @pytest.mark.asyncio
    async def test_get_completion_streaming(self, router, mock_config):
        """Test streaming completion request."""
        user_id = "test_user"
        provider_key = f"{user_id}:{ProviderType.ANTHROPIC.value}"
        
        # Setup mock provider
        mock_provider = MockProvider(mock_config)
        router.providers[provider_key] = mock_provider
        router.user_providers[user_id] = {ProviderType.ANTHROPIC.value: provider_key}
        
        messages = [{"role": "user", "content": "Hello"}]
        
        result = await router.get_completion(
            user_id, messages, preferred_provider=ProviderType.ANTHROPIC, stream=True
        )
        
        # Collect streamed content
        content_chunks = []
        async for chunk in result:
            content_chunks.append(chunk)
        
        assert content_chunks == ["Mock ", "response ", "content"]
    
    @pytest.mark.asyncio
    async def test_get_completion_fallback(self, router):
        """Test fallback to secondary provider."""
        user_id = "test_user"
        
        # Setup failing primary provider
        primary_config = ProviderConfig(type=ProviderType.ANTHROPIC, api_key="test-key")
        primary_provider = MockProvider(primary_config, should_fail=True)
        primary_key = f"{user_id}:{ProviderType.ANTHROPIC.value}"
        
        # Setup working fallback provider
        fallback_config = ProviderConfig(type=ProviderType.OPENAI, api_key="test-key")
        fallback_provider = MockProvider(fallback_config, should_fail=False)
        fallback_key = f"{user_id}:{ProviderType.OPENAI.value}"
        
        # Register providers
        router.providers[primary_key] = primary_provider
        router.providers[fallback_key] = fallback_provider
        router.user_providers[user_id] = {
            ProviderType.ANTHROPIC.value: primary_key,
            ProviderType.OPENAI.value: fallback_key
        }
        
        # Initialize stats
        router.provider_stats[primary_key] = Mock(circuit_state=CircuitState.CLOSED)
        router.provider_stats[fallback_key] = Mock(circuit_state=CircuitState.CLOSED)
        
        messages = [{"role": "user", "content": "Hello"}]
        
        result = await router.get_completion(
            user_id, messages, preferred_provider=ProviderType.ANTHROPIC
        )
        
        assert isinstance(result, CompletionResponse)
        assert result.content == "Mock response content"
        assert result.provider == ProviderType.OPENAI  # Used fallback
        assert result.is_fallback is True
    
    @pytest.mark.asyncio
    async def test_get_completion_no_providers_configured(self, router):
        """Test completion request with no providers configured."""
        user_id = "test_user"
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(NoAvailableProvidersError, match="No providers configured"):
            await router.get_completion(user_id, messages)
    
    @pytest.mark.asyncio
    async def test_get_completion_all_providers_fail(self, router):
        """Test completion request when all providers fail."""
        user_id = "test_user"
        
        # Setup multiple failing providers
        for provider_type in [ProviderType.ANTHROPIC, ProviderType.OPENAI]:
            config = ProviderConfig(type=provider_type, api_key="test-key")
            provider = MockProvider(config, should_fail=True)
            provider_key = f"{user_id}:{provider_type.value}"
            
            router.providers[provider_key] = provider
            router.provider_stats[provider_key] = Mock(circuit_state=CircuitState.CLOSED)
        
        router.user_providers[user_id] = {
            ProviderType.ANTHROPIC.value: f"{user_id}:{ProviderType.ANTHROPIC.value}",
            ProviderType.OPENAI.value: f"{user_id}:{ProviderType.OPENAI.value}"
        }
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(NoAvailableProvidersError, match="All providers failed"):
            await router.get_completion(user_id, messages)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open(self, router, mock_config):
        """Test circuit breaker behavior when open."""
        user_id = "test_user"
        provider_key = f"{user_id}:{ProviderType.ANTHROPIC.value}"
        
        # Setup provider with open circuit breaker
        mock_provider = MockProvider(mock_config)
        router.providers[provider_key] = mock_provider
        router.user_providers[user_id] = {ProviderType.ANTHROPIC.value: provider_key}
        
        # Mock open circuit breaker
        with patch.object(router, '_can_use_provider', return_value=False):
            messages = [{"role": "user", "content": "Hello"}]
            
            with pytest.raises(NoAvailableProvidersError):
                await router.get_completion(user_id, messages)
    
    @pytest.mark.asyncio
    async def test_validate_all_providers(self, router):
        """Test validation of all user providers."""
        user_id = "test_user"
        
        # Setup providers
        for provider_type in [ProviderType.ANTHROPIC, ProviderType.OPENAI]:
            config = ProviderConfig(type=provider_type, api_key="test-key")
            provider = MockProvider(config, should_fail=(provider_type == ProviderType.OPENAI))
            provider_key = f"{user_id}:{provider_type.value}"
            
            router.providers[provider_key] = provider
        
        router.user_providers[user_id] = {
            ProviderType.ANTHROPIC.value: f"{user_id}:{ProviderType.ANTHROPIC.value}",
            ProviderType.OPENAI.value: f"{user_id}:{ProviderType.OPENAI.value}"
        }
        
        results = await router.validate_all_providers(user_id)
        
        assert len(results) == 2
        assert results[ProviderType.ANTHROPIC.value] is True
        assert results[ProviderType.OPENAI.value] is False
    
    @pytest.mark.asyncio
    async def test_health_check_all(self, router):
        """Test health checks for all user providers."""
        user_id = "test_user"
        
        # Setup providers
        for provider_type in [ProviderType.ANTHROPIC, ProviderType.OPENAI]:
            config = ProviderConfig(type=provider_type, api_key="test-key")
            provider = MockProvider(config, should_fail=(provider_type == ProviderType.OPENAI))
            provider_key = f"{user_id}:{provider_type.value}"
            
            router.providers[provider_key] = provider
        
        router.user_providers[user_id] = {
            ProviderType.ANTHROPIC.value: f"{user_id}:{ProviderType.ANTHROPIC.value}",
            ProviderType.OPENAI.value: f"{user_id}:{ProviderType.OPENAI.value}"
        }
        
        results = await router.health_check_all(user_id)
        
        assert len(results) == 2
        assert results[ProviderType.ANTHROPIC.value] is True
        assert results[ProviderType.OPENAI.value] is False
    
    def test_get_provider_stats(self, router):
        """Test getting provider statistics."""
        user_id = "test_user"
        provider_key = f"{user_id}:{ProviderType.ANTHROPIC.value}"
        
        # Setup mock stats
        from src.ai_providers.provider_router import ProviderStats
        stats = ProviderStats()
        stats.total_requests = 100
        stats.successful_requests = 95
        stats.failed_requests = 5
        stats.average_response_time = 1.5
        stats.last_success = datetime.now()
        
        router.provider_stats[provider_key] = stats
        router.user_providers[user_id] = {ProviderType.ANTHROPIC.value: provider_key}
        
        results = router.get_provider_stats(user_id)
        
        assert ProviderType.ANTHROPIC.value in results
        provider_stats = results[ProviderType.ANTHROPIC.value]
        
        assert provider_stats['total_requests'] == 100
        assert provider_stats['successful_requests'] == 95
        assert provider_stats['failed_requests'] == 5
        assert provider_stats['success_rate'] == 95.0
        assert provider_stats['average_response_time'] == 1.5
    
    def test_get_provider_order_preferred(self, router):
        """Test provider order with preferred provider."""
        user_id = "test_user"
        
        # Setup multiple providers
        router.user_providers[user_id] = {
            ProviderType.ANTHROPIC.value: f"{user_id}:{ProviderType.ANTHROPIC.value}",
            ProviderType.OPENAI.value: f"{user_id}:{ProviderType.OPENAI.value}",
            ProviderType.GOOGLE.value: f"{user_id}:{ProviderType.GOOGLE.value}"
        }
        
        provider_order = router._get_provider_order(
            user_id, 
            preferred_provider=ProviderType.OPENAI,
            fallback_order=None,
            cost_optimization=False
        )
        
        # Should start with preferred provider
        assert provider_order[0] == (ProviderType.OPENAI, False)
        
        # Should include all providers
        provider_types = [provider_type for provider_type, _ in provider_order]
        assert set(provider_types) == {ProviderType.ANTHROPIC, ProviderType.OPENAI, ProviderType.GOOGLE}
    
    def test_get_provider_order_custom_fallback(self, router):
        """Test provider order with custom fallback order."""
        user_id = "test_user"
        
        # Setup multiple providers
        router.user_providers[user_id] = {
            ProviderType.ANTHROPIC.value: f"{user_id}:{ProviderType.ANTHROPIC.value}",
            ProviderType.OPENAI.value: f"{user_id}:{ProviderType.OPENAI.value}",
            ProviderType.GOOGLE.value: f"{user_id}:{ProviderType.GOOGLE.value}"
        }
        
        custom_fallback = [ProviderType.GOOGLE, ProviderType.ANTHROPIC, ProviderType.OPENAI]
        
        provider_order = router._get_provider_order(
            user_id,
            preferred_provider=None,
            fallback_order=custom_fallback,
            cost_optimization=False
        )
        
        # Should follow custom fallback order
        expected_order = [ProviderType.GOOGLE, ProviderType.ANTHROPIC, ProviderType.OPENAI]
        actual_order = [provider_type for provider_type, _ in provider_order]
        assert actual_order == expected_order
    
    def test_can_use_provider_closed_circuit(self, router):
        """Test circuit breaker in closed state."""
        from src.ai_providers.provider_router import ProviderStats
        
        provider_key = "test_key"
        stats = ProviderStats()
        stats.circuit_state = CircuitState.CLOSED
        router.provider_stats[provider_key] = stats
        
        assert router._can_use_provider(provider_key) is True
    
    def test_can_use_provider_open_circuit(self, router):
        """Test circuit breaker in open state."""
        from src.ai_providers.provider_router import ProviderStats
        
        provider_key = "test_key"
        stats = ProviderStats()
        stats.circuit_state = CircuitState.OPEN
        stats.circuit_opened_at = datetime.now()
        router.provider_stats[provider_key] = stats
        
        assert router._can_use_provider(provider_key) is False
    
    def test_record_success(self, router):
        """Test recording successful request."""
        from src.ai_providers.provider_router import ProviderStats
        
        provider_key = "test_key"
        stats = ProviderStats()
        router.provider_stats[provider_key] = stats
        
        response_time = 1.5
        router._record_success(provider_key, response_time)
        
        assert stats.total_requests == 1
        assert stats.successful_requests == 1
        assert stats.failed_requests == 0
        assert stats.consecutive_successes == 1
        assert stats.consecutive_failures == 0
        assert stats.average_response_time == response_time
        assert stats.last_success is not None
    
    def test_record_failure(self, router):
        """Test recording failed request."""
        from src.ai_providers.provider_router import ProviderStats
        
        provider_key = "test_key"
        stats = ProviderStats()
        router.provider_stats[provider_key] = stats
        
        error = ProviderError("Test error")
        router._record_failure(provider_key, error)
        
        assert stats.total_requests == 1
        assert stats.successful_requests == 0
        assert stats.failed_requests == 1
        assert stats.consecutive_failures == 1
        assert stats.consecutive_successes == 0
        assert stats.last_failure is not None
    
    def test_circuit_breaker_transition_to_open(self, router):
        """Test circuit breaker transition to open state."""
        from src.ai_providers.provider_router import ProviderStats
        
        provider_key = "test_key"
        stats = ProviderStats()
        router.provider_stats[provider_key] = stats
        
        # Record failures to trigger circuit breaker
        for _ in range(router.circuit_config.failure_threshold):
            router._record_failure(provider_key, ProviderError("Test error"))
        
        assert stats.circuit_state == CircuitState.OPEN
        assert stats.circuit_opened_at is not None
    
    def test_circuit_breaker_transition_to_closed(self, router):
        """Test circuit breaker transition back to closed state."""
        from src.ai_providers.provider_router import ProviderStats
        
        provider_key = "test_key"
        stats = ProviderStats()
        stats.circuit_state = CircuitState.HALF_OPEN
        router.provider_stats[provider_key] = stats
        
        # Record successes to close circuit breaker
        for _ in range(router.circuit_config.success_threshold):
            router._record_success(provider_key, 1.0)
        
        assert stats.circuit_state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_auth_error_removes_provider(self, router, mock_config):
        """Test that authentication errors remove provider from rotation."""
        user_id = "test_user"
        provider_key = f"{user_id}:{ProviderType.ANTHROPIC.value}"
        
        # Setup provider that will fail with auth error
        mock_provider = MockProvider(mock_config)
        mock_provider.complete = AsyncMock(side_effect=ProviderAuthenticationError("Auth failed"))
        
        router.providers[provider_key] = mock_provider
        router.user_providers[user_id] = {ProviderType.ANTHROPIC.value: provider_key}
        router.provider_stats[provider_key] = Mock(circuit_state=CircuitState.CLOSED)
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(NoAvailableProvidersError):
            await router.get_completion(user_id, messages)
        
        # Provider should be removed
        assert provider_key not in router.providers
        assert ProviderType.ANTHROPIC.value not in router.user_providers[user_id]