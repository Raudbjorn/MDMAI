"""
Integration tests for the AI provider authentication layer.

These tests verify that all components work together correctly in realistic scenarios.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import AsyncMock, Mock, patch
from cryptography.fernet import Fernet

from src.ai_providers.base_provider import ProviderType, ProviderConfig
from src.ai_providers.credential_manager import CredentialManager
from src.ai_providers.provider_router import ProviderRouter
from src.ai_providers.usage_tracker import UsageTracker, SpendingLimitExceededException
from src.ai_providers.rate_limiter import RateLimiter
from src.ai_providers.health_checker import HealthChecker
from src.ai_providers.pricing_config import PricingConfigManager


class TestIntegration:
    """Integration tests for the complete authentication layer."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def encryption_key(self):
        """Generate test encryption key."""
        return Fernet.generate_key().decode('utf-8')
    
    @pytest.fixture
    def credential_manager(self, temp_storage, encryption_key):
        """Create CredentialManager for testing."""
        os.environ['MDMAI_ENCRYPTION_KEY'] = encryption_key
        manager = CredentialManager(storage_path=temp_storage)
        yield manager
        if 'MDMAI_ENCRYPTION_KEY' in os.environ:
            del os.environ['MDMAI_ENCRYPTION_KEY']
    
    @pytest.fixture
    def usage_tracker(self, temp_storage):
        """Create UsageTracker for testing."""
        return UsageTracker(storage_path=f"{temp_storage}/usage")
    
    @pytest.fixture
    def rate_limiter(self):
        """Create RateLimiter for testing."""
        return RateLimiter()
    
    @pytest.fixture
    def health_checker(self):
        """Create HealthChecker for testing."""
        return HealthChecker(check_interval_seconds=10)
    
    @pytest.fixture
    def provider_router(self):
        """Create ProviderRouter for testing."""
        return ProviderRouter()
    
    @pytest.fixture
    def pricing_config(self, temp_storage):
        """Create PricingConfigManager for testing."""
        config_path = f"{temp_storage}/pricing.yaml"
        return PricingConfigManager(config_path=config_path)
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(
        self, 
        credential_manager, 
        provider_router, 
        usage_tracker,
        rate_limiter,
        health_checker,
        pricing_config
    ):
        """Test complete end-to-end workflow."""
        user_id = "test_user"
        api_key = "sk-test-api-key"
        
        # 1. Store credentials securely
        encrypted_key = credential_manager.encrypt_api_key(api_key, user_id)
        assert credential_manager.has_credentials(user_id)
        
        # 2. Set up spending limits
        usage_tracker.set_spending_limit(
            user_id,
            daily_limit=5.0,
            weekly_limit=25.0,
            monthly_limit=100.0
        )
        
        # 3. Create provider configuration using the cleaner API
        decrypted_key = credential_manager.get_stored_api_key(user_id)
        assert decrypted_key == api_key
        
        config = ProviderConfig(
            type=ProviderType.ANTHROPIC,
            api_key=decrypted_key,
            model="claude-3-haiku"
        )
        
        # 4. Mock provider for testing
        with patch('src.ai_providers.anthropic_provider_auth.AnthropicProvider') as MockProvider:
            mock_instance = Mock()
            mock_instance.config = config
            mock_instance.complete = AsyncMock(return_value=["Mock response"])
            mock_instance.validate_credentials = AsyncMock(return_value=True)
            mock_instance.health_check = AsyncMock(return_value=True)
            MockProvider.return_value = mock_instance
            
            # Register provider
            provider_key = provider_router.register_provider(user_id, ProviderType.ANTHROPIC, config)
            
            # 5. Apply rate limiting
            delay = await rate_limiter.acquire(ProviderType.ANTHROPIC, user_id)
            assert delay >= 0
            
            # 6. Get completion
            messages = [{"role": "user", "content": "Hello, DM!"}]
            
            response = await provider_router.get_completion(
                user_id, 
                messages, 
                preferred_provider=ProviderType.ANTHROPIC
            )
            
            assert response.content == "Mock response"
            assert response.provider == ProviderType.ANTHROPIC
            
            # 7. Track usage
            cost = pricing_config.calculate_cost(
                ProviderType.ANTHROPIC, 
                "claude-3-haiku", 
                100, 200
            )
            
            usage_record = await usage_tracker.track_usage(
                user_id,
                ProviderType.ANTHROPIC,
                "claude-3-haiku",
                100,
                200,
                "test_session"
            )
            
            assert usage_record.cost > 0
            assert usage_record.success is True
            
            # 8. Record success in rate limiter
            rate_limiter.record_success(ProviderType.ANTHROPIC, user_id)
            
            # 9. Check health
            health_checker.register_provider(user_id, mock_instance)
            health_result = await health_checker.perform_health_check(provider_key)
            
            assert health_result.status.value == "healthy"
    
    @pytest.mark.asyncio
    async def test_spending_limit_enforcement(self, usage_tracker):
        """Test that spending limits are enforced."""
        user_id = "test_user"
        
        # Set low daily limit
        usage_tracker.set_spending_limit(user_id, daily_limit=0.10)
        
        # Track usage that would exceed limit
        with pytest.raises(SpendingLimitExceededException):
            await usage_tracker.track_usage(
                user_id,
                ProviderType.ANTHROPIC,
                "claude-3-opus",
                10000,  # High token count
                10000,  # High token count
                "test_session"
            )
    
    @pytest.mark.asyncio
    async def test_provider_fallback_with_cost_tracking(
        self, 
        provider_router, 
        usage_tracker, 
        pricing_config
    ):
        """Test provider fallback with cost tracking."""
        user_id = "test_user"
        
        # Mock providers - primary fails, fallback succeeds
        with patch('src.ai_providers.anthropic_provider_auth.AnthropicProvider') as MockAnthropic:
            with patch('src.ai_providers.openai_provider_auth.OpenAIProvider') as MockOpenAI:
                # Primary provider fails
                mock_anthropic = Mock()
                mock_anthropic.config = ProviderConfig(type=ProviderType.ANTHROPIC, api_key="key")
                mock_anthropic.complete = AsyncMock(side_effect=Exception("API Error"))
                MockAnthropic.return_value = mock_anthropic
                
                # Fallback provider succeeds
                mock_openai = Mock()
                mock_openai.config = ProviderConfig(type=ProviderType.OPENAI, api_key="key")
                mock_openai.complete = AsyncMock(return_value=["Fallback response"])
                MockOpenAI.return_value = mock_openai
                
                # Register both providers
                provider_router.register_provider(
                    user_id, 
                    ProviderType.ANTHROPIC, 
                    ProviderConfig(type=ProviderType.ANTHROPIC, api_key="key")
                )
                provider_router.register_provider(
                    user_id, 
                    ProviderType.OPENAI, 
                    ProviderConfig(type=ProviderType.OPENAI, api_key="key")
                )
                
                # Initialize circuit breaker states
                from src.ai_providers.provider_router import ProviderStats, CircuitState
                for provider_key in provider_router.provider_stats:
                    provider_router.provider_stats[provider_key] = ProviderStats()
                    provider_router.provider_stats[provider_key].circuit_state = CircuitState.CLOSED
                
                messages = [{"role": "user", "content": "Hello"}]
                
                response = await provider_router.get_completion(
                    user_id, 
                    messages, 
                    preferred_provider=ProviderType.ANTHROPIC
                )
                
                # Should use fallback
                assert response.is_fallback is True
                assert response.provider == ProviderType.OPENAI
                
                # Track the fallback usage
                cost = pricing_config.calculate_cost(
                    ProviderType.OPENAI, 
                    "gpt-4o", 
                    100, 200
                )
                
                usage_record = await usage_tracker.track_usage(
                    user_id,
                    ProviderType.OPENAI,
                    "gpt-4o",
                    100,
                    200,
                    "test_session"
                )
                
                assert usage_record.provider == ProviderType.OPENAI
                assert usage_record.cost > 0
    
    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, rate_limiter, provider_router):
        """Test rate limiting integration with provider router."""
        user_id = "test_user"
        provider_type = ProviderType.ANTHROPIC
        
        # Set aggressive rate limits for testing
        from src.ai_providers.rate_limiter import RateLimitConfig
        config = RateLimitConfig(
            requests_per_minute=2,
            requests_per_hour=10,
            burst_limit=1,
            initial_delay=0.1
        )
        rate_limiter.update_config(provider_type, config)
        
        # Test multiple rapid requests
        delays = []
        for i in range(5):
            delay = await rate_limiter.acquire(provider_type, user_id, priority=5)
            delays.append(delay)
        
        # First request should have no delay, subsequent ones should be delayed
        assert delays[0] == 0.0  # First request
        assert any(delay > 0 for delay in delays[1:])  # Subsequent requests delayed
        
        # Check rate limiting status
        status = rate_limiter.get_status(provider_type, user_id)
        assert status['requests_per_minute_remaining'] >= 0
        assert status['current_delay'] >= 0
    
    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self, health_checker, provider_router):
        """Test health monitoring integration."""
        user_id = "test_user"
        
        # Mock provider with varying health
        with patch('src.ai_providers.anthropic_provider_auth.AnthropicProvider') as MockProvider:
            mock_instance = Mock()
            mock_instance.config = ProviderConfig(type=ProviderType.ANTHROPIC, api_key="key")
            mock_instance.health_check = AsyncMock(return_value=True)
            MockProvider.return_value = mock_instance
            
            # Register provider with both router and health checker
            provider_key = provider_router.register_provider(
                user_id, 
                ProviderType.ANTHROPIC, 
                ProviderConfig(type=ProviderType.ANTHROPIC, api_key="key")
            )
            
            health_checker.register_provider(user_id, mock_instance)
            
            # Perform health check
            result = await health_checker.perform_health_check(provider_key)
            
            assert result.status.value == "healthy"
            assert result.response_time_ms >= 0
            
            # Get provider status from health checker
            status = health_checker.get_provider_status(user_id, ProviderType.ANTHROPIC)
            
            assert status['status'] == "healthy"
            assert status['metrics']['total_checks'] >= 1
    
    def test_pricing_configuration_integration(self, pricing_config):
        """Test pricing configuration integration."""
        # Test getting model pricing
        pricing = pricing_config.get_model_pricing(ProviderType.ANTHROPIC, "claude-3-haiku")
        assert pricing is not None
        assert 'input_price' in pricing
        assert 'output_price' in pricing
        
        # Test cost calculation
        cost = pricing_config.calculate_cost(
            ProviderType.ANTHROPIC,
            "claude-3-haiku",
            1000,  # 1K input tokens
            1000   # 1K output tokens
        )
        assert cost > 0
        
        # Test recommendations
        recommendations = pricing_config.get_recommended_models("quick_responses")
        assert len(recommendations) > 0
        assert any(rec['provider'] == 'anthropic' for rec in recommendations)
        
        # Test spending guidelines
        guidelines = pricing_config.get_spending_guidelines("casual")
        assert 'daily_budget' in guidelines
        assert 'weekly_budget' in guidelines
        assert 'monthly_budget' in guidelines
    
    @pytest.mark.asyncio
    async def test_credential_rotation_workflow(self, credential_manager, provider_router):
        """Test credential rotation workflow."""
        user_id = "test_user"
        old_api_key = "sk-old-key"
        new_api_key = "sk-new-key"
        
        # Store initial credentials
        credential_manager.encrypt_api_key(old_api_key, user_id)
        
        # Verify old key works
        decrypted = credential_manager.decrypt_api_key("", user_id)
        assert decrypted == old_api_key
        
        # Update with new key
        credential_manager.encrypt_api_key(new_api_key, user_id)
        
        # Verify new key works
        decrypted = credential_manager.decrypt_api_key("", user_id)
        assert decrypted == new_api_key
        
        # Verify old key no longer works by trying to create provider config
        config = ProviderConfig(
            type=ProviderType.ANTHROPIC,
            api_key=decrypted,
            model="claude-3-haiku"
        )
        
        assert config.api_key == new_api_key
    
    @pytest.mark.asyncio
    async def test_error_handling_chain(
        self, 
        credential_manager, 
        provider_router, 
        usage_tracker, 
        rate_limiter
    ):
        """Test error handling across the entire chain."""
        user_id = "test_user"
        api_key = "sk-invalid-key"
        
        # Store credentials
        credential_manager.encrypt_api_key(api_key, user_id)
        
        # Set spending limits
        usage_tracker.set_spending_limit(user_id, daily_limit=1.0)
        
        # Mock provider that fails authentication
        with patch('src.ai_providers.anthropic_provider_auth.AnthropicProvider') as MockProvider:
            mock_instance = Mock()
            mock_instance.config = ProviderConfig(type=ProviderType.ANTHROPIC, api_key=api_key)
            
            from src.ai_providers.base_provider import ProviderAuthenticationError
            mock_instance.complete = AsyncMock(side_effect=ProviderAuthenticationError("Invalid key"))
            MockProvider.return_value = mock_instance
            
            # Register provider
            provider_router.register_provider(
                user_id, 
                ProviderType.ANTHROPIC, 
                ProviderConfig(type=ProviderType.ANTHROPIC, api_key=api_key)
            )
            
            # Initialize circuit breaker state
            from src.ai_providers.provider_router import ProviderStats, CircuitState
            provider_key = f"{user_id}:{ProviderType.ANTHROPIC.value}"
            provider_router.provider_stats[provider_key] = ProviderStats()
            provider_router.provider_stats[provider_key].circuit_state = CircuitState.CLOSED
            
            # Apply rate limiting
            await rate_limiter.acquire(ProviderType.ANTHROPIC, user_id)
            
            # Try to get completion - should fail and remove provider
            messages = [{"role": "user", "content": "Hello"}]
            
            from src.ai_providers.base_provider import NoAvailableProvidersError
            with pytest.raises(NoAvailableProvidersError):
                await provider_router.get_completion(user_id, messages)
            
            # Provider should be removed due to auth failure
            assert f"{user_id}:{ProviderType.ANTHROPIC.value}" not in provider_router.providers
            
            # Record the failure in rate limiter
            rate_limiter.record_failure(ProviderType.ANTHROPIC, user_id, is_rate_limit_error=False)
            
            # Check rate limiter status shows failure
            status = rate_limiter.get_status(ProviderType.ANTHROPIC, user_id)
            assert status['consecutive_failures'] > 0
    
    @pytest.mark.asyncio
    async def test_cost_optimization_workflow(
        self, 
        pricing_config, 
        usage_tracker, 
        provider_router
    ):
        """Test cost optimization workflow."""
        user_id = "test_user"
        
        # Set budget constraints
        usage_tracker.set_spending_limit(user_id, daily_limit=2.0)
        
        # Get cost-optimized recommendations
        recommendations = pricing_config.get_recommended_models("quick_responses", "casual")
        
        # Should recommend cheaper models first
        assert len(recommendations) > 0
        
        # Find cheapest recommendation
        cheapest = min(recommendations, key=lambda x: x['cost_per_1k_tokens'])
        
        # Mock the recommended provider
        provider_type = ProviderType(cheapest['provider'])
        model_name = cheapest['model']
        
        with patch('src.ai_providers.google_provider_auth.GoogleProvider') as MockProvider:
            mock_instance = Mock()
            mock_instance.config = ProviderConfig(type=provider_type, api_key="key", model=model_name)
            mock_instance.complete = AsyncMock(return_value=["Cost-optimized response"])
            MockProvider.return_value = mock_instance
            
            # Register the cost-optimized provider
            provider_router.register_provider(
                user_id, 
                provider_type, 
                ProviderConfig(type=provider_type, api_key="key", model=model_name)
            )
            
            # Initialize circuit breaker state
            from src.ai_providers.provider_router import ProviderStats, CircuitState
            provider_key = f"{user_id}:{provider_type.value}"
            provider_router.provider_stats[provider_key] = ProviderStats()
            provider_router.provider_stats[provider_key].circuit_state = CircuitState.CLOSED
            
            # Get completion with cost optimization enabled
            messages = [{"role": "user", "content": "Quick question"}]
            
            response = await provider_router.get_completion(
                user_id, 
                messages, 
                cost_optimization=True
            )
            
            assert response.provider == provider_type
            
            # Track usage with lower cost
            cost = pricing_config.calculate_cost(provider_type, model_name, 100, 100)
            
            usage_record = await usage_tracker.track_usage(
                user_id,
                provider_type,
                model_name,
                100,
                100,
                "test_session"
            )
            
            # Should be within budget
            current_spending = usage_tracker.get_current_spending(user_id)
            assert current_spending['daily'] <= 2.0