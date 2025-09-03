"""Tests for AI provider integration."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.ai_providers.abstract_provider import AbstractProvider
from src.ai_providers.cost_optimizer import CostOptimizer, UsageTracker
from src.ai_providers.error_handler import ErrorHandler, AIProviderError, RetryStrategy
from src.ai_providers.models import (
    AIRequest,
    AIResponse,
    CostBudget,
    ModelSpec,
    ProviderCapability,
    ProviderConfig,
    ProviderType,
    StreamingChunk,
    UsageRecord,
)
from src.ai_providers.provider_manager import AIProviderManager
from src.ai_providers.provider_registry import ProviderRegistry
from src.ai_providers.streaming_manager import StreamingManager
from src.ai_providers.tool_translator import ToolTranslator, MCPTool


class MockProvider(AbstractProvider):
    """Mock provider for testing."""
    
    def __init__(self, provider_type: ProviderType):
        config = ProviderConfig(provider_type=provider_type, api_key="test-key")
        super().__init__(config)
        self._models = {
            "test-model": ModelSpec(
                model_id="test-model",
                provider_type=provider_type,
                display_name="Test Model",
                capabilities=[ProviderCapability.TEXT_GENERATION],
                cost_per_input_token=0.001,
                cost_per_output_token=0.002,
                supports_streaming=True,
                supports_tools=True,
            )
        }
    
    async def _initialize_client(self):
        pass
    
    async def _cleanup_client(self):
        pass
    
    async def _load_models(self):
        pass
    
    async def _generate_response_impl(self, request: AIRequest) -> AIResponse:
        return AIResponse(
            request_id=request.request_id,
            provider_type=self.provider_type,
            model=request.model,
            content="Test response",
            usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            cost=0.2,
            latency_ms=500.0,
        )
    
    async def _stream_response_impl(self, request: AIRequest):
        yield StreamingChunk(
            request_id=request.request_id,
            content="Test ",
        )
        yield StreamingChunk(
            request_id=request.request_id,
            content="streaming response",
        )
        yield StreamingChunk(
            request_id=request.request_id,
            is_complete=True,
            finish_reason="stop",
        )
    
    def _get_supported_capabilities(self):
        return [ProviderCapability.TEXT_GENERATION, ProviderCapability.STREAMING]
    
    async def _perform_health_check(self):
        pass


@pytest.fixture
def mock_provider():
    return MockProvider(ProviderType.ANTHROPIC)


@pytest.fixture
def ai_request():
    return AIRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Test message"}],
        max_tokens=100,
    )


@pytest.fixture
def mcp_tool():
    return MCPTool(
        name="test_tool",
        description="A test tool",
        inputSchema={
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First parameter"}
            },
            "required": ["param1"]
        }
    )


class TestProviderRegistry:
    """Test provider registry functionality."""
    
    @pytest.mark.asyncio
    async def test_register_provider(self, mock_provider):
        registry = ProviderRegistry()
        
        await registry.register_provider(mock_provider, priority=5)
        
        assert mock_provider.provider_type in registry._providers
        assert registry._provider_priorities[mock_provider.provider_type] == 5
    
    @pytest.mark.asyncio
    async def test_get_available_providers(self, mock_provider):
        registry = ProviderRegistry()
        await registry.register_provider(mock_provider)
        
        available = registry.get_available_providers()
        assert len(available) == 1
        assert available[0] == mock_provider
    
    @pytest.mark.asyncio
    async def test_select_provider_priority_strategy(self, mock_provider):
        registry = ProviderRegistry()
        await registry.register_provider(mock_provider, priority=5)
        
        ai_request = AIRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Test"}],
        )
        
        selected = await registry.select_provider(ai_request, strategy="priority")
        assert selected == mock_provider


class TestCostOptimizer:
    """Test cost optimization functionality."""
    
    def test_usage_tracker_record(self):
        tracker = UsageTracker()
        
        request = AIRequest(model="test-model", messages=[])
        response = AIResponse(
            request_id=request.request_id,
            provider_type=ProviderType.ANTHROPIC,
            model="test-model",
            content="Test",
            usage={"input_tokens": 100, "output_tokens": 50},
            cost=0.15,
        )
        
        asyncio.run(
            tracker.record_usage(request, response, ProviderType.ANTHROPIC)
        )
        
        stats = tracker.get_usage_stats()
        assert stats["total_requests"] == 1
        assert stats["total_cost"] == 0.15
        assert stats["total_tokens"] == 150
    
    def test_cost_optimizer_budget_check(self):
        tracker = UsageTracker()
        optimizer = CostOptimizer(tracker)
        
        budget = CostBudget(
            name="Test Budget",
            daily_limit=1.0,
            monthly_limit=30.0,
        )
        optimizer.add_budget(budget)
        
        request = AIRequest(model="test-model", messages=[])
        
        # Should allow request within budget
        allowed, violations = asyncio.run(
            optimizer.check_budget_limits(request, 0.5, ProviderType.ANTHROPIC)
        )
        assert allowed
        assert len(violations) == 0
        
        # Should reject request exceeding budget
        allowed, violations = asyncio.run(
            optimizer.check_budget_limits(request, 2.0, ProviderType.ANTHROPIC)
        )
        assert not allowed
        assert len(violations) > 0


class TestErrorHandler:
    """Test error handling functionality."""
    
    def test_map_provider_error(self):
        handler = ErrorHandler()
        
        # Test generic error
        generic_error = Exception("Connection timeout")
        mapped = handler.map_provider_error(generic_error, ProviderType.OPENAI)
        
        assert isinstance(mapped, AIProviderError)
        assert mapped.provider_type == ProviderType.OPENAI
        assert "Connection timeout" in mapped.message
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self):
        handler = ErrorHandler()
        
        async def successful_function():
            return "success"
        
        result = await handler.execute_with_retry(
            successful_function,
            provider_type=ProviderType.ANTHROPIC,
            max_retries=2,
        )
        
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_failure(self):
        handler = ErrorHandler()
        
        async def failing_function():
            raise Exception("Always fails")
        
        with pytest.raises(AIProviderError):
            await handler.execute_with_retry(
                failing_function,
                provider_type=ProviderType.ANTHROPIC,
                max_retries=1,
            )


class TestToolTranslator:
    """Test tool format translation."""
    
    def test_mcp_to_anthropic_translation(self, mcp_tool):
        translator = ToolTranslator()
        
        anthropic_tools = translator.mcp_to_provider([mcp_tool], ProviderType.ANTHROPIC)
        
        assert len(anthropic_tools) == 1
        assert anthropic_tools[0]["name"] == mcp_tool.name
        assert anthropic_tools[0]["description"] == mcp_tool.description
        assert "input_schema" in anthropic_tools[0]
    
    def test_mcp_to_openai_translation(self, mcp_tool):
        translator = ToolTranslator()
        
        openai_tools = translator.mcp_to_provider([mcp_tool], ProviderType.OPENAI)
        
        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == mcp_tool.name
        assert openai_tools[0]["function"]["description"] == mcp_tool.description
    
    def test_validate_tool_compatibility(self, mcp_tool):
        translator = ToolTranslator()
        
        errors = translator.validate_tool_compatibility(mcp_tool, ProviderType.ANTHROPIC)
        assert len(errors) == 0  # Should be valid
        
        # Test invalid tool
        invalid_tool = MCPTool(name="", description="", inputSchema={})
        errors = translator.validate_tool_compatibility(invalid_tool, ProviderType.ANTHROPIC)
        assert len(errors) > 0


class TestStreamingManager:
    """Test streaming response management."""
    
    @pytest.mark.asyncio
    async def test_streaming_session(self, ai_request):
        manager = StreamingManager()
        
        async def mock_stream_generator():
            yield StreamingChunk(
                request_id=ai_request.request_id,
                content="Hello ",
            )
            yield StreamingChunk(
                request_id=ai_request.request_id,
                content="world!",
                is_complete=True,
            )
        
        chunks = []
        async for chunk in manager.start_stream(
            ai_request, "test-provider", mock_stream_generator()
        ):
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert chunks[0].content == "Hello "
        assert chunks[1].content == "world!"
        assert chunks[1].is_complete


class TestAIProviderManager:
    """Test AI provider manager integration."""
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        manager = AIProviderManager()
        
        # Mock provider configurations
        configs = [
            ProviderConfig(
                provider_type=ProviderType.ANTHROPIC,
                api_key="test-key",
                enabled=True,
            )
        ]
        
        # Mock the provider initialization
        with patch.object(manager, "_initialize_provider") as mock_init:
            mock_init.return_value = AsyncMock()
            
            await manager.initialize(configs)
            
            assert manager._initialized
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_request_integration(self):
        manager = AIProviderManager()
        
        # Mock registry and providers
        with patch.object(manager.registry, "select_provider") as mock_select:
            mock_provider = MockProvider(ProviderType.ANTHROPIC)
            await mock_provider.initialize()
            mock_select.return_value = mock_provider
            
            # Mock cost optimization checks
            with patch.object(
                manager.cost_optimizer, "check_budget_limits"
            ) as mock_budget:
                mock_budget.return_value = (True, [])
                
                request = AIRequest(
                    model="test-model",
                    messages=[{"role": "user", "content": "Test"}],
                )
                
                response = await manager.process_request(request)
                
                assert response.content == "Test response"
                assert response.provider_type == ProviderType.ANTHROPIC


class TestBridgeIntegration:
    """Test MCP Bridge integration."""
    
    @pytest.mark.asyncio
    async def test_ai_request_handling(self):
        from src.ai_providers.bridge_integration import AIProviderBridgeIntegration
        from src.bridge.models import MCPRequest
        
        # Mock provider manager
        mock_manager = MagicMock()
        mock_response = AIResponse(
            request_id="test-123",
            provider_type=ProviderType.ANTHROPIC,
            model="test-model",
            content="Test response",
            cost=0.1,
            latency_ms=100.0,
        )
        mock_manager.process_request = AsyncMock(return_value=mock_response)
        
        integration = AIProviderBridgeIntegration(mock_manager)
        
        mcp_request = MCPRequest(id="mcp-123", method="ai/generate")
        ai_request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Test"}],
        }
        
        # Mock the conversion and processing
        with patch.object(integration, "_convert_to_ai_request") as mock_convert:
            mock_convert.return_value = AIRequest(
                model="test-model",
                messages=[{"role": "user", "content": "Test"}],
            )
            
            from src.ai_providers.bridge_integration import AIProviderRequest
            ai_request = AIProviderRequest(
                model="test-model",
                messages=[{"role": "user", "content": "Test"}],
            )
            
            response = await integration.handle_ai_request(mcp_request, ai_request)
            
            assert response.id == "mcp-123"
            assert response.result["success"]
            assert "ai_response" in response.result


# Integration test that requires actual configuration
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_integration():
    """Full integration test (requires API keys to be configured)."""
    from src.ai_providers.config import get_ai_provider_config
    
    config_manager = get_ai_provider_config()
    
    # Skip if no configuration is available
    if not config_manager.config_path.exists():
        pytest.skip("No AI provider configuration found")
    
    manager = AIProviderManager()
    
    try:
        provider_configs = config_manager.get_provider_configs()
        if not provider_configs:
            pytest.skip("No AI providers configured")
        
        await manager.initialize(provider_configs)
        
        request = AIRequest(
            model="gpt-4o-mini",  # Use a cheap model for testing
            messages=[{"role": "user", "content": "Say 'Hello, World!' and nothing else."}],
            max_tokens=10,
        )
        
        response = await manager.process_request(request, strategy="cost")
        
        assert response.content
        assert response.cost is not None
        assert response.latency_ms is not None
        
    finally:
        await manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])