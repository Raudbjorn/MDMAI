"""Main AI Provider Manager that orchestrates all provider integration components."""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

from structlog import get_logger

from .anthropic_provider import AnthropicProvider
from .cost_optimizer import CostOptimizer, UsageTracker
from .error_handler import ErrorHandler, RetryStrategy, AIProviderError, BudgetExceededError
from .google_provider import GoogleProvider
from .models import (
    AIRequest,
    AIResponse,
    CostBudget,
    MCPTool,
    ProviderConfig,
    ProviderSelection,
    ProviderType,
    StreamingChunk,
)
from .openai_provider import OpenAIProvider
from .provider_registry import ProviderRegistry
from .streaming_manager import StreamingManager, StreamingResponse
from .tool_translator import ToolTranslator

logger = get_logger(__name__)


class AIProviderManager:
    """Main manager for AI provider integration with MCP Bridge.
    
    This class orchestrates all AI provider components including:
    - Provider registration and management
    - Cost optimization and budget enforcement
    - Tool format translation
    - Response streaming
    - Error handling and retry logic
    - Provider selection strategies
    """
    
    def __init__(self):
        self.registry = ProviderRegistry()
        self.usage_tracker = UsageTracker()
        self.cost_optimizer = CostOptimizer(self.usage_tracker)
        self.streaming_manager = StreamingManager()
        self.error_handler = ErrorHandler()
        self.tool_translator = ToolTranslator()
        
        self._initialized = False
        self._provider_classes = {
            ProviderType.ANTHROPIC: AnthropicProvider,
            ProviderType.OPENAI: OpenAIProvider,
            ProviderType.GOOGLE: GoogleProvider,
        }
    
    async def initialize(
        self,
        provider_configs: List[ProviderConfig],
        budgets: Optional[List[CostBudget]] = None,
    ) -> None:
        """Initialize the AI provider manager with configurations.
        
        Args:
            provider_configs: List of provider configurations
            budgets: Optional list of cost budgets
        """
        if self._initialized:
            return
        
        logger.info("Initializing AI Provider Manager")
        
        try:
            # Initialize providers
            for config in provider_configs:
                if not config.enabled:
                    continue
                
                await self._initialize_provider(config)
            
            # Set up cost budgets
            if budgets:
                for budget in budgets:
                    self.cost_optimizer.add_budget(budget)
            
            # Start health monitoring
            await self.registry.start_health_monitoring()
            
            self._initialized = True
            
            logger.info(
                "AI Provider Manager initialized successfully",
                providers=len(self.registry._providers),
                budgets=len(self.cost_optimizer._budgets),
            )
            
        except Exception as e:
            logger.error("Failed to initialize AI Provider Manager", error=str(e))
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the AI provider manager and all providers."""
        if not self._initialized:
            return
        
        logger.info("Shutting down AI Provider Manager")
        
        try:
            await self.registry.shutdown_all_providers()
            self._initialized = False
            
            logger.info("AI Provider Manager shut down successfully")
            
        except Exception as e:
            logger.error("Error during AI Provider Manager shutdown", error=str(e))
    
    async def process_request(
        self,
        request: AIRequest,
        tools: Optional[List[MCPTool]] = None,
        selection: Optional[ProviderSelection] = None,
        strategy: str = "cost",
    ) -> AIResponse:
        """Process an AI request with optimal provider selection.
        
        Args:
            request: The AI request to process
            tools: Optional MCP tools to include
            selection: Provider selection criteria
            strategy: Provider selection strategy
            
        Returns:
            AIResponse from the selected provider
            
        Raises:
            AIProviderError: If request cannot be processed
        """
        if not self._initialized:
            raise RuntimeError("AI Provider Manager not initialized")
        
        logger.info(
            "Processing AI request",
            request_id=request.request_id,
            model=request.model,
            strategy=strategy,
        )
        
        # Select optimal provider
        provider = await self.registry.select_provider(request, selection, strategy)
        if not provider:
            raise RuntimeError("No suitable AI provider available")
        
        # Translate tools if provided
        if tools:
            provider_tools = self.tool_translator.mcp_to_provider(tools, provider.provider_type)
            # Convert to request format expected by the provider
            request.tools = tools
        
        # Check budget constraints
        estimated_cost = self.cost_optimizer.estimate_request_cost(request, provider.provider_type)
        budget_ok, violations = await self.cost_optimizer.check_budget_limits(
            request, estimated_cost, provider.provider_type
        )
        
        if not budget_ok:
            raise RuntimeError(f"Budget limits exceeded: {', '.join(violations)}")
        
        # Execute request with error handling and retry logic
        response = await self.error_handler.retry_with_strategy(
            provider.generate_response,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            provider_type=provider.provider_type,
            request=request,
        )
        
        # Track usage
        await self.usage_tracker.record_usage(
            request, response, provider.provider_type, success=True
        )
        
        logger.info(
            "Successfully processed AI request",
            request_id=request.request_id,
            provider=provider.provider_type.value,
            cost=response.cost,
            latency_ms=response.latency_ms,
        )
        
        return response
    
    async def stream_request(
        self,
        request: AIRequest,
        tools: Optional[List[MCPTool]] = None,
        selection: Optional[ProviderSelection] = None,
        strategy: str = "cost",
    ) -> AsyncGenerator[StreamingChunk, None]:
        """Stream an AI request with optimal provider selection.
        
        Args:
            request: The AI request to stream
            tools: Optional MCP tools to include
            selection: Provider selection criteria
            strategy: Provider selection strategy
            
        Yields:
            StreamingChunk: Streaming response chunks
            
        Raises:
            AIProviderError: If streaming cannot be started
        """
        if not self._initialized:
            raise RuntimeError("AI Provider Manager not initialized")
        
        logger.info(
            "Starting AI request stream",
            request_id=request.request_id,
            model=request.model,
            strategy=strategy,
        )
        
        # Ensure streaming is requested
        request.stream = True
        
        # Update selection to require streaming
        if selection is None:
            selection = ProviderSelection()
        selection.require_streaming = True
        
        # Select optimal provider
        provider = await self.registry.select_provider(request, selection, strategy)
        if not provider:
            raise RuntimeError("No suitable streaming AI provider available")
        
        # Translate tools if provided
        if tools:
            provider_tools = self.tool_translator.mcp_to_provider(tools, provider.provider_type)
            request.tools = tools
        
        # Check budget constraints
        estimated_cost = self.cost_optimizer.estimate_request_cost(request, provider.provider_type)
        budget_ok, violations = await self.cost_optimizer.check_budget_limits(
            request, estimated_cost, provider.provider_type
        )
        
        if not budget_ok:
            raise RuntimeError(f"Budget limits exceeded: {', '.join(violations)}")
        
        try:
            # Create streaming generator with error handling
            async def error_handled_stream():
                try:
                    async for chunk in provider.stream_response(request):
                        yield chunk
                except Exception as e:
                    # Handle and track error
                    ai_error = await self.error_handler.handle_error(e, provider.provider_type)
                    await self.usage_tracker.record_usage(
                        request, None, provider.provider_type, success=False, error_message=str(ai_error)
                    )
                    raise ai_error
            
            # Create streaming response
            streaming_response = await self.streaming_manager.create_streaming_response(
                request.request_id,
                provider.provider_type,
                error_handled_stream(),
            )
            
            # Stream chunks
            async for chunk in streaming_response.stream():
                yield chunk
            
            # Track usage after streaming completes
            final_response = streaming_response.to_response()
            await self.usage_tracker.record_usage(
                request, final_response, provider.provider_type, success=True
            )
            
            logger.info(
                "Successfully completed AI request stream",
                request_id=request.request_id,
                provider=provider.provider_type.value,
            )
            
        except Exception as e:
            logger.error(
                "Error during AI request streaming",
                request_id=request.request_id,
                provider=provider.provider_type.value,
                error=str(e),
            )
            raise
    
    async def discover_tools(self, provider_type: Optional[ProviderType] = None) -> List[MCPTool]:
        """Discover available MCP tools.
        
        Args:
            provider_type: Specific provider to get tools from (optional)
            
        Returns:
            List of available MCP tools
        """
        # This would typically integrate with the MCP bridge to get available tools
        # For now, return empty list as tools come from the MCP server
        return []
    
    async def process_streaming_request(
        self,
        request: AIRequest,
        tools: Optional[List[MCPTool]] = None,
        selection: Optional[ProviderSelection] = None,
        strategy: str = "cost",
    ) -> StreamingResponse:
        """Process a streaming AI request and return a StreamingResponse object.
        
        Args:
            request: The AI request to stream
            tools: Optional MCP tools to include
            selection: Provider selection criteria
            strategy: Provider selection strategy
            
        Returns:
            StreamingResponse: Managed streaming response object
            
        Raises:
            AIProviderError: If streaming cannot be started
        """
        if not self._initialized:
            raise RuntimeError("AI Provider Manager not initialized")
        
        logger.info(
            "Processing streaming AI request",
            request_id=request.request_id,
            model=request.model,
            strategy=strategy,
        )
        
        # Ensure streaming is requested
        request.stream = True
        
        # Update selection to require streaming
        if selection is None:
            selection = ProviderSelection()
        selection.require_streaming = True
        
        # Select optimal provider
        provider = await self.registry.select_provider(request, selection, strategy)
        if not provider:
            raise RuntimeError("No suitable streaming AI provider available")
        
        # Translate tools if provided
        if tools:
            provider_tools = self.tool_translator.mcp_to_provider(tools, provider.provider_type)
            request.tools = tools
        
        # Check budget constraints
        estimated_cost = self.cost_optimizer.estimate_request_cost(request, provider.provider_type)
        budget_ok, violations = await self.cost_optimizer.check_budget_limits(
            request, estimated_cost, provider.provider_type
        )
        
        if not budget_ok:
            raise RuntimeError(f"Budget limits exceeded: {', '.join(violations)}")
        
        # Create error-handled stream generator
        async def error_handled_stream():
            try:
                async for chunk in provider.stream_response(request):
                    yield chunk
            except Exception as e:
                # Handle and track error
                ai_error = await self.error_handler.handle_error(e, provider.provider_type)
                await self.usage_tracker.record_usage(
                    request, None, provider.provider_type, success=False, error_message=str(ai_error)
                )
                raise ai_error
        
        # Create and return streaming response
        streaming_response = await self.streaming_manager.create_streaming_response(
            request.request_id,
            provider.provider_type,
            error_handled_stream(),
        )
        
        return streaming_response
    
    def validate_tool_compatibility(
        self, tool: MCPTool, provider_type: ProviderType
    ) -> List[str]:
        """Validate if a tool is compatible with a specific provider.
        
        Args:
            tool: MCP tool to validate
            provider_type: Target provider type
            
        Returns:
            List of validation errors (empty if valid)
        """
        return self.tool_translator.validate_tool_compatibility(tool, provider_type)
    
    async def get_cost_recommendations(
        self, request: AIRequest
    ) -> Dict[ProviderType, Dict[str, Any]]:
        """Get cost recommendations for a request across providers.
        
        Args:
            request: The AI request to analyze
            
        Returns:
            Dictionary with cost analysis per provider
        """
        recommendations = {}
        
        for provider_type, provider in self.registry._providers.items():
            if not provider.is_available:
                continue
            
            try:
                estimated_cost = self.cost_optimizer.estimate_request_cost(request, provider_type)
                
                # Find cheapest model for this provider
                cheapest = self.cost_optimizer.get_cost_efficient_models(provider_type)
                
                recommendations[provider_type] = {
                    "estimated_cost": estimated_cost,
                    "cheapest_models": cheapest[:3],  # Top 3
                    "provider_usage": self.usage_tracker.get_provider_usage(provider_type),
                    "health_score": provider.health.uptime_percentage,
                }
            except Exception as e:
                logger.error(
                    "Error calculating cost recommendation",
                    provider=provider_type.value,
                    error=str(e),
                )
        
        return recommendations
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get comprehensive manager statistics.
        
        Returns:
            Dictionary with manager statistics
        """
        return {
            "initialized": self._initialized,
            "registry_stats": self.registry.get_registry_stats(),
            "usage_stats": self.usage_tracker.get_usage_stats(),
            "streaming_stats": self.streaming_manager.get_stream_stats(),
            "error_stats": self.error_handler.get_error_stats(),
            "budget_alerts": self.cost_optimizer.get_budget_alerts(),
        }
    
    def configure_provider_priority(self, provider_type: ProviderType, priority: int) -> None:
        """Configure priority for a specific provider.
        
        Args:
            provider_type: Provider to configure
            priority: New priority (higher = preferred)
        """
        if provider_type in self.registry._provider_priorities:
            self.registry._provider_priorities[provider_type] = priority
            logger.info(
                "Updated provider priority",
                provider=provider_type.value,
                priority=priority,
            )
    
    def add_cost_budget(self, budget: CostBudget) -> None:
        """Add a new cost budget.
        
        Args:
            budget: Cost budget configuration
        """
        self.cost_optimizer.add_budget(budget)
    
    def remove_cost_budget(self, budget_id: str) -> None:
        """Remove a cost budget.
        
        Args:
            budget_id: ID of budget to remove
        """
        self.cost_optimizer.remove_budget(budget_id)
    
    async def _initialize_provider(self, config: ProviderConfig) -> None:
        """Initialize a single provider.
        
        Args:
            config: Provider configuration
        """
        if config.provider_type not in self._provider_classes:
            raise ValueError(f"Unsupported provider type: {config.provider_type}")
        
        provider_class = self._provider_classes[config.provider_type]
        provider = provider_class(config)
        
        try:
            await self.registry.register_provider(provider, config.priority)
            
            # Register models for cost optimization
            self.cost_optimizer.register_provider_models(config.provider_type, provider.models)
            
            logger.info(
                "Successfully initialized provider",
                provider=config.provider_type.value,
                models=len(provider.models),
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize provider",
                provider=config.provider_type.value,
                error=str(e),
            )
            raise
    
    # Integration methods for MCP Bridge
    
    async def create_sse_stream(
        self,
        request: AIRequest,
        tools: Optional[List[MCPTool]] = None,
        selection: Optional[ProviderSelection] = None,
        strategy: str = "cost",
    ) -> AsyncGenerator[str, None]:
        """Create SSE stream for MCP Bridge integration.
        
        Args:
            request: The AI request
            tools: Optional MCP tools
            selection: Provider selection criteria
            strategy: Provider selection strategy
            
        Yields:
            str: SSE formatted messages
        """
        provider = await self.registry.select_provider(request, selection, strategy)
        if not provider:
            raise RuntimeError("No suitable provider available")
        
        if tools:
            request.tools = tools
        
        async def stream_generator():
            async for chunk in provider.stream_response(request):
                yield chunk
        
        async for sse_message in self.streaming_manager.create_sse_stream(
            request, provider.provider_type.value, stream_generator()
        ):
            yield sse_message
    
    async def create_websocket_stream(
        self,
        request: AIRequest,
        websocket,
        tools: Optional[List[MCPTool]] = None,
        selection: Optional[ProviderSelection] = None,
        strategy: str = "cost",
    ) -> None:
        """Create WebSocket stream for MCP Bridge integration.
        
        Args:
            request: The AI request
            websocket: WebSocket connection
            tools: Optional MCP tools
            selection: Provider selection criteria
            strategy: Provider selection strategy
        """
        provider = await self.registry.select_provider(request, selection, strategy)
        if not provider:
            raise RuntimeError("No suitable provider available")
        
        if tools:
            request.tools = tools
        
        async def stream_generator():
            async for chunk in provider.stream_response(request):
                yield chunk
        
        await self.streaming_manager.create_websocket_stream(
            request, provider.provider_type.value, stream_generator(), websocket
        )