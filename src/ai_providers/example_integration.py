"""Example integration demonstrating AI provider usage with MCP Bridge."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

from structlog import get_logger

from .config import AIProviderConfigManager
from .models import AIRequest, MCPTool, ProviderSelection, ProviderType
from .provider_manager import AIProviderManager

logger = get_logger(__name__)


class AIProviderExample:
    """Example class demonstrating AI provider integration."""
    
    def __init__(self):
        self.config_manager = AIProviderConfigManager()
        self.provider_manager = AIProviderManager()
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the AI provider system."""
        if self._initialized:
            return
        
        logger.info("Initializing AI Provider Example")
        
        # Get configurations
        provider_configs = self.config_manager.get_provider_configs()
        budgets = self.config_manager.get_budgets()
        
        # Initialize provider manager
        await self.provider_manager.initialize(provider_configs, budgets)
        
        self._initialized = True
        logger.info("AI Provider Example initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the AI provider system."""
        if not self._initialized:
            return
        
        logger.info("Shutting down AI Provider Example")
        await self.provider_manager.shutdown()
        self._initialized = False
    
    # Example 1: Simple text generation
    async def example_simple_generation(self) -> None:
        """Example of simple text generation."""
        print("\n=== Example 1: Simple Text Generation ===")
        
        request = AIRequest(
            model="claude-3-haiku-20240307",  # Use a cost-effective model
            messages=[
                {"role": "user", "content": "Write a haiku about AI"}
            ],
            max_tokens=50,
            temperature=0.7,
        )
        
        try:
            response = await self.provider_manager.process_request(
                request,
                strategy="cost",  # Use cost-optimized strategy
            )
            
            print(f"Provider: {response.provider_type.value}")
            print(f"Model: {response.model}")
            print(f"Response: {response.content}")
            print(f"Cost: ${response.cost:.6f}")
            print(f"Latency: {response.latency_ms:.0f}ms")
            
        except Exception as e:
            logger.error("Simple generation failed", error=str(e))
    
    # Example 2: Tool calling
    async def example_tool_calling(self) -> None:
        """Example of AI with tool calling."""
        print("\n=== Example 2: Tool Calling ===")
        
        # Define MCP tools
        tools = [
            MCPTool(
                name="get_weather",
                description="Get the current weather for a location",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            ),
            MCPTool(
                name="search_web",
                description="Search the web for information",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            ),
        ]
        
        request = AIRequest(
            model="gpt-4o-mini",  # Use OpenAI for this example
            messages=[
                {"role": "user", "content": "What's the weather like in Paris today?"}
            ],
            tools=tools,
            max_tokens=200,
        )
        
        # Use capability-based selection to ensure tool support
        selection = ProviderSelection(
            require_tools=True,
            preferred_providers=[ProviderType.OPENAI, ProviderType.ANTHROPIC],
        )
        
        try:
            response = await self.provider_manager.process_request(
                request,
                tools=tools,
                selection=selection,
                strategy="capability",
            )
            
            print(f"Provider: {response.provider_type.value}")
            print(f"Response: {response.content}")
            
            if response.tool_calls:
                print("Tool calls:")
                for tool_call in response.tool_calls:
                    print(f"  - {tool_call}")
            
        except Exception as e:
            logger.error("Tool calling failed", error=str(e))
    
    # Example 3: Streaming response
    async def example_streaming(self) -> None:
        """Example of streaming response."""
        print("\n=== Example 3: Streaming Response ===")
        
        request = AIRequest(
            model="claude-3-5-sonnet-20241022",
            messages=[
                {"role": "user", "content": "Explain quantum computing in simple terms"}
            ],
            max_tokens=500,
            stream=True,
        )
        
        try:
            # Process streaming request
            streaming_response = await self.provider_manager.process_streaming_request(
                request,
                strategy="priority",
            )
            
            print(f"Provider: {streaming_response.provider_type.value}")
            print("Streaming response:")
            
            # Stream and print chunks
            async for chunk in streaming_response.stream():
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                if chunk.is_complete:
                    print(f"\n\nFinish reason: {chunk.finish_reason}")
            
            # Get aggregated response
            final_response = streaming_response.to_response()
            print(f"Total tokens: {len(final_response.content.split())}")
            
        except Exception as e:
            logger.error("Streaming failed", error=str(e))
    
    # Example 4: Cost optimization
    async def example_cost_optimization(self) -> None:
        """Example of cost optimization strategies."""
        print("\n=== Example 4: Cost Optimization ===")
        
        messages = [
            {"role": "user", "content": "Summarize the key points of machine learning"}
        ]
        
        # Compare costs across different models
        models = [
            "claude-3-haiku-20240307",
            "gpt-3.5-turbo",
            "gemini-1.5-flash",
        ]
        
        for model in models:
            request = AIRequest(
                model=model,
                messages=messages,
                max_tokens=200,
            )
            
            # Estimate cost before sending
            provider_type = self._get_provider_for_model(model)
            if provider_type:
                estimated_cost = self.provider_manager.cost_optimizer.estimate_request_cost(
                    request, provider_type
                )
                print(f"{model}: Estimated cost ${estimated_cost:.6f}")
        
        # Find cheapest option
        cheapest_option = self.provider_manager.cost_optimizer.find_cheapest_provider(
            request,
            available_providers=[
                ProviderType.ANTHROPIC,
                ProviderType.OPENAI,
                ProviderType.GOOGLE,
            ],
        )
        
        if cheapest_option:
            provider, model, cost = cheapest_option
            print(f"\nCheapest option: {provider.value} - {model} at ${cost:.6f}")
            
            # Process with cheapest option
            request.model = model
            response = await self.provider_manager.process_request(
                request,
                strategy="cost",
            )
            print(f"Actual cost: ${response.cost:.6f}")
    
    # Example 5: Error handling and retry
    async def example_error_handling(self) -> None:
        """Example of error handling and retry logic."""
        print("\n=== Example 5: Error Handling ===")
        
        # Simulate a request that might fail
        request = AIRequest(
            model="nonexistent-model",
            messages=[
                {"role": "user", "content": "Test error handling"}
            ],
        )
        
        try:
            # This should trigger error handling
            response = await self.provider_manager.process_request(
                request,
                strategy="failover",
            )
            
        except Exception as e:
            print(f"Expected error: {e}")
            
            # Get error statistics
            error_stats = self.provider_manager.error_handler.get_error_stats()
            print(f"Error stats: {json.dumps(error_stats, indent=2)}")
    
    # Example 6: Budget monitoring
    async def example_budget_monitoring(self) -> None:
        """Example of budget monitoring and alerts."""
        print("\n=== Example 6: Budget Monitoring ===")
        
        # Get current usage
        daily_usage = self.provider_manager.usage_tracker.get_daily_usage()
        monthly_usage = self.provider_manager.usage_tracker.get_monthly_usage()
        
        print(f"Daily usage: ${daily_usage:.2f}")
        print(f"Monthly usage: ${monthly_usage:.2f}")
        
        # Check budget alerts
        alerts = self.provider_manager.cost_optimizer.get_budget_alerts()
        if alerts:
            print("Budget alerts:")
            for alert in alerts:
                print(f"  - {alert['type']}: {alert['percentage']:.0f}% of {alert['limit']}")
        
        # Get usage statistics
        usage_stats = self.provider_manager.usage_tracker.get_usage_stats()
        print(f"\nUsage statistics:")
        print(f"  Total requests: {usage_stats['total_requests']}")
        print(f"  Success rate: {usage_stats['success_rate']:.1%}")
        print(f"  Average cost: ${usage_stats['avg_cost_per_request']:.6f}")
        print(f"  Average latency: {usage_stats['avg_latency_ms']:.0f}ms")
    
    # Example 7: Provider health monitoring
    async def example_health_monitoring(self) -> None:
        """Example of provider health monitoring."""
        print("\n=== Example 7: Health Monitoring ===")
        
        # Perform health check
        health_results = await self.provider_manager.registry.perform_health_check()
        
        for provider_type, health in health_results.items():
            print(f"\n{provider_type.value}:")
            print(f"  Status: {health.status.value}")
            print(f"  Uptime: {health.uptime_percentage:.1f}%")
            print(f"  Avg latency: {health.avg_latency_ms:.0f}ms")
            print(f"  Success count: {health.success_count}")
            print(f"  Error count: {health.error_count}")
    
    # Example 8: Multi-provider comparison
    async def example_multi_provider_comparison(self) -> None:
        """Example of comparing responses from multiple providers."""
        print("\n=== Example 8: Multi-Provider Comparison ===")
        
        messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        providers = [
            ProviderType.ANTHROPIC,
            ProviderType.OPENAI,
            ProviderType.GOOGLE,
        ]
        
        for provider_type in providers:
            # Select specific provider
            selection = ProviderSelection(
                preferred_providers=[provider_type],
            )
            
            request = AIRequest(
                model=self._get_default_model_for_provider(provider_type),
                messages=messages,
                max_tokens=50,
            )
            
            try:
                response = await self.provider_manager.process_request(
                    request,
                    selection=selection,
                    strategy="priority",
                )
                
                print(f"\n{provider_type.value}:")
                print(f"  Response: {response.content.strip()}")
                print(f"  Cost: ${response.cost:.6f}")
                print(f"  Latency: {response.latency_ms:.0f}ms")
                
            except Exception as e:
                print(f"\n{provider_type.value}: Failed - {e}")
    
    # Helper methods
    def _get_provider_for_model(self, model: str) -> Optional[ProviderType]:
        """Get provider type for a model."""
        if "claude" in model:
            return ProviderType.ANTHROPIC
        elif "gpt" in model:
            return ProviderType.OPENAI
        elif "gemini" in model:
            return ProviderType.GOOGLE
        return None
    
    def _get_default_model_for_provider(self, provider_type: ProviderType) -> str:
        """Get default model for a provider."""
        defaults = {
            ProviderType.ANTHROPIC: "claude-3-haiku-20240307",
            ProviderType.OPENAI: "gpt-3.5-turbo",
            ProviderType.GOOGLE: "gemini-1.5-flash",
        }
        return defaults.get(provider_type, "")


async def main():
    """Run all examples."""
    example = AIProviderExample()
    
    try:
        # Initialize
        await example.initialize()
        
        # Run examples
        await example.example_simple_generation()
        await example.example_tool_calling()
        await example.example_streaming()
        await example.example_cost_optimization()
        await example.example_error_handling()
        await example.example_budget_monitoring()
        await example.example_health_monitoring()
        await example.example_multi_provider_comparison()
        
    finally:
        # Shutdown
        await example.shutdown()


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())