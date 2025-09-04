#!/usr/bin/env python3
"""Demonstration of the enhanced usage tracking and cost management system.

This example shows how to integrate the comprehensive usage tracking system
into an AI application with real-time cost monitoring, budget enforcement,
and detailed analytics.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

from src.ai_providers.enhanced_usage_tracker import (
    BudgetLevel,
    ComprehensiveUsageManager,
    CurrencyCode,
    EnhancedBudget,
    UsageGranularity,
    track_usage,
)
from src.ai_providers.models import (
    AIRequest,
    AIResponse,
    ProviderType,
)


# ==================== Configuration ====================

# Storage path for usage data
USAGE_DATA_PATH = Path.home() / ".mdmai" / "usage_demo"

# Budget configurations
BUDGETS = [
    {
        "name": "Development Budget",
        "user_id": "dev_user",
        "daily_limit": Decimal("5.00"),
        "monthly_limit": Decimal("100.00"),
        "per_request_limit": Decimal("0.50"),
        "enforcement_level": BudgetLevel.SOFT,
        "alert_thresholds": [0.5, 0.75, 0.9],
    },
    {
        "name": "Production Budget",
        "tenant_id": "prod_tenant",
        "daily_limit": Decimal("50.00"),
        "monthly_limit": Decimal("1000.00"),
        "enforcement_level": BudgetLevel.ADAPTIVE,
        "degradation_strategies": ["reduce_max_tokens", "use_smaller_model"],
        "fallback_models": ["gpt-3.5-turbo", "claude-3-haiku"],
    },
    {
        "name": "Strict API Budget",
        "hourly_limit": Decimal("1.00"),
        "per_request_limit": Decimal("0.10"),
        "enforcement_level": BudgetLevel.HARD,
        "provider_limits": {
            "openai": Decimal("30.00"),
            "anthropic": Decimal("20.00"),
        },
    },
]


# ==================== Demo Application ====================

class AIApplicationDemo:
    """Demonstration AI application with usage tracking."""
    
    def __init__(self):
        """Initialize the application with usage tracking."""
        self.usage_manager = ComprehensiveUsageManager(
            storage_path=USAGE_DATA_PATH,
            enable_persistence=True,
        )
        self.session_id = None
        self.user_id = None
    
    async def setup(self):
        """Set up budgets and configuration."""
        print("🚀 Setting up enhanced usage tracking system...")
        
        # Add budgets
        for budget_config in BUDGETS:
            budget = EnhancedBudget(**budget_config)
            await self.usage_manager.budget_manager.add_budget(budget)
            print(f"✅ Added budget: {budget.name}")
        
        # Update pricing (example of dynamic pricing updates)
        self.usage_manager.cost_engine.update_model_pricing(
            ProviderType.OPENAI,
            "gpt-4-turbo-2024",
            {
                "input": Decimal("0.01"),
                "output": Decimal("0.03"),
                "cached_input": Decimal("0.005"),
            }
        )
        
        print("✅ Setup complete!\n")
    
    async def simulate_user_session(
        self,
        user_id: str,
        num_requests: int = 5,
    ):
        """Simulate a user session with multiple AI requests."""
        print(f"\n👤 Starting session for user: {user_id}")
        
        # Use session tracking context manager
        async with self.usage_manager.usage_tracker.track_session_async(
            user_id=user_id,
            metadata={"app_version": "1.0", "environment": "demo"}
        ) as session:
            
            self.session_id = session.session_id
            self.user_id = user_id
            
            print(f"📝 Session ID: {session.session_id}")
            
            for i in range(num_requests):
                await self.make_ai_request(
                    f"Request {i+1}: Tell me about {['Python', 'AI', 'Space', 'History', 'Science'][i % 5]}",
                    provider=ProviderType.OPENAI if i % 2 == 0 else ProviderType.ANTHROPIC,
                    model="gpt-4" if i % 2 == 0 else "claude-3-sonnet",
                )
                
                # Add some delay between requests
                await asyncio.sleep(0.5)
            
            # Show session summary
            print(f"\n📊 Session Summary:")
            print(f"  - Total Requests: {session.total_requests}")
            print(f"  - Total Tokens: {session.total_tokens}")
            print(f"  - Total Cost: ${session.total_cost:.4f}")
            print(f"  - Duration: {session.duration}")
    
    async def make_ai_request(
        self,
        prompt: str,
        provider: ProviderType = ProviderType.OPENAI,
        model: str = "gpt-4",
    ):
        """Make an AI request with usage tracking."""
        print(f"\n🤖 Making request to {provider.value}:{model}")
        print(f"   Prompt: {prompt[:50]}...")
        
        # Create request
        request = AIRequest(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.7,
            session_id=self.session_id,
        )
        
        # Check budget before making request
        allowed, reason, strategy = await self.usage_manager.track_request(
            request,
            provider,
            model,
            user_id=self.user_id,
        )
        
        if not allowed:
            print(f"   ❌ Request blocked: {reason}")
            return
        
        if strategy:
            print(f"   ⚠️ Budget optimization applied:")
            for s in strategy.get("strategies", []):
                print(f"      - {s['type']}")
        
        # Simulate API response
        response = AIResponse(
            request_id=request.request_id,
            provider_type=provider,
            model=model,
            content=f"This is a simulated response about {prompt[:20]}...",
            usage={
                "input_tokens": 50 + len(prompt) // 4,
                "output_tokens": 75,
            },
            latency_ms=500.0 + (100 * len(prompt) / 50),  # Simulate variable latency
        )
        
        # Record usage
        record = await self.usage_manager.record_response(
            request,
            response,
            provider,
            model,
            session_id=self.session_id,
            user_id=self.user_id,
            tags=["demo", "test"],
        )
        
        print(f"   ✅ Request completed:")
        print(f"      - Tokens: {record.token_metrics.total_tokens}")
        print(f"      - Cost: ${record.cost_breakdown.total_cost:.6f}")
        print(f"      - Latency: {record.latency_ms:.0f}ms")
        
        # Check for budget alerts
        budget_status = self.usage_manager.budget_manager.get_budget_status(
            user_id=self.user_id
        )
        
        for budget_id, status in budget_status.items():
            if status.get("alerts"):
                print(f"   ⚠️ Budget Alert for {status['name']}:")
                for alert in status["alerts"][-1:]:  # Show latest alert
                    print(f"      - {alert['reason']}")
    
    async def simulate_batch_processing(self):
        """Demonstrate batch processing with cost optimization."""
        print("\n📦 Simulating batch processing...")
        
        # Create batch of requests
        batch_requests = [
            AIRequest(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Batch item {i}"}],
                max_tokens=50,
            )
            for i in range(10)
        ]
        
        # Estimate batch cost
        total_cost, individual_costs = self.usage_manager.cost_engine.estimate_batch_cost(
            batch_requests,
            ProviderType.OPENAI,
            "gpt-3.5-turbo",
            self.usage_manager.token_counter,
        )
        
        print(f"   Batch of {len(batch_requests)} requests:")
        print(f"   - Individual total: ${sum(c.total_cost for c in individual_costs):.4f}")
        print(f"   - Batch total: ${total_cost.total_cost:.4f}")
        print(f"   - Batch discount: ${total_cost.batch_discount:.4f}")
    
    async def demonstrate_analytics(self):
        """Demonstrate analytics and reporting capabilities."""
        print("\n📈 Analytics Dashboard")
        print("=" * 60)
        
        # Get comprehensive analytics
        analytics = self.usage_manager.get_analytics()
        
        # Summary metrics
        summary = analytics.get("summary", {})
        print("\n🔍 Usage Summary:")
        print(f"   Total Cost: ${summary.get('total_cost', 0):.4f}")
        print(f"   Total Requests: {summary.get('total_requests', 0)}")
        print(f"   Success Rate: {summary.get('success_rate', 0)*100:.1f}%")
        print(f"   Total Tokens: {summary.get('total_tokens', 0):,}")
        print(f"   Cache Ratio: {summary.get('cache_ratio', 0)*100:.1f}%")
        print(f"   Avg Cost/Request: ${summary.get('avg_cost_per_request', 0):.4f}")
        print(f"   Avg Cost/1K Tokens: ${summary.get('avg_cost_per_token', 0):.4f}")
        
        # Provider breakdown
        providers = analytics.get("providers", {})
        if providers:
            print("\n🏢 Provider Breakdown:")
            for provider, data in providers.items():
                print(f"   {provider}:")
                print(f"      - Cost: ${data['cost']:.4f} ({data['percentage']*100:.1f}%)")
                print(f"      - Requests: {data['requests']}")
        
        # Model breakdown
        models = analytics.get("models", {})
        if models:
            print("\n🤖 Model Usage:")
            for model, data in sorted(models.items(), key=lambda x: x[1]['cost'], reverse=True)[:5]:
                print(f"   {model}:")
                print(f"      - Cost: ${data['cost']:.4f}")
                print(f"      - Requests: {data['requests']}")
        
        # Cost trends
        trends = analytics.get("cost_trends", {})
        if trends and "current_daily_avg" in trends:
            print("\n📊 Cost Trends:")
            print(f"   Daily Average (7d): ${trends['current_daily_avg']:.4f}")
            print(f"   Monthly Average: ${trends['monthly_avg']:.4f}")
            print(f"   Projected Monthly: ${trends['projected_monthly']:.4f}")
            print(f"   Trend: {trends.get('trend', 'stable')}")
        
        # Top users
        top_users = analytics.get("top_users", [])
        if top_users:
            print("\n👥 Top Users by Cost:")
            for user in top_users[:3]:
                print(f"   {user['user_id']}: ${user['cost']:.4f} ({user['requests']} requests)")
    
    async def demonstrate_export(self):
        """Demonstrate data export capabilities."""
        print("\n💾 Data Export Options")
        print("=" * 60)
        
        # Export as JSON
        json_export = self.usage_manager.export_data(format="json")
        data = json.loads(json_export) if json_export != "[]" else []
        print(f"\n📄 JSON Export: {len(data)} records")
        
        if data:
            # Show sample record
            sample = data[0]
            print("   Sample record:")
            print(f"      - Request ID: {sample.get('request_id', 'N/A')}")
            print(f"      - Provider: {sample.get('provider_type', 'N/A')}")
            print(f"      - Cost: ${sample.get('cost', 0):.6f}")
        
        # Export as CSV
        csv_export = self.usage_manager.export_data(format="csv")
        lines = csv_export.split('\n')
        print(f"\n📊 CSV Export: {len(lines)-1} rows")
        
        # Save exports
        export_dir = USAGE_DATA_PATH / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_file = export_dir / f"usage_export_{timestamp}.json"
        with open(json_file, 'w') as f:
            f.write(json_export)
        print(f"\n✅ JSON saved to: {json_file}")
        
        csv_file = export_dir / f"usage_export_{timestamp}.csv"
        with open(csv_file, 'w') as f:
            f.write(csv_export)
        print(f"✅ CSV saved to: {csv_file}")
    
    async def demonstrate_currency_conversion(self):
        """Demonstrate multi-currency support."""
        print("\n💱 Multi-Currency Cost Calculation")
        print("=" * 60)
        
        # Create a sample request
        request = AIRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Currency test"}],
        )
        
        # Calculate cost in different currencies
        metrics = self.usage_manager.token_counter.count_tokens(
            request.messages,
            ProviderType.OPENAI,
            "gpt-4",
        )
        
        currencies = [CurrencyCode.USD, CurrencyCode.EUR, CurrencyCode.GBP, CurrencyCode.JPY]
        
        print("\nCost in different currencies:")
        for currency in currencies:
            cost = self.usage_manager.cost_engine.calculate_cost(
                ProviderType.OPENAI,
                "gpt-4",
                metrics,
                currency=currency,
            )
            print(f"   {currency.value}: {cost.total_cost:.4f}")
    
    async def run_demo(self):
        """Run the complete demonstration."""
        print("\n" + "="*70)
        print(" Enhanced Usage Tracking and Cost Management Demo ".center(70, "="))
        print("="*70)
        
        # Setup
        await self.setup()
        
        # Simulate different user sessions
        await self.simulate_user_session("dev_user", num_requests=3)
        await self.simulate_user_session("test_user", num_requests=2)
        
        # Batch processing
        await self.simulate_batch_processing()
        
        # Currency conversion
        await self.demonstrate_currency_conversion()
        
        # Analytics
        await self.demonstrate_analytics()
        
        # Budget status
        print("\n💰 Budget Status")
        print("=" * 60)
        budget_status = self.usage_manager.budget_manager.get_budget_status()
        for budget_id, status in budget_status.items():
            print(f"\n{status['name']}:")
            if "daily" in status.get("limits", {}):
                print(f"   Daily: ${status['usage'].get('daily', 0):.2f} / ${status['limits']['daily']:.2f}")
            if "monthly" in status.get("limits", {}):
                print(f"   Monthly: ${status['usage'].get('monthly', 0):.2f} / ${status['limits']['monthly']:.2f}")
        
        # Export data
        await self.demonstrate_export()
        
        print("\n" + "="*70)
        print(" Demo Complete! ".center(70, "="))
        print("="*70)
        print(f"\n📂 Usage data saved to: {USAGE_DATA_PATH}")


# ==================== Advanced Usage Examples ====================

class AdvancedUsageExamples:
    """Advanced usage examples and patterns."""
    
    def __init__(self, usage_manager: ComprehensiveUsageManager):
        self.usage_manager = usage_manager
    
    @track_usage(ProviderType.OPENAI, "gpt-4")
    async def decorated_ai_call(self, request: AIRequest) -> AIResponse:
        """Example of using the decorator for automatic tracking."""
        # Simulate AI call
        await asyncio.sleep(0.1)
        
        return AIResponse(
            request_id=request.request_id,
            provider_type=ProviderType.OPENAI,
            model="gpt-4",
            content="Response from decorated function",
            usage={"input_tokens": 50, "output_tokens": 30},
        )
    
    async def streaming_usage_example(self):
        """Example of tracking streaming responses."""
        print("\n🌊 Streaming Usage Tracking Example")
        
        # Simulate streaming chunks
        chunks = [
            f"Chunk {i}: Some content here. " for i in range(10)
        ]
        
        total_tokens = 0
        for i, chunk in enumerate(chunks):
            # Count tokens for each chunk
            metrics = self.usage_manager.token_counter.count_tokens(
                chunk,
                ProviderType.OPENAI,
                "gpt-4",
                cache=False,  # Don't cache streaming chunks
            )
            
            total_tokens += metrics.text_tokens
            
            # Simulate streaming delay
            await asyncio.sleep(0.05)
            
            if i % 3 == 0:
                print(f"   Streamed {i+1} chunks, {total_tokens} tokens so far...")
        
        print(f"   ✅ Streaming complete: {total_tokens} total tokens")
    
    async def multimodal_usage_example(self):
        """Example of tracking multimodal content."""
        print("\n🎨 Multimodal Usage Tracking Example")
        
        # Create multimodal request
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "detail": "high"},
                    {"type": "text", "text": "And what about this audio?"},
                    {"type": "audio", "duration_seconds": 10},
                ],
            }
        ]
        
        # Count tokens
        metrics = self.usage_manager.token_counter.count_tokens(
            messages,
            ProviderType.OPENAI,
            "gpt-4-vision",
        )
        
        print(f"   Text tokens: {metrics.text_tokens}")
        print(f"   Image tokens: {metrics.image_tokens}")
        print(f"   Audio tokens: {metrics.audio_tokens}")
        print(f"   Total tokens: {metrics.total_tokens}")
        
        # Calculate cost
        cost = self.usage_manager.cost_engine.calculate_cost(
            ProviderType.OPENAI,
            "gpt-4-vision",
            metrics,
        )
        
        print(f"   Total cost: ${cost.total_cost:.4f}")
        print(f"   Image cost: ${cost.image_cost:.4f}")
        print(f"   Audio cost: ${cost.audio_cost:.4f}")


# ==================== Main Entry Point ====================

async def main():
    """Main entry point for the demonstration."""
    # Create and run demo
    demo = AIApplicationDemo()
    await demo.run_demo()
    
    # Show advanced examples
    print("\n" + "="*70)
    print(" Advanced Usage Examples ".center(70, "="))
    print("="*70)
    
    advanced = AdvancedUsageExamples(demo.usage_manager)
    
    # Decorated function example
    request = AIRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "Test decorated call"}],
    )
    response = await advanced.decorated_ai_call(request)
    print(f"\n✅ Decorated call tracked automatically")
    
    # Streaming example
    await advanced.streaming_usage_example()
    
    # Multimodal example
    await advanced.multimodal_usage_example()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())