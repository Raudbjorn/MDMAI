"""
MDMAI Cost Optimization System - Comprehensive Demo

This demo showcases the advanced cost optimization and management capabilities
of the MDMAI multi-provider LLM system including:

- Intelligent provider routing with ML-based cost optimization
- Multi-tier budget enforcement with emergency brakes
- Real-time alerting with trend analysis
- Cost prediction and forecasting
- Token optimization with semantic caching
- Advanced pricing models with volume discounts

Requirements:
- numpy
- scikit-learn
- aiohttp
- structlog
"""

import asyncio
import json
from decimal import Decimal
from datetime import datetime, timedelta

# Import the cost optimization system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cost_optimization import (
    CostManagementSystem,
    CostManagementConfig,
    ProviderType,
    AlertSeverity,
    BudgetLimitType,
    CompressionStrategy,
    ForecastHorizon
)

# Demo configuration
DEMO_USER_ID = "demo_user_123"
DEMO_MESSAGES = [
    {"role": "system", "content": "You are a helpful AI assistant specializing in data analysis and business intelligence."},
    {"role": "user", "content": "Can you analyze this sales data and provide insights on quarterly trends?"},
    {"role": "assistant", "content": "I'd be happy to help analyze your sales data! To provide meaningful insights on quarterly trends, I'll need to see the actual data. Could you please share the sales data you'd like me to analyze?"},
    {"role": "user", "content": "Here's our Q1-Q4 sales data: Q1: $125k, Q2: $180k, Q3: $165k, Q4: $220k. What patterns do you see?"},
    {"role": "assistant", "content": "Based on your quarterly sales data, I can see several interesting patterns:\n\n**Overall Growth**: Your business shows strong year-over-year growth, with Q4 ($220k) being 76% higher than Q1 ($125k).\n\n**Seasonal Trends**: \n- Strong growth from Q1 to Q2 (44% increase)\n- Slight decline in Q3 (8% drop from Q2)\n- Strong finish in Q4 (33% increase from Q3)\n\n**Key Insights**:\n1. The Q3 dip might indicate seasonal factors or market conditions\n2. Q4 shows excellent recovery and growth\n3. Average quarterly growth rate is approximately 21%\n\nWould you like me to analyze any specific factors that might have contributed to these trends?"},
    {"role": "user", "content": "That's very helpful! Can you also forecast what Q1 next year might look like based on this trend?"}
]

AVAILABLE_PROVIDERS = [
    (ProviderType.OPENAI, "gpt-4-turbo"),
    (ProviderType.OPENAI, "gpt-3.5-turbo"),
    (ProviderType.ANTHROPIC, "claude-3-opus-20240229"),
    (ProviderType.ANTHROPIC, "claude-3-sonnet-20240229"),
    (ProviderType.ANTHROPIC, "claude-3-haiku-20240307"),
    (ProviderType.GOOGLE, "gemini-pro")
]


async def demonstrate_cost_optimization():
    """Demonstrate the complete cost optimization system."""
    
    print("🚀 MDMAI Cost Optimization System Demo")
    print("=" * 50)
    
    # Initialize the cost management system
    config = CostManagementConfig()
    config.enable_ml_routing = True
    config.enable_budget_enforcement = True
    config.enable_alerts = True
    config.enable_token_optimization = True
    config.enable_caching = True
    config.default_daily_budget = Decimal("50.0")
    config.default_monthly_budget = Decimal("1500.0")
    
    cost_manager = CostManagementSystem(config)
    await cost_manager.initialize()
    
    print("✅ Cost Management System initialized")
    
    # Create user budget
    await cost_manager.create_user_budget(
        DEMO_USER_ID,
        daily_limit=Decimal("50.0"),
        monthly_limit=Decimal("1500.0")
    )
    
    print(f"✅ Created budget limits for user {DEMO_USER_ID}")
    
    # Demonstrate request optimization
    print("\n📊 REQUEST OPTIMIZATION DEMO")
    print("-" * 30)
    
    optimization_result = await cost_manager.optimize_request(
        user_id=DEMO_USER_ID,
        messages=DEMO_MESSAGES,
        available_providers=AVAILABLE_PROVIDERS,
        max_tokens=512,
        strategy="balanced"
    )
    
    print(f"Request approved: {optimization_result.approved}")
    
    if optimization_result.approved:
        print(f"Selected provider: {optimization_result.provider.value}")
        print(f"Selected model: {optimization_result.model}")
        print(f"Estimated cost: ${optimization_result.estimated_cost}")
        print(f"Optimization strategy: {optimization_result.optimization_strategy}")
        
        if optimization_result.token_optimization:
            token_info = optimization_result.token_optimization
            print(f"Token optimization:")
            print(f"  - Original tokens: {token_info['original_tokens']}")
            print(f"  - Final tokens: {token_info['final_tokens']}")
            print(f"  - Tokens saved: {token_info['tokens_saved']}")
            print(f"  - Compression ratio: {token_info['compression_ratio']:.2f}")
        
        if optimization_result.budget_status:
            budget_status = optimization_result.budget_status
            print(f"Budget status: {budget_status['action']}")
            if budget_status['violations']:
                print(f"Violations: {', '.join(budget_status['violations'])}")
    else:
        print("❌ Request denied")
        print(f"Warnings: {', '.join(optimization_result.warnings)}")
    
    # Simulate usage recording
    if optimization_result.approved:
        usage_record = {
            'timestamp': datetime.utcnow(),
            'provider': optimization_result.provider.value,
            'model': optimization_result.model,
            'input_tokens': 850,
            'output_tokens': 320,
            'cost': float(optimization_result.estimated_cost),
            'latency_ms': 1450.0,
            'success': True,
            'messages': DEMO_MESSAGES
        }
        
        mock_response = {
            'content': 'Based on the trend analysis, Q1 next year could see sales in the range of $180k-$200k...',
            'usage': {'input_tokens': 850, 'output_tokens': 320},
            'model': optimization_result.model
        }
        
        await cost_manager.record_usage(DEMO_USER_ID, usage_record, mock_response)
        print("✅ Usage recorded successfully")
    
    # Demonstrate cost forecasting
    print("\n📈 COST FORECASTING DEMO")
    print("-" * 25)
    
    # Add some historical usage data for better predictions
    await simulate_historical_usage(cost_manager, DEMO_USER_ID)
    
    # Get cost forecast
    forecast = await cost_manager.get_cost_forecast(
        DEMO_USER_ID,
        horizon=ForecastHorizon.MONTHLY,
        periods_ahead=12
    )
    
    if forecast:
        print(f"Monthly cost forecast (12 months):")
        print(f"Total predicted cost: ${forecast['forecast']['total_predicted_cost']:.2f}")
        
        insights = forecast['insights']
        print(f"Average period cost: ${insights['average_period_cost']:.2f}")
        print(f"Trend direction: {insights['trend_analysis']['direction']}")
        print(f"Risk level: {insights['risk_assessment']['risk_level']}")
        
        if insights['peak_periods']:
            print(f"Peak periods: {len(insights['peak_periods'])} identified")
        
        budget_status = forecast['budget_status']
        print(f"Current budget utilization: {len([l for l in budget_status['limits'] if l['utilization_percentage'] > 50])} limits over 50%")
    
    # Demonstrate analytics
    print("\n📊 USER ANALYTICS DEMO")
    print("-" * 22)
    
    analytics = await cost_manager.get_user_analytics(DEMO_USER_ID)
    
    if 'error' not in analytics:
        print(f"Budget status: {len(analytics['budget_status']['limits'])} limits configured")
        
        usage_patterns = analytics['usage_patterns']
        if 'pattern_type' in usage_patterns:
            print(f"Usage pattern: {usage_patterns['pattern_type']}")
            print(f"Pattern confidence: {usage_patterns.get('confidence', 0):.2f}")
        
        recommendations = analytics['recommendations']
        print(f"Recommendations: {len(recommendations)} generated")
        for rec in recommendations[:3]:  # Show first 3
            print(f"  - {rec['title']} ({rec['priority']} priority)")
    
    # Demonstrate system status
    print("\n🏥 SYSTEM STATUS DEMO")
    print("-" * 20)
    
    system_status = await cost_manager.get_system_status()
    
    print(f"System status: {system_status['system_status']}")
    print(f"Requests processed: {system_status['metrics']['requests_processed']}")
    print(f"Tokens optimized: {system_status['metrics']['tokens_optimized']}")
    print(f"Costs saved: ${system_status['metrics']['costs_saved']}")
    
    cache_stats = system_status['cache_stats']
    print(f"Cache hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"Cache size: {cache_stats['cache_size']} entries")
    
    alert_stats = system_status['alert_stats']
    print(f"Active alerts: {alert_stats['active_alerts']}")
    print(f"Alerts (24h): {alert_stats['alerts_last_24h']}")
    
    # Demonstrate advanced features
    print("\n🎯 ADVANCED FEATURES DEMO")
    print("-" * 27)
    
    # Token optimization recommendations
    token_optimizer = cost_manager.token_optimizer
    token_recommendations = token_optimizer.get_optimization_recommendations(DEMO_MESSAGES)
    
    print(f"Current conversation: {token_recommendations['current_tokens']} tokens")
    print(f"Optimization opportunities: {len(token_recommendations['optimization_opportunities'])}")
    
    for opportunity in token_recommendations['optimization_opportunities'][:2]:
        print(f"  - {opportunity['type']}: {opportunity['description']}")
    
    # Provider cost comparison
    pricing_engine = cost_manager.pricing_engine
    
    from cost_optimization.pricing_engine import CostComponent
    
    cost_comparison = pricing_engine.get_cost_comparison({
        CostComponent.INPUT_TOKENS: 1000,
        CostComponent.OUTPUT_TOKENS: 500
    })
    
    if cost_comparison['cheapest_option']:
        cheapest = cost_comparison['cheapest_option']
        most_expensive = cost_comparison['most_expensive_option']
        
        print(f"Cost comparison (1000 input + 500 output tokens):")
        print(f"  Cheapest: {cheapest['provider']}:{cheapest['model']} - ${cheapest['cost']:.4f}")
        print(f"  Most expensive: {most_expensive['provider']}:{most_expensive['model']} - ${most_expensive['cost']:.4f}")
        print(f"  Potential savings: {cost_comparison['cost_range']['savings_percentage']:.1f}%")
    
    print("\n🎉 Demo completed successfully!")
    print("\nKey Benefits Demonstrated:")
    print("✅ Intelligent provider routing with cost optimization")
    print("✅ Multi-tier budget enforcement with emergency brakes")
    print("✅ Real-time alerting and trend analysis")
    print("✅ ML-based cost prediction and forecasting")
    print("✅ Token optimization with semantic caching")
    print("✅ Advanced pricing models with volume discounts")
    print("✅ Comprehensive analytics and recommendations")


async def simulate_historical_usage(cost_manager, user_id: str):
    """Simulate historical usage data for better predictions."""
    
    print("📚 Simulating historical usage data...")
    
    # Generate 30 days of usage data
    base_date = datetime.utcnow() - timedelta(days=30)
    
    historical_records = []
    
    for day in range(30):
        # Simulate 1-5 requests per day with some randomness
        current_date = base_date + timedelta(days=day)
        
        # Weekend pattern (lower usage)
        if current_date.weekday() >= 5:  # Saturday/Sunday
            daily_requests = 1 + (day % 2)  # 1-2 requests
            base_cost = 0.002
        else:
            daily_requests = 2 + (day % 4)  # 2-5 requests
            base_cost = 0.005
        
        for req in range(daily_requests):
            record = {
                'timestamp': current_date + timedelta(hours=req * 3),
                'provider': ['openai', 'anthropic', 'google'][day % 3],
                'model': ['gpt-3.5-turbo', 'claude-3-haiku-20240307', 'gemini-pro'][day % 3],
                'input_tokens': 400 + (day * 10) + (req * 50),
                'output_tokens': 200 + (day * 5) + (req * 25),
                'cost': base_cost + (day * 0.0001) + (req * 0.001),
                'latency_ms': 800 + (req * 100),
                'success': True
            }
            
            historical_records.append(record)
    
    # Add historical data to cost predictor
    cost_manager.cost_predictor.add_usage_data(user_id, historical_records)
    
    # Record some usage for budget tracking
    for record in historical_records[-5:]:  # Last 5 records
        await cost_manager.record_usage(user_id, record)
    
    print(f"✅ Added {len(historical_records)} historical usage records")


async def demonstrate_budget_scenarios():
    """Demonstrate different budget enforcement scenarios."""
    
    print("\n💰 BUDGET ENFORCEMENT SCENARIOS")
    print("=" * 35)
    
    config = CostManagementConfig()
    config.enable_budget_enforcement = True
    config.default_daily_budget = Decimal("10.0")  # Very low budget for demo
    
    cost_manager = CostManagementSystem(config)
    await cost_manager.initialize()
    
    # Test user with tight budget
    test_user = "budget_test_user"
    await cost_manager.create_user_budget(
        test_user,
        daily_limit=Decimal("10.0"),
        monthly_limit=Decimal("300.0")
    )
    
    print(f"Created tight budget for {test_user}: $10/day, $300/month")
    
    # Scenario 1: Normal request within budget
    print("\n📝 Scenario 1: Normal request (within budget)")
    
    result1 = await cost_manager.optimize_request(
        test_user,
        DEMO_MESSAGES[:3],  # Shorter conversation
        AVAILABLE_PROVIDERS,
        max_tokens=256
    )
    
    print(f"Request approved: {result1.approved}")
    if result1.approved:
        print(f"Estimated cost: ${result1.estimated_cost}")
        
        # Record the usage
        usage = {
            'timestamp': datetime.utcnow(),
            'provider': result1.provider.value,
            'model': result1.model,
            'cost': float(result1.estimated_cost),
            'success': True
        }
        await cost_manager.record_usage(test_user, usage)
    
    # Scenario 2: Expensive request that gets downgraded
    print("\n📝 Scenario 2: Expensive request (triggers downgrade)")
    
    expensive_messages = DEMO_MESSAGES + [
        {"role": "user", "content": "Please provide a comprehensive analysis of market trends, competitor analysis, SWOT analysis, financial projections, risk assessment, and strategic recommendations for the next 5 years. Include detailed charts, graphs, and supporting data for each section."},
    ]
    
    result2 = await cost_manager.optimize_request(
        test_user,
        expensive_messages,
        AVAILABLE_PROVIDERS,
        max_tokens=2048  # Large response
    )
    
    print(f"Request approved: {result2.approved}")
    if result2.approved and result2.budget_status:
        print(f"Budget action: {result2.budget_status['action']}")
        if result2.budget_status['modifications']:
            print(f"Applied modifications: {result2.budget_status['modifications']}")
    
    # Scenario 3: Simulate budget exhaustion
    print("\n📝 Scenario 3: Budget exhaustion (request blocked)")
    
    # Simulate multiple expensive requests to exhaust budget
    for i in range(3):
        expensive_usage = {
            'timestamp': datetime.utcnow(),
            'provider': 'openai',
            'model': 'gpt-4-turbo',
            'cost': 3.5,  # Expensive request
            'success': True
        }
        await cost_manager.record_usage(test_user, expensive_usage)
    
    result3 = await cost_manager.optimize_request(
        test_user,
        DEMO_MESSAGES,
        AVAILABLE_PROVIDERS,
        max_tokens=512
    )
    
    print(f"Request approved: {result3.approved}")
    if not result3.approved:
        print(f"Denial reasons: {', '.join(result3.warnings)}")
    
    # Show final budget status
    budget_status = cost_manager.budget_enforcer.get_budget_status(test_user)
    print(f"\nFinal budget status:")
    for limit in budget_status['limits']:
        if limit['enabled']:
            print(f"  {limit['name']}: ${limit['spent']:.2f} / ${limit['amount']:.2f} ({limit['utilization_percentage']:.1f}%)")


async def demonstrate_alerting_system():
    """Demonstrate the alerting and notification system."""
    
    print("\n🚨 ALERTING SYSTEM DEMO")
    print("=" * 25)
    
    config = CostManagementConfig()
    config.enable_alerts = True
    
    cost_manager = CostManagementSystem(config)
    await cost_manager.initialize()
    
    alert_user = "alert_test_user"
    await cost_manager.create_user_budget(
        alert_user,
        daily_limit=Decimal("25.0")
    )
    
    print(f"Set up alerts for {alert_user}")
    
    # Simulate usage that triggers different alert thresholds
    alert_costs = [5.0, 15.0, 22.0, 26.0]  # 20%, 60%, 88%, 104% of $25 budget
    
    for i, cost in enumerate(alert_costs):
        usage = {
            'timestamp': datetime.utcnow(),
            'provider': 'openai',
            'model': 'gpt-3.5-turbo',
            'cost': cost,
            'success': True
        }
        
        await cost_manager.record_usage(alert_user, usage)
        
        print(f"Recorded usage: ${cost} (total: ${sum(alert_costs[:i+1])})")
        
        # Check for new alerts
        notifications = cost_manager.alert_system.get_user_notifications(alert_user)
        print(f"  Active notifications: {len(notifications)}")
    
    # Show alert statistics
    alert_stats = cost_manager.alert_system.get_alert_statistics()
    print(f"\nAlert system statistics:")
    print(f"  Total alerts: {alert_stats['total_alerts']}")
    print(f"  Active alerts: {alert_stats['active_alerts']}")
    print(f"  Alerts (24h): {alert_stats['alerts_last_24h']}")


if __name__ == "__main__":
    print("MDMAI Cost Optimization System - Comprehensive Demo")
    print("This demo requires numpy, scikit-learn, aiohttp, and structlog")
    print()
    
    async def run_all_demos():
        try:
            await demonstrate_cost_optimization()
            await demonstrate_budget_scenarios()
            await demonstrate_alerting_system()
            
        except Exception as e:
            print(f"❌ Demo error: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the demo
    asyncio.run(run_all_demos())