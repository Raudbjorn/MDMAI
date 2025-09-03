# Enhanced Usage Tracking and Cost Management System

## Overview

The Enhanced Usage Tracking and Cost Management System provides comprehensive monitoring, tracking, and optimization of AI provider usage in the MDMAI project. This production-ready system offers accurate token counting, real-time cost calculation, per-user tracking, budget enforcement, and detailed analytics.

## Key Features

### 1. **Advanced Token Counting**
- Provider-specific accuracy (tiktoken for OpenAI)
- Pre and post-request estimation
- Tool/function call accounting
- Multimodal content support (images, audio, video)
- Streaming token accumulation
- LRU caching for performance

### 2. **Real-time Cost Calculation**
- Dynamic pricing models per provider/model
- Input vs output token differentiation
- Currency conversion support (USD, EUR, GBP, JPY, CNY)
- Batch processing discounts
- Volume-based discounts
- Cache-aware pricing

### 3. **Per-User Usage Tracking**
- User and tenant isolation
- Session-based tracking
- Persistent storage (JSON/ChromaDB)
- Historical aggregation (hourly, daily, monthly)
- Export capabilities (CSV, JSON)

### 4. **Budget Management**
- Hard and soft budget limits
- Adaptive degradation strategies
- Multi-level limits (request, hourly, daily, weekly, monthly)
- Provider and model-specific limits
- Alert thresholds and notifications

### 5. **Analytics and Reporting**
- Real-time usage metrics
- Provider comparison
- Cost trend analysis
- Token efficiency metrics
- Top user identification
- Time-series data generation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   ComprehensiveUsageManager                  │
│  (Main integration point for all usage tracking features)    │
└─────────────────┬───────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┬──────────────┬─────────────┐
        ▼                   ▼              ▼             ▼
┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐
│AdvancedToken │  │CostCalculation│ │Enhanced  │  │Budget    │
│Counter       │  │Engine         │ │Usage     │  │Manager   │
│              │  │               │ │Tracker   │  │          │
│-Token count  │  │-Pricing data  │ │-Recording│  │-Limits   │
│-Caching      │  │-Discounts     │ │-Sessions │  │-Alerts   │
│-Multimodal   │  │-Currency conv │ │-Analytics│  │-Strategy │
└──────────────┘  └──────────────┘  └──────────┘  └──────────┘
```

## Installation

```bash
# Install required dependencies
pip install structlog pydantic

# Optional: Install provider-specific libraries for better accuracy
pip install tiktoken  # For OpenAI token counting
pip install anthropic  # For Anthropic support
```

## Quick Start

### Basic Usage

```python
from src.ai_providers.enhanced_usage_tracker import ComprehensiveUsageManager
from src.ai_providers.models import AIRequest, ProviderType

# Initialize the manager
manager = ComprehensiveUsageManager()

# Create a request
request = AIRequest(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
)

# Check budget and track request
allowed, reason, strategy = await manager.track_request(
    request,
    ProviderType.OPENAI,
    "gpt-4",
    user_id="user123",
)

if allowed:
    # Make API call and get response
    response = await make_api_call(request)
    
    # Record usage
    record = await manager.record_response(
        request,
        response,
        ProviderType.OPENAI,
        "gpt-4",
        user_id="user123",
    )
    
    print(f"Cost: ${record.cost_breakdown.total_cost:.4f}")
```

### Session Tracking

```python
# Track a user session
async with manager.usage_tracker.track_session_async(
    user_id="user123",
    tenant_id="tenant456",
) as session:
    # Make multiple API calls within the session
    for i in range(5):
        await make_tracked_call(request)
    
    print(f"Session cost: ${session.total_cost}")
    print(f"Session tokens: {session.total_tokens}")
```

### Budget Configuration

```python
from src.ai_providers.enhanced_usage_tracker import EnhancedBudget, BudgetLevel
from decimal import Decimal

# Create a budget
budget = EnhancedBudget(
    name="Production Budget",
    daily_limit=Decimal("50.00"),
    monthly_limit=Decimal("1000.00"),
    per_request_limit=Decimal("1.00"),
    enforcement_level=BudgetLevel.ADAPTIVE,
    degradation_strategies=["reduce_max_tokens", "use_smaller_model"],
    fallback_models=["gpt-3.5-turbo"],
    alert_thresholds=[0.5, 0.75, 0.9],
)

# Add to manager
await manager.budget_manager.add_budget(budget)
```

### Using the Decorator

```python
from src.ai_providers.enhanced_usage_tracker import track_usage

@track_usage(ProviderType.OPENAI, "gpt-4", tracker=manager.usage_tracker)
async def make_ai_call(request: AIRequest) -> AIResponse:
    # Your API call logic here
    response = await api_client.create_completion(request)
    return response

# Usage is automatically tracked
response = await make_ai_call(request)
```

## Advanced Features

### Token Counting

```python
# Get detailed token metrics
metrics = manager.token_counter.count_tokens(
    messages,
    ProviderType.OPENAI,
    "gpt-4",
)

print(f"Text tokens: {metrics.text_tokens}")
print(f"Tool tokens: {metrics.tool_call_tokens}")
print(f"Image tokens: {metrics.image_tokens}")
print(f"Total: {metrics.total_tokens}")
print(f"Billable: {metrics.billable_tokens}")  # Excludes cached
```

### Cost Calculation

```python
# Calculate costs with detailed breakdown
cost_breakdown = manager.cost_engine.calculate_cost(
    ProviderType.OPENAI,
    "gpt-4",
    metrics,
    currency=CurrencyCode.EUR,  # Get cost in EUR
)

print(f"Input cost: {cost_breakdown.input_cost}")
print(f"Output cost: {cost_breakdown.output_cost}")
print(f"Cache discount: {cost_breakdown.cache_discount}")
print(f"Total: {cost_breakdown.total_cost} {cost_breakdown.currency.value}")
```

### Analytics

```python
# Get comprehensive analytics
analytics = manager.get_analytics(
    user_id="user123",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
)

print(f"Total cost: ${analytics['summary']['total_cost']:.2f}")
print(f"Requests: {analytics['summary']['total_requests']}")
print(f"Success rate: {analytics['summary']['success_rate']*100:.1f}%")

# Provider breakdown
for provider, data in analytics['providers'].items():
    print(f"{provider}: ${data['cost']:.2f} ({data['percentage']*100:.1f}%)")
```

### Data Export

```python
# Export usage data
json_data = manager.export_data(
    format="json",
    user_id="user123",
    start_date=datetime(2024, 1, 1),
)

csv_data = manager.export_data(
    format="csv",
    tenant_id="tenant456",
)

# Save to file
with open("usage_export.json", "w") as f:
    f.write(json_data)
```

## Budget Enforcement Strategies

### 1. **Hard Limits**
Requests are blocked when budget is exceeded:

```python
budget = EnhancedBudget(
    name="Strict Budget",
    daily_limit=Decimal("10.00"),
    enforcement_level=BudgetLevel.HARD,
)
```

### 2. **Soft Limits**
Requests continue but alerts are generated:

```python
budget = EnhancedBudget(
    name="Monitoring Budget",
    monthly_limit=Decimal("100.00"),
    enforcement_level=BudgetLevel.SOFT,
    alert_thresholds=[0.5, 0.75, 0.9],
)
```

### 3. **Adaptive Degradation**
Requests are modified to reduce costs:

```python
budget = EnhancedBudget(
    name="Adaptive Budget",
    daily_limit=Decimal("50.00"),
    enforcement_level=BudgetLevel.ADAPTIVE,
    degradation_strategies=[
        "reduce_max_tokens",     # Reduce response length
        "use_smaller_model",     # Switch to cheaper model
        "disable_tools",         # Disable function calling
        "reduce_temperature",    # Use more deterministic responses
    ],
    fallback_models=["gpt-3.5-turbo", "claude-3-haiku"],
)
```

## Multimodal Support

```python
# Track multimodal content
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this image"},
            {"type": "image", "detail": "high"},
            {"type": "audio", "duration_seconds": 30},
        ],
    }
]

metrics = manager.token_counter.count_tokens(
    messages,
    ProviderType.OPENAI,
    "gpt-4-vision",
)

# Costs are calculated separately for each modality
cost = manager.cost_engine.calculate_cost(
    ProviderType.OPENAI,
    "gpt-4-vision",
    metrics,
)

print(f"Image cost: ${cost.image_cost:.4f}")
print(f"Audio cost: ${cost.audio_cost:.4f}")
```

## Streaming Support

```python
# Track streaming responses
async def track_streaming(chunks):
    total_metrics = TokenCountMetrics()
    
    generator = manager.token_counter.count_streaming_tokens(
        chunks,
        ProviderType.OPENAI,
        "gpt-4",
    )
    
    async for chunk_metrics in generator:
        total_metrics.text_tokens += chunk_metrics.text_tokens
        # Process chunk...
    
    return total_metrics
```

## Performance Considerations

### Caching
- Token counting results are cached using LRU cache (1024 entries)
- Cache keys are generated from content hash for efficiency
- Disable caching for streaming: `cache=False`

### Thread Safety
- All operations are thread-safe using asyncio locks
- Sync operations use threading.RLock for safety
- Concurrent request tracking is fully supported

### Storage
- Usage data is persisted to JSON files by default
- ChromaDB integration available for larger datasets
- Automatic periodic saves every 100 records

## Configuration

### Pricing Updates

```python
# Update model pricing dynamically
manager.cost_engine.update_model_pricing(
    ProviderType.OPENAI,
    "gpt-4-turbo-2024",
    {
        "input": Decimal("0.01"),
        "output": Decimal("0.03"),
        "cached_input": Decimal("0.005"),
        "image": Decimal("0.01"),
        "audio": Decimal("0.006"),
    }
)
```

### Exchange Rates

```python
# Update currency exchange rates
manager.cost_engine.update_exchange_rates({
    CurrencyCode.EUR: Decimal("0.92"),
    CurrencyCode.GBP: Decimal("0.79"),
    CurrencyCode.JPY: Decimal("149.5"),
})
```

## Monitoring and Alerts

### Budget Status

```python
# Get current budget status
status = manager.budget_manager.get_budget_status()

for budget_id, info in status.items():
    print(f"{info['name']}:")
    print(f"  Daily: ${info['usage']['daily']}/{info['limits']['daily']}")
    print(f"  Remaining: ${info['remaining']['daily']}")
    
    # Check alerts
    for alert in info['alerts']:
        print(f"  Alert: {alert['reason']}")
```

### Real-time Monitoring

```python
# Set up real-time monitoring
async def monitor_usage():
    while True:
        analytics = manager.get_analytics()
        
        # Check thresholds
        if analytics['summary']['total_cost'] > 100:
            send_alert("High usage detected!")
        
        await asyncio.sleep(60)  # Check every minute
```

## Testing

```python
# Run comprehensive tests
pytest tests/test_enhanced_usage_tracker.py -v

# Test specific components
pytest tests/test_enhanced_usage_tracker.py::TestAdvancedTokenCounter -v
pytest tests/test_enhanced_usage_tracker.py::TestCostCalculationEngine -v
pytest tests/test_enhanced_usage_tracker.py::TestBudgetManager -v
```

## Best Practices

1. **Always use session tracking** for related requests
2. **Configure budgets** before production deployment
3. **Export data regularly** for backup and analysis
4. **Update pricing** when providers change rates
5. **Monitor alerts** and adjust limits as needed
6. **Use caching** for repeated token counting
7. **Implement degradation strategies** for cost control
8. **Tag requests** for better categorization
9. **Review analytics** weekly/monthly
10. **Test budget enforcement** in staging first

## Troubleshooting

### Issue: Token counts seem inaccurate
- Install `tiktoken` for OpenAI: `pip install tiktoken`
- Verify model name matches provider's naming
- Check if content includes special formatting

### Issue: Costs don't match provider billing
- Update pricing data with latest rates
- Verify currency conversion rates
- Check for provider-specific discounts

### Issue: Budget not enforcing limits
- Verify budget is enabled: `budget.enabled = True`
- Check enforcement level is not SOFT
- Ensure budget applies to user/tenant

### Issue: High memory usage
- Reduce usage record buffer: `deque(maxlen=1000)`
- Export and clear old records regularly
- Use ChromaDB for large-scale storage

## Integration with MDMAI

The enhanced usage tracking system integrates seamlessly with the existing MDMAI architecture:

1. **Provider Manager**: Track all provider requests automatically
2. **Cost Optimizer**: Enhanced with real-time calculations
3. **Health Monitor**: Include cost metrics in health checks
4. **Bridge Integration**: Track MCP tool usage costs
5. **API Layer**: Per-endpoint usage tracking

## Future Enhancements

- GraphQL API for analytics queries
- Webhook notifications for alerts
- Predictive cost modeling
- A/B testing for cost optimization
- Real-time dashboard WebSocket updates
- Machine learning for usage patterns
- Automated budget recommendations
- Provider cost comparison tool

## Support

For issues, questions, or contributions:
- Create an issue in the MDMAI repository
- Review existing tests for usage examples
- Check the demo script: `examples/enhanced_usage_tracking_demo.py`

---

*Last Updated: January 2025*
*Version: 1.0.0*