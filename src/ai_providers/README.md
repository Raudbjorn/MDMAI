# AI Provider Integration Module

## Overview

The AI Provider Integration module provides a unified interface for integrating multiple AI providers (Anthropic Claude, OpenAI GPT, Google Gemini) with the MCP Bridge infrastructure. It handles provider management, cost optimization, tool format translation, and response streaming.

## Architecture

### Core Components

1. **AbstractProvider**: Base class defining the interface for all AI providers
2. **Provider Implementations**: Concrete implementations for each AI provider
3. **ProviderManager**: Orchestrates all provider operations
4. **ProviderRegistry**: Manages provider registration and selection
5. **CostOptimizer**: Tracks usage and optimizes costs
6. **ToolTranslator**: Converts between MCP and provider-specific tool formats
7. **StreamingManager**: Handles streaming responses
8. **ErrorHandler**: Unified error handling with retry logic

## Features

### Multi-Provider Support

- **Anthropic Claude**: Supports Claude 3.5 Sonnet, Claude 3 Haiku
- **OpenAI GPT**: Supports GPT-4o, GPT-4o Mini, GPT-3.5 Turbo
- **Google Gemini**: Supports Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini Pro

### Cost Optimization

- Real-time usage tracking per provider
- Budget enforcement with daily/monthly limits
- Cost-aware provider selection strategies
- Budget alerts and notifications
- Provider-specific budget limits

### Tool Format Translation

- MCP tool format to Anthropic tool use format
- MCP tool format to OpenAI function calling format
- MCP tool format to Google Gemini function format
- Bidirectional translation support

### Provider Management

- Health monitoring with circuit breakers
- Automatic failover between providers
- Load balancing across providers
- Priority-based provider selection
- Capability-based routing

### Error Handling

- Circuit breaker pattern for provider failures
- Exponential backoff retry strategies
- Error classification and mapping
- Rate limit handling
- Budget exceeded handling

## Configuration

### Environment Variables

```bash
# API Keys
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
GOOGLE_API_KEY=your-google-key

# Provider Settings
AI_ANTHROPIC_ENABLED=true
AI_OPENAI_ENABLED=true
AI_GOOGLE_ENABLED=true

# Provider Priorities (higher = preferred)
AI_ANTHROPIC_PRIORITY=10
AI_OPENAI_PRIORITY=5
AI_GOOGLE_PRIORITY=3

# Budget Limits (USD)
AI_DAILY_BUDGET=100.0
AI_MONTHLY_BUDGET=3000.0

# Selection Strategy (cost|priority|capability|load_balanced|failover)
AI_SELECTION_STRATEGY=cost

# Health Monitoring
AI_HEALTH_CHECK_ENABLED=true
AI_HEALTH_CHECK_INTERVAL=300
```

### YAML Configuration

Create `config/ai_providers.yaml`:

```yaml
# Provider configurations
anthropic_enabled: true
openai_enabled: true
google_enabled: true

# Budget settings
daily_budget_limit: 100.0
monthly_budget_limit: 3000.0
budget_alert_thresholds:
  - 0.5
  - 0.8
  - 0.95

# Provider-specific limits
anthropic_daily_limit: 50.0
openai_daily_limit: 30.0
google_daily_limit: 20.0

# Request defaults
default_max_tokens: 2048
default_temperature: 0.7
request_timeout: 30.0
max_retries: 3

# Selection strategy
default_selection_strategy: cost

# Circuit breaker settings
circuit_breaker_failure_threshold: 5
circuit_breaker_recovery_timeout: 60.0
```

## Usage

### Basic Example

```python
from src.ai_providers import AIProviderManager
from src.ai_providers.models import AIRequest
from src.ai_providers.config import AIProviderConfigManager

# Initialize configuration
config_manager = AIProviderConfigManager()
provider_configs = config_manager.get_provider_configs()
budgets = config_manager.get_budgets()

# Initialize provider manager
manager = AIProviderManager()
await manager.initialize(provider_configs, budgets)

# Create request
request = AIRequest(
    model="claude-3-haiku-20240307",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=100,
)

# Process request with cost optimization
response = await manager.process_request(
    request,
    strategy="cost",  # Use cheapest provider
)

print(f"Response: {response.content}")
print(f"Cost: ${response.cost:.6f}")
print(f"Provider: {response.provider_type.value}")
```

### Tool Calling Example

```python
from src.ai_providers.models import MCPTool, ProviderSelection

# Define MCP tools
tools = [
    MCPTool(
        name="get_weather",
        description="Get current weather",
        inputSchema={
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    )
]

# Create request with tools
request = AIRequest(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"}
    ],
    tools=tools,
)

# Select provider with tool support
selection = ProviderSelection(
    require_tools=True,
    preferred_providers=[ProviderType.OPENAI],
)

response = await manager.process_request(
    request,
    tools=tools,
    selection=selection,
    strategy="capability",
)

if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call['function']['name']}")
        print(f"Args: {tool_call['function']['arguments']}")
```

### Streaming Example

```python
# Create streaming request
request = AIRequest(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ],
    max_tokens=500,
    stream=True,
)

# Get streaming response
streaming_response = await manager.process_streaming_request(
    request,
    strategy="priority",
)

# Stream chunks
async for chunk in streaming_response.stream():
    if chunk.content:
        print(chunk.content, end="", flush=True)
    if chunk.is_complete:
        print(f"\nFinished: {chunk.finish_reason}")
```

### Budget Monitoring

```python
# Get current usage
daily_usage = manager.usage_tracker.get_daily_usage()
monthly_usage = manager.usage_tracker.get_monthly_usage()

print(f"Daily usage: ${daily_usage:.2f}")
print(f"Monthly usage: ${monthly_usage:.2f}")

# Check budget alerts
alerts = manager.cost_optimizer.get_budget_alerts()
for alert in alerts:
    print(f"Alert: {alert['type']} - {alert['percentage']:.0f}% of limit")

# Get usage statistics
stats = manager.usage_tracker.get_usage_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Average cost: ${stats['avg_cost_per_request']:.6f}")
```

## Provider Selection Strategies

### Available Strategies

1. **cost**: Select the cheapest provider for the request
2. **priority**: Select based on configured provider priorities
3. **capability**: Select based on required capabilities (tools, streaming, vision)
4. **load_balanced**: Distribute load evenly across providers
5. **failover**: Use primary provider with automatic failover
6. **round_robin**: Rotate through available providers
7. **random**: Random provider selection

### Strategy Configuration

```python
# Cost-optimized selection
response = await manager.process_request(
    request, strategy="cost"
)

# Capability-based selection
selection = ProviderSelection(
    required_capabilities=[
        ProviderCapability.TOOL_CALLING,
        ProviderCapability.STREAMING,
    ],
    exclude_providers=[ProviderType.GOOGLE],
)

response = await manager.process_request(
    request, selection=selection, strategy="capability"
)
```

## Error Handling

### Circuit Breaker

The system implements circuit breaker pattern to handle provider failures:

- **Closed**: Normal operation, requests pass through
- **Open**: Provider failing, requests blocked
- **Half-Open**: Testing recovery with limited requests

### Retry Strategies

- **NONE**: No retry
- **FIXED**: Fixed delay between retries
- **EXPONENTIAL_BACKOFF**: Exponential delay increase
- **LINEAR_BACKOFF**: Linear delay increase
- **JITTERED_EXPONENTIAL**: Exponential with random jitter

### Error Types

- `RateLimitError`: Rate limit exceeded (retryable)
- `QuotaExceededError`: API quota exceeded (non-retryable)
- `AuthenticationError`: Invalid API key (non-retryable)
- `ModelNotFoundError`: Model not available (non-retryable)
- `ServiceUnavailableError`: Temporary service issue (retryable)
- `BudgetExceededError`: Budget limit reached (non-retryable)

## Health Monitoring

### Health Check

```python
# Perform health check
health_results = await manager.registry.perform_health_check()

for provider_type, health in health_results.items():
    print(f"{provider_type.value}:")
    print(f"  Status: {health.status.value}")
    print(f"  Uptime: {health.uptime_percentage:.1f}%")
    print(f"  Avg latency: {health.avg_latency_ms:.0f}ms")
```

### Automatic Monitoring

```python
# Start automatic health monitoring
await manager.registry.start_health_monitoring(interval=300)

# Stop monitoring
await manager.registry.stop_health_monitoring()
```

## Testing

Run tests with:

```bash
# Run all AI provider tests
pytest tests/test_ai_providers.py -v

# Run comprehensive tests
pytest tests/test_ai_providers_complete.py -v

# Run with coverage
pytest tests/test_ai_providers_complete.py --cov=src.ai_providers
```

## Performance Considerations

### Caching

- Response caching for identical requests
- Cost calculation caching
- Model specification caching

### Optimization Tips

1. Use cost-optimized strategy for non-critical requests
2. Enable streaming for long responses
3. Set appropriate timeout values
4. Use budget limits to control costs
5. Monitor provider health regularly
6. Implement request batching where possible

## Troubleshooting

### Common Issues

1. **No providers available**
   - Check API keys are set
   - Verify providers are enabled
   - Check circuit breaker status

2. **Budget exceeded**
   - Review daily/monthly limits
   - Check provider-specific limits
   - Monitor usage patterns

3. **Streaming not working**
   - Ensure provider supports streaming
   - Check model capabilities
   - Verify network connectivity

4. **Tool calls failing**
   - Validate tool schema format
   - Check provider tool support
   - Review tool translation

## Future Enhancements

- [ ] Add support for more providers (Cohere, Mistral, etc.)
- [ ] Implement request batching
- [ ] Add response caching layer
- [ ] Support for fine-tuned models
- [ ] Advanced cost prediction
- [ ] Multi-modal support improvements
- [ ] Provider performance analytics
- [ ] Automatic model selection

## License

See main project LICENSE file.