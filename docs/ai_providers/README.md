# Phase 15: AI Provider Integration

## Overview

Phase 15 introduces comprehensive AI provider integration for the TTRPG Assistant MCP Server, enabling users to connect their own AI accounts (Anthropic Claude, OpenAI GPT, Google Gemini) to interact with MCP tools.

## Architecture

### Core Components

1. **Provider Abstraction Layer** (`abstract_provider.py`)
   - Unified interface for all AI providers
   - Health monitoring and status tracking
   - Capability-based provider selection

2. **Provider Implementations**
   - `anthropic_provider.py` - Anthropic Claude integration
   - `openai_provider.py` - OpenAI GPT integration  
   - `google_provider.py` - Google Gemini integration

3. **Tool Format Translation** (`tool_translator.py`)
   - Converts MCP tools to provider-specific formats
   - Validates tool compatibility across providers
   - Handles schema normalization

4. **Cost Optimization System** (`cost_optimizer.py`)
   - Usage tracking and budget enforcement
   - Cost-aware provider selection
   - Budget alerts and recommendations

5. **Provider Selection Strategies** (`provider_registry.py`)
   - Round-robin, priority-based, cost-optimized selection
   - Load balancing and failover strategies
   - Capability-based filtering

6. **Response Streaming** (`streaming_manager.py`)
   - Unified streaming interface across providers
   - SSE and WebSocket streaming support
   - Stream session management

7. **Error Handling** (`error_handler.py`)
   - Unified error mapping across providers
   - Circuit breaker pattern implementation
   - Configurable retry strategies

8. **Provider Manager** (`provider_manager.py`)
   - Orchestrates all integration components
   - Main entry point for AI provider operations

## Features

### Multi-Provider Support
- **Anthropic Claude**: Claude 3.5 Sonnet, Claude 3 Haiku
- **OpenAI GPT**: GPT-4o, GPT-4o Mini, GPT-3.5 Turbo
- **Google Gemini**: Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini Pro

### Advanced Cost Optimization
- Real-time usage tracking
- Daily and monthly budget enforcement
- Provider-specific budget limits
- Cost recommendations and alerts
- Automatic cheapest provider selection

### Provider Selection Strategies
- **Cost-optimized**: Selects cheapest available provider
- **Priority-based**: Uses configured provider priorities
- **Capability-based**: Matches provider capabilities to requirements
- **Load-balanced**: Distributes requests evenly
- **Failover**: Prefers most reliable providers

### Streaming Support
- Unified streaming interface across all providers
- Server-Sent Events (SSE) support
- WebSocket streaming integration
- Stream session management and monitoring

### Error Handling & Reliability
- Circuit breaker pattern prevents cascade failures
- Configurable retry strategies with exponential backoff
- Provider health monitoring
- Automatic failover to backup providers

## Quick Start

### 1. Configuration

Create `config/ai_providers.yaml`:

```yaml
providers:
  - type: anthropic
    api_key: ${ANTHROPIC_API_KEY}
    enabled: true
    priority: 1
    timeout: 30.0
    max_retries: 3

  - type: openai
    api_key: ${OPENAI_API_KEY}
    enabled: true
    priority: 2
    timeout: 30.0
    max_retries: 3

  - type: google
    api_key: ${GOOGLE_API_KEY}
    enabled: true
    priority: 3
    timeout: 30.0
    max_retries: 3

budgets:
  - name: Daily Budget
    daily_limit: 10.0
    alert_thresholds: [0.5, 0.8, 0.95]
    enabled: true

  - name: Monthly Budget
    monthly_limit: 300.0
    provider_limits:
      anthropic: 150.0
      openai: 100.0
      google: 50.0
    enabled: true
```

### 2. Environment Variables

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"  
export GOOGLE_API_KEY="your-google-key"
```

### 3. Basic Usage

```python
from src.ai_providers import AIProviderManager
from src.ai_providers.models import AIRequest
from src.ai_providers.config import get_ai_provider_config

# Initialize
config_manager = get_ai_provider_config()
manager = AIProviderManager()

# Load configurations
provider_configs = config_manager.get_provider_configs()
cost_budgets = config_manager.get_cost_budgets()

await manager.initialize(provider_configs, cost_budgets)

# Simple request
request = AIRequest(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello, world!"}],
    max_tokens=100
)

response = await manager.process_request(request, strategy="cost")
print(f"Response: {response.content}")
print(f"Cost: ${response.cost:.4f}")
```

### 4. MCP Bridge Integration

The AI providers integrate seamlessly with the existing MCP Bridge:

```python
from src.ai_providers.example_integration import create_ai_enhanced_bridge_server

# Create enhanced bridge server
server = await create_ai_enhanced_bridge_server()

# New endpoints available:
# POST /ai/generate - Generate AI responses
# GET /ai/status - Get provider status
# POST /ai/recommendations - Get cost recommendations
# POST /ai/configure - Configure providers
```

## Advanced Usage

### Tool Calling with MCP Tools

```python
from src.ai_providers.models import MCPTool

# Define MCP tool
weather_tool = MCPTool(
    name="get_weather",
    description="Get weather for a location",
    inputSchema={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
    }
)

# Request with tools
request = AIRequest(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}]
)

response = await manager.process_request(
    request, 
    tools=[weather_tool],
    strategy="capability_based"
)
```

### Streaming Responses

```python
# Streaming request
stream_request = AIRequest(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

async for chunk in manager.stream_request(stream_request):
    if chunk.content:
        print(chunk.content, end="", flush=True)
    if chunk.is_complete:
        break
```

### Provider Selection Criteria

```python
from src.ai_providers.models import ProviderSelection, ProviderType

selection = ProviderSelection(
    preferred_providers=[ProviderType.ANTHROPIC, ProviderType.OPENAI],
    require_streaming=True,
    require_tools=True,
    max_cost_per_request=0.05
)

response = await manager.process_request(
    request, 
    selection=selection,
    strategy="capability_based"
)
```

### Cost Recommendations

```python
# Get cost analysis across providers
recommendations = await manager.get_cost_recommendations(request)

for provider_type, rec in recommendations.items():
    print(f"{provider_type.value}:")
    print(f"  Estimated cost: ${rec['estimated_cost']:.4f}")
    print(f"  Health score: {rec['health_score']:.1f}%")
```

## API Reference

### AIRequest

| Field | Type | Description |
|-------|------|-------------|
| `model` | str | Model identifier |
| `messages` | List[Dict] | Conversation messages |
| `tools` | List[MCPTool] | Optional MCP tools |
| `max_tokens` | int | Maximum output tokens |
| `temperature` | float | Sampling temperature |
| `stream` | bool | Enable streaming |
| `budget_limit` | float | Per-request budget limit |

### ProviderSelection

| Field | Type | Description |
|-------|------|-------------|
| `required_capabilities` | List[ProviderCapability] | Required capabilities |
| `preferred_providers` | List[ProviderType] | Preferred provider order |
| `exclude_providers` | List[ProviderType] | Providers to exclude |
| `max_cost_per_request` | float | Cost limit per request |
| `require_streaming` | bool | Require streaming support |
| `require_tools` | bool | Require tool calling |

### Selection Strategies

- `"cost"` - Select cheapest available provider
- `"priority"` - Use configured provider priorities  
- `"capability_based"` - Match capabilities to requirements
- `"load_balanced"` - Distribute requests evenly
- `"failover"` - Prefer most reliable providers
- `"round_robin"` - Cycle through providers

## Monitoring & Analytics

### Usage Statistics

```python
stats = manager.get_manager_stats()

print(f"Total requests: {stats['usage_stats']['total_requests']}")
print(f"Total cost: ${stats['usage_stats']['total_cost']:.2f}")
print(f"Success rate: {stats['usage_stats']['success_rate']:.1%}")
```

### Provider Health

```python
for provider_type, provider in manager.registry._providers.items():
    health = provider.health
    print(f"{provider_type.value}:")
    print(f"  Status: {health.status.value}")
    print(f"  Uptime: {health.uptime_percentage:.1f}%")
    print(f"  Avg latency: {health.avg_latency_ms:.0f}ms")
```

### Budget Alerts

```python
alerts = manager.cost_optimizer.get_budget_alerts()
for alert in alerts:
    print(f"Alert: {alert['budget_name']} at {alert['percentage']:.1f}%")
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_PROVIDER_ENABLED` | Enable AI integration | `true` |
| `AI_PROVIDER_DEFAULT_STRATEGY` | Default selection strategy | `"cost"` |
| `AI_PROVIDER_MAX_RETRIES` | Default max retries | `3` |
| `AI_PROVIDER_CONFIG_PATH` | Config file path | `config/ai_providers.yaml` |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `GOOGLE_API_KEY` | Google API key | - |

### Provider Settings

Each provider supports:
- `enabled`: Enable/disable provider
- `api_key`: API authentication key
- `base_url`: Custom API endpoint
- `timeout`: Request timeout (seconds)
- `max_retries`: Maximum retry attempts
- `priority`: Selection priority (higher = preferred)

### Budget Configuration

- `daily_limit`: Maximum daily spend (USD)
- `monthly_limit`: Maximum monthly spend (USD)
- `provider_limits`: Per-provider spending limits
- `alert_thresholds`: Budget alert percentages (0.0-1.0)

## Error Handling

### Common Errors

- `RateLimitError`: API rate limits exceeded
- `QuotaExceededError`: API quota/billing limits exceeded
- `AuthenticationError`: Invalid API keys
- `ModelNotFoundError`: Requested model unavailable
- `ServiceUnavailableError`: Provider service down
- `InvalidRequestError`: Malformed request parameters

### Retry Strategies

- `EXPONENTIAL_BACKOFF`: Exponential delay increase
- `LINEAR_BACKOFF`: Linear delay increase
- `JITTERED_EXPONENTIAL`: Exponential with random jitter
- `FIXED`: Fixed delay between retries

### Circuit Breaker

Automatically opens when:
- 5+ consecutive failures (configurable)
- Prevents cascade failures
- Auto-resets after timeout period
- Half-open state for recovery testing

## Performance Considerations

### Optimization Tips

1. **Use cost-optimized strategy** for best price/performance
2. **Enable streaming** for better user experience
3. **Set appropriate budget limits** to control costs
4. **Monitor provider health** and adjust priorities
5. **Use cheaper models** for simple tasks

### Resource Usage

- **Memory**: ~50MB per provider instance
- **Network**: HTTP/2 connections to provider APIs
- **Storage**: Usage logs and configuration files
- **CPU**: Minimal overhead for request routing

## Security

### API Key Management

- Store keys in environment variables
- Use separate keys for different environments
- Rotate keys regularly
- Monitor usage for anomalies

### Request Validation

- Input sanitization and validation
- Token limit enforcement
- Budget limit checks
- Rate limiting protection

### Data Privacy

- No request/response content stored by default
- Usage metadata only (tokens, cost, timing)
- Configurable audit logging
- Secure transmission to providers

## Troubleshooting

### Common Issues

**Provider not available**
```bash
# Check configuration
python -m src.ai_providers.example_integration --example requests
```

**Authentication errors**
```bash
# Verify API keys
export ANTHROPIC_API_KEY="your-key-here"
```

**Budget exceeded**
```python
# Check current usage
stats = manager.get_manager_stats()
print(stats['usage_stats']['total_cost'])
```

**Model not found**
```python
# List available models
for provider in manager.registry._providers.values():
    print(f"{provider.provider_type.value}: {list(provider.models.keys())}")
```

### Debug Mode

```python
import structlog
structlog.get_logger().setLevel("DEBUG")
```

## Testing

### Unit Tests

```bash
pytest tests/test_ai_providers.py -v
```

### Integration Tests

```bash
pytest tests/test_ai_providers.py::test_full_integration -v -m integration
```

### Example Usage

```bash
python -m src.ai_providers.example_integration --example requests
python -m src.ai_providers.example_integration --example bridge
```

## Migration Guide

### From Previous Versions

1. Install new dependencies: `pip install -r requirements.txt`
2. Create AI provider configuration file
3. Set API keys in environment variables
4. Update bridge server initialization
5. Test with example integration

### Breaking Changes

- New configuration format required
- API endpoints added to bridge server
- Additional dependencies for HTTP clients

## Roadmap

### Planned Features

- [ ] Additional provider integrations (Cohere, Together AI)
- [ ] Advanced caching and response deduplication
- [ ] Multi-modal support (images, audio)
- [ ] Batch request optimization
- [ ] Enhanced analytics and reporting
- [ ] Provider-specific optimizations
- [ ] Custom model fine-tuning integration

### Feedback

Please report issues and feature requests in the project repository.

## License

This integration follows the same license as the main TTRPG Assistant project.