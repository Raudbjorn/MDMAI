# MCP Provider Router Protocol Documentation

## Overview

The MCP Provider Router with Fallback system provides a comprehensive JSON-RPC 2.0 compliant protocol for intelligent AI provider routing, automatic failover, health monitoring, and cost optimization. This document outlines the complete protocol specification, tool interfaces, and integration patterns.

## Protocol Specification

### Version Information
- **Protocol Version**: 1.0.0
- **JSON-RPC Version**: 2.0
- **Namespace**: `provider_router`
- **MCP Framework**: FastMCP 2.11.3+

### Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Provider Router Server                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐   │
│  │   Router Core   │  │ Error Handler   │  │ Health Monitor │   │
│  └─────────────────┘  └─────────────────┘  └────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                    AI Provider Manager                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │  Anthropic  │  │   OpenAI    │  │   Google    │  │   ...    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## MCP Tool Specifications

### Core Routing Tools

#### 1. route_request

**Purpose**: Intelligent request routing with fallback support

**Method**: `provider_router/route_request`

**Parameters**:
```json
{
  "request_payload": {
    "model": "string",
    "messages": [{"role": "string", "content": "string"}],
    "temperature": "number (optional)",
    "max_tokens": "number (optional)",
    "stream": "boolean (optional)"
  },
  "routing_options": {
    "strategy": "string (cost|speed|capability|priority|load_balanced|failover|random)",
    "fallback_enabled": "boolean",
    "max_retries": "number",
    "timeout": "number",
    "preferred_providers": ["string"],
    "exclude_providers": ["string"],
    "cost_limit": "number (optional)"
  }
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "success": "boolean",
    "routing_decision": {
      "selected_provider": "string",
      "selected_model": "string",
      "strategy_used": "string",
      "decision_factors": "object",
      "alternatives_considered": ["string"],
      "estimated_cost": "number",
      "confidence_score": "number",
      "fallback_available": "boolean"
    },
    "response_data": "object (optional)",
    "execution_metrics": {
      "total_time_ms": "number",
      "routing_time_ms": "number",
      "provider_time_ms": "number",
      "retries": "number"
    },
    "error_details": "object (optional)"
  },
  "id": "string"
}
```

**Error Codes**:
- `-32002`: No provider available
- `-32003`: Routing failed
- `-32004`: Provider timeout
- `-32200`: Budget exceeded

#### 2. configure_routing

**Purpose**: Dynamic routing configuration management

**Method**: `provider_router/configure_routing`

**Parameters**:
```json
{
  "routing_config": {
    "default_strategy": "string",
    "fallback_chain": ["string"],
    "health_check_interval": "number",
    "retry_config": {
      "max_attempts": "number",
      "backoff_multiplier": "number",
      "max_backoff": "number"
    },
    "cost_thresholds": {
      "daily_warning": "number",
      "daily_limit": "number",
      "monthly_limit": "number"
    },
    "provider_priorities": {
      "provider_name": "number"
    }
  }
}
```

#### 3. get_provider_status

**Purpose**: Real-time provider health and status information

**Method**: `provider_router/get_provider_status`

**Response**:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "providers": [
      {
        "name": "string",
        "type": "string",
        "status": "healthy|degraded|unhealthy|critical",
        "health_score": "number (0-1)",
        "response_time_ms": "number",
        "success_rate": "number (0-1)",
        "current_load": "number",
        "cost_per_1k_tokens": "number",
        "models_available": ["string"],
        "capabilities": ["string"],
        "last_health_check": "string (ISO 8601)"
      }
    ],
    "overall_system_health": {
      "healthy_providers": "number",
      "total_providers": "number",
      "average_uptime": "number",
      "system_status": "string"
    },
    "active_configuration": "object",
    "last_updated": "string (ISO 8601)"
  },
  "id": "string"
}
```

#### 4. test_provider_chain

**Purpose**: Validate fallback chain with test requests

**Method**: `provider_router/test_provider_chain`

**Parameters**:
```json
{
  "test_payload": {
    "model": "string",
    "messages": [{"role": "user", "content": "Test message"}],
    "max_tokens": "number"
  },
  "test_options": {
    "include_costs": "boolean",
    "timeout_per_provider": "number",
    "test_streaming": "boolean"
  }
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "chain_test_results": [
      {
        "provider": "string",
        "success": "boolean",
        "response_time_ms": "number",
        "error": "string (optional)",
        "cost_estimate": "number",
        "model_available": "boolean"
      }
    ],
    "successful_providers": ["string"],
    "failed_providers": ["string"],
    "total_test_time_ms": "number",
    "recommendations": ["string"]
  },
  "id": "string"
}
```

#### 5. force_failover

**Purpose**: Manual provider failover control

**Method**: `provider_router/force_failover`

**Parameters**:
```json
{
  "failover_config": {
    "from_provider": "string",
    "to_provider": "string (optional)",
    "reason": "string",
    "duration": "number (seconds, optional)",
    "automatic_restore": "boolean"
  }
}
```

### Monitoring Tools

#### 6. get_system_health

**Method**: `provider_router/get_system_health`

**Response**: Comprehensive system health including all providers, error rates, performance metrics, and availability status.

#### 7. get_routing_stats

**Method**: `provider_router/get_routing_stats`

**Response**: Detailed routing statistics, provider performance analytics, cost optimization data, and usage patterns.

#### 8. get_active_alerts

**Method**: `provider_router/get_active_alerts`

**Response**: Current health alerts, their severity levels, affected providers, and recommended actions.

## Event Notifications

The MCP Provider Router emits real-time notifications using JSON-RPC 2.0 notification format (no response expected).

### Provider Health Events

#### provider_health_changed
```json
{
  "jsonrpc": "2.0",
  "method": "health/provider_health_changed",
  "params": {
    "event_type": "provider_health_changed",
    "timestamp": "2024-01-01T12:00:00Z",
    "provider_name": "anthropic",
    "previous_status": "healthy",
    "current_status": "degraded",
    "health_metrics": {
      "response_time": {"value": 2500, "unit": "ms"},
      "error_rate": {"value": 0.08, "unit": "ratio"},
      "success_rate": {"value": 0.92, "unit": "ratio"}
    },
    "impact_assessment": {
      "routing_affected": true,
      "fallback_triggered": false,
      "estimated_recovery_time": 300
    }
  }
}
```

### Routing Events

#### failover_triggered
```json
{
  "jsonrpc": "2.0",
  "method": "router/failover_triggered",
  "params": {
    "event_type": "failover_triggered",
    "timestamp": "2024-01-01T12:00:00Z",
    "trigger_reason": "High error rate detected",
    "from_provider": "openai",
    "to_provider": "anthropic",
    "request_id": "req-12345",
    "automatic": true,
    "estimated_downtime": 120
  }
}
```

#### routing_decision_made
```json
{
  "jsonrpc": "2.0",
  "method": "router/routing_decision_made",
  "params": {
    "event_type": "routing_decision_made",
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req-12345",
    "routing_decision": {
      "selected_provider": "anthropic",
      "strategy_used": "cost",
      "decision_factors": {
        "cost_per_token": 0.0003,
        "response_time_ms": 1200,
        "success_rate": 0.99
      },
      "alternatives_considered": ["openai", "google"]
    },
    "context": {
      "fallback_attempt": false,
      "cost_limit_active": true,
      "performance_priority": "balanced"
    }
  }
}
```

### Cost Events

#### cost_threshold_exceeded
```json
{
  "jsonrpc": "2.0",
  "method": "cost/threshold_exceeded",
  "params": {
    "event_type": "cost_threshold_exceeded",
    "timestamp": "2024-01-01T12:00:00Z",
    "threshold_type": "daily_warning",
    "current_usage": 85.50,
    "threshold_limit": 100.00,
    "provider": null,
    "recommended_actions": [
      "Consider switching to lower-cost providers",
      "Monitor usage more frequently",
      "Review optimization opportunities"
    ]
  }
}
```

## Error Handling

### Error Code Ranges

- **-32700 to -32603**: Standard JSON-RPC 2.0 errors
- **-32001 to -32099**: Provider management errors
- **-32100 to -32199**: Routing errors
- **-32200 to -32299**: Cost and budget errors
- **-32300 to -32399**: Health monitoring errors
- **-32400 to -32499**: Configuration errors

### Error Response Format

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32002,
    "message": "No suitable providers available",
    "data": {
      "error_id": "err-12345",
      "category": "provider_error",
      "severity": "high",
      "recovery_strategy": "failover",
      "can_retry": true,
      "context": {
        "request_id": "req-12345",
        "provider": null,
        "timestamp": "2024-01-01T12:00:00Z"
      },
      "suggestions": [
        "Check provider configurations",
        "Verify API keys and credentials",
        "Review provider health status"
      ]
    }
  },
  "id": "req-12345"
}
```

### Circuit Breaker Integration

The system implements circuit breaker patterns for provider resilience:

**States**:
- `CLOSED`: Normal operation
- `OPEN`: Failing, rejecting requests
- `HALF_OPEN`: Testing recovery

**Notifications**:
```json
{
  "jsonrpc": "2.0",
  "method": "error/circuit_breaker_state_changed",
  "params": {
    "provider": "openai",
    "state": "open",
    "failure_count": 5,
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

## Integration Patterns

### FastMCP Integration

```python
from src.ai_providers.mcp_integration import create_provider_router_server
from src.ai_providers.models import ProviderConfig, CostBudget, ProviderType

# Create provider configurations
provider_configs = [
    ProviderConfig(
        provider_type=ProviderType.ANTHROPIC,
        enabled=True,
        priority=100,
        api_key="your-api-key"
    ),
    ProviderConfig(
        provider_type=ProviderType.OPENAI,
        enabled=True,
        priority=90,
        api_key="your-api-key"
    )
]

# Create cost budgets
cost_budgets = [
    CostBudget(
        budget_id="daily-limit",
        daily_limit=100.0,
        monthly_limit=1000.0,
        alert_thresholds=[0.8, 0.9]
    )
]

# Create and start server
server = await create_provider_router_server(
    server_name="my-provider-router",
    provider_configs=provider_configs,
    cost_budgets=cost_budgets,
    auto_start=True
)

# Subscribe to events
def handle_health_change(event_data):
    print(f"Provider health changed: {event_data}")

server.subscribe_to_events("provider_health_changed", handle_health_change)
```

### Event Subscription Example

```python
# Subscribe to multiple event types
server.subscribe_to_events("failover_triggered", lambda event: handle_failover(event))
server.subscribe_to_events("cost_threshold_exceeded", lambda event: handle_cost_alert(event))
server.subscribe_to_events("system_events", lambda event: log_system_event(event))

# Custom event handlers
async def handle_failover(event_data):
    provider = event_data["from_provider"]
    reason = event_data["trigger_reason"]
    
    # Send notification to operations team
    await send_alert(f"Failover triggered for {provider}: {reason}")

async def handle_cost_alert(event_data):
    usage = event_data["current_usage"]
    limit = event_data["threshold_limit"]
    
    # Implement cost control measures
    if usage / limit > 0.95:
        await implement_emergency_cost_controls()
```

## Protocol Compliance

### JSON-RPC 2.0 Requirements

1. **Request Format**: All requests must include `jsonrpc: "2.0"`, `method`, and `id`
2. **Response Format**: All responses include `jsonrpc: "2.0"`, `result` or `error`, and `id`
3. **Notifications**: Include `jsonrpc: "2.0"` and `method`, but no `id`
4. **Error Handling**: Standard error codes with custom extensions

### Validation Requirements

- All parameters validated against Pydantic schemas
- Input sanitization for security
- Type checking and conversion
- Range validation for numeric values

### Performance Characteristics

- **Routing Decision Time**: < 50ms (typical)
- **Health Check Interval**: 60s (configurable)
- **Failover Time**: < 5s (automatic)
- **Event Notification Latency**: < 100ms
- **Circuit Breaker Recovery**: 60s (configurable)

## Security Considerations

1. **API Key Management**: Secure storage and rotation
2. **Input Validation**: Comprehensive parameter validation
3. **Rate Limiting**: Built-in provider rate limiting
4. **Access Control**: Role-based access to management functions
5. **Audit Logging**: Complete audit trail of all operations

## Monitoring and Observability

### Health Metrics
- Response time percentiles (P50, P95, P99)
- Error rates by provider and time window
- Success rates and availability
- Throughput and request volume
- Cost per request and optimization savings

### Alerting Thresholds
- Provider response time > 5s (critical)
- Error rate > 15% (critical)
- Success rate < 85% (critical)
- Cost threshold > 90% (warning)
- Health score < 0.5 (unhealthy)

### Dashboards
- Real-time provider status
- Cost optimization metrics
- Routing decision analytics
- Error trends and resolution
- Performance benchmarks

## Best Practices

### Configuration Management
1. Use environment-specific configurations
2. Implement gradual rollouts for changes
3. Maintain fallback configurations
4. Regular configuration validation

### Error Handling
1. Implement comprehensive retry strategies
2. Use circuit breakers for resilience
3. Provide meaningful error messages
4. Log errors with full context

### Performance Optimization
1. Cache routing decisions when appropriate
2. Use connection pooling for providers
3. Implement request batching where possible
4. Monitor and optimize critical paths

### Cost Management
1. Set appropriate budget limits
2. Monitor usage trends regularly
3. Implement cost alerting
4. Regular cost optimization reviews

This comprehensive MCP protocol integration provides a robust, scalable, and feature-rich solution for AI provider routing with intelligent failover, health monitoring, and cost optimization capabilities.