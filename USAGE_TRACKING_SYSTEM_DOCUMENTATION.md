# Comprehensive Usage Tracking and Cost Management System

## Overview

This comprehensive system provides enterprise-grade usage tracking, cost management, and optimization for the MDMAI TTRPG Assistant MCP Server. The system is designed to handle multiple AI providers (Anthropic, OpenAI, Google, Ollama) with real-time cost calculation, per-user tracking, budget enforcement, and intelligent optimization recommendations.

## Architecture

The system consists of 10 integrated components working together to provide comprehensive cost management:

```
┌─────────────────────────────────────────────────────────────────┐
│                 ComprehensiveUsageSystem                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Token Estimator │  │ Pricing Engine  │  │ Usage Tracker   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │Budget Enforcer  │  │Analytics Engine │  │Cost Optimizer   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐                      │
│  │Metrics Collector│  │ Alert System    │                      │
│  └─────────────────┘  └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
         │                      │                      │
    ┌────▼────┐         ┌──────▼──────┐        ┌──────▼──────┐
    │JSON     │         │ChromaDB     │        │File System │
    │Storage  │         │Vector Store │        │Backups      │
    └─────────┘         └─────────────┘        └─────────────┘
```

## Component Documentation

### 1. Enhanced Token Estimator (`enhanced_token_estimator.py`)

**Purpose**: Accurate token counting for all AI providers with provider-specific optimizations.

**Key Features**:
- Provider-specific tokenizers (tiktoken for OpenAI, heuristic for Anthropic/Google)
- Multimodal content support (text + images)
- Tool call token estimation
- Intelligent caching with 1000-item LRU cache
- Vision token estimation with detail level support

**Token Counting Methods**:
- **OpenAI**: tiktoken with cl100k_base/p50k_base encodings
- **Anthropic**: Character-based heuristic (3.5 chars/token optimized)
- **Google**: SentencePiece approximation (4.2 chars/token)
- **Ollama**: Conservative estimation (4.5 chars/token)

**Configuration**:
```python
estimator = EnhancedTokenEstimator()
input_tokens, output_tokens = estimator.estimate_request_tokens(
    provider_type=ProviderType.OPENAI,
    model="gpt-4",
    messages=messages,
    tools=tools,
    max_output_tokens=1000
)
```

### 2. Real-Time Pricing Engine (`pricing_engine.py`)

**Purpose**: Dynamic cost calculation with support for multiple pricing models and real-time rate updates.

**Pricing Models Supported**:
- **Token-based**: Per 1K input/output tokens (most common)
- **Request-based**: Fixed cost per API call
- **Time-based**: Cost per processing minute
- **Tiered**: Volume discounts based on usage
- **Dynamic**: Demand-based pricing with time-of-day factors

**Current Pricing (as of 2024)**:

| Provider | Model | Input ($/1K) | Output ($/1K) | Min Charge |
|----------|-------|--------------|---------------|------------|
| OpenAI | GPT-4 | $0.030 | $0.060 | $0.0001 |
| OpenAI | GPT-4-Turbo | $0.010 | $0.030 | $0.0001 |
| OpenAI | GPT-3.5-Turbo | $0.0015 | $0.002 | $0.00005 |
| Anthropic | Claude-3-Opus | $0.015 | $0.075 | $0.0001 |
| Anthropic | Claude-3-Sonnet | $0.003 | $0.015 | $0.00005 |
| Anthropic | Claude-3-Haiku | $0.00025 | $0.00125 | $0.00001 |
| Google | Gemini-Pro | $0.0005 | $0.0015 | $0.00001 |

**Dynamic Pricing Factors**:
- **Time-of-day**: 20% premium during business hours, 10% discount off-peak
- **Demand**: Up to 30% premium during high usage periods
- **Provider health**: 50% premium for unhealthy providers, 10% discount for excellent
- **Volume discounts**: 15% discount for >1M tokens/month, 5% for >100K

### 3. Per-User Usage Tracker (`user_usage_tracker.py`)

**Purpose**: Comprehensive per-user tracking with persistent storage in JSON files and ChromaDB.

**Data Schema**:

**UserProfile**:
```json
{
  "user_id": "user_abc123",
  "username": "john_doe",
  "email": "john@example.com",
  "created_at": "2024-01-15T10:30:00Z",
  "user_tier": "premium",
  "is_active": true,
  "preferences": {},
  "metadata": {}
}
```

**UserUsageAggregation** (daily/monthly):
```json
{
  "user_id": "user_abc123",
  "date": "2024-01-15",
  "total_requests": 45,
  "successful_requests": 43,
  "failed_requests": 2,
  "total_input_tokens": 12500,
  "total_output_tokens": 8300,
  "total_cost": 0.85,
  "avg_latency_ms": 1250.0,
  "providers_used": {"anthropic": 25, "openai": 20},
  "models_used": {"claude-3-sonnet": 25, "gpt-3.5-turbo": 20},
  "session_count": 8,
  "unique_sessions": ["session_1", "session_2"]
}
```

**UserSpendingLimits**:
```json
{
  "user_id": "user_abc123",
  "daily_limit": 10.0,
  "weekly_limit": 50.0,
  "monthly_limit": 150.0,
  "per_request_limit": 1.0,
  "per_session_limit": 5.0,
  "provider_limits": {"openai": 30.0, "anthropic": 80.0},
  "alert_thresholds": [0.5, 0.8, 0.95],
  "enabled": true
}
```

### 4. Budget Enforcement System (`budget_enforcer.py`)

**Purpose**: Intelligent budget enforcement with graceful degradation and alternative routing.

**Enforcement Actions**:
1. **ALLOW**: Request proceeds normally
2. **WARN**: Allow with warning message
3. **THROTTLE**: Add delay (exponential backoff: 1s, 2s, 4s, 8s, max 30s)
4. **DOWNGRADE**: Switch to cheaper model automatically
5. **CACHE_ONLY**: Serve only cached responses
6. **QUEUE**: Queue request for later processing
7. **DENY**: Block the request
8. **EMERGENCY_STOP**: Stop all requests with cooldown

**Budget Policies by User Tier**:

| Tier | Warning | Throttle | Downgrade | Hard Limit | Grace Period | Max Overage |
|------|---------|----------|-----------|------------|--------------|-------------|
| Free | 70% | 80% | 90% | 100% | 6h | $2.00 |
| Premium | 80% | 90% | 95% | 100% | 12h | $25.00 |
| Enterprise | 90% | 95% | 98% | 105% | 24h | $100.00 |

**Model Alternatives for Downgrading**:
- **GPT-4** → GPT-4-Turbo (33% cost savings) → GPT-3.5-Turbo (95% cost savings) → Claude-3-Haiku (98% cost savings)
- **Claude-3-Opus** → Claude-3-Sonnet (80% cost savings) → Claude-3-Haiku (98% cost savings) → GPT-3.5-Turbo (90% cost savings)
- **Claude-3-Sonnet** → Claude-3-Haiku (90% cost savings) → GPT-3.5-Turbo (50% cost savings) → Gemini-Pro (83% cost savings)

### 5. Analytics Dashboard (`analytics_dashboard.py`)

**Purpose**: Comprehensive analytics with customizable dashboards and real-time metrics.

**Default Dashboards**:

**Overview Dashboard**:
- Request Volume (line chart)
- Cost Tracking (line chart)
- Token Usage (area chart)
- Performance Metrics (line chart)
- Provider Distribution (pie chart)
- User Activity (gauge)

**Cost Analysis Dashboard**:
- Cost Trend (30-day line chart)
- Cost Breakdown by Provider (bar chart)
- Cost Efficiency Metrics (line chart)

**User Analytics Dashboard**:
- User Activity Heatmap (hourly patterns)
- Top Users by Cost (table)
- Popular Models (bar chart)

**Metric Definitions**:
```python
MetricDefinition(
    metric_id="total_requests",
    name="Total Requests",
    aggregation_method="count",
    chart_type=ChartType.LINE,
    unit="requests",
    warning_threshold=1000,
    critical_threshold=5000
)
```

### 6. Cost Optimization Engine (`cost_optimization_engine.py`)

**Purpose**: AI-powered cost optimization with intelligent recommendations.

**Optimization Types**:
1. **Model Downgrade**: Switch to cheaper models with acceptable quality trade-offs
2. **Provider Switch**: Route to more cost-effective providers
3. **Batch Processing**: Combine similar requests for volume discounts
4. **Caching**: Cache frequent requests to avoid redundant costs
5. **Request Optimization**: Optimize prompt length and parameters
6. **Usage Timing**: Schedule non-urgent requests during off-peak hours
7. **Context Management**: Optimize context window usage
8. **Streaming Optimization**: Use streaming for better perceived performance

**Recommendation Example**:
```json
{
  "recommendation_id": "model_downgrade_user123_202401",
  "optimization_type": "model_downgrade",
  "priority": "high",
  "title": "Switch from GPT-4 to GPT-3.5-Turbo",
  "description": "Replace expensive model usage with cost-effective alternative",
  "potential_savings": 45.50,
  "potential_savings_percentage": 85.0,
  "implementation_effort": "low",
  "quality_impact": "minimal",
  "confidence_score": 0.92,
  "action_items": [
    "Update model configuration from gpt-4 to gpt-3.5-turbo",
    "Test alternative model with sample requests",
    "Monitor quality metrics after implementation"
  ]
}
```

### 7. Metrics Collector (`metrics_collector.py`)

**Purpose**: Comprehensive metrics collection with intelligent retention policies.

**Retention Policies**:

| Environment | Raw Data | Aggregated | Compression | Backup |
|-------------|----------|------------|-------------|--------|
| Development | 7 days | 30 days | 6 hours | Disabled |
| Production | 30 days | 365 days | 7 days | 90 days |
| Enterprise | 90 days | Permanent | 30 days | 365 days |

**Aggregation Levels**:
- **Raw**: Individual metric points
- **Minute**: Per-minute aggregations
- **Hour**: Per-hour aggregations  
- **Day**: Daily aggregations
- **Week**: Weekly aggregations
- **Month**: Monthly aggregations

**Aggregated Metric Structure**:
```json
{
  "metric_name": "requests_total",
  "aggregation_level": "hour",
  "time_bucket": "2024-01-15T14:00:00Z",
  "statistics": {
    "count": 125,
    "sum": 125.0,
    "avg": 1.0,
    "min": 0.0,
    "max": 5.0,
    "std_dev": 0.8,
    "p50": 1.0,
    "p95": 3.0,
    "p99": 4.0
  },
  "context": {
    "unique_users": 15,
    "unique_sessions": 45,
    "unique_providers": 3,
    "unique_models": 8
  }
}
```

### 8. Alert System (`alert_system.py`)

**Purpose**: Intelligent alerting with multi-channel notifications and escalation.

**Alert Types**:
- **Budget Threshold**: Daily/monthly budget warnings and limits
- **Cost Anomaly**: Unusual cost spikes (>200% of baseline)
- **Error Rate**: High error rates (>10% warning, >25% critical)
- **Latency Spike**: Response time issues (>5s warning, >10s critical)
- **Provider Failure**: Provider unavailability or severe degradation
- **Quota Exceeded**: API quota limits reached
- **Usage Spike**: Unusual request volume (>300% of baseline)
- **Security Threat**: Suspicious usage patterns
- **System Health**: Infrastructure issues

**Notification Channels**:
- **Email**: SMTP with HTML formatting and attachments
- **Slack**: Rich messages with action buttons
- **Webhook**: JSON payloads to external systems
- **SMS**: Text messages via Twilio/AWS SNS
- **Discord**: Gaming-focused team notifications
- **Microsoft Teams**: Enterprise chat notifications
- **PagerDuty**: On-call escalation for critical issues
- **Log Only**: Structured logging for monitoring systems

**Alert Escalation**:
```
Initial Alert → Email + Slack
     ↓ (10 minutes, no acknowledgment)
First Escalation → SMS + PagerDuty
     ↓ (30 minutes, no resolution) 
Final Escalation → Manager notification + Emergency protocols
```

**Alert Threshold Configuration**:
```python
AlertThreshold(
    threshold_id="daily_budget_80",
    name="Daily Budget Warning",
    alert_type=AlertType.BUDGET_THRESHOLD,
    severity=AlertSeverity.WARNING,
    metric_name="budget_usage_percentage",
    operator=">=",
    value=80.0,
    time_window=timedelta(minutes=5),
    consecutive_breaches=1,
    cooldown_period=timedelta(minutes=15),
    notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
)
```

## Integration Guide

### Basic Integration

```python
from ai_providers.comprehensive_usage_system import ComprehensiveUsageSystem
from ai_providers.models import AIRequest, ProviderType

async def main():
    # Initialize system
    async with ComprehensiveUsageSystem() as system:
        # Create user
        user_id = await system.create_user(
            username="john_doe",
            email="john@company.com",
            user_tier="premium",
            daily_limit=50.0,
            monthly_limit=500.0
        )
        
        # Process request
        request = AIRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=100
        )
        
        allowed, enforcement, metadata = await system.process_ai_request(
            request, ProviderType.OPENAI, user_id
        )
        
        if allowed:
            # Make actual API call here
            response = await call_openai_api(request)
            
            # Record response
            await system.record_ai_response(
                request, response, ProviderType.OPENAI, user_id, success=True
            )
```

### Advanced Integration with MCP Server

```python
class MCPServerWithUsageTracking:
    def __init__(self):
        self.usage_system = None
    
    async def initialize(self):
        self.usage_system = await create_usage_system(
            storage_path="./data/usage",
            enable_all_features=True
        )
    
    async def handle_ai_request(self, request_data: dict) -> dict:
        # Extract request details
        user_id = request_data.get("user_id", "anonymous")
        messages = request_data.get("messages", [])
        model = request_data.get("model", "gpt-3.5-turbo")
        
        # Create AI request
        ai_request = AIRequest(
            model=model,
            messages=messages,
            max_tokens=request_data.get("max_tokens"),
            temperature=request_data.get("temperature"),
            tools=request_data.get("tools")
        )
        
        # Check budget and get routing recommendation
        allowed, enforcement, metadata = await self.usage_system.process_ai_request(
            ai_request, ProviderType.OPENAI, user_id
        )
        
        if not allowed:
            return {
                "error": "Request denied due to budget limits",
                "reason": enforcement.reason,
                "suggested_action": enforcement.warning_message
            }
        
        # Use alternative if suggested
        if enforcement and enforcement.modified_request:
            ai_request = enforcement.modified_request
            provider = enforcement.suggested_provider or ProviderType.OPENAI
        else:
            provider = ProviderType.OPENAI
        
        try:
            # Make API call
            response = await self.call_provider_api(ai_request, provider)
            
            # Record successful response
            await self.usage_system.record_ai_response(
                ai_request, response, provider, user_id, success=True
            )
            
            return {
                "content": response.content,
                "usage": response.usage,
                "cost": response.cost,
                "metadata": metadata
            }
            
        except Exception as e:
            # Record failed response
            await self.usage_system.record_ai_response(
                ai_request, None, provider, user_id, 
                success=False, error_message=str(e)
            )
            
            return {"error": str(e)}
```

## Configuration

### Environment Variables

```bash
# Database settings
CHROMA_DB_PATH=./data/chromadb
CHROMA_COLLECTION_PREFIX=mdmai_

# Storage settings
USAGE_STORAGE_PATH=./data/usage
ENABLE_CHROMADB=true

# Budget enforcement
ENABLE_BUDGET_ENFORCEMENT=true
DEFAULT_USER_TIER=free
ENABLE_GRACEFUL_DEGRADATION=true

# Analytics
ENABLE_ANALYTICS_DASHBOARD=true
DASHBOARD_REFRESH_INTERVAL=60
METRICS_RETENTION_DAYS=90

# Alerting
ENABLE_ALERTING=true
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USERNAME=alerts@company.com
EMAIL_PASSWORD=your_app_password
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Cost optimization
ENABLE_COST_OPTIMIZATION=true
MIN_OPTIMIZATION_SAVINGS=5.0
OPTIMIZATION_CONFIDENCE_THRESHOLD=0.7
```

### JSON Configuration File

```json
{
  "system": {
    "storage_base_path": "./data",
    "use_chromadb": true,
    "async_processing": true,
    "max_concurrent_operations": 10
  },
  "budget_enforcement": {
    "enabled": true,
    "default_policies": {
      "free": {
        "daily_limit": 5.0,
        "monthly_limit": 50.0,
        "per_request_limit": 0.50
      },
      "premium": {
        "daily_limit": 25.0,
        "monthly_limit": 250.0,
        "per_request_limit": 2.0
      },
      "enterprise": {
        "daily_limit": 100.0,
        "monthly_limit": 1000.0,
        "per_request_limit": 10.0
      }
    }
  },
  "alerting": {
    "enabled": true,
    "channels": {
      "email": {
        "host": "smtp.gmail.com",
        "port": 587,
        "username": "${EMAIL_USERNAME}",
        "password": "${EMAIL_PASSWORD}",
        "recipients": ["admin@company.com"]
      },
      "slack": {
        "webhook_url": "${SLACK_WEBHOOK_URL}",
        "channel": "#ai-alerts"
      }
    }
  }
}
```

## Performance Characteristics

### Latency Targets
- Request processing: < 50ms P95
- Budget enforcement check: < 10ms P95
- Cost calculation: < 5ms P95
- Metrics collection: < 1ms P95

### Throughput Capacity
- Concurrent requests: 1000+/second
- Users supported: 10,000+ active
- Daily requests: 1M+ processed
- Storage growth: ~100MB/month per 1000 active users

### Resource Requirements
- Memory: 512MB base + 1MB per 1000 active users
- Storage: 1GB base + 100MB/month per 1000 users
- CPU: 2 cores minimum, 4+ recommended for production
- Network: 100MB/month per 1000 users for metrics/alerts

## Monitoring and Observability

### Key Metrics to Monitor
- **Usage Metrics**: requests/minute, tokens/second, cost/hour
- **Performance Metrics**: latency P50/P95/P99, error rate, throughput
- **Budget Metrics**: spending rate, budget utilization, enforcement actions
- **System Metrics**: memory usage, disk usage, background task health

### Health Check Endpoint
```python
async def health_check():
    health = await usage_system.get_system_health()
    return {
        "status": health["status"],
        "uptime": health["uptime_seconds"],
        "components": health["components"],
        "active_alerts": len(health.get("alerts", []))
    }
```

### Log Structure
```json
{
  "timestamp": "2024-01-15T14:30:00Z",
  "level": "INFO",
  "component": "budget_enforcer",
  "event": "request_processed",
  "user_id": "user_abc123",
  "request_id": "req_xyz789",
  "provider": "openai",
  "model": "gpt-3.5-turbo",
  "cost": 0.0025,
  "tokens": {"input": 15, "output": 12},
  "action": "allowed",
  "processing_time_ms": 45
}
```

## Security Considerations

### Data Privacy
- User data encrypted at rest using AES-256
- API keys stored in secure environment variables
- PII data can be excluded from analytics
- GDPR-compliant data export and deletion

### Access Control
- Role-based access to different dashboard levels
- API key rotation supported
- Audit logging for all administrative actions
- Rate limiting to prevent abuse

### Compliance Features
- Data retention policies configurable
- Export capabilities for compliance audits
- Anonymization options for analytics
- Secure backup and recovery procedures

## Troubleshooting Guide

### Common Issues

**1. High Memory Usage**
- Check metrics collector buffer sizes
- Reduce cache sizes if memory constrained
- Enable compression for old data
- Consider external storage for large datasets

**2. Budget Enforcement Not Working**
- Verify user limits are configured
- Check policy enablement status
- Review threshold configurations
- Validate cost calculation accuracy

**3. Alerts Not Being Sent**
- Test notification channels individually
- Check rate limiting settings
- Verify webhook URLs and credentials
- Review alert threshold configurations

**4. Performance Issues**
- Monitor background task health
- Check database connection status
- Review aggregation intervals
- Consider scaling storage backend

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Create system with debug configuration
config = SystemConfiguration(
    enable_token_estimation_cache=False,  # Disable for debugging
    metrics_aggregation_interval=30,       # More frequent updates
    async_processing=False                 # Synchronous for debugging
)

system = ComprehensiveUsageSystem(config)
```

## Future Enhancements

### Planned Features
- **ML-based Anomaly Detection**: Advanced pattern recognition for unusual usage
- **Predictive Cost Modeling**: Forecast future costs based on usage trends
- **Multi-tenant Architecture**: Support for organizational hierarchies
- **Advanced Caching**: Smart caching with semantic similarity
- **Real-time Dashboards**: WebSocket-based live updates
- **Mobile Notifications**: Push notifications for critical alerts
- **API Gateway Integration**: Direct integration with popular API gateways
- **Multi-region Support**: Geographic cost optimization

### Integration Roadmap
- **Kubernetes Operator**: Native K8s deployment and management
- **Prometheus Metrics**: Native Prometheus metric export
- **Grafana Dashboards**: Pre-built dashboard templates
- **Datadog Integration**: Native APM and monitoring
- **AWS CloudWatch**: Direct metric publishing
- **Terraform Modules**: Infrastructure as code templates

## Support and Maintenance

### Regular Maintenance Tasks
- Weekly: Review alert thresholds and notification channels
- Monthly: Analyze cost optimization recommendations
- Quarterly: Update pricing models and user tier policies
- Annually: Review data retention policies and compliance requirements

### Backup Procedures
- Daily automated backups of user data and configurations
- Weekly full system backups with retention testing
- Monthly disaster recovery testing
- Quarterly backup restoration validation

### Update Process
1. Test updates in staging environment
2. Review breaking changes and migration requirements
3. Backup current system state
4. Apply updates during maintenance window
5. Validate system health and functionality
6. Monitor for 24 hours post-update

---

*This documentation covers the comprehensive usage tracking and cost management system. For implementation support or questions, please refer to the code comments and examples provided in each module.*