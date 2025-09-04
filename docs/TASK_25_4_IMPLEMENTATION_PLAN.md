# Task 25.4: Implement Usage Tracking and Cost Management - Implementation Plan

## Executive Summary

This document outlines the comprehensive implementation plan for Task 25.4: Implement Usage Tracking and Cost Management, a critical component of the MDMAI system's Phase 25: LLM Provider Authentication Enhancement. This task focuses on intelligent cost management, usage tracking, and budget enforcement across multiple AI providers.

## 1. Task Overview

### 1.1 Requirements
- **REQ-013**: LLM Provider Authentication and Integration
- **REQ-020**: Cost Optimization for AI Providers

### 1.2 Current Status
- **Status**: PLANNED → READY FOR DEV
- **Dependencies**: Task 25.1 (Secure Credential Management), Task 25.2 (Provider Authentication Layer)
- **Integration Points**: Existing cost_optimizer.py, token_estimator.py, provider management system

### 1.3 Success Criteria
- Real-time cost tracking with < 50ms latency
- Per-user usage analytics with persistence
- Budget enforcement with multiple limit types
- Cost optimization recommendations
- 95%+ cost calculation accuracy across providers

## 2. Architecture Overview

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Usage Tracking System                    │
├─────────────────────┬─────────────────────┬─────────────────┤
│   Token Counting    │  Cost Calculation   │   Usage Storage │
│      System         │      Engine         │     Manager     │
├─────────────────────┼─────────────────────┼─────────────────┤
│ • Provider-specific │ • Real-time pricing │ • Local JSON    │
│   token estimation  │ • Multi-currency    │ • ChromaDB      │
│ • Streaming support │ • Volume discounts  │ • Hybrid cache  │
│ • Tool/func calls   │ • Time-based rates  │ • Session mgmt  │
└─────────────────────┴─────────────────────┴─────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────────┐
│          Budget Management & Alerting System                │
├─────────────────────────────┼─────────────────────────────────┤
│   • Hard/Soft/Adaptive limits                               │
│   • Daily/Weekly/Monthly caps                               │
│   • Emergency circuit breakers                              │
│   • Multi-channel notifications (email, webhook, in-app)    │
│   • Trend-based alerts and cost predictions                 │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Architecture

```
Request → Token Estimation → Cost Calculation → Usage Recording → Analytics Update
    ↓           ↓                    ↓                ↓               ↓
Budget Check → Provider Selection → Response Processing → Alert Check → Dashboard
```

## 3. Detailed Component Design

### 3.1 Enhanced Token Counting System

#### Implementation: `src/ai_providers/enhanced_usage_tracker.py`

**Core Features:**
- **Provider-Specific Counting**: Accurate token estimation using tiktoken (OpenAI), custom heuristics (Anthropic, Google)
- **Multimodal Support**: Separate tracking for text, images, audio, video content
- **Tool Call Accounting**: Proper token counting for function/tool invocations
- **Streaming Accumulation**: Real-time token counting for streaming responses
- **LRU Caching**: 1024-entry cache for performance optimization

**Key Classes:**
```python
class AdvancedTokenCounter:
    async def count_tokens(
        self, 
        content: Union[str, List, Dict], 
        provider: ProviderType,
        model: str,
        content_type: ContentType = ContentType.TEXT
    ) -> TokenCount

class TokenCount:
    text_tokens: int
    tool_tokens: int
    image_tokens: int
    audio_tokens: int
    video_tokens: int
    cached_tokens: int
    total_tokens: int
```

### 3.2 Real-Time Cost Calculation Engine

#### Implementation: `src/cost_optimization/pricing_engine.py`

**Features:**
- **Dynamic Pricing Models**: Real-time pricing updates per provider/model
- **Input/Output Differentiation**: Separate pricing for prompt vs completion tokens
- **Volume Discounts**: Automatic application based on usage tiers
- **Multi-Currency Support**: USD, EUR, GBP, JPY, CNY with exchange rates
- **Time-Based Pricing**: Peak/off-peak rate support

**Pricing Structure:**
```python
@dataclass
class ModelPricing:
    model_name: str
    input_price_per_1k: Decimal
    output_price_per_1k: Decimal
    cached_input_discount: float = 0.5
    volume_tiers: List[VolumeTier] = field(default_factory=list)
    peak_multiplier: float = 1.0
    currency: str = "USD"
```

### 3.3 Per-User Usage Tracking Schema

#### Storage Options:

**Local JSON Files:**
```json
{
  "user_profiles": {
    "user_123": {
      "total_cost": "45.67",
      "total_requests": 1234,
      "providers": {
        "anthropic": {"cost": "23.45", "requests": 567},
        "openai": {"cost": "22.22", "requests": 667}
      },
      "daily_usage": {
        "2025-09-03": {"cost": "5.67", "requests": 89}
      }
    }
  }
}
```

**ChromaDB Collections:**
- **usage_records**: Individual request records with embeddings for pattern analysis
- **user_profiles**: Aggregated user statistics
- **session_data**: Session-specific usage tracking
- **cost_analytics**: Pre-computed analytics data

### 3.4 Spending Limits and Budget Enforcement

#### Budget Types:
```python
class BudgetLevel(Enum):
    HARD = "hard"        # Block requests when exceeded
    SOFT = "soft"        # Warn but continue
    ADAPTIVE = "adaptive" # Reduce quality/switch models

class BudgetPeriod(Enum):
    REQUEST = "request"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
```

#### Enforcement Mechanisms:
- **Pre-request Budget Checks**: Validate before making API calls
- **Graceful Degradation**: Automatic model downgrade when approaching limits
- **Emergency Circuit Breakers**: Hard stops for critical budget violations
- **Alert Thresholds**: Configurable alerts at 50%, 75%, 90% of budget

### 3.5 Usage Analytics Dashboard Data Model

#### Analytics Schema:
```python
@dataclass
class UsageAnalytics:
    summary: UsageSummary
    provider_breakdown: Dict[str, ProviderMetrics]
    model_breakdown: Dict[str, ModelMetrics]
    time_series: List[TimeSeriesPoint]
    cost_trends: CostTrendAnalysis
    optimization_recommendations: List[OptimizationSuggestion]
```

#### Aggregation Strategy:
- **Real-time**: Current session and last 24 hours
- **Hourly**: Last 7 days with hour-level granularity
- **Daily**: Last 30 days with day-level aggregation
- **Monthly**: Last 12 months for long-term trends

### 3.6 Cost Optimization Algorithms

#### Implementation: `src/cost_optimization/advanced_optimizer.py`

**Optimization Strategies:**
1. **Provider Selection**: ML-based routing considering cost, quality, latency
2. **Model Selection**: Task-appropriate model matching with cost consideration
3. **Request Batching**: Combine similar requests for efficiency
4. **Context Compression**: Intelligent context pruning to reduce token usage
5. **Caching**: Semantic caching for similar requests

**Key Algorithms:**
```python
class CostOptimizer:
    async def optimize_request(
        self, 
        request: AIRequest, 
        budget_context: BudgetContext
    ) -> OptimizationStrategy

class OptimizationStrategy:
    provider: ProviderType
    model: str
    context_compression: float  # 0.0-1.0
    batch_eligible: bool
    estimated_cost: Decimal
    confidence_score: float
```

## 4. Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
**Files to Create/Modify:**
- `src/ai_providers/enhanced_usage_tracker.py`
- `src/usage_tracking/storage/models.py`
- `src/usage_tracking/storage/json_storage.py`
- `tests/test_enhanced_usage_tracker.py`

**Deliverables:**
- Enhanced token counting system
- Basic usage recording functionality
- Local JSON storage implementation
- Unit tests for core components

### Phase 2: Cost Calculation & Analytics (Week 1-2)
**Files to Create/Modify:**
- `src/cost_optimization/pricing_engine.py`
- `src/usage_tracking/analytics/time_series.py`
- `src/cost_optimization/advanced_optimizer.py`
- `examples/enhanced_usage_tracking_demo.py`

**Deliverables:**
- Real-time cost calculation engine
- Usage analytics and trending
- Cost optimization algorithms
- Comprehensive demo application

### Phase 3: Budget Enforcement (Week 2)
**Files to Create/Modify:**
- `src/cost_optimization/budget_enforcer.py`
- `src/cost_optimization/alert_system.py`
- `src/usage_tracking/storage/hybrid_storage.py`
- `docs/ENHANCED_USAGE_TRACKING.md`

**Deliverables:**
- Budget management system
- Multi-channel alerting
- Hybrid storage implementation
- Complete documentation

### Phase 4: Integration & Testing (Week 2-3)
**Files to Create/Modify:**
- `src/usage_tracking/storage_manager.py`
- `tests/test_cost_optimization_complete.py`
- `src/usage_tracking/__init__.py`
- Integration with existing provider management

**Deliverables:**
- Complete system integration
- Comprehensive test suite
- Performance benchmarks
- Production readiness validation

## 5. Provider Cost Formulas

### 5.1 Anthropic Claude
```python
ANTHROPIC_PRICING = {
    "claude-3-5-sonnet-20241022": ModelPricing(
        model_name="claude-3-5-sonnet-20241022",
        input_price_per_1k=Decimal("3.00"),
        output_price_per_1k=Decimal("15.00"),
        cached_input_discount=0.9,  # 90% discount for cached content
    ),
    "claude-3-5-haiku-20241022": ModelPricing(
        model_name="claude-3-5-haiku-20241022", 
        input_price_per_1k=Decimal("1.00"),
        output_price_per_1k=Decimal("5.00"),
        cached_input_discount=0.9,
    )
}
```

### 5.2 OpenAI GPT
```python
OPENAI_PRICING = {
    "gpt-4o": ModelPricing(
        model_name="gpt-4o",
        input_price_per_1k=Decimal("2.50"),
        output_price_per_1k=Decimal("10.00"),
        cached_input_discount=0.5,  # 50% discount for cached content
    ),
    "gpt-4o-mini": ModelPricing(
        model_name="gpt-4o-mini",
        input_price_per_1k=Decimal("0.15"),
        output_price_per_1k=Decimal("0.60"),
        cached_input_discount=0.5,
    )
}
```

### 5.3 Google Gemini
```python
GOOGLE_PRICING = {
    "gemini-1.5-pro": ModelPricing(
        model_name="gemini-1.5-pro",
        input_price_per_1k=Decimal("1.25"),
        output_price_per_1k=Decimal("5.00"),
        cached_input_discount=0.875,  # 87.5% discount for cached content
    ),
    "gemini-1.5-flash": ModelPricing(
        model_name="gemini-1.5-flash", 
        input_price_per_1k=Decimal("0.075"),
        output_price_per_1k=Decimal("0.30"),
        cached_input_discount=0.875,
    )
}
```

## 6. Metrics Collection and Retention Policies

### 6.1 Data Retention Schedule
- **Raw Usage Records**: 90 days in active storage, 1 year archived
- **Hourly Aggregates**: 30 days active, 6 months archived
- **Daily Aggregates**: 1 year active, 5 years archived
- **Monthly Aggregates**: 5 years active, permanent archive
- **User Profiles**: Permanent retention with anonymization after 2 years inactive

### 6.2 Performance Metrics
```python
@dataclass
class PerformanceMetrics:
    avg_request_latency: float        # Target: < 50ms
    cost_calculation_time: float      # Target: < 10ms
    storage_write_latency: float      # Target: < 100ms
    cache_hit_rate: float            # Target: > 80%
    budget_check_time: float         # Target: < 25ms
```

### 6.3 Business Metrics
```python
@dataclass
class BusinessMetrics:
    total_cost_savings: Decimal       # From optimization
    average_cost_per_request: Decimal
    provider_distribution: Dict[str, float]
    model_efficiency_ratio: Dict[str, float]
    budget_adherence_rate: float      # % requests within budget
```

## 7. Alerting System Design

### 7.1 Alert Types and Thresholds
```python
class AlertType(Enum):
    BUDGET_WARNING = "budget_warning"     # 75% of budget used
    BUDGET_CRITICAL = "budget_critical"   # 90% of budget used
    BUDGET_EXCEEDED = "budget_exceeded"   # 100% of budget used
    COST_ANOMALY = "cost_anomaly"        # Unusual spending pattern
    PROVIDER_FAILURE = "provider_failure" # Provider unavailable
    TOKEN_SPIKE = "token_spike"          # Unusual token usage
```

### 7.2 Notification Channels
- **In-App**: Real-time dashboard notifications
- **Email**: Configurable SMTP with templates
- **Webhook**: HTTP POST to configured endpoints
- **SMS**: Twilio integration for critical alerts

### 7.3 Alert Rules Engine
```python
@dataclass
class AlertRule:
    name: str
    condition: str                    # Python expression
    threshold: Union[float, Decimal]
    severity: AlertSeverity
    channels: List[NotificationChannel]
    cooldown_minutes: int = 60
    enabled: bool = True
```

## 8. Security and Compliance

### 8.1 Data Protection
- **Encryption**: AES-256 for sensitive cost data
- **Access Control**: User-specific data isolation
- **Audit Logging**: All cost-related operations logged
- **Data Anonymization**: PII removal from analytics

### 8.2 Compliance Features
- **GDPR**: Right to deletion and data export
- **SOX**: Immutable audit trails for financial data
- **PCI**: Secure handling of payment-related information
- **HIPAA**: Data segregation for healthcare customers

## 9. Performance Optimization

### 9.1 Caching Strategy
- **L1 Cache**: In-memory LRU (1000 entries, 5-minute TTL)
- **L2 Cache**: Redis distributed cache (10000 entries, 1-hour TTL)  
- **L3 Cache**: Persistent disk cache (unlimited, 24-hour TTL)

### 9.2 Database Optimization
- **Indexing**: Composite indexes on user_id + timestamp
- **Partitioning**: Time-based partitioning for usage records
- **Compression**: GZIP compression for historical data
- **Read Replicas**: Separate analytics queries from writes

### 9.3 Query Optimization
- **Prepared Statements**: Parameterized queries for performance
- **Connection Pooling**: Reuse database connections
- **Batch Processing**: Group similar operations
- **Lazy Loading**: Load data only when needed

## 10. Testing Strategy

### 10.1 Unit Tests (Target: 90% Coverage)
- Token counting accuracy for all providers
- Cost calculation precision
- Budget enforcement logic
- Storage operations
- Alert generation

### 10.2 Integration Tests
- End-to-end usage tracking workflows
- Provider integration validation
- Storage backend compatibility
- Analytics calculation accuracy

### 10.3 Performance Tests
- Load testing with 1000+ concurrent users
- Latency benchmarks for all operations
- Memory usage profiling
- Storage scalability testing

### 10.4 Security Tests
- Data encryption validation
- Access control enforcement
- Audit trail completeness
- Vulnerability scanning

## 11. Deployment Strategy

### 11.1 Environment Configuration
```yaml
# config/usage_tracking.yaml
storage:
  type: "hybrid"  # json, chromadb, hybrid
  json_path: "./data/usage"
  chromadb_host: "localhost:8000"
  
budgets:
  default_daily_limit: 10.00
  default_monthly_limit: 300.00
  enforcement_level: "soft"
  
alerts:
  email_enabled: true
  webhook_url: "https://api.company.com/webhooks/costs"
  
optimization:
  enabled: true
  cache_size: 1000
  compression_threshold: 0.8
```

### 11.2 Migration from Existing System
1. **Data Export**: Extract current usage data from existing cost_optimizer.py
2. **Schema Migration**: Convert to new enhanced format
3. **Parallel Running**: Run both systems during transition
4. **Validation**: Compare results for accuracy
5. **Cutover**: Switch to new system with rollback plan

### 11.3 Monitoring and Observability
- **Health Checks**: System status endpoints
- **Metrics Export**: Prometheus-compatible metrics
- **Distributed Tracing**: OpenTelemetry integration
- **Error Tracking**: Structured error reporting

## 12. Risk Assessment and Mitigation

### 12.1 Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Token counting accuracy | Medium | High | Extensive testing, fallback estimation |
| Performance degradation | Low | Medium | Caching, optimization, load testing |
| Data corruption | Low | High | Backups, validation, checksums |
| Provider API changes | Medium | Medium | Version pinning, adapter patterns |

### 12.2 Business Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Cost tracking errors | Low | High | Validation, reconciliation, alerts |
| Budget overruns | Medium | High | Real-time monitoring, hard limits |
| Data privacy breach | Low | Critical | Encryption, access controls, audits |
| System unavailability | Low | Medium | Redundancy, failover, monitoring |

## 13. Success Metrics

### 13.1 Technical Metrics
- **Accuracy**: > 99% cost calculation accuracy
- **Performance**: < 50ms P95 latency for usage tracking
- **Reliability**: > 99.9% system uptime
- **Scalability**: Support 10,000+ users per instance

### 13.2 Business Metrics
- **Cost Savings**: 15-30% reduction in AI provider costs
- **Budget Compliance**: > 95% of users stay within budgets
- **User Adoption**: > 80% of users enable usage tracking
- **Cost Visibility**: Real-time cost data for all requests

## 14. Future Enhancements

### 14.1 Machine Learning Integration
- **Usage Prediction**: ML models for cost forecasting
- **Anomaly Detection**: AI-powered unusual pattern detection
- **Optimization Recommendations**: Smart cost-saving suggestions
- **Dynamic Budgeting**: Automatic budget adjustment based on patterns

### 14.2 Advanced Analytics
- **Cohort Analysis**: User behavior segmentation
- **ROI Tracking**: Value measurement for AI usage
- **Competitive Benchmarking**: Cost comparison across similar users
- **Predictive Scaling**: Automatic resource allocation

### 14.3 Enterprise Features
- **Multi-tenant Management**: Organization-level controls
- **Advanced Reporting**: Custom dashboards and exports
- **API Integration**: Third-party cost management tools
- **Compliance Automation**: Automated audit trail generation

## 15. Conclusion

Task 25.4 represents a comprehensive upgrade to the MDMAI system's cost management capabilities, providing enterprise-grade usage tracking, budget enforcement, and cost optimization. The implementation follows a phased approach ensuring minimal disruption while delivering immediate value.

**Key Benefits:**
- **Cost Control**: Real-time budget enforcement with multiple limit types
- **Visibility**: Comprehensive analytics and dashboards
- **Optimization**: AI-powered cost reduction recommendations
- **Scalability**: Architecture supports growth from individual users to enterprise

**Implementation Timeline:** 2-3 weeks with parallel development streams
**Resource Requirements:** 1 senior developer, testing resources, documentation
**Risk Level:** Low to Medium with proper testing and gradual rollout

---

**Document Status**: COMPLETE  
**Last Updated**: 2025-09-03  
**Author**: MDMAI Development Team  
**Version**: 1.0