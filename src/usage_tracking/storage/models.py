"""
Storage models and schemas for usage tracking and cost management.

This module defines the data models for:
- Usage records and analytics
- User profiles and spending limits
- Cost tracking and budgets
- Performance metrics
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class StorageType(str, Enum):
    """Storage backend types."""
    CHROMADB = "chromadb"
    JSON = "json"
    HYBRID = "hybrid"


class UsageEventType(str, Enum):
    """Types of usage events to track."""
    API_CALL = "api_call"
    TOKEN_USAGE = "token_usage"
    CONTEXT_ACCESS = "context_access"
    CHARACTER_GENERATION = "character_generation"
    PDF_PROCESSING = "pdf_processing"
    SEARCH_QUERY = "search_query"
    EMBEDDING_GENERATION = "embedding_generation"
    SESSION_START = "session_start"
    SESSION_END = "session_end"


class ProviderType(str, Enum):
    """AI provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    LOCAL = "local"


class CostType(str, Enum):
    """Types of costs to track."""
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    EMBEDDING_TOKENS = "embedding_tokens"
    API_REQUEST = "api_request"
    STORAGE = "storage"
    COMPUTE = "compute"


class TimeAggregation(str, Enum):
    """Time aggregation periods for analytics."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


# Core Usage Models

class UsageRecord(BaseModel):
    """Individual usage record for vector storage."""
    
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Event details
    event_type: UsageEventType
    provider: ProviderType
    model_name: Optional[str] = None
    
    # Quantitative metrics
    token_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    
    # Cost tracking
    cost_usd: Decimal = Decimal("0.00")
    cost_breakdown: Dict[CostType, Decimal] = Field(default_factory=dict)
    
    # Context information
    context_id: Optional[str] = None
    operation: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    
    # Performance metrics
    duration_ms: Optional[int] = None
    latency_ms: Optional[int] = None
    
    # Metadata for vector search
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('cost_usd')
    @classmethod
    def validate_cost(cls, v):
        """Ensure cost is non-negative."""
        if v < 0:
            raise ValueError("Cost cannot be negative")
        return v
    
    @model_validator(mode='before')
    @classmethod
    def validate_tokens(cls, values):
        """Validate token counts are consistent."""
        if isinstance(values, dict):
            token_count = values.get('token_count', 0)
            input_tokens = values.get('input_tokens', 0)
            output_tokens = values.get('output_tokens', 0)
            
            if token_count == 0 and (input_tokens > 0 or output_tokens > 0):
                values['token_count'] = input_tokens + output_tokens
        
        return values


class UsagePattern(BaseModel):
    """Analyzed usage pattern for embeddings."""
    
    pattern_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    pattern_type: str  # "daily_peak", "weekend_usage", "cost_spike", etc.
    
    # Pattern characteristics
    frequency_score: float = Field(ge=0.0, le=1.0)
    cost_impact: Decimal = Decimal("0.00")
    efficiency_score: float = Field(ge=0.0, le=1.0)
    
    # Time characteristics
    start_time: datetime
    end_time: datetime
    duration_hours: float
    
    # Associated records
    record_count: int = 0
    avg_cost_per_record: Decimal = Decimal("0.00")
    
    # Pattern insights
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Embedding metadata
    embedding_model: str = "all-MiniLM-L6-v2"
    last_analyzed: datetime = Field(default_factory=datetime.utcnow)


# User Profile and Budget Models

class SpendingLimit(BaseModel):
    """User spending limits."""
    
    limit_type: str  # "daily", "weekly", "monthly", "total"
    amount_usd: Decimal
    current_spent: Decimal = Decimal("0.00")
    reset_date: Optional[datetime] = None
    alert_threshold: float = 0.8  # Alert at 80% of limit
    hard_limit: bool = True  # Stop usage when exceeded
    
    @property
    def remaining(self) -> Decimal:
        """Calculate remaining budget."""
        return max(Decimal("0.00"), self.amount_usd - self.current_spent)
    
    @property
    def percentage_used(self) -> float:
        """Calculate percentage of limit used."""
        if self.amount_usd == 0:
            return 0.0
        return float(self.current_spent / self.amount_usd)
    
    @property
    def is_exceeded(self) -> bool:
        """Check if limit is exceeded."""
        return self.current_spent >= self.amount_usd


class UserProfile(BaseModel):
    """User profile with usage preferences and limits."""
    
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Profile information
    username: Optional[str] = None
    email: Optional[str] = None
    timezone: str = "UTC"
    
    # Spending limits
    spending_limits: Dict[str, SpendingLimit] = Field(default_factory=dict)
    
    # Usage preferences
    preferred_providers: List[ProviderType] = Field(default_factory=list)
    cost_optimization: bool = True
    usage_alerts: bool = True
    detailed_tracking: bool = True
    
    # Statistics (cached from usage data)
    total_spent: Decimal = Decimal("0.00")
    total_tokens: int = 0
    total_requests: int = 0
    avg_cost_per_request: Decimal = Decimal("0.00")
    
    # Usage patterns
    peak_hours: List[int] = Field(default_factory=list)  # Hours 0-23
    most_used_provider: Optional[ProviderType] = None
    most_used_model: Optional[str] = None
    
    @property
    def has_active_limits(self) -> bool:
        """Check if user has any active spending limits."""
        return len(self.spending_limits) > 0
    
    def get_limit(self, limit_type: str) -> Optional[SpendingLimit]:
        """Get spending limit by type."""
        return self.spending_limits.get(limit_type)
    
    def is_within_limits(self, additional_cost: Decimal) -> Dict[str, bool]:
        """Check if additional cost would exceed any limits."""
        results = {}
        for limit_type, limit in self.spending_limits.items():
            would_exceed = (limit.current_spent + additional_cost) > limit.amount_usd
            results[limit_type] = not would_exceed
        return results


# Analytics Models

class UsageMetrics(BaseModel):
    """Aggregated usage metrics."""
    
    metric_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None  # None for global metrics
    
    # Time period
    period_start: datetime
    period_end: datetime
    aggregation_type: TimeAggregation
    
    # Usage statistics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    
    # Cost statistics
    total_cost: Decimal = Decimal("0.00")
    avg_cost_per_request: Decimal = Decimal("0.00")
    cost_by_provider: Dict[str, Decimal] = Field(default_factory=dict)
    cost_by_model: Dict[str, Decimal] = Field(default_factory=dict)
    
    # Performance statistics
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_duration_ms: float = 0.0
    
    # Provider statistics
    requests_by_provider: Dict[str, int] = Field(default_factory=dict)
    tokens_by_provider: Dict[str, int] = Field(default_factory=dict)
    
    # Error statistics
    error_rate: float = 0.0
    errors_by_type: Dict[str, int] = Field(default_factory=dict)
    
    # Calculated at aggregation time
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


class CostBreakdown(BaseModel):
    """Detailed cost breakdown."""
    
    breakdown_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    period_start: datetime
    period_end: datetime
    
    # Cost categories
    input_token_cost: Decimal = Decimal("0.00")
    output_token_cost: Decimal = Decimal("0.00")
    embedding_cost: Decimal = Decimal("0.00")
    api_request_cost: Decimal = Decimal("0.00")
    storage_cost: Decimal = Decimal("0.00")
    
    # Provider breakdown
    provider_costs: Dict[str, Decimal] = Field(default_factory=dict)
    model_costs: Dict[str, Decimal] = Field(default_factory=dict)
    
    # Operations breakdown
    operation_costs: Dict[str, Decimal] = Field(default_factory=dict)
    
    @property
    def total_cost(self) -> Decimal:
        """Calculate total cost."""
        return (
            self.input_token_cost + 
            self.output_token_cost + 
            self.embedding_cost + 
            self.api_request_cost + 
            self.storage_cost
        )


# Storage Configuration Models

class ChromaDBConfig(BaseModel):
    """ChromaDB collection configuration."""
    
    # Connection settings
    host: str = "localhost"
    port: int = 8000
    persist_directory: str = "./data/chromadb"
    
    # Collection settings
    usage_collection: str = "usage_records"
    patterns_collection: str = "usage_patterns"
    analytics_collection: str = "usage_analytics"
    
    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Performance settings
    batch_size: int = 1000
    max_retries: int = 3
    timeout_seconds: int = 30


class JSONStorageConfig(BaseModel):
    """JSON file storage configuration."""
    
    # Storage paths
    base_path: str = "./data/usage_tracking"
    profiles_path: str = "user_profiles"
    transactions_path: str = "transactions"
    analytics_path: str = "analytics"
    backups_path: str = "backups"
    
    # File management
    max_file_size_mb: int = 100
    files_per_directory: int = 1000
    compression: bool = True
    
    # Retention settings
    retention_days: int = 90
    archive_after_days: int = 30
    backup_frequency_hours: int = 24
    
    # Consistency settings
    use_atomic_writes: bool = True
    fsync: bool = True
    create_backups: bool = True


class HybridStorageConfig(BaseModel):
    """Hybrid storage strategy configuration."""
    
    # Storage preferences
    hot_data_storage: StorageType = StorageType.CHROMADB
    warm_data_storage: StorageType = StorageType.JSON
    cold_data_storage: StorageType = StorageType.JSON
    
    # Thresholds
    hot_data_days: int = 7
    warm_data_days: int = 30
    cache_size_mb: int = 500
    
    # Migration settings
    auto_migrate: bool = True
    migration_batch_size: int = 1000
    
    # Consistency settings
    consistency_level: str = "strong"  # "strong", "eventual", "weak"
    sync_frequency_minutes: int = 15
    conflict_resolution: str = "latest_wins"  # "latest_wins", "merge", "manual"


class StorageSchema(BaseModel):
    """Complete storage schema configuration."""
    
    version: str = "1.0.0"
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)
    json_storage: JSONStorageConfig = Field(default_factory=JSONStorageConfig)
    hybrid: HybridStorageConfig = Field(default_factory=HybridStorageConfig)
    
    # Performance settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    write_batch_size: int = 100
    read_batch_size: int = 1000
    
    # Data validation
    validate_on_write: bool = True
    schema_validation: bool = True
    
    # Monitoring
    enable_metrics: bool = True
    log_slow_queries: bool = True
    slow_query_threshold_ms: int = 1000