"""Enhanced data models for usage tracking."""

from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

from ..models import UsageRecord


class BudgetLevel(Enum):
    """Budget enforcement levels."""
    SOFT = "soft"  # Alert only
    HARD = "hard"  # Block requests
    ADAPTIVE = "adaptive"  # Degrade gracefully


class CurrencyCode(Enum):
    """Supported currency codes."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CNY = "CNY"


class UsageGranularity(Enum):
    """Granularity for usage aggregation."""
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


@dataclass
class TokenCountMetrics:
    """Detailed token counting metrics."""
    
    text_tokens: int = 0
    tool_call_tokens: int = 0
    tool_response_tokens: int = 0
    image_tokens: int = 0
    audio_tokens: int = 0
    video_tokens: int = 0
    system_prompt_tokens: int = 0
    cached_tokens: int = 0
    
    @property
    def total_tokens(self) -> int:
        """Calculate total token count."""
        return sum([
            self.text_tokens,
            self.tool_call_tokens,
            self.tool_response_tokens,
            self.image_tokens,
            self.audio_tokens,
            self.video_tokens,
            self.system_prompt_tokens,
        ])
    
    @property
    def billable_tokens(self) -> int:
        """Calculate billable tokens (excluding cached)."""
        return self.total_tokens - self.cached_tokens


@dataclass
class CostBreakdown:
    """Detailed cost breakdown."""
    
    input_cost: Decimal = field(default_factory=Decimal)
    output_cost: Decimal = field(default_factory=Decimal)
    tool_cost: Decimal = field(default_factory=Decimal)
    image_cost: Decimal = field(default_factory=Decimal)
    audio_cost: Decimal = field(default_factory=Decimal)
    cache_discount: Decimal = field(default_factory=Decimal)
    batch_discount: Decimal = field(default_factory=Decimal)
    volume_discount: Decimal = field(default_factory=Decimal)
    currency: CurrencyCode = CurrencyCode.USD
    exchange_rate: Decimal = field(default_factory=lambda: Decimal("1.0"))
    
    @property
    def subtotal(self) -> Decimal:
        """Calculate subtotal before discounts."""
        return sum([
            self.input_cost,
            self.output_cost,
            self.tool_cost,
            self.image_cost,
            self.audio_cost,
        ])
    
    @property
    def total_discount(self) -> Decimal:
        """Calculate total discount amount."""
        return sum([
            self.cache_discount,
            self.batch_discount,
            self.volume_discount,
        ])
    
    @property
    def total_cost(self) -> Decimal:
        """Calculate final cost after discounts."""
        return max(Decimal("0"), self.subtotal - self.total_discount)
    
    def convert_currency(self, target: CurrencyCode, rate: Decimal) -> "CostBreakdown":
        """Convert to different currency."""
        if self.currency == target:
            return self
        
        factor = rate / self.exchange_rate
        return CostBreakdown(
            input_cost=self.input_cost * factor,
            output_cost=self.output_cost * factor,
            tool_cost=self.tool_cost * factor,
            image_cost=self.image_cost * factor,
            audio_cost=self.audio_cost * factor,
            cache_discount=self.cache_discount * factor,
            batch_discount=self.batch_discount * factor,
            volume_discount=self.volume_discount * factor,
            currency=target,
            exchange_rate=rate,
        )


@dataclass
class UserSession:
    """User session tracking."""
    
    session_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: Decimal = field(default_factory=Decimal)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate session duration."""
        if self.ended_at:
            return self.ended_at - self.started_at
        return datetime.now(timezone.utc) - self.started_at
    
    @property
    def is_active(self) -> bool:
        """Check if session is still active."""
        return self.ended_at is None


@dataclass
class EnhancedUsageRecord(UsageRecord):
    """Enhanced usage record with detailed metrics."""
    
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    token_metrics: TokenCountMetrics = field(default_factory=TokenCountMetrics)
    cost_breakdown: CostBreakdown = field(default_factory=CostBreakdown)
    cache_hit_ratio: float = 0.0
    queue_time_ms: float = 0.0
    processing_time_ms: float = 0.0
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            "token_metrics": asdict(self.token_metrics),
            "cost_breakdown": {
                "input_cost": str(self.cost_breakdown.input_cost),
                "output_cost": str(self.cost_breakdown.output_cost),
                "tool_cost": str(self.cost_breakdown.tool_cost),
                "image_cost": str(self.cost_breakdown.image_cost),
                "audio_cost": str(self.cost_breakdown.audio_cost),
                "cache_discount": str(self.cost_breakdown.cache_discount),
                "batch_discount": str(self.cost_breakdown.batch_discount),
                "volume_discount": str(self.cost_breakdown.volume_discount),
                "total_cost": str(self.cost_breakdown.total_cost),
                "currency": self.cost_breakdown.currency.value,
            },
        }


class EnhancedBudget(BaseModel):
    """Enhanced budget configuration with advanced features."""
    
    budget_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    
    # Limits
    per_request_limit: Optional[Decimal] = None
    per_session_limit: Optional[Decimal] = None
    hourly_limit: Optional[Decimal] = None
    daily_limit: Optional[Decimal] = None
    weekly_limit: Optional[Decimal] = None
    monthly_limit: Optional[Decimal] = None
    yearly_limit: Optional[Decimal] = None
    
    # Provider-specific limits
    provider_limits: Dict[str, Decimal] = Field(default_factory=dict)
    model_limits: Dict[str, Decimal] = Field(default_factory=dict)
    
    # Enforcement
    enforcement_level: BudgetLevel = BudgetLevel.SOFT
    alert_thresholds: List[float] = Field(default=[0.5, 0.75, 0.9, 0.95])
    grace_period_minutes: int = 5
    
    # Degradation strategies
    degradation_strategies: List[str] = Field(default_factory=list)
    fallback_models: List[str] = Field(default_factory=list)
    
    # Metadata
    currency: CurrencyCode = CurrencyCode.USD
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    enabled: bool = True
    
    @validator("alert_thresholds")
    def validate_thresholds(cls, v):
        """Ensure thresholds are sorted and valid."""
        return sorted(set(t for t in v if 0 < t < 1))