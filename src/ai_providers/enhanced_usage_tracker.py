"""Enhanced usage tracking and cost management system for MDMAI.

This module provides comprehensive tracking of AI provider usage with advanced
token counting, real-time cost calculation, per-user tracking, budget enforcement,
and analytics capabilities.
"""

import asyncio
import json
import threading
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from functools import wraps
from collections import OrderedDict
from pathlib import Path
from typing import (
    Any, AsyncIterator, Callable, Deque, Dict, Generator, Iterator,
    List, Optional, Tuple, Union
)
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field, validator

from .models import (
    AIRequest, AIResponse, ProviderType, StreamingChunk, UsageRecord
)
from .token_estimator import TokenEstimator
from ..cost_optimization.pricing_engine import get_pricing_engine

logger = structlog.get_logger(__name__)


# ==================== Enhanced Data Models ====================

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


# ==================== Advanced Token Counter ====================

class AdvancedTokenCounter:
    """Advanced token counting with provider-specific accuracy."""

    def __init__(self):
        self._estimator = TokenEstimator()
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._cache_maxsize = 2048  # Bounded cache size
        self._cache_lock = threading.Lock()

        # Provider-specific tokenizers
        self._tokenizers: Dict[ProviderType, Any] = {}
        self._initialize_tokenizers()

    def _initialize_tokenizers(self) -> None:
        """Initialize provider-specific tokenizers."""
        # Try to load tiktoken for OpenAI
        try:
            import tiktoken
            self._tokenizers[ProviderType.OPENAI] = tiktoken
            logger.info("Loaded tiktoken for accurate OpenAI token counting")
        except ImportError:
            logger.debug("tiktoken not available for OpenAI")

        # Try to load Anthropic tokenizer
        try:
            import anthropic  # noqa: F401
            # Note: Anthropic doesn't expose direct tokenizer, use heuristics
            logger.info("Anthropic SDK available for token estimation")
        except ImportError:
            logger.debug("Anthropic SDK not available")

    def _get_cache_key(self, content: str, provider: ProviderType, model: str) -> str:
        """Generate cache key for token counting."""
        # Use first 100 and last 100 chars for key to handle large content
        content_sig = content[:100] + content[-100:] if len(content) > 200 else content
        return f"{provider.value}:{model}:{hash(content_sig)}"

    def count_tokens(
        self,
        content: Union[str, List[Dict[str, Any]], Dict[str, Any]],
        provider: ProviderType,
        model: str,
        cache: bool = True,
    ) -> TokenCountMetrics:
        """Count tokens with detailed metrics.

        Args:
            content: Content to count tokens for
            provider: AI provider type
            model: Model identifier
            cache: Whether to use caching

        Returns:
            Detailed token count metrics
        """
        metrics = TokenCountMetrics()

        # Handle different content types
        if isinstance(content, str):
            metrics.text_tokens = self._count_text_tokens(content, provider, model, cache)
        elif isinstance(content, list):
            metrics = self._count_message_tokens(content, provider, model, cache)
        elif isinstance(content, dict):
            metrics = self._count_structured_tokens(content, provider, model, cache)

        return metrics

    def _count_text_tokens(
        self,
        text: str,
        provider: ProviderType,
        model: str,
        cache: bool = True,
    ) -> int:
        """Count tokens in plain text."""
        if cache:
            cache_key = self._get_cache_key(text, provider, model)
            with self._cache_lock:
                if cache_key in self._cache:
                    # Move to end for LRU
                    self._cache.move_to_end(cache_key)
                    return self._cache[cache_key]

        # Use provider-specific counting
        if provider == ProviderType.OPENAI and ProviderType.OPENAI in self._tokenizers:
            count = self._count_openai_tokens(text, model)
        else:
            count = self._estimator.estimate_tokens(text, provider, model)

        if cache:
            with self._cache_lock:
                # Implement LRU eviction when cache is full
                if len(self._cache) >= self._cache_maxsize and cache_key not in self._cache:
                    self._cache.popitem(last=False)  # Remove oldest item
                self._cache[cache_key] = count
                self._cache.move_to_end(cache_key)  # Mark as most recent

        return count

    def _count_openai_tokens(self, text: str, model: str) -> int:
        """Count tokens using OpenAI's tiktoken."""
        tiktoken = self._tokenizers[ProviderType.OPENAI]

        # Get appropriate encoding
        try:
            if "gpt-4" in model or "gpt-3.5" in model:
                encoding = tiktoken.encoding_for_model(model)
            else:
                encoding = tiktoken.get_encoding("cl100k_base")

            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Failed to use tiktoken: {e}, falling back to estimation")
            return self._estimator.estimate_tokens(text, ProviderType.OPENAI, model)

    def _count_message_tokens(
        self,
        messages: List[Dict[str, Any]],
        provider: ProviderType,
        model: str,
        cache: bool = True,
    ) -> TokenCountMetrics:
        """Count tokens in message list."""
        metrics = TokenCountMetrics()

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            # Handle different content types
            if isinstance(content, str):
                if role == "system":
                    metrics.system_prompt_tokens += self._count_text_tokens(
                        content, provider, model, cache
                    )
                else:
                    metrics.text_tokens += self._count_text_tokens(
                        content, provider, model, cache
                    )
            elif isinstance(content, list):
                # Handle multimodal content
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        if item_type == "text":
                            metrics.text_tokens += self._count_text_tokens(
                                item.get("text", ""), provider, model, cache
                            )
                        elif item_type == "image":
                            metrics.image_tokens += self._estimate_image_tokens(
                                item, provider, model
                            )
                        elif item_type == "audio":
                            metrics.audio_tokens += self._estimate_audio_tokens(
                                item, provider, model
                            )

            # Handle tool calls
            if "tool_calls" in message:
                metrics.tool_call_tokens += self._count_tool_tokens(
                    message["tool_calls"], provider, model, cache
                )

            if "tool_call_id" in message:
                # This is a tool response
                metrics.tool_response_tokens += self._count_text_tokens(
                    str(content), provider, model, cache
                )

        return metrics

    def _count_structured_tokens(
        self,
        data: Dict[str, Any],
        provider: ProviderType,
        model: str,
        cache: bool = True,
    ) -> TokenCountMetrics:
        """Count tokens in structured data."""
        # Convert to JSON string for estimation
        json_str = json.dumps(data, separators=(',', ':'))
        metrics = TokenCountMetrics()
        metrics.text_tokens = self._count_text_tokens(json_str, provider, model, cache)
        return metrics

    def _count_tool_tokens(
        self,
        tool_calls: List[Dict[str, Any]],
        provider: ProviderType,
        model: str,
        cache: bool = True,
    ) -> int:
        """Count tokens in tool calls."""
        total = 0
        for call in tool_calls:
            # Count function name and arguments
            json_str = json.dumps(call, separators=(',', ':'))
            total += self._count_text_tokens(json_str, provider, model, cache)
        return total

    def _estimate_image_tokens(
        self,
        image_data: Dict[str, Any],
        provider: ProviderType,
        model: str,
    ) -> int:
        """Estimate tokens for image content."""
        # Provider-specific image token estimation
        if provider == ProviderType.OPENAI:
            # GPT-4 Vision pricing model
            detail = image_data.get("detail", "auto")
            if detail == "low":
                return 85  # Fixed cost for low detail
            else:
                # High detail: base + tiles
                # Simplified estimation
                return 170 + 85 * 4  # Base + estimated tiles
        elif provider == ProviderType.ANTHROPIC:
            # Claude vision estimation
            return 250  # Approximate
        elif provider == ProviderType.GOOGLE:
            # Gemini vision estimation
            return 258  # Approximate
        return 200  # Default estimation

    def _estimate_audio_tokens(
        self,
        audio_data: Dict[str, Any],
        provider: ProviderType,
        model: str,
    ) -> int:
        """Estimate tokens for audio content."""
        # Simplified estimation based on duration
        duration_seconds = audio_data.get("duration_seconds", 0)
        if duration_seconds:
            # Approximate: 1 second of audio ≈ 50 tokens
            return int(duration_seconds * 50)
        return 1000  # Default estimation

    def count_streaming_tokens(
        self,
        chunks: Iterator[StreamingChunk],
        provider: ProviderType,
        model: str,
    ) -> Generator[TokenCountMetrics, None, TokenCountMetrics]:
        """Count tokens in streaming response.

        Yields token metrics for each chunk and returns total.
        """
        total_metrics = TokenCountMetrics()

        for chunk in chunks:
            chunk_metrics = TokenCountMetrics()

            if chunk.content:
                chunk_metrics.text_tokens = self._count_text_tokens(
                    chunk.content, provider, model, cache=False
                )

            if chunk.tool_calls:
                chunk_metrics.tool_call_tokens = self._count_tool_tokens(
                    chunk.tool_calls, provider, model, cache=False
                )

            total_metrics.text_tokens += chunk_metrics.text_tokens
            total_metrics.tool_call_tokens += chunk_metrics.tool_call_tokens

            yield chunk_metrics

        return total_metrics


# ==================== Real-time Cost Calculator ====================

class CostCalculationEngine:
    """Real-time cost calculation using centralized pricing engine."""

    def __init__(self):
        self._pricing_engine = get_pricing_engine()
        self._exchange_rates: Dict[CurrencyCode, Decimal] = {
            CurrencyCode.USD: Decimal("1.0"),
            CurrencyCode.EUR: Decimal("0.85"),
            CurrencyCode.GBP: Decimal("0.73"),
            CurrencyCode.JPY: Decimal("110.0"),
            CurrencyCode.CNY: Decimal("6.5"),
        }

    def calculate_cost(
        self,
        provider: ProviderType,
        model: str,
        token_metrics: TokenCountMetrics,
        currency: CurrencyCode = CurrencyCode.USD,
        apply_discounts: bool = True,
    ) -> CostBreakdown:
        """Calculate detailed cost breakdown using centralized pricing engine.

        Args:
            provider: AI provider type
            model: Model identifier
            token_metrics: Token count metrics
            currency: Target currency
            apply_discounts: Whether to apply available discounts

        Returns:
            Detailed cost breakdown
        """
        breakdown = CostBreakdown(currency=currency)

        try:
            # Use centralized pricing engine for cost calculation
            total_cost = self._pricing_engine.calculate_simple_cost(
                provider=provider,
                model=model,
                input_tokens=token_metrics.total_tokens,
                output_tokens=token_metrics.text_tokens,
                cached_tokens=token_metrics.cached_tokens,
                tool_call_tokens=token_metrics.tool_call_tokens,
                image_tokens=token_metrics.image_tokens,
                audio_tokens=token_metrics.audio_tokens
            )

            # For backward compatibility, distribute total cost across components
            # This provides the detailed breakdown expected by existing code
            total_tokens = token_metrics.total_tokens

            if total_tokens > 0:
                # Proportionally distribute costs
                input_ratio = (token_metrics.total_tokens - token_metrics.cached_tokens) / total_tokens
                output_ratio = token_metrics.text_tokens / total_tokens
                tool_ratio = token_metrics.tool_call_tokens / total_tokens
                image_ratio = token_metrics.image_tokens / total_tokens
                audio_ratio = token_metrics.audio_tokens / total_tokens

                breakdown.input_cost = total_cost * Decimal(str(input_ratio))
                breakdown.output_cost = total_cost * Decimal(str(output_ratio))
                breakdown.tool_cost = total_cost * Decimal(str(tool_ratio))
                breakdown.image_cost = total_cost * Decimal(str(image_ratio))
                breakdown.audio_cost = total_cost * Decimal(str(audio_ratio))

                # Handle cached tokens discount
                if token_metrics.cached_tokens > 0:
                    cached_ratio = token_metrics.cached_tokens / total_tokens
                    breakdown.cache_discount = total_cost * Decimal(str(cached_ratio)) * Decimal("0.5")

            # Apply discounts if enabled
            if apply_discounts:
                breakdown = self._apply_discounts(breakdown, token_metrics)

            # Convert currency if needed
            if currency != CurrencyCode.USD:
                breakdown = breakdown.convert_currency(currency, self._exchange_rates[currency])

        except Exception as e:
            logger.error(f"Error calculating cost for {provider.value}:{model}: {e}")

        return breakdown


    def _apply_discounts(
        self,
        breakdown: CostBreakdown,
        token_metrics: TokenCountMetrics,
    ) -> CostBreakdown:
        """Apply available discounts."""
        total_tokens = token_metrics.total_tokens

        # Volume discount tiers
        if total_tokens > 100000:
            breakdown.volume_discount = breakdown.subtotal * Decimal("0.1")  # 10% off
        elif total_tokens > 50000:
            breakdown.volume_discount = breakdown.subtotal * Decimal("0.05")  # 5% off
        elif total_tokens > 10000:
            breakdown.volume_discount = breakdown.subtotal * Decimal("0.02")  # 2% off

        return breakdown

    def estimate_batch_cost(
        self,
        requests: List[AIRequest],
        provider: ProviderType,
        model: str,
        token_counter: AdvancedTokenCounter,
    ) -> Tuple[CostBreakdown, List[CostBreakdown]]:
        """Estimate cost for batch of requests.

        Returns total cost and per-request costs.
        """
        individual_costs = []
        total_metrics = TokenCountMetrics()

        for request in requests:
            # Count tokens for request
            metrics = token_counter.count_tokens(request.messages, provider, model)
            total_metrics.text_tokens += metrics.text_tokens
            total_metrics.system_prompt_tokens += metrics.system_prompt_tokens

            # Calculate individual cost
            cost = self.calculate_cost(provider, model, metrics, apply_discounts=False)
            individual_costs.append(cost)

        # Calculate total with batch discount
        total_cost = self.calculate_cost(provider, model, total_metrics)

        # Apply batch processing discount
        if len(requests) > 10:
            total_cost.batch_discount = total_cost.subtotal * Decimal("0.15")  # 15% batch discount
        elif len(requests) > 5:
            total_cost.batch_discount = total_cost.subtotal * Decimal("0.08")  # 8% batch discount

        return total_cost, individual_costs

    def update_exchange_rates(self, rates: Dict[CurrencyCode, Decimal]) -> None:
        """Update currency exchange rates."""
        self._exchange_rates.update(rates)
        logger.info("Updated exchange rates", rates=rates)

    def update_model_pricing(
        self,
        provider: ProviderType,
        model: str,
        pricing: Dict[str, Decimal],
    ) -> None:
        """Update pricing for a specific model."""
        if provider.value not in self._pricing_data:
            self._pricing_data[provider.value] = {}

        self._pricing_data[provider.value][model] = pricing
        logger.info(f"Updated pricing for {provider.value}:{model}")


# ==================== Enhanced Usage Tracker ====================

class EnhancedUsageTracker:
    """Advanced usage tracking with per-user support and persistence."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        token_counter: Optional[AdvancedTokenCounter] = None,
        cost_engine: Optional[CostCalculationEngine] = None,
    ):
        self._storage_path = storage_path or Path.home() / ".mdmai" / "usage"
        self._storage_path.mkdir(parents=True, exist_ok=True)

        self._token_counter = token_counter or AdvancedTokenCounter()
        self._cost_engine = cost_engine or CostCalculationEngine()

        # In-memory storage
        self._usage_records: Deque[EnhancedUsageRecord] = deque(maxlen=10000)
        self._sessions: Dict[str, UserSession] = {}
        self._user_usage: Dict[str, Dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
        self._tenant_usage: Dict[str, Dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))

        # Aggregated metrics
        self._hourly_usage: Dict[str, Decimal] = defaultdict(Decimal)
        self._daily_usage: Dict[str, Decimal] = defaultdict(Decimal)
        self._monthly_usage: Dict[str, Decimal] = defaultdict(Decimal)
        self._provider_usage: Dict[ProviderType, Decimal] = defaultdict(Decimal)

        # Per-user/tenant provider usage tracking
        self._user_provider_usage: Dict[str, Dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
        self._tenant_provider_usage: Dict[str, Dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))

        # Thread safety
        self._lock = asyncio.Lock()
        self._thread_lock = threading.RLock()

        # Load existing data
        self._load_usage_data()

    def _load_usage_data(self) -> None:
        """Load usage data from storage."""
        try:
            # Load daily summary
            daily_file = self._storage_path / "daily_usage.json"
            if daily_file.exists():
                with open(daily_file, 'r') as f:
                    data = json.load(f)
                    self._daily_usage = defaultdict(
                        Decimal,
                        {k: Decimal(v) for k, v in data.items()}
                    )

            # Load monthly summary
            monthly_file = self._storage_path / "monthly_usage.json"
            if monthly_file.exists():
                with open(monthly_file, 'r') as f:
                    data = json.load(f)
                    self._monthly_usage = defaultdict(
                        Decimal,
                        {k: Decimal(v) for k, v in data.items()}
                    )

            logger.info("Loaded usage data from storage")
        except Exception as e:
            logger.error(f"Failed to load usage data: {e}")

    def _save_usage_data(self) -> None:
        """Save usage data to storage."""
        try:
            # Save daily summary
            daily_file = self._storage_path / "daily_usage.json"
            with open(daily_file, 'w') as f:
                json.dump(
                    {k: str(v) for k, v in self._daily_usage.items()},
                    f,
                    indent=2,
                )

            # Save monthly summary
            monthly_file = self._storage_path / "monthly_usage.json"
            with open(monthly_file, 'w') as f:
                json.dump(
                    {k: str(v) for k, v in self._monthly_usage.items()},
                    f,
                    indent=2,
                )

        except Exception as e:
            logger.error(f"Failed to save usage data: {e}")

    @contextmanager
    def track_session(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[UserSession, None, None]:
        """Context manager for tracking user sessions.

        Example:
            with tracker.track_session(user_id="user123") as session:
                # Make API calls
                pass
        """
        session = UserSession(
            user_id=user_id,
            tenant_id=tenant_id,
            metadata=metadata or {},
        )

        with self._thread_lock:
            self._sessions[session.session_id] = session

        try:
            yield session
        finally:
            session.ended_at = datetime.now(timezone.utc)
            with self._thread_lock:
                if session.session_id in self._sessions:
                    del self._sessions[session.session_id]

                # Update user/tenant totals
                if session.user_id:
                    self._user_usage[session.user_id]["total"] += session.total_cost
                    self._user_usage[session.user_id]["sessions"] += Decimal("1")

                if session.tenant_id:
                    self._tenant_usage[session.tenant_id]["total"] += session.total_cost
                    self._tenant_usage[session.tenant_id]["sessions"] += Decimal("1")

    @asynccontextmanager
    async def track_session_async(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[UserSession]:
        """Async context manager for tracking user sessions."""
        session = UserSession(
            user_id=user_id,
            tenant_id=tenant_id,
            metadata=metadata or {},
        )

        async with self._lock:
            self._sessions[session.session_id] = session

        try:
            yield session
        finally:
            session.ended_at = datetime.now(timezone.utc)
            async with self._lock:
                if session.session_id in self._sessions:
                    del self._sessions[session.session_id]

                # Update user/tenant totals
                if session.user_id:
                    self._user_usage[session.user_id]["total"] += session.total_cost

                if session.tenant_id:
                    self._tenant_usage[session.tenant_id]["total"] += session.total_cost

    async def record_usage(
        self,
        request: AIRequest,
        response: Optional[AIResponse],
        provider: ProviderType,
        model: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> EnhancedUsageRecord:
        """Record detailed usage for a request/response pair.

        Args:
            request: The AI request
            response: The AI response (None for errors)
            provider: Provider that handled the request
            model: Model used
            session_id: Optional session ID
            user_id: Optional user ID
            tenant_id: Optional tenant ID
            tags: Optional tags for categorization

        Returns:
            Enhanced usage record
        """
        # Count tokens
        input_metrics = self._token_counter.count_tokens(
            request.messages, provider, model
        )

        output_metrics = TokenCountMetrics()
        if response and response.content:
            output_metrics = self._token_counter.count_tokens(
                response.content, provider, model
            )

        # Combine metrics
        total_metrics = TokenCountMetrics(
            text_tokens=input_metrics.text_tokens + output_metrics.text_tokens,
            tool_call_tokens=input_metrics.tool_call_tokens,
            tool_response_tokens=output_metrics.tool_response_tokens,
            image_tokens=input_metrics.image_tokens,
            audio_tokens=input_metrics.audio_tokens,
            video_tokens=input_metrics.video_tokens,
            system_prompt_tokens=input_metrics.system_prompt_tokens,
            cached_tokens=input_metrics.cached_tokens,
        )

        # Calculate cost
        cost_breakdown = self._cost_engine.calculate_cost(
            provider, model, total_metrics
        )

        # Create usage record
        record = EnhancedUsageRecord(
            request_id=request.request_id,
            session_id=session_id or request.session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            provider_type=provider,
            model=model,
            input_tokens=input_metrics.total_tokens,
            output_tokens=output_metrics.total_tokens,
            cost=float(cost_breakdown.total_cost),
            latency_ms=response.latency_ms if response else 0.0,
            success=response is not None,
            error_message=None if response else "Request failed",
            token_metrics=total_metrics,
            cost_breakdown=cost_breakdown,
            tags=tags or [],
        )

        async with self._lock:
            # Add to records
            self._usage_records.append(record)

            # Update session if exists
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.total_requests += 1
                session.total_tokens += total_metrics.total_tokens
                session.total_cost += cost_breakdown.total_cost

            # Update aggregated metrics
            now = datetime.now(timezone.utc)
            hour_key = now.strftime("%Y-%m-%d %H:00")
            day_key = now.strftime("%Y-%m-%d")
            month_key = now.strftime("%Y-%m")

            self._hourly_usage[hour_key] += cost_breakdown.total_cost
            self._daily_usage[day_key] += cost_breakdown.total_cost
            self._monthly_usage[month_key] += cost_breakdown.total_cost
            self._provider_usage[provider] += cost_breakdown.total_cost

            # Update user/tenant usage
            if user_id:
                self._user_usage[user_id]["total"] += cost_breakdown.total_cost
                self._user_usage[user_id][day_key] += cost_breakdown.total_cost
                # Track per-user provider usage
                provider_key = provider.value
                self._user_provider_usage[user_id][provider_key] += cost_breakdown.total_cost

            if tenant_id:
                self._tenant_usage[tenant_id]["total"] += cost_breakdown.total_cost
                self._tenant_usage[tenant_id][day_key] += cost_breakdown.total_cost
                # Track per-tenant provider usage
                provider_key = provider.value
                self._tenant_provider_usage[tenant_id][provider_key] += cost_breakdown.total_cost

            # Persist periodically
            if len(self._usage_records) % 100 == 0:
                self._save_usage_data()

        logger.debug(
            "Recorded enhanced usage",
            request_id=request.request_id,
            user_id=user_id,
            cost=float(cost_breakdown.total_cost),
            tokens=total_metrics.total_tokens,
        )

        return record

    def get_usage_analytics(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        granularity: UsageGranularity = UsageGranularity.DAILY,
    ) -> Dict[str, Any]:
        """Get comprehensive usage analytics.

        Args:
            user_id: Filter by user
            tenant_id: Filter by tenant
            start_date: Start of analysis period
            end_date: End of analysis period
            granularity: Aggregation granularity

        Returns:
            Dictionary with analytics data
        """
        with self._thread_lock:
            # Filter records
            filtered_records = [
                r for r in self._usage_records
                if (not user_id or r.user_id == user_id)
                and (not tenant_id or r.tenant_id == tenant_id)
                and (not start_date or r.timestamp >= start_date)
                and (not end_date or r.timestamp <= end_date)
            ]

            if not filtered_records:
                return {"message": "No usage data found for the specified criteria"}

            # Calculate metrics
            total_cost = sum(Decimal(str(r.cost)) for r in filtered_records)
            total_requests = len(filtered_records)
            successful_requests = sum(1 for r in filtered_records if r.success)
            total_tokens = sum(r.input_tokens + r.output_tokens for r in filtered_records)

            # Provider breakdown
            provider_costs = defaultdict(Decimal)
            provider_requests = defaultdict(int)
            for record in filtered_records:
                provider_costs[record.provider_type.value] += Decimal(str(record.cost))
                provider_requests[record.provider_type.value] += 1

            # Model breakdown
            model_costs = defaultdict(Decimal)
            model_requests = defaultdict(int)
            for record in filtered_records:
                model_costs[record.model] += Decimal(str(record.cost))
                model_requests[record.model] += 1

            # Time series data
            time_series = self._generate_time_series(
                filtered_records, granularity
            )

            # Cost efficiency metrics
            avg_cost_per_request = total_cost / total_requests if total_requests > 0 else Decimal("0")
            avg_cost_per_token = total_cost / total_tokens if total_tokens > 0 else Decimal("0")

            # Cache effectiveness
            cached_tokens = sum(r.token_metrics.cached_tokens for r in filtered_records)
            cache_ratio = cached_tokens / total_tokens if total_tokens > 0 else 0

            return {
                "summary": {
                    "total_cost": float(total_cost),
                    "total_requests": total_requests,
                    "successful_requests": successful_requests,
                    "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
                    "total_tokens": total_tokens,
                    "cached_tokens": cached_tokens,
                    "cache_ratio": cache_ratio,
                    "avg_cost_per_request": float(avg_cost_per_request),
                    "avg_cost_per_token": float(avg_cost_per_token * 1000),  # Per 1K tokens
                },
                "providers": {
                    provider: {
                        "cost": float(cost),
                        "requests": provider_requests[provider],
                        "percentage": float(cost / total_cost) if total_cost > 0 else 0,
                    }
                    for provider, cost in provider_costs.items()
                },
                "models": {
                    model: {
                        "cost": float(cost),
                        "requests": model_requests[model],
                        "percentage": float(cost / total_cost) if total_cost > 0 else 0,
                    }
                    for model, cost in model_costs.items()
                },
                "time_series": time_series,
                "top_users": self._get_top_users(filtered_records, limit=10),
                "cost_trends": self._calculate_cost_trends(filtered_records),
            }

    def _generate_time_series(
        self,
        records: List[EnhancedUsageRecord],
        granularity: UsageGranularity,
    ) -> List[Dict[str, Any]]:
        """Generate time series data."""
        time_buckets = defaultdict(lambda: {"cost": Decimal("0"), "requests": 0, "tokens": 0})

        for record in records:
            # Determine bucket key based on granularity
            if granularity == UsageGranularity.HOURLY:
                bucket_key = record.timestamp.strftime("%Y-%m-%d %H:00")
            elif granularity == UsageGranularity.DAILY:
                bucket_key = record.timestamp.strftime("%Y-%m-%d")
            elif granularity == UsageGranularity.WEEKLY:
                # Get week start
                week_start = record.timestamp - timedelta(days=record.timestamp.weekday())
                bucket_key = week_start.strftime("%Y-%m-%d")
            elif granularity == UsageGranularity.MONTHLY:
                bucket_key = record.timestamp.strftime("%Y-%m")
            else:
                bucket_key = record.timestamp.strftime("%Y-%m-%d")

            time_buckets[bucket_key]["cost"] += Decimal(str(record.cost))
            time_buckets[bucket_key]["requests"] += 1
            time_buckets[bucket_key]["tokens"] += record.input_tokens + record.output_tokens

        # Convert to list and sort
        time_series = [
            {
                "period": key,
                "cost": float(values["cost"]),
                "requests": values["requests"],
                "tokens": values["tokens"],
            }
            for key, values in sorted(time_buckets.items())
        ]

        return time_series

    def _get_top_users(
        self,
        records: List[EnhancedUsageRecord],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get top users by cost."""
        user_costs = defaultdict(Decimal)
        user_requests = defaultdict(int)

        for record in records:
            if record.user_id:
                user_costs[record.user_id] += Decimal(str(record.cost))
                user_requests[record.user_id] += 1

        # Sort by cost
        sorted_users = sorted(
            user_costs.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:limit]

        return [
            {
                "user_id": user_id,
                "cost": float(cost),
                "requests": user_requests[user_id],
            }
            for user_id, cost in sorted_users
        ]

    def _calculate_cost_trends(
        self,
        records: List[EnhancedUsageRecord],
    ) -> Dict[str, Any]:
        """Calculate cost trends and projections."""
        if not records:
            return {}

        # Sort records by timestamp
        sorted_records = sorted(records, key=lambda r: r.timestamp)

        # Calculate daily costs
        daily_costs = defaultdict(Decimal)
        for record in sorted_records:
            day_key = record.timestamp.strftime("%Y-%m-%d")
            daily_costs[day_key] += Decimal(str(record.cost))

        if len(daily_costs) < 2:
            return {"message": "Insufficient data for trend analysis"}

        # Calculate moving averages
        costs_list = list(daily_costs.values())
        ma_7day = sum(costs_list[-7:]) / min(7, len(costs_list)) if costs_list else Decimal("0")
        ma_30day = sum(costs_list[-30:]) / min(30, len(costs_list)) if costs_list else Decimal("0")

        # Simple linear projection
        if len(costs_list) >= 7:
            recent_trend = (costs_list[-1] - costs_list[-7]) / 7
            projected_30day = costs_list[-1] + (recent_trend * 30)
        else:
            projected_30day = ma_7day * 30 if ma_7day else Decimal("0")

        return {
            "current_daily_avg": float(ma_7day),
            "monthly_avg": float(ma_30day),
            "projected_monthly": float(projected_30day),
            "trend": "increasing" if len(costs_list) >= 2 and costs_list[-1] > costs_list[-2] else "stable",
        }

    def export_usage_data(
        self,
        format: str = "json",
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Union[str, bytes]:
        """Export usage data in specified format.

        Args:
            format: Export format (json, csv)
            user_id: Filter by user
            tenant_id: Filter by tenant
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Exported data as string or bytes
        """
        with self._thread_lock:
            # Filter records
            filtered_records = [
                r for r in self._usage_records
                if (not user_id or r.user_id == user_id)
                and (not tenant_id or r.tenant_id == tenant_id)
                and (not start_date or r.timestamp >= start_date)
                and (not end_date or r.timestamp <= end_date)
            ]

        if format == "json":
            return json.dumps(
                [r.to_dict() for r in filtered_records],
                indent=2,
                default=str,
            )
        elif format == "csv":
            import csv
            import io

            output = io.StringIO()
            if filtered_records:
                fieldnames = [
                    "timestamp", "request_id", "user_id", "tenant_id",
                    "provider", "model", "input_tokens", "output_tokens",
                    "total_cost", "success", "latency_ms",
                ]
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()

                for record in filtered_records:
                    writer.writerow({
                        "timestamp": record.timestamp.isoformat(),
                        "request_id": record.request_id,
                        "user_id": record.user_id or "",
                        "tenant_id": record.tenant_id or "",
                        "provider": record.provider_type.value,
                        "model": record.model,
                        "input_tokens": record.input_tokens,
                        "output_tokens": record.output_tokens,
                        "total_cost": float(record.cost_breakdown.total_cost),
                        "success": record.success,
                        "latency_ms": record.latency_ms,
                    })

            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")


# ==================== Budget Manager ====================

class BudgetManager:
    """Advanced budget management with enforcement and alerts."""

    def __init__(
        self,
        usage_tracker: EnhancedUsageTracker,
        cost_engine: CostCalculationEngine,
    ):
        self._usage_tracker = usage_tracker
        self._cost_engine = cost_engine
        self._budgets: Dict[str, EnhancedBudget] = {}
        self._alerts: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self._lock = asyncio.Lock()

    async def add_budget(self, budget: EnhancedBudget) -> None:
        """Add a budget for monitoring."""
        async with self._lock:
            self._budgets[budget.budget_id] = budget
            logger.info(f"Added budget: {budget.name} (ID: {budget.budget_id})")

    async def check_budget(
        self,
        request: AIRequest,
        provider: ProviderType,
        model: str,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Check if request is within budget limits.

        Returns:
            Tuple of (allowed, rejection_reason, degradation_strategy)
        """
        # Estimate request cost
        token_counter = AdvancedTokenCounter()
        metrics = token_counter.count_tokens(request.messages, provider, model)
        estimated_cost = self._cost_engine.calculate_cost(provider, model, metrics)

        async with self._lock:
            for budget in self._budgets.values():
                if not budget.enabled:
                    continue

                # Check if budget applies to user/tenant
                if budget.user_id and budget.user_id != user_id:
                    continue
                if budget.tenant_id and budget.tenant_id != tenant_id:
                    continue

                # Check various limits
                violation, reason = await self._check_budget_limits(
                    budget, estimated_cost.total_cost, provider, model, user_id, tenant_id
                )

                if violation:
                    if budget.enforcement_level == BudgetLevel.HARD:
                        return False, reason, None
                    elif budget.enforcement_level == BudgetLevel.ADAPTIVE:
                        strategy = self._get_degradation_strategy(budget, estimated_cost.total_cost)
                        return True, None, strategy
                    else:  # SOFT
                        await self._create_alert(budget, reason, estimated_cost.total_cost)

        return True, None, None

    async def _check_budget_limits(
        self,
        budget: EnhancedBudget,
        estimated_cost: Decimal,
        provider: ProviderType,
        model: str,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Check specific budget limits."""
        now = datetime.now(timezone.utc)

        # Per-request limit
        if budget.per_request_limit and estimated_cost > budget.per_request_limit:
            return True, f"Request cost ${estimated_cost:.4f} exceeds limit ${budget.per_request_limit:.4f}"

        # Hourly limit
        if budget.hourly_limit:
            hour_key = now.strftime("%Y-%m-%d %H:00")
            current_hourly = self._usage_tracker._hourly_usage.get(hour_key, Decimal("0"))
            if current_hourly + estimated_cost > budget.hourly_limit:
                return True, f"Hourly budget exceeded: ${current_hourly + estimated_cost:.4f} > ${budget.hourly_limit:.4f}"

        # Daily limit
        if budget.daily_limit:
            day_key = now.strftime("%Y-%m-%d")
            current_daily = self._usage_tracker._daily_usage.get(day_key, Decimal("0"))
            if current_daily + estimated_cost > budget.daily_limit:
                return True, f"Daily budget exceeded: ${current_daily + estimated_cost:.4f} > ${budget.daily_limit:.4f}"

        # Monthly limit
        if budget.monthly_limit:
            month_key = now.strftime("%Y-%m")
            current_monthly = self._usage_tracker._monthly_usage.get(month_key, Decimal("0"))
            if current_monthly + estimated_cost > budget.monthly_limit:
                return True, f"Monthly budget exceeded: ${current_monthly + estimated_cost:.4f} > ${budget.monthly_limit:.4f}"

        # Provider-specific limits
        provider_key = provider.value
        if provider_key in budget.provider_limits:
            # Check appropriate provider usage based on budget scope
            if budget.user_id and user_id:
                # Per-user provider limit
                current_provider = self._usage_tracker._user_provider_usage[user_id].get(provider_key, Decimal("0"))
                scope_desc = f"user {user_id}"
            elif budget.tenant_id and tenant_id:
                # Per-tenant provider limit
                current_provider = self._usage_tracker._tenant_provider_usage[tenant_id].get(provider_key, Decimal("0"))
                scope_desc = f"tenant {tenant_id}"
            else:
                # Global provider limit
                current_provider = self._usage_tracker._provider_usage.get(provider, Decimal("0"))
                scope_desc = "global"

            if current_provider + estimated_cost > budget.provider_limits[provider_key]:
                return True, f"Provider budget exceeded for {provider_key} ({scope_desc}): ${current_provider + estimated_cost:.4f} > ${budget.provider_limits[provider_key]:.4f}"

        # Model-specific limits
        if model in budget.model_limits:
            # Would need to track model-specific usage
            pass

        return False, None

    def _get_degradation_strategy(
        self,
        budget: EnhancedBudget,
        estimated_cost: Decimal,
    ) -> Dict[str, Any]:
        """Get degradation strategy when approaching limits."""
        strategies = []

        # Check available degradation strategies
        if "reduce_max_tokens" in budget.degradation_strategies:
            strategies.append({
                "type": "reduce_max_tokens",
                "factor": 0.5,  # Reduce by 50%
            })

        if "use_smaller_model" in budget.degradation_strategies and budget.fallback_models:
            strategies.append({
                "type": "switch_model",
                "model": budget.fallback_models[0],
            })

        if "disable_tools" in budget.degradation_strategies:
            strategies.append({
                "type": "disable_tools",
            })

        if "reduce_temperature" in budget.degradation_strategies:
            strategies.append({
                "type": "reduce_temperature",
                "value": 0.3,
            })

        return {
            "strategies": strategies,
            "reason": "Budget limit approaching",
            "estimated_savings": float(estimated_cost * Decimal("0.3")),  # Estimated 30% savings
        }

    async def _create_alert(
        self,
        budget: EnhancedBudget,
        reason: str,
        estimated_cost: Decimal,
    ) -> None:
        """Create a budget alert."""
        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "budget_id": budget.budget_id,
            "budget_name": budget.name,
            "reason": reason,
            "estimated_cost": float(estimated_cost),
            "type": "budget_warning",
        }

        self._alerts.append(alert)
        logger.warning("Budget alert created", **alert)

    def get_budget_status(
        self,
        budget_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get current budget status and usage."""
        results = {}

        for bid, budget in self._budgets.items():
            if budget_id and bid != budget_id:
                continue
            if user_id and budget.user_id != user_id:
                continue
            if tenant_id and budget.tenant_id != tenant_id:
                continue

            now = datetime.now(timezone.utc)
            status = {
                "budget_id": budget.budget_id,
                "name": budget.name,
                "enabled": budget.enabled,
                "limits": {},
                "usage": {},
                "remaining": {},
                "alerts": [],
            }

            # Check various time periods
            if budget.hourly_limit:
                hour_key = now.strftime("%Y-%m-%d %H:00")
                usage = self._usage_tracker._hourly_usage.get(hour_key, Decimal("0"))
                status["limits"]["hourly"] = float(budget.hourly_limit)
                status["usage"]["hourly"] = float(usage)
                status["remaining"]["hourly"] = float(budget.hourly_limit - usage)

            if budget.daily_limit:
                day_key = now.strftime("%Y-%m-%d")
                usage = self._usage_tracker._daily_usage.get(day_key, Decimal("0"))
                status["limits"]["daily"] = float(budget.daily_limit)
                status["usage"]["daily"] = float(usage)
                status["remaining"]["daily"] = float(budget.daily_limit - usage)

            if budget.monthly_limit:
                month_key = now.strftime("%Y-%m")
                usage = self._usage_tracker._monthly_usage.get(month_key, Decimal("0"))
                status["limits"]["monthly"] = float(budget.monthly_limit)
                status["usage"]["monthly"] = float(usage)
                status["remaining"]["monthly"] = float(budget.monthly_limit - usage)

            # Get recent alerts for this budget
            status["alerts"] = [
                alert for alert in self._alerts
                if alert.get("budget_id") == budget.budget_id
            ][-10:]  # Last 10 alerts

            results[budget.budget_id] = status

        return results


# ==================== Usage Tracking Decorator ====================

def track_usage(
    provider: ProviderType,
    model: str,
    tracker: Optional[EnhancedUsageTracker] = None,
):
    """Decorator for automatic usage tracking.

    Example:
        @track_usage(ProviderType.OPENAI, "gpt-4", tracker=my_tracker)
        async def make_ai_call(request: AIRequest) -> AIResponse:
            # Make API call
            return response
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, AIRequest):
                    request = arg
                    break
            if not request and "request" in kwargs:
                request = kwargs["request"]

            if not request:
                logger.warning("No AIRequest found in tracked function")
                return await func(*args, **kwargs)

            # Get tracker
            usage_tracker = tracker or EnhancedUsageTracker()

            # Execute function
            start_time = datetime.now(timezone.utc)
            response = None
            error = None

            try:
                response = await func(*args, **kwargs)
                return response
            except Exception as e:
                error = str(e)
                raise
            finally:
                # Calculate latency
                latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

                # Create response object if needed
                if response and not isinstance(response, AIResponse):
                    response_obj = AIResponse(
                        request_id=request.request_id,
                        provider_type=provider,
                        model=model,
                        content=str(response),
                        latency_ms=latency_ms,
                    )
                else:
                    response_obj = response
                    if response_obj:
                        response_obj.latency_ms = latency_ms

                # Record usage
                await usage_tracker.record_usage(
                    request=request,
                    response=response_obj,
                    provider=provider,
                    model=model,
                )

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # For sync functions, we need to run in event loop
            import asyncio
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ==================== Main Integration Class ====================

class ComprehensiveUsageManager:
    """Main class integrating all usage tracking and cost management features."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        enable_persistence: bool = True,
    ):
        self.token_counter = AdvancedTokenCounter()
        self.cost_engine = CostCalculationEngine()
        self.usage_tracker = EnhancedUsageTracker(
            storage_path=storage_path,
            token_counter=self.token_counter,
            cost_engine=self.cost_engine,
        )
        self.budget_manager = BudgetManager(
            usage_tracker=self.usage_tracker,
            cost_engine=self.cost_engine,
        )

        self._enable_persistence = enable_persistence

        logger.info("Initialized Comprehensive Usage Manager")

    async def track_request(
        self,
        request: AIRequest,
        provider: ProviderType,
        model: str,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Track a request and check budgets.

        Returns:
            Tuple of (allowed, rejection_reason, degradation_strategy)
        """
        # Check budget first
        allowed, reason, strategy = await self.budget_manager.check_budget(
            request, provider, model, user_id, tenant_id
        )

        if not allowed:
            logger.warning(f"Request blocked by budget: {reason}")
            return False, reason, None

        return True, None, strategy

    async def record_response(
        self,
        request: AIRequest,
        response: Optional[AIResponse],
        provider: ProviderType,
        model: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> EnhancedUsageRecord:
        """Record usage for a completed request."""
        return await self.usage_tracker.record_usage(
            request, response, provider, model,
            session_id, user_id, tenant_id, tags,
        )

    def get_analytics(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get comprehensive analytics."""
        return self.usage_tracker.get_usage_analytics(
            user_id, tenant_id, start_date, end_date
        )

    def export_data(
        self,
        format: str = "json",
        **filters,
    ) -> Union[str, bytes]:
        """Export usage data."""
        return self.usage_tracker.export_usage_data(format, **filters)


# Export main components
__all__ = [
    "ComprehensiveUsageManager",
    "EnhancedUsageTracker",
    "AdvancedTokenCounter",
    "CostCalculationEngine",
    "BudgetManager",
    "EnhancedBudget",
    "UserSession",
    "track_usage",
    "TokenCountMetrics",
    "CostBreakdown",
    "EnhancedUsageRecord",
]