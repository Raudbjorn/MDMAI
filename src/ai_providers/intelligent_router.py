"""
Intelligent Provider Router with Advanced Selection Algorithm
Task 25.3: Develop Provider Router with Fallback
"""

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from structlog import get_logger

from .models import (
    AIRequest,
    ProviderType,
    ProviderCapability,
    ModelSpec,
    ProviderSelection,
    CostTier,
)
from .abstract_provider import AbstractProvider
from .health_monitor import HealthMonitor, ErrorType
from .config.model_config import get_model_config_manager, ModelConfigManager
from .utils.cost_utils import (
    estimate_input_tokens,
    estimate_output_tokens,
    estimate_request_cost,
    assess_request_complexity,
)

logger = get_logger(__name__)


class SelectionStrategy(Enum):
    """Advanced provider selection strategies."""
    
    COST_OPTIMIZED = "cost_optimized"
    SPEED_OPTIMIZED = "speed_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    RELIABILITY_FOCUSED = "reliability_focused"
    LOAD_BALANCED = "load_balanced"
    ADAPTIVE = "adaptive"
    WEIGHTED_COMPOSITE = "weighted_composite"


@dataclass
class SelectionCriteria:
    """Comprehensive selection criteria with weights."""
    
    # Primary weights (must sum to 1.0)
    cost_weight: float = 0.3
    speed_weight: float = 0.3
    quality_weight: float = 0.2
    reliability_weight: float = 0.2
    
    # Performance targets
    max_latency_ms: Optional[float] = None
    max_cost_per_request: Optional[float] = None
    min_quality_score: Optional[float] = None
    min_reliability_percentage: Optional[float] = None
    
    # Hard requirements
    required_capabilities: List[ProviderCapability] = field(default_factory=list)
    excluded_providers: List[ProviderType] = field(default_factory=list)
    
    # Context preferences
    prefer_streaming: bool = False
    prefer_tools: bool = False
    prefer_vision: bool = False
    
    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.cost_weight + self.speed_weight + self.quality_weight + self.reliability_weight
        if not math.isclose(total, 1.0, rel_tol=1e-3):
            raise ValueError(f"Selection weights must sum to 1.0, got {total}")


@dataclass
class ProviderScore:
    """Detailed scoring for a provider candidate."""
    
    provider_type: ProviderType
    model_id: str
    total_score: float
    
    # Component scores (0.0 to 1.0)
    cost_score: float
    speed_score: float
    quality_score: float
    reliability_score: float
    
    # Detailed metrics
    estimated_cost: float
    estimated_latency_ms: float
    quality_rating: float
    reliability_percentage: float
    
    # Reasoning
    selection_reason: str
    warnings: List[str] = field(default_factory=list)


class IntelligentRouter:
    """
    Advanced provider router with intelligent selection algorithm.
    
    Features:
    - Multi-tier decision logic
    - Weighted composite scoring
    - Real-time performance tracking
    - Adaptive learning from request patterns
    - Circuit breaker integration
    - Cost-quality optimization
    """
    
    def __init__(
        self,
        health_monitor: HealthMonitor,
        default_criteria: Optional[SelectionCriteria] = None,
        config_manager: Optional[ModelConfigManager] = None,
    ):
        self.health_monitor = health_monitor
        self.default_criteria = default_criteria or SelectionCriteria()
        self.config_manager = config_manager or get_model_config_manager()
        
        # Performance tracking
        self._performance_history: Dict[ProviderType, List[Tuple[datetime, float, float]]] = {}
        self._model_quality_ratings: Dict[str, float] = {}
        self._adaptive_learning_enabled = True
        
        # Load balancing
        self._provider_loads: Dict[ProviderType, int] = {}
        self._last_selection: Dict[str, ProviderType] = {}  # request_pattern -> last_provider
        
        # Strategy implementations
        self._strategy_functions = {
            SelectionStrategy.COST_OPTIMIZED: self._cost_optimized_selection,
            SelectionStrategy.SPEED_OPTIMIZED: self._speed_optimized_selection,
            SelectionStrategy.QUALITY_OPTIMIZED: self._quality_optimized_selection,
            SelectionStrategy.RELIABILITY_FOCUSED: self._reliability_focused_selection,
            SelectionStrategy.LOAD_BALANCED: self._load_balanced_selection,
            SelectionStrategy.ADAPTIVE: self._adaptive_selection,
            SelectionStrategy.WEIGHTED_COMPOSITE: self._weighted_composite_selection,
        }
    
    async def select_optimal_provider(
        self,
        request: AIRequest,
        available_providers: List[AbstractProvider],
        strategy: SelectionStrategy = SelectionStrategy.WEIGHTED_COMPOSITE,
        criteria: Optional[SelectionCriteria] = None,
    ) -> Optional[ProviderScore]:
        """
        Select optimal provider using intelligent routing algorithm.
        
        Args:
            request: The AI request
            available_providers: List of available providers
            strategy: Selection strategy to use
            criteria: Custom selection criteria
            
        Returns:
            ProviderScore with detailed selection reasoning
        """
        if not available_providers:
            logger.warning("No providers available for selection")
            return None
        
        # Use provided criteria or default
        selection_criteria = criteria or self.default_criteria
        
        logger.info(
            "Starting intelligent provider selection",
            strategy=strategy.value,
            providers=len(available_providers),
            model=request.model,
        )
        
        # Tier 1: Availability filtering
        healthy_providers = await self._filter_by_availability(available_providers)
        if not healthy_providers:
            logger.error("No healthy providers available")
            return None
        
        # Tier 2: Capability matching
        capable_providers = self._filter_by_capabilities(
            healthy_providers, request, selection_criteria
        )
        if not capable_providers:
            logger.error("No providers match capability requirements")
            return None
        
        # Tier 3: Strategy-based optimization
        strategy_func = self._strategy_functions.get(strategy, self._weighted_composite_selection)
        selected_score = await strategy_func(capable_providers, request, selection_criteria)
        
        if selected_score:
            # Update load tracking
            self._provider_loads[selected_score.provider_type] = (
                self._provider_loads.get(selected_score.provider_type, 0) + 1
            )
            
            # Learn from selection for adaptive improvement
            if self._adaptive_learning_enabled:
                await self._record_selection_pattern(request, selected_score)
            
            logger.info(
                "Selected optimal provider",
                provider=selected_score.provider_type.value,
                model=selected_score.model_id,
                score=selected_score.total_score,
                strategy=strategy.value,
                reason=selected_score.selection_reason,
            )
        
        return selected_score
    
    async def _filter_by_availability(
        self, providers: List[AbstractProvider]
    ) -> List[AbstractProvider]:
        """Tier 1: Filter providers by availability and health."""
        healthy_providers = []
        
        for provider in providers:
            # Basic availability check
            if not provider.is_available:
                continue
            
            # Health metrics check
            metrics = self.health_monitor.get_metrics(provider.provider_type)
            if not metrics:
                # No health data yet, assume available
                healthy_providers.append(provider)
                continue
            
            # Circuit breaker check
            if metrics.circuit_open_until and datetime.now() < metrics.circuit_open_until:
                logger.debug(
                    "Provider excluded due to circuit breaker",
                    provider=provider.provider_type.value,
                )
                continue
            
            # Rate limit check
            if metrics.status.value in ["rate_limited", "quota_exceeded"]:
                logger.debug(
                    "Provider excluded due to rate/quota limits",
                    provider=provider.provider_type.value,
                    status=metrics.status.value,
                )
                continue
            
            # Reliability threshold (configurable)
            if metrics.uptime_percentage < 80.0:  # 80% uptime minimum
                logger.debug(
                    "Provider excluded due to low uptime",
                    provider=provider.provider_type.value,
                    uptime=metrics.uptime_percentage,
                )
                continue
            
            healthy_providers.append(provider)
        
        logger.debug(f"Availability filtering: {len(healthy_providers)}/{len(providers)} providers healthy")
        return healthy_providers
    
    def _filter_by_capabilities(
        self,
        providers: List[AbstractProvider],
        request: AIRequest,
        criteria: SelectionCriteria,
    ) -> List[AbstractProvider]:
        """Tier 2: Filter providers by capability requirements."""
        capable_providers = []
        
        for provider in providers:
            # Check if provider has the requested model
            if request.model and request.model not in provider.models:
                continue
            
            model_spec = provider.models.get(request.model)
            if not model_spec or not model_spec.is_available:
                continue
            
            # Check context length requirements
            estimated_tokens = estimate_input_tokens(request.messages)
            if estimated_tokens > model_spec.context_length:
                continue
            
            # Check required capabilities
            if criteria.required_capabilities:
                provider_caps = model_spec.capabilities
                if not all(cap in provider_caps for cap in criteria.required_capabilities):
                    continue
            
            # Check excluded providers
            if provider.provider_type in criteria.excluded_providers:
                continue
            
            # Check streaming requirement
            if criteria.prefer_streaming and not model_spec.supports_streaming:
                continue
            
            # Check tool calling requirement
            if criteria.prefer_tools and not model_spec.supports_tools:
                continue
            
            # Check vision requirement
            if criteria.prefer_vision and not model_spec.supports_vision:
                continue
            
            capable_providers.append(provider)
        
        logger.debug(f"Capability filtering: {len(capable_providers)}/{len(providers)} providers capable")
        return capable_providers
    
    async def _weighted_composite_selection(
        self,
        providers: List[AbstractProvider],
        request: AIRequest,
        criteria: SelectionCriteria,
    ) -> Optional[ProviderScore]:
        """Primary selection strategy using weighted composite scoring."""
        if not providers:
            return None
        
        scored_providers = []
        
        for provider in providers:
            model_spec = provider.models[request.model]
            
            # Calculate component scores
            cost_score = await self._calculate_cost_score(provider, model_spec, request)
            speed_score = await self._calculate_speed_score(provider, model_spec, request)
            quality_score = self._calculate_quality_score(provider, model_spec, request)
            reliability_score = self._calculate_reliability_score(provider)
            
            # Calculate weighted total
            total_score = (
                cost_score * criteria.cost_weight +
                speed_score * criteria.speed_weight +
                quality_score * criteria.quality_weight +
                reliability_score * criteria.reliability_weight
            )
            
            # Calculate detailed metrics for reporting
            estimated_cost = await self._estimate_request_cost(provider, model_spec, request)
            estimated_latency = await self._estimate_request_latency(provider, model_spec)
            quality_rating = self._get_model_quality_rating(request.model)
            
            metrics = self.health_monitor.get_metrics(provider.provider_type)
            reliability_percentage = metrics.uptime_percentage if metrics else 95.0
            
            # Generate selection reasoning
            reason_parts = []
            if cost_score > 0.8:
                reason_parts.append("excellent cost efficiency")
            if speed_score > 0.8:
                reason_parts.append("high speed performance")
            if quality_score > 0.8:
                reason_parts.append("superior quality")
            if reliability_score > 0.9:
                reason_parts.append("exceptional reliability")
            
            selection_reason = f"Selected for {', '.join(reason_parts) or 'balanced performance'}"
            
            # Check for warnings
            warnings = []
            if estimated_latency > 5000:  # 5 seconds
                warnings.append("High estimated latency")
            if reliability_percentage < 95:
                warnings.append(f"Lower reliability ({reliability_percentage:.1f}%)")
            
            provider_score = ProviderScore(
                provider_type=provider.provider_type,
                model_id=request.model,
                total_score=total_score,
                cost_score=cost_score,
                speed_score=speed_score,
                quality_score=quality_score,
                reliability_score=reliability_score,
                estimated_cost=estimated_cost,
                estimated_latency_ms=estimated_latency,
                quality_rating=quality_rating,
                reliability_percentage=reliability_percentage,
                selection_reason=selection_reason,
                warnings=warnings,
            )
            
            scored_providers.append(provider_score)
        
        # Sort by total score (descending)
        scored_providers.sort(key=lambda x: x.total_score, reverse=True)
        
        # Apply final validation checks
        best_score = scored_providers[0]
        
        # Check hard limits
        if criteria.max_cost_per_request and best_score.estimated_cost > criteria.max_cost_per_request:
            logger.warning(
                "Best provider exceeds cost limit",
                provider=best_score.provider_type.value,
                cost=best_score.estimated_cost,
                limit=criteria.max_cost_per_request,
            )
            # Try next best option
            for score in scored_providers[1:]:
                if score.estimated_cost <= criteria.max_cost_per_request:
                    return score
            return None
        
        if criteria.max_latency_ms and best_score.estimated_latency_ms > criteria.max_latency_ms:
            logger.warning(
                "Best provider exceeds latency limit",
                provider=best_score.provider_type.value,
                latency=best_score.estimated_latency_ms,
                limit=criteria.max_latency_ms,
            )
            # Try next best option
            for score in scored_providers[1:]:
                if score.estimated_latency_ms <= criteria.max_latency_ms:
                    return score
            return None
        
        return best_score
    
    async def _cost_optimized_selection(
        self,
        providers: List[AbstractProvider],
        request: AIRequest,
        criteria: SelectionCriteria,
    ) -> Optional[ProviderScore]:
        """Cost-optimized selection strategy."""
        best_provider = None
        lowest_cost = float('inf')
        
        for provider in providers:
            model_spec = provider.models[request.model]
            cost = await self._estimate_request_cost(provider, model_spec, request)
            
            if cost < lowest_cost:
                lowest_cost = cost
                best_provider = provider
        
        if best_provider:
            return ProviderScore(
                provider_type=best_provider.provider_type,
                model_id=request.model,
                total_score=1.0,
                cost_score=1.0,
                speed_score=0.5,
                quality_score=0.5,
                reliability_score=0.5,
                estimated_cost=lowest_cost,
                estimated_latency_ms=await self._estimate_request_latency(best_provider, model_spec),
                quality_rating=self._get_model_quality_rating(request.model),
                reliability_percentage=95.0,
                selection_reason="Lowest cost option",
            )
        
        return None
    
    async def _speed_optimized_selection(
        self,
        providers: List[AbstractProvider],
        request: AIRequest,
        criteria: SelectionCriteria,
    ) -> Optional[ProviderScore]:
        """Speed-optimized selection strategy."""
        best_provider = None
        lowest_latency = float('inf')
        
        for provider in providers:
            model_spec = provider.models[request.model]
            latency = await self._estimate_request_latency(provider, model_spec)
            
            if latency < lowest_latency:
                lowest_latency = latency
                best_provider = provider
        
        if best_provider:
            return ProviderScore(
                provider_type=best_provider.provider_type,
                model_id=request.model,
                total_score=1.0,
                cost_score=0.5,
                speed_score=1.0,
                quality_score=0.5,
                reliability_score=0.5,
                estimated_cost=await self._estimate_request_cost(best_provider, model_spec, request),
                estimated_latency_ms=lowest_latency,
                quality_rating=self._get_model_quality_rating(request.model),
                reliability_percentage=95.0,
                selection_reason="Fastest response time",
            )
        
        return None
    
    async def _quality_optimized_selection(
        self,
        providers: List[AbstractProvider],
        request: AIRequest,
        criteria: SelectionCriteria,
    ) -> Optional[ProviderScore]:
        """Quality-optimized selection strategy."""
        best_provider = None
        highest_quality = 0.0
        
        for provider in providers:
            quality_rating = self._get_model_quality_rating(request.model)
            
            if quality_rating > highest_quality:
                highest_quality = quality_rating
                best_provider = provider
        
        if best_provider:
            model_spec = best_provider.models[request.model]
            return ProviderScore(
                provider_type=best_provider.provider_type,
                model_id=request.model,
                total_score=1.0,
                cost_score=0.5,
                speed_score=0.5,
                quality_score=1.0,
                reliability_score=0.5,
                estimated_cost=await self._estimate_request_cost(best_provider, model_spec, request),
                estimated_latency_ms=await self._estimate_request_latency(best_provider, model_spec),
                quality_rating=highest_quality,
                reliability_percentage=95.0,
                selection_reason="Highest quality model",
            )
        
        return None
    
    async def _reliability_focused_selection(
        self,
        providers: List[AbstractProvider],
        request: AIRequest,
        criteria: SelectionCriteria,
    ) -> Optional[ProviderScore]:
        """Reliability-focused selection strategy."""
        best_provider = None
        highest_reliability = 0.0
        
        for provider in providers:
            reliability = self._calculate_reliability_score(provider)
            
            if reliability > highest_reliability:
                highest_reliability = reliability
                best_provider = provider
        
        if best_provider:
            model_spec = best_provider.models[request.model]
            metrics = self.health_monitor.get_metrics(best_provider.provider_type)
            reliability_percentage = metrics.uptime_percentage if metrics else 95.0
            
            return ProviderScore(
                provider_type=best_provider.provider_type,
                model_id=request.model,
                total_score=1.0,
                cost_score=0.5,
                speed_score=0.5,
                quality_score=0.5,
                reliability_score=1.0,
                estimated_cost=await self._estimate_request_cost(best_provider, model_spec, request),
                estimated_latency_ms=await self._estimate_request_latency(best_provider, model_spec),
                quality_rating=self._get_model_quality_rating(request.model),
                reliability_percentage=reliability_percentage,
                selection_reason="Most reliable provider",
            )
        
        return None
    
    async def _load_balanced_selection(
        self,
        providers: List[AbstractProvider],
        request: AIRequest,
        criteria: SelectionCriteria,
    ) -> Optional[ProviderScore]:
        """Load-balanced selection strategy."""
        # Find provider with lowest current load
        best_provider = None
        lowest_load = float('inf')
        
        for provider in providers:
            current_load = self._provider_loads.get(provider.provider_type, 0)
            
            if current_load < lowest_load:
                lowest_load = current_load
                best_provider = provider
        
        if best_provider:
            model_spec = best_provider.models[request.model]
            return ProviderScore(
                provider_type=best_provider.provider_type,
                model_id=request.model,
                total_score=1.0,
                cost_score=0.7,
                speed_score=0.7,
                quality_score=0.7,
                reliability_score=0.7,
                estimated_cost=await self._estimate_request_cost(best_provider, model_spec, request),
                estimated_latency_ms=await self._estimate_request_latency(best_provider, model_spec),
                quality_rating=self._get_model_quality_rating(request.model),
                reliability_percentage=95.0,
                selection_reason="Load balancing optimization",
            )
        
        return None
    
    async def _adaptive_selection(
        self,
        providers: List[AbstractProvider],
        request: AIRequest,
        criteria: SelectionCriteria,
    ) -> Optional[ProviderScore]:
        """Adaptive selection based on learned patterns."""
        # Generate request pattern signature
        pattern = self._generate_request_pattern(request)
        
        # Check if we have learned preferences for this pattern
        if pattern in self._last_selection:
            preferred_provider = self._last_selection[pattern]
            
            # Verify preferred provider is still available and capable
            for provider in providers:
                if provider.provider_type == preferred_provider:
                    model_spec = provider.models[request.model]
                    return ProviderScore(
                        provider_type=provider.provider_type,
                        model_id=request.model,
                        total_score=1.0,
                        cost_score=0.8,
                        speed_score=0.8,
                        quality_score=0.8,
                        reliability_score=0.8,
                        estimated_cost=await self._estimate_request_cost(provider, model_spec, request),
                        estimated_latency_ms=await self._estimate_request_latency(provider, model_spec),
                        quality_rating=self._get_model_quality_rating(request.model),
                        reliability_percentage=95.0,
                        selection_reason="Adaptive learning preference",
                    )
        
        # Fall back to weighted composite if no learned preference
        return await self._weighted_composite_selection(providers, request, criteria)
    
    # Helper methods for scoring calculations
    
    async def _calculate_cost_score(
        self, provider: AbstractProvider, model_spec: ModelSpec, request: AIRequest
    ) -> float:
        """Calculate cost score (0.0 to 1.0, higher is better)."""
        try:
            cost = await self._estimate_request_cost(provider, model_spec, request)
            # Use normalization from config manager
            return self.config_manager.normalization_config.normalize_cost(cost)
        except Exception:
            return 0.5  # Default score
    
    async def _calculate_speed_score(
        self, provider: AbstractProvider, model_spec: ModelSpec, request: AIRequest
    ) -> float:
        """Calculate speed score (0.0 to 1.0, higher is better)."""
        try:
            latency_ms = await self._estimate_request_latency(provider, model_spec)
            # Use normalization from config manager
            return self.config_manager.normalization_config.normalize_latency(latency_ms)
        except Exception:
            return 0.5  # Default score
    
    def _calculate_quality_score(
        self, provider: AbstractProvider, model_spec: ModelSpec, request: AIRequest
    ) -> float:
        """Calculate quality score (0.0 to 1.0, higher is better)."""
        # Try to get quality from model profile first
        model_profile = self.config_manager.get_model_profile(request.model)
        if model_profile:
            base_score = model_profile.quality_score
        else:
            # Fallback to tier-based scoring
            tier_scores = {
                CostTier.FREE: 0.3,
                CostTier.LOW: 0.5,
                CostTier.MEDIUM: 0.7,
                CostTier.HIGH: 0.9,
                CostTier.PREMIUM: 1.0,
            }
            base_score = tier_scores.get(model_spec.cost_tier, 0.5)
        
        # Bonus for additional capabilities
        capability_bonus = min(0.2, len(model_spec.capabilities) * 0.05)
        
        # Consider learned quality rating
        learned_rating = self._get_model_quality_rating(request.model)
        
        # Use normalization for final score
        raw_score = base_score + capability_bonus + (learned_rating * 0.1)
        return self.config_manager.normalization_config.normalize_quality(raw_score)
    
    def _calculate_reliability_score(self, provider: AbstractProvider) -> float:
        """Calculate reliability score (0.0 to 1.0, higher is better)."""
        metrics = self.health_monitor.get_metrics(provider.provider_type)
        if not metrics:
            return 0.8  # Default for unknown providers
        
        # Base score from uptime percentage
        uptime_score = metrics.uptime_percentage / 100.0
        
        # Penalty for recent errors
        error_penalty = 0.0
        total_errors = sum(metrics.error_counts.values())
        if metrics.total_requests > 0:
            error_rate = total_errors / metrics.total_requests
            error_penalty = error_rate * 0.5  # Up to 50% penalty
        
        # Bonus for consistent performance
        consistency_bonus = 0.0
        if metrics.consecutive_failures == 0:
            consistency_bonus = 0.1
        
        return max(0.0, min(1.0, uptime_score - error_penalty + consistency_bonus))
    
    async def _estimate_request_cost(
        self, provider: AbstractProvider, model_spec: ModelSpec, request: AIRequest
    ) -> float:
        """Estimate request cost in USD."""
        # Use centralized cost estimation
        input_tokens = estimate_input_tokens(request.messages)
        output_tokens = estimate_output_tokens(request, model_spec)
        
        return estimate_request_cost(
            provider.provider_type,
            request.model,
            input_tokens,
            output_tokens,
            self.config_manager
        )
    
    async def _estimate_request_latency(
        self, provider: AbstractProvider, model_spec: ModelSpec
    ) -> float:
        """Estimate request latency in milliseconds."""
        metrics = self.health_monitor.get_metrics(provider.provider_type)
        
        if metrics and metrics.avg_latency_ms > 0:
            return metrics.avg_latency_ms
        
        # Try to get latency from model profile
        model_profile = self.config_manager.get_model_profile(model_spec.model_id)
        if model_profile:
            return model_profile.avg_latency_ms
        
        # Fallback estimation based on model size/complexity
        base_latency = {
            CostTier.FREE: 2000,
            CostTier.LOW: 1500,
            CostTier.MEDIUM: 3000,
            CostTier.HIGH: 5000,
            CostTier.PREMIUM: 8000,
        }.get(model_spec.cost_tier, 3000)
        
        return base_latency
    
    
    def _get_model_quality_rating(self, model_id: str) -> float:
        """Get learned quality rating for a model."""
        return self._model_quality_ratings.get(model_id, 0.5)
    
    def _generate_request_pattern(self, request: AIRequest) -> str:
        """Generate a pattern signature for request learning."""
        # Create pattern based on request characteristics
        pattern_components = [
            request.model,
            "streaming" if request.stream else "non_streaming",
            "tools" if request.tools else "no_tools",
            f"max_tokens_{request.max_tokens or 0}",
        ]
        
        return "_".join(pattern_components)
    
    async def _record_selection_pattern(
        self, request: AIRequest, selected_score: ProviderScore
    ) -> None:
        """Record selection for adaptive learning."""
        pattern = self._generate_request_pattern(request)
        self._last_selection[pattern] = selected_score.provider_type
    
    def get_router_statistics(self) -> Dict[str, Any]:
        """Get comprehensive router statistics."""
        return {
            "provider_loads": dict(self._provider_loads),
            "learned_patterns": len(self._last_selection),
            "quality_ratings": dict(self._model_quality_ratings),
            "adaptive_learning": self._adaptive_learning_enabled,
        }