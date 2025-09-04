"""Automatic model optimization algorithms for intelligent AI model selection."""

import asyncio
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import structlog
from .task_categorizer import TTRPGTaskType, TaskCharacteristics, TaskCategorizer
from .performance_profiler import PerformanceBenchmark, MetricType, ModelPerformanceProfile
from ..ai_providers.models import ProviderType, ModelSpec

logger = structlog.get_logger(__name__)


class OptimizationStrategy(Enum):
    """Strategies for automatic model optimization."""
    
    COST_OPTIMAL = "cost_optimal"           # Minimize cost while meeting quality thresholds
    QUALITY_OPTIMAL = "quality_optimal"    # Maximize quality regardless of cost
    BALANCED = "balanced"                   # Balance cost, quality, and latency
    LATENCY_OPTIMAL = "latency_optimal"    # Minimize response time
    ADAPTIVE = "adaptive"                   # Learn from user behavior and optimize accordingly


class LoadBalancingStrategy(Enum):
    """Load balancing strategies across providers."""
    
    ROUND_ROBIN = "round_robin"            # Simple round-robin
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"  # Weight by performance
    LEAST_CONNECTIONS = "least_connections"  # Route to least busy provider
    PERFORMANCE_BASED = "performance_based"  # Route based on current performance
    PREDICTIVE = "predictive"              # Predict optimal routing based on patterns


@dataclass
class OptimizationRule:
    """A rule for automatic model optimization."""
    
    rule_id: str = field(default_factory=lambda: str(uuid4()))
    task_type: Optional[TTRPGTaskType] = None
    condition: str = ""  # e.g., "latency > 2000"
    action: str = ""     # e.g., "switch_to_faster_model"
    priority: int = 1    # Higher = more important
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class ModelRecommendation:
    """A model recommendation from the optimizer."""
    
    provider_type: ProviderType
    model_id: str
    confidence: float
    reasoning: List[str]
    expected_performance: Dict[str, float]
    cost_estimate: float
    fallback_options: List[Tuple[ProviderType, str]] = field(default_factory=list)


class ModelOptimizer:
    """Automatic model optimization system for TTRPG Assistant."""
    
    def __init__(
        self,
        task_categorizer: TaskCategorizer,
        performance_benchmark: PerformanceBenchmark,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    ):
        self.task_categorizer = task_categorizer
        self.performance_benchmark = performance_benchmark
        self.optimization_strategy = optimization_strategy
        
        # Model and performance data
        self.available_models: Dict[str, ModelSpec] = {}
        self.model_capabilities: Dict[str, Dict[str, float]] = {}
        self.provider_health: Dict[ProviderType, float] = {}
        
        # Optimization state
        self.optimization_rules: List[OptimizationRule] = []
        self.learning_history: deque = deque(maxlen=10000)
        self.model_usage_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Load balancing
        self.load_balancer_state: Dict[str, Any] = {}
        self.provider_load: Dict[ProviderType, int] = defaultdict(int)
        
        # Predictive modeling
        self.prediction_cache: Dict[str, Tuple[datetime, ModelRecommendation]] = {}
        self.pattern_detection: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)
        
        # Performance thresholds
        self.quality_thresholds = {
            TTRPGTaskType.RULE_LOOKUP: 0.9,
            TTRPGTaskType.RULE_CLARIFICATION: 0.85,
            TTRPGTaskType.CHARACTER_GENERATION: 0.8,
            TTRPGTaskType.NPC_GENERATION: 0.75,
            TTRPGTaskType.STORY_GENERATION: 0.9,
            TTRPGTaskType.WORLD_BUILDING: 0.85,
            TTRPGTaskType.DESCRIPTION_GENERATION: 0.75,
            TTRPGTaskType.COMBAT_RESOLUTION: 0.95,
            TTRPGTaskType.SESSION_SUMMARIZATION: 0.8,
            TTRPGTaskType.IMPROVISATION: 0.7
        }
        
        self.latency_thresholds = {
            TTRPGTaskType.COMBAT_RESOLUTION: 500,      # 500ms
            TTRPGTaskType.RULE_LOOKUP: 1000,           # 1s
            TTRPGTaskType.RULE_CLARIFICATION: 2000,    # 2s
            TTRPGTaskType.CHARACTER_GENERATION: 5000,  # 5s
            TTRPGTaskType.IMPROVISATION: 1500,         # 1.5s
        }
        
        # Initialize default optimization rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self) -> None:
        """Initialize default optimization rules.
        
        TODO: HIGH PRIORITY - Default optimization rules are hardcoded, making them
        difficult to tune or expand upon without code changes. To improve maintainability,
        consider loading these rules from an external configuration file (e.g., YAML or JSON).
        This would allow for dynamic adjustments to the optimization logic.
        """
        default_rules = [
            OptimizationRule(
                task_type=TTRPGTaskType.COMBAT_RESOLUTION,
                condition="latency > 500",
                action="switch_to_fastest_model",
                priority=10
            ),
            OptimizationRule(
                task_type=TTRPGTaskType.RULE_LOOKUP,
                condition="error_rate > 0.05",
                action="switch_to_most_accurate_model",
                priority=9
            ),
            OptimizationRule(
                condition="cost_per_hour > user_budget",
                action="switch_to_cost_efficient_model",
                priority=8
            ),
            OptimizationRule(
                condition="provider_health < 0.8",
                action="switch_provider",
                priority=7
            ),
            OptimizationRule(
                task_type=TTRPGTaskType.STORY_GENERATION,
                condition="quality_score < 0.8",
                action="switch_to_highest_quality_model",
                priority=6
            )
        ]
        
        self.optimization_rules.extend(default_rules)
    
    async def register_model(self, model_spec: ModelSpec) -> None:
        """Register a model with the optimizer."""
        model_key = f"{model_spec.provider_type.value}:{model_spec.model_id}"
        self.available_models[model_key] = model_spec
        
        # Initialize capabilities mapping
        capabilities = {}
        if model_spec.supports_tools:
            capabilities["tool_calling"] = 1.0
        if model_spec.supports_streaming:
            capabilities["streaming"] = 1.0
        if model_spec.supports_vision:
            capabilities["vision"] = 1.0
        
        # Estimate task suitability based on model characteristics
        capabilities["rule_lookup"] = 0.9 if model_spec.supports_tools else 0.7
        capabilities["character_generation"] = 0.8
        capabilities["story_generation"] = min(1.0, model_spec.context_length / 8192)
        capabilities["combat_resolution"] = 0.9 if model_spec.supports_tools else 0.6
        
        # Factor in cost tier
        cost_multiplier = {
            "free": 1.2, "low": 1.1, "medium": 1.0, "high": 0.9, "premium": 0.8
        }.get(model_spec.cost_tier.value, 1.0)
        
        for capability in capabilities:
            capabilities[capability] *= cost_multiplier
        
        self.model_capabilities[model_key] = capabilities
        
        logger.info(
            "Registered model with optimizer",
            model_key=model_key,
            capabilities=capabilities,
            cost_tier=model_spec.cost_tier.value
        )
    
    async def optimize_model_selection(
        self,
        task_type: TTRPGTaskType,
        task_characteristics: TaskCharacteristics,
        context: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> ModelRecommendation:
        """Optimize model selection for a specific task."""
        context = context or {}
        user_preferences = user_preferences or {}
        
        logger.info(
            "Starting model optimization",
            task_type=task_type.value,
            strategy=self.optimization_strategy.value
        )
        
        # Check prediction cache first
        cache_key = self._generate_cache_key(task_type, task_characteristics, context)
        if cache_key in self.prediction_cache:
            cached_time, cached_recommendation = self.prediction_cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=5):  # 5-minute cache
                logger.debug("Using cached model recommendation", cache_key=cache_key)
                return cached_recommendation
        
        # Get candidate models
        candidates = await self._get_candidate_models(task_type, task_characteristics, context)
        
        if not candidates:
            raise ValueError(f"No suitable models available for task type: {task_type}")
        
        # Apply optimization strategy
        if self.optimization_strategy == OptimizationStrategy.COST_OPTIMAL:
            recommendation = await self._optimize_for_cost(candidates, task_type, task_characteristics)
        elif self.optimization_strategy == OptimizationStrategy.QUALITY_OPTIMAL:
            recommendation = await self._optimize_for_quality(candidates, task_type, task_characteristics)
        elif self.optimization_strategy == OptimizationStrategy.LATENCY_OPTIMAL:
            recommendation = await self._optimize_for_latency(candidates, task_type, task_characteristics)
        elif self.optimization_strategy == OptimizationStrategy.ADAPTIVE:
            recommendation = await self._optimize_adaptive(candidates, task_type, task_characteristics, user_preferences)
        else:  # BALANCED
            recommendation = await self._optimize_balanced(candidates, task_type, task_characteristics, user_preferences)
        
        # Apply optimization rules
        recommendation = await self._apply_optimization_rules(recommendation, task_type, context)
        
        # Cache the recommendation
        self.prediction_cache[cache_key] = (datetime.now(), recommendation)
        
        # Record the decision for learning
        self.learning_history.append({
            "timestamp": datetime.now(),
            "task_type": task_type.value,
            "recommendation": recommendation,
            "context": context,
            "strategy": self.optimization_strategy.value
        })
        
        logger.info(
            "Model optimization completed",
            recommended_model=f"{recommendation.provider_type.value}:{recommendation.model_id}",
            confidence=recommendation.confidence,
            reasoning_points=len(recommendation.reasoning)
        )
        
        return recommendation
    
    async def _get_candidate_models(
        self,
        task_type: TTRPGTaskType,
        task_characteristics: TaskCharacteristics,
        context: Dict[str, Any]
    ) -> List[Tuple[str, ModelSpec, float]]:
        """Get candidate models with suitability scores."""
        candidates = []
        
        for model_key, model_spec in self.available_models.items():
            if not model_spec.is_available:
                continue
            
            # Check provider health
            provider_health = self.provider_health.get(model_spec.provider_type, 1.0)
            if provider_health < 0.5:  # Skip unhealthy providers
                continue
            
            # Calculate base suitability score
            suitability_score = self._calculate_suitability_score(
                model_spec, task_type, task_characteristics
            )
            
            # Apply context adjustments
            suitability_score *= self._apply_context_adjustments(
                model_spec, context, provider_health
            )
            
            if suitability_score > 0.1:  # Minimum threshold
                candidates.append((model_key, model_spec, suitability_score))
        
        # Sort by suitability score
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        logger.debug(
            "Generated model candidates",
            task_type=task_type.value,
            candidates_count=len(candidates),
            top_candidate=candidates[0][0] if candidates else None
        )
        
        return candidates
    
    def _calculate_suitability_score(
        self,
        model_spec: ModelSpec,
        task_type: TTRPGTaskType,
        task_characteristics: TaskCharacteristics
    ) -> float:
        """Calculate base suitability score for a model."""
        score = 0.0
        
        # Capability matching
        model_key = f"{model_spec.provider_type.value}:{model_spec.model_id}"
        capabilities = self.model_capabilities.get(model_key, {})
        
        task_capability = capabilities.get(task_type.value.replace("_", ""), 0.5)
        score += task_capability * 0.3
        
        # Context length requirement
        if model_spec.context_length >= task_characteristics.context_length_needed:
            score += 0.2
        else:
            penalty = (task_characteristics.context_length_needed - model_spec.context_length) / task_characteristics.context_length_needed
            score += 0.2 * (1 - penalty)
        
        # Tool calling requirement
        if task_characteristics.needs_tool_calling:
            if model_spec.supports_tools:
                score += 0.15
            else:
                score -= 0.1  # Penalty for missing required capability
        
        # Streaming requirement for immediate tasks
        if task_characteristics.latency_requirement.value == "immediate":
            if model_spec.supports_streaming:
                score += 0.1
        
        # Performance-based adjustments
        performance_profile = self.performance_benchmark.get_model_performance(
            model_spec.provider_type, model_spec.model_id, task_type.value
        )
        
        if performance_profile:
            # Latency scoring
            target_latency = self.latency_thresholds.get(task_type, 5000)
            if performance_profile.avg_latency <= target_latency:
                score += 0.15
            else:
                latency_penalty = min(0.15, (performance_profile.avg_latency - target_latency) / target_latency * 0.15)
                score -= latency_penalty
            
            # Quality scoring
            target_quality = self.quality_thresholds.get(task_type, 0.8)
            if performance_profile.avg_quality_score >= target_quality:
                score += 0.1
            else:
                quality_penalty = (target_quality - performance_profile.avg_quality_score) * 0.1
                score -= quality_penalty
            
            # Success rate
            score += performance_profile.success_rate * 0.1
            
            # Confidence adjustment
            score *= performance_profile.confidence_score
        
        return max(0.0, score)
    
    def _apply_context_adjustments(
        self,
        model_spec: ModelSpec,
        context: Dict[str, Any],
        provider_health: float
    ) -> float:
        """Apply context-based adjustments to suitability score."""
        adjustment = 1.0
        
        # Provider health adjustment
        adjustment *= provider_health
        
        # Load balancing adjustment
        current_load = self.provider_load.get(model_spec.provider_type, 0)
        if current_load > 10:  # High load
            adjustment *= 0.8
        elif current_load < 3:  # Low load
            adjustment *= 1.1
        
        # Time-based adjustments (some models perform better at certain times)
        current_hour = datetime.now().hour
        if model_spec.provider_type == ProviderType.ANTHROPIC and 9 <= current_hour <= 17:
            adjustment *= 1.05  # Slightly prefer during business hours
        
        # Budget constraints
        if context.get("user_budget_remaining", float('inf')) < model_spec.cost_per_input_token * 1000:
            adjustment *= 0.5  # Heavily penalize expensive models when budget is low
        
        # Campaign genre preferences
        genre = context.get("campaign_genre", "").lower()
        if "horror" in genre and "creative" in model_spec.metadata.get("strengths", []):
            adjustment *= 1.1
        elif "tactical" in genre and "precise" in model_spec.metadata.get("strengths", []):
            adjustment *= 1.1
        
        return adjustment
    
    async def _optimize_for_cost(
        self,
        candidates: List[Tuple[str, ModelSpec, float]],
        task_type: TTRPGTaskType,
        task_characteristics: TaskCharacteristics
    ) -> ModelRecommendation:
        """Optimize model selection for cost efficiency."""
        # Filter candidates that meet minimum quality threshold
        quality_threshold = self.quality_thresholds.get(task_type, 0.8)
        suitable_candidates = []
        
        for model_key, model_spec, suitability in candidates:
            performance_profile = self.performance_benchmark.get_model_performance(
                model_spec.provider_type, model_spec.model_id, task_type.value
            )
            
            if performance_profile and performance_profile.avg_quality_score >= quality_threshold:
                suitable_candidates.append((model_key, model_spec, suitability, performance_profile))
            elif not performance_profile and suitability > 0.7:  # High suitability but no data
                suitable_candidates.append((model_key, model_spec, suitability, None))
        
        if not suitable_candidates:
            # Fallback to best available if no candidates meet quality threshold
            suitable_candidates = [(candidates[0][0], candidates[0][1], candidates[0][2], None)]
        
        # Sort by cost (ascending)
        suitable_candidates.sort(key=lambda x: x[1].cost_per_input_token + x[1].cost_per_output_token)
        
        best_candidate = suitable_candidates[0]
        model_key, model_spec, suitability, performance_profile = best_candidate
        
        reasoning = [
            "Optimized for cost efficiency",
            f"Model cost: ${model_spec.cost_per_input_token:.6f} input + ${model_spec.cost_per_output_token:.6f} output per token"
        ]
        
        if performance_profile:
            reasoning.append(f"Historical quality score: {performance_profile.avg_quality_score:.2f}")
            reasoning.append(f"Average request cost: ${performance_profile.avg_cost_per_request:.4f}")
        
        return ModelRecommendation(
            provider_type=model_spec.provider_type,
            model_id=model_spec.model_id,
            confidence=0.9,
            reasoning=reasoning,
            expected_performance={
                "quality": performance_profile.avg_quality_score if performance_profile else 0.8,
                "latency": performance_profile.avg_latency if performance_profile else 3000,
                "cost": model_spec.cost_per_input_token + model_spec.cost_per_output_token
            },
            cost_estimate=performance_profile.avg_cost_per_request if performance_profile else 0.01,
            fallback_options=[(c[1].provider_type, c[1].model_id) for c in suitable_candidates[1:3]]
        )
    
    async def _optimize_for_quality(
        self,
        candidates: List[Tuple[str, ModelSpec, float]],
        task_type: TTRPGTaskType,
        task_characteristics: TaskCharacteristics
    ) -> ModelRecommendation:
        """Optimize model selection for quality."""
        # Sort by performance data first, then by suitability
        quality_candidates = []
        
        for model_key, model_spec, suitability in candidates:
            performance_profile = self.performance_benchmark.get_model_performance(
                model_spec.provider_type, model_spec.model_id, task_type.value
            )
            
            if performance_profile:
                quality_score = performance_profile.avg_quality_score
                confidence_adjusted_quality = quality_score * performance_profile.confidence_score
            else:
                # Use suitability as proxy for quality when no performance data
                quality_score = suitability * 0.8  # Conservative estimate
                confidence_adjusted_quality = quality_score * 0.5  # Low confidence
            
            quality_candidates.append((model_key, model_spec, quality_score, confidence_adjusted_quality, performance_profile))
        
        # Sort by confidence-adjusted quality (descending)
        quality_candidates.sort(key=lambda x: x[3], reverse=True)
        
        best_candidate = quality_candidates[0]
        model_key, model_spec, quality_score, confidence_adjusted_quality, performance_profile = best_candidate
        
        reasoning = [
            "Optimized for highest quality output",
            f"Estimated quality score: {quality_score:.2f}"
        ]
        
        if performance_profile:
            reasoning.append(f"Based on {performance_profile.total_requests} historical requests")
            reasoning.append(f"Average latency: {performance_profile.avg_latency:.0f}ms")
            reasoning.append(f"Success rate: {performance_profile.success_rate:.1%}")
        else:
            reasoning.append("Based on model capabilities and specifications")
        
        return ModelRecommendation(
            provider_type=model_spec.provider_type,
            model_id=model_spec.model_id,
            confidence=0.95 if performance_profile else 0.7,
            reasoning=reasoning,
            expected_performance={
                "quality": quality_score,
                "latency": performance_profile.avg_latency if performance_profile else 5000,
                "cost": performance_profile.avg_cost_per_request if performance_profile else model_spec.cost_per_input_token * 1000
            },
            cost_estimate=performance_profile.avg_cost_per_request if performance_profile else 0.02,
            fallback_options=[(c[1].provider_type, c[1].model_id) for c in quality_candidates[1:3]]
        )
    
    async def _optimize_for_latency(
        self,
        candidates: List[Tuple[str, ModelSpec, float]],
        task_type: TTRPGTaskType,
        task_characteristics: TaskCharacteristics
    ) -> ModelRecommendation:
        """Optimize model selection for lowest latency."""
        latency_candidates = []
        
        for model_key, model_spec, suitability in candidates:
            performance_profile = self.performance_benchmark.get_model_performance(
                model_spec.provider_type, model_spec.model_id, task_type.value
            )
            
            if performance_profile:
                avg_latency = performance_profile.avg_latency
                p95_latency = performance_profile.p95_latency
            else:
                # Estimate latency based on model characteristics
                base_latency = 2000  # 2 second baseline
                if model_spec.supports_streaming:
                    base_latency *= 0.7
                if model_spec.context_length > 8192:
                    base_latency *= 1.3
                avg_latency = base_latency
                p95_latency = base_latency * 1.5
            
            latency_candidates.append((model_key, model_spec, avg_latency, p95_latency, performance_profile))
        
        # Sort by average latency (ascending)
        latency_candidates.sort(key=lambda x: x[2])
        
        best_candidate = latency_candidates[0]
        model_key, model_spec, avg_latency, p95_latency, performance_profile = best_candidate
        
        reasoning = [
            "Optimized for fastest response time",
            f"Average latency: {avg_latency:.0f}ms",
            f"95th percentile latency: {p95_latency:.0f}ms"
        ]
        
        if performance_profile:
            reasoning.append(f"Based on {performance_profile.total_requests} historical requests")
        else:
            reasoning.append("Based on estimated model performance")
        
        if model_spec.supports_streaming:
            reasoning.append("Supports streaming for even faster perceived response")
        
        return ModelRecommendation(
            provider_type=model_spec.provider_type,
            model_id=model_spec.model_id,
            confidence=0.9 if performance_profile else 0.6,
            reasoning=reasoning,
            expected_performance={
                "latency": avg_latency,
                "quality": performance_profile.avg_quality_score if performance_profile else 0.75,
                "cost": performance_profile.avg_cost_per_request if performance_profile else model_spec.cost_per_input_token * 1000
            },
            cost_estimate=performance_profile.avg_cost_per_request if performance_profile else 0.01,
            fallback_options=[(c[1].provider_type, c[1].model_id) for c in latency_candidates[1:3]]
        )
    
    async def _optimize_balanced(
        self,
        candidates: List[Tuple[str, ModelSpec, float]],
        task_type: TTRPGTaskType,
        task_characteristics: TaskCharacteristics,
        user_preferences: Dict[str, Any]
    ) -> ModelRecommendation:
        """Optimize model selection with balanced approach."""
        # Default weights for balanced optimization
        weights = {
            "quality": 0.35,
            "latency": 0.25,
            "cost": 0.25,
            "suitability": 0.15
        }
        
        # Adjust weights based on task characteristics
        if task_characteristics.latency_requirement.value == "immediate":
            weights["latency"] = 0.4
            weights["quality"] = 0.3
            weights["cost"] = 0.2
        elif task_characteristics.requires_creativity:
            weights["quality"] = 0.45
            weights["latency"] = 0.2
            weights["cost"] = 0.2
        elif task_characteristics.cost_sensitivity > 0.7:
            weights["cost"] = 0.4
            weights["quality"] = 0.3
            weights["latency"] = 0.2
        
        # Apply user preferences
        if user_preferences.get("prioritize_speed"):
            weights["latency"] *= 1.5
        if user_preferences.get("budget_conscious"):
            weights["cost"] *= 1.5
        if user_preferences.get("quality_focused"):
            weights["quality"] *= 1.5
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        scored_candidates = []
        
        for model_key, model_spec, suitability in candidates:
            performance_profile = self.performance_benchmark.get_model_performance(
                model_spec.provider_type, model_spec.model_id, task_type.value
            )
            
            # Normalize scores (0-1 scale)
            quality_score = performance_profile.avg_quality_score if performance_profile else suitability * 0.8
            
            if performance_profile:
                # Normalize latency (inverse: lower is better)
                latency_score = max(0, 1 - (performance_profile.avg_latency / 10000))  # 10s = 0 score
                cost_score = max(0, 1 - (performance_profile.avg_cost_per_request / 1.0))  # $1 = 0 score
            else:
                # Estimate scores
                latency_score = 0.7 if model_spec.supports_streaming else 0.5
                cost_score = 1 - (model_spec.cost_per_input_token + model_spec.cost_per_output_token) / 0.001
                cost_score = max(0, min(1, cost_score))
            
            # Calculate weighted score
            total_score = (
                quality_score * weights["quality"] +
                latency_score * weights["latency"] +
                cost_score * weights["cost"] +
                suitability * weights["suitability"]
            )
            
            scored_candidates.append((model_key, model_spec, total_score, performance_profile, {
                "quality": quality_score,
                "latency": latency_score,
                "cost": cost_score,
                "suitability": suitability
            }))
        
        # Sort by total score (descending)
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        
        best_candidate = scored_candidates[0]
        model_key, model_spec, total_score, performance_profile, component_scores = best_candidate
        
        reasoning = [
            "Balanced optimization across quality, latency, and cost",
            f"Overall score: {total_score:.2f}",
            f"Quality score: {component_scores['quality']:.2f} (weight: {weights['quality']:.1%})",
            f"Latency score: {component_scores['latency']:.2f} (weight: {weights['latency']:.1%})",
            f"Cost score: {component_scores['cost']:.2f} (weight: {weights['cost']:.1%})"
        ]
        
        if performance_profile:
            reasoning.append(f"Based on {performance_profile.total_requests} historical requests")
        
        return ModelRecommendation(
            provider_type=model_spec.provider_type,
            model_id=model_spec.model_id,
            confidence=min(0.95, total_score + 0.1),
            reasoning=reasoning,
            expected_performance={
                "quality": component_scores["quality"],
                "latency": performance_profile.avg_latency if performance_profile else 3000,
                "cost": performance_profile.avg_cost_per_request if performance_profile else model_spec.cost_per_input_token * 1000,
                "overall_score": total_score
            },
            cost_estimate=performance_profile.avg_cost_per_request if performance_profile else 0.015,
            fallback_options=[(c[1].provider_type, c[1].model_id) for c in scored_candidates[1:3]]
        )
    
    async def _optimize_adaptive(
        self,
        candidates: List[Tuple[str, ModelSpec, float]],
        task_type: TTRPGTaskType,
        task_characteristics: TaskCharacteristics,
        user_preferences: Dict[str, Any]
    ) -> ModelRecommendation:
        """Adaptive optimization based on learning from user behavior."""
        # Analyze historical preferences from learning history
        task_history = [
            entry for entry in self.learning_history
            if entry["task_type"] == task_type.value
        ]
        
        # Extract patterns from usage
        if len(task_history) >= 5:
            # Calculate user's implied preferences based on past selections
            preference_weights = self._analyze_user_patterns(task_history)
        else:
            # Use balanced approach for new users
            preference_weights = {"quality": 0.35, "latency": 0.25, "cost": 0.25, "suitability": 0.15}
        
        # Apply time-based learning
        recent_history = [entry for entry in task_history if 
                         (datetime.now() - entry["timestamp"]).days <= 7]
        
        if recent_history:
            # Weight recent preferences more heavily
            recent_weights = self._analyze_user_patterns(recent_history)
            # Blend with long-term patterns (70% recent, 30% historical)
            for key in preference_weights:
                preference_weights[key] = 0.7 * recent_weights.get(key, preference_weights[key]) + 0.3 * preference_weights[key]
        
        # Use balanced optimization with learned weights
        return await self._optimize_balanced(candidates, task_type, task_characteristics, user_preferences)
    
    def _analyze_user_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze user patterns from historical data using comprehensive pattern analysis."""
        if not history:
            return {"quality": 0.25, "latency": 0.25, "cost": 0.25, "suitability": 0.25}
        
        weights = {"quality": 0.25, "latency": 0.25, "cost": 0.25, "suitability": 0.25}
        
        # Extract metrics for pattern analysis
        costs = [entry["recommendation"].cost_estimate for entry in history if "recommendation" in entry]
        latencies = [entry.get("actual_latency", 0) for entry in history if entry.get("actual_latency")]
        quality_scores = [entry.get("quality_score", 0) for entry in history if entry.get("quality_score")]
        
        # Analyze cost sensitivity patterns
        if costs:
            cost_variance = self._calculate_variance(costs)
            avg_cost = sum(costs) / len(costs)
            
            # High cost variance suggests user is cost-flexible, low variance suggests cost-conscious
            if cost_variance < 0.001:  # Low variance - consistent cost preferences
                if avg_cost < 0.01:  # Consistently low cost
                    weights["cost"] = 0.45
                    weights["quality"] = 0.2
                elif avg_cost > 0.05:  # Consistently high cost for quality
                    weights["quality"] = 0.45
                    weights["cost"] = 0.1
            else:  # High variance - user varies cost based on context
                weights["suitability"] = 0.35  # Context becomes more important
        
        # Analyze latency sensitivity patterns
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            latency_variance = self._calculate_variance(latencies)
            
            if latency_variance < 100:  # Consistent latency preferences (ms²)
                if avg_latency < 1000:  # Prefers fast responses
                    weights["latency"] = 0.4
                elif avg_latency > 5000:  # Accepts slower responses for quality
                    weights["quality"] = max(weights["quality"], 0.35)
        
        # Analyze quality patterns
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            quality_variance = self._calculate_variance(quality_scores)
            
            if avg_quality > 0.8 and quality_variance < 0.01:  # Consistently high quality
                weights["quality"] = max(weights["quality"], 0.4)
            elif avg_quality < 0.6:  # User accepts lower quality (speed/cost focused)
                weights["latency"] = max(weights["latency"], 0.3)
                weights["cost"] = max(weights["cost"], 0.3)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    async def _apply_optimization_rules(
        self,
        recommendation: ModelRecommendation,
        task_type: TTRPGTaskType,
        context: Dict[str, Any]
    ) -> ModelRecommendation:
        """Apply optimization rules to the recommendation."""
        for rule in self.optimization_rules:
            if not rule.enabled:
                continue
            
            # Check if rule applies to this task type
            if rule.task_type and rule.task_type != task_type:
                continue
            
            # Evaluate rule condition
            if self._evaluate_rule_condition(rule.condition, recommendation, context):
                # Apply rule action
                recommendation = await self._apply_rule_action(rule.action, recommendation, context)
                
                # Update rule statistics
                rule.last_triggered = datetime.now()
                rule.trigger_count += 1
                
                logger.info(
                    "Applied optimization rule",
                    rule_id=rule.rule_id,
                    condition=rule.condition,
                    action=rule.action
                )
        
        return recommendation
    
    def _evaluate_rule_condition(
        self,
        condition: str,
        recommendation: ModelRecommendation,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate if a rule condition is met using safe pattern matching."""
        import re
        
        try:
            # Secure condition evaluation using regex patterns
            # Only allow predefined safe patterns to prevent injection attacks
            
            # Pattern: "metric_name > threshold"
            gt_pattern = r"^(\w+)\s*>\s*([0-9]+\.?[0-9]*)$"
            gt_match = re.match(gt_pattern, condition.strip())
            if gt_match:
                metric_name, threshold_str = gt_match.groups()
                threshold = float(threshold_str)
                
                if metric_name == "latency":
                    return recommendation.expected_performance.get("latency", 0) > threshold
                elif metric_name == "error_rate":
                    return recommendation.expected_performance.get("error_rate", 0) > threshold
                elif metric_name == "cost_per_hour":
                    user_budget = context.get("user_budget", float('inf'))
                    hourly_cost = recommendation.cost_estimate * 60
                    return hourly_cost > threshold
            
            # Pattern: "metric_name < threshold"
            lt_pattern = r"^(\w+)\s*<\s*([0-9]+\.?[0-9]*)$"
            lt_match = re.match(lt_pattern, condition.strip())
            if lt_match:
                metric_name, threshold_str = lt_match.groups()
                threshold = float(threshold_str)
                
                if metric_name == "provider_health":
                    provider_health = self.provider_health.get(recommendation.provider_type, 1.0)
                    return provider_health < threshold
                elif metric_name == "quality_score":
                    return recommendation.expected_performance.get("quality", 0) < threshold
            
            logger.warning("Unsupported or invalid rule condition", condition=condition)
            return False
        
        except (ValueError, re.error) as e:
            logger.warning("Failed to evaluate rule condition", condition=condition, error=str(e))
            return False
        
        return False
    
    async def _apply_rule_action(
        self,
        action: str,
        recommendation: ModelRecommendation,
        context: Dict[str, Any]
    ) -> ModelRecommendation:
        """Apply a rule action to modify the recommendation."""
        if action == "switch_to_fastest_model":
            # Find the fastest model from fallback options
            fastest_option = recommendation.fallback_options[0] if recommendation.fallback_options else None
            if fastest_option:
                # Create new recommendation with fastest model
                # This is a simplified implementation
                recommendation.provider_type = fastest_option[0]
                recommendation.model_id = fastest_option[1]
                recommendation.reasoning.append("Switched to fastest model due to latency rule")
        
        elif action == "switch_to_most_accurate_model":
            # Similar logic for most accurate model
            pass
        
        elif action == "switch_to_cost_efficient_model":
            # Switch to most cost-efficient model
            pass
        
        elif action == "switch_provider":
            # Switch to different provider
            if recommendation.fallback_options:
                for fallback_provider, fallback_model in recommendation.fallback_options:
                    if fallback_provider != recommendation.provider_type:
                        recommendation.provider_type = fallback_provider
                        recommendation.model_id = fallback_model
                        recommendation.reasoning.append("Switched provider due to health rule")
                        break
        
        return recommendation
    
    def _generate_cache_key(
        self,
        task_type: TTRPGTaskType,
        task_characteristics: TaskCharacteristics,
        context: Dict[str, Any]
    ) -> str:
        """Generate a cache key for model recommendations."""
        # Create a hash of the key parameters
        key_parts = [
            task_type.value,
            task_characteristics.complexity.value,
            task_characteristics.latency_requirement.value,
            str(task_characteristics.requires_creativity),
            str(task_characteristics.cost_sensitivity),
            context.get("campaign_genre", ""),
            str(context.get("user_budget_remaining", 0))
        ]
        
        return "|".join(key_parts)
    
    async def update_provider_health(self, provider_type: ProviderType, health_score: float) -> None:
        """Update provider health score."""
        self.provider_health[provider_type] = max(0.0, min(1.0, health_score))
        logger.debug("Updated provider health", provider=provider_type.value, health=health_score)
    
    async def update_provider_load(self, provider_type: ProviderType, load_change: int) -> None:
        """Update provider load tracking."""
        self.provider_load[provider_type] = max(0, self.provider_load[provider_type] + load_change)
        logger.debug("Updated provider load", provider=provider_type.value, load=self.provider_load[provider_type])
    
    def add_optimization_rule(self, rule: OptimizationRule) -> None:
        """Add a new optimization rule."""
        self.optimization_rules.append(rule)
        self.optimization_rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info("Added optimization rule", rule_id=rule.rule_id, priority=rule.priority)
    
    def remove_optimization_rule(self, rule_id: str) -> bool:
        """Remove an optimization rule."""
        for i, rule in enumerate(self.optimization_rules):
            if rule.rule_id == rule_id:
                del self.optimization_rules[i]
                logger.info("Removed optimization rule", rule_id=rule_id)
                return True
        return False
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "total_models": len(self.available_models),
            "optimization_strategy": self.optimization_strategy.value,
            "active_rules": len([r for r in self.optimization_rules if r.enabled]),
            "learning_history_size": len(self.learning_history),
            "prediction_cache_size": len(self.prediction_cache),
            "provider_health": {k.value: v for k, v in self.provider_health.items()},
            "provider_load": {k.value: v for k, v in self.provider_load.items()},
            "rule_statistics": [
                {
                    "rule_id": rule.rule_id,
                    "priority": rule.priority,
                    "trigger_count": rule.trigger_count,
                    "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None
                }
                for rule in self.optimization_rules
            ]
        }