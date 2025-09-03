"""Cost optimization recommendation engine with intelligent algorithms."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import defaultdict
import json
from pathlib import Path
from structlog import get_logger

from .models import ProviderType, CostTier
from .user_usage_tracker import UserUsageTracker, UserUsageAggregation
from .pricing_engine import PricingEngine, ModelPricing
from .analytics_dashboard import AnalyticsDashboard

logger = get_logger(__name__)


class OptimizationType(Enum):
    """Types of cost optimizations."""
    
    MODEL_DOWNGRADE = "model_downgrade"  # Switch to cheaper model
    PROVIDER_SWITCH = "provider_switch"  # Switch to cheaper provider  
    BATCH_PROCESSING = "batch_processing"  # Batch similar requests
    CACHING = "caching"  # Cache frequent requests
    REQUEST_OPTIMIZATION = "request_optimization"  # Optimize request parameters
    USAGE_TIMING = "usage_timing"  # Optimize request timing
    CONTEXT_MANAGEMENT = "context_management"  # Optimize context windows
    STREAMING_OPTIMIZATION = "streaming_optimization"  # Optimize streaming usage


class Priority(Enum):
    """Priority levels for optimization recommendations."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OptimizationRecommendation:
    """A cost optimization recommendation."""
    
    recommendation_id: str
    optimization_type: OptimizationType
    priority: Priority
    
    title: str
    description: str
    reasoning: str
    
    # Financial impact
    potential_savings: float  # USD per month
    potential_savings_percentage: float
    implementation_effort: str  # low, medium, high
    
    # Specific recommendations
    current_configuration: Dict[str, Any] = field(default_factory=dict)
    recommended_configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Quality impact assessment
    quality_impact: str = "none"  # none, minimal, moderate, significant
    quality_impact_description: str = ""
    
    # Implementation details
    action_items: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    # Metadata
    affected_users: List[str] = field(default_factory=list)
    data_period_analyzed: str = "30d"
    confidence_score: float = 0.0  # 0.0 to 1.0
    
    created_at: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "recommendation_id": self.recommendation_id,
            "optimization_type": self.optimization_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "reasoning": self.reasoning,
            "potential_savings": self.potential_savings,
            "potential_savings_percentage": self.potential_savings_percentage,
            "implementation_effort": self.implementation_effort,
            "current_configuration": self.current_configuration,
            "recommended_configuration": self.recommended_configuration,
            "quality_impact": self.quality_impact,
            "quality_impact_description": self.quality_impact_description,
            "action_items": self.action_items,
            "risks": self.risks,
            "prerequisites": self.prerequisites,
            "affected_users": self.affected_users,
            "data_period_analyzed": self.data_period_analyzed,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
        }


@dataclass
class UsagePattern:
    """Identified usage pattern for optimization analysis."""
    
    pattern_id: str
    pattern_type: str
    description: str
    
    # Pattern characteristics
    frequency: int  # requests per day/week/month
    cost_impact: float  # USD
    affected_users: List[str]
    
    # Pattern data
    models_used: Dict[str, int]  # model -> usage count
    providers_used: Dict[str, int]  # provider -> usage count
    avg_tokens_per_request: float
    avg_cost_per_request: float
    peak_usage_hours: List[int]  # hours of day
    
    optimization_opportunities: List[str] = field(default_factory=list)
    confidence: float = 0.0


class CostOptimizationEngine:
    """Intelligent cost optimization recommendation engine."""
    
    def __init__(
        self,
        usage_tracker: UserUsageTracker,
        pricing_engine: PricingEngine,
        analytics_dashboard: AnalyticsDashboard,
        storage_path: Optional[str] = None
    ):
        self.usage_tracker = usage_tracker
        self.pricing_engine = pricing_engine
        self.analytics_dashboard = analytics_dashboard
        
        self.storage_path = Path(storage_path) if storage_path else Path("./data/optimization")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Analysis configuration
        self.analysis_periods = {
            "short_term": timedelta(days=7),
            "medium_term": timedelta(days=30),
            "long_term": timedelta(days=90)
        }
        
        # Optimization thresholds
        self.optimization_thresholds = {
            "min_cost_impact": 1.0,  # USD - minimum cost impact to consider
            "min_savings_percentage": 5.0,  # % - minimum savings percentage
            "max_quality_impact_acceptable": "moderate",
            "min_confidence_score": 0.7,
        }
        
        # Model quality tiers for optimization
        self.model_quality_tiers = {
            "premium": ["gpt-4", "claude-3-opus", "gemini-ultra"],
            "high": ["gpt-4-turbo", "claude-3-sonnet", "gemini-pro"],
            "medium": ["gpt-3.5-turbo", "claude-3-haiku"],
            "low": ["text-davinci-002", "text-curie-001"],
        }
        
        # Cached analysis results
        self.usage_patterns: Dict[str, UsagePattern] = {}
        self.recommendations: Dict[str, OptimizationRecommendation] = {}
        self._last_analysis_time: Optional[datetime] = None
        
        logger.info("Cost optimization engine initialized")
    
    async def analyze_usage_patterns(self, analysis_period: timedelta = None) -> List[UsagePattern]:
        """Analyze usage patterns to identify optimization opportunities."""
        if analysis_period is None:
            analysis_period = self.analysis_periods["medium_term"]
        
        end_time = datetime.now()
        start_time = end_time - analysis_period
        
        patterns = []
        
        # Analyze per-user patterns
        for user_id, user_daily_usage in self.usage_tracker.daily_usage.items():
            user_pattern = await self._analyze_user_patterns(user_id, start_time, end_time)
            if user_pattern:
                patterns.append(user_pattern)
        
        # Analyze cross-user patterns
        global_patterns = await self._analyze_global_patterns(start_time, end_time)
        patterns.extend(global_patterns)
        
        # Cache patterns
        for pattern in patterns:
            self.usage_patterns[pattern.pattern_id] = pattern
        
        logger.info("Usage pattern analysis completed", patterns_found=len(patterns))
        return patterns
    
    async def _analyze_user_patterns(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[UsagePattern]:
        """Analyze usage patterns for a specific user."""
        # Get user's usage data
        user_summary = self.usage_tracker.get_user_usage_summary(user_id, days=30)
        
        if user_summary["statistics"]["total_requests"] < 10:
            return None  # Not enough data
        
        daily_usage = user_summary["daily_usage"]
        if not daily_usage:
            return None
        
        # Aggregate usage statistics
        total_requests = 0
        total_cost = 0.0
        total_tokens = 0
        models_used = defaultdict(int)
        providers_used = defaultdict(int)
        hourly_usage = defaultdict(int)
        
        for date_str, agg_data in daily_usage.items():
            if isinstance(agg_data, dict):
                total_requests += agg_data.get("total_requests", 0)
                total_cost += agg_data.get("total_cost", 0.0)
                total_tokens += agg_data.get("total_input_tokens", 0) + agg_data.get("total_output_tokens", 0)
                
                for model, count in agg_data.get("models_used", {}).items():
                    models_used[model] += count
                
                for provider, count in agg_data.get("providers_used", {}).items():
                    providers_used[provider] += count
        
        if total_requests == 0:
            return None
        
        avg_cost_per_request = total_cost / total_requests
        avg_tokens_per_request = total_tokens / total_requests
        
        # Identify optimization opportunities
        opportunities = []
        
        # Check if user is using expensive models
        expensive_models = []
        for model, count in models_used.items():
            if model in self.model_quality_tiers["premium"] and count > total_requests * 0.1:
                expensive_models.append(model)
        
        if expensive_models:
            opportunities.append(f"High usage of premium models: {', '.join(expensive_models)}")
        
        # Check cost efficiency
        if avg_cost_per_request > 0.01:  # More than 1 cent per request
            opportunities.append("High cost per request suggests potential for model optimization")
        
        # Check provider diversity
        if len(providers_used) == 1 and total_cost > 10.0:
            opportunities.append("Single provider usage - potential for cross-provider optimization")
        
        pattern = UsagePattern(
            pattern_id=f"user_{user_id}_{start_time.date()}",
            pattern_type="user_specific",
            description=f"Usage pattern for user {user_id}",
            frequency=total_requests,
            cost_impact=total_cost,
            affected_users=[user_id],
            models_used=dict(models_used),
            providers_used=dict(providers_used),
            avg_tokens_per_request=avg_tokens_per_request,
            avg_cost_per_request=avg_cost_per_request,
            peak_usage_hours=[],  # Would need hourly data
            optimization_opportunities=opportunities,
            confidence=0.8 if total_requests > 50 else 0.5
        )
        
        return pattern
    
    async def _analyze_global_patterns(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[UsagePattern]:
        """Analyze global usage patterns across all users."""
        patterns = []
        
        # Aggregate global statistics
        global_models = defaultdict(int)
        global_providers = defaultdict(int)
        global_costs = defaultdict(float)
        total_requests = 0
        total_cost = 0.0
        
        for user_id, user_daily in self.usage_tracker.daily_usage.items():
            for date_str, agg in user_daily.items():
                try:
                    date = datetime.fromisoformat(date_str)
                    if start_time <= date <= end_time:
                        total_requests += agg.total_requests
                        total_cost += agg.total_cost
                        
                        for model, count in agg.models_used.items():
                            global_models[model] += count
                            global_costs[model] += agg.total_cost * (count / max(agg.total_requests, 1))
                        
                        for provider, count in agg.providers_used.items():
                            global_providers[provider] += count
                except:
                    continue
        
        if total_requests < 100:  # Not enough data for global analysis
            return patterns
        
        # Find expensive model patterns
        expensive_models = []
        for model, usage_count in global_models.items():
            if usage_count > total_requests * 0.05:  # Used in >5% of requests
                avg_cost_per_request = global_costs[model] / usage_count
                if avg_cost_per_request > 0.005:  # More than 0.5 cents per request
                    expensive_models.append((model, usage_count, avg_cost_per_request))
        
        if expensive_models:
            expensive_models.sort(key=lambda x: x[2], reverse=True)
            top_expensive = expensive_models[0]
            
            pattern = UsagePattern(
                pattern_id=f"global_expensive_model_{start_time.date()}",
                pattern_type="expensive_model_usage",
                description=f"High usage of expensive model: {top_expensive[0]}",
                frequency=top_expensive[1],
                cost_impact=global_costs[top_expensive[0]],
                affected_users=list(self.usage_tracker.daily_usage.keys()),
                models_used={top_expensive[0]: top_expensive[1]},
                providers_used=dict(global_providers),
                avg_tokens_per_request=0,  # Would need detailed data
                avg_cost_per_request=top_expensive[2],
                peak_usage_hours=[],
                optimization_opportunities=[
                    f"Consider switching from {top_expensive[0]} to cheaper alternatives",
                    "Implement smart routing to use cheaper models when appropriate"
                ],
                confidence=0.9
            )
            patterns.append(pattern)
        
        # Find provider imbalance patterns
        if len(global_providers) > 1:
            sorted_providers = sorted(global_providers.items(), key=lambda x: x[1], reverse=True)
            dominant_provider = sorted_providers[0]
            
            if dominant_provider[1] > total_requests * 0.8:  # One provider handles >80%
                pattern = UsagePattern(
                    pattern_id=f"global_provider_imbalance_{start_time.date()}",
                    pattern_type="provider_imbalance",
                    description=f"Heavy reliance on single provider: {dominant_provider[0]}",
                    frequency=dominant_provider[1],
                    cost_impact=total_cost * 0.8,
                    affected_users=list(self.usage_tracker.daily_usage.keys()),
                    models_used=dict(global_models),
                    providers_used={dominant_provider[0]: dominant_provider[1]},
                    avg_tokens_per_request=0,
                    avg_cost_per_request=total_cost / total_requests,
                    peak_usage_hours=[],
                    optimization_opportunities=[
                        "Diversify provider usage for better cost optimization",
                        "Implement provider failover and cost comparison"
                    ],
                    confidence=0.8
                )
                patterns.append(pattern)
        
        return patterns
    
    async def generate_recommendations(
        self,
        user_id: Optional[str] = None,
        analysis_period: timedelta = None
    ) -> List[OptimizationRecommendation]:
        """Generate cost optimization recommendations."""
        if analysis_period is None:
            analysis_period = self.analysis_periods["medium_term"]
        
        # Analyze patterns first
        patterns = await self.analyze_usage_patterns(analysis_period)
        
        recommendations = []
        
        # Generate recommendations based on patterns
        for pattern in patterns:
            pattern_recommendations = await self._generate_pattern_recommendations(pattern)
            recommendations.extend(pattern_recommendations)
        
        # Generate model-specific recommendations
        model_recommendations = await self._generate_model_recommendations(user_id)
        recommendations.extend(model_recommendations)
        
        # Generate provider-specific recommendations
        provider_recommendations = await self._generate_provider_recommendations(user_id)
        recommendations.extend(provider_recommendations)
        
        # Generate timing-based recommendations
        timing_recommendations = await self._generate_timing_recommendations(user_id)
        recommendations.extend(timing_recommendations)
        
        # Filter and prioritize recommendations
        filtered_recommendations = self._filter_recommendations(recommendations)
        prioritized_recommendations = self._prioritize_recommendations(filtered_recommendations)
        
        # Cache recommendations
        for rec in prioritized_recommendations:
            self.recommendations[rec.recommendation_id] = rec
        
        # Save recommendations
        await self._save_recommendations(prioritized_recommendations)
        
        logger.info("Cost optimization recommendations generated", 
                   count=len(prioritized_recommendations),
                   user_id=user_id)
        
        return prioritized_recommendations
    
    async def _generate_pattern_recommendations(
        self,
        pattern: UsagePattern
    ) -> List[OptimizationRecommendation]:
        """Generate recommendations based on usage patterns."""
        recommendations = []
        
        if pattern.pattern_type == "expensive_model_usage":
            # Find cheaper alternatives
            expensive_model = list(pattern.models_used.keys())[0]
            alternatives = await self._find_model_alternatives(expensive_model)
            
            if alternatives:
                best_alternative = alternatives[0]  # Cheapest option
                
                # Calculate potential savings
                current_monthly_cost = pattern.cost_impact * (30 / 7)  # Extrapolate to monthly
                potential_savings = current_monthly_cost * (1 - best_alternative["cost_ratio"])
                savings_percentage = (1 - best_alternative["cost_ratio"]) * 100
                
                recommendation = OptimizationRecommendation(
                    recommendation_id=f"model_downgrade_{pattern.pattern_id}",
                    optimization_type=OptimizationType.MODEL_DOWNGRADE,
                    priority=Priority.HIGH if potential_savings > 50 else Priority.MEDIUM,
                    title=f"Switch from {expensive_model} to {best_alternative['model']}",
                    description=f"Replace expensive model usage with more cost-effective alternative",
                    reasoning=f"Analysis shows {expensive_model} is used {pattern.frequency} times with high cost per request. "
                             f"Switching to {best_alternative['model']} could maintain similar quality while reducing costs.",
                    potential_savings=potential_savings,
                    potential_savings_percentage=savings_percentage,
                    implementation_effort="medium",
                    current_configuration={"model": expensive_model},
                    recommended_configuration={"model": best_alternative["model"]},
                    quality_impact=best_alternative["quality_impact"],
                    quality_impact_description=best_alternative["quality_description"],
                    action_items=[
                        f"Update model configuration from {expensive_model} to {best_alternative['model']}",
                        "Test alternative model with sample requests",
                        "Monitor quality metrics after implementation",
                        "Set up A/B testing if needed"
                    ],
                    risks=[
                        "Potential quality degradation",
                        "Need to adjust prompts for new model",
                        "Different response formats may require code changes"
                    ],
                    affected_users=pattern.affected_users,
                    confidence_score=pattern.confidence * 0.9
                )
                recommendations.append(recommendation)
        
        elif pattern.pattern_type == "provider_imbalance":
            # Recommend provider diversification
            dominant_provider = list(pattern.providers_used.keys())[0]
            
            recommendation = OptimizationRecommendation(
                recommendation_id=f"provider_diversification_{pattern.pattern_id}",
                optimization_type=OptimizationType.PROVIDER_SWITCH,
                priority=Priority.MEDIUM,
                title="Diversify AI Provider Usage",
                description="Balance usage across multiple providers for cost optimization",
                reasoning=f"Currently {dominant_provider} handles majority of requests. "
                         "Diversifying can reduce costs and improve reliability.",
                potential_savings=pattern.cost_impact * 0.15,  # Estimate 15% savings
                potential_savings_percentage=15.0,
                implementation_effort="high",
                current_configuration={"primary_provider": dominant_provider},
                recommended_configuration={
                    "providers": ["anthropic", "openai", "google"],
                    "routing_strategy": "cost_optimized"
                },
                quality_impact="minimal",
                quality_impact_description="Minimal impact with smart routing",
                action_items=[
                    "Implement provider comparison logic",
                    "Set up fallback providers",
                    "Create cost-based routing rules",
                    "Monitor provider performance and costs"
                ],
                risks=[
                    "Integration complexity",
                    "Need to handle different API formats",
                    "Potential inconsistency in responses"
                ],
                affected_users=pattern.affected_users,
                confidence_score=pattern.confidence * 0.7
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _generate_model_recommendations(self, user_id: Optional[str] = None) -> List[OptimizationRecommendation]:
        """Generate model-specific recommendations."""
        recommendations = []
        
        # Analyze current model usage
        if user_id:
            user_summary = self.usage_tracker.get_user_usage_summary(user_id)
            model_usage = {}
            model_costs = {}
            
            for date_str, daily_data in user_summary.get("daily_usage", {}).items():
                if isinstance(daily_data, dict):
                    for model, count in daily_data.get("models_used", {}).items():
                        model_usage[model] = model_usage.get(model, 0) + count
                        # Estimate model cost (simplified)
                        daily_cost = daily_data.get("total_cost", 0)
                        total_requests = daily_data.get("total_requests", 1)
                        model_costs[model] = model_costs.get(model, 0) + (daily_cost * count / total_requests)
        else:
            # Global model analysis
            model_usage = defaultdict(int)
            model_costs = defaultdict(float)
            
            for user_daily in self.usage_tracker.daily_usage.values():
                for agg in user_daily.values():
                    for model, count in agg.models_used.items():
                        model_usage[model] += count
                        model_costs[model] += agg.total_cost * (count / max(agg.total_requests, 1))
        
        # Find optimization opportunities for each model
        for model, usage_count in model_usage.items():
            if usage_count < 5:  # Skip low-usage models
                continue
            
            model_cost = model_costs.get(model, 0)
            avg_cost_per_use = model_cost / usage_count
            
            # Check if model is in expensive tier
            if model in self.model_quality_tiers["premium"] and avg_cost_per_use > 0.01:
                alternatives = await self._find_model_alternatives(model)
                
                if alternatives:
                    best_alt = alternatives[0]
                    monthly_savings = model_cost * (1 - best_alt["cost_ratio"]) * 4  # Weekly to monthly
                    
                    if monthly_savings > self.optimization_thresholds["min_cost_impact"]:
                        recommendation = OptimizationRecommendation(
                            recommendation_id=f"model_opt_{model}_{datetime.now().strftime('%Y%m')}",
                            optimization_type=OptimizationType.MODEL_DOWNGRADE,
                            priority=Priority.HIGH if monthly_savings > 20 else Priority.MEDIUM,
                            title=f"Optimize {model} Usage",
                            description=f"Switch to {best_alt['model']} for cost savings",
                            reasoning=f"Model {model} has high usage ({usage_count} requests) with high cost per request "
                                     f"(${avg_cost_per_use:.4f}). Alternative available with similar capabilities.",
                            potential_savings=monthly_savings,
                            potential_savings_percentage=(1 - best_alt["cost_ratio"]) * 100,
                            implementation_effort="low",
                            current_configuration={"model": model},
                            recommended_configuration={"model": best_alt["model"]},
                            quality_impact=best_alt["quality_impact"],
                            quality_impact_description=best_alt["quality_description"],
                            action_items=[
                                f"Replace {model} with {best_alt['model']} in configuration",
                                "Test with representative workload",
                                "Monitor response quality",
                                "Rollback if quality issues detected"
                            ],
                            confidence_score=0.8
                        )
                        recommendations.append(recommendation)
        
        return recommendations
    
    async def _generate_provider_recommendations(self, user_id: Optional[str] = None) -> List[OptimizationRecommendation]:
        """Generate provider-specific recommendations."""
        recommendations = []
        
        # Get provider cost comparison
        provider_costs = {}
        provider_usage = {}
        
        if user_id:
            user_summary = self.usage_tracker.get_user_usage_summary(user_id)
            for daily_data in user_summary.get("daily_usage", {}).values():
                if isinstance(daily_data, dict):
                    for provider, count in daily_data.get("providers_used", {}).items():
                        provider_usage[provider] = provider_usage.get(provider, 0) + count
                        daily_cost = daily_data.get("total_cost", 0)
                        total_requests = daily_data.get("total_requests", 1)
                        provider_costs[provider] = provider_costs.get(provider, 0) + (daily_cost * count / total_requests)
        
        if len(provider_costs) < 2:
            return recommendations  # Need multiple providers for comparison
        
        # Find most and least expensive providers
        sorted_providers = sorted(provider_costs.items(), key=lambda x: x[1] / provider_usage.get(x[0], 1), reverse=True)
        most_expensive = sorted_providers[0]
        least_expensive = sorted_providers[-1]
        
        expensive_avg_cost = most_expensive[1] / provider_usage[most_expensive[0]]
        cheap_avg_cost = least_expensive[1] / provider_usage[least_expensive[0]]
        
        if expensive_avg_cost > cheap_avg_cost * 1.5:  # 50% more expensive
            savings_ratio = 1 - (cheap_avg_cost / expensive_avg_cost)
            monthly_savings = most_expensive[1] * savings_ratio * 4  # Weekly to monthly
            
            recommendation = OptimizationRecommendation(
                recommendation_id=f"provider_switch_{most_expensive[0]}_{datetime.now().strftime('%Y%m')}",
                optimization_type=OptimizationType.PROVIDER_SWITCH,
                priority=Priority.MEDIUM,
                title=f"Consider Switching from {most_expensive[0]} to {least_expensive[0]}",
                description="Switch to more cost-effective provider for similar requests",
                reasoning=f"Provider {most_expensive[0]} costs ${expensive_avg_cost:.4f} per request while "
                         f"{least_expensive[0]} costs ${cheap_avg_cost:.4f} per request.",
                potential_savings=monthly_savings,
                potential_savings_percentage=savings_ratio * 100,
                implementation_effort="medium",
                current_configuration={"primary_provider": most_expensive[0]},
                recommended_configuration={"primary_provider": least_expensive[0]},
                quality_impact="minimal",
                quality_impact_description="Similar quality expected with different provider",
                action_items=[
                    f"Test {least_expensive[0]} with current workloads",
                    "Compare response quality and latency",
                    "Gradually migrate traffic",
                    "Monitor performance metrics"
                ],
                risks=[
                    "API differences may require code changes",
                    "Different rate limits and quotas",
                    "Potential quality variations"
                ],
                confidence_score=0.7
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _generate_timing_recommendations(self, user_id: Optional[str] = None) -> List[OptimizationRecommendation]:
        """Generate timing-based optimization recommendations."""
        recommendations = []
        
        # This would analyze peak usage times and suggest off-peak usage
        # For now, providing a general recommendation
        
        recommendation = OptimizationRecommendation(
            recommendation_id=f"timing_optimization_{datetime.now().strftime('%Y%m')}",
            optimization_type=OptimizationType.USAGE_TIMING,
            priority=Priority.LOW,
            title="Optimize Request Timing",
            description="Batch non-urgent requests during off-peak hours",
            reasoning="Batching requests during off-peak times can reduce costs through volume discounts and better resource utilization.",
            potential_savings=10.0,  # Estimated
            potential_savings_percentage=5.0,
            implementation_effort="medium",
            quality_impact="none",
            quality_impact_description="No impact on response quality",
            action_items=[
                "Identify non-urgent batch processing tasks",
                "Schedule batch jobs during off-peak hours (2-6 AM)",
                "Implement request queuing system",
                "Monitor cost savings"
            ],
            risks=[
                "Increased latency for batched requests",
                "Complexity in request scheduling"
            ],
            confidence_score=0.6
        )
        recommendations.append(recommendation)
        
        return recommendations
    
    async def _find_model_alternatives(self, current_model: str) -> List[Dict[str, Any]]:
        """Find cheaper alternatives for a given model."""
        alternatives = []
        
        # Get current model pricing
        current_pricing = None
        for provider_type in [ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.GOOGLE]:
            pricing = self.pricing_engine.get_model_pricing(provider_type, current_model)
            if pricing:
                current_pricing = pricing
                break
        
        if not current_pricing:
            return alternatives
        
        current_cost_per_token = (current_pricing.input_token_rate + current_pricing.output_token_rate) / 2
        
        # Find alternatives in pricing models
        for pricing in self.pricing_engine.pricing_models.values():
            if pricing.model_id == current_model:
                continue
            
            alt_cost_per_token = (pricing.input_token_rate + pricing.output_token_rate) / 2
            
            if alt_cost_per_token < current_cost_per_token:
                cost_ratio = alt_cost_per_token / current_cost_per_token
                
                # Determine quality impact
                quality_impact = "unknown"
                quality_description = "Quality impact needs evaluation"
                
                if current_model in self.model_quality_tiers["premium"]:
                    if pricing.model_id in self.model_quality_tiers["high"]:
                        quality_impact = "minimal"
                        quality_description = "Slight quality reduction expected"
                    elif pricing.model_id in self.model_quality_tiers["medium"]:
                        quality_impact = "moderate"
                        quality_description = "Noticeable quality reduction likely"
                    else:
                        quality_impact = "significant"
                        quality_description = "Significant quality reduction expected"
                
                alternatives.append({
                    "model": pricing.model_id,
                    "provider": pricing.provider_type.value,
                    "cost_ratio": cost_ratio,
                    "savings_percentage": (1 - cost_ratio) * 100,
                    "quality_impact": quality_impact,
                    "quality_description": quality_description
                })
        
        # Sort by cost savings
        alternatives.sort(key=lambda x: x["cost_ratio"])
        
        return alternatives[:5]  # Return top 5 alternatives
    
    def _filter_recommendations(self, recommendations: List[OptimizationRecommendation]) -> List[OptimizationRecommendation]:
        """Filter recommendations based on thresholds and criteria."""
        filtered = []
        
        for rec in recommendations:
            # Check minimum cost impact
            if rec.potential_savings < self.optimization_thresholds["min_cost_impact"]:
                continue
            
            # Check minimum savings percentage
            if rec.potential_savings_percentage < self.optimization_thresholds["min_savings_percentage"]:
                continue
            
            # Check confidence score
            if rec.confidence_score < self.optimization_thresholds["min_confidence_score"]:
                continue
            
            # Check quality impact acceptance
            max_acceptable = self.optimization_thresholds["max_quality_impact_acceptable"]
            quality_levels = ["none", "minimal", "moderate", "significant"]
            if quality_levels.index(rec.quality_impact) > quality_levels.index(max_acceptable):
                continue
            
            filtered.append(rec)
        
        return filtered
    
    def _prioritize_recommendations(self, recommendations: List[OptimizationRecommendation]) -> List[OptimizationRecommendation]:
        """Prioritize recommendations based on impact and effort."""
        def priority_score(rec: OptimizationRecommendation) -> float:
            # Calculate priority score based on multiple factors
            savings_score = min(rec.potential_savings / 100, 1.0)  # Normalize to 0-1
            percentage_score = min(rec.potential_savings_percentage / 50, 1.0)  # Normalize to 0-1
            confidence_score = rec.confidence_score
            
            # Effort penalty
            effort_penalty = {
                "low": 0,
                "medium": -0.1,
                "high": -0.2
            }.get(rec.implementation_effort, 0)
            
            # Quality impact penalty
            quality_penalty = {
                "none": 0,
                "minimal": -0.05,
                "moderate": -0.15,
                "significant": -0.3
            }.get(rec.quality_impact, 0)
            
            return (savings_score + percentage_score + confidence_score) / 3 + effort_penalty + quality_penalty
        
        # Sort by priority score
        recommendations.sort(key=priority_score, reverse=True)
        
        # Assign priority levels based on ranking
        for i, rec in enumerate(recommendations):
            if i < len(recommendations) * 0.2:  # Top 20%
                rec.priority = Priority.CRITICAL
            elif i < len(recommendations) * 0.4:  # Next 20%
                rec.priority = Priority.HIGH
            elif i < len(recommendations) * 0.7:  # Next 30%
                rec.priority = Priority.MEDIUM
            else:  # Bottom 30%
                rec.priority = Priority.LOW
        
        return recommendations
    
    async def _save_recommendations(self, recommendations: List[OptimizationRecommendation]) -> None:
        """Save recommendations to storage."""
        try:
            recommendations_data = {
                "generated_at": datetime.now().isoformat(),
                "recommendations": [rec.to_dict() for rec in recommendations]
            }
            
            file_path = self.storage_path / f"recommendations_{datetime.now().strftime('%Y%m%d')}.json"
            with open(file_path, 'w') as f:
                json.dump(recommendations_data, f, indent=2)
                
        except Exception as e:
            logger.error("Failed to save recommendations", error=str(e))
    
    async def get_recommendation_summary(self) -> Dict[str, Any]:
        """Get a summary of current recommendations."""
        if not self.recommendations:
            return {"total": 0, "by_priority": {}, "by_type": {}, "total_potential_savings": 0.0}
        
        by_priority = defaultdict(int)
        by_type = defaultdict(int)
        total_savings = 0.0
        
        for rec in self.recommendations.values():
            by_priority[rec.priority.value] += 1
            by_type[rec.optimization_type.value] += 1
            total_savings += rec.potential_savings
        
        return {
            "total": len(self.recommendations),
            "by_priority": dict(by_priority),
            "by_type": dict(by_type),
            "total_potential_savings": total_savings,
            "last_analysis": self._last_analysis_time.isoformat() if self._last_analysis_time else None
        }
    
    async def implement_recommendation(self, recommendation_id: str) -> Dict[str, Any]:
        """Implement a specific recommendation (placeholder for integration)."""
        if recommendation_id not in self.recommendations:
            raise ValueError(f"Recommendation not found: {recommendation_id}")
        
        rec = self.recommendations[recommendation_id]
        
        # This would integrate with the actual system to implement changes
        # For now, just return implementation plan
        
        return {
            "recommendation_id": recommendation_id,
            "implementation_plan": rec.action_items,
            "estimated_implementation_time": rec.implementation_effort,
            "prerequisites": rec.prerequisites,
            "risks": rec.risks,
            "monitoring_metrics": [
                "cost_per_request",
                "response_quality",
                "latency",
                "error_rate"
            ]
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        # Clear old cached data
        cutoff_date = datetime.now() - timedelta(days=7)
        
        expired_patterns = [
            pattern_id for pattern_id, pattern in self.usage_patterns.items()
            if pattern_id.endswith(cutoff_date.date().isoformat())
        ]
        
        for pattern_id in expired_patterns:
            del self.usage_patterns[pattern_id]
        
        # Clear expired recommendations
        expired_recs = [
            rec_id for rec_id, rec in self.recommendations.items()
            if rec.valid_until and rec.valid_until < datetime.now()
        ]
        
        for rec_id in expired_recs:
            del self.recommendations[rec_id]
        
        logger.info("Cost optimization engine cleanup completed",
                   expired_patterns=len(expired_patterns),
                   expired_recommendations=len(expired_recs))