"""A/B testing framework for AI model comparison and optimization."""

import asyncio
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import structlog
from scipy import stats
from .task_categorizer import TTRPGTaskType
from .performance_profiler import PerformanceBenchmark, MetricType
from ..ai_providers.models import ProviderType

logger = structlog.get_logger(__name__)


class ExperimentStatus(Enum):
    """Status of A/B test experiments."""
    
    DRAFT = "draft"                    # Experiment designed but not started
    ACTIVE = "active"                  # Currently running
    PAUSED = "paused"                  # Temporarily stopped
    COMPLETED = "completed"            # Finished successfully
    TERMINATED = "terminated"          # Stopped before completion
    FAILED = "failed"                  # Failed due to error


class ExperimentType(Enum):
    """Types of A/B test experiments."""
    
    MODEL_COMPARISON = "model_comparison"        # Compare different models
    PROVIDER_COMPARISON = "provider_comparison"  # Compare providers
    PARAMETER_OPTIMIZATION = "parameter_optimization"  # Optimize model parameters
    STRATEGY_COMPARISON = "strategy_comparison"  # Compare selection strategies
    COST_OPTIMIZATION = "cost_optimization"     # Optimize cost vs quality
    LATENCY_OPTIMIZATION = "latency_optimization"  # Optimize response time


class StatisticalSignificance(Enum):
    """Levels of statistical significance."""
    
    NOT_SIGNIFICANT = "not_significant"  # p > 0.05
    MARGINAL = "marginal"               # 0.01 < p <= 0.05
    SIGNIFICANT = "significant"         # 0.001 < p <= 0.01
    HIGHLY_SIGNIFICANT = "highly_significant"  # p <= 0.001


@dataclass
class ExperimentVariant:
    """A variant in an A/B test experiment."""
    
    variant_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    
    # Model configuration
    provider_type: Optional[ProviderType] = None
    model_id: Optional[str] = None
    
    # Selection strategy configuration
    strategy_config: Dict[str, Any] = field(default_factory=dict)
    
    # Traffic allocation
    traffic_allocation: float = 0.5  # Percentage of traffic (0.0-1.0)
    
    # Performance tracking
    sample_size: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    
    # Metrics
    latency_samples: List[float] = field(default_factory=list)
    quality_samples: List[float] = field(default_factory=list)
    cost_samples: List[float] = field(default_factory=list)
    satisfaction_samples: List[float] = field(default_factory=list)
    
    # Statistical data
    mean_latency: float = 0.0
    mean_quality: float = 0.0
    mean_cost: float = 0.0
    mean_satisfaction: float = 0.0
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment."""
    
    experiment_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    experiment_type: ExperimentType = ExperimentType.MODEL_COMPARISON
    
    # Target criteria
    task_types: List[TTRPGTaskType] = field(default_factory=list)
    user_segments: List[str] = field(default_factory=list)  # User groups to include
    
    # Variants
    variants: List[ExperimentVariant] = field(default_factory=list)
    
    # Test parameters
    minimum_sample_size: int = 100
    maximum_duration_days: int = 14
    significance_threshold: float = 0.05
    minimum_effect_size: float = 0.1  # Minimum meaningful difference
    
    # Primary metric to optimize
    primary_metric: str = "quality"  # "quality", "latency", "cost", "satisfaction"
    
    # Status and timing
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metadata
    created_by: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Results of an A/B test experiment."""
    
    experiment_id: str
    winning_variant: Optional[str] = None
    statistical_significance: StatisticalSignificance = StatisticalSignificance.NOT_SIGNIFICANT
    p_value: float = 1.0
    effect_size: float = 0.0
    confidence_level: float = 0.95
    
    # Detailed results per variant
    variant_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Recommendations
    recommendation: str = ""
    reasoning: List[str] = field(default_factory=list)
    
    # Rollout plan
    suggested_rollout_percentage: float = 0.0
    rollout_timeline: Optional[str] = None


class ABTestingFramework:
    """A/B testing framework for AI model optimization."""
    
    def __init__(self, performance_benchmark: PerformanceBenchmark):
        self.performance_benchmark = performance_benchmark
        
        # Experiment storage
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.experiment_results: Dict[str, ExperimentResult] = {}
        self.active_experiments: Dict[str, ExperimentConfig] = {}
        
        # Traffic routing
        self.user_assignments: Dict[str, Dict[str, str]] = {}  # user_id -> {experiment_id: variant_id}
        self.assignment_cache: Dict[str, Tuple[datetime, Dict[str, str]]] = {}
        
        # Statistics and analytics
        self.statistical_cache: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}
        
    async def create_experiment(
        self,
        name: str,
        description: str,
        experiment_type: ExperimentType,
        variants: List[Dict[str, Any]],
        task_types: Optional[List[TTRPGTaskType]] = None,
        minimum_sample_size: int = 100,
        maximum_duration_days: int = 14,
        primary_metric: str = "quality"
    ) -> str:
        """Create a new A/B test experiment."""
        
        experiment = ExperimentConfig(
            name=name,
            description=description,
            experiment_type=experiment_type,
            task_types=task_types or [],
            minimum_sample_size=minimum_sample_size,
            maximum_duration_days=maximum_duration_days,
            primary_metric=primary_metric
        )
        
        # Create variants
        total_allocation = 0.0
        for variant_data in variants:
            variant = ExperimentVariant(
                name=variant_data["name"],
                description=variant_data.get("description", ""),
                provider_type=variant_data.get("provider_type"),
                model_id=variant_data.get("model_id"),
                strategy_config=variant_data.get("strategy_config", {}),
                traffic_allocation=variant_data.get("traffic_allocation", 1.0 / len(variants))
            )
            experiment.variants.append(variant)
            total_allocation += variant.traffic_allocation
        
        # Normalize traffic allocation
        if total_allocation != 1.0:
            for variant in experiment.variants:
                variant.traffic_allocation /= total_allocation
        
        # Store experiment
        self.experiments[experiment.experiment_id] = experiment
        
        logger.info(
            "Created A/B test experiment",
            experiment_id=experiment.experiment_id,
            name=name,
            variant_count=len(experiment.variants)
        )
        
        return experiment.experiment_id
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an A/B test experiment."""
        if experiment_id not in self.experiments:
            logger.error("Experiment not found", experiment_id=experiment_id)
            return False
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.DRAFT:
            logger.error("Experiment cannot be started", 
                        experiment_id=experiment_id, 
                        status=experiment.status.value)
            return False
        
        # Validate experiment configuration
        if len(experiment.variants) < 2:
            logger.error("Experiment must have at least 2 variants", experiment_id=experiment_id)
            return False
        
        # Start the experiment
        experiment.status = ExperimentStatus.ACTIVE
        experiment.started_at = datetime.now()
        
        # Add to active experiments
        self.active_experiments[experiment_id] = experiment
        
        logger.info(
            "Started A/B test experiment",
            experiment_id=experiment_id,
            name=experiment.name,
            duration_days=experiment.maximum_duration_days
        )
        
        return True
    
    async def assign_user_to_variant(
        self,
        user_id: str,
        task_type: TTRPGTaskType,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[str, str]]:
        """Assign a user to a variant for applicable experiments.
        
        Returns:
            Tuple of (experiment_id, variant_id) if assigned, None otherwise
        """
        applicable_experiments = []
        
        # Find applicable active experiments
        for experiment_id, experiment in self.active_experiments.items():
            # Check if task type matches
            if experiment.task_types and task_type not in experiment.task_types:
                continue
            
            # Check if user segments match (if specified)
            if experiment.user_segments:
                user_segment = context.get("user_segment", "default") if context else "default"
                if user_segment not in experiment.user_segments:
                    continue
            
            applicable_experiments.append(experiment)
        
        if not applicable_experiments:
            return None
        
        # For multiple experiments, choose the highest priority one
        # (In this simple implementation, choose the first one)
        experiment = applicable_experiments[0]
        
        # Check if user is already assigned to this experiment
        if user_id in self.user_assignments:
            if experiment.experiment_id in self.user_assignments[user_id]:
                variant_id = self.user_assignments[user_id][experiment.experiment_id]
                return experiment.experiment_id, variant_id
        
        # Assign user to variant based on traffic allocation
        variant = self._select_variant_for_user(user_id, experiment)
        
        # Store assignment
        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = {}
        
        self.user_assignments[user_id][experiment.experiment_id] = variant.variant_id
        
        logger.debug(
            "Assigned user to experiment variant",
            user_id=user_id,
            experiment_id=experiment.experiment_id,
            variant_id=variant.variant_id,
            variant_name=variant.name
        )
        
        return experiment.experiment_id, variant.variant_id
    
    def _select_variant_for_user(self, user_id: str, experiment: ExperimentConfig) -> ExperimentVariant:
        """Select a variant for a user based on traffic allocation."""
        # Use deterministic SHA-256 hash for consistent assignment across processes
        hash_input = f"{user_id}:{experiment.experiment_id}".encode('utf-8')
        hash_value = int(hashlib.sha256(hash_input).hexdigest(), 16) % 10000 / 10000.0  # Convert to 0-1 range
        
        # Select variant based on traffic allocation
        cumulative_allocation = 0.0
        for variant in experiment.variants:
            cumulative_allocation += variant.traffic_allocation
            if hash_value <= cumulative_allocation:
                return variant
        
        # Fallback to last variant
        return experiment.variants[-1]
    
    async def record_experiment_result(
        self,
        experiment_id: str,
        variant_id: str,
        metrics: Dict[str, float],
        success: bool = True
    ) -> None:
        """Record results for an experiment variant."""
        if experiment_id not in self.active_experiments:
            return
        
        experiment = self.active_experiments[experiment_id]
        
        # Find the variant
        variant = None
        for v in experiment.variants:
            if v.variant_id == variant_id:
                variant = v
                break
        
        if not variant:
            logger.warning("Variant not found", experiment_id=experiment_id, variant_id=variant_id)
            return
        
        # Update variant metrics
        variant.total_requests += 1
        if success:
            variant.successful_requests += 1
        
        # Record metric samples
        if "latency" in metrics:
            variant.latency_samples.append(metrics["latency"])
        if "quality" in metrics:
            variant.quality_samples.append(metrics["quality"])
        if "cost" in metrics:
            variant.cost_samples.append(metrics["cost"])
        if "satisfaction" in metrics:
            variant.satisfaction_samples.append(metrics["satisfaction"])
        
        # Update sample size
        variant.sample_size = len(variant.quality_samples)  # Use quality as primary sample count
        
        # Update means
        if variant.latency_samples:
            variant.mean_latency = statistics.mean(variant.latency_samples)
        if variant.quality_samples:
            variant.mean_quality = statistics.mean(variant.quality_samples)
        if variant.cost_samples:
            variant.mean_cost = statistics.mean(variant.cost_samples)
        if variant.satisfaction_samples:
            variant.mean_satisfaction = statistics.mean(variant.satisfaction_samples)
        
        # Check if experiment should be completed
        await self._check_experiment_completion(experiment_id)
        
        logger.debug(
            "Recorded experiment result",
            experiment_id=experiment_id,
            variant_id=variant_id,
            sample_size=variant.sample_size,
            success=success
        )
    
    async def _check_experiment_completion(self, experiment_id: str) -> None:
        """Check if an experiment should be completed."""
        if experiment_id not in self.active_experiments:
            return
        
        experiment = self.active_experiments[experiment_id]
        
        # Check minimum sample size
        min_sample_reached = all(
            variant.sample_size >= experiment.minimum_sample_size 
            for variant in experiment.variants
        )
        
        # Check maximum duration
        max_duration_reached = False
        if experiment.started_at:
            duration = datetime.now() - experiment.started_at
            max_duration_reached = duration.days >= experiment.maximum_duration_days
        
        # Check for early stopping due to statistical significance
        early_stopping = False
        if min_sample_reached and all(v.sample_size >= 20 for v in experiment.variants):
            result = await self._calculate_experiment_result(experiment)
            if result.statistical_significance in [StatisticalSignificance.SIGNIFICANT, 
                                                  StatisticalSignificance.HIGHLY_SIGNIFICANT]:
                early_stopping = True
        
        # Complete experiment if conditions are met
        if min_sample_reached and (max_duration_reached or early_stopping):
            await self.complete_experiment(experiment_id)
    
    async def complete_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Complete an A/B test experiment and generate results."""
        if experiment_id not in self.active_experiments:
            logger.error("Active experiment not found", experiment_id=experiment_id)
            return None
        
        experiment = self.active_experiments[experiment_id]
        
        # Calculate final results
        result = await self._calculate_experiment_result(experiment)
        
        # Update experiment status
        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = datetime.now()
        
        # Remove from active experiments
        del self.active_experiments[experiment_id]
        
        # Store results
        self.experiment_results[experiment_id] = result
        
        logger.info(
            "Completed A/B test experiment",
            experiment_id=experiment_id,
            winning_variant=result.winning_variant,
            statistical_significance=result.statistical_significance.value,
            p_value=result.p_value
        )
        
        return result
    
    async def _calculate_experiment_result(self, experiment: ExperimentConfig) -> ExperimentResult:
        """Calculate statistical results for an experiment."""
        result = ExperimentResult(experiment_id=experiment.experiment_id)
        
        # Get primary metric samples for each variant
        primary_metric = experiment.primary_metric
        variant_samples = {}
        
        for variant in experiment.variants:
            if primary_metric == "latency":
                samples = variant.latency_samples
            elif primary_metric == "quality":
                samples = variant.quality_samples
            elif primary_metric == "cost":
                samples = variant.cost_samples
            elif primary_metric == "satisfaction":
                samples = variant.satisfaction_samples
            else:
                samples = variant.quality_samples  # Default to quality
            
            if samples:
                variant_samples[variant.variant_id] = samples
                
                # Store detailed results
                result.variant_results[variant.variant_id] = {
                    "sample_size": len(samples),
                    "mean": statistics.mean(samples),
                    "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                    "success_rate": variant.successful_requests / max(variant.total_requests, 1)
                }
        
        if len(variant_samples) < 2:
            result.recommendation = "Insufficient data for statistical comparison"
            return result
        
        # Perform appropriate statistical test based on number of variants
        variant_ids = list(variant_samples.keys())
        valid_samples = {vid: samples for vid, samples in variant_samples.items() if len(samples) > 1}
        
        if len(valid_samples) < 2:
            result.recommendation = "Insufficient data for statistical comparison (need at least 2 samples per variant)"
            return result
        
        if len(valid_samples) == 2:
            # Use t-test for exactly two variants
            variant_ids = list(valid_samples.keys())
            samples_a = valid_samples[variant_ids[0]]
            samples_b = valid_samples[variant_ids[1]]
            
            # Calculate t-statistic and p-value
            mean_a = statistics.mean(samples_a)
            mean_b = statistics.mean(samples_b)
            std_a = statistics.stdev(samples_a)
            std_b = statistics.stdev(samples_b)
            n_a = len(samples_a)
            n_b = len(samples_b)
            
            # Pooled standard error
            pooled_se = math.sqrt((std_a ** 2 / n_a) + (std_b ** 2 / n_b))
            
            if pooled_se > 0:
                t_stat = abs(mean_a - mean_b) / pooled_se
                
                # Use scipy's t-distribution CDF for accurate p-value
                degrees_of_freedom = n_a + n_b - 2
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=degrees_of_freedom))
                
                result.p_value = p_value
                result.effect_size = abs(mean_a - mean_b) / max(std_a, std_b, 0.001)
                
                # Determine winning variant
                if primary_metric == "cost":  # Lower is better for cost
                    result.winning_variant = variant_ids[0] if mean_a < mean_b else variant_ids[1]
                elif primary_metric == "latency":  # Lower is better for latency
                    result.winning_variant = variant_ids[0] if mean_a < mean_b else variant_ids[1]
                else:  # Higher is better for quality/satisfaction
                    result.winning_variant = variant_ids[0] if mean_a > mean_b else variant_ids[1]
        
        else:
            # Use ANOVA for multiple variants (more than 2)
            sample_groups = list(valid_samples.values())
            
            # Perform one-way ANOVA
            f_stat, p_value = stats.f_oneway(*sample_groups)
            result.p_value = p_value
            
            # Calculate effect size using eta-squared
            # First, calculate overall mean and sum of squares
            all_values = [val for samples in sample_groups for val in samples]
            overall_mean = statistics.mean(all_values)
            total_ss = sum((val - overall_mean) ** 2 for val in all_values)
            
            # Calculate between-group sum of squares
            between_ss = sum(len(samples) * (statistics.mean(samples) - overall_mean) ** 2 
                           for samples in sample_groups)
            
            result.effect_size = between_ss / total_ss if total_ss > 0 else 0.0
            
            # For multiple variants, determine the best performing variant
            if primary_metric == "cost":  # Lower is better
                best_variant_id = min(valid_samples.keys(), 
                                    key=lambda vid: statistics.mean(valid_samples[vid]))
            elif primary_metric == "latency":  # Lower is better
                best_variant_id = min(valid_samples.keys(), 
                                    key=lambda vid: statistics.mean(valid_samples[vid]))
            else:  # Higher is better for quality/satisfaction
                best_variant_id = max(valid_samples.keys(), 
                                    key=lambda vid: statistics.mean(valid_samples[vid]))
            
            result.winning_variant = best_variant_id
        
        # Determine statistical significance (common for both t-test and ANOVA)
        if result.p_value <= 0.001:
            result.statistical_significance = StatisticalSignificance.HIGHLY_SIGNIFICANT
        elif result.p_value <= 0.01:
            result.statistical_significance = StatisticalSignificance.SIGNIFICANT
        elif result.p_value <= 0.05:
            result.statistical_significance = StatisticalSignificance.MARGINAL
        else:
            result.statistical_significance = StatisticalSignificance.NOT_SIGNIFICANT
        
        # Generate recommendations
        result.reasoning = self._generate_recommendations(experiment, result)
        
        # Calculate rollout percentage
        if result.statistical_significance in [StatisticalSignificance.SIGNIFICANT, 
                                             StatisticalSignificance.HIGHLY_SIGNIFICANT]:
            if result.effect_size > 0.2:  # Large effect
                result.suggested_rollout_percentage = 1.0
                result.rollout_timeline = "immediate"
            elif result.effect_size > 0.1:  # Medium effect
                result.suggested_rollout_percentage = 0.5
                result.rollout_timeline = "gradual_over_1_week"
            else:  # Small effect
                result.suggested_rollout_percentage = 0.25
                result.rollout_timeline = "gradual_over_2_weeks"
        
        return result
    
    
    def _generate_recommendations(
        self, 
        experiment: ExperimentConfig, 
        result: ExperimentResult
    ) -> List[str]:
        """Generate recommendations based on experiment results."""
        recommendations = []
        
        if result.statistical_significance == StatisticalSignificance.NOT_SIGNIFICANT:
            recommendations.append("No statistically significant difference found between variants")
            recommendations.append("Consider running the experiment longer or with more traffic")
            
            # Check if one variant is trending better
            if result.variant_results:
                variant_means = {
                    vid: data["mean"] 
                    for vid, data in result.variant_results.items()
                }
                best_variant = max(variant_means.items(), key=lambda x: x[1])
                recommendations.append(f"Variant {best_variant[0]} shows slight advantage but needs more data")
        
        elif result.winning_variant:
            winning_data = result.variant_results[result.winning_variant]
            recommendations.append(f"Variant {result.winning_variant} is the clear winner")
            recommendations.append(
                f"Improvement: {result.effect_size:.1%} with {result.statistical_significance.value} significance"
            )
            
            # Add context-specific recommendations
            if experiment.experiment_type == ExperimentType.MODEL_COMPARISON:
                recommendations.append("Consider rolling out the winning model to all users")
            elif experiment.experiment_type == ExperimentType.COST_OPTIMIZATION:
                recommendations.append("Winning variant provides better cost efficiency")
            elif experiment.experiment_type == ExperimentType.LATENCY_OPTIMIZATION:
                recommendations.append("Winning variant provides faster response times")
        
        # Add general recommendations based on sample size
        total_samples = sum(data["sample_size"] for data in result.variant_results.values())
        if total_samples < 200:
            recommendations.append("Consider collecting more data for higher confidence")
        
        return recommendations
    
    async def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an experiment."""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        status = {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "experiment_type": experiment.experiment_type.value,
            "created_at": experiment.created_at.isoformat(),
            "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
            "completed_at": experiment.completed_at.isoformat() if experiment.completed_at else None,
            "variants": []
        }
        
        for variant in experiment.variants:
            variant_status = {
                "variant_id": variant.variant_id,
                "name": variant.name,
                "traffic_allocation": variant.traffic_allocation,
                "sample_size": variant.sample_size,
                "total_requests": variant.total_requests,
                "success_rate": variant.successful_requests / max(variant.total_requests, 1),
                "mean_latency": variant.mean_latency,
                "mean_quality": variant.mean_quality,
                "mean_cost": variant.mean_cost
            }
            status["variants"].append(variant_status)
        
        # Add results if completed
        if experiment_id in self.experiment_results:
            result = self.experiment_results[experiment_id]
            status["results"] = {
                "winning_variant": result.winning_variant,
                "statistical_significance": result.statistical_significance.value,
                "p_value": result.p_value,
                "effect_size": result.effect_size,
                "recommendation": result.recommendation,
                "suggested_rollout_percentage": result.suggested_rollout_percentage
            }
        
        return status
    
    async def pause_experiment(self, experiment_id: str) -> bool:
        """Pause an active experiment."""
        if experiment_id not in self.active_experiments:
            return False
        
        experiment = self.active_experiments[experiment_id]
        experiment.status = ExperimentStatus.PAUSED
        
        logger.info("Paused experiment", experiment_id=experiment_id)
        return True
    
    async def resume_experiment(self, experiment_id: str) -> bool:
        """Resume a paused experiment."""
        if experiment_id not in self.active_experiments:
            return False
        
        experiment = self.active_experiments[experiment_id]
        if experiment.status != ExperimentStatus.PAUSED:
            return False
        
        experiment.status = ExperimentStatus.ACTIVE
        
        logger.info("Resumed experiment", experiment_id=experiment_id)
        return True
    
    async def terminate_experiment(self, experiment_id: str, reason: str = "") -> bool:
        """Terminate an experiment before completion."""
        if experiment_id not in self.active_experiments:
            return False
        
        experiment = self.active_experiments[experiment_id]
        experiment.status = ExperimentStatus.TERMINATED
        experiment.completed_at = datetime.now()
        
        # Remove from active experiments
        del self.active_experiments[experiment_id]
        
        logger.info("Terminated experiment", experiment_id=experiment_id, reason=reason)
        return True
    
    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Get summary of all experiments."""
        experiments = []
        
        for experiment_id, experiment in self.experiments.items():
            exp_summary = {
                "experiment_id": experiment_id,
                "name": experiment.name,
                "status": experiment.status.value,
                "experiment_type": experiment.experiment_type.value,
                "variant_count": len(experiment.variants),
                "created_at": experiment.created_at.isoformat(),
                "total_samples": sum(v.sample_size for v in experiment.variants)
            }
            
            if experiment_id in self.experiment_results:
                result = self.experiment_results[experiment_id]
                exp_summary["winning_variant"] = result.winning_variant
                exp_summary["statistical_significance"] = result.statistical_significance.value
            
            experiments.append(exp_summary)
        
        return experiments
    
    async def export_experiment_data(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Export detailed experiment data for analysis."""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        export_data = {
            "experiment_config": {
                "experiment_id": experiment_id,
                "name": experiment.name,
                "description": experiment.description,
                "experiment_type": experiment.experiment_type.value,
                "task_types": [tt.value for tt in experiment.task_types],
                "minimum_sample_size": experiment.minimum_sample_size,
                "maximum_duration_days": experiment.maximum_duration_days,
                "primary_metric": experiment.primary_metric,
                "status": experiment.status.value,
                "created_at": experiment.created_at.isoformat(),
                "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
                "completed_at": experiment.completed_at.isoformat() if experiment.completed_at else None
            },
            "variants": [],
            "raw_data": {}
        }
        
        for variant in experiment.variants:
            variant_data = {
                "variant_id": variant.variant_id,
                "name": variant.name,
                "description": variant.description,
                "provider_type": variant.provider_type.value if variant.provider_type else None,
                "model_id": variant.model_id,
                "strategy_config": variant.strategy_config,
                "traffic_allocation": variant.traffic_allocation,
                "sample_size": variant.sample_size,
                "total_requests": variant.total_requests,
                "successful_requests": variant.successful_requests,
                "mean_latency": variant.mean_latency,
                "mean_quality": variant.mean_quality,
                "mean_cost": variant.mean_cost,
                "mean_satisfaction": variant.mean_satisfaction
            }
            export_data["variants"].append(variant_data)
            
            # Include raw sample data
            export_data["raw_data"][variant.variant_id] = {
                "latency_samples": variant.latency_samples,
                "quality_samples": variant.quality_samples,
                "cost_samples": variant.cost_samples,
                "satisfaction_samples": variant.satisfaction_samples
            }
        
        # Include results if available
        if experiment_id in self.experiment_results:
            result = self.experiment_results[experiment_id]
            export_data["results"] = {
                "winning_variant": result.winning_variant,
                "statistical_significance": result.statistical_significance.value,
                "p_value": result.p_value,
                "effect_size": result.effect_size,
                "confidence_level": result.confidence_level,
                "variant_results": result.variant_results,
                "recommendation": result.recommendation,
                "reasoning": result.reasoning,
                "suggested_rollout_percentage": result.suggested_rollout_percentage,
                "rollout_timeline": result.rollout_timeline
            }
        
        return export_data
    
    async def cleanup_completed_experiments(self, retention_days: int = 90) -> int:
        """Clean up old completed experiments."""
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        cleaned_count = 0
        
        experiment_ids_to_remove = []
        
        for experiment_id, experiment in self.experiments.items():
            if experiment.status in [ExperimentStatus.COMPLETED, ExperimentStatus.TERMINATED]:
                if experiment.completed_at and experiment.completed_at < cutoff_time:
                    experiment_ids_to_remove.append(experiment_id)
        
        # Remove old experiments
        for experiment_id in experiment_ids_to_remove:
            del self.experiments[experiment_id]
            if experiment_id in self.experiment_results:
                del self.experiment_results[experiment_id]
            cleaned_count += 1
        
        # Clean up user assignments for removed experiments
        for user_id in list(self.user_assignments.keys()):
            user_experiments = self.user_assignments[user_id]
            for experiment_id in experiment_ids_to_remove:
                if experiment_id in user_experiments:
                    del user_experiments[experiment_id]
            
            # Remove user if no active assignments
            if not user_experiments:
                del self.user_assignments[user_id]
        
        # Clear statistical cache
        self.statistical_cache.clear()
        self.assignment_cache.clear()
        
        logger.info(
            "Cleaned up completed experiments",
            cleaned_count=cleaned_count,
            retention_days=retention_days,
            remaining_experiments=len(self.experiments)
        )
        
        return cleaned_count