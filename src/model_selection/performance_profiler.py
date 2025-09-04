"""Performance profiling and benchmarking framework for AI model selection."""

import asyncio
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import structlog
from ..ai_providers.models import ProviderType, ModelSpec

logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """Types of performance metrics to track."""
    
    LATENCY = "latency"              # Response time in milliseconds
    THROUGHPUT = "throughput"        # Tokens per second
    COST = "cost"                    # USD per request
    QUALITY = "quality"              # Quality score (0.0-1.0)
    ERROR_RATE = "error_rate"        # Error percentage
    TOKEN_EFFICIENCY = "token_efficiency"  # Output tokens / input tokens
    CONTEXT_UTILIZATION = "context_utilization"  # Used context / max context
    SUCCESS_RATE = "success_rate"    # Successful requests percentage


@dataclass
class PerformanceMetric:
    """A single performance measurement."""
    
    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: str = field(default_factory=lambda: str(uuid4()))
    provider_type: Optional[ProviderType] = None
    model_id: Optional[str] = None
    task_type: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformanceProfile:
    """Performance profile for a specific model."""
    
    provider_type: ProviderType
    model_id: str
    
    # Aggregated metrics
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    avg_throughput: float = 0.0
    avg_cost_per_request: float = 0.0
    avg_quality_score: float = 0.0
    error_rate: float = 0.0
    success_rate: float = 0.0
    
    # Task-specific performance
    task_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Time-based trends
    hourly_performance: Dict[int, Dict[str, float]] = field(default_factory=dict)
    daily_trends: Dict[str, List[float]] = field(default_factory=dict)
    
    # Metadata
    total_requests: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.0  # Based on sample size and variance


class PerformanceBenchmark:
    """Benchmarking framework for model performance comparison."""
    
    def __init__(self, retention_days: int = 30, max_samples_per_model: int = 1000):
        self.retention_days = retention_days
        self.max_samples_per_model = max_samples_per_model
        
        # Storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples_per_model))
        self.model_profiles: Dict[str, ModelPerformanceProfile] = {}
        self.task_benchmarks: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Real-time tracking
        self.current_requests: Dict[str, Dict[str, Any]] = {}
        self.performance_cache: Dict[str, Tuple[datetime, Dict[str, float]]] = {}
        
        # Statistics tracking
        self.quality_thresholds = {
            "excellent": 0.9,
            "good": 0.7,
            "acceptable": 0.5,
            "poor": 0.3
        }
        
        self.latency_targets = {
            "immediate": 500,    # ms
            "fast": 2000,       # ms
            "standard": 5000,   # ms
            "relaxed": 15000    # ms
        }
    
    async def start_request_tracking(
        self,
        request_id: str,
        provider_type: ProviderType,
        model_id: str,
        task_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Start tracking performance for a request."""
        self.current_requests[request_id] = {
            "provider_type": provider_type,
            "model_id": model_id,
            "task_type": task_type,
            "context": context or {},
            "start_time": time.time(),
            "metrics": {}
        }
        
        logger.debug(
            "Started request tracking",
            request_id=request_id,
            provider=provider_type.value,
            model=model_id,
            task_type=task_type
        )
    
    async def record_metric(
        self,
        request_id: str,
        metric_type: MetricType,
        value: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a performance metric for a tracked request."""
        if request_id not in self.current_requests:
            logger.warning("Attempting to record metric for untracked request", request_id=request_id)
            return
        
        request_data = self.current_requests[request_id]
        
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            request_id=request_id,
            provider_type=request_data["provider_type"],
            model_id=request_data["model_id"],
            task_type=request_data["task_type"],
            context={**request_data["context"], **(context or {})}
        )
        
        # Store metric
        model_key = f"{request_data['provider_type'].value}:{request_data['model_id']}"
        self.metrics_history[model_key].append(metric)
        
        # Update current request
        request_data["metrics"][metric_type] = value
        
        logger.debug(
            "Recorded performance metric",
            request_id=request_id,
            metric_type=metric_type.value,
            value=value
        )
    
    async def complete_request_tracking(
        self,
        request_id: str,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> Dict[str, float]:
        """Complete tracking for a request and return final metrics."""
        if request_id not in self.current_requests:
            logger.warning("Attempting to complete tracking for untracked request", request_id=request_id)
            return {}
        
        request_data = self.current_requests[request_id]
        
        # Calculate final latency if not already recorded
        if MetricType.LATENCY not in request_data["metrics"]:
            latency = (time.time() - request_data["start_time"]) * 1000  # Convert to ms
            await self.record_metric(request_id, MetricType.LATENCY, latency)
        
        # Record success/failure
        await self.record_metric(
            request_id,
            MetricType.SUCCESS_RATE,
            1.0 if success else 0.0,
            {"error_message": error_message} if error_message else None
        )
        
        # Update model profile
        model_key = f"{request_data['provider_type'].value}:{request_data['model_id']}"
        await self._update_model_profile(model_key, request_data)
        
        # Get final metrics
        final_metrics = request_data["metrics"].copy()
        
        # Cleanup
        del self.current_requests[request_id]
        
        logger.info(
            "Completed request tracking",
            request_id=request_id,
            success=success,
            metrics=final_metrics
        )
        
        return final_metrics
    
    async def _update_model_profile(self, model_key: str, request_data: Dict[str, Any]) -> None:
        """Update the performance profile for a model."""
        provider_type = request_data["provider_type"]
        model_id = request_data["model_id"]
        task_type = request_data["task_type"]
        
        if model_key not in self.model_profiles:
            self.model_profiles[model_key] = ModelPerformanceProfile(
                provider_type=provider_type,
                model_id=model_id
            )
        
        profile = self.model_profiles[model_key]
        metrics = [m for m in self.metrics_history[model_key]]
        
        if not metrics:
            return
        
        # Calculate aggregated metrics
        latencies = [m.value for m in metrics if m.metric_type == MetricType.LATENCY]
        throughputs = [m.value for m in metrics if m.metric_type == MetricType.THROUGHPUT]
        costs = [m.value for m in metrics if m.metric_type == MetricType.COST]
        qualities = [m.value for m in metrics if m.metric_type == MetricType.QUALITY]
        successes = [m.value for m in metrics if m.metric_type == MetricType.SUCCESS_RATE]
        
        if latencies:
            profile.avg_latency = statistics.mean(latencies)
            if len(latencies) >= 20:
                profile.p95_latency = sorted_latencies[max(p95_index, len(sorted_latencies) - 1)]
        
        if throughputs:
            profile.avg_throughput = statistics.mean(throughputs)
        
        if costs:
            profile.avg_cost_per_request = statistics.mean(costs)
        
        if qualities:
            profile.avg_quality_score = statistics.mean(qualities)
        
        if successes:
            profile.success_rate = statistics.mean(successes)
            profile.error_rate = 1.0 - profile.success_rate
        
        # Task-specific performance
        task_metrics = [m for m in metrics if m.task_type == task_type]
        if task_metrics:
            if task_type not in profile.task_performance:
                profile.task_performance[task_type] = {}
            
            task_latencies = [m.value for m in task_metrics if m.metric_type == MetricType.LATENCY]
            task_qualities = [m.value for m in task_metrics if m.metric_type == MetricType.QUALITY]
            
            if task_latencies:
                profile.task_performance[task_type]["avg_latency"] = statistics.mean(task_latencies)
            if task_qualities:
                profile.task_performance[task_type]["avg_quality"] = statistics.mean(task_qualities)
        
        # Time-based trends
        current_hour = datetime.now().hour
        if current_hour not in profile.hourly_performance:
            profile.hourly_performance[current_hour] = {}
        
        hour_metrics = [m for m in metrics if m.timestamp.hour == current_hour and 
                       m.timestamp.date() == datetime.now().date()]
        
        if hour_metrics:
            hour_latencies = [m.value for m in hour_metrics if m.metric_type == MetricType.LATENCY]
            if hour_latencies:
                profile.hourly_performance[current_hour]["avg_latency"] = statistics.mean(hour_latencies)
        
        # Update metadata
        profile.total_requests = len(metrics)
        profile.last_updated = datetime.now()
        profile.confidence_score = min(1.0, len(metrics) / 100)  # 100 samples = full confidence
    
    def get_model_performance(
        self,
        provider_type: ProviderType,
        model_id: str,
        task_type: Optional[str] = None
    ) -> Optional[ModelPerformanceProfile]:
        """Get performance profile for a specific model."""
        model_key = f"{provider_type.value}:{model_id}"
        profile = self.model_profiles.get(model_key)
        
        if not profile:
            return None
        
        # Return task-specific subset if requested
        if task_type and task_type in profile.task_performance:
            # Create a copy with task-specific metrics
            task_profile = ModelPerformanceProfile(
                provider_type=provider_type,
                model_id=model_id,
                avg_latency=profile.task_performance[task_type].get("avg_latency", profile.avg_latency),
                avg_quality_score=profile.task_performance[task_type].get("avg_quality", profile.avg_quality_score),
                total_requests=len([m for m in self.metrics_history[model_key] if m.task_type == task_type]),
                confidence_score=profile.confidence_score
            )
            return task_profile
        
        return profile
    
    def compare_models(
        self,
        models: List[Tuple[ProviderType, str]],
        task_type: Optional[str] = None,
        metric_weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """Compare multiple models and return ranked results.
        
        Args:
            models: List of (provider_type, model_id) tuples
            task_type: Optional task type for specific comparison
            metric_weights: Optional weights for different metrics
            
        Returns:
            List of (model_key, overall_score, detailed_metrics)
        """
        default_weights = {
            "latency": 0.3,
            "quality": 0.3,
            "cost": 0.2,
            "success_rate": 0.2
        }
        weights = metric_weights or default_weights
        
        results = []
        
        for provider_type, model_id in models:
            profile = self.get_model_performance(provider_type, model_id, task_type)
            
            if not profile:
                continue
            
            # Normalize metrics for comparison
            normalized_latency = max(0, 1 - (profile.avg_latency / 10000))  # 10s = 0 score
            normalized_quality = profile.avg_quality_score
            normalized_cost = max(0, 1 - (profile.avg_cost_per_request / 1.0))  # $1 = 0 score
            normalized_success = profile.success_rate
            
            # Calculate weighted score
            overall_score = (
                normalized_latency * weights.get("latency", 0) +
                normalized_quality * weights.get("quality", 0) +
                normalized_cost * weights.get("cost", 0) +
                normalized_success * weights.get("success_rate", 0)
            )
            
            # Apply confidence penalty
            overall_score *= profile.confidence_score
            
            detailed_metrics = {
                "latency": profile.avg_latency,
                "quality": profile.avg_quality_score,
                "cost": profile.avg_cost_per_request,
                "success_rate": profile.success_rate,
                "confidence": profile.confidence_score,
                "total_requests": profile.total_requests
            }
            
            model_key = f"{provider_type.value}:{model_id}"
            results.append((model_key, overall_score, detailed_metrics))
        
        # Sort by overall score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(
            "Model comparison completed",
            models_count=len(models),
            task_type=task_type,
            top_model=results[0][0] if results else None
        )
        
        return results
    
    def get_performance_insights(
        self,
        provider_type: Optional[ProviderType] = None,
        task_type: Optional[str] = None,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """Get performance insights and recommendations."""
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        
        insights = {
            "summary": {},
            "trends": {},
            "recommendations": [],
            "alerts": []
        }
        
        # Filter models based on criteria
        relevant_models = []
        for model_key, profile in self.model_profiles.items():
            if provider_type and profile.provider_type != provider_type:
                continue
            
            # Check if model has recent activity
            recent_metrics = [
                m for m in self.metrics_history[model_key]
                if m.timestamp > cutoff_time
            ]
            
            if recent_metrics:
                relevant_models.append((model_key, profile, recent_metrics))
        
        if not relevant_models:
            return insights
        
        # Summary statistics
        all_latencies = []
        all_qualities = []
        all_costs = []
        
        for model_key, profile, recent_metrics in relevant_models:
            latencies = [m.value for m in recent_metrics if m.metric_type == MetricType.LATENCY]
            qualities = [m.value for m in recent_metrics if m.metric_type == MetricType.QUALITY]
            costs = [m.value for m in recent_metrics if m.metric_type == MetricType.COST]
            
            all_latencies.extend(latencies)
            all_qualities.extend(qualities)
            all_costs.extend(costs)
        
        if all_latencies:
            insights["summary"]["avg_latency"] = statistics.mean(all_latencies)
            insights["summary"]["p95_latency"] = statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) >= 20 else max(all_latencies)
        
        if all_qualities:
            insights["summary"]["avg_quality"] = statistics.mean(all_qualities)
        
        if all_costs:
            insights["summary"]["avg_cost"] = statistics.mean(all_costs)
        
        # Trends analysis
        for model_key, profile, recent_metrics in relevant_models:
            if len(recent_metrics) < 10:  # Need sufficient data
                continue
            
            # Latency trend
            latency_values = [m.value for m in recent_metrics if m.metric_type == MetricType.LATENCY]
            if len(latency_values) >= 10:
                recent_avg = statistics.mean(latency_values[-5:])  # Last 5 requests
                earlier_avg = statistics.mean(latency_values[:5])   # First 5 requests
                trend = (recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0
                
                insights["trends"][model_key] = {
                    "latency_trend": trend,
                    "improving": trend < -0.1,  # 10% improvement
                    "degrading": trend > 0.1    # 10% degradation
                }
        
        # Recommendations
        best_models = self.compare_models(
            [(p.provider_type, p.model_id) for _, p, _ in relevant_models],
            task_type=task_type
        )
        
        if best_models:
            best_model = best_models[0]
            insights["recommendations"].append({
                "type": "best_model",
                "message": f"Best performing model: {best_model[0]}",
                "score": best_model[1],
                "metrics": best_model[2]
            })
        
        # Performance alerts
        for model_key, profile, recent_metrics in relevant_models:
            # High error rate alert
            if profile.error_rate > 0.1:  # > 10% error rate
                insights["alerts"].append({
                    "type": "high_error_rate",
                    "model": model_key,
                    "error_rate": profile.error_rate,
                    "severity": "high" if profile.error_rate > 0.25 else "medium"
                })
            
            # High latency alert
            if profile.avg_latency > 10000:  # > 10 seconds
                insights["alerts"].append({
                    "type": "high_latency",
                    "model": model_key,
                    "avg_latency": profile.avg_latency,
                    "severity": "high" if profile.avg_latency > 20000 else "medium"
                })
            
            # Low quality alert
            if profile.avg_quality_score < 0.5:
                insights["alerts"].append({
                    "type": "low_quality",
                    "model": model_key,
                    "quality_score": profile.avg_quality_score,
                    "severity": "medium"
                })
        
        return insights
    
    def export_benchmarks(self) -> Dict[str, Any]:
        """Export all benchmark data for analysis or backup."""
        return {
            "model_profiles": {
                key: {
                    "provider_type": profile.provider_type.value,
                    "model_id": profile.model_id,
                    "avg_latency": profile.avg_latency,
                    "p95_latency": profile.p95_latency,
                    "avg_throughput": profile.avg_throughput,
                    "avg_cost_per_request": profile.avg_cost_per_request,
                    "avg_quality_score": profile.avg_quality_score,
                    "error_rate": profile.error_rate,
                    "success_rate": profile.success_rate,
                    "task_performance": profile.task_performance,
                    "total_requests": profile.total_requests,
                    "confidence_score": profile.confidence_score,
                    "last_updated": profile.last_updated.isoformat()
                }
                for key, profile in self.model_profiles.items()
            },
            "export_timestamp": datetime.now().isoformat(),
            "retention_days": self.retention_days,
            "total_metrics": sum(len(deque_obj) for deque_obj in self.metrics_history.values())
        }
    
    async def cleanup_old_data(self) -> int:
        """Clean up old performance data based on retention policy."""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        removed_count = 0
        
        for model_key in list(self.metrics_history.keys()):
            metrics_deque = self.metrics_history[model_key]
            original_length = len(metrics_deque)
            
            # Filter out old metrics
            filtered_metrics = deque(
                (m for m in metrics_deque if m.timestamp > cutoff_time),
                maxlen=self.max_samples_per_model
            )
            
            self.metrics_history[model_key] = filtered_metrics
            removed_count += original_length - len(filtered_metrics)
            
            # If no recent metrics, remove the model profile
            if len(filtered_metrics) == 0:
                if model_key in self.model_profiles:
                    del self.model_profiles[model_key]
        
        # Clear performance cache
        self.performance_cache.clear()
        
        logger.info(
            "Performance data cleanup completed",
            removed_metrics=removed_count,
            active_models=len(self.model_profiles)
        )
        
        return removed_count