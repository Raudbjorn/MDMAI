"""
Advanced Cost Optimization Architecture with Real-Time Tracking
Task 25.3: Develop Provider Router with Fallback
"""

import asyncio
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Deque

from structlog import get_logger

from .models import (
    AIRequest,
    AIResponse,
    ProviderType,
    CostTier,
    ModelSpec,
)

logger = get_logger(__name__)


class OptimizationStrategy(Enum):
    """Cost optimization strategies."""
    
    MINIMIZE_COST = "minimize_cost"
    COST_QUALITY_BALANCE = "cost_quality_balance"
    COST_SPEED_BALANCE = "cost_speed_balance"
    DYNAMIC_ARBITRAGE = "dynamic_arbitrage"
    BUDGET_AWARE = "budget_aware"
    PREDICTIVE_SCALING = "predictive_scaling"


class AlertSeverity(Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class CostMetrics:
    """Comprehensive cost tracking metrics."""
    
    # Time-based costs
    hourly_cost: float = 0.0
    daily_cost: float = 0.0
    weekly_cost: float = 0.0
    monthly_cost: float = 0.0
    
    # Request-based costs
    cost_per_request: float = 0.0
    cost_per_token: float = 0.0
    cost_per_successful_request: float = 0.0
    
    # Provider breakdown
    provider_costs: Dict[ProviderType, float] = field(default_factory=dict)
    model_costs: Dict[str, float] = field(default_factory=dict)
    
    # Token metrics
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_tokens_per_request: float = 0.0
    
    # Efficiency metrics
    cost_efficiency_score: float = 0.0
    waste_percentage: float = 0.0
    optimization_savings: float = 0.0
    
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class BudgetAlert:
    """Budget alert configuration and state."""
    
    alert_id: str
    name: str
    budget_type: str  # "daily", "weekly", "monthly", "per_request"
    threshold: float
    current_usage: float
    percentage_used: float
    severity: AlertSeverity
    message: str
    triggered_at: datetime
    acknowledged: bool = False
    auto_actions: List[str] = field(default_factory=list)


@dataclass
class CostPrediction:
    """Cost prediction for future periods."""
    
    prediction_period: str  # "1h", "1d", "1w", "1m"
    predicted_cost: float
    confidence_level: float
    trend: str  # "increasing", "decreasing", "stable"
    factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ArbitrageOpportunity:
    """Provider arbitrage opportunity."""
    
    primary_provider: ProviderType
    alternative_provider: ProviderType
    model_pair: Tuple[str, str]  # (primary_model, alternative_model)
    cost_difference: float
    quality_difference: float
    speed_difference: float
    confidence_score: float
    potential_savings_per_request: float
    detected_at: datetime = field(default_factory=datetime.now)


class AdvancedCostOptimizer:
    """
    Advanced cost optimizer with real-time tracking and predictive capabilities.
    
    Features:
    - Real-time cost tracking and alerts
    - Multi-dimensional optimization strategies
    - Provider arbitrage detection
    - Predictive cost modeling
    - Dynamic budget enforcement
    - Cost-quality tradeoff analysis
    - Automated cost optimization
    - Usage pattern analysis
    """
    
    def __init__(
        self,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.COST_QUALITY_BALANCE,
        enable_real_time_tracking: bool = True,
        enable_predictive_modeling: bool = True,
    ):
        self.optimization_strategy = optimization_strategy
        self.enable_real_time_tracking = enable_real_time_tracking
        self.enable_predictive_modeling = enable_predictive_modeling
        
        # Cost tracking
        self.cost_metrics = CostMetrics()
        self.cost_history: Deque[Tuple[datetime, float]] = deque(maxlen=10000)
        self.request_costs: Deque[Tuple[datetime, str, ProviderType, float]] = deque(maxlen=5000)
        
        # Budget management
        self.budget_limits: Dict[str, float] = {}
        self.budget_alerts: List[BudgetAlert] = []
        self.alert_thresholds = [0.5, 0.7, 0.85, 0.95]  # 50%, 70%, 85%, 95%
        
        # Optimization tracking
        self.provider_performance: Dict[ProviderType, Dict[str, Any]] = defaultdict(dict)
        self.model_performance: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.arbitrage_opportunities: List[ArbitrageOpportunity] = []
        
        # Predictive modeling
        self.usage_patterns: Dict[str, List[float]] = defaultdict(list)  # hourly patterns
        self.cost_predictions: List[CostPrediction] = []
        
        # Background tasks
        self._tracking_task: Optional[asyncio.Task] = None
        self._prediction_task: Optional[asyncio.Task] = None
        self._optimization_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.cost_tracking_interval = 60  # seconds
        self.prediction_interval = 300  # 5 minutes
        self.optimization_interval = 600  # 10 minutes
    
    async def start(self) -> None:
        """Start the advanced cost optimizer."""
        logger.info("Starting advanced cost optimizer")
        
        if self.enable_real_time_tracking:
            self._tracking_task = asyncio.create_task(self._real_time_tracking_loop())
        
        if self.enable_predictive_modeling:
            self._prediction_task = asyncio.create_task(self._predictive_modeling_loop())
        
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        
        logger.info("Advanced cost optimizer started")
    
    async def stop(self) -> None:
        """Stop the advanced cost optimizer."""
        logger.info("Stopping advanced cost optimizer")
        
        for task in [self._tracking_task, self._prediction_task, self._optimization_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Advanced cost optimizer stopped")
    
    async def optimize_request_routing(
        self,
        request: AIRequest,
        available_providers: Dict[ProviderType, List[str]],  # provider -> models
        performance_data: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Optimize request routing based on cost and performance.
        
        Args:
            request: The AI request
            available_providers: Available providers and their models
            performance_data: Real-time performance metrics
            
        Returns:
            Optimization recommendation
        """
        # Analyze request characteristics
        request_analysis = self._analyze_request_cost_impact(request)
        
        # Get provider options with cost estimates
        provider_options = []
        for provider_type, models in available_providers.items():
            for model in models:
                if request.model and model != request.model:
                    continue
                
                option = await self._analyze_provider_option(
                    provider_type, model, request, request_analysis, performance_data
                )
                provider_options.append(option)
        
        # Apply optimization strategy
        if self.optimization_strategy == OptimizationStrategy.MINIMIZE_COST:
            best_option = min(provider_options, key=lambda x: x["estimated_cost"])
        
        elif self.optimization_strategy == OptimizationStrategy.COST_QUALITY_BALANCE:
            best_option = self._optimize_cost_quality_balance(provider_options)
        
        elif self.optimization_strategy == OptimizationStrategy.COST_SPEED_BALANCE:
            best_option = self._optimize_cost_speed_balance(provider_options)
        
        elif self.optimization_strategy == OptimizationStrategy.DYNAMIC_ARBITRAGE:
            best_option = self._find_arbitrage_opportunity(provider_options)
        
        elif self.optimization_strategy == OptimizationStrategy.BUDGET_AWARE:
            best_option = self._optimize_budget_aware(provider_options, request)
        
        else:
            best_option = self._optimize_predictive_scaling(provider_options, request_analysis)
        
        # Generate optimization report
        optimization_report = {
            "recommended_provider": best_option["provider"],
            "recommended_model": best_option["model"],
            "estimated_cost": best_option["estimated_cost"],
            "estimated_savings": self._calculate_savings(provider_options, best_option),
            "optimization_reason": best_option.get("reason", "Cost optimized selection"),
            "confidence_score": best_option.get("confidence", 0.8),
            "alternatives": sorted(provider_options, key=lambda x: x["cost_efficiency_score"], reverse=True)[:3],
            "cost_impact": request_analysis,
        }
        
        return optimization_report
    
    def _analyze_request_cost_impact(self, request: AIRequest) -> Dict[str, Any]:
        """Analyze the cost impact characteristics of a request."""
        # Estimate token usage
        input_tokens = self._estimate_input_tokens(request.messages)
        output_tokens = request.max_tokens or 1000
        
        # Categorize request
        content = " ".join(str(msg.get("content", "")) for msg in request.messages)
        content_length = len(content)
        
        complexity_score = self._assess_complexity(content)
        
        return {
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "content_length": content_length,
            "complexity_score": complexity_score,
            "has_tools": bool(request.tools),
            "requires_streaming": request.stream,
            "session_context": bool(request.session_id),
            "cost_category": self._categorize_cost_impact(input_tokens, output_tokens, complexity_score),
        }
    
    async def _analyze_provider_option(
        self,
        provider_type: ProviderType,
        model: str,
        request: AIRequest,
        request_analysis: Dict[str, Any],
        performance_data: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze a provider/model option for cost optimization."""
        # Get performance metrics
        perf_data = performance_data.get(f"{provider_type.value}:{model}", {})
        
        # Estimate costs
        estimated_cost = self._estimate_request_cost(
            provider_type, model, 
            request_analysis["estimated_input_tokens"],
            request_analysis["estimated_output_tokens"]
        )
        
        # Calculate efficiency metrics
        quality_score = perf_data.get("quality_score", 0.5)
        speed_score = 1.0 / max(1.0, perf_data.get("avg_latency_ms", 3000) / 1000.0)
        reliability_score = perf_data.get("success_rate", 0.9)
        
        # Cost efficiency calculation
        cost_efficiency_score = (quality_score * speed_score * reliability_score) / max(0.001, estimated_cost)
        
        # Provider-specific adjustments
        provider_adjustment = self._get_provider_cost_adjustment(provider_type)
        adjusted_cost = estimated_cost * provider_adjustment
        
        return {
            "provider": provider_type,
            "model": model,
            "estimated_cost": adjusted_cost,
            "base_cost": estimated_cost,
            "quality_score": quality_score,
            "speed_score": speed_score,
            "reliability_score": reliability_score,
            "cost_efficiency_score": cost_efficiency_score,
            "provider_adjustment": provider_adjustment,
            "confidence": min(1.0, perf_data.get("request_count", 0) / 100.0),  # Confidence based on data
        }
    
    def _optimize_cost_quality_balance(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize for cost-quality balance."""
        # Weight cost vs quality (60% cost, 40% quality)
        for option in options:
            cost_score = 1.0 / (1.0 + option["estimated_cost"])  # Inverse cost
            quality_score = option["quality_score"]
            option["balance_score"] = cost_score * 0.6 + quality_score * 0.4
            option["reason"] = "Optimized for cost-quality balance"
        
        return max(options, key=lambda x: x["balance_score"])
    
    def _optimize_cost_speed_balance(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize for cost-speed balance."""
        for option in options:
            cost_score = 1.0 / (1.0 + option["estimated_cost"])
            speed_score = option["speed_score"]
            option["balance_score"] = cost_score * 0.5 + speed_score * 0.5
            option["reason"] = "Optimized for cost-speed balance"
        
        return max(options, key=lambda x: x["balance_score"])
    
    def _find_arbitrage_opportunity(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find provider arbitrage opportunities."""
        # Sort by cost efficiency
        sorted_options = sorted(options, key=lambda x: x["cost_efficiency_score"], reverse=True)
        
        best_option = sorted_options[0]
        
        # Check for significant arbitrage opportunities
        if len(sorted_options) > 1:
            second_best = sorted_options[1]
            cost_difference = second_best["estimated_cost"] - best_option["estimated_cost"]
            quality_difference = best_option["quality_score"] - second_best["quality_score"]
            
            if cost_difference > 0.01 and quality_difference < 0.1:  # Significant savings, minimal quality loss
                arbitrage = ArbitrageOpportunity(
                    primary_provider=second_best["provider"],
                    alternative_provider=best_option["provider"],
                    model_pair=(second_best["model"], best_option["model"]),
                    cost_difference=cost_difference,
                    quality_difference=quality_difference,
                    speed_difference=best_option["speed_score"] - second_best["speed_score"],
                    confidence_score=min(best_option["confidence"], second_best["confidence"]),
                    potential_savings_per_request=cost_difference,
                )
                
                self.arbitrage_opportunities.append(arbitrage)
                best_option["reason"] = f"Arbitrage opportunity: {cost_difference:.4f} savings"
        
        return best_option
    
    def _optimize_budget_aware(self, options: List[Dict[str, Any]], request: AIRequest) -> Dict[str, Any]:
        """Optimize considering budget constraints."""
        # Check current budget usage
        current_daily_cost = self._get_current_period_cost("daily")
        daily_budget = self.budget_limits.get("daily", float('inf'))
        budget_remaining = daily_budget - current_daily_cost
        
        # Filter options by budget constraints
        affordable_options = []
        for option in options:
            if option["estimated_cost"] <= budget_remaining:
                affordable_options.append(option)
            elif len(affordable_options) == 0:  # No affordable options, take cheapest
                affordable_options.append(option)
        
        if not affordable_options:
            affordable_options = [min(options, key=lambda x: x["estimated_cost"])]
        
        # Among affordable options, pick best quality
        best_option = max(affordable_options, key=lambda x: x["quality_score"])
        best_option["reason"] = f"Budget-aware selection (remaining: ${budget_remaining:.4f})"
        
        return best_option
    
    def _optimize_predictive_scaling(
        self, options: List[Dict[str, Any]], request_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize based on predictive scaling."""
        current_hour = datetime.now().hour
        
        # Get predicted load for next few hours
        predicted_load = self._predict_hourly_load(current_hour)
        
        # Adjust selection based on predicted scaling needs
        if predicted_load > 1.5:  # High load predicted
            # Prioritize speed and reliability
            for option in options:
                option["scaling_score"] = (
                    option["speed_score"] * 0.5 + 
                    option["reliability_score"] * 0.3 + 
                    (1.0 / (1.0 + option["estimated_cost"])) * 0.2
                )
        else:  # Normal/low load
            # Prioritize cost optimization
            for option in options:
                option["scaling_score"] = (
                    (1.0 / (1.0 + option["estimated_cost"])) * 0.6 + 
                    option["quality_score"] * 0.4
                )
        
        best_option = max(options, key=lambda x: x["scaling_score"])
        best_option["reason"] = f"Predictive scaling optimization (load factor: {predicted_load:.2f})"
        
        return best_option
    
    async def track_request_cost(
        self,
        request: AIRequest,
        response: AIResponse,
        provider_type: ProviderType,
        model: str,
    ) -> None:
        """Track the actual cost of a completed request."""
        actual_cost = response.cost
        timestamp = datetime.now()
        
        # Update cost history
        self.cost_history.append((timestamp, actual_cost))
        self.request_costs.append((timestamp, request.request_id, provider_type, actual_cost))
        
        # Update provider performance
        provider_perf = self.provider_performance[provider_type]
        provider_perf["total_cost"] = provider_perf.get("total_cost", 0.0) + actual_cost
        provider_perf["request_count"] = provider_perf.get("request_count", 0) + 1
        provider_perf["avg_cost_per_request"] = provider_perf["total_cost"] / provider_perf["request_count"]
        
        # Update model performance
        model_perf = self.model_performance[model]
        model_perf["total_cost"] = model_perf.get("total_cost", 0.0) + actual_cost
        model_perf["request_count"] = model_perf.get("request_count", 0) + 1
        model_perf["avg_cost_per_request"] = model_perf["total_cost"] / model_perf["request_count"]
        
        # Update real-time metrics
        await self._update_cost_metrics()
        
        # Check budget alerts
        await self._check_budget_alerts()
        
        # Log significant cost events
        if actual_cost > 1.0:  # High cost request
            logger.warning(
                "High cost request detected",
                request_id=request.request_id,
                cost=actual_cost,
                provider=provider_type.value,
                model=model,
            )
    
    async def _update_cost_metrics(self) -> None:
        """Update comprehensive cost metrics."""
        now = datetime.now()
        
        # Calculate time-based costs
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)
        one_week_ago = now - timedelta(weeks=1)
        one_month_ago = now - timedelta(days=30)
        
        self.cost_metrics.hourly_cost = sum(
            cost for timestamp, cost in self.cost_history
            if timestamp >= one_hour_ago
        )
        
        self.cost_metrics.daily_cost = sum(
            cost for timestamp, cost in self.cost_history
            if timestamp >= one_day_ago
        )
        
        self.cost_metrics.weekly_cost = sum(
            cost for timestamp, cost in self.cost_history
            if timestamp >= one_week_ago
        )
        
        self.cost_metrics.monthly_cost = sum(
            cost for timestamp, cost in self.cost_history
            if timestamp >= one_month_ago
        )
        
        # Calculate provider breakdown
        self.cost_metrics.provider_costs = {}
        for timestamp, request_id, provider, cost in self.request_costs:
            if timestamp >= one_day_ago:
                self.cost_metrics.provider_costs[provider] = (
                    self.cost_metrics.provider_costs.get(provider, 0.0) + cost
                )
        
        # Calculate efficiency metrics
        total_requests_24h = len([
            1 for timestamp, _, _, _ in self.request_costs
            if timestamp >= one_day_ago
        ])
        
        if total_requests_24h > 0:
            self.cost_metrics.cost_per_request = self.cost_metrics.daily_cost / total_requests_24h
        
        self.cost_metrics.last_updated = now
    
    async def _check_budget_alerts(self) -> None:
        """Check for budget threshold violations and generate alerts."""
        current_costs = {
            "hourly": self.cost_metrics.hourly_cost,
            "daily": self.cost_metrics.daily_cost,
            "weekly": self.cost_metrics.weekly_cost,
            "monthly": self.cost_metrics.monthly_cost,
        }
        
        for period, current_cost in current_costs.items():
            budget_limit = self.budget_limits.get(period)
            if not budget_limit:
                continue
            
            usage_percentage = current_cost / budget_limit
            
            for threshold in self.alert_thresholds:
                if usage_percentage >= threshold:
                    # Check if alert already exists
                    existing_alert = next(
                        (alert for alert in self.budget_alerts 
                         if alert.budget_type == period and alert.threshold == threshold),
                        None
                    )
                    
                    if not existing_alert:
                        severity = self._determine_alert_severity(usage_percentage)
                        alert = BudgetAlert(
                            alert_id=f"{period}_{threshold}_{datetime.now().timestamp()}",
                            name=f"{period.title()} Budget Alert",
                            budget_type=period,
                            threshold=threshold,
                            current_usage=current_cost,
                            percentage_used=usage_percentage * 100,
                            severity=severity,
                            message=f"{period.title()} costs have reached {usage_percentage*100:.1f}% of budget",
                            triggered_at=datetime.now(),
                        )
                        
                        self.budget_alerts.append(alert)
                        
                        logger.warning(
                            "Budget alert triggered",
                            period=period,
                            usage_percentage=usage_percentage * 100,
                            current_cost=current_cost,
                            budget_limit=budget_limit,
                            severity=severity.value,
                        )
                        
                        # Execute auto-actions if configured
                        await self._execute_budget_auto_actions(alert)
    
    def _determine_alert_severity(self, usage_percentage: float) -> AlertSeverity:
        """Determine alert severity based on usage percentage."""
        if usage_percentage >= 0.95:
            return AlertSeverity.EMERGENCY
        elif usage_percentage >= 0.85:
            return AlertSeverity.CRITICAL
        elif usage_percentage >= 0.70:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    async def _execute_budget_auto_actions(self, alert: BudgetAlert) -> None:
        """Execute automated actions when budget alerts are triggered."""
        # Example auto-actions based on severity
        if alert.severity == AlertSeverity.CRITICAL:
            # Switch to more cost-effective models
            self.optimization_strategy = OptimizationStrategy.MINIMIZE_COST
            logger.info("Switched to cost minimization strategy due to budget alert")
        
        elif alert.severity == AlertSeverity.EMERGENCY:
            # More aggressive cost controls
            self.optimization_strategy = OptimizationStrategy.MINIMIZE_COST
            # Could also implement request throttling or other protective measures
            logger.critical("Emergency budget measures activated")
    
    def _predict_hourly_load(self, current_hour: int) -> float:
        """Predict load factor for the current hour."""
        # Use historical hourly patterns
        hour_pattern = self.usage_patterns.get(f"hour_{current_hour}", [])
        
        if hour_pattern:
            # Simple average-based prediction
            avg_load = sum(hour_pattern) / len(hour_pattern)
            return avg_load
        
        return 1.0  # Default neutral load factor
    
    def _estimate_input_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate input tokens from messages."""
        total_chars = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total_chars += len(item.get("text", ""))
        
        return total_chars // 4  # Rough approximation
    
    def _estimate_request_cost(
        self, provider_type: ProviderType, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Estimate request cost for a specific provider and model."""
        # Default cost rates (should be loaded from model specs)
        cost_rates = {
            ProviderType.ANTHROPIC: {
                "claude-3-opus": {"input": 0.015, "output": 0.075},
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            },
            ProviderType.OPENAI: {
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            },
            ProviderType.GOOGLE: {
                "gemini-pro": {"input": 0.0005, "output": 0.0015},
            },
        }
        
        provider_rates = cost_rates.get(provider_type, {})
        model_rates = provider_rates.get(model, {"input": 0.001, "output": 0.002})
        
        input_cost = (input_tokens / 1000) * model_rates["input"]
        output_cost = (output_tokens / 1000) * model_rates["output"]
        
        return input_cost + output_cost
    
    def _assess_complexity(self, content: str) -> float:
        """Assess content complexity for cost estimation."""
        # Simple heuristic-based complexity assessment
        complexity_indicators = [
            ("code", 0.3),
            ("analysis", 0.2),
            ("complex", 0.2),
            ("detailed", 0.1),
            ("explain", 0.1),
        ]
        
        content_lower = content.lower()
        complexity_score = 0.0
        
        for indicator, weight in complexity_indicators:
            if indicator in content_lower:
                complexity_score += weight
        
        # Length-based complexity
        if len(content) > 1000:
            complexity_score += 0.2
        
        return min(1.0, complexity_score)
    
    def _categorize_cost_impact(self, input_tokens: int, output_tokens: int, complexity: float) -> str:
        """Categorize the cost impact of a request."""
        total_tokens = input_tokens + output_tokens
        
        if total_tokens > 10000 or complexity > 0.7:
            return "high_cost"
        elif total_tokens > 5000 or complexity > 0.4:
            return "medium_cost"
        else:
            return "low_cost"
    
    def _get_provider_cost_adjustment(self, provider_type: ProviderType) -> float:
        """Get provider-specific cost adjustment factor."""
        # Based on historical performance, reliability, etc.
        adjustments = {
            ProviderType.ANTHROPIC: 1.0,
            ProviderType.OPENAI: 1.05,  # Slight premium for reliability
            ProviderType.GOOGLE: 0.95,  # Slight discount
        }
        
        return adjustments.get(provider_type, 1.0)
    
    def _calculate_savings(self, options: List[Dict[str, Any]], selected_option: Dict[str, Any]) -> float:
        """Calculate potential savings from optimization."""
        if not options:
            return 0.0
        
        # Compare with most expensive option
        max_cost = max(option["estimated_cost"] for option in options)
        return max_cost - selected_option["estimated_cost"]
    
    def _get_current_period_cost(self, period: str) -> float:
        """Get current cost for a specific period."""
        period_costs = {
            "hourly": self.cost_metrics.hourly_cost,
            "daily": self.cost_metrics.daily_cost,
            "weekly": self.cost_metrics.weekly_cost,
            "monthly": self.cost_metrics.monthly_cost,
        }
        
        return period_costs.get(period, 0.0)
    
    # Background monitoring loops
    
    async def _real_time_tracking_loop(self) -> None:
        """Background loop for real-time cost tracking."""
        while True:
            try:
                await asyncio.sleep(self.cost_tracking_interval)
                await self._update_cost_metrics()
                await self._check_budget_alerts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in real-time tracking loop", error=str(e))
    
    async def _predictive_modeling_loop(self) -> None:
        """Background loop for predictive cost modeling."""
        while True:
            try:
                await asyncio.sleep(self.prediction_interval)
                await self._generate_cost_predictions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in predictive modeling loop", error=str(e))
    
    async def _optimization_loop(self) -> None:
        """Background loop for optimization analysis."""
        while True:
            try:
                await asyncio.sleep(self.optimization_interval)
                await self._analyze_optimization_opportunities()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in optimization loop", error=str(e))
    
    async def _generate_cost_predictions(self) -> None:
        """Generate cost predictions for various time periods."""
        # Simple trend-based prediction
        recent_costs = [cost for timestamp, cost in self.cost_history[-100:]]
        
        if len(recent_costs) < 10:
            return  # Not enough data
        
        # Calculate trend
        mid_point = len(recent_costs) // 2
        first_half_avg = sum(recent_costs[:mid_point]) / mid_point
        second_half_avg = sum(recent_costs[mid_point:]) / (len(recent_costs) - mid_point)
        
        trend_factor = second_half_avg / first_half_avg if first_half_avg > 0 else 1.0
        
        # Generate predictions
        current_hourly = self.cost_metrics.hourly_cost
        
        predictions = [
            CostPrediction(
                prediction_period="1h",
                predicted_cost=current_hourly * trend_factor,
                confidence_level=0.7,
                trend="increasing" if trend_factor > 1.05 else "decreasing" if trend_factor < 0.95 else "stable",
            ),
            CostPrediction(
                prediction_period="1d",
                predicted_cost=current_hourly * 24 * trend_factor,
                confidence_level=0.6,
                trend="increasing" if trend_factor > 1.05 else "decreasing" if trend_factor < 0.95 else "stable",
            ),
        ]
        
        self.cost_predictions.extend(predictions)
        # Keep only recent predictions
        cutoff = datetime.now() - timedelta(hours=24)
        self.cost_predictions = [p for p in self.cost_predictions if p.created_at >= cutoff]
    
    async def _analyze_optimization_opportunities(self) -> None:
        """Analyze current spending for optimization opportunities."""
        # Analyze provider cost distribution
        total_daily_cost = self.cost_metrics.daily_cost
        if total_daily_cost > 0:
            for provider, cost in self.cost_metrics.provider_costs.items():
                cost_percentage = cost / total_daily_cost
                
                if cost_percentage > 0.5:  # Provider taking >50% of costs
                    logger.info(
                        "High cost provider identified",
                        provider=provider.value,
                        cost_percentage=cost_percentage * 100,
                        daily_cost=cost,
                    )
        
        # Clean up old arbitrage opportunities
        cutoff = datetime.now() - timedelta(hours=1)
        self.arbitrage_opportunities = [
            opp for opp in self.arbitrage_opportunities
            if opp.detected_at >= cutoff
        ]
    
    def set_budget_limit(self, period: str, limit: float) -> None:
        """Set budget limit for a period."""
        self.budget_limits[period] = limit
        logger.info(f"Set {period} budget limit to ${limit:.2f}")
    
    def get_cost_analysis(self) -> Dict[str, Any]:
        """Get comprehensive cost analysis."""
        return {
            "current_metrics": {
                "hourly_cost": self.cost_metrics.hourly_cost,
                "daily_cost": self.cost_metrics.daily_cost,
                "weekly_cost": self.cost_metrics.weekly_cost,
                "monthly_cost": self.cost_metrics.monthly_cost,
                "cost_per_request": self.cost_metrics.cost_per_request,
            },
            "provider_breakdown": dict(self.cost_metrics.provider_costs),
            "active_alerts": len(self.budget_alerts),
            "optimization_strategy": self.optimization_strategy.value,
            "arbitrage_opportunities": len(self.arbitrage_opportunities),
            "cost_predictions": len(self.cost_predictions),
            "budget_limits": dict(self.budget_limits),
        }
    
    def get_budget_alerts(self) -> List[Dict[str, Any]]:
        """Get current budget alerts."""
        return [
            {
                "alert_id": alert.alert_id,
                "name": alert.name,
                "budget_type": alert.budget_type,
                "percentage_used": alert.percentage_used,
                "severity": alert.severity.value,
                "message": alert.message,
                "triggered_at": alert.triggered_at.isoformat(),
                "acknowledged": alert.acknowledged,
            }
            for alert in self.budget_alerts
        ]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a budget alert."""
        for alert in self.budget_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False