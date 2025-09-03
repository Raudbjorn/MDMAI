"""Intelligent Model Selector - Main orchestrator for comprehensive AI model selection."""

import asyncio
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import structlog
from .task_categorizer import TTRPGTaskType, TaskCategorizer, TaskCharacteristics
from .performance_profiler import PerformanceBenchmark, MetricType
from .model_optimizer import ModelOptimizer, ModelRecommendation, OptimizationStrategy
from .preference_learner import PreferenceLearner, UserFeedback, FeedbackType
from .ab_testing import ABTestingFramework, ExperimentType
from .context_aware_selector import ContextAwareSelector, SessionPhase, CampaignGenre
from .decision_tree import ModelSelectionDecisionTree, ModelScore
from ..ai_providers.models import ProviderType, ModelSpec

logger = structlog.get_logger(__name__)


class SelectionMode(Enum):
    """Model selection modes."""
    
    AUTOMATIC = "automatic"              # Fully automatic selection
    DECISION_TREE = "decision_tree"      # Use decision tree approach
    OPTIMIZATION = "optimization"       # Use optimization algorithm
    USER_PREFERENCE = "user_preference" # Prioritize user preferences
    CONTEXT_AWARE = "context_aware"     # Context-driven selection
    AB_TEST = "ab_test"                 # A/B testing mode
    HYBRID = "hybrid"                   # Combine multiple approaches


@dataclass
class SelectionRequest:
    """Request for intelligent model selection."""
    
    request_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    session_id: Optional[str] = None
    campaign_id: Optional[str] = None
    
    # Request content
    user_input: str = ""
    task_context: Dict[str, Any] = field(default_factory=dict)
    
    # Selection preferences
    selection_mode: SelectionMode = SelectionMode.AUTOMATIC
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    
    # Constraints
    max_cost: Optional[float] = None
    max_latency_ms: Optional[int] = None
    min_quality: Optional[float] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1


@dataclass
class SelectionResult:
    """Result of intelligent model selection."""
    
    request_id: str
    
    # Selected model
    selected_provider: ProviderType
    selected_model: str
    confidence: float
    
    # Alternative options
    alternatives: List[Tuple[ProviderType, str, float]] = field(default_factory=list)
    
    # Selection reasoning
    selection_method: str = ""
    reasoning: List[str] = field(default_factory=list)
    factors_considered: List[str] = field(default_factory=list)
    
    # Performance predictions
    expected_latency_ms: float = 0.0
    expected_quality_score: float = 0.0
    expected_cost: float = 0.0
    
    # Metadata
    selection_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class IntelligentModelSelector:
    """Main orchestrator for comprehensive AI model selection."""
    
    def __init__(self):
        # Core components
        self.task_categorizer = TaskCategorizer()
        self.performance_benchmark = PerformanceBenchmark()
        self.model_optimizer = ModelOptimizer(self.task_categorizer, self.performance_benchmark)
        self.preference_learner = PreferenceLearner()
        self.ab_testing = ABTestingFramework(self.performance_benchmark)
        self.context_selector = ContextAwareSelector(
            self.task_categorizer, self.model_optimizer, 
            self.performance_benchmark, self.preference_learner
        )
        self.decision_tree = ModelSelectionDecisionTree()
        
        # State management
        self.available_models: Dict[str, ModelSpec] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.selection_history: deque = deque(maxlen=1000)
        
        # Configuration
        self.default_selection_mode = SelectionMode.HYBRID
        self.enable_ab_testing = True
        self.enable_learning = True
        
        # Performance tracking
        self.selection_stats = {
            "total_selections": 0,
            "successful_selections": 0,
            "average_selection_time": 0.0,
            "mode_usage": defaultdict(int),
            "provider_usage": defaultdict(int)
        }
    
    async def initialize(self, model_specs: List[ModelSpec]) -> None:
        """Initialize the intelligent model selector with available models."""
        logger.info("Initializing Intelligent Model Selector")
        
        # Register models with all components
        for model_spec in model_specs:
            model_key = f"{model_spec.provider_type.value}:{model_spec.model_id}"
            self.available_models[model_key] = model_spec
            
            # Register with optimizer
            await self.model_optimizer.register_model(model_spec)
            
            # Register with decision tree
            self.decision_tree.model_registry[model_key] = model_spec
        
        logger.info(
            "Intelligent Model Selector initialized",
            total_models=len(model_specs),
            providers=[spec.provider_type.value for spec in model_specs]
        )
    
    async def select_model(self, request: SelectionRequest) -> SelectionResult:
        """Main method for intelligent model selection."""
        start_time = datetime.now()
        
        logger.info(
            "Starting intelligent model selection",
            request_id=request.request_id,
            user_id=request.user_id,
            selection_mode=request.selection_mode.value,
            input_length=len(request.user_input)
        )
        
        try:
            # 1. Check for A/B testing assignment first
            if self.enable_ab_testing and request.selection_mode in [SelectionMode.AUTOMATIC, SelectionMode.AB_TEST]:
                ab_assignment = await self._check_ab_testing(request)
                if ab_assignment:
                    selection_result = await self._create_result_from_ab_assignment(request, ab_assignment)
                    await self._record_selection(request, selection_result, "ab_testing")
                    return selection_result
            
            # 2. Route to appropriate selection method
            if request.selection_mode == SelectionMode.DECISION_TREE:
                result = await self._select_via_decision_tree(request)
            elif request.selection_mode == SelectionMode.OPTIMIZATION:
                result = await self._select_via_optimization(request)
            elif request.selection_mode == SelectionMode.USER_PREFERENCE:
                result = await self._select_via_user_preference(request)
            elif request.selection_mode == SelectionMode.CONTEXT_AWARE:
                result = await self._select_via_context_aware(request)
            elif request.selection_mode == SelectionMode.HYBRID:
                result = await self._select_via_hybrid(request)
            else:  # AUTOMATIC
                result = await self._select_automatic(request)
            
            # 3. Post-process and finalize result
            result.selection_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # 4. Record selection for learning
            await self._record_selection(request, result, result.selection_method)
            
            # 5. Update statistics
            self._update_selection_stats(request, result)
            
            logger.info(
                "Model selection completed",
                request_id=request.request_id,
                selected_model=f"{result.selected_provider.value}:{result.selected_model}",
                confidence=result.confidence,
                selection_time_ms=result.selection_time_ms
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Error in model selection",
                request_id=request.request_id,
                error=str(e)
            )
            
            # Return fallback selection
            return await self._create_fallback_result(request)
    
    async def _check_ab_testing(self, request: SelectionRequest) -> Optional[Tuple[str, str]]:
        """Check if user should be assigned to an A/B test."""
        if not self.enable_ab_testing:
            return None
        
        # Categorize the task to check for applicable experiments
        task_type, _ = self.task_categorizer.categorize_task(request.user_input)
        
        # Check for assignment
        assignment = await self.ab_testing.assign_user_to_variant(
            request.user_id, task_type, request.task_context
        )
        
        return assignment
    
    async def _create_result_from_ab_assignment(
        self,
        request: SelectionRequest,
        assignment: Tuple[str, str]
    ) -> SelectionResult:
        """Create selection result from A/B test assignment."""
        experiment_id, variant_id = assignment
        
        # Get experiment details to determine model
        experiment = self.ab_testing.active_experiments.get(experiment_id)
        if not experiment:
            return await self._create_fallback_result(request)
        
        # Find the variant
        variant = None
        for v in experiment.variants:
            if v.variant_id == variant_id:
                variant = v
                break
        
        if not variant or not variant.provider_type or not variant.model_id:
            return await self._create_fallback_result(request)
        
        return SelectionResult(
            request_id=request.request_id,
            selected_provider=variant.provider_type,
            selected_model=variant.model_id,
            confidence=0.8,  # A/B test assignment has reasonable confidence
            selection_method="ab_testing",
            reasoning=[
                f"Selected via A/B testing experiment: {experiment.name}",
                f"Assigned to variant: {variant.name}",
                f"Experiment type: {experiment.experiment_type.value}"
            ],
            factors_considered=["ab_testing_assignment", "experiment_design"],
            expected_latency_ms=3000,  # Default estimates
            expected_quality_score=0.8,
            expected_cost=0.02
        )
    
    async def _select_via_decision_tree(self, request: SelectionRequest) -> SelectionResult:
        """Select model using decision tree approach."""
        # Build context for decision tree
        context = await self._build_comprehensive_context(request)
        
        # Get available models
        available_models = [(spec.provider_type, spec.model_id) for spec in self.available_models.values()]
        
        # Evaluate decision tree
        leaf_node = await self.decision_tree.evaluate_decision_tree(context, available_models)
        
        if leaf_node and leaf_node.recommended_models:
            # Use top recommendation from decision tree
            provider, model, score = leaf_node.recommended_models[0]
            
            return SelectionResult(
                request_id=request.request_id,
                selected_provider=provider,
                selected_model=model,
                confidence=score * leaf_node.confidence,
                alternatives=[(p, m, s) for p, m, s in leaf_node.recommended_models[1:3]],
                selection_method="decision_tree",
                reasoning=[
                    f"Selected via decision tree node: {leaf_node.name}",
                    f"Node description: {leaf_node.description}",
                    f"Decision tree confidence: {leaf_node.confidence:.2f}"
                ],
                factors_considered=["task_type", "context_conditions", "decision_rules"],
                expected_latency_ms=2500,
                expected_quality_score=0.8,
                expected_cost=0.015
            )
        
        # Fallback to automatic selection
        return await self._select_automatic(request)
    
    async def _select_via_optimization(self, request: SelectionRequest) -> SelectionResult:
        """Select model using optimization algorithm."""
        # Categorize task
        task_type, confidence = self.task_categorizer.categorize_task(request.user_input)
        task_characteristics = self.task_categorizer.get_task_characteristics(task_type)
        
        # Build context
        context = await self._build_comprehensive_context(request)
        
        # Get user preferences
        user_preferences = await self.preference_learner.get_user_preferences(
            request.user_id, task_type, request.campaign_id
        )
        
        # Get recommendation from optimizer
        recommendation = await self.model_optimizer.optimize_model_selection(
            task_type, task_characteristics, context, user_preferences
        )
        
        return SelectionResult(
            request_id=request.request_id,
            selected_provider=recommendation.provider_type,
            selected_model=recommendation.model_id,
            confidence=recommendation.confidence,
            alternatives=recommendation.fallback_options,
            selection_method="optimization",
            reasoning=recommendation.reasoning,
            factors_considered=["task_analysis", "optimization_strategy", "user_preferences"],
            expected_latency_ms=recommendation.expected_performance.get("latency", 3000),
            expected_quality_score=recommendation.expected_performance.get("quality", 0.8),
            expected_cost=recommendation.cost_estimate
        )
    
    async def _select_via_user_preference(self, request: SelectionRequest) -> SelectionResult:
        """Select model prioritizing user preferences."""
        # Get user preferences
        user_preferences = await self.preference_learner.get_user_preferences(request.user_id)
        
        if user_preferences.get("confidence_score", 0) < 0.3:
            # Not enough preference data, fall back to automatic
            return await self._select_automatic(request)
        
        # Find preferred provider/model
        provider_prefs = user_preferences.get("provider_preferences", {})
        model_prefs = user_preferences.get("model_preferences", {})
        
        # Score available models by user preference
        model_scores = []
        for model_key, spec in self.available_models.items():
            provider_score = provider_prefs.get(spec.provider_type.value, 0.5)
            model_score = model_prefs.get(model_key, 0.5)
            combined_score = (provider_score + model_score) / 2
            model_scores.append((spec.provider_type, spec.model_id, combined_score))
        
        # Sort by preference score
        model_scores.sort(key=lambda x: x[2], reverse=True)
        
        if model_scores:
            provider, model, score = model_scores[0]
            
            return SelectionResult(
                request_id=request.request_id,
                selected_provider=provider,
                selected_model=model,
                confidence=score,
                alternatives=[(p, m, s) for p, m, s in model_scores[1:3]],
                selection_method="user_preference",
                reasoning=[
                    "Selected based on your historical preferences",
                    f"Provider preference: {provider_prefs.get(provider.value, 0.5):.2f}",
                    f"Model preference: {model_prefs.get(model_key, 0.5):.2f}",
                    f"Preference confidence: {user_preferences['confidence_score']:.2f}"
                ],
                factors_considered=["user_preferences", "historical_feedback", "preference_confidence"],
                expected_latency_ms=3000,
                expected_quality_score=0.8,
                expected_cost=0.02
            )
        
        return await self._create_fallback_result(request)
    
    async def _select_via_context_aware(self, request: SelectionRequest) -> SelectionResult:
        """Select model using context-aware logic."""
        recommendation = await self.context_selector.select_optimal_model(
            request.user_id, request.user_input, request.session_id, request.campaign_id
        )
        
        return SelectionResult(
            request_id=request.request_id,
            selected_provider=recommendation.provider_type,
            selected_model=recommendation.model_id,
            confidence=recommendation.confidence,
            alternatives=recommendation.fallback_options,
            selection_method="context_aware",
            reasoning=recommendation.reasoning,
            factors_considered=["context_analysis", "session_state", "campaign_context"],
            expected_latency_ms=recommendation.expected_performance.get("latency", 3000),
            expected_quality_score=recommendation.expected_performance.get("quality", 0.8),
            expected_cost=recommendation.cost_estimate
        )
    
    async def _select_via_hybrid(self, request: SelectionRequest) -> SelectionResult:
        """Select model using hybrid approach combining multiple methods."""
        # Get recommendations from multiple approaches
        context_result = await self._select_via_context_aware(request)
        optimization_result = await self._select_via_optimization(request)
        
        # Try decision tree if context/optimization disagree
        decision_tree_result = None
        if context_result.selected_model != optimization_result.selected_model:
            decision_tree_result = await self._select_via_decision_tree(request)
        
        # Score the recommendations
        candidates = [
            (context_result.selected_provider, context_result.selected_model, 
             context_result.confidence * 0.4, "context_aware"),
            (optimization_result.selected_provider, optimization_result.selected_model,
             optimization_result.confidence * 0.4, "optimization")
        ]
        
        if decision_tree_result:
            candidates.append((
                decision_tree_result.selected_provider, decision_tree_result.selected_model,
                decision_tree_result.confidence * 0.2, "decision_tree"
            ))
        
        # Select the highest scoring recommendation
        best_candidate = max(candidates, key=lambda x: x[2])
        provider, model, score, method = best_candidate
        
        # Build comprehensive reasoning
        reasoning = [
            f"Hybrid selection using {method} approach",
            f"Context-aware suggested: {context_result.selected_provider.value}:{context_result.selected_model}",
            f"Optimization suggested: {optimization_result.selected_provider.value}:{optimization_result.selected_model}"
        ]
        
        if decision_tree_result:
            reasoning.append(f"Decision tree suggested: {decision_tree_result.selected_provider.value}:{decision_tree_result.selected_model}")
        
        return SelectionResult(
            request_id=request.request_id,
            selected_provider=provider,
            selected_model=model,
            confidence=min(0.95, score / 0.4),  # Normalize confidence
            selection_method="hybrid",
            reasoning=reasoning,
            factors_considered=["multiple_approaches", "consensus_analysis", "weighted_scoring"],
            expected_latency_ms=3000,
            expected_quality_score=0.8,
            expected_cost=0.02
        )
    
    async def _select_automatic(self, request: SelectionRequest) -> SelectionResult:
        """Automatic selection using the best available method."""
        # For automatic mode, choose the best approach based on available data
        
        # Check user preference data quality
        user_preferences = await self.preference_learner.get_user_preferences(request.user_id)
        preference_confidence = user_preferences.get("confidence_score", 0.0)
        
        # If user has strong preferences, use context-aware (which includes preferences)
        if preference_confidence > 0.7:
            result = await self._select_via_context_aware(request)
            result.selection_method = "automatic_context_aware"
            return result
        
        # If moderate preferences, use hybrid approach
        elif preference_confidence > 0.3:
            result = await self._select_via_hybrid(request)
            result.selection_method = "automatic_hybrid"
            return result
        
        # For new users, use optimization approach
        else:
            result = await self._select_via_optimization(request)
            result.selection_method = "automatic_optimization"
            return result
    
    async def _build_comprehensive_context(self, request: SelectionRequest) -> Dict[str, Any]:
        """Build comprehensive context for model selection."""
        # Categorize task
        task_type, confidence = self.task_categorizer.categorize_task(request.user_input)
        task_characteristics = self.task_categorizer.get_task_characteristics(task_type)
        
        context = {
            "task_type": task_type.value,
            "task_confidence": confidence,
            "latency_requirement_ms": request.max_latency_ms or task_characteristics.latency_requirement.value,
            "quality_requirement": request.min_quality or 0.8,
            "cost_budget": request.max_cost or 1.0,
            "timestamp": request.timestamp,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "campaign_id": request.campaign_id
        }
        
        # Add task context
        context.update(request.task_context)
        
        # Add session context if available
        if request.session_id and request.session_id in self.context_selector.session_contexts:
            session_ctx = self.context_selector.session_contexts[request.session_id]
            context["session_context"] = {
                "phase": session_ctx.phase.value,
                "in_combat": session_ctx.in_combat,
                "session_duration": session_ctx.total_session_duration.total_seconds() / 60
            }
        
        # Add campaign context if available
        if request.campaign_id and request.campaign_id in self.context_selector.campaign_contexts:
            campaign_ctx = self.context_selector.campaign_contexts[request.campaign_id]
            context["campaign_context"] = {
                "genre": campaign_ctx.genre.value,
                "tone": campaign_ctx.tone,
                "complexity_level": campaign_ctx.complexity_level
            }
        
        # Add system context
        context["system_context"] = {
            "provider_health": dict(self.context_selector.system_context.provider_health),
            "provider_load": dict(self.context_selector.system_context.provider_load),
            "budget_remaining": self.context_selector.system_context.total_budget_remaining,
            "peak_usage": self.context_selector.system_context.peak_usage_detected
        }
        
        return context
    
    async def _create_fallback_result(self, request: SelectionRequest) -> SelectionResult:
        """Create a fallback result when other methods fail."""
        # Use a reliable default model
        default_provider = ProviderType.ANTHROPIC
        default_model = "claude-3-5-sonnet"
        
        # Check if default is available
        if f"{default_provider.value}:{default_model}" not in self.available_models:
            # Find any available model
            if self.available_models:
                first_model = next(iter(self.available_models.values()))
                default_provider = first_model.provider_type
                default_model = first_model.model_id
            else:
                # No models available - this shouldn't happen
                logger.error("No models available for fallback", request_id=request.request_id)
                default_provider = ProviderType.ANTHROPIC
                default_model = "claude-3-5-sonnet"
        
        return SelectionResult(
            request_id=request.request_id,
            selected_provider=default_provider,
            selected_model=default_model,
            confidence=0.5,
            selection_method="fallback",
            reasoning=[
                "Fallback selection due to error in primary selection methods",
                f"Using reliable default: {default_provider.value}:{default_model}"
            ],
            factors_considered=["fallback_logic", "model_availability"],
            expected_latency_ms=3000,
            expected_quality_score=0.8,
            expected_cost=0.02
        )
    
    async def _record_selection(
        self,
        request: SelectionRequest,
        result: SelectionResult,
        method: str
    ) -> None:
        """Record selection for learning and analytics."""
        selection_record = {
            "request_id": request.request_id,
            "user_id": request.user_id,
            "timestamp": datetime.now(),
            "task_type": self.task_categorizer.categorize_task(request.user_input)[0].value,
            "selected_provider": result.selected_provider.value,
            "selected_model": result.selected_model,
            "selection_method": method,
            "confidence": result.confidence,
            "selection_time_ms": result.selection_time_ms
        }
        
        self.selection_history.append(selection_record)
        
        # Start performance tracking
        if self.enable_learning:
            await self.performance_benchmark.start_request_tracking(
                request.request_id,
                result.selected_provider,
                result.selected_model,
                selection_record["task_type"],
                {"selection_method": method}
            )
    
    def _update_selection_stats(self, request: SelectionRequest, result: SelectionResult) -> None:
        """Update selection statistics."""
        self.selection_stats["total_selections"] += 1
        self.selection_stats["mode_usage"][request.selection_mode.value] += 1
        self.selection_stats["provider_usage"][result.selected_provider.value] += 1
        
        # Update average selection time
        current_avg = self.selection_stats["average_selection_time"]
        total = self.selection_stats["total_selections"]
        new_avg = (current_avg * (total - 1) + result.selection_time_ms) / total
        self.selection_stats["average_selection_time"] = new_avg
    
    async def record_feedback(
        self,
        request_id: str,
        feedback_type: FeedbackType,
        rating: Optional[float] = None,
        success: bool = True,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Record feedback for a previous selection."""
        # Find the selection record
        selection_record = None
        for record in reversed(self.selection_history):
            if record["request_id"] == request_id:
                selection_record = record
                break
        
        if not selection_record:
            logger.warning("Selection record not found for feedback", request_id=request_id)
            return
        
        # Create feedback object
        feedback = UserFeedback(
            user_id=selection_record["user_id"],
            task_type=TTRPGTaskType(selection_record["task_type"]),
            provider_type=ProviderType(selection_record["selected_provider"]),
            model_id=selection_record["selected_model"],
            feedback_type=feedback_type,
            rating=rating
        )
        
        # Record with preference learner
        if self.enable_learning:
            await self.preference_learner.record_feedback(feedback)
        
        # Complete performance tracking
        if performance_metrics:
            for metric_name, value in performance_metrics.items():
                if metric_name == "latency":
                    await self.performance_benchmark.record_metric(
                        request_id, MetricType.LATENCY, value
                    )
                elif metric_name == "quality":
                    await self.performance_benchmark.record_metric(
                        request_id, MetricType.QUALITY, value
                    )
                elif metric_name == "cost":
                    await self.performance_benchmark.record_metric(
                        request_id, MetricType.COST, value
                    )
        
        await self.performance_benchmark.complete_request_tracking(request_id, success)
        
        # Update statistics
        if success:
            self.selection_stats["successful_selections"] += 1
        
        logger.info(
            "Recorded feedback for model selection",
            request_id=request_id,
            feedback_type=feedback_type.value,
            success=success
        )
    
    async def get_selection_insights(
        self,
        user_id: Optional[str] = None,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """Get insights about model selection patterns and performance."""
        insights = {
            "selection_statistics": dict(self.selection_stats),
            "performance_insights": {},
            "user_preferences": {},
            "recommendations": []
        }
        
        # Get performance insights
        insights["performance_insights"] = self.performance_benchmark.get_performance_insights(
            time_range_hours=time_range_hours
        )
        
        # Get user preferences if specified
        if user_id:
            insights["user_preferences"] = await self.preference_learner.get_user_preferences(user_id)
        
        # Get context insights
        context_insights = self.context_selector.get_context_insights(time_range_hours=time_range_hours)
        insights.update(context_insights)
        
        # Generate recommendations
        if self.selection_stats["total_selections"] > 100:
            success_rate = self.selection_stats["successful_selections"] / self.selection_stats["total_selections"]
            
            if success_rate < 0.8:
                insights["recommendations"].append({
                    "type": "improve_selection_accuracy",
                    "message": f"Selection success rate is {success_rate:.1%}. Consider adjusting selection criteria.",
                    "priority": "high"
                })
            
            # Analyze mode effectiveness
            most_used_mode = max(self.selection_stats["mode_usage"].items(), key=lambda x: x[1])
            insights["recommendations"].append({
                "type": "mode_usage_analysis",
                "message": f"Most used selection mode: {most_used_mode[0]} ({most_used_mode[1]} times)",
                "priority": "info"
            })
        
        return insights
    
    async def cleanup_old_data(self, retention_hours: int = 48) -> Dict[str, int]:
        """Clean up old data across all components."""
        cleanup_results = {}
        
        # Clean up performance data
        cleanup_results["performance_metrics"] = await self.performance_benchmark.cleanup_old_data()
        
        # Clean up preference data
        cleanup_results["preference_data"] = await self.preference_learner.cleanup_old_data(
            retention_days=retention_hours // 24
        )
        
        # Clean up context data
        cleanup_results["context_data"] = await self.context_selector.cleanup_old_context_data(
            retention_hours=retention_hours
        )
        
        # Clean up A/B testing data
        cleanup_results["ab_testing_data"] = await self.ab_testing.cleanup_completed_experiments(
            retention_days=retention_hours // 24
        )
        
        # Clean up selection history
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        original_count = len(self.selection_history)
        
        # Filter selection history
        filtered_history = deque(
            (record for record in self.selection_history 
             if record["timestamp"] > cutoff_time),
            maxlen=1000
        )
        
        self.selection_history = filtered_history
        cleanup_results["selection_history"] = original_count - len(filtered_history)
        
        total_cleaned = sum(cleanup_results.values())
        
        logger.info(
            "Completed data cleanup across all components",
            total_items_cleaned=total_cleaned,
            retention_hours=retention_hours,
            breakdown=cleanup_results
        )
        
        return cleanup_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "intelligent_selector": {
                "available_models": len(self.available_models),
                "active_sessions": len(self.active_sessions),
                "selection_history_size": len(self.selection_history),
                "default_mode": self.default_selection_mode.value,
                "ab_testing_enabled": self.enable_ab_testing,
                "learning_enabled": self.enable_learning
            },
            "components": {
                "task_categorizer": "active",
                "performance_benchmark": {
                    "tracked_models": len(self.performance_benchmark.model_profiles),
                    "active_requests": len(self.performance_benchmark.current_requests)
                },
                "model_optimizer": {
                    "optimization_strategy": self.model_optimizer.optimization_strategy.value,
                    "registered_models": len(self.model_optimizer.available_models)
                },
                "preference_learner": {
                    "user_profiles": len(self.preference_learner.user_profiles),
                    "total_feedback": sum(len(hist) for hist in self.preference_learner.feedback_history.values())
                },
                "ab_testing": {
                    "active_experiments": len(self.ab_testing.active_experiments),
                    "total_experiments": len(self.ab_testing.experiments)
                },
                "context_selector": {
                    "active_sessions": len(self.context_selector.session_contexts),
                    "campaign_contexts": len(self.context_selector.campaign_contexts)
                },
                "decision_tree": {
                    "tree_statistics": self.decision_tree.get_decision_statistics()
                }
            },
            "performance": {
                "selection_stats": dict(self.selection_stats),
                "average_selection_time_ms": self.selection_stats["average_selection_time"]
            }
        }