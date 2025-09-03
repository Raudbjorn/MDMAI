"""Decision tree and scoring algorithm for comprehensive AI model selection."""

import asyncio
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

import structlog
from .task_categorizer import TTRPGTaskType, TaskCharacteristics, TaskComplexity, TaskLatencyRequirement
from .model_optimizer import ModelRecommendation, OptimizationStrategy
from .performance_profiler import ModelPerformanceProfile
from .context_aware_selector import SessionPhase, CampaignGenre
from ..ai_providers.models import ProviderType, ModelSpec

logger = structlog.get_logger(__name__)


class DecisionCriterion(Enum):
    """Criteria used in decision tree nodes."""
    
    TASK_TYPE = "task_type"
    LATENCY_REQUIREMENT = "latency_requirement"
    QUALITY_REQUIREMENT = "quality_requirement"
    COST_BUDGET = "cost_budget"
    PROVIDER_HEALTH = "provider_health"
    USER_PREFERENCE = "user_preference"
    SYSTEM_LOAD = "system_load"
    SESSION_PHASE = "session_phase"
    CAMPAIGN_GENRE = "campaign_genre"
    HISTORICAL_PERFORMANCE = "historical_performance"


class ComparisonOperator(Enum):
    """Comparison operators for decision criteria."""
    
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"


@dataclass
class DecisionCondition:
    """A condition in a decision tree node."""
    
    criterion: DecisionCriterion
    operator: ComparisonOperator
    value: Union[str, float, int, List[Any]]
    weight: float = 1.0  # Weight for scoring


@dataclass
class DecisionTreeNode:
    """A node in the decision tree."""
    
    node_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    
    # Conditions for this node
    conditions: List[DecisionCondition] = field(default_factory=list)
    
    # Child nodes
    children: List["DecisionTreeNode"] = field(default_factory=list)
    
    # If this is a leaf node, model recommendation
    recommended_models: List[Tuple[ProviderType, str, float]] = field(default_factory=list)  # (provider, model, score)
    
    # Node metadata
    confidence: float = 1.0
    priority: int = 0
    success_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Statistics
    evaluation_count: int = 0
    success_count: int = 0


@dataclass
class ModelScore:
    """Comprehensive scoring for a model candidate."""
    
    provider_type: ProviderType
    model_id: str
    
    # Individual component scores (0.0-1.0)
    task_suitability_score: float = 0.0
    performance_score: float = 0.0
    cost_score: float = 0.0
    reliability_score: float = 0.0
    user_preference_score: float = 0.0
    context_fit_score: float = 0.0
    
    # Weighted final score
    final_score: float = 0.0
    confidence: float = 0.0
    
    # Detailed breakdown
    scoring_details: Dict[str, float] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)


class ModelSelectionDecisionTree:
    """Comprehensive decision tree for AI model selection with scoring algorithm."""
    
    def __init__(self):
        self.root_node = self._build_default_decision_tree()
        self.model_registry: Dict[str, ModelSpec] = {}
        self.performance_profiles: Dict[str, ModelPerformanceProfile] = {}
        
        # Scoring weights (configurable)
        self.scoring_weights = {
            "task_suitability": 0.25,
            "performance": 0.20,
            "cost": 0.15,
            "reliability": 0.15,
            "user_preference": 0.15,
            "context_fit": 0.10
        }
        
        # Model capability matrix
        self.model_capabilities = self._initialize_model_capabilities()
        
        # Performance thresholds
        self.performance_thresholds = {
            "latency_excellent": 500,    # ms
            "latency_good": 2000,
            "latency_acceptable": 5000,
            "quality_excellent": 0.9,
            "quality_good": 0.8,
            "quality_acceptable": 0.7
        }
        
        # Cost efficiency targets
        self.cost_targets = {
            "free_tier": 0.0,
            "budget": 0.01,      # $0.01 per request
            "standard": 0.05,    # $0.05 per request
            "premium": 0.20      # $0.20 per request
        }
    
    def _initialize_model_capabilities(self) -> Dict[str, Dict[str, float]]:
        """Initialize comprehensive model capability matrix."""
        return {
            # Anthropic Claude models
            "anthropic:claude-3-5-sonnet": {
                "rule_lookup": 0.95,
                "rule_clarification": 0.95,
                "character_generation": 0.90,
                "npc_generation": 0.90,
                "story_generation": 0.95,
                "world_building": 0.90,
                "description_generation": 0.95,
                "combat_resolution": 0.85,
                "session_summarization": 0.90,
                "improvisation": 0.90,
                "creativity": 0.95,
                "accuracy": 0.90,
                "consistency": 0.85,
                "tool_calling": 0.90,
                "structured_output": 0.85,
                "multi_step_reasoning": 0.90,
                "speed": 0.70,
                "cost_efficiency": 0.60
            },
            
            "anthropic:claude-3-5-haiku": {
                "rule_lookup": 0.90,
                "rule_clarification": 0.85,
                "character_generation": 0.80,
                "npc_generation": 0.80,
                "story_generation": 0.85,
                "world_building": 0.80,
                "description_generation": 0.85,
                "combat_resolution": 0.90,
                "session_summarization": 0.85,
                "improvisation": 0.85,
                "creativity": 0.80,
                "accuracy": 0.85,
                "consistency": 0.80,
                "tool_calling": 0.85,
                "structured_output": 0.80,
                "multi_step_reasoning": 0.80,
                "speed": 0.95,
                "cost_efficiency": 0.95
            },
            
            # OpenAI GPT models
            "openai:gpt-4o": {
                "rule_lookup": 0.90,
                "rule_clarification": 0.90,
                "character_generation": 0.85,
                "npc_generation": 0.85,
                "story_generation": 0.90,
                "world_building": 0.85,
                "description_generation": 0.90,
                "combat_resolution": 0.90,
                "session_summarization": 0.85,
                "improvisation": 0.85,
                "creativity": 0.85,
                "accuracy": 0.90,
                "consistency": 0.90,
                "tool_calling": 0.95,
                "structured_output": 0.95,
                "multi_step_reasoning": 0.90,
                "speed": 0.75,
                "cost_efficiency": 0.70
            },
            
            "openai:gpt-4o-mini": {
                "rule_lookup": 0.85,
                "rule_clarification": 0.80,
                "character_generation": 0.75,
                "npc_generation": 0.75,
                "story_generation": 0.80,
                "world_building": 0.75,
                "description_generation": 0.80,
                "combat_resolution": 0.85,
                "session_summarization": 0.80,
                "improvisation": 0.80,
                "creativity": 0.75,
                "accuracy": 0.80,
                "consistency": 0.85,
                "tool_calling": 0.90,
                "structured_output": 0.90,
                "multi_step_reasoning": 0.80,
                "speed": 0.90,
                "cost_efficiency": 0.90
            },
            
            # Google Gemini models
            "google:gemini-1.5-pro": {
                "rule_lookup": 0.85,
                "rule_clarification": 0.85,
                "character_generation": 0.80,
                "npc_generation": 0.80,
                "story_generation": 0.85,
                "world_building": 0.80,
                "description_generation": 0.85,
                "combat_resolution": 0.85,
                "session_summarization": 0.80,
                "improvisation": 0.80,
                "creativity": 0.80,
                "accuracy": 0.85,
                "consistency": 0.85,
                "tool_calling": 0.85,
                "structured_output": 0.85,
                "multi_step_reasoning": 0.85,
                "speed": 0.80,
                "cost_efficiency": 0.75
            },
            
            "google:gemini-1.5-flash": {
                "rule_lookup": 0.80,
                "rule_clarification": 0.75,
                "character_generation": 0.70,
                "npc_generation": 0.70,
                "story_generation": 0.75,
                "world_building": 0.70,
                "description_generation": 0.75,
                "combat_resolution": 0.80,
                "session_summarization": 0.75,
                "improvisation": 0.75,
                "creativity": 0.70,
                "accuracy": 0.75,
                "consistency": 0.80,
                "tool_calling": 0.80,
                "structured_output": 0.80,
                "multi_step_reasoning": 0.75,
                "speed": 0.95,
                "cost_efficiency": 0.85
            }
        }
    
    def _build_default_decision_tree(self) -> DecisionTreeNode:
        """Build the default decision tree for model selection."""
        root = DecisionTreeNode(
            name="Root",
            description="Root decision node for model selection"
        )
        
        # Level 1: Task Type Branching
        combat_node = DecisionTreeNode(
            name="Combat Tasks",
            description="Combat resolution and immediate response tasks",
            conditions=[
                DecisionCondition(
                    DecisionCriterion.TASK_TYPE,
                    ComparisonOperator.IN,
                    [TTRPGTaskType.COMBAT_RESOLUTION.value, TTRPGTaskType.IMPROVISATION.value]
                )
            ]
        )
        
        creative_node = DecisionTreeNode(
            name="Creative Tasks", 
            description="Story generation, world building, and creative content",
            conditions=[
                DecisionCondition(
                    DecisionCriterion.TASK_TYPE,
                    ComparisonOperator.IN,
                    [TTRPGTaskType.STORY_GENERATION.value, TTRPGTaskType.WORLD_BUILDING.value, 
                     TTRPGTaskType.DESCRIPTION_GENERATION.value]
                )
            ]
        )
        
        rules_node = DecisionTreeNode(
            name="Rules Tasks",
            description="Rule lookup, clarification, and system mechanics",
            conditions=[
                DecisionCondition(
                    DecisionCriterion.TASK_TYPE,
                    ComparisonOperator.IN,
                    [TTRPGTaskType.RULE_LOOKUP.value, TTRPGTaskType.RULE_CLARIFICATION.value]
                )
            ]
        )
        
        generation_node = DecisionTreeNode(
            name="Generation Tasks",
            description="Character and NPC generation",
            conditions=[
                DecisionCondition(
                    DecisionCriterion.TASK_TYPE,
                    ComparisonOperator.IN,
                    [TTRPGTaskType.CHARACTER_GENERATION.value, TTRPGTaskType.NPC_GENERATION.value]
                )
            ]
        )
        
        root.children = [combat_node, creative_node, rules_node, generation_node]
        
        # Level 2: Combat Task Sub-decisions
        combat_immediate = DecisionTreeNode(
            name="Immediate Combat Response",
            description="Ultra-fast combat resolution",
            conditions=[
                DecisionCondition(
                    DecisionCriterion.LATENCY_REQUIREMENT,
                    ComparisonOperator.LESS_THAN,
                    1000  # <1 second
                )
            ],
            recommended_models=[
                (ProviderType.ANTHROPIC, "claude-3-5-haiku", 0.95),
                (ProviderType.OPENAI, "gpt-4o-mini", 0.90),
                (ProviderType.GOOGLE, "gemini-1.5-flash", 0.85)
            ]
        )
        
        combat_standard = DecisionTreeNode(
            name="Standard Combat Resolution",
            description="Balance of speed and accuracy for combat",
            conditions=[
                DecisionCondition(
                    DecisionCriterion.LATENCY_REQUIREMENT,
                    ComparisonOperator.GREATER_EQUAL,
                    1000
                )
            ],
            recommended_models=[
                (ProviderType.OPENAI, "gpt-4o", 0.90),
                (ProviderType.ANTHROPIC, "claude-3-5-sonnet", 0.85),
                (ProviderType.ANTHROPIC, "claude-3-5-haiku", 0.80)
            ]
        )
        
        combat_node.children = [combat_immediate, combat_standard]
        
        # Level 2: Creative Task Sub-decisions
        creative_premium = DecisionTreeNode(
            name="Premium Creative Content",
            description="Highest quality creative generation",
            conditions=[
                DecisionCondition(
                    DecisionCriterion.QUALITY_REQUIREMENT,
                    ComparisonOperator.GREATER_THAN,
                    0.9
                )
            ],
            recommended_models=[
                (ProviderType.ANTHROPIC, "claude-3-5-sonnet", 0.95),
                (ProviderType.OPENAI, "gpt-4o", 0.90),
                (ProviderType.GOOGLE, "gemini-1.5-pro", 0.80)
            ]
        )
        
        creative_balanced = DecisionTreeNode(
            name="Balanced Creative Content",
            description="Good quality creative content with reasonable cost",
            conditions=[
                DecisionCondition(
                    DecisionCriterion.QUALITY_REQUIREMENT,
                    ComparisonOperator.LESS_EQUAL,
                    0.9
                )
            ],
            recommended_models=[
                (ProviderType.ANTHROPIC, "claude-3-5-haiku", 0.85),
                (ProviderType.OPENAI, "gpt-4o-mini", 0.80),
                (ProviderType.ANTHROPIC, "claude-3-5-sonnet", 0.75)
            ]
        )
        
        creative_node.children = [creative_premium, creative_balanced]
        
        # Level 2: Rules Task Sub-decisions
        rules_accurate = DecisionTreeNode(
            name="High Accuracy Rules",
            description="Maximum accuracy for rule interpretations",
            conditions=[
                DecisionCondition(
                    DecisionCriterion.QUALITY_REQUIREMENT,
                    ComparisonOperator.GREATER_THAN,
                    0.9
                )
            ],
            recommended_models=[
                (ProviderType.ANTHROPIC, "claude-3-5-sonnet", 0.95),
                (ProviderType.OPENAI, "gpt-4o", 0.90),
                (ProviderType.ANTHROPIC, "claude-3-5-haiku", 0.85)
            ]
        )
        
        rules_fast = DecisionTreeNode(
            name="Fast Rules Lookup",
            description="Quick rule lookups with good accuracy",
            conditions=[
                DecisionCondition(
                    DecisionCriterion.LATENCY_REQUIREMENT,
                    ComparisonOperator.LESS_THAN,
                    2000
                )
            ],
            recommended_models=[
                (ProviderType.ANTHROPIC, "claude-3-5-haiku", 0.90),
                (ProviderType.OPENAI, "gpt-4o-mini", 0.85),
                (ProviderType.GOOGLE, "gemini-1.5-flash", 0.80)
            ]
        )
        
        rules_node.children = [rules_accurate, rules_fast]
        
        # Level 2: Generation Task Sub-decisions  
        generation_detailed = DecisionTreeNode(
            name="Detailed Character Generation",
            description="Comprehensive character and NPC creation",
            conditions=[
                DecisionCondition(
                    DecisionCriterion.QUALITY_REQUIREMENT,
                    ComparisonOperator.GREATER_THAN,
                    0.8
                )
            ],
            recommended_models=[
                (ProviderType.ANTHROPIC, "claude-3-5-sonnet", 0.90),
                (ProviderType.OPENAI, "gpt-4o", 0.85),
                (ProviderType.ANTHROPIC, "claude-3-5-haiku", 0.80)
            ]
        )
        
        generation_quick = DecisionTreeNode(
            name="Quick Generation",
            description="Fast character/NPC generation for immediate use",
            conditions=[
                DecisionCondition(
                    DecisionCriterion.LATENCY_REQUIREMENT,
                    ComparisonOperator.LESS_THAN,
                    3000
                )
            ],
            recommended_models=[
                (ProviderType.ANTHROPIC, "claude-3-5-haiku", 0.85),
                (ProviderType.OPENAI, "gpt-4o-mini", 0.80),
                (ProviderType.GOOGLE, "gemini-1.5-flash", 0.75)
            ]
        )
        
        generation_node.children = [generation_detailed, generation_quick]
        
        return root
    
    async def evaluate_decision_tree(
        self,
        context: Dict[str, Any],
        available_models: List[Tuple[ProviderType, str]]
    ) -> Optional[DecisionTreeNode]:
        """Evaluate the decision tree and return the best matching leaf node."""
        
        def evaluate_node(node: DecisionTreeNode) -> bool:
            """Evaluate if a node's conditions are met."""
            if not node.conditions:
                return True  # No conditions means always match
            
            for condition in node.conditions:
                if not self._evaluate_condition(condition, context):
                    return False
            return True
        
        def find_leaf(node: DecisionTreeNode) -> Optional[DecisionTreeNode]:
            """Find the best matching leaf node."""
            if not node.children:
                # This is a leaf node
                return node if evaluate_node(node) else None
            
            # Check children
            for child in node.children:
                if evaluate_node(child):
                    leaf = find_leaf(child)
                    if leaf:
                        return leaf
            
            # No matching child found
            return None
        
        # Start evaluation from root
        result = find_leaf(self.root_node)
        
        if result:
            # Update node statistics
            result.evaluation_count += 1
            result.last_updated = datetime.now()
        
        logger.debug(
            "Decision tree evaluation completed",
            result_node=result.name if result else None,
            context_keys=list(context.keys())
        )
        
        return result
    
    def _evaluate_condition(self, condition: DecisionCondition, context: Dict[str, Any]) -> bool:
        """Evaluate a single condition against the context."""
        try:
            # Get the context value based on criterion
            context_value = self._get_context_value(condition.criterion, context)
            
            if context_value is None:
                return False
            
            # Evaluate based on operator
            if condition.operator == ComparisonOperator.EQUALS:
                return context_value == condition.value
            elif condition.operator == ComparisonOperator.NOT_EQUALS:
                return context_value != condition.value
            elif condition.operator == ComparisonOperator.GREATER_THAN:
                return float(context_value) > float(condition.value)
            elif condition.operator == ComparisonOperator.LESS_THAN:
                return float(context_value) < float(condition.value)
            elif condition.operator == ComparisonOperator.GREATER_EQUAL:
                return float(context_value) >= float(condition.value)
            elif condition.operator == ComparisonOperator.LESS_EQUAL:
                return float(context_value) <= float(condition.value)
            elif condition.operator == ComparisonOperator.IN:
                return context_value in condition.value
            elif condition.operator == ComparisonOperator.NOT_IN:
                return context_value not in condition.value
            elif condition.operator == ComparisonOperator.CONTAINS:
                return condition.value in str(context_value)
            
            return False
            
        except (ValueError, TypeError, KeyError) as e:
            logger.warning("Error evaluating decision condition", condition=condition, error=str(e))
            return False
    
    def _get_context_value(self, criterion: DecisionCriterion, context: Dict[str, Any]) -> Any:
        """Extract the relevant context value for a criterion."""
        if criterion == DecisionCriterion.TASK_TYPE:
            return context.get("task_type")
        elif criterion == DecisionCriterion.LATENCY_REQUIREMENT:
            return context.get("latency_requirement_ms", 5000)
        elif criterion == DecisionCriterion.QUALITY_REQUIREMENT:
            return context.get("quality_requirement", 0.8)
        elif criterion == DecisionCriterion.COST_BUDGET:
            return context.get("cost_budget", 1.0)
        elif criterion == DecisionCriterion.PROVIDER_HEALTH:
            provider = context.get("provider_type")
            if provider:
                return context.get("system_context", {}).get("provider_health", {}).get(provider, 1.0)
        elif criterion == DecisionCriterion.USER_PREFERENCE:
            return context.get("user_preferences", {})
        elif criterion == DecisionCriterion.SYSTEM_LOAD:
            return context.get("system_context", {}).get("active_requests", 0)
        elif criterion == DecisionCriterion.SESSION_PHASE:
            return context.get("session_context", {}).get("phase")
        elif criterion == DecisionCriterion.CAMPAIGN_GENRE:
            return context.get("campaign_context", {}).get("genre")
        elif criterion == DecisionCriterion.HISTORICAL_PERFORMANCE:
            return context.get("historical_performance", {})
        
        return None
    
    async def score_model_comprehensive(
        self,
        provider_type: ProviderType,
        model_id: str,
        task_type: TTRPGTaskType,
        task_characteristics: TaskCharacteristics,
        context: Dict[str, Any],
        user_preferences: Dict[str, Any],
        performance_profile: Optional[ModelPerformanceProfile] = None
    ) -> ModelScore:
        """Comprehensive scoring algorithm for a model candidate."""
        
        model_key = f"{provider_type.value}:{model_id}"
        score = ModelScore(provider_type=provider_type, model_id=model_id)
        
        # 1. Task Suitability Score
        score.task_suitability_score = await self._calculate_task_suitability_score(
            model_key, task_type, task_characteristics
        )
        
        # 2. Performance Score
        score.performance_score = await self._calculate_performance_score(
            model_key, task_type, performance_profile, context
        )
        
        # 3. Cost Score
        score.cost_score = await self._calculate_cost_score(
            model_key, task_characteristics, context, user_preferences
        )
        
        # 4. Reliability Score
        score.reliability_score = await self._calculate_reliability_score(
            provider_type, model_key, context, performance_profile
        )
        
        # 5. User Preference Score
        score.user_preference_score = await self._calculate_user_preference_score(
            provider_type, model_key, task_type, user_preferences
        )
        
        # 6. Context Fit Score
        score.context_fit_score = await self._calculate_context_fit_score(
            model_key, task_type, context
        )
        
        # Calculate weighted final score
        score.final_score = (
            score.task_suitability_score * self.scoring_weights["task_suitability"] +
            score.performance_score * self.scoring_weights["performance"] +
            score.cost_score * self.scoring_weights["cost"] +
            score.reliability_score * self.scoring_weights["reliability"] +
            score.user_preference_score * self.scoring_weights["user_preference"] +
            score.context_fit_score * self.scoring_weights["context_fit"]
        )
        
        # Calculate confidence based on data availability
        score.confidence = await self._calculate_confidence_score(
            performance_profile, user_preferences, context
        )
        
        # Generate detailed reasoning
        score.reasoning = self._generate_scoring_reasoning(score, task_type, context)
        
        # Store detailed breakdown
        score.scoring_details = {
            "task_suitability": score.task_suitability_score,
            "performance": score.performance_score,
            "cost": score.cost_score,
            "reliability": score.reliability_score,
            "user_preference": score.user_preference_score,
            "context_fit": score.context_fit_score,
            "final_score": score.final_score,
            "confidence": score.confidence
        }
        
        logger.debug(
            "Comprehensive model scoring completed",
            model=model_key,
            task_type=task_type.value,
            final_score=score.final_score,
            confidence=score.confidence
        )
        
        return score
    
    async def _calculate_task_suitability_score(
        self,
        model_key: str,
        task_type: TTRPGTaskType,
        task_characteristics: TaskCharacteristics
    ) -> float:
        """Calculate how suitable a model is for the specific task."""
        if model_key not in self.model_capabilities:
            return 0.5  # Default score for unknown models
        
        capabilities = self.model_capabilities[model_key]
        task_key = task_type.value.replace("_", "")
        
        # Base task suitability
        base_score = capabilities.get(task_key, 0.5)
        
        # Adjust based on task characteristics
        adjustments = 0.0
        
        if task_characteristics.requires_creativity:
            adjustments += capabilities.get("creativity", 0.5) * 0.1
        
        if task_characteristics.needs_tool_calling:
            adjustments += capabilities.get("tool_calling", 0.5) * 0.1
        
        if task_characteristics.needs_structured_output:
            adjustments += capabilities.get("structured_output", 0.5) * 0.1
        
        if task_characteristics.needs_multi_step_reasoning:
            adjustments += capabilities.get("multi_step_reasoning", 0.5) * 0.1
        
        # Accuracy requirements
        if task_characteristics.requires_accuracy:
            adjustments += capabilities.get("accuracy", 0.5) * 0.1
        
        # Consistency requirements
        if task_characteristics.requires_consistency:
            adjustments += capabilities.get("consistency", 0.5) * 0.05
        
        final_score = min(1.0, base_score + adjustments)
        return max(0.0, final_score)
    
    async def _calculate_performance_score(
        self,
        model_key: str,
        task_type: TTRPGTaskType,
        performance_profile: Optional[ModelPerformanceProfile],
        context: Dict[str, Any]
    ) -> float:
        """Calculate performance score based on historical data."""
        if not performance_profile:
            # Use capability-based estimation
            capabilities = self.model_capabilities.get(model_key, {})
            speed_score = capabilities.get("speed", 0.5)
            quality_score = capabilities.get(task_type.value.replace("_", ""), 0.5)
            return (speed_score + quality_score) / 2
        
        score = 0.0
        
        # Latency score
        latency_requirement = context.get("latency_requirement_ms", 5000)
        if performance_profile.avg_latency <= self.performance_thresholds["latency_excellent"]:
            latency_score = 1.0
        elif performance_profile.avg_latency <= self.performance_thresholds["latency_good"]:
            latency_score = 0.8
        elif performance_profile.avg_latency <= latency_requirement:
            latency_score = 0.6
        else:
            latency_score = max(0.0, 0.6 - (performance_profile.avg_latency - latency_requirement) / latency_requirement)
        
        # Quality score
        if performance_profile.avg_quality_score >= self.performance_thresholds["quality_excellent"]:
            quality_score = 1.0
        elif performance_profile.avg_quality_score >= self.performance_thresholds["quality_good"]:
            quality_score = 0.8
        elif performance_profile.avg_quality_score >= self.performance_thresholds["quality_acceptable"]:
            quality_score = 0.6
        else:
            quality_score = max(0.0, performance_profile.avg_quality_score)
        
        # Success rate score
        success_score = performance_profile.success_rate
        
        # Weighted average
        score = (latency_score * 0.4 + quality_score * 0.4 + success_score * 0.2)
        
        # Apply confidence adjustment
        score *= performance_profile.confidence_score
        
        return score
    
    async def _calculate_cost_score(
        self,
        model_key: str,
        task_characteristics: TaskCharacteristics,
        context: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> float:
        """Calculate cost efficiency score."""
        # Get estimated cost per request
        model_spec = self.model_registry.get(model_key)
        if not model_spec:
            # Use capability-based estimation
            capabilities = self.model_capabilities.get(model_key, {})
            cost_efficiency = capabilities.get("cost_efficiency", 0.5)
            return cost_efficiency
        
        # Estimate cost based on typical usage
        estimated_input_tokens = task_characteristics.context_length_needed * 0.75  # Assume 75% utilization
        estimated_output_tokens = task_characteristics.typical_output_length * 1.3   # Account for token efficiency
        
        total_cost = (
            model_spec.cost_per_input_token * estimated_input_tokens / 1000 +
            model_spec.cost_per_output_token * estimated_output_tokens / 1000
        )
        
        # Score based on cost efficiency
        user_budget = context.get("cost_budget", self.cost_targets["standard"])
        cost_sensitivity = user_preferences.get("cost_sensitivity", 0.5)
        
        if total_cost <= self.cost_targets["budget"]:
            cost_score = 1.0
        elif total_cost <= user_budget:
            cost_score = 0.8
        elif total_cost <= self.cost_targets["premium"]:
            cost_score = 0.6 - (cost_sensitivity * 0.2)
        else:
            cost_score = max(0.0, 0.4 - (cost_sensitivity * 0.3))
        
        return cost_score
    
    async def _calculate_reliability_score(
        self,
        provider_type: ProviderType,
        model_key: str,
        context: Dict[str, Any],
        performance_profile: Optional[ModelPerformanceProfile]
    ) -> float:
        """Calculate reliability score based on provider health and model stability."""
        score = 0.0
        
        # Provider health
        system_context = context.get("system_context", {})
        provider_health = system_context.get("provider_health", {}).get(provider_type, 1.0)
        health_score = provider_health
        
        # Provider load
        provider_load = system_context.get("provider_load", {}).get(provider_type, 0)
        if provider_load < 5:
            load_score = 1.0
        elif provider_load < 15:
            load_score = 0.8
        elif provider_load < 30:
            load_score = 0.6
        else:
            load_score = 0.4
        
        # Historical reliability
        if performance_profile:
            reliability_score = performance_profile.success_rate
        else:
            # Use default based on provider
            provider_reliability = {
                ProviderType.ANTHROPIC: 0.95,
                ProviderType.OPENAI: 0.93,
                ProviderType.GOOGLE: 0.90
            }
            reliability_score = provider_reliability.get(provider_type, 0.85)
        
        # Rate limiting risk
        rate_limit_risk = provider_type in system_context.get("rate_limits_approaching", set())
        rate_limit_score = 0.7 if rate_limit_risk else 1.0
        
        # Weighted average
        score = (health_score * 0.3 + load_score * 0.2 + reliability_score * 0.3 + rate_limit_score * 0.2)
        
        return score
    
    async def _calculate_user_preference_score(
        self,
        provider_type: ProviderType,
        model_key: str,
        task_type: TTRPGTaskType,
        user_preferences: Dict[str, Any]
    ) -> float:
        """Calculate score based on user preferences."""
        if user_preferences.get("confidence_score", 0) < 0.2:
            return 0.5  # Neutral score for users with little preference data
        
        score = 0.5  # Base score
        
        # Provider preference
        provider_prefs = user_preferences.get("provider_preferences", {})
        if provider_type.value in provider_prefs:
            provider_score = provider_prefs[provider_type.value]
            score += (provider_score - 0.5) * 0.3
        
        # Model preference
        model_prefs = user_preferences.get("model_preferences", {})
        if model_key in model_prefs:
            model_score = model_prefs[model_key]
            score += (model_score - 0.5) * 0.4
        
        # Task-specific preferences
        task_prefs = user_preferences.get("task_specific", {})
        if task_prefs:
            # Adjust based on user's preferred detail level, creativity, etc.
            preferred_creativity = task_prefs.get("preferred_creativity", 0.5)
            model_creativity = self.model_capabilities.get(model_key, {}).get("creativity", 0.5)
            creativity_match = 1 - abs(preferred_creativity - model_creativity)
            score += creativity_match * 0.2
            
            # Speed vs quality preference
            speed_importance = task_prefs.get("speed_importance", 0.5)
            model_speed = self.model_capabilities.get(model_key, {}).get("speed", 0.5)
            if speed_importance > 0.7:
                score += model_speed * 0.1
            elif speed_importance < 0.3:
                # Quality focused - penalize very fast models slightly
                quality_bonus = (1 - model_speed) * 0.1
                score += quality_bonus
        
        return max(0.0, min(1.0, score))
    
    async def _calculate_context_fit_score(
        self,
        model_key: str,
        task_type: TTRPGTaskType,
        context: Dict[str, Any]
    ) -> float:
        """Calculate how well the model fits the current context."""
        score = 0.5  # Base score
        
        capabilities = self.model_capabilities.get(model_key, {})
        
        # Session context adjustments
        session_context = context.get("session_context", {})
        
        if session_context.get("in_combat"):
            # Combat situations favor fast, accurate models
            speed_bonus = capabilities.get("speed", 0.5) * 0.2
            accuracy_bonus = capabilities.get("accuracy", 0.5) * 0.2
            score += speed_bonus + accuracy_bonus
        
        session_phase = session_context.get("phase")
        if session_phase == SessionPhase.COMBAT.value:
            score += capabilities.get("combat_resolution", 0.5) * 0.3
        elif session_phase in [SessionPhase.ROLEPLAY.value, SessionPhase.SOCIAL_INTERACTION.value]:
            score += capabilities.get("creativity", 0.5) * 0.2
        
        # Campaign context adjustments
        campaign_context = context.get("campaign_context", {})
        genre = campaign_context.get("genre")
        
        if genre == CampaignGenre.HORROR.value:
            # Horror campaigns benefit from creative, atmospheric responses
            creativity_bonus = capabilities.get("creativity", 0.5) * 0.2
            score += creativity_bonus
        elif genre == CampaignGenre.COMEDY.value:
            # Comedy benefits from creativity but consistency less important
            creativity_bonus = capabilities.get("creativity", 0.5) * 0.3
            score += creativity_bonus
        
        tone = campaign_context.get("tone", "balanced")
        if tone == "serious" and task_type in [TTRPGTaskType.STORY_GENERATION, TTRPGTaskType.DESCRIPTION_GENERATION]:
            consistency_bonus = capabilities.get("consistency", 0.5) * 0.2
            score += consistency_bonus
        
        # System context adjustments
        system_context = context.get("system_context", {})
        
        if system_context.get("peak_usage"):
            # During peak usage, prefer faster models
            speed_bonus = capabilities.get("speed", 0.5) * 0.3
            score += speed_bonus
        
        if system_context.get("budget_remaining", float('inf')) < 10:
            # Low budget favors cost-efficient models
            cost_bonus = capabilities.get("cost_efficiency", 0.5) * 0.4
            score += cost_bonus
        
        return max(0.0, min(1.0, score))
    
    async def _calculate_confidence_score(
        self,
        performance_profile: Optional[ModelPerformanceProfile],
        user_preferences: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence in the scoring based on available data."""
        confidence_factors = []
        
        # Performance data confidence
        if performance_profile:
            profile_confidence = performance_profile.confidence_score
            confidence_factors.append(profile_confidence)
        else:
            confidence_factors.append(0.3)  # Low confidence without performance data
        
        # User preference confidence
        user_confidence = user_preferences.get("confidence_score", 0.0)
        confidence_factors.append(user_confidence)
        
        # Context completeness confidence
        context_factors = context.get("context_factors", {})
        context_completeness = len(context_factors) / 5.0  # Assuming 5 main context types
        confidence_factors.append(min(1.0, context_completeness))
        
        # Overall confidence is the average of all factors
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        
        return overall_confidence
    
    def _generate_scoring_reasoning(
        self,
        score: ModelScore,
        task_type: TTRPGTaskType,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate human-readable reasoning for the score."""
        reasoning = []
        
        # Task suitability reasoning
        if score.task_suitability_score > 0.8:
            reasoning.append(f"Excellent match for {task_type.value} tasks")
        elif score.task_suitability_score > 0.6:
            reasoning.append(f"Good suitability for {task_type.value} tasks")
        else:
            reasoning.append(f"Limited suitability for {task_type.value} tasks")
        
        # Performance reasoning
        if score.performance_score > 0.8:
            reasoning.append("Strong historical performance")
        elif score.performance_score > 0.6:
            reasoning.append("Adequate historical performance")
        else:
            reasoning.append("Performance concerns based on historical data")
        
        # Cost reasoning
        if score.cost_score > 0.8:
            reasoning.append("Excellent cost efficiency")
        elif score.cost_score > 0.6:
            reasoning.append("Reasonable cost for the task")
        else:
            reasoning.append("Higher cost may impact budget")
        
        # Reliability reasoning
        if score.reliability_score > 0.8:
            reasoning.append("High reliability and availability")
        elif score.reliability_score > 0.6:
            reasoning.append("Generally reliable")
        else:
            reasoning.append("Reliability concerns detected")
        
        # User preference reasoning
        if score.user_preference_score > 0.7:
            reasoning.append("Matches your preferences well")
        elif score.user_preference_score > 0.5:
            reasoning.append("Neutral preference match")
        else:
            reasoning.append("May not match your typical preferences")
        
        # Context reasoning
        session_context = context.get("session_context", {})
        if session_context.get("in_combat"):
            reasoning.append("Optimized for combat situations")
        
        campaign_context = context.get("campaign_context", {})
        if campaign_context.get("genre"):
            reasoning.append(f"Suitable for {campaign_context['genre']} campaigns")
        
        return reasoning
    
    async def get_top_models(
        self,
        task_type: TTRPGTaskType,
        task_characteristics: TaskCharacteristics,
        context: Dict[str, Any],
        user_preferences: Dict[str, Any],
        available_models: List[Tuple[ProviderType, str]],
        top_n: int = 3
    ) -> List[ModelScore]:
        """Get the top N models for a given task and context."""
        
        model_scores = []
        
        for provider_type, model_id in available_models:
            # Get performance profile if available
            performance_profile = self.performance_profiles.get(f"{provider_type.value}:{model_id}")
            
            # Calculate comprehensive score
            score = await self.score_model_comprehensive(
                provider_type, model_id, task_type, task_characteristics,
                context, user_preferences, performance_profile
            )
            
            model_scores.append(score)
        
        # Sort by final score (descending)
        model_scores.sort(key=lambda s: s.final_score, reverse=True)
        
        # Return top N
        return model_scores[:top_n]
    
    async def update_node_performance(
        self,
        node_id: str,
        success: bool,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Update decision tree node performance based on actual results."""
        
        def find_node_by_id(node: DecisionTreeNode, target_id: str) -> Optional[DecisionTreeNode]:
            if node.node_id == target_id:
                return node
            for child in node.children:
                result = find_node_by_id(child, target_id)
                if result:
                    return result
            return None
        
        node = find_node_by_id(self.root_node, node_id)
        if not node:
            logger.warning("Node not found for performance update", node_id=node_id)
            return
        
        # Update success statistics
        if success:
            node.success_count += 1
        
        node.evaluation_count += 1
        node.success_rate = node.success_count / node.evaluation_count
        node.last_updated = datetime.now()
        
        # Log performance update
        logger.debug(
            "Updated decision tree node performance",
            node_id=node_id,
            node_name=node.name,
            success=success,
            success_rate=node.success_rate
        )
    
    def export_decision_tree(self) -> Dict[str, Any]:
        """Export the decision tree structure for analysis or backup."""
        
        def export_node(node: DecisionTreeNode) -> Dict[str, Any]:
            return {
                "node_id": node.node_id,
                "name": node.name,
                "description": node.description,
                "conditions": [
                    {
                        "criterion": cond.criterion.value,
                        "operator": cond.operator.value,
                        "value": cond.value,
                        "weight": cond.weight
                    }
                    for cond in node.conditions
                ],
                "recommended_models": [
                    {
                        "provider": provider.value,
                        "model": model,
                        "score": score
                    }
                    for provider, model, score in node.recommended_models
                ],
                "confidence": node.confidence,
                "priority": node.priority,
                "success_rate": node.success_rate,
                "evaluation_count": node.evaluation_count,
                "children": [export_node(child) for child in node.children]
            }
        
        return {
            "decision_tree": export_node(self.root_node),
            "scoring_weights": self.scoring_weights,
            "performance_thresholds": self.performance_thresholds,
            "cost_targets": self.cost_targets,
            "export_timestamp": datetime.now().isoformat()
        }
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get statistics about decision tree usage and performance."""
        
        def collect_stats(node: DecisionTreeNode, stats: Dict[str, Any]) -> None:
            stats["total_nodes"] += 1
            stats["total_evaluations"] += node.evaluation_count
            stats["total_successes"] += node.success_count
            
            if not node.children:  # Leaf node
                stats["leaf_nodes"] += 1
                if node.evaluation_count > 0:
                    stats["active_leaf_nodes"] += 1
                    stats["leaf_success_rates"].append(node.success_rate)
            
            for child in node.children:
                collect_stats(child, stats)
        
        stats = {
            "total_nodes": 0,
            "leaf_nodes": 0,
            "active_leaf_nodes": 0,
            "total_evaluations": 0,
            "total_successes": 0,
            "leaf_success_rates": []
        }
        
        collect_stats(self.root_node, stats)
        
        # Calculate aggregate statistics
        if stats["total_evaluations"] > 0:
            stats["overall_success_rate"] = stats["total_successes"] / stats["total_evaluations"]
        else:
            stats["overall_success_rate"] = 0.0
        
        if stats["leaf_success_rates"]:
            stats["average_leaf_success_rate"] = sum(stats["leaf_success_rates"]) / len(stats["leaf_success_rates"])
            stats["best_leaf_success_rate"] = max(stats["leaf_success_rates"])
            stats["worst_leaf_success_rate"] = min(stats["leaf_success_rates"])
        else:
            stats["average_leaf_success_rate"] = 0.0
            stats["best_leaf_success_rate"] = 0.0
            stats["worst_leaf_success_rate"] = 0.0
        
        return stats