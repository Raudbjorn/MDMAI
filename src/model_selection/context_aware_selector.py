"""Context-aware model switching logic for intelligent AI model selection."""

import asyncio
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import structlog
from .task_categorizer import TTRPGTaskType, TaskCharacteristics, TaskCategorizer
from .model_optimizer import ModelOptimizer, ModelRecommendation
from .performance_profiler import PerformanceBenchmark
from .preference_learner import PreferenceLearner
from ..ai_providers.models import ProviderType, ModelSpec

logger = structlog.get_logger(__name__)


class ContextType(Enum):
    """Types of context that influence model selection."""
    
    SESSION_STATE = "session_state"           # Combat, roleplay, planning phases
    CAMPAIGN_CONTEXT = "campaign_context"     # Genre, theme, setting details  
    USER_PROFILE = "user_profile"             # User preferences and history
    SYSTEM_STATE = "system_state"             # Provider health, load, costs
    TEMPORAL_CONTEXT = "temporal_context"     # Time of day, session duration
    PERFORMANCE_CONTEXT = "performance_context"  # Recent performance metrics


class SessionPhase(Enum):
    """Current phase of a TTRPG session."""
    
    PLANNING = "planning"                     # Session planning, prep
    INTRODUCTION = "introduction"             # Session start, recap
    EXPLORATION = "exploration"               # Investigation, travel, discovery  
    SOCIAL_INTERACTION = "social_interaction" # NPC conversations, negotiations
    COMBAT = "combat"                         # Active combat encounters
    PUZZLE_SOLVING = "puzzle_solving"         # Riddles, traps, challenges
    ROLEPLAY = "roleplay"                     # Character development, drama
    CONCLUSION = "conclusion"                 # Session wrap-up, XP, planning


class CampaignGenre(Enum):
    """Campaign genre affecting model selection."""
    
    FANTASY = "fantasy"                       # Traditional D&D, fantasy settings
    SCI_FI = "sci_fi"                        # Space opera, cyberpunk, future
    HORROR = "horror"                        # Gothic horror, cosmic horror
    MODERN = "modern"                        # Contemporary settings
    HISTORICAL = "historical"                # Period-accurate settings
    SUPERHERO = "superhero"                  # Comic book style campaigns
    MYSTERY = "mystery"                      # Investigation-focused games
    COMEDY = "comedy"                        # Light-hearted, humorous tone
    GRITTY = "gritty"                        # Dark, realistic themes
    HIGH_FANTASY = "high_fantasy"            # Epic, magical campaigns


@dataclass
class SessionContext:
    """Current session context information."""
    
    session_id: str = field(default_factory=lambda: str(uuid4()))
    phase: SessionPhase = SessionPhase.PLANNING
    
    # Session state
    is_active: bool = False
    in_combat: bool = False
    combat_round: int = 0
    initiative_active: bool = False
    
    # Timing information  
    session_start_time: Optional[datetime] = None
    current_scene_duration: timedelta = field(default_factory=timedelta)
    total_session_duration: timedelta = field(default_factory=timedelta)
    
    # Participants
    active_players: List[str] = field(default_factory=list)
    dm_user_id: Optional[str] = None
    
    # Recent activity
    recent_actions: deque = field(default_factory=lambda: deque(maxlen=10))
    last_task_type: Optional[TTRPGTaskType] = None
    task_sequence: List[TTRPGTaskType] = field(default_factory=list)
    
    # Performance tracking
    recent_response_times: deque = field(default_factory=lambda: deque(maxlen=5))
    recent_quality_scores: deque = field(default_factory=lambda: deque(maxlen=5))


@dataclass 
class CampaignContext:
    """Campaign-specific context information."""
    
    campaign_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    genre: CampaignGenre = CampaignGenre.FANTASY
    
    # Campaign characteristics
    tone: str = "balanced"  # "serious", "light", "dramatic", "comedic"
    complexity_level: str = "moderate"  # "simple", "moderate", "complex"
    player_experience: str = "mixed"  # "new", "experienced", "mixed"
    
    # Content preferences
    violence_level: str = "moderate"  # "minimal", "moderate", "high"
    romance_content: bool = False
    political_intrigue: bool = False
    cosmic_themes: bool = False
    
    # System and rules
    game_system: str = "D&D 5e"
    homebrew_content: bool = False
    house_rules: List[str] = field(default_factory=list)
    
    # Narrative elements
    current_story_arc: Optional[str] = None
    major_npcs: List[str] = field(default_factory=list)
    key_locations: List[str] = field(default_factory=list)
    ongoing_plots: List[str] = field(default_factory=list)


@dataclass
class SystemContext:
    """Current system state context."""
    
    # Provider health and availability
    provider_health: Dict[ProviderType, float] = field(default_factory=dict)
    provider_load: Dict[ProviderType, int] = field(default_factory=dict)
    provider_costs: Dict[ProviderType, float] = field(default_factory=dict)
    
    # Resource constraints
    total_budget_remaining: float = 0.0
    hourly_budget_remaining: float = 0.0
    rate_limits_approaching: Set[ProviderType] = field(default_factory=set)
    
    # Performance metrics
    recent_error_rates: Dict[ProviderType, float] = field(default_factory=dict)
    average_response_times: Dict[ProviderType, float] = field(default_factory=dict)
    
    # System load
    active_requests: int = 0
    queue_length: int = 0
    peak_usage_detected: bool = False


class ContextAwareSelector:
    """Context-aware model selection system for TTRPG Assistant."""
    
    def __init__(
        self,
        task_categorizer: TaskCategorizer,
        model_optimizer: ModelOptimizer,
        performance_benchmark: PerformanceBenchmark,
        preference_learner: PreferenceLearner
    ):
        self.task_categorizer = task_categorizer
        self.model_optimizer = model_optimizer
        self.performance_benchmark = performance_benchmark
        self.preference_learner = preference_learner
        
        # Context storage
        self.session_contexts: Dict[str, SessionContext] = {}
        self.campaign_contexts: Dict[str, CampaignContext] = {}
        self.system_context = SystemContext()
        
        # Context-specific optimization rules
        self.context_rules = self._initialize_context_rules()
        
        # Caching for performance
        self.context_cache: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}
        self.selection_cache: Dict[str, Tuple[datetime, ModelRecommendation]] = {}
        
        # Learning and adaptation
        self.context_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.model_performance_by_context: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    def _initialize_context_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize context-specific optimization rules."""
        return {
            # Combat phase rules
            f"{SessionPhase.COMBAT.value}": {
                "latency_multiplier": 2.0,      # Double importance of speed
                "accuracy_multiplier": 1.5,     # Higher importance of accuracy
                "creativity_multiplier": 0.7,   # Less importance on creativity
                "preferred_models": ["claude-3-5-haiku", "gpt-4o-mini"],  # Fast models
                "max_response_time": 500        # 500ms max for combat
            },
            
            # Story generation rules
            f"{TTRPGTaskType.STORY_GENERATION.value}": {
                "creativity_multiplier": 2.0,   # Double importance of creativity
                "quality_multiplier": 1.8,      # High quality importance
                "cost_multiplier": 0.8,         # Less cost sensitivity
                "preferred_models": ["claude-3-5-sonnet", "gpt-4o"],  # High-quality models
                "min_context_length": 8192      # Need larger context
            },
            
            # Horror genre rules
            f"{CampaignGenre.HORROR.value}": {
                "atmosphere_importance": 2.0,   # Prioritize atmospheric responses
                "creativity_multiplier": 1.5,   # Higher creativity for mood
                "preferred_providers": [ProviderType.ANTHROPIC],  # Better at creative writing
                "tone_keywords": ["dark", "atmospheric", "suspenseful"]
            },
            
            # New player rules
            "player_experience_new": {
                "clarity_multiplier": 2.0,      # Prioritize clear explanations
                "complexity_penalty": 1.5,      # Penalize overly complex responses
                "preferred_detail_level": "comprehensive",
                "explanation_priority": True
            },
            
            # High system load rules
            "high_system_load": {
                "latency_multiplier": 1.8,      # Prioritize speed under load
                "cost_multiplier": 1.3,         # Slightly more cost sensitive
                "preferred_providers": [],       # Determined dynamically
                "fallback_strategy": "fastest_available"
            }
        }
    
    async def select_optimal_model(
        self,
        user_id: str,
        user_input: str,
        session_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
        force_refresh: bool = False
    ) -> ModelRecommendation:
        """Select the optimal model based on comprehensive context analysis."""
        
        logger.info(
            "Starting context-aware model selection",
            user_id=user_id,
            session_id=session_id,
            campaign_id=campaign_id,
            input_length=len(user_input)
        )
        
        # Check cache first
        if not force_refresh:
            cache_key = self._generate_selection_cache_key(user_id, user_input, session_id, campaign_id)
            if cache_key in self.selection_cache:
                cached_time, cached_recommendation = self.selection_cache[cache_key]
                if datetime.now() - cached_time < timedelta(minutes=2):  # 2-minute cache
                    logger.debug("Using cached model selection", cache_key=cache_key)
                    return cached_recommendation
        
        # 1. Categorize the task
        task_type, confidence = self.task_categorizer.categorize_task(
            user_input,
            context=await self._build_task_context(user_id, session_id, campaign_id)
        )
        
        # 2. Get task characteristics
        task_characteristics = self.task_categorizer.get_task_characteristics(task_type)
        
        # 3. Gather all context information
        context_analysis = await self._analyze_comprehensive_context(
            user_id, session_id, campaign_id, task_type, task_characteristics
        )
        
        # 4. Get user preferences
        user_preferences = await self.preference_learner.get_user_preferences(
            user_id, task_type, campaign_id
        )
        
        # 5. Apply context-aware adjustments to task characteristics
        adjusted_characteristics = await self._apply_context_adjustments(
            task_characteristics, context_analysis, user_preferences
        )
        
        # 6. Get base model recommendation from optimizer
        base_recommendation = await self.model_optimizer.optimize_model_selection(
            task_type, adjusted_characteristics, context_analysis, user_preferences
        )
        
        # 7. Apply context-specific overrides and refinements
        final_recommendation = await self._apply_context_overrides(
            base_recommendation, context_analysis, task_type
        )
        
        # 8. Cache the recommendation
        cache_key = self._generate_selection_cache_key(user_id, user_input, session_id, campaign_id)
        self.selection_cache[cache_key] = (datetime.now(), final_recommendation)
        
        # 9. Record the decision for learning
        await self._record_context_decision(
            user_id, task_type, context_analysis, final_recommendation
        )
        
        logger.info(
            "Context-aware model selection completed",
            user_id=user_id,
            selected_model=f"{final_recommendation.provider_type.value}:{final_recommendation.model_id}",
            task_type=task_type.value,
            confidence=final_recommendation.confidence,
            context_factors=len(context_analysis)
        )
        
        return final_recommendation
    
    async def _build_task_context(
        self,
        user_id: str,
        session_id: Optional[str],
        campaign_id: Optional[str]
    ) -> Dict[str, Any]:
        """Build context dictionary for task categorization."""
        context = {}
        
        # Session context
        if session_id and session_id in self.session_contexts:
            session_ctx = self.session_contexts[session_id]
            context.update({
                "in_combat": session_ctx.in_combat,
                "session_active": session_ctx.is_active,
                "session_phase": session_ctx.phase.value,
                "recent_tasks": [t.value for t in session_ctx.task_sequence[-3:]]
            })
        
        # Campaign context
        if campaign_id and campaign_id in self.campaign_contexts:
            campaign_ctx = self.campaign_contexts[campaign_id]
            context.update({
                "campaign_genre": campaign_ctx.genre.value,
                "campaign_tone": campaign_ctx.tone,
                "player_experience": campaign_ctx.player_experience
            })
        
        # User preferences
        user_prefs = await self.preference_learner.get_user_preferences(user_id)
        context["user_preferences"] = user_prefs
        
        return context
    
    async def _analyze_comprehensive_context(
        self,
        user_id: str,
        session_id: Optional[str],
        campaign_id: Optional[str],
        task_type: TTRPGTaskType,
        task_characteristics: TaskCharacteristics
    ) -> Dict[str, Any]:
        """Analyze comprehensive context for model selection."""
        context_analysis = {
            "timestamp": datetime.now(),
            "user_id": user_id,
            "task_type": task_type.value,
            "context_factors": {}
        }
        
        # Session context analysis
        if session_id and session_id in self.session_contexts:
            session_ctx = self.session_contexts[session_id]
            context_analysis["context_factors"]["session"] = {
                "phase": session_ctx.phase.value,
                "in_combat": session_ctx.in_combat,
                "session_duration_minutes": session_ctx.total_session_duration.total_seconds() / 60,
                "recent_performance": {
                    "avg_response_time": sum(session_ctx.recent_response_times) / len(session_ctx.recent_response_times) 
                    if session_ctx.recent_response_times else 0,
                    "avg_quality": sum(session_ctx.recent_quality_scores) / len(session_ctx.recent_quality_scores)
                    if session_ctx.recent_quality_scores else 0
                },
                "task_sequence_pattern": self._analyze_task_sequence(session_ctx.task_sequence)
            }
        
        # Campaign context analysis
        if campaign_id and campaign_id in self.campaign_contexts:
            campaign_ctx = self.campaign_contexts[campaign_id]
            context_analysis["context_factors"]["campaign"] = {
                "genre": campaign_ctx.genre.value,
                "tone": campaign_ctx.tone,
                "complexity_level": campaign_ctx.complexity_level,
                "player_experience": campaign_ctx.player_experience,
                "content_flags": {
                    "violence_level": campaign_ctx.violence_level,
                    "romance_content": campaign_ctx.romance_content,
                    "political_intrigue": campaign_ctx.political_intrigue
                }
            }
        
        # System context analysis
        context_analysis["context_factors"]["system"] = {
            "provider_health": dict(self.system_context.provider_health),
            "provider_load": dict(self.system_context.provider_load),
            "budget_remaining": self.system_context.total_budget_remaining,
            "rate_limits_approaching": [p.value for p in self.system_context.rate_limits_approaching],
            "peak_usage": self.system_context.peak_usage_detected
        }
        
        # Temporal context analysis
        now = datetime.now()
        context_analysis["context_factors"]["temporal"] = {
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
            "is_weekend": now.weekday() >= 5,
            "is_peak_hours": 18 <= now.hour <= 22,  # Evening gaming hours
            "time_zone_offset": 0  # Could be user-specific
        }
        
        # Performance context analysis
        recent_performance = await self._analyze_recent_performance(user_id, task_type)
        context_analysis["context_factors"]["performance"] = recent_performance
        
        return context_analysis
    
    def _analyze_task_sequence(self, task_sequence: List[TTRPGTaskType]) -> Dict[str, Any]:
        """Analyze patterns in recent task sequence."""
        if len(task_sequence) < 2:
            return {"pattern": "insufficient_data"}
        
        recent_tasks = task_sequence[-5:]  # Last 5 tasks
        
        # Check for common patterns
        patterns = {
            "repeating_task": len(set(recent_tasks)) == 1,
            "alternating_pattern": len(recent_tasks) >= 4 and 
                                 recent_tasks[0] == recent_tasks[2] and recent_tasks[1] == recent_tasks[3],
            "escalating_complexity": self._is_escalating_complexity(recent_tasks),
            "combat_sequence": TTRPGTaskType.COMBAT_RESOLUTION in recent_tasks,
            "story_focus": TTRPGTaskType.STORY_GENERATION in recent_tasks or 
                          TTRPGTaskType.DESCRIPTION_GENERATION in recent_tasks
        }
        
        return {
            "recent_tasks": [t.value for t in recent_tasks],
            "patterns": patterns,
            "dominant_task": max(recent_tasks, key=recent_tasks.count).value if recent_tasks else None
        }
    
    def _is_escalating_complexity(self, tasks: List[TTRPGTaskType]) -> bool:
        """Check if task complexity is escalating."""
        complexity_scores = {
            TTRPGTaskType.RULE_LOOKUP: 1,
            TTRPGTaskType.CHARACTER_GENERATION: 2,
            TTRPGTaskType.COMBAT_RESOLUTION: 2,
            TTRPGTaskType.DESCRIPTION_GENERATION: 3,
            TTRPGTaskType.STORY_GENERATION: 4,
            TTRPGTaskType.WORLD_BUILDING: 4
        }
        
        scores = [complexity_scores.get(task, 2) for task in tasks]
        if len(scores) < 3:
            return False
        
        # Check if generally increasing
        increasing_count = sum(1 for i in range(1, len(scores)) if scores[i] > scores[i-1])
        return increasing_count > len(scores) // 2
    
    async def _analyze_recent_performance(
        self,
        user_id: str,
        task_type: TTRPGTaskType
    ) -> Dict[str, Any]:
        """Analyze recent performance for context."""
        # This would typically query the performance benchmark system
        # For now, return mock data structure
        return {
            "recent_latency_trend": "stable",  # "improving", "degrading", "stable"
            "recent_quality_trend": "stable",
            "recent_error_rate": 0.02,
            "satisfaction_trend": "improving",
            "model_switches_recent": 1,
            "performance_issues": []
        }
    
    async def _apply_context_adjustments(
        self,
        task_characteristics: TaskCharacteristics,
        context_analysis: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> TaskCharacteristics:
        """Apply context-aware adjustments to task characteristics."""
        # Create a copy to modify
        adjusted = TaskCharacteristics(
            task_type=task_characteristics.task_type,
            complexity=task_characteristics.complexity,
            latency_requirement=task_characteristics.latency_requirement,
            requires_creativity=task_characteristics.requires_creativity,
            requires_accuracy=task_characteristics.requires_accuracy,
            requires_consistency=task_characteristics.requires_consistency,
            context_length_needed=task_characteristics.context_length_needed,
            typical_output_length=task_characteristics.typical_output_length,
            needs_tool_calling=task_characteristics.needs_tool_calling,
            needs_structured_output=task_characteristics.needs_structured_output,
            needs_multi_step_reasoning=task_characteristics.needs_multi_step_reasoning,
            genre_sensitivity=task_characteristics.genre_sensitivity,
            personality_awareness=task_characteristics.personality_awareness,
            rules_knowledge=task_characteristics.rules_knowledge,
            cost_sensitivity=task_characteristics.cost_sensitivity,
            quality_importance=task_characteristics.quality_importance
        )
        
        context_factors = context_analysis.get("context_factors", {})
        
        # Session context adjustments
        session_ctx = context_factors.get("session", {})
        if session_ctx.get("in_combat"):
            # Combat situations require faster responses
            adjusted.latency_requirement = min(adjusted.latency_requirement, 
                                             self.task_categorizer.task_characteristics[TTRPGTaskType.COMBAT_RESOLUTION].latency_requirement)
            adjusted.quality_importance *= 1.2  # Higher accuracy importance
            adjusted.cost_sensitivity *= 0.8    # Less cost sensitive in combat
        
        if session_ctx.get("session_duration_minutes", 0) > 180:  # Long session (3+ hours)
            adjusted.cost_sensitivity *= 1.2    # More cost conscious in long sessions
        
        # Campaign context adjustments
        campaign_ctx = context_factors.get("campaign", {})
        campaign_genre = campaign_ctx.get("genre")
        
        if campaign_genre == CampaignGenre.HORROR.value:
            adjusted.requires_creativity = True
            adjusted.personality_awareness = True
            adjusted.typical_output_length = int(adjusted.typical_output_length * 1.3)  # More atmospheric detail
        
        elif campaign_genre == CampaignGenre.COMEDY.value:
            adjusted.requires_creativity = True
            adjusted.requires_consistency = False  # Allow more playful responses
        
        if campaign_ctx.get("player_experience") == "new":
            adjusted.typical_output_length = int(adjusted.typical_output_length * 1.4)  # More explanation
            adjusted.context_length_needed = int(adjusted.context_length_needed * 1.2)  # More context for clarity
        
        # System context adjustments
        system_ctx = context_factors.get("system", {})
        if system_ctx.get("budget_remaining", 0) < 10:  # Low budget
            adjusted.cost_sensitivity = max(0.8, adjusted.cost_sensitivity)
        
        if system_ctx.get("peak_usage"):
            adjusted.latency_requirement = min(adjusted.latency_requirement,
                                             self.task_categorizer.task_characteristics[TTRPGTaskType.IMPROVISATION].latency_requirement)
        
        # User preference adjustments
        if user_preferences.get("confidence_score", 0) > 0.5:  # Established preferences
            # Apply learned preferences
            if user_preferences.get("detail_level_preference", 0.5) > 0.7:
                adjusted.typical_output_length = int(adjusted.typical_output_length * 1.5)
            elif user_preferences.get("detail_level_preference", 0.5) < 0.3:
                adjusted.typical_output_length = int(adjusted.typical_output_length * 0.7)
            
            if user_preferences.get("speed_vs_quality", 0.5) > 0.7:  # Quality focused
                adjusted.quality_importance *= 1.3
                adjusted.cost_sensitivity *= 0.8
            elif user_preferences.get("speed_vs_quality", 0.5) < 0.3:  # Speed focused
                adjusted.latency_requirement = min(adjusted.latency_requirement,
                                                 self.task_categorizer.task_characteristics[TTRPGTaskType.IMPROVISATION].latency_requirement)
        
        return adjusted
    
    async def _apply_context_overrides(
        self,
        base_recommendation: ModelRecommendation,
        context_analysis: Dict[str, Any],
        task_type: TTRPGTaskType
    ) -> ModelRecommendation:
        """Apply context-specific overrides to the base recommendation."""
        context_factors = context_analysis.get("context_factors", {})
        
        # Check for hard context rules that override the base recommendation
        overrides_applied = []
        
        # Combat override - force fast model
        session_ctx = context_factors.get("session", {})
        if session_ctx.get("in_combat") and base_recommendation.expected_performance.get("latency", 0) > 1000:
            # Find fastest available model
            fast_models = [
                (ProviderType.ANTHROPIC, "claude-3-5-haiku"),
                (ProviderType.OPENAI, "gpt-4o-mini"),
                (ProviderType.GOOGLE, "gemini-1.5-flash")
            ]
            
            for provider, model in fast_models:
                if provider in context_factors.get("system", {}).get("provider_health", {}):
                    if context_factors["system"]["provider_health"][provider] > 0.8:
                        base_recommendation.provider_type = provider
                        base_recommendation.model_id = model
                        base_recommendation.reasoning.append("Override: Combat situation requires fastest model")
                        overrides_applied.append("combat_speed_override")
                        break
        
        # Budget constraint override
        system_ctx = context_factors.get("system", {})
        if system_ctx.get("budget_remaining", float('inf')) < 5:  # Very low budget
            budget_models = [
                (ProviderType.ANTHROPIC, "claude-3-5-haiku"),
                (ProviderType.OPENAI, "gpt-4o-mini")
            ]
            
            for provider, model in budget_models:
                if provider in system_ctx.get("provider_health", {}):
                    if system_ctx["provider_health"][provider] > 0.7:
                        base_recommendation.provider_type = provider
                        base_recommendation.model_id = model
                        base_recommendation.reasoning.append("Override: Budget constraints require cost-efficient model")
                        overrides_applied.append("budget_override")
                        break
        
        # Provider health override
        current_provider_health = system_ctx.get("provider_health", {}).get(base_recommendation.provider_type, 1.0)
        if current_provider_health < 0.5:  # Provider is unhealthy
            # Find healthiest alternative
            healthy_providers = [
                (provider, health) for provider, health in system_ctx.get("provider_health", {}).items()
                if health > 0.8
            ]
            
            if healthy_providers:
                best_provider = max(healthy_providers, key=lambda x: x[1])[0]
                # Use default model for the healthy provider
                default_models = {
                    ProviderType.ANTHROPIC: "claude-3-5-sonnet",
                    ProviderType.OPENAI: "gpt-4o",
                    ProviderType.GOOGLE: "gemini-1.5-pro"
                }
                
                base_recommendation.provider_type = best_provider
                base_recommendation.model_id = default_models.get(best_provider, "claude-3-5-sonnet")
                base_recommendation.reasoning.append("Override: Primary provider unhealthy, switched to healthy alternative")
                overrides_applied.append("health_override")
        
        # Genre-specific model preferences
        campaign_ctx = context_factors.get("campaign", {})
        genre = campaign_ctx.get("genre")
        
        if genre == CampaignGenre.HORROR.value and task_type in [TTRPGTaskType.DESCRIPTION_GENERATION, TTRPGTaskType.STORY_GENERATION]:
            # Prefer Claude for atmospheric content
            if ProviderType.ANTHROPIC in system_ctx.get("provider_health", {}):
                if system_ctx["provider_health"][ProviderType.ANTHROPIC] > 0.7:
                    base_recommendation.provider_type = ProviderType.ANTHROPIC
                    base_recommendation.model_id = "claude-3-5-sonnet"
                    base_recommendation.reasoning.append("Override: Horror genre prefers Anthropic for atmospheric content")
                    overrides_applied.append("genre_preference_override")
        
        # Log applied overrides
        if overrides_applied:
            logger.info(
                "Applied context overrides to model recommendation",
                overrides=overrides_applied,
                final_model=f"{base_recommendation.provider_type.value}:{base_recommendation.model_id}"
            )
        
        return base_recommendation
    
    async def _record_context_decision(
        self,
        user_id: str,
        task_type: TTRPGTaskType,
        context_analysis: Dict[str, Any],
        recommendation: ModelRecommendation
    ) -> None:
        """Record context-aware decision for learning."""
        decision_key = f"{task_type.value}:{recommendation.provider_type.value}:{recommendation.model_id}"
        
        # Extract key context factors
        context_signature = []
        context_factors = context_analysis.get("context_factors", {})
        
        if "session" in context_factors:
            session_ctx = context_factors["session"]
            if session_ctx.get("in_combat"):
                context_signature.append("combat")
            context_signature.append(session_ctx.get("phase", "unknown"))
        
        if "campaign" in context_factors:
            campaign_ctx = context_factors["campaign"]
            context_signature.append(campaign_ctx.get("genre", "unknown"))
            context_signature.append(campaign_ctx.get("tone", "balanced"))
        
        if "system" in context_factors:
            system_ctx = context_factors["system"]
            if system_ctx.get("peak_usage"):
                context_signature.append("peak_load")
            if system_ctx.get("budget_remaining", float('inf')) < 10:
                context_signature.append("low_budget")
        
        context_key = "|".join(context_signature)
        self.context_patterns[context_key][decision_key] += 1
        
        logger.debug(
            "Recorded context-aware decision",
            user_id=user_id,
            context_key=context_key,
            decision=decision_key,
            confidence=recommendation.confidence
        )
    
    def _generate_selection_cache_key(
        self,
        user_id: str,
        user_input: str,
        session_id: Optional[str],
        campaign_id: Optional[str]
    ) -> str:
        """Generate cache key for model selection."""
        # Create a simplified hash of key factors
        key_parts = [
            user_id,
            str(len(user_input)),  # Use length instead of full input for privacy
            session_id or "no_session",
            campaign_id or "no_campaign"
        ]
        
        # Add context state indicators
        if session_id and session_id in self.session_contexts:
            session_ctx = self.session_contexts[session_id]
            key_parts.extend([
                session_ctx.phase.value,
                str(session_ctx.in_combat)
            ])
        
        return "|".join(key_parts)
    
    async def update_session_context(
        self,
        session_id: str,
        phase: Optional[SessionPhase] = None,
        in_combat: Optional[bool] = None,
        task_type: Optional[TTRPGTaskType] = None,
        response_time: Optional[float] = None,
        quality_score: Optional[float] = None
    ) -> None:
        """Update session context information."""
        if session_id not in self.session_contexts:
            self.session_contexts[session_id] = SessionContext(session_id=session_id)
        
        context = self.session_contexts[session_id]
        
        if phase is not None:
            context.phase = phase
        
        if in_combat is not None:
            context.in_combat = in_combat
        
        if task_type is not None:
            context.last_task_type = task_type
            context.task_sequence.append(task_type)
            # Keep only last 20 tasks
            if len(context.task_sequence) > 20:
                context.task_sequence = context.task_sequence[-20:]
        
        if response_time is not None:
            context.recent_response_times.append(response_time)
        
        if quality_score is not None:
            context.recent_quality_scores.append(quality_score)
        
        logger.debug(
            "Updated session context",
            session_id=session_id,
            phase=phase.value if phase else None,
            in_combat=in_combat,
            task_sequence_length=len(context.task_sequence)
        )
    
    async def update_campaign_context(
        self,
        campaign_id: str,
        **kwargs
    ) -> None:
        """Update campaign context information."""
        if campaign_id not in self.campaign_contexts:
            self.campaign_contexts[campaign_id] = CampaignContext(campaign_id=campaign_id)
        
        context = self.campaign_contexts[campaign_id]
        
        for key, value in kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)
        
        logger.debug("Updated campaign context", campaign_id=campaign_id, updates=list(kwargs.keys()))
    
    async def update_system_context(
        self,
        provider_health: Optional[Dict[ProviderType, float]] = None,
        provider_load: Optional[Dict[ProviderType, int]] = None,
        budget_remaining: Optional[float] = None,
        peak_usage: Optional[bool] = None
    ) -> None:
        """Update system context information."""
        if provider_health is not None:
            self.system_context.provider_health.update(provider_health)
        
        if provider_load is not None:
            self.system_context.provider_load.update(provider_load)
        
        if budget_remaining is not None:
            self.system_context.total_budget_remaining = budget_remaining
        
        if peak_usage is not None:
            self.system_context.peak_usage_detected = peak_usage
        
        logger.debug(
            "Updated system context",
            provider_health_count=len(self.system_context.provider_health),
            budget_remaining=self.system_context.total_budget_remaining,
            peak_usage=self.system_context.peak_usage_detected
        )
    
    def get_context_insights(
        self,
        session_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """Get insights about context patterns and their impact on model selection."""
        insights = {
            "context_patterns": {},
            "model_preferences_by_context": {},
            "performance_by_context": {},
            "recommendations": []
        }
        
        # Analyze context patterns
        for context_key, decisions in self.context_patterns.items():
            if decisions:
                most_common = max(decisions.items(), key=lambda x: x[1])
                insights["context_patterns"][context_key] = {
                    "most_common_decision": most_common[0],
                    "frequency": most_common[1],
                    "total_decisions": sum(decisions.values()),
                    "decision_variety": len(decisions)
                }
        
        # Generate recommendations based on patterns
        if insights["context_patterns"]:
            stable_patterns = [
                context for context, data in insights["context_patterns"].items()
                if data["decision_variety"] <= 2 and data["total_decisions"] >= 5
            ]
            
            if stable_patterns:
                insights["recommendations"].append({
                    "type": "stable_context_patterns",
                    "message": f"Found {len(stable_patterns)} stable context patterns that could benefit from caching",
                    "contexts": stable_patterns
                })
        
        return insights
    
    async def cleanup_old_context_data(self, retention_hours: int = 48) -> int:
        """Clean up old context data."""
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        cleaned_count = 0
        
        # Clean up session contexts
        session_ids_to_remove = []
        for session_id, context in self.session_contexts.items():
            if context.session_start_time and context.session_start_time < cutoff_time:
                session_ids_to_remove.append(session_id)
        
        for session_id in session_ids_to_remove:
            del self.session_contexts[session_id]
            cleaned_count += 1
        
        # Clean up caches
        cache_keys_to_remove = []
        for cache_key, (timestamp, _) in self.selection_cache.items():
            if timestamp < cutoff_time:
                cache_keys_to_remove.append(cache_key)
        
        for cache_key in cache_keys_to_remove:
            del self.selection_cache[cache_key]
            cleaned_count += 1
        
        # Clean up context patterns (keep frequently used ones)
        for context_key in list(self.context_patterns.keys()):
            total_uses = sum(self.context_patterns[context_key].values())
            if total_uses < 3:  # Remove patterns with very few uses
                del self.context_patterns[context_key]
                cleaned_count += 1
        
        logger.info(
            "Cleaned up old context data",
            cleaned_count=cleaned_count,
            retention_hours=retention_hours,
            active_sessions=len(self.session_contexts)
        )
        
        return cleaned_count