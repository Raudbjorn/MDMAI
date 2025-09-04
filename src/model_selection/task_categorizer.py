"""Task-based categorization system for TTRPG operations."""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Set, Tuple
import re
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)


class TTRPGTaskType(Enum):
    """Categories of TTRPG tasks for model selection optimization."""
    
    # Core rule operations
    RULE_LOOKUP = "rule_lookup"
    RULE_CLARIFICATION = "rule_clarification"
    RULE_INTERPRETATION = "rule_interpretation"
    
    # Character and NPC operations
    CHARACTER_GENERATION = "character_generation"
    CHARACTER_OPTIMIZATION = "character_optimization"
    NPC_GENERATION = "npc_generation"
    NPC_PERSONALITY = "npc_personality"
    
    # Narrative and story operations
    STORY_GENERATION = "story_generation"
    WORLD_BUILDING = "world_building"
    DESCRIPTION_GENERATION = "description_generation"
    DIALOGUE_GENERATION = "dialogue_generation"
    
    # Combat and mechanics
    COMBAT_RESOLUTION = "combat_resolution"
    DAMAGE_CALCULATION = "damage_calculation"
    INITIATIVE_TRACKING = "initiative_tracking"
    
    # Campaign management
    SESSION_PLANNING = "session_planning"
    SESSION_SUMMARIZATION = "session_summarization"
    CAMPAIGN_CONTINUITY = "campaign_continuity"
    
    # Creative and improvisational
    IMPROVISATION = "improvisation"
    CREATIVE_PROBLEM_SOLVING = "creative_problem_solving"
    PLOT_HOOK_GENERATION = "plot_hook_generation"
    
    # Analysis and search
    CONTENT_SEARCH = "content_search"
    CROSS_REFERENCE = "cross_reference"
    LORE_ANALYSIS = "lore_analysis"


class TaskComplexity(Enum):
    """Task complexity levels affecting model selection."""
    
    SIMPLE = "simple"          # Basic lookups, simple calculations
    MODERATE = "moderate"      # Standard generation, rule interpretation
    COMPLEX = "complex"        # Multi-step reasoning, complex narratives
    CREATIVE = "creative"      # High creativity, improvisational content


class TaskLatencyRequirement(IntEnum):
    """Latency requirements for different task types (in milliseconds)."""
    
    IMMEDIATE = 500     # <500ms - Combat, initiative
    FAST = 2000         # <2s - Rule lookups, simple queries
    STANDARD = 5000     # <5s - Character generation, descriptions
    RELAXED = 15000     # <15s - Complex narratives, world building


@dataclass
class TaskCharacteristics:
    """Characteristics of a TTRPG task that influence model selection."""
    
    task_type: TTRPGTaskType
    complexity: TaskComplexity
    latency_requirement: TaskLatencyRequirement
    requires_creativity: bool = False
    requires_accuracy: bool = True
    requires_consistency: bool = True
    context_length_needed: int = 4096
    typical_output_length: int = 500
    
    # Model capability requirements
    needs_tool_calling: bool = False
    needs_structured_output: bool = False
    needs_multi_step_reasoning: bool = False
    
    # Domain-specific requirements
    genre_sensitivity: bool = True      # Needs genre-appropriate responses
    personality_awareness: bool = False  # Needs personality matching
    rules_knowledge: bool = True        # Needs rules system knowledge
    
    # Cost and performance preferences
    cost_sensitivity: float = 0.5       # 0.0 = cost no object, 1.0 = minimize cost
    quality_importance: float = 0.8     # 0.0 = acceptable quality, 1.0 = highest quality


class TaskCategorizer:
    """Categorizes TTRPG requests into task types for optimal model selection."""
    
    def __init__(self):
        self.task_patterns = self._build_task_patterns()
        self.task_characteristics = self._build_task_characteristics()
        self.keyword_weights = self._build_keyword_weights()
        
    def _build_task_patterns(self) -> Dict[TTRPGTaskType, List[re.Pattern]]:
        """Build regex patterns for task detection."""
        return {
            TTRPGTaskType.RULE_LOOKUP: [
                re.compile(r'\b(?:what|how|rule|mechanic|stat|roll|check)\b.*\b(?:for|when|if)\b', re.I),
                re.compile(r'\blookup?\b.*\b(?:rule|stat|ability|spell|item)\b', re.I),
                re.compile(r'\b(?:find|search|get)\b.*\b(?:rule|mechanic|ability)\b', re.I),
            ],
            
            TTRPGTaskType.RULE_CLARIFICATION: [
                re.compile(r'\b(?:clarify|explain|interpret|mean)\b.*\brule\b', re.I),
                re.compile(r'\b(?:how does|what happens when|does this mean)\b', re.I),
                re.compile(r'\bconfused about\b|\bdoes this apply\b|\bcan I\b', re.I),
            ],
            
            TTRPGTaskType.CHARACTER_GENERATION: [
                re.compile(r'\b(?:generate|create|make|build)\b.*\b(?:character|pc|player)\b', re.I),
                re.compile(r'\bcharacter\b.*\b(?:sheet|stats|build|creation)\b', re.I),
                re.compile(r'\b(?:roll|assign|calculate)\b.*\b(?:stats|abilities|attributes)\b', re.I),
            ],
            
            TTRPGTaskType.NPC_GENERATION: [
                re.compile(r'\b(?:generate|create|make|build)\b.*\bnpc\b', re.I),
                re.compile(r'\bnpc\b.*\b(?:stats|personality|background|role)\b', re.I),
                re.compile(r'\b(?:villager|merchant|guard|noble|innkeeper)\b', re.I),
            ],
            
            TTRPGTaskType.STORY_GENERATION: [
                re.compile(r'\b(?:generate|create|write|tell)\b.*\b(?:story|tale|narrative|plot)\b', re.I),
                re.compile(r'\bstory\b.*\b(?:arc|hook|idea|concept)\b', re.I),
                re.compile(r'\b(?:adventure|quest|mission|scenario)\b.*\b(?:idea|concept|outline)\b', re.I),
            ],
            
            TTRPGTaskType.WORLD_BUILDING: [
                re.compile(r'\b(?:generate|create|design|describe)\b.*\b(?:world|setting|location|city|dungeon)\b', re.I),
                re.compile(r'\b(?:town|city|village|castle|temple|tavern)\b.*\b(?:layout|description|details)\b', re.I),
                re.compile(r'\bworld\b.*\b(?:building|creation|design)\b', re.I),
            ],
            
            TTRPGTaskType.DESCRIPTION_GENERATION: [
                re.compile(r'\b(?:describe|detail|paint|set)\b.*\b(?:scene|environment|atmosphere|mood)\b', re.I),
                re.compile(r'\bdescription\b.*\b(?:of|for)\b.*\b(?:room|area|character|item)\b', re.I),
                re.compile(r'\b(?:vivid|atmospheric|immersive)\b.*\bdescription\b', re.I),
            ],
            
            TTRPGTaskType.COMBAT_RESOLUTION: [
                re.compile(r'\b(?:combat|fight|battle|attack|damage|hit)\b', re.I),
                re.compile(r'\b(?:initiative|turn|action|reaction)\b.*\b(?:order|sequence)\b', re.I),
                re.compile(r'\b(?:calculate|determine|resolve)\b.*\b(?:damage|hit|attack)\b', re.I),
            ],
            
            TTRPGTaskType.SESSION_SUMMARIZATION: [
                re.compile(r'\b(?:summarize|recap|review)\b.*\b(?:session|game|adventure)\b', re.I),
                re.compile(r'\bsession\b.*\b(?:notes|summary|recap|highlights)\b', re.I),
                re.compile(r'\b(?:what happened|key events|important moments)\b', re.I),
            ],
            
            TTRPGTaskType.IMPROVISATION: [
                re.compile(r'\b(?:improv|improvise|on the spot|quick|random)\b', re.I),
                re.compile(r'\b(?:unexpected|surprise|spontaneous)\b.*\b(?:event|encounter|twist)\b', re.I),
                re.compile(r'\bneed something\b.*\b(?:quick|fast|now)\b', re.I),
            ],
        }
    
    def _build_task_characteristics(self) -> Dict[TTRPGTaskType, TaskCharacteristics]:
        """Build characteristics for each task type."""
        return {
            TTRPGTaskType.RULE_LOOKUP: TaskCharacteristics(
                task_type=TTRPGTaskType.RULE_LOOKUP,
                complexity=TaskComplexity.SIMPLE,
                latency_requirement=TaskLatencyRequirement.FAST,
                requires_creativity=False,
                requires_accuracy=True,
                requires_consistency=True,
                context_length_needed=2048,
                typical_output_length=200,
                needs_tool_calling=True,
                rules_knowledge=True,
                cost_sensitivity=0.3,
                quality_importance=0.9,
            ),
            
            TTRPGTaskType.RULE_CLARIFICATION: TaskCharacteristics(
                task_type=TTRPGTaskType.RULE_CLARIFICATION,
                complexity=TaskComplexity.MODERATE,
                latency_requirement=TaskLatencyRequirement.FAST,
                requires_creativity=False,
                requires_accuracy=True,
                requires_consistency=True,
                context_length_needed=4096,
                typical_output_length=400,
                needs_multi_step_reasoning=True,
                rules_knowledge=True,
                cost_sensitivity=0.4,
                quality_importance=0.85,
            ),
            
            TTRPGTaskType.CHARACTER_GENERATION: TaskCharacteristics(
                task_type=TTRPGTaskType.CHARACTER_GENERATION,
                complexity=TaskComplexity.MODERATE,
                latency_requirement=TaskLatencyRequirement.STANDARD,
                requires_creativity=True,
                requires_accuracy=True,
                requires_consistency=True,
                context_length_needed=4096,
                typical_output_length=800,
                needs_structured_output=True,
                needs_tool_calling=True,
                genre_sensitivity=True,
                rules_knowledge=True,
                cost_sensitivity=0.5,
                quality_importance=0.8,
            ),
            
            TTRPGTaskType.NPC_GENERATION: TaskCharacteristics(
                task_type=TTRPGTaskType.NPC_GENERATION,
                complexity=TaskComplexity.MODERATE,
                latency_requirement=TaskLatencyRequirement.STANDARD,
                requires_creativity=True,
                requires_accuracy=True,
                requires_consistency=False,
                context_length_needed=4096,
                typical_output_length=600,
                needs_structured_output=True,
                personality_awareness=True,
                genre_sensitivity=True,
                cost_sensitivity=0.6,
                quality_importance=0.75,
            ),
            
            TTRPGTaskType.STORY_GENERATION: TaskCharacteristics(
                task_type=TTRPGTaskType.STORY_GENERATION,
                complexity=TaskComplexity.COMPLEX,
                latency_requirement=TaskLatencyRequirement.RELAXED,
                requires_creativity=True,
                requires_accuracy=False,
                requires_consistency=True,
                context_length_needed=8192,
                typical_output_length=1500,
                needs_multi_step_reasoning=True,
                personality_awareness=True,
                genre_sensitivity=True,
                cost_sensitivity=0.7,
                quality_importance=0.9,
            ),
            
            TTRPGTaskType.WORLD_BUILDING: TaskCharacteristics(
                task_type=TTRPGTaskType.WORLD_BUILDING,
                complexity=TaskComplexity.COMPLEX,
                latency_requirement=TaskLatencyRequirement.RELAXED,
                requires_creativity=True,
                requires_accuracy=False,
                requires_consistency=True,
                context_length_needed=8192,
                typical_output_length=1200,
                needs_multi_step_reasoning=True,
                genre_sensitivity=True,
                cost_sensitivity=0.7,
                quality_importance=0.85,
            ),
            
            TTRPGTaskType.DESCRIPTION_GENERATION: TaskCharacteristics(
                task_type=TTRPGTaskType.DESCRIPTION_GENERATION,
                complexity=TaskComplexity.CREATIVE,
                latency_requirement=TaskLatencyRequirement.STANDARD,
                requires_creativity=True,
                requires_accuracy=False,
                requires_consistency=False,
                context_length_needed=4096,
                typical_output_length=400,
                personality_awareness=True,
                genre_sensitivity=True,
                cost_sensitivity=0.6,
                quality_importance=0.75,
            ),
            
            TTRPGTaskType.COMBAT_RESOLUTION: TaskCharacteristics(
                task_type=TTRPGTaskType.COMBAT_RESOLUTION,
                complexity=TaskComplexity.MODERATE,
                latency_requirement=TaskLatencyRequirement.IMMEDIATE,
                requires_creativity=False,
                requires_accuracy=True,
                requires_consistency=True,
                context_length_needed=2048,
                typical_output_length=300,
                needs_tool_calling=True,
                needs_multi_step_reasoning=True,
                rules_knowledge=True,
                cost_sensitivity=0.2,
                quality_importance=0.9,
            ),
            
            TTRPGTaskType.SESSION_SUMMARIZATION: TaskCharacteristics(
                task_type=TTRPGTaskType.SESSION_SUMMARIZATION,
                complexity=TaskComplexity.COMPLEX,
                latency_requirement=TaskLatencyRequirement.RELAXED,
                requires_creativity=False,
                requires_accuracy=True,
                requires_consistency=True,
                context_length_needed=16384,
                typical_output_length=800,
                needs_multi_step_reasoning=True,
                cost_sensitivity=0.6,
                quality_importance=0.8,
            ),
            
            TTRPGTaskType.IMPROVISATION: TaskCharacteristics(
                task_type=TTRPGTaskType.IMPROVISATION,
                complexity=TaskComplexity.CREATIVE,
                latency_requirement=TaskLatencyRequirement.FAST,
                requires_creativity=True,
                requires_accuracy=False,
                requires_consistency=False,
                context_length_needed=4096,
                typical_output_length=300,
                personality_awareness=True,
                genre_sensitivity=True,
                cost_sensitivity=0.4,
                quality_importance=0.7,
            ),
        }
    
    def _build_keyword_weights(self) -> Dict[str, Dict[TTRPGTaskType, float]]:
        """Build keyword-to-task weights for classification."""
        return {
            # Rule-related keywords
            "rule": {TTRPGTaskType.RULE_LOOKUP: 0.8, TTRPGTaskType.RULE_CLARIFICATION: 0.6},
            "mechanic": {TTRPGTaskType.RULE_LOOKUP: 0.7, TTRPGTaskType.RULE_CLARIFICATION: 0.5},
            "how": {TTRPGTaskType.RULE_CLARIFICATION: 0.6, TTRPGTaskType.RULE_LOOKUP: 0.4},
            "what": {TTRPGTaskType.RULE_LOOKUP: 0.5, TTRPGTaskType.RULE_CLARIFICATION: 0.4},
            
            # Character keywords
            "character": {TTRPGTaskType.CHARACTER_GENERATION: 0.8, TTRPGTaskType.NPC_GENERATION: 0.3},
            "npc": {TTRPGTaskType.NPC_GENERATION: 0.9},
            "generate": {TTRPGTaskType.CHARACTER_GENERATION: 0.6, TTRPGTaskType.NPC_GENERATION: 0.6, TTRPGTaskType.STORY_GENERATION: 0.5},
            "create": {TTRPGTaskType.CHARACTER_GENERATION: 0.5, TTRPGTaskType.NPC_GENERATION: 0.5, TTRPGTaskType.WORLD_BUILDING: 0.4},
            
            # Story keywords
            "story": {TTRPGTaskType.STORY_GENERATION: 0.9},
            "narrative": {TTRPGTaskType.STORY_GENERATION: 0.8},
            "plot": {TTRPGTaskType.STORY_GENERATION: 0.7},
            "adventure": {TTRPGTaskType.STORY_GENERATION: 0.6, TTRPGTaskType.WORLD_BUILDING: 0.3},
            
            # World-building keywords
            "world": {TTRPGTaskType.WORLD_BUILDING: 0.9},
            "setting": {TTRPGTaskType.WORLD_BUILDING: 0.8},
            "location": {TTRPGTaskType.WORLD_BUILDING: 0.7, TTRPGTaskType.DESCRIPTION_GENERATION: 0.4},
            "city": {TTRPGTaskType.WORLD_BUILDING: 0.6},
            "dungeon": {TTRPGTaskType.WORLD_BUILDING: 0.7},
            
            # Description keywords
            "describe": {TTRPGTaskType.DESCRIPTION_GENERATION: 0.8},
            "description": {TTRPGTaskType.DESCRIPTION_GENERATION: 0.9},
            "scene": {TTRPGTaskType.DESCRIPTION_GENERATION: 0.7},
            "atmosphere": {TTRPGTaskType.DESCRIPTION_GENERATION: 0.6},
            
            # Combat keywords
            "combat": {TTRPGTaskType.COMBAT_RESOLUTION: 0.9},
            "fight": {TTRPGTaskType.COMBAT_RESOLUTION: 0.8},
            "battle": {TTRPGTaskType.COMBAT_RESOLUTION: 0.8},
            "damage": {TTRPGTaskType.COMBAT_RESOLUTION: 0.7},
            "initiative": {TTRPGTaskType.COMBAT_RESOLUTION: 0.6},
            
            # Session keywords
            "session": {TTRPGTaskType.SESSION_SUMMARIZATION: 0.8},
            "summarize": {TTRPGTaskType.SESSION_SUMMARIZATION: 0.9},
            "recap": {TTRPGTaskType.SESSION_SUMMARIZATION: 0.8},
            
            # Improvisation keywords
            "improv": {TTRPGTaskType.IMPROVISATION: 0.9},
            "random": {TTRPGTaskType.IMPROVISATION: 0.6, TTRPGTaskType.CHARACTER_GENERATION: 0.3},
            "quick": {TTRPGTaskType.IMPROVISATION: 0.5},
            "unexpected": {TTRPGTaskType.IMPROVISATION: 0.7},
        }
    
    def categorize_task(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[TTRPGTaskType, float]:
        """Categorize a user input into a TTRPG task type.
        
        Args:
            user_input: The user's request or query
            context: Optional context information (session state, campaign info, etc.)
            
        Returns:
            Tuple of (task_type, confidence_score)
        """
        scores = {}
        
        # Pattern-based scoring
        for task_type, patterns in self.task_patterns.items():
            pattern_score = 0.0
            for pattern in patterns:
                if pattern.search(user_input):
                    pattern_score += 0.3
            scores[task_type] = pattern_score
        
        # Keyword-based scoring
        words = re.findall(r'\b\w+\b', user_input.lower())
        for word in words:
            if word in self.keyword_weights:
                for task_type, weight in self.keyword_weights[word].items():
                    scores[task_type] = scores.get(task_type, 0) + weight * 0.1
        
        # Context-based adjustments
        if context:
            scores = self._apply_context_adjustments(scores, context)
        
        # Find best match
        if not scores:
            return TTRPGTaskType.RULE_LOOKUP, 0.1  # Default fallback
        
        best_task = max(scores.items(), key=lambda x: x[1])
        confidence = min(best_task[1], 1.0)  # Cap at 1.0
        
        logger.info(
            "Task categorization completed",
            input_length=len(user_input),
            task_type=best_task[0].value,
            confidence=confidence,
            top_scores={k.value: v for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]}
        )
        
        return best_task[0], confidence
    
    def _apply_context_adjustments(
        self,
        scores: Dict[TTRPGTaskType, float],
        context: Dict[str, Any]
    ) -> Dict[TTRPGTaskType, float]:
        """Apply context-based adjustments to task scores."""
        adjusted_scores = scores.copy()
        
        # Session context adjustments
        if context.get("in_combat"):
            adjusted_scores[TTRPGTaskType.COMBAT_RESOLUTION] = adjusted_scores.get(TTRPGTaskType.COMBAT_RESOLUTION, 0) + 0.3
            adjusted_scores[TTRPGTaskType.IMPROVISATION] = adjusted_scores.get(TTRPGTaskType.IMPROVISATION, 0) + 0.2
        
        if context.get("session_active"):
            adjusted_scores[TTRPGTaskType.IMPROVISATION] = adjusted_scores.get(TTRPGTaskType.IMPROVISATION, 0) + 0.1
        
        # Campaign context adjustments
        if context.get("campaign_genre"):
            genre = context["campaign_genre"].lower()
            if "horror" in genre:
                adjusted_scores[TTRPGTaskType.DESCRIPTION_GENERATION] = adjusted_scores.get(TTRPGTaskType.DESCRIPTION_GENERATION, 0) + 0.1
            elif "combat" in genre or "tactical" in genre:
                adjusted_scores[TTRPGTaskType.COMBAT_RESOLUTION] = adjusted_scores.get(TTRPGTaskType.COMBAT_RESOLUTION, 0) + 0.1
        
        # User preference adjustments
        if context.get("user_preferences"):
            prefs = context["user_preferences"]
            if prefs.get("prefers_creative_responses"):
                for creative_task in [TTRPGTaskType.STORY_GENERATION, TTRPGTaskType.DESCRIPTION_GENERATION, TTRPGTaskType.IMPROVISATION]:
                    adjusted_scores[creative_task] = adjusted_scores.get(creative_task, 0) + 0.1
        
        return adjusted_scores
    
    def get_task_characteristics(self, task_type: TTRPGTaskType) -> TaskCharacteristics:
        """Get characteristics for a specific task type."""
        return self.task_characteristics.get(task_type, self.task_characteristics[TTRPGTaskType.RULE_LOOKUP])
    
    def analyze_request_complexity(self, user_input: str) -> TaskComplexity:
        """Analyze the complexity of a request based on content."""
        word_count = len(user_input.split())
        
        # Complex indicators
        complex_indicators = [
            "multi-step", "complex", "detailed", "comprehensive", "elaborate",
            "intricate", "nuanced", "sophisticated", "advanced"
        ]
        
        # Simple indicators  
        simple_indicators = [
            "quick", "simple", "basic", "fast", "just", "only", "brief"
        ]
        
        has_complex = any(indicator in user_input.lower() for indicator in complex_indicators)
        has_simple = any(indicator in user_input.lower() for indicator in simple_indicators)
        
        if has_complex:
            return TaskComplexity.COMPLEX
        elif has_simple:
            return TaskComplexity.SIMPLE
        elif word_count > 50:
            return TaskComplexity.COMPLEX
        elif word_count > 20:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE