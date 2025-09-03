"""
Model-Specific Router with Advanced Capability Matching
Task 25.3: Develop Provider Router with Fallback
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from structlog import get_logger
from .config.config_loader import get_default_config_loader, ConfigurationError

from .models import (
    AIRequest,
    ProviderType,
    ProviderCapability,
    ModelSpec,
    CostTier,
)
from .abstract_provider import AbstractProvider

logger = get_logger(__name__)


class ModelCategory(Enum):
    """Categories of AI models based on their primary use case."""
    
    CONVERSATIONAL = "conversational"
    CODE_GENERATION = "code_generation"
    CREATIVE_WRITING = "creative_writing"
    ANALYSIS_REASONING = "analysis_reasoning"
    MULTIMODAL = "multimodal"
    FUNCTION_CALLING = "function_calling"
    SPEED_OPTIMIZED = "speed_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"


class ModelTier(Enum):
    """Performance tiers for models."""
    
    FLAGSHIP = "flagship"      # Best quality, highest cost
    BALANCED = "balanced"      # Good quality/cost ratio
    EFFICIENT = "efficient"    # Fast and cost-effective
    SPECIALIZED = "specialized" # Specific use cases


@dataclass
class ModelProfile:
    """Detailed profile of a model's capabilities and characteristics."""
    
    model_id: str
    provider_type: ProviderType
    category: ModelCategory
    tier: ModelTier
    
    # Core specifications
    context_length: int
    max_output_tokens: int
    supports_streaming: bool = False
    supports_tools: bool = False
    supports_vision: bool = False
    supports_json_mode: bool = False
    
    # Performance characteristics
    avg_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    quality_score: float = 0.0  # 0.0 to 1.0
    
    # Cost information
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    cost_tier: CostTier = CostTier.MEDIUM
    
    # Capability scores (0.0 to 1.0)
    reasoning_capability: float = 0.5
    creativity_capability: float = 0.5
    coding_capability: float = 0.5
    multimodal_capability: float = 0.0
    tool_use_capability: float = 0.0
    
    # Usage patterns and recommendations
    recommended_for: List[str] = field(default_factory=list)
    not_recommended_for: List[str] = field(default_factory=list)
    
    # Context and constraints
    optimal_context_usage: float = 0.7  # Fraction of context length for optimal performance
    min_tokens_for_quality: int = 100
    
    # Metadata
    release_date: Optional[datetime] = None
    deprecation_date: Optional[datetime] = None
    is_preview: bool = False


@dataclass
class RoutingRule:
    """Rule for model selection based on request characteristics."""
    
    name: str
    description: str
    priority: int  # Higher priority rules are evaluated first
    
    # Matching criteria
    request_patterns: List[str] = field(default_factory=list)  # Regex patterns
    required_capabilities: List[ProviderCapability] = field(default_factory=list)
    preferred_categories: List[ModelCategory] = field(default_factory=list)
    preferred_tiers: List[ModelTier] = field(default_factory=list)
    
    # Constraints
    max_cost_per_request: Optional[float] = None
    max_latency_ms: Optional[float] = None
    min_quality_score: Optional[float] = None
    
    # Context requirements
    min_context_length: Optional[int] = None
    context_utilization_threshold: float = 0.8
    
    # Output requirements
    min_output_tokens: Optional[int] = None
    requires_json_output: bool = False
    
    enabled: bool = True


class ModelRouter:
    """
    Advanced model-specific router with capability matching.
    
    Features:
    - Model categorization and profiling
    - Intelligent capability matching
    - Context-aware routing
    - Performance-based selection
    - Cost optimization per model
    - Quality requirements matching
    - Tool compatibility validation
    """
    
    def __init__(self, config_loader=None):
        self.model_profiles: Dict[str, ModelProfile] = {}
        self.provider_model_mapping: Dict[ProviderType, List[str]] = {}
        self.routing_rules: List[RoutingRule] = []
        
        # Model compatibility matrix
        self.tool_compatibility: Dict[str, Set[str]] = {}  # model_id -> set of compatible tool types
        
        # Performance tracking
        self.model_performance_history: Dict[str, List[Tuple[datetime, float, float]]] = {}
        
        # Configuration loader for external configs
        self.config_loader = config_loader or get_default_config_loader()
        
        # Initialize with external configuration first, fallback to defaults
        self._initialize_from_external_config()
    
    def register_model(
        self,
        provider_type: ProviderType,
        model_spec: ModelSpec,
        profile: Optional[ModelProfile] = None,
    ) -> None:
        """Register a model with the router."""
        model_id = model_spec.model_id
        
        if profile:
            self.model_profiles[model_id] = profile
        else:
            # Create default profile from model spec
            self.model_profiles[model_id] = self._create_profile_from_spec(model_spec)
        
        # Update provider mapping
        if provider_type not in self.provider_model_mapping:
            self.provider_model_mapping[provider_type] = []
        
        if model_id not in self.provider_model_mapping[provider_type]:
            self.provider_model_mapping[provider_type].append(model_id)
        
        logger.info(
            "Registered model with router",
            model=model_id,
            provider=provider_type.value,
            category=self.model_profiles[model_id].category.value,
        )
    
    async def select_optimal_model(
        self,
        request: AIRequest,
        available_providers: List[AbstractProvider],
        preferences: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[AbstractProvider, str]]:
        """
        Select optimal model based on request characteristics and preferences.
        
        Args:
            request: The AI request
            available_providers: List of available providers
            preferences: Optional preferences dict
            
        Returns:
            Tuple of (provider, model_id) or None
        """
        logger.debug(
            "Starting model selection",
            request_id=request.request_id,
            requested_model=request.model,
            providers=len(available_providers),
        )
        
        # If specific model requested and available, validate and return
        if request.model:
            for provider in available_providers:
                if request.model in provider.models:
                    if self._validate_model_compatibility(request, request.model):
                        return provider, request.model
                    else:
                        logger.warning(
                            "Requested model not compatible with request requirements",
                            model=request.model,
                            request_id=request.request_id,
                        )
        
        # Analyze request characteristics
        request_analysis = self._analyze_request(request)
        
        # Find candidate models
        candidates = self._find_candidate_models(request, available_providers, request_analysis)
        
        if not candidates:
            logger.warning("No candidate models found")
            return None
        
        # Apply routing rules
        filtered_candidates = self._apply_routing_rules(request, candidates, request_analysis)
        
        # Score and rank candidates
        scored_candidates = self._score_model_candidates(
            request, filtered_candidates, request_analysis, preferences
        )
        
        if not scored_candidates:
            logger.warning("No scored candidates found")
            return None
        
        # Select best candidate
        best_candidate = scored_candidates[0]  # Already sorted by score
        provider, model_id, score = best_candidate
        
        logger.info(
            "Selected optimal model",
            provider=provider.provider_type.value,
            model=model_id,
            score=score,
            request_id=request.request_id,
        )
        
        return provider, model_id
    
    def _analyze_request(self, request: AIRequest) -> Dict[str, Any]:
        """Analyze request to determine characteristics and requirements."""
        analysis = {
            "estimated_input_tokens": 0,
            "requires_tools": bool(request.tools),
            "requires_streaming": request.stream,
            "requires_vision": False,
            "content_type": "text",
            "complexity_level": "medium",
            "task_type": "general",
            "quality_priority": False,
            "speed_priority": False,
            "cost_priority": False,
        }
        
        # Analyze message content
        total_content = ""
        for message in request.messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total_content += content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            total_content += item.get("text", "")
                        elif item.get("type") in ["image", "image_url"]:
                            analysis["requires_vision"] = True
                            analysis["content_type"] = "multimodal"
        
        # Estimate input tokens
        analysis["estimated_input_tokens"] = len(total_content) // 4
        
        # Detect task type from content patterns
        content_lower = total_content.lower()
        
        if any(keyword in content_lower for keyword in ["code", "function", "class", "python", "javascript", "programming"]):
            analysis["task_type"] = "coding"
            analysis["complexity_level"] = "high"
        
        elif any(keyword in content_lower for keyword in ["analyze", "reason", "explain", "logic", "problem"]):
            analysis["task_type"] = "reasoning"
            analysis["complexity_level"] = "high"
            analysis["quality_priority"] = True
        
        elif any(keyword in content_lower for keyword in ["create", "write", "story", "poem", "creative"]):
            analysis["task_type"] = "creative"
            analysis["quality_priority"] = True
        
        elif any(keyword in content_lower for keyword in ["quick", "fast", "simple", "brief"]):
            analysis["speed_priority"] = True
            analysis["complexity_level"] = "low"
        
        # Check for JSON output requirements
        if "json" in content_lower or "format" in content_lower:
            analysis["requires_json"] = True
        
        # Analyze context length requirements
        if analysis["estimated_input_tokens"] > 8000:
            analysis["requires_long_context"] = True
        
        return analysis
    
    def _find_candidate_models(
        self,
        request: AIRequest,
        available_providers: List[AbstractProvider],
        analysis: Dict[str, Any],
    ) -> List[Tuple[AbstractProvider, str]]:
        """Find all models that could potentially handle the request."""
        candidates = []
        
        for provider in available_providers:
            for model_id, model_spec in provider.models.items():
                if not model_spec.is_available:
                    continue
                
                # Check basic compatibility
                if not self._check_basic_compatibility(model_spec, analysis):
                    continue
                
                # Check if model is registered in our profiles
                if model_id not in self.model_profiles:
                    # Create basic profile for unknown model
                    self.register_model(provider.provider_type, model_spec)
                
                candidates.append((provider, model_id))
        
        logger.debug(f"Found {len(candidates)} candidate models")
        return candidates
    
    def _check_basic_compatibility(self, model_spec: ModelSpec, analysis: Dict[str, Any]) -> bool:
        """Check basic compatibility requirements."""
        # Context length check
        if analysis["estimated_input_tokens"] > model_spec.context_length:
            return False
        
        # Capability checks
        if analysis["requires_tools"] and not model_spec.supports_tools:
            return False
        
        if analysis["requires_streaming"] and not model_spec.supports_streaming:
            return False
        
        if analysis["requires_vision"] and not model_spec.supports_vision:
            return False
        
        return True
    
    def _apply_routing_rules(
        self,
        request: AIRequest,
        candidates: List[Tuple[AbstractProvider, str]],
        analysis: Dict[str, Any],
    ) -> List[Tuple[AbstractProvider, str]]:
        """Apply routing rules to filter candidates."""
        # Sort rules by priority (descending)
        active_rules = [rule for rule in self.routing_rules if rule.enabled]
        active_rules.sort(key=lambda r: r.priority, reverse=True)
        
        filtered_candidates = candidates.copy()
        
        for rule in active_rules:
            if not self._rule_matches_request(rule, request, analysis):
                continue
            
            # Apply rule filtering
            rule_candidates = []
            for provider, model_id in filtered_candidates:
                if self._model_matches_rule(model_id, rule):
                    rule_candidates.append((provider, model_id))
            
            if rule_candidates:  # Only apply if rule leaves some candidates
                filtered_candidates = rule_candidates
                logger.debug(
                    "Applied routing rule",
                    rule=rule.name,
                    candidates_before=len(filtered_candidates),
                    candidates_after=len(rule_candidates),
                )
        
        return filtered_candidates
    
    def _rule_matches_request(
        self, rule: RoutingRule, request: AIRequest, analysis: Dict[str, Any]
    ) -> bool:
        """Check if a routing rule matches the request."""
        # Check request patterns
        if rule.request_patterns:
            content = " ".join(str(msg.get("content", "")) for msg in request.messages)
            if not any(re.search(pattern, content, re.IGNORECASE) for pattern in rule.request_patterns):
                return False
        
        # Check required capabilities
        for capability in rule.required_capabilities:
            if capability == ProviderCapability.TOOL_CALLING and not analysis["requires_tools"]:
                return False
            elif capability == ProviderCapability.STREAMING and not analysis["requires_streaming"]:
                return False
            elif capability == ProviderCapability.VISION and not analysis["requires_vision"]:
                return False
        
        return True
    
    def _model_matches_rule(self, model_id: str, rule: RoutingRule) -> bool:
        """Check if a model matches a routing rule."""
        if model_id not in self.model_profiles:
            return False
        
        profile = self.model_profiles[model_id]
        
        # Check preferred categories
        if rule.preferred_categories and profile.category not in rule.preferred_categories:
            return False
        
        # Check preferred tiers
        if rule.preferred_tiers and profile.tier not in rule.preferred_tiers:
            return False
        
        # Check constraints
        if rule.max_cost_per_request:
            estimated_cost = self._estimate_model_cost(profile, 1000, 1000)  # Sample calculation
            if estimated_cost > rule.max_cost_per_request:
                return False
        
        if rule.max_latency_ms and profile.avg_latency_ms > rule.max_latency_ms:
            return False
        
        if rule.min_quality_score and profile.quality_score < rule.min_quality_score:
            return False
        
        if rule.min_context_length and profile.context_length < rule.min_context_length:
            return False
        
        return True
    
    def _score_model_candidates(
        self,
        request: AIRequest,
        candidates: List[Tuple[AbstractProvider, str]],
        analysis: Dict[str, Any],
        preferences: Optional[Dict[str, Any]],
    ) -> List[Tuple[AbstractProvider, str, float]]:
        """Score and rank model candidates."""
        scored_candidates = []
        
        for provider, model_id in candidates:
            if model_id not in self.model_profiles:
                continue
            
            profile = self.model_profiles[model_id]
            score = self._calculate_model_score(profile, request, analysis, preferences)
            
            scored_candidates.append((provider, model_id, score))
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        
        return scored_candidates
    
    def _calculate_model_score(
        self,
        profile: ModelProfile,
        request: AIRequest,
        analysis: Dict[str, Any],
        preferences: Optional[Dict[str, Any]],
    ) -> float:
        """Calculate composite score for a model."""
        score = 0.0
        
        # Base capability score
        if analysis["task_type"] == "coding":
            score += profile.coding_capability * 0.4
        elif analysis["task_type"] == "reasoning":
            score += profile.reasoning_capability * 0.4
        elif analysis["task_type"] == "creative":
            score += profile.creativity_capability * 0.4
        else:
            score += profile.quality_score * 0.3
        
        # Multimodal bonus
        if analysis["requires_vision"] and profile.supports_vision:
            score += profile.multimodal_capability * 0.3
        
        # Tool use bonus
        if analysis["requires_tools"] and profile.supports_tools:
            score += profile.tool_use_capability * 0.2
        
        # Performance factors
        if analysis["speed_priority"]:
            if profile.tokens_per_second > 0:
                speed_score = min(1.0, profile.tokens_per_second / 100.0)  # Normalize
                score += speed_score * 0.3
        
        if analysis["quality_priority"]:
            score += profile.quality_score * 0.3
        
        # Cost efficiency (lower cost = higher score)
        if analysis["cost_priority"]:
            estimated_cost = self._estimate_model_cost(profile, 
                                                     analysis["estimated_input_tokens"],
                                                     request.max_tokens or 1000)
            if estimated_cost > 0:
                cost_score = max(0.0, 1.0 - (estimated_cost / 1.0))  # Normalize to $1
                score += cost_score * 0.2
        
        # Context utilization efficiency
        context_usage = analysis["estimated_input_tokens"] / profile.context_length
        if context_usage <= profile.optimal_context_usage:
            score += 0.1  # Bonus for efficient context usage
        elif context_usage > 0.9:
            score -= 0.2  # Penalty for near-limit usage
        
        # Tier bonuses
        tier_bonuses = {
            ModelTier.FLAGSHIP: 0.1,
            ModelTier.BALANCED: 0.05,
            ModelTier.EFFICIENT: 0.0,
            ModelTier.SPECIALIZED: 0.15,  # Bonus if it's specialized for the task
        }
        score += tier_bonuses.get(profile.tier, 0.0)
        
        # Apply preferences if provided
        if preferences:
            if preferences.get("prefer_speed") and profile.tokens_per_second > 50:
                score += 0.2
            if preferences.get("prefer_quality") and profile.quality_score > 0.8:
                score += 0.2
            if preferences.get("prefer_cost") and profile.cost_tier in [CostTier.FREE, CostTier.LOW]:
                score += 0.2
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def _validate_model_compatibility(self, request: AIRequest, model_id: str) -> bool:
        """Validate that a specific model is compatible with the request."""
        if model_id not in self.model_profiles:
            return True  # Unknown model, assume compatible
        
        profile = self.model_profiles[model_id]
        analysis = self._analyze_request(request)
        
        # Basic checks
        if analysis["estimated_input_tokens"] > profile.context_length:
            return False
        
        if analysis["requires_tools"] and not profile.supports_tools:
            return False
        
        if analysis["requires_streaming"] and not profile.supports_streaming:
            return False
        
        if analysis["requires_vision"] and not profile.supports_vision:
            return False
        
        return True
    
    def _estimate_model_cost(self, profile: ModelProfile, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for using a model."""
        input_cost = (input_tokens / 1000) * profile.cost_per_input_token
        output_cost = (output_tokens / 1000) * profile.cost_per_output_token
        return input_cost + output_cost
    
    def _create_profile_from_spec(self, model_spec: ModelSpec) -> ModelProfile:
        """Create a basic model profile from a model spec."""
        # Infer category from model name
        model_name = model_spec.model_id.lower()
        category = ModelCategory.CONVERSATIONAL  # Default
        
        if "code" in model_name or "codex" in model_name:
            category = ModelCategory.CODE_GENERATION
        elif "vision" in model_name or "multimodal" in model_name:
            category = ModelCategory.MULTIMODAL
        elif "fast" in model_name or "turbo" in model_name:
            category = ModelCategory.SPEED_OPTIMIZED
        elif "pro" in model_name or "opus" in model_name:
            category = ModelCategory.QUALITY_OPTIMIZED
        
        # Infer tier from cost
        tier = ModelTier.BALANCED
        if model_spec.cost_tier == CostTier.PREMIUM:
            tier = ModelTier.FLAGSHIP
        elif model_spec.cost_tier in [CostTier.FREE, CostTier.LOW]:
            tier = ModelTier.EFFICIENT
        
        return ModelProfile(
            model_id=model_spec.model_id,
            provider_type=model_spec.provider_type,
            category=category,
            tier=tier,
            context_length=model_spec.context_length,
            max_output_tokens=model_spec.max_output_tokens,
            supports_streaming=model_spec.supports_streaming,
            supports_tools=model_spec.supports_tools,
            supports_vision=model_spec.supports_vision,
            cost_per_input_token=model_spec.cost_per_input_token,
            cost_per_output_token=model_spec.cost_per_output_token,
            cost_tier=model_spec.cost_tier,
        )
    
    def _initialize_default_profiles(self) -> None:
        """Initialize default model profiles for known models."""
        # Claude models
        claude_profiles = [
            ModelProfile(
                model_id="claude-3-opus",
                provider_type=ProviderType.ANTHROPIC,
                category=ModelCategory.QUALITY_OPTIMIZED,
                tier=ModelTier.FLAGSHIP,
                context_length=200000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
                quality_score=0.95,
                reasoning_capability=0.95,
                creativity_capability=0.90,
                coding_capability=0.90,
                multimodal_capability=0.85,
                tool_use_capability=0.90,
                cost_tier=CostTier.PREMIUM,
                recommended_for=["complex reasoning", "creative writing", "code analysis", "vision tasks"],
            ),
            ModelProfile(
                model_id="claude-3-sonnet",
                provider_type=ProviderType.ANTHROPIC,
                category=ModelCategory.BALANCED,
                tier=ModelTier.BALANCED,
                context_length=200000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
                quality_score=0.85,
                reasoning_capability=0.85,
                creativity_capability=0.80,
                coding_capability=0.85,
                multimodal_capability=0.80,
                tool_use_capability=0.85,
                cost_tier=CostTier.MEDIUM,
                recommended_for=["general tasks", "balanced performance", "most use cases"],
            ),
            ModelProfile(
                model_id="claude-3-haiku",
                provider_type=ProviderType.ANTHROPIC,
                category=ModelCategory.SPEED_OPTIMIZED,
                tier=ModelTier.EFFICIENT,
                context_length=200000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
                quality_score=0.75,
                tokens_per_second=150,
                reasoning_capability=0.75,
                creativity_capability=0.70,
                coding_capability=0.75,
                multimodal_capability=0.70,
                tool_use_capability=0.80,
                cost_tier=CostTier.LOW,
                recommended_for=["quick responses", "simple tasks", "cost-sensitive applications"],
            ),
        ]
        
        # GPT models
        gpt_profiles = [
            ModelProfile(
                model_id="gpt-4",
                provider_type=ProviderType.OPENAI,
                category=ModelCategory.QUALITY_OPTIMIZED,
                tier=ModelTier.FLAGSHIP,
                context_length=8192,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_tools=True,
                quality_score=0.90,
                reasoning_capability=0.90,
                creativity_capability=0.85,
                coding_capability=0.90,
                tool_use_capability=0.85,
                cost_tier=CostTier.HIGH,
                recommended_for=["complex reasoning", "code generation", "analysis"],
            ),
            ModelProfile(
                model_id="gpt-3.5-turbo",
                provider_type=ProviderType.OPENAI,
                category=ModelCategory.SPEED_OPTIMIZED,
                tier=ModelTier.EFFICIENT,
                context_length=4096,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_tools=True,
                quality_score=0.75,
                tokens_per_second=100,
                reasoning_capability=0.70,
                creativity_capability=0.75,
                coding_capability=0.80,
                tool_use_capability=0.75,
                cost_tier=CostTier.LOW,
                recommended_for=["general chat", "quick tasks", "cost-effective solutions"],
            ),
        ]
        
        # Register all profiles
        for profile in claude_profiles + gpt_profiles:
            self.model_profiles[profile.model_id] = profile
    
    def _initialize_default_routing_rules(self) -> None:
        """Initialize default routing rules."""
        self.routing_rules = [
            RoutingRule(
                name="coding_tasks",
                description="Route coding tasks to code-optimized models",
                priority=100,
                request_patterns=[
                    r"(write|create|generate).*(code|function|class|script)",
                    r"(python|javascript|java|c\+\+|rust|go)\s+(code|function)",
                    r"(debug|fix|optimize).*(code|bug|error)",
                ],
                preferred_categories=[ModelCategory.CODE_GENERATION, ModelCategory.QUALITY_OPTIMIZED],
                preferred_tiers=[ModelTier.FLAGSHIP, ModelTier.BALANCED],
                min_quality_score=0.8,
            ),
            RoutingRule(
                name="vision_tasks",
                description="Route vision tasks to multimodal models",
                priority=95,
                required_capabilities=[ProviderCapability.VISION],
                preferred_categories=[ModelCategory.MULTIMODAL],
                min_quality_score=0.7,
            ),
            RoutingRule(
                name="quick_responses",
                description="Route simple tasks to fast models",
                priority=70,
                request_patterns=[
                    r"(quick|fast|brief|short).*(answer|response|reply)",
                    r"(simple|easy|basic).*(question|task)",
                ],
                preferred_categories=[ModelCategory.SPEED_OPTIMIZED],
                preferred_tiers=[ModelTier.EFFICIENT],
                max_latency_ms=5000,
            ),
            RoutingRule(
                name="tool_calling",
                description="Route function calling to tool-capable models",
                priority=90,
                required_capabilities=[ProviderCapability.TOOL_CALLING],
                preferred_categories=[ModelCategory.FUNCTION_CALLING],
                min_quality_score=0.75,
            ),
            RoutingRule(
                name="creative_writing",
                description="Route creative tasks to creative-optimized models",
                priority=80,
                request_patterns=[
                    r"(write|create|compose).*(story|poem|article|creative)",
                    r"(creative|artistic|imaginative)",
                ],
                preferred_categories=[ModelCategory.CREATIVE_WRITING, ModelCategory.QUALITY_OPTIMIZED],
                min_quality_score=0.8,
            ),
        ]
    
    def _initialize_from_external_config(self) -> None:
        """Initialize model profiles and routing rules from external configuration files."""
        try:
            # Load external configuration
            profiles_config = self.config_loader.load_model_profiles()
            rules_config = self.config_loader.load_routing_rules()
            
            # Initialize profiles from config
            self._load_profiles_from_config(profiles_config)
            
            # Initialize routing rules from config  
            self._load_routing_rules_from_config(rules_config)
            
            logger.info("Successfully initialized from external configuration",
                       profile_count=len(self.model_profiles),
                       rule_count=len(self.routing_rules))
            
        except ConfigurationError as e:
            logger.warning("Failed to load external configuration, falling back to defaults",
                          error=str(e))
            # Fallback to hardcoded defaults
            self._initialize_default_profiles()
            self._initialize_default_routing_rules()
        except Exception as e:
            logger.error("Unexpected error loading configuration, using defaults",
                        error=str(e))
            # Fallback to hardcoded defaults
            self._initialize_default_profiles()
            self._initialize_default_routing_rules()
    
    def _load_profiles_from_config(self, profiles_config: Dict[str, Any]) -> None:
        """Load model profiles from configuration dictionary."""
        for model_id, profile_data in profiles_config.items():
            try:
                # Map string values to enums
                provider_type = ProviderType(profile_data['provider_type'].upper())
                category = ModelCategory(profile_data['category'].upper())
                tier = ModelTier(profile_data['tier'].upper())
                cost_tier = CostTier(profile_data['cost_tier'].upper())
                
                # Extract capabilities
                caps = profile_data['capabilities']
                
                # Create ModelProfile
                profile = ModelProfile(
                    model_id=model_id,
                    provider_type=provider_type,
                    category=category,
                    tier=tier,
                    context_length=profile_data['context_length'],
                    max_output_tokens=profile_data['max_output_tokens'],
                    supports_streaming=profile_data['supports_streaming'],
                    supports_tools=profile_data['supports_tools'],
                    supports_vision=profile_data['supports_vision'],
                    quality_score=caps['quality_score'],
                    reasoning_capability=caps['reasoning_capability'],
                    creativity_capability=caps['creativity_capability'],
                    coding_capability=caps['coding_capability'],
                    multimodal_capability=caps['multimodal_capability'],
                    tool_use_capability=caps['tool_use_capability'],
                    cost_tier=cost_tier,
                    recommended_for=profile_data['recommended_for'],
                )
                
                self.model_profiles[model_id] = profile
                
                # Update provider mapping
                if provider_type not in self.provider_model_mapping:
                    self.provider_model_mapping[provider_type] = []
                self.provider_model_mapping[provider_type].append(model_id)
                
                logger.debug("Loaded model profile from config",
                           model_id=model_id, provider=provider_type.value)
                           
            except (KeyError, ValueError) as e:
                logger.error("Failed to load model profile",
                           model_id=model_id, error=str(e))
                raise ConfigurationError(f"Invalid profile config for {model_id}: {e}")
    
    def _load_routing_rules_from_config(self, rules_config: List[Dict[str, Any]]) -> None:
        """Load routing rules from configuration list."""
        for rule_data in rules_config:
            try:
                # Parse categories and tiers
                preferred_categories = []
                if rule_data.get('preferred_categories'):
                    for cat in rule_data['preferred_categories']:
                        try:
                            preferred_categories.append(ModelCategory(cat.upper()))
                        except ValueError:
                            logger.warning("Invalid category in routing rule",
                                         category=cat, rule=rule_data['name'])
                
                preferred_tiers = []
                if rule_data.get('preferred_tiers'):
                    for tier in rule_data['preferred_tiers']:
                        try:
                            preferred_tiers.append(ModelTier(tier.upper()))
                        except ValueError:
                            logger.warning("Invalid tier in routing rule",
                                         tier=tier, rule=rule_data['name'])
                
                # Parse required capabilities
                required_capabilities = []
                if rule_data.get('required_capabilities'):
                    for cap in rule_data['required_capabilities']:
                        try:
                            required_capabilities.append(ProviderCapability(cap.upper()))
                        except ValueError:
                            logger.warning("Invalid capability in routing rule",
                                         capability=cap, rule=rule_data['name'])
                
                # Create RoutingRule
                rule = RoutingRule(
                    name=rule_data['name'],
                    description=rule_data['description'],
                    priority=rule_data['priority'],
                    request_patterns=rule_data.get('request_patterns', []),
                    preferred_categories=preferred_categories,
                    preferred_tiers=preferred_tiers,
                    required_capabilities=required_capabilities,
                    min_quality_score=rule_data.get('min_quality_score'),
                    max_latency_ms=rule_data.get('max_latency_ms'),
                )
                
                self.routing_rules.append(rule)
                
                logger.debug("Loaded routing rule from config",
                           rule_name=rule_data['name'],
                           priority=rule_data['priority'])
                           
            except (KeyError, ValueError) as e:
                logger.error("Failed to load routing rule",
                           rule=rule_data.get('name', 'unnamed'), error=str(e))
                raise ConfigurationError(f"Invalid routing rule config: {e}")
        
        # Sort rules by priority (highest first)
        self.routing_rules.sort(key=lambda r: r.priority, reverse=True)
    
    def add_routing_rule(self, rule: RoutingRule) -> None:
        """Add a custom routing rule."""
        self.routing_rules.append(rule)
        self.routing_rules.sort(key=lambda r: r.priority, reverse=True)
        
        logger.info(
            "Added routing rule",
            rule_name=rule.name,
            priority=rule.priority,
        )
    
    def get_model_recommendations(
        self, request: AIRequest, max_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """Get model recommendations for a request."""
        analysis = self._analyze_request(request)
        recommendations = []
        
        for model_id, profile in self.model_profiles.items():
            if not self._validate_model_compatibility(request, model_id):
                continue
            
            score = self._calculate_model_score(profile, request, analysis, None)
            estimated_cost = self._estimate_model_cost(
                profile, analysis["estimated_input_tokens"], request.max_tokens or 1000
            )
            
            recommendations.append({
                "model_id": model_id,
                "provider": profile.provider_type.value,
                "category": profile.category.value,
                "tier": profile.tier.value,
                "score": score,
                "estimated_cost": estimated_cost,
                "avg_latency_ms": profile.avg_latency_ms,
                "quality_score": profile.quality_score,
                "recommended_for": profile.recommended_for,
                "reasoning": self._generate_recommendation_reasoning(profile, analysis),
            })
        
        # Sort by score and limit results
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:max_recommendations]
    
    def _generate_recommendation_reasoning(
        self, profile: ModelProfile, analysis: Dict[str, Any]
    ) -> str:
        """Generate reasoning for why a model is recommended."""
        reasons = []
        
        if analysis["task_type"] == "coding" and profile.coding_capability > 0.8:
            reasons.append("excellent coding capabilities")
        
        if analysis["requires_vision"] and profile.supports_vision:
            reasons.append("supports vision/image analysis")
        
        if analysis["speed_priority"] and profile.tokens_per_second > 100:
            reasons.append("fast response times")
        
        if analysis["quality_priority"] and profile.quality_score > 0.85:
            reasons.append("high-quality outputs")
        
        if profile.tier == ModelTier.EFFICIENT:
            reasons.append("cost-effective")
        
        return f"Recommended for {', '.join(reasons)}" if reasons else "General purpose model"
    
    def get_router_statistics(self) -> Dict[str, Any]:
        """Get comprehensive router statistics."""
        category_counts = {}
        tier_counts = {}
        provider_counts = {}
        
        for profile in self.model_profiles.values():
            category_counts[profile.category.value] = category_counts.get(profile.category.value, 0) + 1
            tier_counts[profile.tier.value] = tier_counts.get(profile.tier.value, 0) + 1
            provider_counts[profile.provider_type.value] = provider_counts.get(profile.provider_type.value, 0) + 1
        
        return {
            "total_models": len(self.model_profiles),
            "total_routing_rules": len(self.routing_rules),
            "active_routing_rules": len([r for r in self.routing_rules if r.enabled]),
            "category_distribution": category_counts,
            "tier_distribution": tier_counts,
            "provider_distribution": provider_counts,
            "tool_compatibility_entries": len(self.tool_compatibility),
        }