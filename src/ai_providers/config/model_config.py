"""
Centralized Model Configuration Module
Addresses PR #59 review issues: centralized model specs, costs, and routing rules
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import json
import os
from pathlib import Path

from ..models import ProviderType, CostTier, ProviderCapability


@dataclass
class ModelCostConfig:
    """Model cost configuration."""
    
    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float
    context_window: int
    max_output_tokens: int
    cost_tier: CostTier
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token counts."""
        input_cost = (input_tokens / 1000.0) * self.input_cost_per_1k_tokens
        output_cost = (output_tokens / 1000.0) * self.output_cost_per_1k_tokens
        return input_cost + output_cost


@dataclass
class ModelProfile:
    """Complete model profile including capabilities and routing preferences."""
    
    model_id: str
    provider: ProviderType
    cost_config: ModelCostConfig
    capabilities: List[ProviderCapability] = field(default_factory=list)
    
    # Performance characteristics
    avg_latency_ms: float = 3000.0
    reliability_score: float = 0.95
    quality_score: float = 0.8
    
    # Routing preferences
    preferred_for: List[str] = field(default_factory=list)  # Task types this model excels at
    fallback_priority: int = 10  # Lower number = higher priority
    
    # Feature support
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_vision: bool = False
    supports_json_mode: bool = False
    
    # Operational limits
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 60000
    max_parallel_requests: int = 10


@dataclass
class RoutingRuleConfig:
    """Configuration for intelligent routing rules."""
    
    rule_id: str
    name: str
    description: str
    
    # Conditions for rule activation
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Actions when rule matches
    preferred_providers: List[ProviderType] = field(default_factory=list)
    excluded_providers: List[ProviderType] = field(default_factory=list)
    required_capabilities: List[ProviderCapability] = field(default_factory=list)
    
    # Performance requirements
    max_latency_ms: Optional[float] = None
    max_cost_per_request: Optional[float] = None
    min_quality_score: Optional[float] = None
    
    priority: int = 50  # Rule priority (lower = higher priority)
    enabled: bool = True


@dataclass
class FallbackTierConfig:
    """Configuration for fallback tiers."""
    
    tier_name: str
    providers: List[ProviderType]
    selection_strategy: str
    max_attempts: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True
    
    # Relaxed criteria for fallback
    allow_higher_cost_factor: float = 1.5  # Allow up to 50% higher cost
    allow_higher_latency_factor: float = 2.0  # Allow up to 2x latency
    
    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 300


@dataclass
class NormalizationConfig:
    """Configuration for score normalization."""
    
    # Dynamic ranges based on actual data
    cost_range: tuple[float, float] = (0.0001, 1.0)  # Min and max cost per request
    latency_range: tuple[float, float] = (100, 30000)  # Min and max latency in ms
    quality_range: tuple[float, float] = (0.0, 1.0)  # Quality score range
    reliability_range: tuple[float, float] = (0.0, 1.0)  # Reliability percentage range
    
    def normalize_cost(self, cost: float) -> float:
        """Normalize cost to 0-1 range (inverted - lower cost = higher score)."""
        min_cost, max_cost = self.cost_range
        if cost <= min_cost:
            return 1.0
        if cost >= max_cost:
            return 0.0
        return 1.0 - ((cost - min_cost) / (max_cost - min_cost))
    
    def normalize_latency(self, latency_ms: float) -> float:
        """Normalize latency to 0-1 range (inverted - lower latency = higher score)."""
        min_lat, max_lat = self.latency_range
        if latency_ms <= min_lat:
            return 1.0
        if latency_ms >= max_lat:
            return 0.0
        return 1.0 - ((latency_ms - min_lat) / (max_lat - min_lat))
    
    def normalize_quality(self, quality: float) -> float:
        """Normalize quality score to 0-1 range."""
        min_qual, max_qual = self.quality_range
        return max(0.0, min(1.0, (quality - min_qual) / (max_qual - min_qual)))
    
    def normalize_reliability(self, reliability: float) -> float:
        """Normalize reliability to 0-1 range."""
        min_rel, max_rel = self.reliability_range
        return max(0.0, min(1.0, (reliability - min_rel) / (max_rel - min_rel)))
    
    def update_ranges(self, metrics: Dict[str, List[float]]) -> None:
        """Update normalization ranges based on observed data."""
        if "costs" in metrics and metrics["costs"]:
            self.cost_range = (min(metrics["costs"]), max(metrics["costs"]))
        if "latencies" in metrics and metrics["latencies"]:
            self.latency_range = (min(metrics["latencies"]), max(metrics["latencies"]))
        if "qualities" in metrics and metrics["qualities"]:
            self.quality_range = (min(metrics["qualities"]), max(metrics["qualities"]))
        if "reliabilities" in metrics and metrics["reliabilities"]:
            self.reliability_range = (min(metrics["reliabilities"]), max(metrics["reliabilities"]))


class ModelConfigManager:
    """
    Centralized configuration manager for all model-related settings.
    Addresses review issues: centralized costs, routing rules, and normalization.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.model_profiles: Dict[str, ModelProfile] = {}
        self.routing_rules: List[RoutingRuleConfig] = []
        self.fallback_tiers: Dict[str, FallbackTierConfig] = {}
        self.normalization_config = NormalizationConfig()
        
        # Load configurations
        self._load_default_configs()
        if config_path and config_path.exists():
            self._load_custom_configs()
    
    def _get_default_config_path(self) -> Path:
        """Get default configuration path."""
        return Path(__file__).parent / "model_configs.json"
    
    def _load_default_configs(self) -> None:
        """Load default model configurations."""
        # Default model profiles
        self.model_profiles = {
            # Anthropic models
            "claude-3-opus": ModelProfile(
                model_id="claude-3-opus",
                provider=ProviderType.ANTHROPIC,
                cost_config=ModelCostConfig(
                    input_cost_per_1k_tokens=0.015,
                    output_cost_per_1k_tokens=0.075,
                    context_window=200000,
                    max_output_tokens=4096,
                    cost_tier=CostTier.PREMIUM,
                ),
                capabilities=[
                    ProviderCapability.GENERAL,
                    ProviderCapability.CODING,
                    ProviderCapability.ANALYSIS,
                ],
                avg_latency_ms=8000,
                reliability_score=0.98,
                quality_score=0.95,
                preferred_for=["complex_analysis", "coding", "creative_writing"],
                supports_tools=True,
                supports_vision=True,
            ),
            "claude-3-sonnet": ModelProfile(
                model_id="claude-3-sonnet",
                provider=ProviderType.ANTHROPIC,
                cost_config=ModelCostConfig(
                    input_cost_per_1k_tokens=0.003,
                    output_cost_per_1k_tokens=0.015,
                    context_window=200000,
                    max_output_tokens=4096,
                    cost_tier=CostTier.MEDIUM,
                ),
                capabilities=[
                    ProviderCapability.GENERAL,
                    ProviderCapability.CODING,
                ],
                avg_latency_ms=5000,
                reliability_score=0.97,
                quality_score=0.88,
                preferred_for=["general_tasks", "code_review"],
                supports_tools=True,
                supports_vision=True,
            ),
            "claude-3-haiku": ModelProfile(
                model_id="claude-3-haiku",
                provider=ProviderType.ANTHROPIC,
                cost_config=ModelCostConfig(
                    input_cost_per_1k_tokens=0.00025,
                    output_cost_per_1k_tokens=0.00125,
                    context_window=200000,
                    max_output_tokens=4096,
                    cost_tier=CostTier.LOW,
                ),
                capabilities=[ProviderCapability.GENERAL],
                avg_latency_ms=1500,
                reliability_score=0.96,
                quality_score=0.75,
                preferred_for=["simple_tasks", "quick_responses"],
                supports_vision=True,
            ),
            
            # OpenAI models
            "gpt-4": ModelProfile(
                model_id="gpt-4",
                provider=ProviderType.OPENAI,
                cost_config=ModelCostConfig(
                    input_cost_per_1k_tokens=0.03,
                    output_cost_per_1k_tokens=0.06,
                    context_window=128000,
                    max_output_tokens=4096,
                    cost_tier=CostTier.PREMIUM,
                ),
                capabilities=[
                    ProviderCapability.GENERAL,
                    ProviderCapability.CODING,
                    ProviderCapability.ANALYSIS,
                ],
                avg_latency_ms=10000,
                reliability_score=0.99,
                quality_score=0.93,
                preferred_for=["complex_reasoning", "math", "coding"],
                supports_tools=True,
                supports_json_mode=True,
            ),
            "gpt-3.5-turbo": ModelProfile(
                model_id="gpt-3.5-turbo",
                provider=ProviderType.OPENAI,
                cost_config=ModelCostConfig(
                    input_cost_per_1k_tokens=0.0005,
                    output_cost_per_1k_tokens=0.0015,
                    context_window=16385,
                    max_output_tokens=4096,
                    cost_tier=CostTier.LOW,
                ),
                capabilities=[ProviderCapability.GENERAL],
                avg_latency_ms=2000,
                reliability_score=0.98,
                quality_score=0.70,
                preferred_for=["simple_queries", "fast_responses"],
                supports_tools=True,
                supports_json_mode=True,
            ),
            
            # Google models
            "gemini-pro": ModelProfile(
                model_id="gemini-pro",
                provider=ProviderType.GOOGLE,
                cost_config=ModelCostConfig(
                    input_cost_per_1k_tokens=0.0005,
                    output_cost_per_1k_tokens=0.0015,
                    context_window=32768,
                    max_output_tokens=2048,
                    cost_tier=CostTier.LOW,
                ),
                capabilities=[ProviderCapability.GENERAL],
                avg_latency_ms=3000,
                reliability_score=0.94,
                quality_score=0.78,
                preferred_for=["general_conversation", "summarization"],
                supports_tools=True,
            ),
        }
        
        # Default routing rules
        self.routing_rules = [
            RoutingRuleConfig(
                rule_id="coding_preference",
                name="Coding Task Routing",
                description="Route coding tasks to specialized models",
                conditions={"task_type": "coding", "complexity": "high"},
                preferred_providers=[ProviderType.ANTHROPIC, ProviderType.OPENAI],
                required_capabilities=[ProviderCapability.CODING],
                min_quality_score=0.85,
                priority=10,
            ),
            RoutingRuleConfig(
                rule_id="budget_constraint",
                name="Budget Constrained Routing",
                description="Route to low-cost providers when budget is tight",
                conditions={"budget_remaining_percentage": "<20"},
                preferred_providers=[ProviderType.GOOGLE],
                max_cost_per_request=0.1,
                priority=5,
            ),
            RoutingRuleConfig(
                rule_id="speed_critical",
                name="Speed Critical Routing",
                description="Route time-sensitive requests to fast providers",
                conditions={"speed_requirement": "critical"},
                max_latency_ms=3000,
                priority=15,
            ),
        ]
        
        # Default fallback tiers
        self.fallback_tiers = {
            "primary": FallbackTierConfig(
                tier_name="primary",
                providers=[ProviderType.ANTHROPIC, ProviderType.OPENAI],
                selection_strategy="weighted_composite",
                max_attempts=2,
            ),
            "secondary": FallbackTierConfig(
                tier_name="secondary",
                providers=[ProviderType.GOOGLE, ProviderType.OPENAI],
                selection_strategy="speed_optimized",
                max_attempts=2,
                allow_higher_cost_factor=1.5,
            ),
            "emergency": FallbackTierConfig(
                tier_name="emergency",
                providers=[ProviderType.ANTHROPIC, ProviderType.OPENAI, ProviderType.GOOGLE],
                selection_strategy="any_available",
                max_attempts=3,
                allow_higher_cost_factor=3.0,
                allow_higher_latency_factor=3.0,
            ),
        }
    
    def _load_custom_configs(self) -> None:
        """Load custom configurations from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
                
                # Override model profiles
                if "model_profiles" in config_data:
                    for model_id, profile_data in config_data["model_profiles"].items():
                        self._update_model_profile(model_id, profile_data)
                
                # Override routing rules
                if "routing_rules" in config_data:
                    self._update_routing_rules(config_data["routing_rules"])
                
                # Override fallback tiers
                if "fallback_tiers" in config_data:
                    self._update_fallback_tiers(config_data["fallback_tiers"])
                
                # Override normalization ranges
                if "normalization" in config_data:
                    self._update_normalization_config(config_data["normalization"])
                    
        except FileNotFoundError:
            pass  # Use defaults if custom config doesn't exist
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid configuration file: {e}")
    
    def _update_model_profile(self, model_id: str, profile_data: Dict[str, Any]) -> None:
        """Update a model profile from configuration data."""
        if model_id in self.model_profiles:
            profile = self.model_profiles[model_id]
            
            # Update cost config
            if "cost_config" in profile_data:
                cost_data = profile_data["cost_config"]
                profile.cost_config.input_cost_per_1k_tokens = cost_data.get(
                    "input_cost_per_1k_tokens", 
                    profile.cost_config.input_cost_per_1k_tokens
                )
                profile.cost_config.output_cost_per_1k_tokens = cost_data.get(
                    "output_cost_per_1k_tokens", 
                    profile.cost_config.output_cost_per_1k_tokens
                )
            
            # Update performance characteristics
            if "performance" in profile_data:
                perf_data = profile_data["performance"]
                profile.avg_latency_ms = perf_data.get("avg_latency_ms", profile.avg_latency_ms)
                profile.reliability_score = perf_data.get("reliability_score", profile.reliability_score)
                profile.quality_score = perf_data.get("quality_score", profile.quality_score)
    
    def _update_routing_rules(self, rules_data: List[Dict[str, Any]]) -> None:
        """Update routing rules from configuration data."""
        for rule_data in rules_data:
            rule_id = rule_data.get("rule_id")
            existing_rule = next((r for r in self.routing_rules if r.rule_id == rule_id), None)
            
            if existing_rule:
                # Update existing rule
                for key, value in rule_data.items():
                    if hasattr(existing_rule, key):
                        setattr(existing_rule, key, value)
            else:
                # Add new rule
                self.routing_rules.append(RoutingRuleConfig(**rule_data))
    
    def _update_fallback_tiers(self, tiers_data: Dict[str, Dict[str, Any]]) -> None:
        """Update fallback tier configurations."""
        for tier_name, tier_data in tiers_data.items():
            if tier_name in self.fallback_tiers:
                tier = self.fallback_tiers[tier_name]
                for key, value in tier_data.items():
                    if hasattr(tier, key):
                        setattr(tier, key, value)
            else:
                self.fallback_tiers[tier_name] = FallbackTierConfig(
                    tier_name=tier_name,
                    **tier_data
                )
    
    def _update_normalization_config(self, norm_data: Dict[str, Any]) -> None:
        """Update normalization configuration."""
        if "cost_range" in norm_data:
            self.normalization_config.cost_range = tuple(norm_data["cost_range"])
        if "latency_range" in norm_data:
            self.normalization_config.latency_range = tuple(norm_data["latency_range"])
        if "quality_range" in norm_data:
            self.normalization_config.quality_range = tuple(norm_data["quality_range"])
        if "reliability_range" in norm_data:
            self.normalization_config.reliability_range = tuple(norm_data["reliability_range"])
    
    def get_model_profile(self, model_id: str) -> Optional[ModelProfile]:
        """Get model profile by ID."""
        return self.model_profiles.get(model_id)
    
    def get_model_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a specific model and token counts."""
        profile = self.get_model_profile(model_id)
        if profile:
            return profile.cost_config.calculate_cost(input_tokens, output_tokens)
        # Default fallback cost
        return (input_tokens / 1000.0) * 0.001 + (output_tokens / 1000.0) * 0.002
    
    def get_routing_rules(self, enabled_only: bool = True) -> List[RoutingRuleConfig]:
        """Get routing rules, optionally filtered by enabled status."""
        if enabled_only:
            return [r for r in self.routing_rules if r.enabled]
        return self.routing_rules
    
    def get_fallback_tier(self, tier_name: str) -> Optional[FallbackTierConfig]:
        """Get fallback tier configuration by name."""
        return self.fallback_tiers.get(tier_name)
    
    def save_config(self, path: Optional[Path] = None) -> None:
        """Save current configuration to JSON file."""
        save_path = path or self.config_path
        
        config_data = {
            "model_profiles": {
                model_id: {
                    "cost_config": {
                        "input_cost_per_1k_tokens": profile.cost_config.input_cost_per_1k_tokens,
                        "output_cost_per_1k_tokens": profile.cost_config.output_cost_per_1k_tokens,
                    },
                    "performance": {
                        "avg_latency_ms": profile.avg_latency_ms,
                        "reliability_score": profile.reliability_score,
                        "quality_score": profile.quality_score,
                    }
                }
                for model_id, profile in self.model_profiles.items()
            },
            "routing_rules": [
                {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "priority": rule.priority,
                    "enabled": rule.enabled,
                    "conditions": rule.conditions,
                }
                for rule in self.routing_rules
            ],
            "fallback_tiers": {
                tier_name: {
                    "providers": [p.value for p in tier.providers],
                    "selection_strategy": tier.selection_strategy,
                    "max_attempts": tier.max_attempts,
                }
                for tier_name, tier in self.fallback_tiers.items()
            },
            "normalization": {
                "cost_range": list(self.normalization_config.cost_range),
                "latency_range": list(self.normalization_config.latency_range),
                "quality_range": list(self.normalization_config.quality_range),
                "reliability_range": list(self.normalization_config.reliability_range),
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(config_data, f, indent=2)


# Global instance for easy access
_config_manager: Optional[ModelConfigManager] = None


def get_model_config_manager() -> ModelConfigManager:
    """Get or create the global model config manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ModelConfigManager()
    return _config_manager


def reload_config(config_path: Optional[Path] = None) -> None:
    """Reload configuration from file."""
    global _config_manager
    _config_manager = ModelConfigManager(config_path)