"""Configuration loader for model selection components."""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from ...ai_providers.models import ProviderType
from ..preference_learner import FeedbackType


@dataclass
class DecisionTreeConfig:
    """Configuration for decision tree model selection."""
    scoring_weights: Dict[str, float] = field(default_factory=dict)
    performance_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    decision_tree: Dict[str, Any] = field(default_factory=dict)
    default_recommendations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PreferenceLearnerConfig:
    """Configuration for preference learning."""
    learning_rate: float = 0.1
    decay_factor: float = 0.95
    min_confidence: float = 0.3
    max_sessions_per_user: int = 50
    feedback_weights: Dict[str, float] = field(default_factory=dict)
    confidence_thresholds: Dict[str, float] = field(default_factory=dict)
    session_analysis: Dict[str, Any] = field(default_factory=dict)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    pattern_detection: Dict[str, Any] = field(default_factory=dict)
    data_retention: Dict[str, Any] = field(default_factory=dict)
    advanced_learning: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSelectionConfig:
    """Complete configuration for model selection system."""
    decision_tree: DecisionTreeConfig = field(default_factory=DecisionTreeConfig)
    preference_learner: PreferenceLearnerConfig = field(default_factory=PreferenceLearnerConfig)


class ConfigLoader:
    """Loads and manages configuration for model selection components."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize config loader.
        
        Args:
            config_dir: Directory containing config files. Defaults to this module's config dir.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent
        self.config_dir = Path(config_dir)
        self._config: Optional[ModelSelectionConfig] = None
    
    def load_config(self, force_reload: bool = False) -> ModelSelectionConfig:
        """Load configuration from YAML files.
        
        Args:
            force_reload: Whether to force reload even if config is cached.
            
        Returns:
            Complete model selection configuration.
        """
        if self._config is not None and not force_reload:
            return self._config
        
        config = ModelSelectionConfig()
        
        # Load decision tree config
        decision_tree_path = self.config_dir / "decision_tree_config.yaml"
        if decision_tree_path.exists():
            with open(decision_tree_path, 'r', encoding='utf-8') as f:
                dt_data = yaml.safe_load(f)
                config.decision_tree = DecisionTreeConfig(
                    scoring_weights=dt_data.get('scoring_weights', {}),
                    performance_thresholds=dt_data.get('performance_thresholds', {}),
                    decision_tree=dt_data.get('decision_tree', {}),
                    default_recommendations=dt_data.get('default_recommendations', [])
                )
        
        # Load preference learner config
        pref_path = self.config_dir / "preference_learner_config.yaml"
        if pref_path.exists():
            with open(pref_path, 'r', encoding='utf-8') as f:
                pref_data = yaml.safe_load(f)
                config.preference_learner = PreferenceLearnerConfig(
                    learning_rate=pref_data.get('learning_rate', 0.1),
                    decay_factor=pref_data.get('decay_factor', 0.95),
                    min_confidence=pref_data.get('min_confidence', 0.3),
                    max_sessions_per_user=pref_data.get('max_sessions_per_user', 50),
                    feedback_weights=pref_data.get('feedback_weights', {}),
                    confidence_thresholds=pref_data.get('confidence_thresholds', {}),
                    session_analysis=pref_data.get('session_analysis', {}),
                    user_profile=pref_data.get('user_profile', {}),
                    pattern_detection=pref_data.get('pattern_detection', {}),
                    data_retention=pref_data.get('data_retention', {}),
                    advanced_learning=pref_data.get('advanced_learning', {})
                )
        
        self._config = config
        return config
    
    def get_feedback_weights(self) -> Dict[FeedbackType, float]:
        """Get feedback weights as FeedbackType enum mapping."""
        config = self.load_config()
        weights = {}
        
        for feedback_name, weight in config.preference_learner.feedback_weights.items():
            try:
                feedback_type = FeedbackType[feedback_name]
                weights[feedback_type] = weight
            except KeyError:
                # Skip unknown feedback types
                continue
        
        return weights
    
    def get_scoring_weights(self) -> Dict[str, float]:
        """Get decision tree scoring weights."""
        config = self.load_config()
        return config.decision_tree.scoring_weights
    
    def get_performance_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get performance thresholds for decision tree."""
        config = self.load_config()
        return config.decision_tree.performance_thresholds
    
    def build_decision_tree_from_config(self):
        """Build decision tree structure from configuration.
        
        This method would parse the YAML structure and create DecisionTreeNode objects.
        Implementation depends on the specific decision tree node structure.
        """
        config = self.load_config()
        tree_config = config.decision_tree.decision_tree
        
        # TODO: Implement dynamic tree building from config
        # This would replace the hardcoded _build_default_decision_tree method
        # For now, return None to indicate config-based building is not yet implemented
        return None
    
    def get_default_recommendations(self) -> List[Tuple[ProviderType, str, float]]:
        """Get default model recommendations from config."""
        config = self.load_config()
        recommendations = []
        
        for rec in config.decision_tree.default_recommendations:
            try:
                provider = ProviderType[rec['provider'].upper()]
                model = rec['model']
                score = rec['score']
                recommendations.append((provider, model, score))
            except (KeyError, ValueError):
                # Skip invalid recommendations
                continue
        
        return recommendations
    
    def update_config_value(self, path: str, value: Any) -> None:
        """Update a configuration value at runtime.
        
        Args:
            path: Dot-separated path to the config value (e.g., 'preference_learner.learning_rate')
            value: New value to set
        """
        config = self.load_config()
        
        parts = path.split('.')
        if len(parts) < 2:
            raise ValueError("Config path must have at least 2 parts (component.setting)")
        
        component = parts[0]
        setting_path = parts[1:]
        
        # Navigate to the target setting
        if component == 'decision_tree':
            target = config.decision_tree
        elif component == 'preference_learner':
            target = config.preference_learner
        else:
            raise ValueError(f"Unknown config component: {component}")
        
        # Set the value
        for part in setting_path[:-1]:
            target = getattr(target, part)
        setattr(target, setting_path[-1], value)
    
    def save_config(self, config: ModelSelectionConfig) -> None:
        """Save configuration back to YAML files.
        
        Args:
            config: Configuration to save
        """
        # Save decision tree config
        dt_path = self.config_dir / "decision_tree_config.yaml"
        dt_data = {
            'scoring_weights': config.decision_tree.scoring_weights,
            'performance_thresholds': config.decision_tree.performance_thresholds,
            'decision_tree': config.decision_tree.decision_tree,
            'default_recommendations': config.decision_tree.default_recommendations
        }
        
        with open(dt_path, 'w', encoding='utf-8') as f:
            yaml.dump(dt_data, f, default_flow_style=False, sort_keys=False)
        
        # Save preference learner config
        pref_path = self.config_dir / "preference_learner_config.yaml"
        pref_data = {
            'learning_rate': config.preference_learner.learning_rate,
            'decay_factor': config.preference_learner.decay_factor,
            'min_confidence': config.preference_learner.min_confidence,
            'max_sessions_per_user': config.preference_learner.max_sessions_per_user,
            'feedback_weights': config.preference_learner.feedback_weights,
            'confidence_thresholds': config.preference_learner.confidence_thresholds,
            'session_analysis': config.preference_learner.session_analysis,
            'user_profile': config.preference_learner.user_profile,
            'pattern_detection': config.preference_learner.pattern_detection,
            'data_retention': config.preference_learner.data_retention,
            'advanced_learning': config.preference_learner.advanced_learning
        }
        
        with open(pref_path, 'w', encoding='utf-8') as f:
            yaml.dump(pref_data, f, default_flow_style=False, sort_keys=False)


# Global config loader instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """Get the global config loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def load_model_selection_config() -> ModelSelectionConfig:
    """Load the model selection configuration."""
    return get_config_loader().load_config()