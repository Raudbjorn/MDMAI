"""Model Selection System for MDMAI TTRPG Assistant.

This package implements Task 25.5: a comprehensive model selection strategy
that intelligently chooses AI models based on task type, context, user preferences,
and performance metrics.

Key Components:
- TaskCategorizer: Categorizes TTRPG tasks for optimal model selection
- PerformanceBenchmark: Profiles and benchmarks model performance
- ModelOptimizer: Implements automatic model optimization algorithms
- PreferenceLearner: Learns user preferences from feedback
- ABTestingFramework: A/B tests models for continuous improvement
- ContextAwareSelector: Context-aware model switching logic
- DecisionTree: Decision tree and scoring algorithms
- IntelligentModelSelector: Main orchestrator
"""

from .task_categorizer import (
    TTRPGTaskType,
    TaskComplexity,
    TaskLatencyRequirement,
    TaskCharacteristics,
    TaskCategorizer
)

from .performance_profiler import (
    MetricType,
    PerformanceMetric,
    ModelPerformanceProfile,
    PerformanceBenchmark
)

from .model_optimizer import (
    OptimizationStrategy,
    LoadBalancingStrategy,
    OptimizationRule,
    ModelRecommendation,
    ModelOptimizer
)

from .preference_learner import (
    FeedbackType,
    PreferenceCategory,
    UserFeedback,
    UserPreferenceProfile,
    PreferenceLearner
)

from .ab_testing import (
    ExperimentStatus,
    ExperimentType,
    StatisticalSignificance,
    ExperimentVariant,
    ExperimentConfig,
    ExperimentResult,
    ABTestingFramework
)

from .context_aware_selector import (
    ContextType,
    SessionPhase,
    CampaignGenre,
    SessionContext,
    CampaignContext,
    SystemContext,
    ContextAwareSelector
)

from .decision_tree import (
    DecisionCriterion,
    ComparisonOperator,
    DecisionCondition,
    DecisionTreeNode,
    ModelScore,
    ModelSelectionDecisionTree
)

from .intelligent_model_selector import (
    SelectionMode,
    SelectionRequest,
    SelectionResult,
    IntelligentModelSelector
)

__version__ = "1.0.0"
__author__ = "MDMAI Development Team"

# Export the main interface
__all__ = [
    # Task Categorization
    "TTRPGTaskType",
    "TaskComplexity", 
    "TaskLatencyRequirement",
    "TaskCharacteristics",
    "TaskCategorizer",
    
    # Performance Profiling
    "MetricType",
    "PerformanceMetric",
    "ModelPerformanceProfile", 
    "PerformanceBenchmark",
    
    # Model Optimization
    "OptimizationStrategy",
    "LoadBalancingStrategy",
    "OptimizationRule",
    "ModelRecommendation",
    "ModelOptimizer",
    
    # Preference Learning
    "FeedbackType",
    "PreferenceCategory",
    "UserFeedback",
    "UserPreferenceProfile",
    "PreferenceLearner",
    
    # A/B Testing
    "ExperimentStatus",
    "ExperimentType", 
    "StatisticalSignificance",
    "ExperimentVariant",
    "ExperimentConfig",
    "ExperimentResult",
    "ABTestingFramework",
    
    # Context-Aware Selection
    "ContextType",
    "SessionPhase",
    "CampaignGenre",
    "SessionContext",
    "CampaignContext", 
    "SystemContext",
    "ContextAwareSelector",
    
    # Decision Tree
    "DecisionCriterion",
    "ComparisonOperator",
    "DecisionCondition",
    "DecisionTreeNode",
    "ModelScore",
    "ModelSelectionDecisionTree",
    
    # Main Interface
    "SelectionMode",
    "SelectionRequest",
    "SelectionResult", 
    "IntelligentModelSelector"
]