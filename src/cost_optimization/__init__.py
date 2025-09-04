"""
Cost Optimization Module for MDMAI Multi-Provider LLM System.

This module provides comprehensive cost optimization and management capabilities:
- Advanced cost optimization algorithms with ML-based prediction
- Intelligent budget enforcement with multi-tier limits
- Sophisticated alerting system with trend analysis
- Cost prediction engine with pattern recognition
- Provider pricing models with real-time cost formulas
- Token optimization strategies with context compression
"""

from .advanced_optimizer import (
    AdvancedCostOptimizer,
    CostOptimizationStrategy,
    ModelPerformanceMetrics,
    PricingModel,
    TokenOptimizer as AdvancedTokenOptimizer,
    UsagePredictor
)

from .alert_system import (
    AlertSeverity,
    AlertChannel,
    AlertType,
    AlertSystem,
    TrendAnalyzer,
    Alert,
    AlertRule
)

from .budget_enforcer import (
    BudgetEnforcer,
    BudgetLimit,
    BudgetLimitType,
    BudgetAction,
    VelocityAlert,
    SpendingVelocityMonitor,
    DegradationStrategy
)

from .cost_predictor import (
    CostPredictor,
    CostForecast,
    ForecastHorizon,
    PatternRecognizer,
    UsagePattern
)

from .pricing_engine import (
    PricingEngine,
    PricingTier,
    CostComponent,
    ModelPricingInfo,
    ProviderPricingConfig
)

from .token_optimizer import (
    TokenOptimizer,
    TokenEstimator,
    MessageImportanceScorer,
    SemanticCache,
    CompressionStrategy,
    MessageType
)

__all__ = [
    # Advanced Optimizer
    'AdvancedCostOptimizer',
    'CostOptimizationStrategy',
    'ModelPerformanceMetrics',
    'PricingModel',
    'AdvancedTokenOptimizer',
    'UsagePredictor',
    
    # Alert System
    'AlertSeverity',
    'AlertChannel',
    'AlertType',
    'AlertSystem',
    'TrendAnalyzer',
    'Alert',
    'AlertRule',
    
    # Budget Enforcer
    'BudgetEnforcer',
    'BudgetLimit',
    'BudgetLimitType',
    'BudgetAction',
    'VelocityAlert',
    'SpendingVelocityMonitor',
    'DegradationStrategy',
    
    # Cost Predictor
    'CostPredictor',
    'CostForecast',
    'ForecastHorizon',
    'PatternRecognizer',
    'UsagePattern',
    
    # Pricing Engine
    'PricingEngine',
    'PricingTier',
    'CostComponent',
    'ModelPricingInfo',
    'ProviderPricingConfig',
    
    # Token Optimizer
    'TokenOptimizer',
    'TokenEstimator',
    'MessageImportanceScorer',
    'SemanticCache',
    'CompressionStrategy',
    'MessageType',
]

# Version info
__version__ = '1.0.0'
__author__ = 'MDMAI Development Team'
__description__ = 'Advanced cost optimization and management for multi-provider LLM systems'