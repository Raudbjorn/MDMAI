"""
AI Provider Authentication Layer for MDMAI TTRPG Assistant.

This package provides a comprehensive authentication and management system for AI providers
including Anthropic Claude, OpenAI GPT, and Google Gemini.

Key Features:
- Secure credential management with AES-256 encryption
- Provider authentication and validation
- Intelligent routing with fallback and circuit breaker patterns
- Usage tracking and cost management
- Rate limiting with exponential backoff
- Health monitoring and alerting
- TTRPG-optimized model selection

Main Components:
- BaseAIProvider: Abstract base class for all AI providers
- CredentialManager: Secure credential storage and encryption
- ProviderRouter: Intelligent provider routing with fallback
- UsageTracker: Usage tracking and spending limits
- RateLimiter: Advanced rate limiting with backoff strategies
- HealthChecker: Provider health monitoring and alerting
- PricingConfigManager: Dynamic pricing configuration

Legacy Components (still available):
- AbstractProvider, ProviderRegistry, CostOptimizer (from existing codebase)
"""

# Version information
__version__ = "1.0.0"
__author__ = "MDMAI TTRPG Assistant Team"

# Import new authentication layer components
try:
    from .base_provider import (
        BaseAIProvider,
        ProviderType,
        ProviderConfig,
        CompletionResponse,
        ProviderError,
        ProviderAuthenticationError,
        ProviderRateLimitError,
        ProviderTimeoutError,
        ProviderQuotaExceededError,
        ProviderInvalidRequestError,
        NoAvailableProvidersError,
    )

    from .credential_manager import CredentialManager

    from .provider_router import (
        ProviderRouter,
        CircuitState,
        CircuitBreakerConfig,
        ProviderStats,
    )

    from .usage_tracker import (
        UsageTracker,
        UsageRecord,
        SpendingLimit,
        SpendingLimitExceededException,
    )

    from .rate_limiter import (
        RateLimiter,
        RateLimitConfig,
        BackoffStrategy,
    )

    from .health_checker import (
        HealthChecker,
        HealthStatus,
        HealthCheckResult,
        HealthMetrics,
        AlertSeverity,
        AlertRule,
    )

    from .pricing_config import (
        PricingConfigManager,
        PricingConfigError,
        get_pricing_manager,
    )

    # Import new provider implementations
    from .anthropic_provider_auth import AnthropicProvider as AnthropicAuthProvider
    from .openai_provider_auth import OpenAIProvider as OpenAIAuthProvider
    from .google_provider_auth import GoogleProvider as GoogleAuthProvider
    
    # New authentication layer is available
    AUTH_LAYER_AVAILABLE = True
    
except ImportError as e:
    # Fallback if new components are not available
    AUTH_LAYER_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"New authentication layer components not available: {e}")

# Import legacy components (existing codebase)
try:
    from .abstract_provider import AbstractProvider, ProviderCapability, ProviderStatus
    from .anthropic_provider import AnthropicProvider
    from .cost_optimizer import CostOptimizer, UsageTracker as LegacyUsageTracker
    from .error_handler import AIProviderError, ErrorHandler
    from .google_provider import GoogleProvider
    from .openai_provider import OpenAIProvider
    from .provider_manager import AIProviderManager
    from .provider_registry import ProviderRegistry
    from .streaming_manager import StreamingManager, StreamingResponse
    from .tool_translator import ToolTranslator
    
    LEGACY_COMPONENTS_AVAILABLE = True
    
except ImportError as e:
    LEGACY_COMPONENTS_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Legacy components not available: {e}")

# Build __all__ list dynamically based on what's available
__all__ = ["__version__", "__author__"]

if AUTH_LAYER_AVAILABLE:
    __all__.extend([
        # Base components
        "BaseAIProvider",
        "ProviderType",
        "ProviderConfig",
        "CompletionResponse",
        
        # Exceptions
        "ProviderError",
        "ProviderAuthenticationError",
        "ProviderRateLimitError",
        "ProviderTimeoutError",
        "ProviderQuotaExceededError",
        "ProviderInvalidRequestError",
        "NoAvailableProvidersError",
        "SpendingLimitExceededException",
        "PricingConfigError",
        
        # Core managers
        "CredentialManager",
        "ProviderRouter",
        "UsageTracker",
        "RateLimiter",
        "HealthChecker",
        "PricingConfigManager",
        
        # Configuration classes
        "CircuitBreakerConfig",
        "RateLimitConfig",
        "SpendingLimit",
        "AlertRule",
        
        # Data classes
        "ProviderStats",
        "UsageRecord",
        "HealthCheckResult",
        "HealthMetrics",
        
        # Enums
        "CircuitState",
        "BackoffStrategy",
        "HealthStatus",
        "AlertSeverity",
        
        # New provider implementations
        "AnthropicAuthProvider",
        "OpenAIAuthProvider", 
        "GoogleAuthProvider",
        
        # Utility functions
        "get_pricing_manager",
    ])

if LEGACY_COMPONENTS_AVAILABLE:
    __all__.extend([
        "AbstractProvider",
        "ProviderCapability", 
        "ProviderStatus",
        "AnthropicProvider",
        "OpenAIProvider", 
        "GoogleProvider",
        "AIProviderManager",
        "ProviderRegistry",
        "CostOptimizer",
        "LegacyUsageTracker", 
        "ToolTranslator",
        "StreamingManager",
        "StreamingResponse",
        "ErrorHandler",
        "AIProviderError",
    ])