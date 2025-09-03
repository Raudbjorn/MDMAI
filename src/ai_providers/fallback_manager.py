"""
Advanced Fallback Strategy Manager with Circuit Breaker Patterns
Task 25.3: Develop Provider Router with Fallback
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

from structlog import get_logger

from .models import (
    AIRequest,
    AIResponse,
    ProviderType,
    ProviderCapability,
)
from .abstract_provider import AbstractProvider
from .health_monitor import HealthMonitor, ErrorType
from .intelligent_router import IntelligentRouter, SelectionStrategy, SelectionCriteria
from .config.model_config import get_model_config_manager, ModelConfigManager
from .utils.cost_utils import classify_error, ErrorClassification

logger = get_logger(__name__)


class FallbackTier(Enum):
    """Fallback tiers in order of preference."""
    
    PRIMARY = "primary"
    SECONDARY = "secondary"
    EMERGENCY = "emergency"
    LOCAL = "local"


class CircuitState(Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit tripped, blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class FallbackRule:
    """Rule defining fallback behavior for specific conditions."""
    
    name: str
    description: str
    trigger_conditions: List[str]  # Error types or conditions that trigger this fallback
    fallback_tier: FallbackTier
    fallback_providers: List[ProviderType]
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True
    enabled: bool = True


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration for a provider."""
    
    provider_type: ProviderType
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: int = 60
    half_open_max_calls: int = 3
    recovery_timeout_seconds: int = 300  # 5 minutes
    
    # Current state
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    state_changed_time: datetime = field(default_factory=datetime.now)


@dataclass
class FallbackAttempt:
    """Records details of a fallback attempt."""
    
    request_id: str
    original_provider: ProviderType
    fallback_provider: ProviderType
    fallback_tier: FallbackTier
    attempt_number: int
    trigger_reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = False
    error_message: Optional[str] = None
    latency_ms: Optional[float] = None


class FallbackManager:
    """
    Advanced fallback strategy manager with circuit breaker patterns.
    
    Features:
    - Multi-tier fallback strategies
    - Circuit breaker per provider
    - Intelligent fallback routing
    - Performance-based degradation
    - Emergency local model fallback
    - Comprehensive failure analysis
    """
    
    def __init__(
        self,
        intelligent_router: IntelligentRouter,
        health_monitor: HealthMonitor,
        circuit_breaker_configs: Optional[List[CircuitBreakerConfig]] = None,
        fallback_rules: Optional[List[FallbackRule]] = None,
        config_manager: Optional[ModelConfigManager] = None,
    ):
        self.router = intelligent_router
        self.health_monitor = health_monitor
        self.config_manager = config_manager or get_model_config_manager()
        
        # Circuit breaker management
        self.circuit_breakers: Dict[ProviderType, CircuitBreakerConfig] = {}
        if circuit_breaker_configs:
            for config in circuit_breaker_configs:
                self.circuit_breakers[config.provider_type] = config
        
        # Fallback rules
        self.fallback_rules: List[FallbackRule] = fallback_rules or self._create_default_fallback_rules()
        
        # Tracking
        self.fallback_history: List[FallbackAttempt] = []
        self.provider_tiers: Dict[ProviderType, FallbackTier] = {}
        
        # Local model integration (placeholder for local model providers)
        self.local_models_available = False
        self.local_model_provider: Optional[AbstractProvider] = None
        
        # Performance thresholds for degradation
        self.performance_thresholds = {
            "max_latency_ms": 30000,  # 30 seconds
            "min_success_rate": 0.8,   # 80%
            "max_cost_per_token": 0.01  # $0.01 per 1K tokens
        }
    
    async def execute_with_fallback(
        self,
        request: AIRequest,
        available_providers: List[AbstractProvider],
        primary_strategy: SelectionStrategy = SelectionStrategy.WEIGHTED_COMPOSITE,
        max_fallback_attempts: int = 3,
    ) -> Tuple[AIResponse, List[FallbackAttempt]]:
        """
        Execute request with intelligent fallback strategy.
        
        Args:
            request: The AI request
            available_providers: List of available providers
            primary_strategy: Primary selection strategy
            max_fallback_attempts: Maximum fallback attempts
            
        Returns:
            Tuple of (response, fallback_attempts)
        """
        fallback_attempts = []
        last_error = None
        
        logger.info(
            "Starting request execution with fallback",
            request_id=request.request_id,
            providers=len(available_providers),
            strategy=primary_strategy.value,
        )
        
        # Tier 1: Primary provider selection
        primary_providers = self._filter_providers_by_tier(
            available_providers, FallbackTier.PRIMARY
        )
        
        if primary_providers:
            try:
                response, attempt = await self._attempt_request(
                    request, primary_providers, primary_strategy, 
                    FallbackTier.PRIMARY, attempt_number=1
                )
                if response:
                    fallback_attempts.append(attempt)
                    return response, fallback_attempts
            except Exception as e:
                last_error = e
                attempt = self._create_failed_attempt(
                    request, None, FallbackTier.PRIMARY, 1, str(e)
                )
                fallback_attempts.append(attempt)
                logger.warning(
                    "Primary provider failed",
                    error=str(e),
                    request_id=request.request_id,
                )
        
        # Tier 2: Secondary fallback with different strategy
        secondary_providers = self._filter_providers_by_tier(
            available_providers, FallbackTier.SECONDARY
        )
        
        if secondary_providers and len(fallback_attempts) < max_fallback_attempts:
            try:
                # Use speed-optimized strategy for secondary
                response, attempt = await self._attempt_request(
                    request, secondary_providers, SelectionStrategy.SPEED_OPTIMIZED,
                    FallbackTier.SECONDARY, attempt_number=2
                )
                if response:
                    fallback_attempts.append(attempt)
                    return response, fallback_attempts
            except Exception as e:
                last_error = e
                attempt = self._create_failed_attempt(
                    request, None, FallbackTier.SECONDARY, 2, str(e)
                )
                fallback_attempts.append(attempt)
                logger.warning(
                    "Secondary provider failed",
                    error=str(e),
                    request_id=request.request_id,
                )
        
        # Tier 3: Emergency fallback with relaxed criteria
        emergency_providers = self._filter_providers_by_tier(
            available_providers, FallbackTier.EMERGENCY
        )
        
        if emergency_providers and len(fallback_attempts) < max_fallback_attempts:
            try:
                # Use most relaxed criteria for emergency
                emergency_criteria = SelectionCriteria(
                    cost_weight=0.1, speed_weight=0.4, quality_weight=0.2, reliability_weight=0.3,
                    max_latency_ms=60000,  # Allow up to 60 seconds
                    max_cost_per_request=10.0,  # Allow high cost in emergency
                )
                
                selected_score = await self.router.select_optimal_provider(
                    request, emergency_providers, SelectionStrategy.RELIABILITY_FOCUSED, emergency_criteria
                )
                
                if selected_score:
                    provider = next(
                        p for p in emergency_providers 
                        if p.provider_type == selected_score.provider_type
                    )
                    
                    start_time = datetime.now()
                    response = await provider.generate_response(request)
                    latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                    
                    attempt = FallbackAttempt(
                        request_id=request.request_id,
                        original_provider=primary_providers[0].provider_type if primary_providers else ProviderType.ANTHROPIC,
                        fallback_provider=provider.provider_type,
                        fallback_tier=FallbackTier.EMERGENCY,
                        attempt_number=len(fallback_attempts) + 1,
                        trigger_reason="Emergency fallback after primary/secondary failures",
                        success=True,
                        latency_ms=latency_ms,
                    )
                    fallback_attempts.append(attempt)
                    
                    logger.info(
                        "Emergency fallback successful",
                        provider=provider.provider_type.value,
                        request_id=request.request_id,
                    )
                    
                    return response, fallback_attempts
                    
            except Exception as e:
                last_error = e
                attempt = self._create_failed_attempt(
                    request, None, FallbackTier.EMERGENCY, len(fallback_attempts) + 1, str(e)
                )
                fallback_attempts.append(attempt)
                logger.error(
                    "Emergency provider failed",
                    error=str(e),
                    request_id=request.request_id,
                )
        
        # Tier 4: Local model fallback (if available)
        if (self.local_models_available and self.local_model_provider and 
            len(fallback_attempts) < max_fallback_attempts):
            try:
                logger.info(
                    "Attempting local model fallback",
                    request_id=request.request_id,
                )
                
                start_time = datetime.now()
                response = await self.local_model_provider.generate_response(request)
                latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                attempt = FallbackAttempt(
                    request_id=request.request_id,
                    original_provider=primary_providers[0].provider_type if primary_providers else ProviderType.ANTHROPIC,
                    fallback_provider=ProviderType.ANTHROPIC,  # Placeholder
                    fallback_tier=FallbackTier.LOCAL,
                    attempt_number=len(fallback_attempts) + 1,
                    trigger_reason="Local model fallback after all cloud providers failed",
                    success=True,
                    latency_ms=latency_ms,
                )
                fallback_attempts.append(attempt)
                
                logger.info(
                    "Local model fallback successful",
                    request_id=request.request_id,
                )
                
                return response, fallback_attempts
                
            except Exception as e:
                last_error = e
                attempt = self._create_failed_attempt(
                    request, None, FallbackTier.LOCAL, len(fallback_attempts) + 1, str(e)
                )
                fallback_attempts.append(attempt)
        
        # All fallback attempts exhausted
        logger.error(
            "All fallback attempts exhausted",
            request_id=request.request_id,
            attempts=len(fallback_attempts),
            last_error=str(last_error) if last_error else "Unknown",
        )
        
        raise Exception(f"All providers failed after {len(fallback_attempts)} attempts: {last_error}")
    
    async def _attempt_request(
        self,
        request: AIRequest,
        providers: List[AbstractProvider],
        strategy: SelectionStrategy,
        tier: FallbackTier,
        attempt_number: int,
    ) -> Tuple[Optional[AIResponse], FallbackAttempt]:
        """Attempt request with specified providers and strategy."""
        # Filter providers through circuit breakers
        available_providers = []
        for provider in providers:
            if self._is_circuit_closed(provider.provider_type):
                available_providers.append(provider)
        
        if not available_providers:
            raise Exception("All providers have open circuit breakers")
        
        # Select optimal provider
        selected_score = await self.router.select_optimal_provider(
            request, available_providers, strategy
        )
        
        if not selected_score:
            raise Exception("No suitable provider found")
        
        # Find the actual provider instance
        selected_provider = next(
            p for p in available_providers 
            if p.provider_type == selected_score.provider_type
        )
        
        # Execute request with circuit breaker protection
        start_time = datetime.now()
        try:
            response = await selected_provider.generate_response(request)
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Record successful execution
            await self._record_circuit_breaker_success(selected_provider.provider_type)
            
            attempt = FallbackAttempt(
                request_id=request.request_id,
                original_provider=selected_provider.provider_type,
                fallback_provider=selected_provider.provider_type,
                fallback_tier=tier,
                attempt_number=attempt_number,
                trigger_reason=f"{tier.value} tier execution",
                success=True,
                latency_ms=latency_ms,
            )
            
            return response, attempt
            
        except Exception as e:
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Record failure and potentially trip circuit breaker
            await self._record_circuit_breaker_failure(selected_provider.provider_type, e)
            
            attempt = FallbackAttempt(
                request_id=request.request_id,
                original_provider=selected_provider.provider_type,
                fallback_provider=selected_provider.provider_type,
                fallback_tier=tier,
                attempt_number=attempt_number,
                trigger_reason=f"{tier.value} tier execution failed",
                success=False,
                error_message=str(e),
                latency_ms=latency_ms,
            )
            
            raise e
    
    def _filter_providers_by_tier(
        self, providers: List[AbstractProvider], tier: FallbackTier
    ) -> List[AbstractProvider]:
        """Filter providers by fallback tier assignment."""
        if tier == FallbackTier.LOCAL:
            return [self.local_model_provider] if self.local_model_provider else []
        
        # Get tier configuration from config manager
        tier_config = self.config_manager.get_fallback_tier(tier.value.lower())
        if tier_config:
            preferred_types = tier_config.providers
        else:
            # Fallback to defaults if config not found
            tier_mapping = {
                FallbackTier.PRIMARY: [ProviderType.ANTHROPIC, ProviderType.OPENAI],
                FallbackTier.SECONDARY: [ProviderType.GOOGLE, ProviderType.OPENAI],
                FallbackTier.EMERGENCY: [ProviderType.ANTHROPIC, ProviderType.OPENAI, ProviderType.GOOGLE],
            }
            preferred_types = tier_mapping.get(tier, [])
        
        return [
            provider for provider in providers
            if provider.provider_type in preferred_types
        ]
    
    def _is_circuit_closed(self, provider_type: ProviderType) -> bool:
        """Check if circuit breaker is closed (allowing requests)."""
        if provider_type not in self.circuit_breakers:
            return True  # No circuit breaker configured, assume closed
        
        breaker = self.circuit_breakers[provider_type]
        
        if breaker.state == CircuitState.CLOSED:
            return True
        
        if breaker.state == CircuitState.OPEN:
            # Check if timeout has elapsed to move to half-open
            if (datetime.now() - breaker.state_changed_time).total_seconds() >= breaker.timeout_seconds:
                breaker.state = CircuitState.HALF_OPEN
                breaker.state_changed_time = datetime.now()
                breaker.success_count = 0
                logger.info(
                    "Circuit breaker moved to half-open",
                    provider=provider_type.value,
                )
                return True
            return False
        
        if breaker.state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open state
            return breaker.success_count < breaker.half_open_max_calls
        
        return False
    
    async def _record_circuit_breaker_success(self, provider_type: ProviderType) -> None:
        """Record successful request for circuit breaker."""
        if provider_type not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[provider_type]
        
        if breaker.state == CircuitState.HALF_OPEN:
            breaker.success_count += 1
            if breaker.success_count >= breaker.success_threshold:
                # Close the circuit
                breaker.state = CircuitState.CLOSED
                breaker.state_changed_time = datetime.now()
                breaker.failure_count = 0
                logger.info(
                    "Circuit breaker closed after successful recovery",
                    provider=provider_type.value,
                )
        
        elif breaker.state == CircuitState.CLOSED:
            # Reset failure count on success
            breaker.failure_count = 0
    
    async def _record_circuit_breaker_failure(
        self, provider_type: ProviderType, error: Exception
    ) -> None:
        """Record failed request for circuit breaker."""
        if provider_type not in self.circuit_breakers:
            # Create default circuit breaker
            self.circuit_breakers[provider_type] = CircuitBreakerConfig(
                provider_type=provider_type
            )
        
        breaker = self.circuit_breakers[provider_type]
        breaker.failure_count += 1
        breaker.last_failure_time = datetime.now()
        
        # Classify error using centralized classification
        error_class = classify_error(error)
        
        # Check if we should trip the circuit
        if (breaker.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN] and
            breaker.failure_count >= breaker.failure_threshold):
            
            breaker.state = CircuitState.OPEN
            breaker.state_changed_time = datetime.now()
            logger.warning(
                "Circuit breaker opened due to failures",
                provider=provider_type.value,
                failure_count=breaker.failure_count,
                error_type=error_class.value,
                error=str(error),
            )
        
        elif breaker.state == CircuitState.HALF_OPEN:
            # Return to open state if failure in half-open
            breaker.state = CircuitState.OPEN
            breaker.state_changed_time = datetime.now()
            logger.warning(
                "Circuit breaker returned to open after half-open failure",
                provider=provider_type.value,
                error_type=error_class.value,
            )
    
    def _create_failed_attempt(
        self,
        request: AIRequest,
        provider: Optional[AbstractProvider],
        tier: FallbackTier,
        attempt_number: int,
        error_message: str,
    ) -> FallbackAttempt:
        """Create a failed fallback attempt record."""
        return FallbackAttempt(
            request_id=request.request_id,
            original_provider=provider.provider_type if provider else ProviderType.ANTHROPIC,
            fallback_provider=provider.provider_type if provider else ProviderType.ANTHROPIC,
            fallback_tier=tier,
            attempt_number=attempt_number,
            trigger_reason=f"Failed to execute {tier.value} tier request",
            success=False,
            error_message=error_message,
        )
    
    def _create_default_fallback_rules(self) -> List[FallbackRule]:
        """Create default fallback rules based on error classifications."""
        # Use error classifications from centralized enum
        return [
            FallbackRule(
                name="rate_limit_fallback",
                description="Fallback when rate limited",
                trigger_conditions=[
                    ErrorClassification.RATE_LIMIT.value,
                    ErrorClassification.TOO_MANY_REQUESTS.value,
                ],
                fallback_tier=FallbackTier.SECONDARY,
                fallback_providers=[ProviderType.GOOGLE, ProviderType.OPENAI],
                max_retries=2,
                retry_delay_seconds=5.0,
            ),
            FallbackRule(
                name="timeout_fallback",
                description="Fallback on timeout errors",
                trigger_conditions=[
                    ErrorClassification.TIMEOUT.value,
                    ErrorClassification.CONNECTION_ERROR.value,
                ],
                fallback_tier=FallbackTier.SECONDARY,
                fallback_providers=[ProviderType.OPENAI, ProviderType.ANTHROPIC],
                max_retries=3,
                retry_delay_seconds=2.0,
            ),
            FallbackRule(
                name="quota_exceeded_fallback",
                description="Fallback when quota exceeded",
                trigger_conditions=[ErrorClassification.QUOTA_EXCEEDED.value],
                fallback_tier=FallbackTier.EMERGENCY,
                fallback_providers=[ProviderType.ANTHROPIC, ProviderType.OPENAI, ProviderType.GOOGLE],
                max_retries=1,
            ),
            FallbackRule(
                name="service_error_fallback",
                description="Fallback on service errors",
                trigger_conditions=[
                    ErrorClassification.SERVICE_UNAVAILABLE.value,
                    ErrorClassification.INTERNAL_SERVER_ERROR.value,
                    ErrorClassification.BAD_GATEWAY.value,
                    ErrorClassification.GATEWAY_TIMEOUT.value,
                ],
                fallback_tier=FallbackTier.SECONDARY,
                fallback_providers=[ProviderType.GOOGLE, ProviderType.ANTHROPIC],
                max_retries=2,
                exponential_backoff=True,
            ),
        ]
    
    def configure_local_model_provider(self, provider: AbstractProvider) -> None:
        """Configure local model provider for emergency fallback."""
        self.local_model_provider = provider
        self.local_models_available = True
        logger.info("Local model provider configured for emergency fallback")
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fallback statistics."""
        total_attempts = len(self.fallback_history)
        successful_fallbacks = sum(1 for attempt in self.fallback_history if attempt.success)
        
        # Statistics by tier
        tier_stats = {}
        for tier in FallbackTier:
            tier_attempts = [a for a in self.fallback_history if a.fallback_tier == tier]
            tier_successes = [a for a in tier_attempts if a.success]
            
            tier_stats[tier.value] = {
                "attempts": len(tier_attempts),
                "successes": len(tier_successes),
                "success_rate": len(tier_successes) / len(tier_attempts) if tier_attempts else 0,
                "avg_latency_ms": (
                    sum(a.latency_ms for a in tier_successes if a.latency_ms) / len(tier_successes)
                    if tier_successes else 0
                ),
            }
        
        # Circuit breaker states
        circuit_states = {
            provider.value: {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "success_count": breaker.success_count,
                "last_failure": breaker.last_failure_time.isoformat() if breaker.last_failure_time else None,
            }
            for provider, breaker in self.circuit_breakers.items()
        }
        
        return {
            "total_fallback_attempts": total_attempts,
            "successful_fallbacks": successful_fallbacks,
            "overall_success_rate": successful_fallbacks / total_attempts if total_attempts else 0,
            "tier_statistics": tier_stats,
            "circuit_breaker_states": circuit_states,
            "local_models_available": self.local_models_available,
            "fallback_rules_active": len([r for r in self.fallback_rules if r.enabled]),
        }
    
    def get_circuit_breaker_recommendations(self) -> List[str]:
        """Get recommendations based on circuit breaker states."""
        recommendations = []
        
        for provider_type, breaker in self.circuit_breakers.items():
            if breaker.state == CircuitState.OPEN:
                time_until_retry = breaker.timeout_seconds - (
                    datetime.now() - breaker.state_changed_time
                ).total_seconds()
                
                if time_until_retry > 0:
                    recommendations.append(
                        f"Provider {provider_type.value} circuit breaker is OPEN. "
                        f"Will retry in {time_until_retry:.0f} seconds."
                    )
                else:
                    recommendations.append(
                        f"Provider {provider_type.value} circuit breaker ready to test recovery."
                    )
            
            elif breaker.failure_count > breaker.failure_threshold * 0.7:
                recommendations.append(
                    f"Provider {provider_type.value} approaching failure threshold "
                    f"({breaker.failure_count}/{breaker.failure_threshold}). Monitor closely."
                )
        
        return recommendations