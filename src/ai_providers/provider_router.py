"""
Provider router with fallback and circuit breaker for MDMAI TTRPG Assistant.

This module provides intelligent routing between AI providers with automatic
fallback, circuit breaker patterns, and load balancing.
"""

import logging
import asyncio
import time
from typing import Dict, Optional, List, Union, AsyncIterator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .base_provider import (
    BaseAIProvider, ProviderType, ProviderConfig, CompletionResponse,
    ProviderError, ProviderAuthenticationError, ProviderRateLimitError,
    ProviderTimeoutError, ProviderQuotaExceededError, ProviderInvalidRequestError,
    NoAvailableProvidersError
)
from .anthropic_provider_auth import AnthropicProvider
from .openai_provider_auth import OpenAIProvider
from .google_provider_auth import GoogleProvider

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, requests are blocked
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Number of failures to trigger open state
    recovery_timeout: int = 60  # Seconds to wait before trying half-open
    success_threshold: int = 3   # Successes needed in half-open to close
    timeout_threshold: float = 30.0  # Request timeout threshold


@dataclass
class ProviderStats:
    """Statistics for a provider."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    circuit_state: CircuitState = CircuitState.CLOSED
    circuit_opened_at: Optional[datetime] = None


class ProviderRouter:
    """
    Intelligent provider router with fallback and circuit breaker.
    
    Features:
    - Automatic provider fallback on failures
    - Circuit breaker pattern for failing providers
    - Load balancing based on response times
    - Provider health monitoring
    - Cost optimization routing
    - Request statistics tracking
    """
    
    def __init__(self, circuit_config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize provider router.
        
        Args:
            circuit_config: Configuration for circuit breaker behavior
        """
        self.providers: Dict[str, BaseAIProvider] = {}
        self.user_providers: Dict[str, Dict[str, str]] = {}  # user_id -> {provider_type -> provider_key}
        self.provider_stats: Dict[str, ProviderStats] = {}
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        
        # Default fallback order (can be customized per user)
        self.default_fallback_order = [
            ProviderType.ANTHROPIC,
            ProviderType.OPENAI,
            ProviderType.GOOGLE
        ]
        
        # Provider factory mapping
        self.provider_factories = {
            ProviderType.ANTHROPIC: AnthropicProvider,
            ProviderType.OPENAI: OpenAIProvider,
            ProviderType.GOOGLE: GoogleProvider
        }
        
        logger.info("ProviderRouter initialized with circuit breaker configuration")
    
    def register_provider(
        self, 
        user_id: str, 
        provider_type: ProviderType,
        config: ProviderConfig,
        is_primary: bool = False
    ) -> str:
        """
        Register a provider for a user.
        
        Args:
            user_id: User identifier
            provider_type: Type of provider
            config: Provider configuration
            is_primary: Whether this is the primary provider for the user
            
        Returns:
            str: Provider key for tracking
            
        Raises:
            ValueError: If provider type is not supported
        """
        if provider_type not in self.provider_factories:
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
        # Create provider instance
        provider_class = self.provider_factories[provider_type]
        provider = provider_class(config)
        
        # Generate provider key
        provider_key = f"{user_id}:{provider_type.value}"
        
        # Register provider
        self.providers[provider_key] = provider
        self.provider_stats[provider_key] = ProviderStats()
        
        # Track user's providers
        if user_id not in self.user_providers:
            self.user_providers[user_id] = {}
        
        self.user_providers[user_id][provider_type.value] = provider_key
        
        logger.info(f"Registered {provider_type.value} provider for user {user_id} (primary: {is_primary})")
        return provider_key
    
    def unregister_provider(self, user_id: str, provider_type: ProviderType) -> bool:
        """
        Unregister a provider for a user.
        
        Args:
            user_id: User identifier
            provider_type: Type of provider to remove
            
        Returns:
            bool: True if provider was removed
        """
        provider_key = f"{user_id}:{provider_type.value}"
        
        if provider_key in self.providers:
            del self.providers[provider_key]
            del self.provider_stats[provider_key]
            
            if user_id in self.user_providers:
                self.user_providers[user_id].pop(provider_type.value, None)
            
            logger.info(f"Unregistered {provider_type.value} provider for user {user_id}")
            return True
        
        return False
    
    async def get_completion(
        self, 
        user_id: str, 
        messages: List[Dict],
        preferred_provider: Optional[ProviderType] = None,
        fallback_order: Optional[List[ProviderType]] = None,
        stream: bool = False,
        cost_optimization: bool = True
    ) -> Union[CompletionResponse, AsyncIterator[str]]:
        """
        Get completion with automatic fallback and circuit breaker.
        
        Args:
            user_id: User identifier
            messages: Chat messages
            preferred_provider: Preferred provider to try first
            fallback_order: Custom fallback order
            stream: Whether to stream the response
            cost_optimization: Whether to optimize for cost
            
        Returns:
            CompletionResponse or AsyncIterator[str] for streaming
            
        Raises:
            NoAvailableProvidersError: If all providers fail
        """
        # Determine provider order
        providers_to_try = self._get_provider_order(
            user_id, preferred_provider, fallback_order, cost_optimization
        )
        
        if not providers_to_try:
            raise NoAvailableProvidersError(
                f"No providers configured for user {user_id}"
            )
        
        last_error = None
        
        # Try each provider in order
        for provider_type, is_fallback in providers_to_try:
            provider_key = f"{user_id}:{provider_type.value}"
            
            if provider_key not in self.providers:
                logger.debug(f"Provider {provider_type.value} not configured for user {user_id}")
                continue
            
            # Check circuit breaker
            if not self._can_use_provider(provider_key):
                logger.debug(f"Circuit breaker open for provider {provider_key}")
                continue
            
            provider = self.providers[provider_key]
            stats = self.provider_stats[provider_key]
            
            try:
                start_time = time.time()
                
                if stream:
                    # Return streaming iterator
                    return self._stream_with_monitoring(
                        provider, provider_key, messages, start_time
                    )
                else:
                    # Get non-streaming response
                    content_chunks = []
                    async for chunk in provider.complete(messages, stream=False):
                        content_chunks.append(chunk)
                    
                    content = "".join(content_chunks)
                    elapsed = time.time() - start_time
                    
                    # Update statistics
                    self._record_success(provider_key, elapsed)
                    
                    return CompletionResponse(
                        content=content,
                        provider=provider_type,
                        is_fallback=is_fallback
                    )
                    
            except ProviderRateLimitError as e:
                logger.warning(f"Provider {provider_type.value} rate limited: {e}")
                self._record_failure(provider_key, e)
                last_error = e
                continue
                
            except ProviderAuthenticationError as e:
                logger.error(f"Provider {provider_type.value} auth failed: {e}")
                self._record_failure(provider_key, e)
                # Remove failed auth provider from rotation
                self.unregister_provider(user_id, provider_type)
                last_error = e
                continue
                
            except ProviderTimeoutError as e:
                logger.warning(f"Provider {provider_type.value} timed out: {e}")
                self._record_failure(provider_key, e)
                last_error = e
                continue
                
            except ProviderQuotaExceededError as e:
                logger.error(f"Provider {provider_type.value} quota exceeded: {e}")
                self._record_failure(provider_key, e)
                last_error = e
                continue
                
            except Exception as e:
                logger.exception(f"Unexpected error with provider {provider_type.value}: {e}")
                self._record_failure(provider_key, ProviderError(str(e)))
                last_error = e
                continue
        
        # All providers failed
        failed_providers = [provider_type.value for provider_type, _ in providers_to_try]
        raise NoAvailableProvidersError(
            f"All providers failed for user {user_id}: {last_error}",
            failed_providers=failed_providers
        )
    
    async def validate_all_providers(self, user_id: str) -> Dict[str, bool]:
        """
        Validate all providers for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict[str, bool]: Provider validation results
        """
        results = {}
        
        if user_id not in self.user_providers:
            return results
        
        for provider_type_str, provider_key in self.user_providers[user_id].items():
            if provider_key in self.providers:
                provider = self.providers[provider_key]
                try:
                    is_valid = await provider.validate_credentials()
                    results[provider_type_str] = is_valid
                except Exception as e:
                    logger.error(f"Validation failed for {provider_type_str}: {e}")
                    results[provider_type_str] = False
            else:
                results[provider_type_str] = False
        
        return results
    
    async def health_check_all(self, user_id: str) -> Dict[str, bool]:
        """
        Run health checks on all providers for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict[str, bool]: Health check results
        """
        results = {}
        
        if user_id not in self.user_providers:
            return results
        
        # Run health checks in parallel
        tasks = {}
        for provider_type_str, provider_key in self.user_providers[user_id].items():
            if provider_key in self.providers:
                provider = self.providers[provider_key]
                tasks[provider_type_str] = asyncio.create_task(provider.health_check())
        
        # Wait for all health checks
        for provider_type_str, task in tasks.items():
            try:
                results[provider_type_str] = await task
            except Exception as e:
                logger.error(f"Health check failed for {provider_type_str}: {e}")
                results[provider_type_str] = False
        
        return results
    
    def get_provider_stats(self, user_id: str) -> Dict[str, Dict]:
        """
        Get statistics for all providers for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict[str, Dict]: Provider statistics
        """
        stats = {}
        
        if user_id not in self.user_providers:
            return stats
        
        for provider_type_str, provider_key in self.user_providers[user_id].items():
            if provider_key in self.provider_stats:
                provider_stats = self.provider_stats[provider_key]
                stats[provider_type_str] = {
                    'total_requests': provider_stats.total_requests,
                    'successful_requests': provider_stats.successful_requests,
                    'failed_requests': provider_stats.failed_requests,
                    'success_rate': (provider_stats.successful_requests / max(provider_stats.total_requests, 1)) * 100,
                    'average_response_time': provider_stats.average_response_time,
                    'last_success': provider_stats.last_success.isoformat() if provider_stats.last_success else None,
                    'last_failure': provider_stats.last_failure.isoformat() if provider_stats.last_failure else None,
                    'circuit_state': provider_stats.circuit_state.value,
                    'consecutive_failures': provider_stats.consecutive_failures
                }
        
        return stats
    
    def _get_provider_order(
        self,
        user_id: str,
        preferred_provider: Optional[ProviderType],
        fallback_order: Optional[List[ProviderType]],
        cost_optimization: bool
    ) -> List[tuple[ProviderType, bool]]:
        """
        Determine the order of providers to try.
        
        Returns:
            List[tuple[ProviderType, bool]]: List of (provider_type, is_fallback) tuples
        """
        if user_id not in self.user_providers:
            return []
        
        available_providers = [
            ProviderType(provider_type) 
            for provider_type in self.user_providers[user_id].keys()
        ]
        
        if not available_providers:
            return []
        
        providers_to_try = []
        
        # Add preferred provider first
        if preferred_provider and preferred_provider in available_providers:
            providers_to_try.append((preferred_provider, False))
            available_providers.remove(preferred_provider)
        
        # Add remaining providers in fallback order
        fallback_list = fallback_order or self.default_fallback_order
        
        if cost_optimization:
            # Sort by cost efficiency (lower cost first)
            available_providers.sort(key=lambda p: self._get_cost_score(user_id, p))
        
        for provider_type in fallback_list:
            if provider_type in available_providers:
                providers_to_try.append((provider_type, True))
                available_providers.remove(provider_type)
        
        # Add any remaining providers
        for provider_type in available_providers:
            providers_to_try.append((provider_type, True))
        
        return providers_to_try
    
    def _get_cost_score(self, user_id: str, provider_type: ProviderType) -> float:
        """
        Get cost score for provider (lower is better).
        
        This is a simplified cost scoring. In practice, you'd consider:
        - Model pricing
        - Expected token usage
        - Success rates
        - Response times
        """
        provider_key = f"{user_id}:{provider_type.value}"
        if provider_key in self.provider_stats:
            stats = self.provider_stats[provider_key]
            # Factor in success rate and response time
            success_rate = stats.successful_requests / max(stats.total_requests, 1)
            response_penalty = min(stats.average_response_time / 5.0, 2.0)  # Cap at 2x penalty
            return (1.0 - success_rate) + response_penalty
        
        # Default scores based on typical pricing
        cost_scores = {
            ProviderType.ANTHROPIC: 2.0,  # Medium cost, high quality
            ProviderType.OPENAI: 3.0,     # Higher cost
            ProviderType.GOOGLE: 1.0      # Lower cost
        }
        
        return cost_scores.get(provider_type, 5.0)
    
    def _can_use_provider(self, provider_key: str) -> bool:
        """Check if provider can be used based on circuit breaker state."""
        if provider_key not in self.provider_stats:
            return True
        
        stats = self.provider_stats[provider_key]
        
        if stats.circuit_state == CircuitState.CLOSED:
            return True
        elif stats.circuit_state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if (stats.circuit_opened_at and 
                datetime.now() - stats.circuit_opened_at > timedelta(seconds=self.circuit_config.recovery_timeout)):
                stats.circuit_state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker for {provider_key} moved to HALF_OPEN")
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def _record_success(self, provider_key: str, response_time: float):
        """Record a successful request."""
        stats = self.provider_stats[provider_key]
        
        stats.total_requests += 1
        stats.successful_requests += 1
        stats.consecutive_successes += 1
        stats.consecutive_failures = 0
        stats.last_success = datetime.now()
        
        # Update average response time
        if stats.average_response_time == 0:
            stats.average_response_time = response_time
        else:
            # Exponential moving average
            stats.average_response_time = (0.8 * stats.average_response_time) + (0.2 * response_time)
        
        # Circuit breaker logic
        if stats.circuit_state == CircuitState.HALF_OPEN:
            if stats.consecutive_successes >= self.circuit_config.success_threshold:
                stats.circuit_state = CircuitState.CLOSED
                logger.info(f"Circuit breaker for {provider_key} moved to CLOSED")
    
    def _record_failure(self, provider_key: str, error: Exception):
        """Record a failed request."""
        stats = self.provider_stats[provider_key]
        
        stats.total_requests += 1
        stats.failed_requests += 1
        stats.consecutive_failures += 1
        stats.consecutive_successes = 0
        stats.last_failure = datetime.now()
        
        # Circuit breaker logic
        if (stats.circuit_state == CircuitState.CLOSED and 
            stats.consecutive_failures >= self.circuit_config.failure_threshold):
            stats.circuit_state = CircuitState.OPEN
            stats.circuit_opened_at = datetime.now()
            logger.warning(f"Circuit breaker for {provider_key} moved to OPEN after {stats.consecutive_failures} failures")
        elif stats.circuit_state == CircuitState.HALF_OPEN:
            stats.circuit_state = CircuitState.OPEN
            stats.circuit_opened_at = datetime.now()
            logger.warning(f"Circuit breaker for {provider_key} moved back to OPEN")
    
    async def _stream_with_monitoring(
        self, 
        provider: BaseAIProvider, 
        provider_key: str, 
        messages: List[Dict],
        start_time: float
    ) -> AsyncIterator[str]:
        """Stream response while monitoring for success/failure."""
        try:
            chunk_count = 0
            async for chunk in provider.complete(messages, stream=True):
                chunk_count += 1
                yield chunk
            
            # Record success if we got any chunks
            if chunk_count > 0:
                elapsed = time.time() - start_time
                self._record_success(provider_key, elapsed)
        except Exception as e:
            self._record_failure(provider_key, e)
            raise