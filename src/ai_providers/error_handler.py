"""Unified error handling and retry logic for AI providers."""

import asyncio
import random
import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from structlog import get_logger

from .models import ProviderType

logger = get_logger(__name__)


class AIProviderError(Exception):
    """Base exception for AI provider errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        provider_type: Optional[ProviderType] = None,
        retryable: bool = False,
        retry_after: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.provider_type = provider_type
        self.retryable = retryable
        self.retry_after = retry_after
        self.details = details or {}
        self.timestamp = datetime.now()


class RateLimitError(AIProviderError):
    """Rate limit exceeded error."""
    
    def __init__(self, message: str, retry_after: float, **kwargs):
        super().__init__(message, retryable=True, retry_after=retry_after, **kwargs)


class QuotaExceededError(AIProviderError):
    """API quota exceeded error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=False, **kwargs)


class AuthenticationError(AIProviderError):
    """Authentication/authorization error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=False, **kwargs)


class ModelNotFoundError(AIProviderError):
    """Model not found or unavailable error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=False, **kwargs)


class InvalidRequestError(AIProviderError):
    """Invalid request parameters error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=False, **kwargs)


class ServiceUnavailableError(AIProviderError):
    """Service temporarily unavailable error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=True, **kwargs)


class TimeoutError(AIProviderError):
    """Request timeout error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=True, **kwargs)


class BudgetExceededError(AIProviderError):
    """Budget limit exceeded error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=False, **kwargs)


class RetryStrategy(Enum):
    """Retry strategy types."""
    
    NONE = "none"
    FIXED = "fixed"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    JITTERED_EXPONENTIAL = "jittered_exponential"


class CircuitBreaker:
    """Circuit breaker for handling provider failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self._state = "closed"  # closed, open, half_open
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
    
    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        if self._state == "open":
            if self._last_failure_time:
                time_since_failure = (datetime.now() - self._last_failure_time).total_seconds()
                if time_since_failure >= self.recovery_timeout:
                    self._state = "half_open"
                    self._half_open_calls = 0
        return self._state
    
    def call_succeeded(self) -> None:
        """Record a successful call."""
        if self.state == "half_open":
            self._success_count += 1
            self._half_open_calls += 1
            
            if self._half_open_calls >= self.half_open_max_calls:
                # Enough successful calls, close the circuit
                self._state = "closed"
                self._failure_count = 0
                self._success_count = 0
                logger.info("Circuit breaker closed after successful recovery")
        
        elif self.state == "closed":
            self._failure_count = max(0, self._failure_count - 1)
    
    def call_failed(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        
        if self.state == "closed":
            if self._failure_count >= self.failure_threshold:
                self._state = "open"
                logger.warning(
                    "Circuit breaker opened",
                    failures=self._failure_count,
                    threshold=self.failure_threshold,
                )
        
        elif self.state == "half_open":
            # Failed while testing recovery, reopen
            self._state = "open"
            logger.warning("Circuit breaker reopened during recovery test")
    
    def is_open(self) -> bool:
        """Check if circuit is open (failing)."""
        return self.state == "open"
    
    def is_closed(self) -> bool:
        """Check if circuit is closed (working)."""
        return self.state == "closed"
    
    def reset(self) -> None:
        """Reset the circuit breaker."""
        self._state = "closed"
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0


class ErrorHandler:
    """Unified error handler with retry logic and circuit breaker pattern."""
    
    def __init__(self):
        self._circuit_breakers: Dict[ProviderType, CircuitBreaker] = {}
        self._error_history: List[AIProviderError] = []
        self._retry_configs = {
            RetryStrategy.FIXED: {"delay": 1.0, "max_retries": 3},
            RetryStrategy.EXPONENTIAL_BACKOFF: {
                "initial_delay": 1.0,
                "max_delay": 60.0,
                "multiplier": 2.0,
                "max_retries": 5,
            },
            RetryStrategy.LINEAR_BACKOFF: {
                "initial_delay": 1.0,
                "increment": 2.0,
                "max_delay": 30.0,
                "max_retries": 5,
            },
            RetryStrategy.JITTERED_EXPONENTIAL: {
                "initial_delay": 1.0,
                "max_delay": 60.0,
                "multiplier": 2.0,
                "jitter_range": 0.3,
                "max_retries": 5,
            },
        }
    
    def get_circuit_breaker(self, provider_type: ProviderType) -> CircuitBreaker:
        """Get or create circuit breaker for a provider.
        
        Args:
            provider_type: Provider type
            
        Returns:
            Circuit breaker instance
        """
        if provider_type not in self._circuit_breakers:
            self._circuit_breakers[provider_type] = CircuitBreaker()
        return self._circuit_breakers[provider_type]
    
    async def handle_error(
        self,
        error: Exception,
        provider_type: Optional[ProviderType] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AIProviderError:
        """Handle and classify an error.
        
        Args:
            error: The error to handle
            provider_type: Provider that generated the error
            context: Additional context information
            
        Returns:
            Classified AIProviderError
        """
        # Convert to AIProviderError if needed
        if isinstance(error, AIProviderError):
            ai_error = error
        else:
            ai_error = self._classify_error(error, provider_type, context)
        
        # Record error
        self._error_history.append(ai_error)
        
        # Update circuit breaker if applicable
        if provider_type:
            breaker = self.get_circuit_breaker(provider_type)
            breaker.call_failed()
        
        logger.error(
            "AI provider error handled",
            error_type=type(ai_error).__name__,
            message=ai_error.message,
            provider=provider_type.value if provider_type else None,
            retryable=ai_error.retryable,
        )
        
        return ai_error
    
    async def retry_with_strategy(
        self,
        func: Callable,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        provider_type: Optional[ProviderType] = None,
        **kwargs,
    ) -> Any:
        """Execute function with retry strategy.
        
        Args:
            func: Async function to execute
            strategy: Retry strategy to use
            provider_type: Provider type for circuit breaking
            **kwargs: Arguments to pass to function
            
        Returns:
            Function result
            
        Raises:
            AIProviderError: If all retries fail
        """
        if strategy == RetryStrategy.NONE:
            return await func(**kwargs)
        
        config = self._retry_configs[strategy]
        max_retries = config["max_retries"]
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # Check circuit breaker
                if provider_type:
                    breaker = self.get_circuit_breaker(provider_type)
                    if breaker.is_open():
                        raise ServiceUnavailableError(
                            f"Circuit breaker open for {provider_type.value}",
                            provider_type=provider_type,
                        )
                
                # Execute function
                result = await func(**kwargs)
                
                # Record success
                if provider_type:
                    breaker.call_succeeded()
                
                return result
                
            except Exception as e:
                last_error = await self.handle_error(e, provider_type)
                
                # Check if retryable
                if not last_error.retryable or attempt >= max_retries:
                    raise last_error
                
                # Calculate delay
                delay = self._calculate_delay(strategy, attempt, config, last_error)
                
                logger.info(
                    "Retrying after error",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    delay=delay,
                    error=str(last_error),
                )
                
                await asyncio.sleep(delay)
        
        raise last_error or AIProviderError("All retry attempts failed")
    
    def _classify_error(
        self,
        error: Exception,
        provider_type: Optional[ProviderType] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AIProviderError:
        """Classify a generic error into specific AIProviderError.
        
        Args:
            error: Error to classify
            provider_type: Provider type
            context: Additional context
            
        Returns:
            Classified AIProviderError
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Check for common error patterns
        if "rate limit" in error_str or "too many requests" in error_str:
            retry_after = self._extract_retry_after(error_str)
            return RateLimitError(
                str(error),
                retry_after=retry_after or 60.0,
                provider_type=provider_type,
            )
        
        elif "quota" in error_str or "limit exceeded" in error_str:
            return QuotaExceededError(str(error), provider_type=provider_type)
        
        elif "unauthorized" in error_str or "api key" in error_str:
            return AuthenticationError(str(error), provider_type=provider_type)
        
        elif "model" in error_str and "not found" in error_str:
            return ModelNotFoundError(str(error), provider_type=provider_type)
        
        elif "invalid" in error_str or "bad request" in error_str:
            return InvalidRequestError(str(error), provider_type=provider_type)
        
        elif "timeout" in error_str or error_type == "TimeoutError":
            return TimeoutError(str(error), provider_type=provider_type)
        
        elif "service unavailable" in error_str or "503" in error_str:
            return ServiceUnavailableError(str(error), provider_type=provider_type)
        
        else:
            # Generic retryable error for network issues
            if error_type in ["ConnectionError", "HTTPError", "RequestException"]:
                return AIProviderError(
                    str(error),
                    provider_type=provider_type,
                    retryable=True,
                )
            
            # Non-retryable by default
            return AIProviderError(
                str(error),
                provider_type=provider_type,
                retryable=False,
            )
    
    def _calculate_delay(
        self,
        strategy: RetryStrategy,
        attempt: int,
        config: Dict[str, Any],
        error: Optional[AIProviderError] = None,
    ) -> float:
        """Calculate retry delay based on strategy.
        
        Args:
            strategy: Retry strategy
            attempt: Current attempt number
            config: Strategy configuration
            error: Error that triggered retry
            
        Returns:
            Delay in seconds
        """
        # Use error's retry_after if available
        if error and error.retry_after:
            return error.retry_after
        
        if strategy == RetryStrategy.FIXED:
            return config["delay"]
        
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config["initial_delay"] * (config["multiplier"] ** attempt)
            return min(delay, config["max_delay"])
        
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config["initial_delay"] + (config["increment"] * attempt)
            return min(delay, config["max_delay"])
        
        elif strategy == RetryStrategy.JITTERED_EXPONENTIAL:
            base_delay = config["initial_delay"] * (config["multiplier"] ** attempt)
            jitter_range = config["jitter_range"]
            jitter = random.uniform(-jitter_range, jitter_range) * base_delay
            delay = base_delay + jitter
            return min(max(0, delay), config["max_delay"])
        
        return 1.0  # Default fallback
    
    def _extract_retry_after(self, error_str: str) -> Optional[float]:
        """Extract retry-after value from error message.
        
        Args:
            error_str: Error message
            
        Returns:
            Retry after value in seconds, or None
        """
        # Look for patterns like "retry after 60 seconds"
        match = re.search(r"retry[\s\-_]*after[\s:]**([\d.]+)\s*(?:seconds?)?", error_str, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        return None
    
    def get_error_stats(
        self,
        provider_type: Optional[ProviderType] = None,
        time_window: Optional[timedelta] = None,
    ) -> Dict[str, Any]:
        """Get error statistics.
        
        Args:
            provider_type: Filter by provider
            time_window: Time window for stats
            
        Returns:
            Error statistics
        """
        now = datetime.now()
        cutoff_time = now - time_window if time_window else None
        
        filtered_errors = [
            e for e in self._error_history
            if (not provider_type or e.provider_type == provider_type)
            and (not cutoff_time or e.timestamp >= cutoff_time)
        ]
        
        if not filtered_errors:
            return {
                "total_errors": 0,
                "error_types": {},
                "providers": {},
            }
        
        # Aggregate stats
        error_types = {}
        providers = {}
        
        for error in filtered_errors:
            error_type = type(error).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            if error.provider_type:
                provider_name = error.provider_type.value
                providers[provider_name] = providers.get(provider_name, 0) + 1
        
        return {
            "total_errors": len(filtered_errors),
            "error_types": error_types,
            "providers": providers,
            "retryable_errors": sum(1 for e in filtered_errors if e.retryable),
            "non_retryable_errors": sum(1 for e in filtered_errors if not e.retryable),
            "recent_errors": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "type": type(e).__name__,
                    "message": e.message,
                    "provider": e.provider_type.value if e.provider_type else None,
                    "retryable": e.retryable,
                }
                for e in filtered_errors[-10:]  # Last 10 errors
            ],
        }
    
    def clear_error_history(self, older_than: Optional[timedelta] = None) -> int:
        """Clear error history.
        
        Args:
            older_than: Clear only errors older than this duration
            
        Returns:
            Number of errors cleared
        """
        if older_than:
            cutoff_time = datetime.now() - older_than
            original_count = len(self._error_history)
            self._error_history = [
                e for e in self._error_history
                if e.timestamp >= cutoff_time
            ]
            return original_count - len(self._error_history)
        else:
            count = len(self._error_history)
            self._error_history.clear()
            return count