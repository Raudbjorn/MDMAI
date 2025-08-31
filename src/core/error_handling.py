"""
Comprehensive error handling system for MDMAI TTRPG Assistant.

This module provides hierarchical error classification, retry logic with exponential
backoff, circuit breaker pattern, and error recovery mechanisms.
"""

import asyncio
import functools
import logging
import random
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from typing_extensions import ParamSpec

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""

    CRITICAL = auto()  # System-breaking errors requiring immediate attention
    HIGH = auto()  # Service failures that impact functionality
    MEDIUM = auto()  # Recoverable errors that may retry
    LOW = auto()  # Minor issues that can be logged and ignored
    INFO = auto()  # Informational errors for debugging


class ErrorCategory(Enum):
    """Error categories for classification and routing."""

    SYSTEM = auto()  # System-level errors
    SERVICE = auto()  # External service errors
    DATABASE = auto()  # Database operation errors
    NETWORK = auto()  # Network connectivity errors
    VALIDATION = auto()  # Input validation errors
    AUTHENTICATION = auto()  # Auth/permission errors
    CONFIGURATION = auto()  # Configuration errors
    RESOURCE = auto()  # Resource availability errors
    BUSINESS_LOGIC = auto()  # Application logic errors
    MCP_PROTOCOL = auto()  # MCP protocol-specific errors


class BaseError(Exception):
    """Base exception class with enhanced error information."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        retry_after: Optional[int] = None,
    ) -> None:
        """
        Initialize base error with comprehensive metadata.

        Args:
            message: Human-readable error message
            severity: Error severity level
            category: Error category for classification
            error_code: Unique error code for tracking
            context: Additional context information
            recoverable: Whether error is recoverable
            retry_after: Seconds to wait before retry
        """
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.error_code = error_code or self._generate_error_code()
        self.context = context or {}
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.timestamp = datetime.utcnow()

    def _generate_error_code(self) -> str:
        """Generate unique error code based on category and timestamp."""
        timestamp = int(time.time() * 1000) % 100000
        return f"{self.category.name[:3]}-{self.severity.name[:3]}-{timestamp}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.name,
            "category": self.category.name,
            "context": self.context,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "timestamp": self.timestamp.isoformat(),
        }


# Hierarchical custom exceptions
class SystemError(BaseError):
    """System-level critical errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYSTEM,
            recoverable=False,
            **kwargs,
        )


class ServiceError(BaseError):
    """External service errors."""

    def __init__(self, message: str, service_name: str, **kwargs: Any) -> None:
        context = kwargs.pop("context", {})
        context["service_name"] = service_name
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SERVICE,
            context=context,
            **kwargs,
        )


class DatabaseError(BaseError):
    """Database operation errors."""

    def __init__(self, message: str, operation: str, **kwargs: Any) -> None:
        context = kwargs.pop("context", {})
        context["operation"] = operation
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATABASE,
            context=context,
            **kwargs,
        )


class NetworkError(BaseError):
    """Network connectivity errors."""

    def __init__(self, message: str, endpoint: str, **kwargs: Any) -> None:
        context = kwargs.pop("context", {})
        context["endpoint"] = endpoint
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            context=context,
            **kwargs,
        )


class ValidationError(BaseError):
    """Input validation errors."""

    def __init__(
        self, message: str, field: Optional[str] = None, **kwargs: Any
    ) -> None:
        context = kwargs.pop("context", {})
        if field:
            context["field"] = field
        super().__init__(
            message,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.VALIDATION,
            context=context,
            recoverable=False,
            **kwargs,
        )


class AuthenticationError(BaseError):
    """Authentication and authorization errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AUTHENTICATION,
            recoverable=False,
            **kwargs,
        )


class ConfigurationError(BaseError):
    """Configuration errors."""

    def __init__(self, message: str, config_key: str, **kwargs: Any) -> None:
        context = kwargs.pop("context", {})
        context["config_key"] = config_key
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.CONFIGURATION,
            context=context,
            recoverable=False,
            **kwargs,
        )


class ResourceError(BaseError):
    """Resource availability errors."""

    def __init__(self, message: str, resource_type: str, **kwargs: Any) -> None:
        context = kwargs.pop("context", {})
        context["resource_type"] = resource_type
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.RESOURCE,
            context=context,
            **kwargs,
        )


class MCPProtocolError(BaseError):
    """MCP protocol-specific errors."""

    def __init__(self, message: str, tool_name: Optional[str] = None, **kwargs: Any) -> None:
        context = kwargs.pop("context", {})
        if tool_name:
            context["tool_name"] = tool_name
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MCP_PROTOCOL,
            context=context,
            **kwargs,
        )


class CircuitBreakerError(ServiceError):
    """Circuit breaker open error."""

    def __init__(self, service_name: str, reset_time: datetime, **kwargs: Any) -> None:
        message = f"Circuit breaker open for service: {service_name}"
        retry_after = int((reset_time - datetime.utcnow()).total_seconds())
        super().__init__(
            message,
            service_name=service_name,
            recoverable=True,
            retry_after=retry_after,
            **kwargs,
        )


class RetryConfig:
    """Configuration for retry logic."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: Optional[List[Type[Exception]]] = None,
        ignore_on: Optional[List[Type[Exception]]] = None,
    ) -> None:
        """
        Initialize retry configuration.

        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Add random jitter to delays
            retry_on: List of exceptions to retry on
            ignore_on: List of exceptions to not retry on
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on = retry_on or [Exception]
        self.ignore_on = ignore_on or []

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number with exponential backoff."""
        delay = min(self.initial_delay * (self.exponential_base ** attempt), self.max_delay)
        if self.jitter:
            delay = delay * (0.5 + random.random())
        return delay

    def should_retry(self, exception: Exception) -> bool:
        """Determine if exception should trigger retry."""
        if any(isinstance(exception, exc_type) for exc_type in self.ignore_on):
            return False
        if isinstance(exception, BaseError) and not exception.recoverable:
            return False
        return any(isinstance(exception, exc_type) for exc_type in self.retry_on)


def _handle_retry_attempt(
    config: RetryConfig,
    func_name: str,
    exception: Exception,
    attempt: int,
) -> tuple[bool, Optional[float]]:
    """
    Handle retry attempt logic.
    
    Args:
        config: Retry configuration
        func_name: Function name for logging
        exception: Exception that occurred
        attempt: Current attempt number (0-indexed)
    
    Returns:
        Tuple of (should_continue, delay_seconds)
    """
    if not config.should_retry(exception):
        logger.error(
            f"Non-retryable error in {func_name}: {exception}",
            extra={"attempt": attempt + 1, "function": func_name},
        )
        return False, None
    
    if attempt < config.max_attempts - 1:
        delay = config.calculate_delay(attempt)
        logger.warning(
            f"Retry attempt {attempt + 1}/{config.max_attempts} "
            f"for {func_name} after {delay:.2f}s delay. Error: {exception}",
            extra={
                "attempt": attempt + 1,
                "max_attempts": config.max_attempts,
                "delay": delay,
                "function": func_name,
            },
        )
        return True, delay
    else:
        logger.error(
            f"Max retries ({config.max_attempts}) exceeded for {func_name}",
            extra={
                "max_attempts": config.max_attempts,
                "function": func_name,
            },
        )
        return True, None


def retry(config: Optional[RetryConfig] = None) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for retry logic with exponential backoff.

    Args:
        config: Retry configuration

    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    should_continue, delay = _handle_retry_attempt(
                        config, func.__name__, e, attempt
                    )
                    
                    if not should_continue:
                        raise
                    
                    if delay is not None:
                        time.sleep(delay)
            
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected error in retry logic for {func.__name__}")

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)  # type: ignore
                except Exception as e:
                    last_exception = e
                    should_continue, delay = _handle_retry_attempt(
                        config, func.__name__, e, attempt
                    )
                    
                    if not should_continue:
                        raise
                    
                    if delay is not None:
                        await asyncio.sleep(delay)
            
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected error in retry logic for {func.__name__}")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()  # Normal operation
    OPEN = auto()  # Blocking calls
    HALF_OPEN = auto()  # Testing recovery


class CircuitBreaker:
    """Circuit breaker pattern implementation for external service calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to monitor
            name: Circuit breaker name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name or "CircuitBreaker"
        
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = CircuitBreakerState.CLOSED
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self._state != CircuitBreakerState.OPEN:
            return False
        
        if self._last_failure_time is None:
            return True
        
        return (datetime.utcnow() - self._last_failure_time).total_seconds() >= self.recovery_timeout

    async def call(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        async with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                else:
                    reset_time = self._last_failure_time + timedelta(seconds=self.recovery_timeout)
                    raise CircuitBreakerError(self.name, reset_time)

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            async with self._lock:
                if self._state == CircuitBreakerState.HALF_OPEN:
                    self._state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
                    logger.info(f"Circuit breaker {self.name} recovered, entering CLOSED state")
            
            return result
            
        except self.expected_exception as e:
            async with self._lock:
                self._failure_count += 1
                self._last_failure_time = datetime.utcnow()
                
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitBreakerState.OPEN
                    logger.error(
                        f"Circuit breaker {self.name} opened after {self._failure_count} failures",
                        extra={
                            "failure_count": self._failure_count,
                            "threshold": self.failure_threshold,
                        },
                    )
            raise

    def reset(self) -> None:
        """Manually reset circuit breaker."""
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        logger.info(f"Circuit breaker {self.name} manually reset")


@contextmanager
def error_handler(
    default_return: Any = None,
    log_errors: bool = True,
    reraise: bool = False,
    error_callback: Optional[Callable[[Exception], None]] = None,
) -> Any:
    """
    Context manager for standardized error handling.

    Args:
        default_return: Default value to return on error
        log_errors: Whether to log errors
        reraise: Whether to reraise exceptions
        error_callback: Optional callback for error handling

    Yields:
        Context for error handling
    """
    try:
        yield
    except Exception as e:
        if log_errors:
            logger.error(f"Error in context: {e}", exc_info=True)
        
        if error_callback:
            error_callback(e)
        
        if reraise:
            raise
        
        return default_return


@asynccontextmanager
async def async_error_handler(
    default_return: Any = None,
    log_errors: bool = True,
    reraise: bool = False,
    error_callback: Optional[Callable[[Exception], None]] = None,
) -> Any:
    """
    Async context manager for standardized error handling.

    Args:
        default_return: Default value to return on error
        log_errors: Whether to log errors
        reraise: Whether to reraise exceptions
        error_callback: Optional callback for error handling

    Yields:
        Context for error handling
    """
    try:
        yield
    except Exception as e:
        if log_errors:
            logger.error(f"Error in async context: {e}", exc_info=True)
        
        if error_callback:
            if asyncio.iscoroutinefunction(error_callback):
                await error_callback(e)
            else:
                error_callback(e)
        
        if reraise:
            raise
        
        return default_return


class ErrorAggregator:
    """Aggregate and analyze errors for pattern detection."""

    def __init__(self, window_size: int = 100, time_window: float = 3600.0) -> None:
        """
        Initialize error aggregator.

        Args:
            window_size: Maximum number of errors to track
            time_window: Time window in seconds for error analysis
        """
        self.window_size = window_size
        self.time_window = time_window
        self.errors: List[BaseError] = []
        self._lock = asyncio.Lock()

    async def add_error(self, error: BaseError) -> None:
        """Add error to aggregator for analysis."""
        async with self._lock:
            self.errors.append(error)
            
            # Remove old errors outside time window
            cutoff_time = datetime.utcnow() - timedelta(seconds=self.time_window)
            self.errors = [e for e in self.errors if e.timestamp > cutoff_time]
            
            # Limit to window size
            if len(self.errors) > self.window_size:
                self.errors = self.errors[-self.window_size:]

    async def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics and patterns."""
        async with self._lock:
            if not self.errors:
                return {
                    "total_errors": 0,
                    "error_rate": 0.0,
                    "categories": {},
                    "severities": {},
                    "top_errors": [],
                }
            
            total = len(self.errors)
            time_span = (self.errors[-1].timestamp - self.errors[0].timestamp).total_seconds()
            error_rate = total / max(time_span, 1.0)
            
            # Count by category
            categories: Dict[str, int] = {}
            for error in self.errors:
                categories[error.category.name] = categories.get(error.category.name, 0) + 1
            
            # Count by severity
            severities: Dict[str, int] = {}
            for error in self.errors:
                severities[error.severity.name] = severities.get(error.severity.name, 0) + 1
            
            # Find top error codes
            error_codes: Dict[str, int] = {}
            for error in self.errors:
                error_codes[error.error_code] = error_codes.get(error.error_code, 0) + 1
            
            top_errors = sorted(error_codes.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "total_errors": total,
                "error_rate": error_rate,
                "categories": categories,
                "severities": severities,
                "top_errors": top_errors,
                "time_span_seconds": time_span,
            }

    async def detect_patterns(self) -> List[Dict[str, Any]]:
        """Detect error patterns and anomalies."""
        patterns = []
        
        async with self._lock:
            if len(self.errors) < 10:
                return patterns
            
            # Detect error bursts
            burst_threshold = 5
            burst_window = 60.0  # seconds
            
            for i in range(len(self.errors) - burst_threshold + 1):
                window_errors = self.errors[i:i + burst_threshold]
                time_diff = (window_errors[-1].timestamp - window_errors[0].timestamp).total_seconds()
                
                if time_diff < burst_window:
                    patterns.append({
                        "type": "error_burst",
                        "count": burst_threshold,
                        "duration_seconds": time_diff,
                        "start_time": window_errors[0].timestamp.isoformat(),
                        "categories": list({e.category.name for e in window_errors}),
                    })
            
            # Detect repeated errors
            error_messages: Dict[str, int] = {}
            for error in self.errors:
                error_messages[error.message] = error_messages.get(error.message, 0) + 1
            
            for message, count in error_messages.items():
                if count > 3:
                    patterns.append({
                        "type": "repeated_error",
                        "message": message,
                        "count": count,
                        "percentage": (count / len(self.errors)) * 100,
                    })
        
        return patterns


# Global error aggregator instance
error_aggregator = ErrorAggregator()