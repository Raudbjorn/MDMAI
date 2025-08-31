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
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import IntEnum, StrEnum, auto
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

from typing_extensions import ParamSpec, Self

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class ErrorSeverity(IntEnum):
    """Error severity levels for classification and handling."""

    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

    def __str__(self) -> str:
        return self.name


class ErrorCategory(StrEnum):
    """Error categories for classification and routing."""

    SYSTEM = auto()
    SERVICE = auto()
    DATABASE = auto()
    NETWORK = auto()
    VALIDATION = auto()
    AUTHENTICATION = auto()
    CONFIGURATION = auto()
    RESOURCE = auto()
    BUSINESS_LOGIC = auto()
    MCP_PROTOCOL = auto()


@dataclass(frozen=True)
class ErrorContext:
    """Immutable error context information."""

    error_code: str
    severity: ErrorSeverity
    category: ErrorCategory
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recoverable: bool = True
    retry_after: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_code": self.error_code,
            "severity": self.severity.name,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "metadata": self.metadata,
        }


class BaseError(Exception):
    """Base exception class with enhanced error information."""

    def __init__(
        self,
        message: str,
        *,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        retry_after: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_context = ErrorContext(
            error_code=error_code or self._generate_error_code(category, severity),
            severity=severity,
            category=category,
            recoverable=recoverable,
            retry_after=retry_after,
            metadata=context or {},
        )

    @staticmethod
    def _generate_error_code(category: ErrorCategory, severity: ErrorSeverity) -> str:
        """Generate unique error code based on category and timestamp."""
        timestamp = int(time.time() * 1000) % 100000
        return f"{category.value[:3].upper()}-{severity.name[:3]}-{timestamp}"

    @property
    def severity(self) -> ErrorSeverity:
        """Get error severity."""
        return self.error_context.severity

    @property
    def category(self) -> ErrorCategory:
        """Get error category."""
        return self.error_context.category

    @property
    def recoverable(self) -> bool:
        """Check if error is recoverable."""
        return self.error_context.recoverable

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "message": self.message,
            **self.error_context.to_dict(),
        }


# Specialized error classes with explicit definitions for better clarity
class SystemError(BaseError):
    """System-level critical error."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            severity=kwargs.pop("severity", ErrorSeverity.CRITICAL),
            category=kwargs.pop("category", ErrorCategory.SYSTEM),
            recoverable=kwargs.pop("recoverable", False),
            **kwargs
        )


class ServiceError(BaseError):
    """Service-level error."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            severity=kwargs.pop("severity", ErrorSeverity.HIGH),
            category=kwargs.pop("category", ErrorCategory.SERVICE),
            recoverable=kwargs.pop("recoverable", True),
            **kwargs
        )


class DatabaseError(BaseError):
    """Database operation error."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            severity=kwargs.pop("severity", ErrorSeverity.HIGH),
            category=kwargs.pop("category", ErrorCategory.DATABASE),
            recoverable=kwargs.pop("recoverable", True),
            **kwargs
        )


class NetworkError(BaseError):
    """Network communication error."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            severity=kwargs.pop("severity", ErrorSeverity.MEDIUM),
            category=kwargs.pop("category", ErrorCategory.NETWORK),
            recoverable=kwargs.pop("recoverable", True),
            **kwargs
        )


class ValidationError(BaseError):
    """Input validation error."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            severity=kwargs.pop("severity", ErrorSeverity.LOW),
            category=kwargs.pop("category", ErrorCategory.VALIDATION),
            recoverable=kwargs.pop("recoverable", False),
            **kwargs
        )


class AuthenticationError(BaseError):
    """Authentication/authorization error."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            severity=kwargs.pop("severity", ErrorSeverity.HIGH),
            category=kwargs.pop("category", ErrorCategory.AUTHENTICATION),
            recoverable=kwargs.pop("recoverable", False),
            **kwargs
        )


class ConfigurationError(BaseError):
    """Configuration error."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            severity=kwargs.pop("severity", ErrorSeverity.CRITICAL),
            category=kwargs.pop("category", ErrorCategory.CONFIGURATION),
            recoverable=kwargs.pop("recoverable", False),
            **kwargs
        )


class ResourceError(BaseError):
    """Resource availability error."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            severity=kwargs.pop("severity", ErrorSeverity.MEDIUM),
            category=kwargs.pop("category", ErrorCategory.RESOURCE),
            recoverable=kwargs.pop("recoverable", True),
            **kwargs
        )


class MCPProtocolError(BaseError):
    """MCP protocol error."""
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            severity=kwargs.pop("severity", ErrorSeverity.HIGH),
            category=kwargs.pop("category", ErrorCategory.MCP_PROTOCOL),
            recoverable=kwargs.pop("recoverable", True),
            **kwargs
        )


class CircuitBreakerError(ServiceError):
    """Circuit breaker open error."""

    def __init__(
        self, service_name: str, reset_time: datetime, **kwargs: Any
    ) -> None:
        retry_after = int((reset_time - datetime.now(timezone.utc)).total_seconds())
        super().__init__(
            f"Circuit breaker open for service: {service_name}",
            retry_after=max(0, retry_after),
            **kwargs,
        )


@dataclass
class RetryConfig:
    """Configuration for retry logic with sensible defaults."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    ignore_on: List[Type[Exception]] = field(default_factory=list)

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number with exponential backoff."""
        delay = min(
            self.initial_delay * (self.exponential_base**attempt), self.max_delay
        )
        if self.jitter:
            delay *= 0.5 + random.random()
        return delay

    def should_retry(self, exception: Exception) -> bool:
        """Determine if exception should trigger retry using pattern matching."""
        match exception:
            case BaseError() as e if not e.recoverable:
                return False
            case _ if any(isinstance(exception, exc_type) for exc_type in self.ignore_on):
                return False
            case _ if any(isinstance(exception, exc_type) for exc_type in self.retry_on):
                return True
            case _:
                return False


def retry(
    config: Optional[RetryConfig] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for retry logic with exponential backoff.

    Supports both sync and async functions automatically.
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if not config.should_retry(e) or attempt >= config.max_attempts - 1:
                        logger.error(
                            f"Non-retryable error or max attempts reached in {func.__name__}: {e}"
                        )
                        raise

                    delay = config.calculate_delay(attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_attempts} for {func.__name__} "
                        f"after {delay:.2f}s. Error: {e}"
                    )
                    time.sleep(delay)

            raise last_exception or RuntimeError(f"Unexpected retry error in {func.__name__}")

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)  # type: ignore
                except Exception as e:
                    last_exception = e
                    if not config.should_retry(e) or attempt >= config.max_attempts - 1:
                        logger.error(
                            f"Non-retryable error or max attempts reached in {func.__name__}: {e}"
                        )
                        raise

                    delay = config.calculate_delay(attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_attempts} for {func.__name__} "
                        f"after {delay:.2f}s. Error: {e}"
                    )
                    await asyncio.sleep(delay)

            raise last_exception or RuntimeError(f"Unexpected retry error in {func.__name__}")

        return async_wrapper if is_async else sync_wrapper  # type: ignore

    return decorator


class CircuitBreakerState(StrEnum):
    """Circuit breaker states."""

    CLOSED = auto()  # Normal operation
    OPEN = auto()  # Blocking calls
    HALF_OPEN = auto()  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker pattern implementation for external service calls."""

    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception
    _state: CircuitBreakerState = field(default=CircuitBreakerState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: Optional[datetime] = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self._state == CircuitBreakerState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self._state != CircuitBreakerState.OPEN:
            return False

        if self._last_failure_time is None:
            return True

        elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout

    async def call(
        self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                else:
                    reset_time = self._last_failure_time + timedelta(
                        seconds=self.recovery_timeout
                    )
                    raise CircuitBreakerError(self.name, reset_time)

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.to_thread(func, *args, **kwargs)

            # Success - update state
            async with self._lock:
                if self._state == CircuitBreakerState.HALF_OPEN:
                    self._state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
                    logger.info(f"Circuit breaker {self.name} recovered")

            return result

        except Exception as e:
            if isinstance(e, self.expected_exception):
                async with self._lock:
                    self._failure_count += 1
                    self._last_failure_time = datetime.now(timezone.utc)

                    if self._failure_count >= self.failure_threshold:
                        self._state = CircuitBreakerState.OPEN
                        logger.error(
                            f"Circuit breaker {self.name} opened after "
                            f"{self._failure_count} failures"
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
    *,
    default_return: Any = None,
    log_errors: bool = True,
    reraise: bool = False,
    error_callback: Optional[Callable[[Exception], None]] = None,
) -> Generator[None, None, Any]:
    """Context manager for standardized error handling."""
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
    *,
    default_return: Any = None,
    log_errors: bool = True,
    reraise: bool = False,
    error_callback: Optional[
        Union[Callable[[Exception], None], Callable[[Exception], Coroutine[Any, Any, None]]]
    ] = None,
):
    """Async context manager for standardized error handling."""
    try:
        yield
    except Exception as e:
        if log_errors:
            logger.error(f"Error in async context: {e}", exc_info=True)

        if error_callback:
            if asyncio.iscoroutinefunction(error_callback):
                await error_callback(e)
            else:
                error_callback(e)  # type: ignore

        if reraise:
            raise
        # Note: Cannot return value from async context manager except block


@dataclass
class ErrorStats:
    """Statistics for error tracking."""

    total_errors: int = 0
    error_rate: float = 0.0
    categories: Dict[str, int] = field(default_factory=dict)
    severities: Dict[str, int] = field(default_factory=dict)
    top_errors: List[tuple[str, int]] = field(default_factory=list)
    time_span_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_errors": self.total_errors,
            "error_rate": self.error_rate,
            "categories": self.categories,
            "severities": self.severities,
            "top_errors": self.top_errors,
            "time_span_seconds": self.time_span_seconds,
        }


class ErrorAggregator:
    """Aggregate and analyze errors for pattern detection."""

    def __init__(
        self, window_size: int = 100, time_window: float = 3600.0
    ) -> None:
        self.window_size = window_size
        self.time_window = time_window
        self.errors: deque[BaseError] = deque(maxlen=window_size)
        self._lock = asyncio.Lock()

    async def add_error(self, error: BaseError) -> None:
        """Add error to aggregator for analysis."""
        async with self._lock:
            # Remove old errors outside time window
            cutoff_time = datetime.now(timezone.utc) - timedelta(
                seconds=self.time_window
            )
            while self.errors and self.errors[0].error_context.timestamp < cutoff_time:
                self.errors.popleft()

            self.errors.append(error)

    async def get_error_stats(self) -> ErrorStats:
        """Get error statistics and patterns."""
        async with self._lock:
            if not self.errors:
                return ErrorStats()

            total = len(self.errors)
            first_error = self.errors[0].error_context.timestamp
            last_error = self.errors[-1].error_context.timestamp
            time_span = (last_error - first_error).total_seconds()
            error_rate = total / max(time_span, 1.0)

            # Count by category and severity
            categories: Dict[str, int] = {}
            severities: Dict[str, int] = {}
            error_codes: Dict[str, int] = {}

            for error in self.errors:
                ctx = error.error_context
                categories[ctx.category.value] = categories.get(ctx.category.value, 0) + 1
                severities[ctx.severity.name] = severities.get(ctx.severity.name, 0) + 1
                error_codes[ctx.error_code] = error_codes.get(ctx.error_code, 0) + 1

            top_errors = sorted(
                error_codes.items(), key=lambda x: x[1], reverse=True
            )[:5]

            return ErrorStats(
                total_errors=total,
                error_rate=error_rate,
                categories=categories,
                severities=severities,
                top_errors=top_errors,
                time_span_seconds=time_span,
            )

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
                window_errors = list(self.errors)[i : i + burst_threshold]
                time_diff = (
                    window_errors[-1].error_context.timestamp
                    - window_errors[0].error_context.timestamp
                ).total_seconds()

                if time_diff < burst_window:
                    patterns.append(
                        {
                            "type": "error_burst",
                            "count": burst_threshold,
                            "duration_seconds": time_diff,
                            "start_time": window_errors[
                                0
                            ].error_context.timestamp.isoformat(),
                            "categories": list(
                                {e.error_context.category.value for e in window_errors}
                            ),
                        }
                    )

            # Detect repeated errors
            error_messages: Dict[str, int] = {}
            for error in self.errors:
                error_messages[error.message] = error_messages.get(error.message, 0) + 1

            for message, count in error_messages.items():
                if count > 3:
                    patterns.append(
                        {
                            "type": "repeated_error",
                            "message": message,
                            "count": count,
                            "percentage": (count / len(self.errors)) * 100,
                        }
                    )

        return patterns

    def clear(self) -> None:
        """Clear all tracked errors."""
        self.errors.clear()


# Global error aggregator instance
error_aggregator = ErrorAggregator()


# Utility function for batch error handling
async def handle_errors_batch(
    operations: List[Callable[[], Coroutine[Any, Any, T]]],
    *,
    continue_on_error: bool = True,
    max_concurrent: int = 10,
) -> List[Union[T, Exception]]:
    """
    Execute multiple operations with error handling.

    Returns list of results or exceptions for each operation.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_error_handling(
        operation: Callable[[], Coroutine[Any, Any, T]]
    ) -> Union[T, Exception]:
        async with semaphore:
            try:
                return await operation()
            except Exception as e:
                if not continue_on_error:
                    raise
                return e

    return await asyncio.gather(
        *[run_with_error_handling(op) for op in operations],
        return_exceptions=False,
    )