"""
Comprehensive tests for error handling system.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.error_handling import (
    BaseError,
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerState,
    ConfigurationError,
    DatabaseError,
    ErrorAggregator,
    ErrorCategory,
    ErrorSeverity,
    MCPProtocolError,
    NetworkError,
    RetryConfig,
    ServiceError,
    SystemError,
    ValidationError,
    async_error_handler,
    error_handler,
    retry,
)


class TestBaseError:
    """Test BaseError class and hierarchy."""

    def test_base_error_initialization(self):
        """Test BaseError initialization with all parameters."""
        error = BaseError(
            message="Test error",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SERVICE,
            error_code="TEST-001",
            context={"key": "value"},
            recoverable=False,
            retry_after=30,
        )
        
        assert error.message == "Test error"
        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.SERVICE
        assert error.error_code == "TEST-001"
        assert error.context == {"key": "value"}
        assert error.recoverable is False
        assert error.retry_after == 30
        assert isinstance(error.timestamp, datetime)

    def test_auto_generated_error_code(self):
        """Test automatic error code generation."""
        error = BaseError("Test error")
        assert error.error_code is not None
        assert error.error_code.startswith("SYS-MED-")  # System-Medium prefix

    def test_error_to_dict(self):
        """Test error serialization to dictionary."""
        error = BaseError(
            message="Test error",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.DATABASE,
            context={"operation": "insert"},
        )
        
        error_dict = error.to_dict()
        assert error_dict["message"] == "Test error"
        assert error_dict["severity"] == "CRITICAL"
        assert error_dict["category"] == "DATABASE"
        assert error_dict["context"] == {"operation": "insert"}
        assert error_dict["recoverable"] is True

    def test_system_error(self):
        """Test SystemError initialization."""
        error = SystemError("Critical system failure")
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.category == ErrorCategory.SYSTEM
        assert error.recoverable is False

    def test_service_error(self):
        """Test ServiceError with service name."""
        error = ServiceError("Service unavailable", service_name="AuthService")
        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.SERVICE
        assert error.context["service_name"] == "AuthService"

    def test_database_error(self):
        """Test DatabaseError with operation."""
        error = DatabaseError("Connection failed", operation="SELECT")
        assert error.category == ErrorCategory.DATABASE
        assert error.context["operation"] == "SELECT"

    def test_validation_error(self):
        """Test ValidationError with field."""
        error = ValidationError("Invalid email", field="email")
        assert error.severity == ErrorSeverity.LOW
        assert error.category == ErrorCategory.VALIDATION
        assert error.context["field"] == "email"
        assert error.recoverable is False


class TestRetryMechanism:
    """Test retry decorator and configuration."""

    def test_retry_config_initialization(self):
        """Test RetryConfig initialization."""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=2.0,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=False,
        )
        
        assert config.max_attempts == 5
        assert config.initial_delay == 2.0
        assert config.max_delay == 30.0
        assert config.exponential_base == 3.0
        assert config.jitter is False

    def test_calculate_delay_without_jitter(self):
        """Test delay calculation without jitter."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=False,
        )
        
        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 8.0

    def test_calculate_delay_with_max_delay(self):
        """Test delay calculation respects max_delay."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=5.0,
            jitter=False,
        )
        
        assert config.calculate_delay(10) == 5.0  # Should be capped at max_delay

    def test_should_retry_logic(self):
        """Test should_retry decision logic."""
        config = RetryConfig(
            retry_on=[NetworkError],
            ignore_on=[ValidationError],
        )
        
        assert config.should_retry(NetworkError("Network error", "api.example.com"))
        assert not config.should_retry(ValidationError("Invalid input"))
        assert not config.should_retry(BaseError("Non-recoverable", recoverable=False))

    def test_sync_retry_decorator_success(self):
        """Test retry decorator with successful function."""
        attempt_count = 0
        
        @retry(RetryConfig(max_attempts=3))
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise NetworkError("Connection failed", "api.example.com")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert attempt_count == 2

    def test_sync_retry_decorator_max_attempts(self):
        """Test retry decorator reaching max attempts."""
        attempt_count = 0
        
        @retry(RetryConfig(max_attempts=3, initial_delay=0.01))
        def always_fails():
            nonlocal attempt_count
            attempt_count += 1
            raise NetworkError("Connection failed", "api.example.com")
        
        with pytest.raises(NetworkError):
            always_fails()
        
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_async_retry_decorator(self):
        """Test retry decorator with async function."""
        attempt_count = 0
        
        @retry(RetryConfig(max_attempts=3, initial_delay=0.01))
        async def async_flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise NetworkError("Connection failed", "api.example.com")
            return "async_success"
        
        result = await async_flaky_function()
        assert result == "async_success"
        assert attempt_count == 2


class TestCircuitBreaker:
    """Test circuit breaker pattern implementation."""

    def test_circuit_breaker_initialization(self):
        """Test CircuitBreaker initialization."""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=10.0,
            expected_exception=NetworkError,
            name="TestBreaker",
        )
        
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 10.0
        assert cb.expected_exception == NetworkError
        assert cb.name == "TestBreaker"
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
        
        async def failing_function():
            raise Exception("Failed")
        
        # First failure
        with pytest.raises(Exception):
            await cb.call(failing_function)
        assert cb.state == CircuitBreakerState.CLOSED
        
        # Second failure - should open
        with pytest.raises(Exception):
            await cb.call(failing_function)
        assert cb.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_rejects_when_open(self):
        """Test circuit breaker rejects calls when open."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
        
        async def failing_function():
            raise Exception("Failed")
        
        # Open the circuit
        with pytest.raises(Exception):
            await cb.call(failing_function)
        
        # Should reject with CircuitBreakerError
        with pytest.raises(CircuitBreakerError) as exc_info:
            await cb.call(failing_function)
        
        assert "Circuit breaker open" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery to half-open and closed."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        call_count = 0
        
        async def recovering_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise Exception("Failed")
            return "success"
        
        # Open the circuit
        with pytest.raises(Exception):
            await cb.call(recovering_function)
        assert cb.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(0.15)
        
        # Should enter half-open and succeed
        result = await cb.call(recovering_function)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_manual_reset(self):
        """Test manual circuit breaker reset."""
        cb = CircuitBreaker(failure_threshold=1)
        cb._state = CircuitBreakerState.OPEN
        cb._failure_count = 5
        
        cb.reset()
        
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb._failure_count == 0


class TestErrorAggregator:
    """Test error aggregation and pattern detection."""

    @pytest.mark.asyncio
    async def test_error_aggregator_initialization(self):
        """Test ErrorAggregator initialization."""
        aggregator = ErrorAggregator(window_size=50, time_window=1800.0)
        
        assert aggregator.window_size == 50
        assert aggregator.time_window == 1800.0
        assert len(aggregator.errors) == 0

    @pytest.mark.asyncio
    async def test_add_error_to_aggregator(self):
        """Test adding errors to aggregator."""
        aggregator = ErrorAggregator()
        
        error1 = BaseError("Error 1", severity=ErrorSeverity.HIGH)
        error2 = BaseError("Error 2", severity=ErrorSeverity.LOW)
        
        await aggregator.add_error(error1)
        await aggregator.add_error(error2)
        
        assert len(aggregator.errors) == 2

    @pytest.mark.asyncio
    async def test_error_aggregator_window_limit(self):
        """Test error aggregator respects window size."""
        aggregator = ErrorAggregator(window_size=3)
        
        for i in range(5):
            error = BaseError(f"Error {i}")
            await aggregator.add_error(error)
        
        assert len(aggregator.errors) == 3
        assert aggregator.errors[0].message == "Error 2"  # Oldest kept

    @pytest.mark.asyncio
    async def test_get_error_statistics(self):
        """Test error statistics calculation."""
        aggregator = ErrorAggregator()
        
        # Add various errors
        await aggregator.add_error(
            BaseError("DB error", category=ErrorCategory.DATABASE)
        )
        await aggregator.add_error(
            BaseError("Network error", category=ErrorCategory.NETWORK)
        )
        await aggregator.add_error(
            BaseError("DB error 2", category=ErrorCategory.DATABASE)
        )
        await aggregator.add_error(
            BaseError("Critical", severity=ErrorSeverity.CRITICAL)
        )
        
        stats = await aggregator.get_error_stats()
        
        assert stats["total_errors"] == 4
        assert stats["categories"]["DATABASE"] == 2
        assert stats["categories"]["NETWORK"] == 1
        assert stats["severities"]["CRITICAL"] == 1

    @pytest.mark.asyncio
    async def test_detect_error_patterns(self):
        """Test error pattern detection."""
        aggregator = ErrorAggregator()
        
        # Create error burst
        for i in range(6):
            error = BaseError(f"Burst error {i % 2}")
            await aggregator.add_error(error)
        
        # Add more errors with delay
        await asyncio.sleep(0.1)
        for i in range(5):
            error = BaseError("Repeated error")
            await aggregator.add_error(error)
        
        patterns = await aggregator.detect_patterns()
        
        # Should detect burst and repeated errors
        burst_patterns = [p for p in patterns if p["type"] == "error_burst"]
        repeated_patterns = [p for p in patterns if p["type"] == "repeated_error"]
        
        assert len(burst_patterns) > 0
        assert len(repeated_patterns) > 0
        assert repeated_patterns[0]["message"] == "Repeated error"
        assert repeated_patterns[0]["count"] == 5


class TestErrorHandlers:
    """Test error handler context managers."""

    def test_error_handler_context_success(self):
        """Test error_handler context manager with success."""
        with error_handler() as handler:
            result = 1 + 1
        
        assert result == 2

    def test_error_handler_context_with_error(self):
        """Test error_handler context manager with error."""
        callback_called = False
        error_received = None
        
        def error_callback(e):
            nonlocal callback_called, error_received
            callback_called = True
            error_received = e
        
        with error_handler(
            default_return="default",
            error_callback=error_callback,
            reraise=False,
        ):
            raise ValueError("Test error")
        
        assert callback_called
        assert isinstance(error_received, ValueError)

    def test_error_handler_reraise(self):
        """Test error_handler context manager with reraise."""
        with pytest.raises(ValueError):
            with error_handler(reraise=True):
                raise ValueError("Test error")

    @pytest.mark.asyncio
    async def test_async_error_handler_success(self):
        """Test async_error_handler context manager with success."""
        async with async_error_handler():
            result = await asyncio.sleep(0, result=42)
        
        assert result == 42

    @pytest.mark.asyncio
    async def test_async_error_handler_with_error(self):
        """Test async_error_handler context manager with error."""
        callback_called = False
        
        async def async_callback(e):
            nonlocal callback_called
            callback_called = True
        
        async with async_error_handler(
            error_callback=async_callback,
            reraise=False,
        ):
            raise RuntimeError("Async error")
        
        assert callback_called


class TestIntegration:
    """Integration tests for error handling system."""

    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(self):
        """Test retry mechanism with circuit breaker."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        attempt_count = 0
        
        @retry(RetryConfig(max_attempts=5, initial_delay=0.01))
        async def protected_function():
            nonlocal attempt_count
            attempt_count += 1
            
            async def inner():
                if attempt_count <= 2:
                    raise NetworkError("Failed", "api.example.com")
                return "success"
            
            return await cb.call(inner)
        
        # First two attempts will fail and open circuit
        with pytest.raises(CircuitBreakerError):
            await protected_function()
        
        # Circuit should be open
        assert cb.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_error_aggregation_with_recovery(self):
        """Test error aggregation with recovery mechanisms."""
        aggregator = ErrorAggregator()
        
        # Simulate various errors
        errors = [
            DatabaseError("Connection lost", "SELECT"),
            NetworkError("Timeout", "api.example.com"),
            ValidationError("Invalid input"),
            ServiceError("Service down", "AuthService"),
        ]
        
        for error in errors:
            await aggregator.add_error(error)
        
        stats = await aggregator.get_error_stats()
        
        assert stats["total_errors"] == 4
        assert len(stats["categories"]) == 4
        assert stats["categories"]["DATABASE"] == 1
        assert stats["categories"]["NETWORK"] == 1