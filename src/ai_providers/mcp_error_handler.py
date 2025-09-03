"""
MCP Error Handling and Recovery System for Provider Router.

This module implements comprehensive error handling, recovery strategies,
and fault tolerance mechanisms for the MCP Provider Router with Fallback system.

Key Features:
- Hierarchical error classification and handling
- Automatic error recovery strategies
- Circuit breaker patterns for provider resilience
- Error propagation and notification
- Retry mechanisms with exponential backoff
- Dead letter queue for failed requests
- Error analytics and reporting
"""

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field
from structlog import get_logger

from .mcp_protocol_schemas import (
    MCPProviderErrorCode,
    JSONRPCErrorCode,
    create_error_response,
    ProviderEventType,
    create_notification
)

logger = get_logger(__name__)


# Error Severity and Classification
class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for classification."""
    PROVIDER_ERROR = "provider_error"
    ROUTING_ERROR = "routing_error"
    CONFIGURATION_ERROR = "configuration_error"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    SYSTEM_ERROR = "system_error"


class RecoveryStrategy(str, Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FAILOVER = "failover"
    CIRCUIT_BREAK = "circuit_break"
    BACKOFF = "backoff"
    ESCALATE = "escalate"
    IGNORE = "ignore"
    MANUAL_INTERVENTION = "manual_intervention"


# Error Models
class ErrorContext(BaseModel):
    """Context information for errors."""
    request_id: Optional[str] = None
    provider: Optional[str] = None
    method: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    user_context: Dict[str, Any] = Field(default_factory=dict)
    system_state: Dict[str, Any] = Field(default_factory=dict)


class MCPError(BaseModel):
    """Structured MCP error representation."""
    error_id: str = Field(..., description="Unique error identifier")
    category: ErrorCategory = Field(..., description="Error category")
    severity: ErrorSeverity = Field(..., description="Error severity")
    code: Union[JSONRPCErrorCode, MCPProviderErrorCode] = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error details")
    context: ErrorContext = Field(default_factory=ErrorContext, description="Error context")
    recovery_strategy: RecoveryStrategy = Field(default=RecoveryStrategy.RETRY, description="Recommended recovery")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    can_retry: bool = Field(default=True, description="Whether error is retryable")
    upstream_errors: List["MCPError"] = Field(default_factory=list, description="Cascade error chain")


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests  
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker implementation for provider resilience."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 3,
        timeout: float = 30.0
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.timeout = timeout
        
        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitBreakerState.CLOSED
        self.last_failure_time: Optional[datetime] = None
        self.state_change_callbacks: List[Callable] = []
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                await self._notify_state_change()
            else:
                raise MCPCircuitBreakerError("Circuit breaker is OPEN")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)
            await self._on_success()
            return result
            
        except asyncio.TimeoutError:
            await self._on_failure()
            raise MCPTimeoutError(f"Operation timed out after {self.timeout}s")
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info("Circuit breaker reset to CLOSED")
                await self._notify_state_change()
        else:
            self.failure_count = 0
    
    async def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker failed during HALF_OPEN, returning to OPEN")
            await self._notify_state_change()
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            await self._notify_state_change()
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self.last_failure_time:
            return True
        
        return (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout
    
    async def _notify_state_change(self):
        """Notify callbacks of state changes."""
        for callback in self.state_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.state)
                else:
                    callback(self.state)
            except Exception as e:
                logger.error("Circuit breaker callback failed", error=str(e))
    
    def add_state_change_callback(self, callback: Callable):
        """Add state change callback."""
        self.state_change_callbacks.append(callback)


# Custom Exception Classes
class MCPRouterError(Exception):
    """Base exception for MCP router errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Union[JSONRPCErrorCode, MCPProviderErrorCode],
        category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
        can_retry: bool = True
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.recovery_strategy = recovery_strategy
        self.can_retry = can_retry


class MCPProviderUnavailableError(MCPRouterError):
    """Provider unavailable error."""
    
    def __init__(self, message: str, provider: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            error_code=MCPProviderErrorCode.NO_PROVIDER_AVAILABLE,
            category=ErrorCategory.PROVIDER_ERROR,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_strategy=RecoveryStrategy.FAILOVER,
            can_retry=True
        )
        if self.context:
            self.context.provider = provider


class MCPRoutingFailedError(MCPRouterError):
    """Routing failed error."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            error_code=MCPProviderErrorCode.ROUTING_FAILED,
            category=ErrorCategory.ROUTING_ERROR,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_strategy=RecoveryStrategy.RETRY,
            can_retry=True
        )


class MCPBudgetExceededError(MCPRouterError):
    """Budget exceeded error."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            error_code=MCPProviderErrorCode.BUDGET_EXCEEDED,
            category=ErrorCategory.VALIDATION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_strategy=RecoveryStrategy.ESCALATE,
            can_retry=False
        )


class MCPConfigurationError(MCPRouterError):
    """Configuration error."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            error_code=MCPProviderErrorCode.CONFIGURATION_ERROR,
            category=ErrorCategory.CONFIGURATION_ERROR,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            can_retry=False
        )


class MCPTimeoutError(MCPRouterError):
    """Timeout error."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            error_code=MCPProviderErrorCode.PROVIDER_TIMEOUT,
            category=ErrorCategory.TIMEOUT_ERROR,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_strategy=RecoveryStrategy.BACKOFF,
            can_retry=True
        )


class MCPCircuitBreakerError(MCPRouterError):
    """Circuit breaker error."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message=message,
            error_code=MCPProviderErrorCode.PROVIDER_UNAVAILABLE,
            category=ErrorCategory.PROVIDER_ERROR,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_strategy=RecoveryStrategy.CIRCUIT_BREAK,
            can_retry=False
        )


# Error Analytics and Reporting
class ErrorAnalytics:
    """Error analytics and reporting system."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.error_history: deque = deque(maxlen=max_history)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_trends: Dict[str, List[datetime]] = defaultdict(list)
        self.provider_error_rates: Dict[str, List[datetime]] = defaultdict(list)
    
    def record_error(self, error: MCPError):
        """Record an error for analytics."""
        self.error_history.append(error)
        
        # Update error counts
        self.error_counts[error.category] += 1
        self.error_counts[f"{error.category}:{error.code.name}"] += 1
        
        if error.context.provider:
            self.error_counts[f"provider:{error.context.provider}"] += 1
            self.provider_error_rates[error.context.provider].append(datetime.now())
        
        # Update trends
        now = datetime.now()
        self.error_trends[error.category].append(now)
        self.error_trends[error.severity].append(now)
        
        # Clean old trend data (keep last 24 hours)
        cutoff = now - timedelta(hours=24)
        for trend_list in self.error_trends.values():
            while trend_list and trend_list[0] < cutoff:
                trend_list.pop(0)
        
        for provider_list in self.provider_error_rates.values():
            while provider_list and provider_list[0] < cutoff:
                provider_list.pop(0)
    
    def get_error_rate(self, time_window: timedelta = timedelta(hours=1)) -> float:
        """Get error rate within time window."""
        now = datetime.now()
        cutoff = now - time_window
        
        recent_errors = [
            error for error in self.error_history
            if datetime.fromisoformat(error.context.timestamp) >= cutoff
        ]
        
        return len(recent_errors) / time_window.total_seconds() * 3600  # errors per hour
    
    def get_provider_error_rate(self, provider: str, time_window: timedelta = timedelta(hours=1)) -> float:
        """Get error rate for specific provider."""
        now = datetime.now()
        cutoff = now - time_window
        
        recent_errors = [
            ts for ts in self.provider_error_rates[provider]
            if ts >= cutoff
        ]
        
        return len(recent_errors) / time_window.total_seconds() * 3600
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary."""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        recent_errors_hour = [
            error for error in self.error_history
            if datetime.fromisoformat(error.context.timestamp) >= last_hour
        ]
        
        recent_errors_day = [
            error for error in self.error_history
            if datetime.fromisoformat(error.context.timestamp) >= last_day
        ]
        
        return {
            "total_errors": len(self.error_history),
            "errors_last_hour": len(recent_errors_hour),
            "errors_last_day": len(recent_errors_day),
            "error_rate_per_hour": self.get_error_rate(),
            "error_categories": dict(self.error_counts),
            "top_error_types": sorted(
                [(k, v) for k, v in self.error_counts.items() if ":" in k],
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "provider_error_rates": {
                provider: self.get_provider_error_rate(provider)
                for provider in self.provider_error_rates.keys()
            }
        }


# Main Error Handler
class MCPErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.analytics = ErrorAnalytics()
        self.retry_policies: Dict[ErrorCategory, Dict[str, Any]] = self._get_default_retry_policies()
        self.error_callbacks: List[Callable] = []
        self.dead_letter_queue: deque = deque(maxlen=1000)
        self._recovery_handlers: Dict[RecoveryStrategy, Callable] = {}
        
        # Initialize recovery handlers
        self._setup_recovery_handlers()
    
    def _get_default_retry_policies(self) -> Dict[ErrorCategory, Dict[str, Any]]:
        """Get default retry policies for different error categories."""
        return {
            ErrorCategory.PROVIDER_ERROR: {
                "max_retries": 3,
                "backoff_multiplier": 2,
                "max_backoff": 60,
                "jitter": True
            },
            ErrorCategory.NETWORK_ERROR: {
                "max_retries": 5,
                "backoff_multiplier": 1.5,
                "max_backoff": 30,
                "jitter": True
            },
            ErrorCategory.TIMEOUT_ERROR: {
                "max_retries": 2,
                "backoff_multiplier": 3,
                "max_backoff": 120,
                "jitter": False
            },
            ErrorCategory.RATE_LIMIT_ERROR: {
                "max_retries": 3,
                "backoff_multiplier": 5,
                "max_backoff": 300,
                "jitter": True
            },
            ErrorCategory.CONFIGURATION_ERROR: {
                "max_retries": 0,
                "backoff_multiplier": 0,
                "max_backoff": 0,
                "jitter": False
            }
        }
    
    def _setup_recovery_handlers(self):
        """Setup recovery strategy handlers."""
        self._recovery_handlers = {
            RecoveryStrategy.RETRY: self._handle_retry_recovery,
            RecoveryStrategy.FAILOVER: self._handle_failover_recovery,
            RecoveryStrategy.CIRCUIT_BREAK: self._handle_circuit_break_recovery,
            RecoveryStrategy.BACKOFF: self._handle_backoff_recovery,
            RecoveryStrategy.ESCALATE: self._handle_escalate_recovery,
            RecoveryStrategy.IGNORE: self._handle_ignore_recovery,
            RecoveryStrategy.MANUAL_INTERVENTION: self._handle_manual_intervention_recovery
        }
    
    async def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        request_data: Optional[Dict[str, Any]] = None
    ) -> MCPError:
        """
        Handle and classify an error with recovery strategy.
        
        Args:
            error: The exception that occurred
            context: Error context information
            request_data: Original request data for recovery
        
        Returns:
            Structured MCP error with recovery information
        """
        # Convert exception to structured error
        mcp_error = self._classify_error(error, context)
        
        # Record error for analytics
        self.analytics.record_error(mcp_error)
        
        # Log error
        logger.error(
            "MCP error occurred",
            error_id=mcp_error.error_id,
            category=mcp_error.category,
            severity=mcp_error.severity,
            message=mcp_error.message,
            provider=mcp_error.context.provider
        )
        
        # Notify error callbacks
        await self._notify_error_callbacks(mcp_error)
        
        # Apply recovery strategy if request data available
        if request_data and mcp_error.can_retry:
            recovery_handler = self._recovery_handlers.get(mcp_error.recovery_strategy)
            if recovery_handler:
                try:
                    await recovery_handler(mcp_error, request_data)
                except Exception as recovery_error:
                    logger.error(
                        "Error recovery failed",
                        original_error=mcp_error.error_id,
                        recovery_error=str(recovery_error)
                    )
        
        return mcp_error
    
    def _classify_error(self, error: Exception, context: Optional[ErrorContext] = None) -> MCPError:
        """Classify an exception into structured MCP error."""
        import uuid
        
        error_id = str(uuid.uuid4())
        
        # Handle MCP router errors (already classified)
        if isinstance(error, MCPRouterError):
            return MCPError(
                error_id=error_id,
                category=error.category,
                severity=error.severity,
                code=error.error_code,
                message=error.message,
                context=error.context,
                recovery_strategy=error.recovery_strategy,
                can_retry=error.can_retry
            )
        
        # Classify standard exceptions
        if isinstance(error, asyncio.TimeoutError):
            return MCPError(
                error_id=error_id,
                category=ErrorCategory.TIMEOUT_ERROR,
                severity=ErrorSeverity.MEDIUM,
                code=MCPProviderErrorCode.PROVIDER_TIMEOUT,
                message="Operation timed out",
                context=context or ErrorContext(),
                recovery_strategy=RecoveryStrategy.BACKOFF
            )
        
        elif isinstance(error, ConnectionError):
            return MCPError(
                error_id=error_id,
                category=ErrorCategory.NETWORK_ERROR,
                severity=ErrorSeverity.HIGH,
                code=MCPProviderErrorCode.PROVIDER_UNAVAILABLE,
                message=f"Connection error: {str(error)}",
                context=context or ErrorContext(),
                recovery_strategy=RecoveryStrategy.RETRY
            )
        
        elif "rate limit" in str(error).lower():
            return MCPError(
                error_id=error_id,
                category=ErrorCategory.RATE_LIMIT_ERROR,
                severity=ErrorSeverity.MEDIUM,
                code=MCPProviderErrorCode.PROVIDER_RATE_LIMITED,
                message=f"Rate limit exceeded: {str(error)}",
                context=context or ErrorContext(),
                recovery_strategy=RecoveryStrategy.BACKOFF
            )
        
        elif "authentication" in str(error).lower() or "unauthorized" in str(error).lower():
            return MCPError(
                error_id=error_id,
                category=ErrorCategory.AUTHENTICATION_ERROR,
                severity=ErrorSeverity.HIGH,
                code=MCPProviderErrorCode.CONFIGURATION_ERROR,
                message=f"Authentication error: {str(error)}",
                context=context or ErrorContext(),
                recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                can_retry=False
            )
        
        else:
            # Generic system error
            return MCPError(
                error_id=error_id,
                category=ErrorCategory.SYSTEM_ERROR,
                severity=ErrorSeverity.HIGH,
                code=MCPProviderErrorCode.PROVIDER_UNAVAILABLE,
                message=f"System error: {str(error)}",
                context=context or ErrorContext(),
                recovery_strategy=RecoveryStrategy.RETRY
            )
    
    async def _notify_error_callbacks(self, error: MCPError):
        """Notify registered error callbacks."""
        for callback in self.error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error)
                else:
                    callback(error)
            except Exception as e:
                logger.error("Error callback failed", error=str(e))
    
    # Recovery Strategy Handlers
    async def _handle_retry_recovery(self, error: MCPError, request_data: Dict[str, Any]):
        """Handle retry recovery strategy."""
        policy = self.retry_policies.get(error.category, self.retry_policies[ErrorCategory.PROVIDER_ERROR])
        
        if error.retry_count < policy["max_retries"]:
            backoff_time = min(
                policy["backoff_multiplier"] ** error.retry_count,
                policy["max_backoff"]
            )
            
            if policy["jitter"]:
                import random
                backoff_time *= (0.5 + random.random() * 0.5)
            
            logger.info(
                "Scheduling retry",
                error_id=error.error_id,
                retry_count=error.retry_count + 1,
                backoff_time=backoff_time
            )
            
            # In practice, this would schedule the actual retry
            # For now, just increment the retry count
            error.retry_count += 1
    
    async def _handle_failover_recovery(self, error: MCPError, request_data: Dict[str, Any]):
        """Handle failover recovery strategy."""
        logger.info("Initiating failover recovery", error_id=error.error_id, provider=error.context.provider)
        
        # Emit failover event
        await self._emit_error_event(
            ProviderEventType.FAILOVER_TRIGGERED,
            {
                "error_id": error.error_id,
                "from_provider": error.context.provider,
                "reason": error.message,
                "automatic": True
            }
        )
    
    async def _handle_circuit_break_recovery(self, error: MCPError, request_data: Dict[str, Any]):
        """Handle circuit breaker recovery strategy."""
        if error.context.provider:
            circuit_breaker = self.get_circuit_breaker(error.context.provider)
            logger.info(
                "Circuit breaker recovery",
                provider=error.context.provider,
                state=circuit_breaker.state,
                failure_count=circuit_breaker.failure_count
            )
    
    async def _handle_backoff_recovery(self, error: MCPError, request_data: Dict[str, Any]):
        """Handle backoff recovery strategy."""
        policy = self.retry_policies.get(error.category, self.retry_policies[ErrorCategory.TIMEOUT_ERROR])
        backoff_time = min(policy["backoff_multiplier"] ** error.retry_count, policy["max_backoff"])
        
        logger.info(
            "Applying backoff strategy",
            error_id=error.error_id,
            backoff_time=backoff_time
        )
    
    async def _handle_escalate_recovery(self, error: MCPError, request_data: Dict[str, Any]):
        """Handle escalation recovery strategy."""
        logger.warning("Error escalated for manual review", error_id=error.error_id)
        
        # Emit escalation event
        await self._emit_error_event(
            "error_escalated",
            {
                "error_id": error.error_id,
                "category": error.category,
                "severity": error.severity,
                "message": error.message
            }
        )
    
    async def _handle_ignore_recovery(self, error: MCPError, request_data: Dict[str, Any]):
        """Handle ignore recovery strategy."""
        logger.debug("Error ignored per recovery strategy", error_id=error.error_id)
    
    async def _handle_manual_intervention_recovery(self, error: MCPError, request_data: Dict[str, Any]):
        """Handle manual intervention recovery strategy."""
        logger.critical(
            "Manual intervention required",
            error_id=error.error_id,
            category=error.category,
            message=error.message
        )
        
        # Add to dead letter queue
        self.dead_letter_queue.append({
            "error": error,
            "request_data": request_data,
            "timestamp": datetime.now().isoformat()
        })
        
        # Emit manual intervention event
        await self._emit_error_event(
            "manual_intervention_required",
            {
                "error_id": error.error_id,
                "category": error.category,
                "severity": error.severity,
                "message": error.message,
                "provider": error.context.provider
            }
        )
    
    async def _emit_error_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit error-related events."""
        notification = create_notification(
            method=f"error/{event_type}",
            params={
                "timestamp": datetime.now().isoformat(),
                **event_data
            }
        )
        
        logger.debug("Emitting error event", event_type=event_type)
    
    def get_circuit_breaker(self, provider: str) -> CircuitBreaker:
        """Get or create circuit breaker for provider."""
        if provider not in self.circuit_breakers:
            self.circuit_breakers[provider] = CircuitBreaker()
            
            # Add callback for circuit breaker state changes
            async def on_state_change(state: CircuitBreakerState):
                logger.info(f"Circuit breaker state changed for {provider}: {state}")
                await self._emit_error_event(
                    "circuit_breaker_state_changed",
                    {
                        "provider": provider,
                        "state": state.value,
                        "failure_count": self.circuit_breakers[provider].failure_count
                    }
                )
            
            self.circuit_breakers[provider].add_state_change_callback(on_state_change)
        
        return self.circuit_breakers[provider]
    
    def add_error_callback(self, callback: Callable):
        """Add error callback for notifications."""
        self.error_callbacks.append(callback)
    
    def remove_error_callback(self, callback: Callable):
        """Remove error callback."""
        if callback in self.error_callbacks:
            self.error_callbacks.remove(callback)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        return {
            "analytics": self.analytics.get_error_summary(),
            "circuit_breakers": {
                provider: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count,
                    "success_count": cb.success_count
                }
                for provider, cb in self.circuit_breakers.items()
            },
            "dead_letter_queue_size": len(self.dead_letter_queue),
            "retry_policies": self.retry_policies
        }
    
    def create_jsonrpc_error_response(self, error: MCPError, request_id: Optional[str] = None):
        """Create JSON-RPC error response from MCP error."""
        return create_error_response(
            error_code=error.code,
            message=error.message,
            request_id=request_id,
            error_data={
                "error_id": error.error_id,
                "category": error.category,
                "severity": error.severity,
                "recovery_strategy": error.recovery_strategy,
                "can_retry": error.can_retry,
                "retry_count": error.retry_count,
                "context": error.context.dict(),
                "details": error.details
            }
        )


# Export main classes
__all__ = [
    "MCPErrorHandler",
    "CircuitBreaker",
    "ErrorAnalytics",
    "MCPError",
    "ErrorContext",
    "ErrorSeverity",
    "ErrorCategory", 
    "RecoveryStrategy",
    "CircuitBreakerState",
    
    # Exception classes
    "MCPRouterError",
    "MCPProviderUnavailableError",
    "MCPRoutingFailedError", 
    "MCPBudgetExceededError",
    "MCPConfigurationError",
    "MCPTimeoutError",
    "MCPCircuitBreakerError"
]