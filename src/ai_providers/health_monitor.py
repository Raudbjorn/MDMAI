"""Enhanced health monitoring with detailed error tracking."""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

from structlog import get_logger

from .models import ProviderType, ProviderStatus

logger = get_logger(__name__)


class ErrorType(Enum):
    """Specific error types for detailed tracking."""
    
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    QUOTA_EXCEEDED = "quota_exceeded"
    INVALID_REQUEST = "invalid_request"
    TIMEOUT = "timeout"
    SERVICE_ERROR = "service_error"
    MODEL_NOT_FOUND = "model_not_found"
    BUDGET_EXCEEDED = "budget_exceeded"
    UNKNOWN = "unknown"


@dataclass
class ErrorMetric:
    """Single error occurrence with details."""
    
    error_type: ErrorType
    provider_type: ProviderType
    model: Optional[str]
    message: str
    timestamp: datetime
    retryable: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthMetrics:
    """Comprehensive health metrics for a provider."""
    
    provider_type: ProviderType
    status: ProviderStatus
    uptime_percentage: float = 100.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Detailed error tracking
    error_counts: Dict[ErrorType, int] = field(default_factory=lambda: defaultdict(int))
    recent_errors: Deque[ErrorMetric] = field(default_factory=lambda: deque(maxlen=100))
    
    # Performance metrics
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Rate limit tracking
    rate_limit_hits: int = 0
    rate_limit_reset: Optional[datetime] = None
    rate_limit_remaining: Optional[int] = None
    
    # Quota tracking
    quota_used: float = 0.0
    quota_limit: Optional[float] = None
    quota_reset: Optional[datetime] = None
    
    # Circuit breaker state
    consecutive_failures: int = 0
    circuit_open_until: Optional[datetime] = None
    
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)


class HealthMonitor:
    """Advanced health monitoring with detailed error tracking and analysis."""
    
    def __init__(
        self,
        check_interval: int = 300,  # 5 minutes
        error_window: timedelta = timedelta(hours=1),
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 60,
    ):
        """Initialize health monitor.
        
        Args:
            check_interval: Health check interval in seconds
            error_window: Time window for error analysis
            circuit_breaker_threshold: Consecutive failures to open circuit
            circuit_breaker_timeout: Seconds to wait before testing recovery
        """
        self.check_interval = check_interval
        self.error_window = error_window
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        
        self._metrics: Dict[ProviderType, HealthMetrics] = {}
        self._check_task: Optional[asyncio.Task] = None
        self._provider_checks: Dict[ProviderType, callable] = {}
    
    def register_provider(
        self,
        provider_type: ProviderType,
        health_check_func: callable,
    ) -> None:
        """Register a provider with its health check function.
        
        Args:
            provider_type: Provider to monitor
            health_check_func: Async function to perform health check
        """
        self._metrics[provider_type] = HealthMetrics(
            provider_type=provider_type,
            status=ProviderStatus.AVAILABLE,
        )
        self._provider_checks[provider_type] = health_check_func
        
        logger.info("Registered provider for health monitoring", provider=provider_type.value)
    
    async def start(self) -> None:
        """Start health monitoring."""
        if not self._check_task:
            self._check_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Started health monitoring", interval=self.check_interval)
    
    async def stop(self) -> None:
        """Stop health monitoring."""
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None
            logger.info("Stopped health monitoring")
    
    async def record_request(
        self,
        provider_type: ProviderType,
        success: bool,
        latency_ms: Optional[float] = None,
        model: Optional[str] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Record a request outcome.
        
        Args:
            provider_type: Provider that handled the request
            success: Whether request succeeded
            latency_ms: Request latency in milliseconds
            model: Model used
            error: Exception if request failed
        """
        if provider_type not in self._metrics:
            self._metrics[provider_type] = HealthMetrics(
                provider_type=provider_type,
                status=ProviderStatus.AVAILABLE,
            )
        
        metrics = self._metrics[provider_type]
        metrics.total_requests += 1
        
        if success:
            metrics.successful_requests += 1
            metrics.last_success = datetime.now()
            metrics.consecutive_failures = 0
            
            # Update latency metrics
            if latency_ms:
                self._update_latency_metrics(metrics, latency_ms)
            
            # Check if circuit breaker can be closed
            if metrics.circuit_open_until and datetime.now() > metrics.circuit_open_until:
                metrics.status = ProviderStatus.AVAILABLE
                metrics.circuit_open_until = None
                logger.info("Circuit breaker closed", provider=provider_type.value)
        
        else:
            metrics.failed_requests += 1
            metrics.last_failure = datetime.now()
            metrics.consecutive_failures += 1
            
            # Record error details
            if error:
                error_type = self._classify_error(error)
                metrics.error_counts[error_type] += 1
                
                error_metric = ErrorMetric(
                    error_type=error_type,
                    provider_type=provider_type,
                    model=model,
                    message=str(error),
                    timestamp=datetime.now(),
                    retryable=self._is_retryable_error(error_type),
                    details=self._extract_error_details(error),
                )
                metrics.recent_errors.append(error_metric)
                
                # Handle specific error types
                self._handle_specific_error(metrics, error_type, error)
            
            # Check circuit breaker
            if metrics.consecutive_failures >= self.circuit_breaker_threshold:
                metrics.status = ProviderStatus.ERROR
                metrics.circuit_open_until = datetime.now() + timedelta(
                    seconds=self.circuit_breaker_timeout
                )
                logger.warning(
                    "Circuit breaker opened",
                    provider=provider_type.value,
                    failures=metrics.consecutive_failures,
                )
        
        # Update uptime percentage
        if metrics.total_requests > 0:
            metrics.uptime_percentage = (
                metrics.successful_requests / metrics.total_requests * 100
            )
        
        metrics.last_updated = datetime.now()
    
    def get_metrics(self, provider_type: ProviderType) -> Optional[HealthMetrics]:
        """Get health metrics for a provider.
        
        Args:
            provider_type: Provider to get metrics for
            
        Returns:
            Health metrics or None if not monitored
        """
        return self._metrics.get(provider_type)
    
    def get_error_analysis(
        self,
        provider_type: Optional[ProviderType] = None,
    ) -> Dict[str, Any]:
        """Get detailed error analysis.
        
        Args:
            provider_type: Optional specific provider to analyze
            
        Returns:
            Error analysis report
        """
        analysis = {}
        
        providers = [provider_type] if provider_type else self._metrics.keys()
        
        for provider in providers:
            if provider not in self._metrics:
                continue
            
            metrics = self._metrics[provider]
            cutoff = datetime.now() - self.error_window
            
            # Filter recent errors
            recent_errors = [
                e for e in metrics.recent_errors
                if e.timestamp >= cutoff
            ]
            
            # Aggregate by error type
            error_breakdown = defaultdict(list)
            for error in recent_errors:
                error_breakdown[error.error_type].append({
                    "timestamp": error.timestamp.isoformat(),
                    "message": error.message[:100],  # Truncate long messages
                    "retryable": error.retryable,
                })
            
            provider_analysis = {
                "status": metrics.status.value,
                "uptime_percentage": metrics.uptime_percentage,
                "total_errors": len(recent_errors),
                "error_types": {
                    error_type.value: {
                        "count": metrics.error_counts[error_type],
                        "percentage": (
                            metrics.error_counts[error_type] / metrics.failed_requests * 100
                            if metrics.failed_requests > 0 else 0
                        ),
                        "recent": error_breakdown[error_type][:5],  # Last 5 of each type
                    }
                    for error_type in ErrorType
                    if metrics.error_counts[error_type] > 0
                },
                "circuit_breaker": {
                    "open": metrics.circuit_open_until is not None,
                    "consecutive_failures": metrics.consecutive_failures,
                    "open_until": (
                        metrics.circuit_open_until.isoformat()
                        if metrics.circuit_open_until else None
                    ),
                },
                "rate_limiting": {
                    "hits": metrics.rate_limit_hits,
                    "remaining": metrics.rate_limit_remaining,
                    "reset": (
                        metrics.rate_limit_reset.isoformat()
                        if metrics.rate_limit_reset else None
                    ),
                },
            }
            
            analysis[provider.value] = provider_analysis
        
        return analysis
    
    def get_recommendations(
        self,
        provider_type: ProviderType,
    ) -> List[str]:
        """Get actionable recommendations based on health metrics.
        
        Args:
            provider_type: Provider to analyze
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if provider_type not in self._metrics:
            return ["Provider not monitored"]
        
        metrics = self._metrics[provider_type]
        
        # Check uptime
        if metrics.uptime_percentage < 95:
            recommendations.append(
                f"Low uptime ({metrics.uptime_percentage:.1f}%). Consider using fallback providers."
            )
        
        # Check error patterns
        if metrics.error_counts[ErrorType.RATE_LIMIT] > 5:
            recommendations.append(
                "Frequent rate limiting. Consider implementing request throttling or upgrading plan."
            )
        
        if metrics.error_counts[ErrorType.AUTHENTICATION] > 0:
            recommendations.append(
                "Authentication errors detected. Verify API credentials."
            )
        
        if metrics.error_counts[ErrorType.TIMEOUT] > metrics.total_requests * 0.1:
            recommendations.append(
                "High timeout rate (>10%). Consider increasing timeout or using faster models."
            )
        
        if metrics.error_counts[ErrorType.QUOTA_EXCEEDED] > 0:
            recommendations.append(
                "Quota exceeded. Monitor usage and consider upgrading limits."
            )
        
        # Check circuit breaker
        if metrics.circuit_open_until:
            recommendations.append(
                f"Circuit breaker open until {metrics.circuit_open_until.isoformat()}. "
                "Provider temporarily disabled due to failures."
            )
        
        # Performance recommendations
        if metrics.p99_latency_ms > 10000:  # 10 seconds
            recommendations.append(
                f"High P99 latency ({metrics.p99_latency_ms:.0f}ms). "
                "Consider using streaming or different models."
            )
        
        return recommendations if recommendations else ["Provider operating normally"]
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify an error into specific type.
        
        Args:
            error: Exception to classify
            
        Returns:
            Error type
        """
        error_str = str(error).lower()
        error_type_name = type(error).__name__
        
        # Network errors
        if any(x in error_str for x in ["connection", "network", "dns", "socket"]):
            return ErrorType.NETWORK
        
        # Authentication
        if any(x in error_str for x in ["unauthorized", "authentication", "api key", "403"]):
            return ErrorType.AUTHENTICATION
        
        # Rate limiting
        if any(x in error_str for x in ["rate limit", "too many requests", "429"]):
            return ErrorType.RATE_LIMIT
        
        # Quota
        if any(x in error_str for x in ["quota", "limit exceeded", "usage limit"]):
            return ErrorType.QUOTA_EXCEEDED
        
        # Invalid request
        if any(x in error_str for x in ["invalid", "bad request", "400"]):
            return ErrorType.INVALID_REQUEST
        
        # Timeout
        if any(x in error_str for x in ["timeout", "timed out"]):
            return ErrorType.TIMEOUT
        
        # Model not found
        if any(x in error_str for x in ["model not found", "model does not exist"]):
            return ErrorType.MODEL_NOT_FOUND
        
        # Budget
        if any(x in error_str for x in ["budget", "cost limit"]):
            return ErrorType.BUDGET_EXCEEDED
        
        # Service errors
        if any(x in error_str for x in ["service", "500", "502", "503", "504"]):
            return ErrorType.SERVICE_ERROR
        
        return ErrorType.UNKNOWN
    
    def _is_retryable_error(self, error_type: ErrorType) -> bool:
        """Check if error type is retryable.
        
        Args:
            error_type: Error type
            
        Returns:
            True if retryable
        """
        retryable_types = {
            ErrorType.NETWORK,
            ErrorType.RATE_LIMIT,
            ErrorType.TIMEOUT,
            ErrorType.SERVICE_ERROR,
        }
        return error_type in retryable_types
    
    def _extract_error_details(self, error: Exception) -> Dict[str, Any]:
        """Extract additional details from error.
        
        Args:
            error: Exception to analyze
            
        Returns:
            Error details dictionary
        """
        details = {
            "type": type(error).__name__,
            "module": type(error).__module__,
        }
        
        # Extract HTTP status code if present
        if hasattr(error, "status_code"):
            details["status_code"] = error.status_code
        
        # Extract response if present
        if hasattr(error, "response"):
            try:
                details["response"] = str(error.response)[:200]
            except (AttributeError, Exception):
                pass
        
        return details
    
    def _handle_specific_error(
        self,
        metrics: HealthMetrics,
        error_type: ErrorType,
        error: Exception,
    ) -> None:
        """Handle specific error types.
        
        Args:
            metrics: Provider metrics
            error_type: Classified error type
            error: Original exception
        """
        if error_type == ErrorType.RATE_LIMIT:
            metrics.rate_limit_hits += 1
            metrics.status = ProviderStatus.RATE_LIMITED
            
            # Try to extract reset time
            if hasattr(error, "retry_after"):
                metrics.rate_limit_reset = datetime.now() + timedelta(
                    seconds=error.retry_after
                )
        
        elif error_type == ErrorType.QUOTA_EXCEEDED:
            metrics.status = ProviderStatus.QUOTA_EXCEEDED
        
        elif error_type == ErrorType.AUTHENTICATION:
            metrics.status = ProviderStatus.ERROR
    
    def _update_latency_metrics(
        self,
        metrics: HealthMetrics,
        latency_ms: float,
    ) -> None:
        """Update latency metrics with exponential moving average.
        
        Args:
            metrics: Provider metrics
            latency_ms: New latency measurement
        """
        alpha = 0.1  # Smoothing factor
        
        if metrics.avg_latency_ms == 0:
            metrics.avg_latency_ms = latency_ms
        else:
            metrics.avg_latency_ms = (
                alpha * latency_ms + (1 - alpha) * metrics.avg_latency_ms
            )
        
        # Update percentiles (simplified - should use proper percentile tracking)
        metrics.p95_latency_ms = max(metrics.p95_latency_ms, latency_ms * 0.95)
        metrics.p99_latency_ms = max(metrics.p99_latency_ms, latency_ms * 0.99)
    
    async def _monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health monitoring loop", error=str(e))
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all registered providers."""
        for provider_type, check_func in self._provider_checks.items():
            try:
                # Perform health check
                is_healthy = await check_func()
                
                # Record result
                await self.record_request(
                    provider_type=provider_type,
                    success=is_healthy,
                )
                
            except Exception as e:
                # Record health check failure
                await self.record_request(
                    provider_type=provider_type,
                    success=False,
                    error=e,
                )
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for analysis.
        
        Returns:
            All health metrics
        """
        return {
            provider.value: {
                "status": metrics.status.value,
                "uptime": metrics.uptime_percentage,
                "requests": {
                    "total": metrics.total_requests,
                    "successful": metrics.successful_requests,
                    "failed": metrics.failed_requests,
                },
                "errors": {
                    error_type.value: count
                    for error_type, count in metrics.error_counts.items()
                },
                "latency": {
                    "avg_ms": metrics.avg_latency_ms,
                    "p95_ms": metrics.p95_latency_ms,
                    "p99_ms": metrics.p99_latency_ms,
                },
                "last_success": (
                    metrics.last_success.isoformat()
                    if metrics.last_success else None
                ),
                "last_failure": (
                    metrics.last_failure.isoformat()
                    if metrics.last_failure else None
                ),
            }
            for provider, metrics in self._metrics.items()
        }