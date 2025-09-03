"""
Health checker for AI provider monitoring in MDMAI TTRPG Assistant.

This module provides comprehensive health monitoring for AI providers with
automated checks, alerting, and recovery mechanisms.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .base_provider import BaseAIProvider, ProviderType, ProviderError

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    provider_type: ProviderType
    status: HealthStatus
    response_time_ms: float
    timestamp: datetime
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'provider_type': self.provider_type.value,
            'status': self.status.value,
            'response_time_ms': self.response_time_ms,
            'timestamp': self.timestamp.isoformat(),
            'error_message': self.error_message,
            'details': self.details
        }


@dataclass
class HealthMetrics:
    """Health metrics for a provider."""
    total_checks: int = 0
    successful_checks: int = 0
    failed_checks: int = 0
    average_response_time: float = 0.0
    uptime_percentage: float = 100.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_checks == 0:
            return 100.0
        return (self.successful_checks / self.total_checks) * 100.0


@dataclass
class AlertRule:
    """Configuration for health alerts."""
    name: str
    condition: Callable[[HealthMetrics], bool]
    severity: AlertSeverity
    message_template: str
    cooldown_minutes: int = 30
    enabled: bool = True
    last_triggered: Optional[datetime] = None


class HealthChecker:
    """
    Comprehensive health checker for AI providers.
    
    Features:
    - Periodic health checks
    - Real-time monitoring
    - Alert system
    - Historical metrics
    - Automated recovery actions
    - Dashboard integration
    """
    
    def __init__(self, check_interval_seconds: int = 300):
        """
        Initialize health checker.
        
        Args:
            check_interval_seconds: Interval between health checks
        """
        self.check_interval = check_interval_seconds
        self.providers: Dict[str, BaseAIProvider] = {}
        self.metrics: Dict[str, HealthMetrics] = {}
        self.check_history: Dict[str, List[HealthCheckResult]] = {}
        self.alert_handlers: List[Callable] = []
        self.alert_rules: List[AlertRule] = []
        
        # Monitoring state
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()
        
        # Default alert rules
        self._setup_default_alerts()
        
        logger.info(f"HealthChecker initialized with {check_interval_seconds}s interval")
    
    def register_provider(self, user_id: str, provider: BaseAIProvider):
        """
        Register a provider for health monitoring.
        
        Args:
            user_id: User identifier
            provider: Provider instance to monitor
        """
        provider_key = f"{user_id}:{provider.config.type.value}"
        self.providers[provider_key] = provider
        self.metrics[provider_key] = HealthMetrics()
        self.check_history[provider_key] = []
        
        # Start monitoring task
        self._start_monitoring_task(provider_key)
        
        logger.info(f"Registered provider {provider.config.type.value} for user {user_id} for health monitoring")
    
    def unregister_provider(self, user_id: str, provider_type: ProviderType):
        """
        Unregister a provider from health monitoring.
        
        Args:
            user_id: User identifier
            provider_type: Provider type to unregister
        """
        provider_key = f"{user_id}:{provider_type.value}"
        
        # Stop monitoring task
        if provider_key in self._monitoring_tasks:
            self._monitoring_tasks[provider_key].cancel()
            del self._monitoring_tasks[provider_key]
        
        # Clean up data
        self.providers.pop(provider_key, None)
        self.metrics.pop(provider_key, None)
        self.check_history.pop(provider_key, None)
        
        logger.info(f"Unregistered provider {provider_type.value} for user {user_id} from health monitoring")
    
    async def perform_health_check(self, provider_key: str) -> HealthCheckResult:
        """
        Perform a health check on a specific provider.
        
        Args:
            provider_key: Provider key to check
            
        Returns:
            HealthCheckResult: Result of the health check
        """
        if provider_key not in self.providers:
            return HealthCheckResult(
                provider_type=ProviderType.ANTHROPIC,  # Default
                status=HealthStatus.UNKNOWN,
                response_time_ms=0.0,
                timestamp=datetime.utcnow(),
                error_message="Provider not registered"
            )
        
        provider = self.providers[provider_key]
        start_time = time.time()
        
        try:
            # Perform health check
            is_healthy = await asyncio.wait_for(
                provider.health_check(),
                timeout=30.0  # 30 second timeout
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            result = HealthCheckResult(
                provider_type=provider.config.type,
                status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                timestamp=datetime.utcnow(),
                details={'timeout_used': False}
            )
            
            # Update metrics
            self._update_metrics(provider_key, result)
            
            return result
            
        except asyncio.TimeoutError:
            response_time_ms = (time.time() - start_time) * 1000
            
            result = HealthCheckResult(
                provider_type=provider.config.type,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                timestamp=datetime.utcnow(),
                error_message="Health check timed out",
                details={'timeout_used': True, 'timeout_seconds': 30}
            )
            
            self._update_metrics(provider_key, result)
            return result
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            
            result = HealthCheckResult(
                provider_type=provider.config.type,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                timestamp=datetime.utcnow(),
                error_message=str(e),
                details={'exception_type': type(e).__name__}
            )
            
            self._update_metrics(provider_key, result)
            return result
    
    async def check_all_providers(self, user_id: Optional[str] = None) -> Dict[str, HealthCheckResult]:
        """
        Perform health checks on all registered providers.
        
        Args:
            user_id: Optional user filter
            
        Returns:
            Dict[str, HealthCheckResult]: Results for all providers
        """
        providers_to_check = self.providers.keys()
        
        if user_id:
            providers_to_check = [
                key for key in providers_to_check 
                if key.startswith(f"{user_id}:")
            ]
        
        # Run health checks in parallel
        tasks = {
            provider_key: asyncio.create_task(self.perform_health_check(provider_key))
            for provider_key in providers_to_check
        }
        
        results = {}
        for provider_key, task in tasks.items():
            try:
                results[provider_key] = await task
            except Exception as e:
                logger.error(f"Failed to check provider {provider_key}: {e}")
                results[provider_key] = HealthCheckResult(
                    provider_type=ProviderType.ANTHROPIC,  # Default
                    status=HealthStatus.UNKNOWN,
                    response_time_ms=0.0,
                    timestamp=datetime.utcnow(),
                    error_message=f"Check failed: {e}"
                )
        
        return results
    
    def get_provider_status(self, user_id: str, provider_type: ProviderType) -> Dict[str, Any]:
        """
        Get current status for a specific provider.
        
        Args:
            user_id: User identifier
            provider_type: Provider type
            
        Returns:
            Dict[str, Any]: Status information
        """
        provider_key = f"{user_id}:{provider_type.value}"
        
        if provider_key not in self.metrics:
            return {
                'status': HealthStatus.UNKNOWN.value,
                'metrics': {},
                'last_check': None
            }
        
        metrics = self.metrics[provider_key]
        history = self.check_history[provider_key]
        
        return {
            'status': self._determine_overall_status(provider_key).value,
            'metrics': {
                'total_checks': metrics.total_checks,
                'success_rate': metrics.success_rate,
                'average_response_time_ms': metrics.average_response_time,
                'uptime_percentage': metrics.uptime_percentage,
                'consecutive_failures': metrics.consecutive_failures,
                'consecutive_successes': metrics.consecutive_successes,
                'last_success': metrics.last_success.isoformat() if metrics.last_success else None,
                'last_failure': metrics.last_failure.isoformat() if metrics.last_failure else None
            },
            'last_check': history[-1].to_dict() if history else None,
            'recent_checks': [check.to_dict() for check in history[-10:]]  # Last 10 checks
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Dict[str, Any]: System status information
        """
        if not self.providers:
            return {
                'overall_status': HealthStatus.UNKNOWN.value,
                'total_providers': 0,
                'healthy_providers': 0,
                'degraded_providers': 0,
                'unhealthy_providers': 0,
                'providers': {}
            }
        
        provider_statuses = {}
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        for provider_key in self.providers:
            status = self._determine_overall_status(provider_key)
            provider_statuses[provider_key] = status
            status_counts[status] += 1
        
        # Determine overall system status
        total_providers = len(self.providers)
        if status_counts[HealthStatus.UNHEALTHY] > total_providers * 0.5:
            overall_status = HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0 or status_counts[HealthStatus.UNHEALTHY] > 0:
            overall_status = HealthStatus.DEGRADED
        elif status_counts[HealthStatus.HEALTHY] == total_providers:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        return {
            'overall_status': overall_status.value,
            'total_providers': total_providers,
            'healthy_providers': status_counts[HealthStatus.HEALTHY],
            'degraded_providers': status_counts[HealthStatus.DEGRADED],
            'unhealthy_providers': status_counts[HealthStatus.UNHEALTHY],
            'unknown_providers': status_counts[HealthStatus.UNKNOWN],
            'providers': {key: status.value for key, status in provider_statuses.items()}
        }
    
    def add_alert_handler(self, handler: Callable[[str, AlertSeverity, str], None]):
        """
        Add an alert handler function.
        
        Args:
            handler: Function to handle alerts (provider_key, severity, message)
        """
        self.alert_handlers.append(handler)
        logger.info("Added alert handler")
    
    def add_alert_rule(self, rule: AlertRule):
        """
        Add a custom alert rule.
        
        Args:
            rule: Alert rule to add
        """
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    async def start_monitoring(self):
        """Start continuous monitoring for all registered providers."""
        if not self.providers:
            logger.warning("No providers registered for monitoring")
            return
        
        # Start monitoring tasks for all providers
        for provider_key in self.providers:
            if provider_key not in self._monitoring_tasks:
                self._start_monitoring_task(provider_key)
        
        logger.info(f"Started monitoring {len(self.providers)} providers")
    
    async def stop_monitoring(self):
        """Stop all monitoring tasks."""
        self._shutdown_event.set()
        
        # Cancel all monitoring tasks
        for task in self._monitoring_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks.values(), return_exceptions=True)
        
        self._monitoring_tasks.clear()
        logger.info("Stopped all monitoring tasks")
    
    def _start_monitoring_task(self, provider_key: str):
        """Start monitoring task for a specific provider."""
        if provider_key in self._monitoring_tasks:
            self._monitoring_tasks[provider_key].cancel()
        
        self._monitoring_tasks[provider_key] = asyncio.create_task(
            self._monitoring_loop(provider_key)
        )
    
    async def _monitoring_loop(self, provider_key: str):
        """Continuous monitoring loop for a provider."""
        logger.debug(f"Started monitoring loop for {provider_key}")
        
        try:
            while not self._shutdown_event.is_set():
                # Perform health check
                result = await self.perform_health_check(provider_key)
                
                # Store result in history
                self.check_history[provider_key].append(result)
                
                # Limit history size
                if len(self.check_history[provider_key]) > 100:
                    self.check_history[provider_key] = self.check_history[provider_key][-50:]
                
                # Check alert rules
                await self._check_alerts(provider_key)
                
                # Wait for next check
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.check_interval
                    )
                    break  # Shutdown event was set
                except asyncio.TimeoutError:
                    continue  # Normal timeout, continue monitoring
                    
        except asyncio.CancelledError:
            logger.debug(f"Monitoring loop cancelled for {provider_key}")
        except Exception as e:
            logger.error(f"Monitoring loop error for {provider_key}: {e}")
    
    def _update_metrics(self, provider_key: str, result: HealthCheckResult):
        """Update metrics based on health check result."""
        metrics = self.metrics[provider_key]
        
        metrics.total_checks += 1
        
        if result.status == HealthStatus.HEALTHY:
            metrics.successful_checks += 1
            metrics.consecutive_successes += 1
            metrics.consecutive_failures = 0
            metrics.last_success = result.timestamp
            
        else:
            metrics.failed_checks += 1
            metrics.consecutive_failures += 1
            metrics.consecutive_successes = 0
            metrics.last_failure = result.timestamp
        
        # Update average response time (exponential moving average)
        if metrics.average_response_time == 0:
            metrics.average_response_time = result.response_time_ms
        else:
            metrics.average_response_time = (
                0.8 * metrics.average_response_time + 
                0.2 * result.response_time_ms
            )
        
        # Update uptime percentage
        metrics.uptime_percentage = (metrics.successful_checks / metrics.total_checks) * 100
    
    def _determine_overall_status(self, provider_key: str) -> HealthStatus:
        """Determine overall status based on recent history."""
        if provider_key not in self.check_history:
            return HealthStatus.UNKNOWN
        
        history = self.check_history[provider_key]
        if not history:
            return HealthStatus.UNKNOWN
        
        # Check recent results (last 5 checks)
        recent_checks = history[-5:]
        recent_failures = sum(1 for check in recent_checks if check.status != HealthStatus.HEALTHY)
        
        if recent_failures == 0:
            return HealthStatus.HEALTHY
        elif recent_failures <= 2:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY
    
    async def _check_alerts(self, provider_key: str):
        """Check alert rules for a provider."""
        if provider_key not in self.metrics:
            return
        
        metrics = self.metrics[provider_key]
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown
            if (rule.last_triggered and 
                datetime.utcnow() - rule.last_triggered < timedelta(minutes=rule.cooldown_minutes)):
                continue
            
            # Evaluate condition
            try:
                if rule.condition(metrics):
                    # Trigger alert
                    # Properly escape all format string special characters to prevent injection
                    safe_provider_key = (
                        provider_key
                        .replace("{", "{{")  # Escape opening braces
                        .replace("}", "}}")  # Escape closing braces
                        .replace("%", "%%")  # Escape percent signs
                    )
                    message = rule.message_template.format(
                        provider_key=safe_provider_key,
                        metrics=metrics
                    )
                    
                    await self._send_alert(provider_key, rule.severity, message)
                    rule.last_triggered = datetime.utcnow()
                    
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.name}: {e}")
    
    async def _send_alert(self, provider_key: str, severity: AlertSeverity, message: str):
        """Send alert to all registered handlers."""
        logger.log(
            logging.ERROR if severity == AlertSeverity.CRITICAL else logging.WARNING,
            f"ALERT [{severity.value.upper()}] {provider_key}: {message}"
        )
        
        for handler in self.alert_handlers:
            try:
                # Handle both sync and async handlers
                if asyncio.iscoroutinefunction(handler):
                    await handler(provider_key, severity, message)
                else:
                    handler(provider_key, severity, message)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        self.alert_rules = [
            AlertRule(
                name="High Failure Rate",
                condition=lambda m: m.total_checks > 10 and m.success_rate < 80,
                severity=AlertSeverity.WARNING,
                message_template="Provider {provider_key} has high failure rate: {metrics.success_rate:.1f}%",
                cooldown_minutes=60
            ),
            AlertRule(
                name="Extended Downtime",
                condition=lambda m: m.consecutive_failures >= 5,
                severity=AlertSeverity.ERROR,
                message_template="Provider {provider_key} has been down for {metrics.consecutive_failures} consecutive checks",
                cooldown_minutes=30
            ),
            AlertRule(
                name="Critical Downtime",
                condition=lambda m: m.consecutive_failures >= 10,
                severity=AlertSeverity.CRITICAL,
                message_template="Provider {provider_key} has been down for {metrics.consecutive_failures} consecutive checks - CRITICAL",
                cooldown_minutes=15
            ),
            AlertRule(
                name="Slow Response Time",
                condition=lambda m: m.average_response_time > 10000,  # 10 seconds
                severity=AlertSeverity.WARNING,
                message_template="Provider {provider_key} has slow response time: {metrics.average_response_time:.0f}ms",
                cooldown_minutes=60
            )
        ]