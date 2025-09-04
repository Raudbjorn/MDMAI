"""
MCP Health Monitoring System for Provider Router.

This module implements comprehensive health monitoring, status tracking,
and alerting for AI providers in the MCP ecosystem.

Key Features:
- Real-time provider health monitoring
- Performance metrics collection
- Health trend analysis
- Automated alerting and notifications
- Health-based routing decisions
- Provider availability tracking
- Service level monitoring
"""

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from structlog import get_logger

from .mcp_protocol_schemas import (
    ProviderEventType,
    create_notification,
    JSONRPCNotification
)
from .models import ProviderStatus, ProviderHealth

logger = get_logger(__name__)


# Health Status and Metrics
class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class HealthMetricType(str, Enum):
    """Types of health metrics."""
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    THROUGHPUT = "throughput"
    AVAILABILITY = "availability"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"
    REQUEST_VOLUME = "request_volume"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Health Models
class HealthMetric(BaseModel):
    """Individual health metric."""
    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    unit: str = Field(..., description="Metric unit")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    threshold_min: Optional[float] = Field(None, description="Minimum acceptable value")
    threshold_max: Optional[float] = Field(None, description="Maximum acceptable value")
    is_healthy: bool = Field(default=True, description="Whether metric is within healthy range")


class ProviderHealthStatus(BaseModel):
    """Comprehensive provider health status."""
    provider_name: str = Field(..., description="Provider identifier")
    overall_status: HealthStatus = Field(..., description="Overall health status")
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())
    uptime_percentage: float = Field(..., ge=0.0, le=100.0, description="Uptime percentage")
    metrics: Dict[str, HealthMetric] = Field(default_factory=dict, description="Health metrics")
    alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Active alerts")
    health_score: float = Field(..., ge=0.0, le=1.0, description="Computed health score")
    last_successful_request: Optional[str] = Field(None, description="Last successful request timestamp")
    last_failed_request: Optional[str] = Field(None, description="Last failed request timestamp")
    consecutive_failures: int = Field(default=0, description="Consecutive failure count")
    availability_windows: List[Dict[str, Any]] = Field(default_factory=list, description="Recent availability windows")


class HealthThreshold(BaseModel):
    """Health threshold configuration."""
    metric_type: HealthMetricType = Field(..., description="Type of metric")
    warning_threshold: float = Field(..., description="Warning level threshold")
    critical_threshold: float = Field(..., description="Critical level threshold")
    comparison_operator: str = Field(..., description="Comparison operator (gt, lt, eq)")
    evaluation_window: int = Field(default=300, description="Evaluation window in seconds")
    alert_after_violations: int = Field(default=3, description="Alert after N violations")


class HealthAlert(BaseModel):
    """Health alert model."""
    alert_id: str = Field(..., description="Unique alert identifier")
    provider_name: str = Field(..., description="Provider name")
    metric_type: HealthMetricType = Field(..., description="Metric that triggered alert")
    severity: AlertSeverity = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    current_value: float = Field(..., description="Current metric value")
    threshold_value: float = Field(..., description="Threshold that was violated")
    first_triggered: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_triggered: str = Field(default_factory=lambda: datetime.now().isoformat())
    times_triggered: int = Field(default=1, description="Number of times triggered")
    is_active: bool = Field(default=True, description="Whether alert is active")
    auto_resolve: bool = Field(default=True, description="Whether alert auto-resolves")


# Health Monitoring System
class ProviderHealthMonitor:
    """Comprehensive health monitoring system for providers."""
    
    def __init__(
        self,
        check_interval: int = 60,
        metric_retention_hours: int = 24,
        alert_retention_hours: int = 168  # 1 week
    ):
        self.check_interval = check_interval
        self.metric_retention_hours = metric_retention_hours
        self.alert_retention_hours = alert_retention_hours
        
        # Health data storage
        self.provider_health: Dict[str, ProviderHealthStatus] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_timeseries: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1440)))  # 24h at 1min intervals
        
        # Alert management
        self.active_alerts: Dict[str, HealthAlert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.alert_callbacks: List[Callable] = []
        
        # Thresholds and configuration
        self.health_thresholds: Dict[str, List[HealthThreshold]] = defaultdict(list)
        self.provider_configs: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring: bool = False
        
        # Default thresholds
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self):
        """Setup default health thresholds."""
        default_thresholds = [
            HealthThreshold(
                metric_type=HealthMetricType.RESPONSE_TIME,
                warning_threshold=2000,  # 2 seconds
                critical_threshold=5000,  # 5 seconds
                comparison_operator="gt",
                evaluation_window=300,
                alert_after_violations=3
            ),
            HealthThreshold(
                metric_type=HealthMetricType.ERROR_RATE,
                warning_threshold=0.05,  # 5%
                critical_threshold=0.15,  # 15%
                comparison_operator="gt",
                evaluation_window=300,
                alert_after_violations=2
            ),
            HealthThreshold(
                metric_type=HealthMetricType.SUCCESS_RATE,
                warning_threshold=0.95,  # 95%
                critical_threshold=0.85,  # 85%
                comparison_operator="lt",
                evaluation_window=600,
                alert_after_violations=3
            ),
            HealthThreshold(
                metric_type=HealthMetricType.AVAILABILITY,
                warning_threshold=0.99,  # 99%
                critical_threshold=0.95,  # 95%
                comparison_operator="lt",
                evaluation_window=3600,  # 1 hour
                alert_after_violations=1
            )
        ]
        
        # Apply default thresholds to all providers
        for threshold in default_thresholds:
            self.health_thresholds["*"].append(threshold)
    
    async def start_monitoring(self):
        """Start the health monitoring system."""
        if self.is_monitoring:
            logger.warning("Health monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info(
            "Health monitoring started",
            check_interval=self.check_interval,
            retention_hours=self.metric_retention_hours
        )
    
    async def stop_monitoring(self):
        """Stop the health monitoring system."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Perform health checks for all registered providers
                await self._perform_health_checks()
                
                # Evaluate thresholds and trigger alerts
                await self._evaluate_health_thresholds()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                # Wait for next check interval
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health monitoring loop", error=str(e))
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_checks(self):
        """Perform health checks on all providers."""
        check_timestamp = datetime.now()
        
        for provider_name in self.provider_configs.keys():
            try:
                # Get current health metrics
                metrics = await self._collect_provider_metrics(provider_name)
                
                # Update provider health status
                await self._update_provider_health(provider_name, metrics, check_timestamp)
                
                # Store metrics in time series
                await self._store_metrics_timeseries(provider_name, metrics, check_timestamp)
                
            except Exception as e:
                logger.error(
                    "Health check failed for provider",
                    provider=provider_name,
                    error=str(e)
                )
                
                # Record health check failure
                await self._record_health_check_failure(provider_name, str(e))
    
    async def _collect_provider_metrics(self, provider_name: str) -> Dict[str, HealthMetric]:
        """Collect health metrics for a provider."""
        # This would integrate with the actual provider to collect real metrics
        # For now, we'll return simulated metrics
        
        import random
        
        metrics = {}
        
        # Simulate response time
        base_response_time = 800 + random.gauss(0, 200)
        metrics[HealthMetricType.RESPONSE_TIME.value] = HealthMetric(
            name="Response Time",
            value=max(0, base_response_time),
            unit="ms",
            threshold_max=2000
        )
        
        # Simulate error rate
        base_error_rate = 0.02 + random.gauss(0, 0.01)
        metrics[HealthMetricType.ERROR_RATE.value] = HealthMetric(
            name="Error Rate",
            value=max(0, min(1, base_error_rate)),
            unit="ratio",
            threshold_max=0.05
        )
        
        # Simulate success rate
        success_rate = 1.0 - base_error_rate
        metrics[HealthMetricType.SUCCESS_RATE.value] = HealthMetric(
            name="Success Rate",
            value=max(0, min(1, success_rate)),
            unit="ratio",
            threshold_min=0.95
        )
        
        # Simulate throughput
        base_throughput = 50 + random.gauss(0, 10)
        metrics[HealthMetricType.THROUGHPUT.value] = HealthMetric(
            name="Throughput",
            value=max(0, base_throughput),
            unit="req/min",
            threshold_min=10
        )
        
        return metrics
    
    async def _update_provider_health(
        self,
        provider_name: str,
        metrics: Dict[str, HealthMetric],
        timestamp: datetime
    ):
        """Update provider health status based on metrics."""
        
        # Calculate health score
        health_score = self._calculate_health_score(metrics)
        
        # Determine overall status
        overall_status = self._determine_health_status(health_score, metrics)
        
        # Update or create provider health status
        if provider_name not in self.provider_health:
            self.provider_health[provider_name] = ProviderHealthStatus(
                provider_name=provider_name,
                overall_status=overall_status,
                health_score=health_score,
                uptime_percentage=100.0,
                metrics=metrics
            )
        else:
            current_health = self.provider_health[provider_name]
            previous_status = current_health.overall_status
            
            # Update health status
            current_health.overall_status = overall_status
            current_health.health_score = health_score
            current_health.metrics = metrics
            current_health.last_updated = timestamp.isoformat()
            
            # Update uptime calculation
            current_health.uptime_percentage = self._calculate_uptime(provider_name)
            
            # Record successful request if healthy
            if overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                current_health.last_successful_request = timestamp.isoformat()
                current_health.consecutive_failures = 0
            else:
                current_health.last_failed_request = timestamp.isoformat()
                current_health.consecutive_failures += 1
            
            # Emit status change event if changed
            if previous_status != overall_status:
                await self._emit_health_change_event(
                    provider_name, previous_status, overall_status, metrics
                )
        
        # Store in history
        self.health_history[provider_name].append({
            "timestamp": timestamp.isoformat(),
            "status": overall_status.value,
            "health_score": health_score,
            "metrics": {k: v.value for k, v in metrics.items()}
        })
    
    def _calculate_health_score(self, metrics: Dict[str, HealthMetric]) -> float:
        """Calculate overall health score from metrics."""
        score = 0.0
        total_weight = 0.0
        
        # Metric weights
        weights = {
            HealthMetricType.SUCCESS_RATE.value: 0.4,
            HealthMetricType.RESPONSE_TIME.value: 0.3,
            HealthMetricType.ERROR_RATE.value: 0.2,
            HealthMetricType.THROUGHPUT.value: 0.1
        }
        
        for metric_name, metric in metrics.items():
            weight = weights.get(metric_name, 0.1)
            metric_score = self._calculate_metric_score(metric)
            
            score += metric_score * weight
            total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_metric_score(self, metric: HealthMetric) -> float:
        """Calculate score for individual metric."""
        if metric.name == HealthMetricType.SUCCESS_RATE.value:
            return metric.value  # Already 0-1
        elif metric.name == HealthMetricType.ERROR_RATE.value:
            return max(0, 1.0 - (metric.value / 0.1))  # Inverse, 10% error = 0 score
        elif metric.name == HealthMetricType.RESPONSE_TIME.value:
            # Good if under 1000ms, bad if over 5000ms
            return max(0, min(1, (5000 - metric.value) / 4000))
        elif metric.name == HealthMetricType.THROUGHPUT.value:
            # Scale throughput to 0-1 (assuming 100 req/min is perfect)
            return min(1, metric.value / 100)
        else:
            return 0.5  # Neutral for unknown metrics
    
    def _determine_health_status(self, health_score: float, metrics: Dict[str, HealthMetric]) -> HealthStatus:
        """Determine overall health status from score and metrics."""
        # Check for critical conditions
        for metric in metrics.values():
            if metric.name == HealthMetricType.SUCCESS_RATE.value and metric.value < 0.5:
                return HealthStatus.CRITICAL
            if metric.name == HealthMetricType.ERROR_RATE.value and metric.value > 0.5:
                return HealthStatus.CRITICAL
        
        # Use health score
        if health_score >= 0.9:
            return HealthStatus.HEALTHY
        elif health_score >= 0.7:
            return HealthStatus.DEGRADED
        elif health_score >= 0.5:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.CRITICAL
    
    def _calculate_uptime(self, provider_name: str) -> float:
        """Calculate uptime percentage for provider."""
        if provider_name not in self.health_history:
            return 100.0
        
        history = list(self.health_history[provider_name])
        if not history:
            return 100.0
        
        # Calculate uptime over last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_history = [
            h for h in history
            if datetime.fromisoformat(h["timestamp"]) >= cutoff_time
        ]
        
        if not recent_history:
            return 100.0
        
        healthy_count = sum(
            1 for h in recent_history
            if h["status"] in [HealthStatus.HEALTHY.value, HealthStatus.DEGRADED.value]
        )
        
        return (healthy_count / len(recent_history)) * 100
    
    async def _store_metrics_timeseries(
        self,
        provider_name: str,
        metrics: Dict[str, HealthMetric],
        timestamp: datetime
    ):
        """Store metrics in time series for trending."""
        for metric_name, metric in metrics.items():
            self.metric_timeseries[provider_name][metric_name].append({
                "timestamp": timestamp.isoformat(),
                "value": metric.value
            })
    
    async def _evaluate_health_thresholds(self):
        """Evaluate health thresholds and trigger alerts."""
        for provider_name, health_status in self.provider_health.items():
            # Get thresholds for this provider (specific + global)
            thresholds = (
                self.health_thresholds.get(provider_name, []) +
                self.health_thresholds.get("*", [])
            )
            
            for threshold in thresholds:
                await self._evaluate_threshold(provider_name, threshold, health_status.metrics)
    
    async def _evaluate_threshold(
        self,
        provider_name: str,
        threshold: HealthThreshold,
        metrics: Dict[str, HealthMetric]
    ):
        """Evaluate a specific threshold against current metrics."""
        metric_name = threshold.metric_type.value
        
        if metric_name not in metrics:
            return
        
        current_metric = metrics[metric_name]
        current_value = current_metric.value
        
        # Check if threshold is violated
        violated = self._check_threshold_violation(
            current_value,
            threshold.warning_threshold,
            threshold.comparison_operator
        )
        
        critical_violated = self._check_threshold_violation(
            current_value,
            threshold.critical_threshold,
            threshold.comparison_operator
        )
        
        # Generate alert if threshold violated
        if violated or critical_violated:
            severity = AlertSeverity.CRITICAL if critical_violated else AlertSeverity.WARNING
            threshold_value = threshold.critical_threshold if critical_violated else threshold.warning_threshold
            
            await self._generate_alert(
                provider_name=provider_name,
                metric_type=threshold.metric_type,
                severity=severity,
                current_value=current_value,
                threshold_value=threshold_value,
                message=f"{metric_name} {threshold.comparison_operator} {threshold_value}"
            )
    
    def _check_threshold_violation(self, value: float, threshold: float, operator: str) -> bool:
        """Check if value violates threshold."""
        if operator == "gt":
            return value > threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "eq":
            return value == threshold
        elif operator == "gte":
            return value >= threshold
        elif operator == "lte":
            return value <= threshold
        else:
            return False
    
    async def _generate_alert(
        self,
        provider_name: str,
        metric_type: HealthMetricType,
        severity: AlertSeverity,
        current_value: float,
        threshold_value: float,
        message: str
    ):
        """Generate and manage health alert."""
        import uuid
        
        # Create unique alert key
        alert_key = f"{provider_name}:{metric_type.value}:{severity.value}"
        
        # Check if alert already exists
        if alert_key in self.active_alerts:
            # Update existing alert
            existing_alert = self.active_alerts[alert_key]
            existing_alert.last_triggered = datetime.now().isoformat()
            existing_alert.times_triggered += 1
            existing_alert.current_value = current_value
        else:
            # Create new alert
            alert = HealthAlert(
                alert_id=str(uuid.uuid4()),
                provider_name=provider_name,
                metric_type=metric_type,
                severity=severity,
                message=message,
                current_value=current_value,
                threshold_value=threshold_value
            )
            
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)
            
            # Notify alert callbacks
            await self._notify_alert_callbacks(alert)
            
            # Emit alert event
            await self._emit_alert_event(alert)
    
    async def _emit_health_change_event(
        self,
        provider_name: str,
        previous_status: HealthStatus,
        current_status: HealthStatus,
        metrics: Dict[str, HealthMetric]
    ):
        """Emit health status change event."""
        event_data = {
            "provider_name": provider_name,
            "previous_status": previous_status.value,
            "current_status": current_status.value,
            "health_metrics": {
                name: {
                    "value": metric.value,
                    "unit": metric.unit,
                    "is_healthy": metric.is_healthy
                }
                for name, metric in metrics.items()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        notification = create_notification(
            method=f"health/{ProviderEventType.PROVIDER_HEALTH_CHANGED.value}",
            params=event_data
        )
        
        logger.info(
            "Provider health status changed",
            provider=provider_name,
            previous=previous_status.value,
            current=current_status.value
        )
    
    async def _emit_alert_event(self, alert: HealthAlert):
        """Emit health alert event."""
        event_data = {
            "alert_id": alert.alert_id,
            "provider_name": alert.provider_name,
            "metric_type": alert.metric_type.value,
            "severity": alert.severity.value,
            "message": alert.message,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold_value,
            "timestamp": alert.first_triggered
        }
        
        notification = create_notification(
            method=f"health/alert_{alert.severity.value}",
            params=event_data
        )
        
        logger.warning(
            "Health alert triggered",
            provider=alert.provider_name,
            metric=alert.metric_type.value,
            severity=alert.severity.value,
            value=alert.current_value
        )
    
    async def _notify_alert_callbacks(self, alert: HealthAlert):
        """Notify registered alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error("Alert callback failed", error=str(e))
    
    async def _record_health_check_failure(self, provider_name: str, error_message: str):
        """Record a health check failure."""
        if provider_name in self.provider_health:
            self.provider_health[provider_name].overall_status = HealthStatus.UNKNOWN
            self.provider_health[provider_name].consecutive_failures += 1
        
        logger.error(
            "Health check failure",
            provider=provider_name,
            error=error_message
        )
    
    async def _cleanup_old_data(self):
        """Clean up old health data and metrics."""
        cutoff_time = datetime.now() - timedelta(hours=self.metric_retention_hours)
        alert_cutoff = datetime.now() - timedelta(hours=self.alert_retention_hours)
        
        # Clean up health history
        for provider_name, history in self.health_history.items():
            while history and datetime.fromisoformat(history[0]["timestamp"]) < cutoff_time:
                history.popleft()
        
        # Clean up metric timeseries
        for provider_name, metrics in self.metric_timeseries.items():
            for metric_name, timeseries in metrics.items():
                while timeseries and datetime.fromisoformat(timeseries[0]["timestamp"]) < cutoff_time:
                    timeseries.popleft()
        
        # Clean up old alerts
        while self.alert_history and datetime.fromisoformat(self.alert_history[0].first_triggered) < alert_cutoff:
            self.alert_history.popleft()
        
        # Remove inactive alerts older than cutoff
        inactive_alerts = [
            key for key, alert in self.active_alerts.items()
            if not alert.is_active and datetime.fromisoformat(alert.last_triggered) < alert_cutoff
        ]
        
        for key in inactive_alerts:
            del self.active_alerts[key]
    
    # Public API Methods
    def register_provider(self, provider_name: str, config: Dict[str, Any] = None):
        """Register a provider for health monitoring."""
        self.provider_configs[provider_name] = config or {}
        
        logger.info("Provider registered for health monitoring", provider=provider_name)
    
    def unregister_provider(self, provider_name: str):
        """Unregister a provider from health monitoring."""
        if provider_name in self.provider_configs:
            del self.provider_configs[provider_name]
        
        if provider_name in self.provider_health:
            del self.provider_health[provider_name]
        
        logger.info("Provider unregistered from health monitoring", provider=provider_name)
    
    def add_health_threshold(self, provider_name: str, threshold: HealthThreshold):
        """Add health threshold for specific provider."""
        self.health_thresholds[provider_name].append(threshold)
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for health alerts."""
        self.alert_callbacks.append(callback)
    
    def get_provider_health(self, provider_name: str) -> Optional[ProviderHealthStatus]:
        """Get current health status for provider."""
        return self.provider_health.get(provider_name)
    
    def get_all_provider_health(self) -> Dict[str, ProviderHealthStatus]:
        """Get health status for all providers."""
        return self.provider_health.copy()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        total_providers = len(self.provider_health)
        healthy_providers = sum(
            1 for health in self.provider_health.values()
            if health.overall_status == HealthStatus.HEALTHY
        )
        
        degraded_providers = sum(
            1 for health in self.provider_health.values()
            if health.overall_status == HealthStatus.DEGRADED
        )
        
        unhealthy_providers = sum(
            1 for health in self.provider_health.values()
            if health.overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
        )
        
        return {
            "total_providers": total_providers,
            "healthy_providers": healthy_providers,
            "degraded_providers": degraded_providers,
            "unhealthy_providers": unhealthy_providers,
            "overall_health_percentage": (healthy_providers + degraded_providers * 0.5) / total_providers * 100 if total_providers > 0 else 100,
            "active_alerts": len(self.active_alerts),
            "critical_alerts": sum(1 for alert in self.active_alerts.values() if alert.severity == AlertSeverity.CRITICAL),
            "last_updated": datetime.now().isoformat()
        }
    
    def get_provider_metrics_history(
        self,
        provider_name: str,
        metric_type: str,
        hours: int = 1
    ) -> List[Dict[str, Any]]:
        """Get historical metrics for provider."""
        if provider_name not in self.metric_timeseries:
            return []
        
        if metric_type not in self.metric_timeseries[provider_name]:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        timeseries = self.metric_timeseries[provider_name][metric_type]
        
        return [
            point for point in timeseries
            if datetime.fromisoformat(point["timestamp"]) >= cutoff_time
        ]


# Export main classes
__all__ = [
    "ProviderHealthMonitor",
    "ProviderHealthStatus",
    "HealthMetric",
    "HealthThreshold",
    "HealthAlert",
    "HealthStatus",
    "HealthMetricType",
    "AlertSeverity"
]