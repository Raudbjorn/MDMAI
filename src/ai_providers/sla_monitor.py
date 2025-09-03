"""
Comprehensive Monitoring and SLA Systems
Task 25.3: Develop Provider Router with Fallback
"""

import asyncio
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Deque, Callable

from structlog import get_logger

from .models import ProviderType, AIRequest, AIResponse
from .health_monitor import HealthMonitor

logger = get_logger(__name__)


class SLAMetric(Enum):
    """SLA metrics to monitor."""
    
    AVAILABILITY = "availability"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"
    SUCCESS_RATE = "success_rate"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    COST_EFFICIENCY = "cost_efficiency"
    QUALITY_SCORE = "quality_score"


class SLAStatus(Enum):
    """SLA compliance status."""
    
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of monitoring alerts."""
    
    SLA_VIOLATION = "sla_violation"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CAPACITY_THRESHOLD = "capacity_threshold"
    ANOMALY_DETECTION = "anomaly_detection"
    HEALTH_CHECK_FAILURE = "health_check_failure"


@dataclass
class SLATarget:
    """SLA target definition."""
    
    metric: SLAMetric
    target_value: float
    threshold_warning: float  # Warning threshold (before violation)
    threshold_critical: float  # Critical threshold
    measurement_window: timedelta  # Time window for measurement
    provider_type: Optional[ProviderType] = None  # None means all providers
    model: Optional[str] = None  # None means all models
    enabled: bool = True
    
    def __post_init__(self):
        """Validate threshold relationships."""
        if self.metric in [SLAMetric.AVAILABILITY, SLAMetric.SUCCESS_RATE, SLAMetric.QUALITY_SCORE]:
            # Higher is better metrics
            assert self.threshold_warning >= self.threshold_critical
        else:
            # Lower is better metrics (latency, error rate)
            assert self.threshold_warning <= self.threshold_critical


@dataclass
class SLAViolation:
    """Record of an SLA violation."""
    
    violation_id: str
    sla_target: SLATarget
    provider_type: ProviderType
    model: Optional[str]
    actual_value: float
    target_value: float
    severity: SLAStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    root_cause: Optional[str] = None
    mitigation_actions: List[str] = field(default_factory=list)
    resolved: bool = False


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics for SLA monitoring."""
    
    provider_type: ProviderType
    model: Optional[str] = None
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Latency tracking
    latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Availability tracking
    uptime_seconds: float = 0.0
    downtime_seconds: float = 0.0
    availability_percentage: float = 100.0
    
    # Throughput tracking
    requests_per_minute: float = 0.0
    tokens_per_minute: float = 0.0
    
    # Cost metrics
    total_cost: float = 0.0
    cost_per_request: float = 0.0
    cost_efficiency: float = 0.0  # requests per dollar
    
    # Quality tracking
    quality_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    avg_quality_score: float = 0.0
    
    # Timestamps
    request_timestamps: Deque[datetime] = field(default_factory=lambda: deque(maxlen=1000))
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class MonitoringAlert:
    """Monitoring system alert."""
    
    alert_id: str
    alert_type: AlertType
    severity: SLAStatus
    provider_type: ProviderType
    model: Optional[str]
    message: str
    details: Dict[str, Any]
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    auto_resolved: bool = False
    escalation_level: int = 0


class SLAMonitor:
    """
    Comprehensive SLA monitoring and alerting system.
    
    Features:
    - Multi-dimensional SLA tracking
    - Real-time performance monitoring
    - Automated violation detection
    - Alert management and escalation
    - Performance trending and analytics
    - Root cause analysis
    - SLA reporting and dashboards
    """
    
    def __init__(
        self,
        health_monitor: HealthMonitor,
        measurement_interval: int = 60,  # seconds
        alert_retention_hours: int = 168,  # 1 week
    ):
        self.health_monitor = health_monitor
        self.measurement_interval = measurement_interval
        self.alert_retention_hours = alert_retention_hours
        
        # SLA configuration
        self.sla_targets: List[SLATarget] = []
        self.performance_metrics: Dict[Tuple[ProviderType, Optional[str]], PerformanceMetrics] = {}
        
        # Violation tracking
        self.active_violations: Dict[str, SLAViolation] = {}
        self.violation_history: List[SLAViolation] = []
        
        # Alert management
        self.active_alerts: Dict[str, MonitoringAlert] = {}
        self.alert_history: List[MonitoringAlert] = []
        
        # Performance tracking
        self.performance_trends: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
        # Anomaly detection
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        self.anomaly_threshold = 2.0  # Standard deviations
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._analytics_task: Optional[asyncio.Task] = None
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[MonitoringAlert], None]] = []
        
        # Initialize default SLA targets
        self._initialize_default_slas()
    
    async def start(self) -> None:
        """Start the SLA monitoring system."""
        logger.info("Starting SLA monitoring system")
        
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._analytics_task = asyncio.create_task(self._analytics_loop())
        
        logger.info("SLA monitoring system started")
    
    async def stop(self) -> None:
        """Stop the SLA monitoring system."""
        logger.info("Stopping SLA monitoring system")
        
        for task in [self._monitoring_task, self._analytics_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("SLA monitoring system stopped")
    
    def add_sla_target(self, target: SLATarget) -> None:
        """Add an SLA target to monitor."""
        self.sla_targets.append(target)
        logger.info(
            "Added SLA target",
            metric=target.metric.value,
            target_value=target.target_value,
            provider=target.provider_type.value if target.provider_type else "all",
        )
    
    def remove_sla_target(self, metric: SLAMetric, provider_type: Optional[ProviderType] = None) -> bool:
        """Remove an SLA target."""
        initial_count = len(self.sla_targets)
        self.sla_targets = [
            target for target in self.sla_targets
            if not (target.metric == metric and target.provider_type == provider_type)
        ]
        
        removed = len(self.sla_targets) < initial_count
        if removed:
            logger.info(
                "Removed SLA target",
                metric=metric.value,
                provider=provider_type.value if provider_type else "all",
            )
        
        return removed
    
    async def record_request(
        self,
        request: AIRequest,
        response: Optional[AIResponse],
        provider_type: ProviderType,
        model: str,
        success: bool,
        latency_ms: float,
        error: Optional[Exception] = None,
    ) -> None:
        """Record a request for SLA monitoring."""
        key = (provider_type, model)
        
        # Ensure metrics exist
        if key not in self.performance_metrics:
            self.performance_metrics[key] = PerformanceMetrics(
                provider_type=provider_type,
                model=model,
            )
        
        metrics = self.performance_metrics[key]
        now = datetime.now()
        
        # Update request counts
        metrics.total_requests += 1
        metrics.request_timestamps.append(now)
        
        if success:
            metrics.successful_requests += 1
            metrics.last_success = now
        else:
            metrics.failed_requests += 1
            metrics.last_failure = now
        
        # Update latency
        metrics.latencies.append(latency_ms)
        
        # Update cost if response available
        if response and response.cost:
            metrics.total_cost += response.cost
        
        # Update quality score if available
        if response and hasattr(response, 'quality_score'):
            metrics.quality_scores.append(response.quality_score)
        
        # Recalculate derived metrics
        await self._update_derived_metrics(metrics)
        
        # Check for immediate SLA violations
        await self._check_immediate_violations(provider_type, model, metrics)
    
    async def _update_derived_metrics(self, metrics: PerformanceMetrics) -> None:
        """Update derived metrics from raw data."""
        now = datetime.now()
        
        # Calculate latency percentiles
        if metrics.latencies:
            sorted_latencies = sorted(metrics.latencies)
            metrics.avg_latency_ms = sum(sorted_latencies) / len(sorted_latencies)
            
            if len(sorted_latencies) >= 20:  # Minimum data for percentiles
                p95_idx = int(len(sorted_latencies) * 0.95)
                p99_idx = int(len(sorted_latencies) * 0.99)
                metrics.p95_latency_ms = sorted_latencies[p95_idx]
                metrics.p99_latency_ms = sorted_latencies[p99_idx]
        
        # Calculate throughput (requests per minute)
        one_minute_ago = now - timedelta(minutes=1)
        recent_requests = [
            ts for ts in metrics.request_timestamps
            if ts >= one_minute_ago
        ]
        metrics.requests_per_minute = len(recent_requests)
        
        # Calculate success rate and availability
        if metrics.total_requests > 0:
            success_rate = metrics.successful_requests / metrics.total_requests
            metrics.availability_percentage = success_rate * 100
        
        # Calculate cost efficiency
        if metrics.total_cost > 0:
            metrics.cost_per_request = metrics.total_cost / metrics.total_requests
            metrics.cost_efficiency = metrics.total_requests / metrics.total_cost
        
        # Calculate average quality score
        if metrics.quality_scores:
            metrics.avg_quality_score = sum(metrics.quality_scores) / len(metrics.quality_scores)
        
        metrics.last_updated = now
    
    async def _check_immediate_violations(
        self,
        provider_type: ProviderType,
        model: str,
        metrics: PerformanceMetrics,
    ) -> None:
        """Check for immediate SLA violations after a request."""
        for target in self.sla_targets:
            if not target.enabled:
                continue
            
            # Check if target applies to this provider/model
            if target.provider_type and target.provider_type != provider_type:
                continue
            if target.model and target.model != model:
                continue
            
            # Get current metric value
            current_value = self._get_metric_value(target.metric, metrics)
            if current_value is None:
                continue
            
            # Check violation status
            violation_status = self._check_violation_status(target, current_value)
            
            if violation_status != SLAStatus.COMPLIANT:
                await self._handle_sla_violation(
                    target, provider_type, model, current_value, violation_status
                )
    
    def _get_metric_value(self, metric: SLAMetric, metrics: PerformanceMetrics) -> Optional[float]:
        """Get the current value of a specific metric."""
        metric_map = {
            SLAMetric.AVAILABILITY: metrics.availability_percentage,
            SLAMetric.LATENCY_P95: metrics.p95_latency_ms,
            SLAMetric.LATENCY_P99: metrics.p99_latency_ms,
            SLAMetric.SUCCESS_RATE: (
                metrics.successful_requests / metrics.total_requests * 100
                if metrics.total_requests > 0 else 100.0
            ),
            SLAMetric.THROUGHPUT: metrics.requests_per_minute,
            SLAMetric.ERROR_RATE: (
                metrics.failed_requests / metrics.total_requests * 100
                if metrics.total_requests > 0 else 0.0
            ),
            SLAMetric.COST_EFFICIENCY: metrics.cost_efficiency,
            SLAMetric.QUALITY_SCORE: metrics.avg_quality_score,
        }
        
        return metric_map.get(metric)
    
    def _check_violation_status(self, target: SLATarget, current_value: float) -> SLAStatus:
        """Check if a metric value violates SLA thresholds."""
        if target.metric in [SLAMetric.AVAILABILITY, SLAMetric.SUCCESS_RATE, 
                           SLAMetric.QUALITY_SCORE, SLAMetric.COST_EFFICIENCY]:
            # Higher is better metrics
            if current_value >= target.target_value:
                return SLAStatus.COMPLIANT
            elif current_value >= target.threshold_warning:
                return SLAStatus.WARNING
            elif current_value >= target.threshold_critical:
                return SLAStatus.VIOLATION
            else:
                return SLAStatus.CRITICAL
        else:
            # Lower is better metrics
            if current_value <= target.target_value:
                return SLAStatus.COMPLIANT
            elif current_value <= target.threshold_warning:
                return SLAStatus.WARNING
            elif current_value <= target.threshold_critical:
                return SLAStatus.VIOLATION
            else:
                return SLAStatus.CRITICAL
    
    async def _handle_sla_violation(
        self,
        target: SLATarget,
        provider_type: ProviderType,
        model: Optional[str],
        actual_value: float,
        severity: SLAStatus,
    ) -> None:
        """Handle an SLA violation."""
        violation_key = f"{target.metric.value}_{provider_type.value}_{model or 'all'}"
        
        # Check if violation already exists
        if violation_key in self.active_violations:
            # Update existing violation
            violation = self.active_violations[violation_key]
            violation.actual_value = actual_value
            violation.severity = severity
        else:
            # Create new violation
            violation = SLAViolation(
                violation_id=f"{violation_key}_{datetime.now().timestamp()}",
                sla_target=target,
                provider_type=provider_type,
                model=model,
                actual_value=actual_value,
                target_value=target.target_value,
                severity=severity,
                start_time=datetime.now(),
            )
            
            self.active_violations[violation_key] = violation
            
            # Create alert
            await self._create_sla_alert(violation)
            
            logger.warning(
                "SLA violation detected",
                metric=target.metric.value,
                provider=provider_type.value,
                model=model,
                actual_value=actual_value,
                target_value=target.target_value,
                severity=severity.value,
            )
    
    async def _create_sla_alert(self, violation: SLAViolation) -> None:
        """Create an alert for an SLA violation."""
        alert = MonitoringAlert(
            alert_id=f"sla_{violation.violation_id}",
            alert_type=AlertType.SLA_VIOLATION,
            severity=violation.severity,
            provider_type=violation.provider_type,
            model=violation.model,
            message=f"SLA violation: {violation.sla_target.metric.value} "
                   f"({violation.actual_value:.2f} vs target {violation.target_value:.2f})",
            details={
                "metric": violation.sla_target.metric.value,
                "actual_value": violation.actual_value,
                "target_value": violation.target_value,
                "threshold_warning": violation.sla_target.threshold_warning,
                "threshold_critical": violation.sla_target.threshold_critical,
                "measurement_window": violation.sla_target.measurement_window.total_seconds(),
            },
            triggered_at=violation.start_time,
        )
        
        self.active_alerts[alert.alert_id] = alert
        
        # Execute alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error("Error executing alert callback", error=str(e))
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.measurement_interval)
                await self._perform_sla_checks()
                await self._detect_anomalies()
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
    
    async def _perform_sla_checks(self) -> None:
        """Perform comprehensive SLA checks."""
        for target in self.sla_targets:
            if not target.enabled:
                continue
            
            # Get relevant metrics
            relevant_metrics = [
                (key, metrics) for key, metrics in self.performance_metrics.items()
                if (target.provider_type is None or key[0] == target.provider_type) and
                   (target.model is None or key[1] == target.model)
            ]
            
            for key, metrics in relevant_metrics:
                provider_type, model = key
                current_value = self._get_metric_value(target.metric, metrics)
                
                if current_value is not None:
                    violation_status = self._check_violation_status(target, current_value)
                    
                    if violation_status != SLAStatus.COMPLIANT:
                        await self._handle_sla_violation(
                            target, provider_type, model, current_value, violation_status
                        )
    
    async def _detect_anomalies(self) -> None:
        """Detect performance anomalies."""
        for key, metrics in self.performance_metrics.items():
            provider_type, model = key
            baseline_key = f"{provider_type.value}_{model or 'all'}"
            
            # Update baseline metrics
            if baseline_key not in self.baseline_metrics:
                self.baseline_metrics[baseline_key] = {}
            
            baseline = self.baseline_metrics[baseline_key]
            
            # Check for latency anomalies
            if metrics.latencies and len(metrics.latencies) >= 10:
                recent_latencies = list(metrics.latencies)[-10:]
                avg_recent = sum(recent_latencies) / len(recent_latencies)
                
                if "avg_latency" in baseline:
                    baseline_avg = baseline["avg_latency"]
                    baseline_std = baseline.get("latency_std", baseline_avg * 0.1)
                    
                    z_score = abs(avg_recent - baseline_avg) / max(baseline_std, 1.0)
                    
                    if z_score > self.anomaly_threshold:
                        await self._create_anomaly_alert(
                            provider_type, model, "latency", avg_recent, baseline_avg
                        )
                
                # Update baseline
                all_latencies = list(metrics.latencies)
                baseline["avg_latency"] = sum(all_latencies) / len(all_latencies)
                if len(all_latencies) > 1:
                    baseline["latency_std"] = statistics.stdev(all_latencies)
    
    async def _create_anomaly_alert(
        self,
        provider_type: ProviderType,
        model: Optional[str],
        metric_name: str,
        current_value: float,
        baseline_value: float,
    ) -> None:
        """Create an anomaly detection alert."""
        alert = MonitoringAlert(
            alert_id=f"anomaly_{provider_type.value}_{model or 'all'}_{metric_name}_{datetime.now().timestamp()}",
            alert_type=AlertType.ANOMALY_DETECTION,
            severity=SLAStatus.WARNING,
            provider_type=provider_type,
            model=model,
            message=f"Performance anomaly detected in {metric_name}: "
                   f"current {current_value:.2f} vs baseline {baseline_value:.2f}",
            details={
                "metric": metric_name,
                "current_value": current_value,
                "baseline_value": baseline_value,
                "deviation": abs(current_value - baseline_value) / baseline_value * 100,
            },
            triggered_at=datetime.now(),
        )
        
        self.active_alerts[alert.alert_id] = alert
        
        logger.info(
            "Performance anomaly detected",
            provider=provider_type.value,
            model=model,
            metric=metric_name,
            current=current_value,
            baseline=baseline_value,
        )
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old alerts and violations."""
        cutoff = datetime.now() - timedelta(hours=self.alert_retention_hours)
        
        # Clean up resolved violations
        resolved_violations = [
            key for key, violation in self.active_violations.items()
            if violation.resolved and violation.end_time and violation.end_time < cutoff
        ]
        
        for key in resolved_violations:
            violation = self.active_violations.pop(key)
            self.violation_history.append(violation)
        
        # Clean up old alerts
        old_alerts = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.resolved_at and alert.resolved_at < cutoff
        ]
        
        for alert_id in old_alerts:
            alert = self.active_alerts.pop(alert_id)
            self.alert_history.append(alert)
        
        # Limit history size
        max_history = 1000
        if len(self.violation_history) > max_history:
            self.violation_history = self.violation_history[-max_history:]
        if len(self.alert_history) > max_history:
            self.alert_history = self.alert_history[-max_history:]
    
    async def _analytics_loop(self) -> None:
        """Background analytics and reporting loop."""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                await self._update_performance_trends()
                await self._generate_insights()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in analytics loop", error=str(e))
    
    async def _update_performance_trends(self) -> None:
        """Update performance trend data."""
        now = datetime.now()
        
        for key, metrics in self.performance_metrics.items():
            provider_type, model = key
            trend_key = f"{provider_type.value}_{model or 'all'}"
            
            # Record current performance metrics
            trend_data = [
                (now, metrics.avg_latency_ms),
                (now, metrics.availability_percentage),
                (now, metrics.requests_per_minute),
                (now, metrics.cost_efficiency),
            ]
            
            for timestamp, value in trend_data:
                if trend_key not in self.performance_trends:
                    self.performance_trends[trend_key] = []
                
                self.performance_trends[trend_key].append((timestamp, value))
                
                # Keep only recent data (last 24 hours)
                cutoff = now - timedelta(hours=24)
                self.performance_trends[trend_key] = [
                    (ts, val) for ts, val in self.performance_trends[trend_key]
                    if ts >= cutoff
                ]
    
    async def _generate_insights(self) -> None:
        """Generate performance insights and recommendations."""
        # Placeholder for advanced analytics
        # Could include trend analysis, pattern recognition, etc.
        pass
    
    def _initialize_default_slas(self) -> None:
        """Initialize default SLA targets."""
        default_targets = [
            SLATarget(
                metric=SLAMetric.AVAILABILITY,
                target_value=99.0,  # 99% availability
                threshold_warning=98.0,
                threshold_critical=95.0,
                measurement_window=timedelta(hours=1),
            ),
            SLATarget(
                metric=SLAMetric.LATENCY_P95,
                target_value=5000.0,  # 5 seconds
                threshold_warning=8000.0,
                threshold_critical=15000.0,
                measurement_window=timedelta(minutes=15),
            ),
            SLATarget(
                metric=SLAMetric.SUCCESS_RATE,
                target_value=95.0,  # 95% success rate
                threshold_warning=90.0,
                threshold_critical=85.0,
                measurement_window=timedelta(hours=1),
            ),
        ]
        
        self.sla_targets.extend(default_targets)
    
    def add_alert_callback(self, callback: Callable[[MonitoringAlert], None]) -> None:
        """Add a callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info("Alert acknowledged", alert_id=alert_id)
            return True
        return False
    
    def resolve_alert(self, alert_id: str, auto_resolved: bool = False) -> bool:
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.now()
            alert.auto_resolved = auto_resolved
            logger.info("Alert resolved", alert_id=alert_id, auto_resolved=auto_resolved)
            return True
        return False
    
    def get_sla_compliance_report(
        self, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get comprehensive SLA compliance report."""
        if not start_time:
            start_time = datetime.now() - timedelta(days=1)
        if not end_time:
            end_time = datetime.now()
        
        # Calculate compliance for each target
        compliance_data = []
        
        for target in self.sla_targets:
            if not target.enabled:
                continue
            
            # Find violations in the time period
            relevant_violations = [
                v for v in self.violation_history + list(self.active_violations.values())
                if (v.sla_target.metric == target.metric and
                    v.sla_target.provider_type == target.provider_type and
                    v.start_time >= start_time and v.start_time <= end_time)
            ]
            
            total_violation_time = sum(
                (v.duration or timedelta()).total_seconds()
                for v in relevant_violations if v.duration
            )
            
            period_seconds = (end_time - start_time).total_seconds()
            compliance_percentage = (
                (period_seconds - total_violation_time) / period_seconds * 100
                if period_seconds > 0 else 100.0
            )
            
            compliance_data.append({
                "metric": target.metric.value,
                "provider": target.provider_type.value if target.provider_type else "all",
                "model": target.model or "all",
                "target_value": target.target_value,
                "compliance_percentage": compliance_percentage,
                "violations": len(relevant_violations),
                "total_violation_seconds": total_violation_time,
            })
        
        # Overall statistics
        active_alerts_count = len(self.active_alerts)
        active_violations_count = len(self.active_violations)
        
        return {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_hours": (end_time - start_time).total_seconds() / 3600,
            },
            "summary": {
                "total_sla_targets": len(self.sla_targets),
                "active_violations": active_violations_count,
                "active_alerts": active_alerts_count,
                "overall_compliance": (
                    sum(data["compliance_percentage"] for data in compliance_data) /
                    len(compliance_data) if compliance_data else 100.0
                ),
            },
            "sla_targets": compliance_data,
            "performance_metrics": {
                f"{key[0].value}_{key[1] or 'all'}": {
                    "total_requests": metrics.total_requests,
                    "success_rate": (
                        metrics.successful_requests / metrics.total_requests * 100
                        if metrics.total_requests > 0 else 100.0
                    ),
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "availability_percentage": metrics.availability_percentage,
                    "cost_per_request": metrics.cost_per_request,
                }
                for key, metrics in self.performance_metrics.items()
            }
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts."""
        return [
            {
                "alert_id": alert.alert_id,
                "type": alert.alert_type.value,
                "severity": alert.severity.value,
                "provider": alert.provider_type.value,
                "model": alert.model,
                "message": alert.message,
                "details": alert.details,
                "triggered_at": alert.triggered_at.isoformat(),
                "acknowledged": alert.acknowledged,
                "escalation_level": alert.escalation_level,
            }
            for alert in self.active_alerts.values()
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all providers."""
        total_requests = sum(m.total_requests for m in self.performance_metrics.values())
        total_successful = sum(m.successful_requests for m in self.performance_metrics.values())
        total_cost = sum(m.total_cost for m in self.performance_metrics.values())
        
        # Average metrics across providers
        avg_latency = statistics.mean([
            m.avg_latency_ms for m in self.performance_metrics.values() 
            if m.avg_latency_ms > 0
        ]) if self.performance_metrics else 0.0
        
        avg_availability = statistics.mean([
            m.availability_percentage for m in self.performance_metrics.values()
        ]) if self.performance_metrics else 100.0
        
        return {
            "total_requests": total_requests,
            "overall_success_rate": (
                total_successful / total_requests * 100 if total_requests > 0 else 100.0
            ),
            "average_latency_ms": avg_latency,
            "average_availability": avg_availability,
            "total_cost": total_cost,
            "cost_per_request": total_cost / total_requests if total_requests > 0 else 0.0,
            "active_providers": len(set(key[0] for key in self.performance_metrics.keys())),
            "monitored_models": len(self.performance_metrics),
        }