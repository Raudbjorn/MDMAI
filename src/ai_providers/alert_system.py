"""Comprehensive alerting system with thresholds, notifications, and escalation."""

import asyncio
import json
import aiosmtplib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
from collections import defaultdict, deque
import aiohttp
import aiofiles
from structlog import get_logger

from .models import ProviderType
from .user_usage_tracker import UserUsageTracker, UserUsageAlert
from .metrics_collector import MetricsCollector, MetricSnapshot, AggregatedMetric
from .budget_enforcer import BudgetEnforcer, EnforcementAction

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of alerts."""
    
    BUDGET_THRESHOLD = "budget_threshold"
    COST_ANOMALY = "cost_anomaly"
    ERROR_RATE = "error_rate"
    LATENCY_SPIKE = "latency_spike"
    PROVIDER_FAILURE = "provider_failure"
    QUOTA_EXCEEDED = "quota_exceeded"
    USAGE_SPIKE = "usage_spike"
    SECURITY_THREAT = "security_threat"
    SYSTEM_HEALTH = "system_health"
    CUSTOM_METRIC = "custom_metric"


class NotificationChannel(Enum):
    """Notification delivery channels."""
    
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DISCORD = "discord"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"
    LOG_ONLY = "log_only"


class AlertStatus(Enum):
    """Alert lifecycle status."""
    
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    SUPPRESSED = "suppressed"


@dataclass
class AlertThreshold:
    """Alert threshold configuration."""
    
    threshold_id: str
    name: str
    description: str
    alert_type: AlertType
    severity: AlertSeverity
    
    # Threshold conditions
    metric_name: str
    operator: str  # >, <, >=, <=, ==, !=
    value: float
    time_window: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    
    # Evaluation settings
    consecutive_breaches: int = 1  # How many consecutive evaluations must breach
    evaluation_interval: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    cooldown_period: timedelta = field(default_factory=lambda: timedelta(minutes=15))
    
    # Filtering
    filters: Dict[str, Any] = field(default_factory=dict)  # user_id, provider, etc.
    
    # Notification settings
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    escalation_after: Optional[timedelta] = None
    escalation_channels: List[NotificationChannel] = field(default_factory=list)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    enabled: bool = True
    
    def matches_filters(self, context: Dict[str, Any]) -> bool:
        """Check if alert context matches threshold filters."""
        if not self.filters:
            return True
        
        for filter_key, filter_value in self.filters.items():
            if filter_key not in context:
                return False
            if context[filter_key] != filter_value:
                return False
        
        return True


@dataclass
class Alert:
    """Active alert instance."""
    
    alert_id: str
    threshold_id: str
    alert_type: AlertType
    severity: AlertSeverity
    status: AlertStatus
    
    # Alert details
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold_value: float
    
    # Context information
    user_id: Optional[str] = None
    provider_type: Optional[str] = None
    model: Optional[str] = None
    session_id: Optional[str] = None
    
    # Timestamps
    first_occurrence: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_occurrence: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Escalation tracking
    escalation_level: int = 0
    escalated_at: Optional[datetime] = None
    
    # Occurrence tracking
    occurrence_count: int = 1
    breach_count: int = 1
    
    # Notification tracking
    notifications_sent: List[str] = field(default_factory=list)  # Channel names
    
    # Additional metadata
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "threshold_id": self.threshold_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "user_id": self.user_id,
            "provider_type": self.provider_type,
            "model": self.model,
            "session_id": self.session_id,
            "first_occurrence": self.first_occurrence.isoformat(),
            "last_occurrence": self.last_occurrence.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "escalation_level": self.escalation_level,
            "escalated_at": self.escalated_at.isoformat() if self.escalated_at else None,
            "occurrence_count": self.occurrence_count,
            "breach_count": self.breach_count,
            "notifications_sent": self.notifications_sent,
            "context": self.context,
            "tags": self.tags
        }


@dataclass
class NotificationConfig:
    """Configuration for notification channels."""
    
    channel: NotificationChannel
    enabled: bool = True
    
    # Email configuration
    email_host: Optional[str] = None
    email_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    email_from: Optional[str] = None
    email_to: List[str] = field(default_factory=list)
    email_use_tls: bool = True
    
    # Slack configuration
    slack_webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None
    slack_token: Optional[str] = None
    
    # Webhook configuration
    webhook_url: Optional[str] = None
    webhook_method: str = "POST"
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    webhook_auth: Optional[Dict[str, str]] = None
    
    # SMS configuration (via various providers)
    sms_provider: Optional[str] = None  # twilio, aws_sns, etc.
    sms_config: Dict[str, str] = field(default_factory=dict)
    sms_recipients: List[str] = field(default_factory=list)
    
    # Rate limiting
    rate_limit_per_hour: int = 100
    rate_limit_per_day: int = 500
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: timedelta = field(default_factory=lambda: timedelta(minutes=1))


class AlertSystem:
    """Comprehensive alerting system with intelligent thresholds and notifications."""
    
    def __init__(
        self,
        usage_tracker: UserUsageTracker,
        metrics_collector: MetricsCollector,
        budget_enforcer: BudgetEnforcer,
        storage_path: Optional[str] = None
    ):
        self.usage_tracker = usage_tracker
        self.metrics_collector = metrics_collector
        self.budget_enforcer = budget_enforcer
        
        self.storage_path = Path(storage_path) if storage_path else Path("./data/alerts")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Alert configuration
        self.thresholds: Dict[str, AlertThreshold] = {}
        self.notification_configs: Dict[NotificationChannel, NotificationConfig] = {}
        
        # Active alerts
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Evaluation state
        self.threshold_states: Dict[str, Dict[str, Any]] = {}  # Track breach counts, etc.
        self.notification_rate_limits: Dict[str, deque] = {}  # Track notification rates
        
        # Background processing
        self._evaluation_task: Optional[asyncio.Task] = None
        self._escalation_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.alert_stats = {
            "total_alerts": 0,
            "alerts_by_severity": {severity.value: 0 for severity in AlertSeverity},
            "alerts_by_type": {alert_type.value: 0 for alert_type in AlertType},
            "notifications_sent": 0,
            "escalations_performed": 0,
            "avg_resolution_time": 0.0,
            "last_evaluation": None
        }
        
        # Initialize default thresholds and configurations
        self._initialize_default_thresholds()
        self._initialize_default_notifications()
        
        # Start background processes
        asyncio.create_task(self._start_background_processes())
        
        logger.info("Alert system initialized", storage_path=str(self.storage_path))
    
    def _initialize_default_thresholds(self) -> None:
        """Initialize default alert thresholds."""
        default_thresholds = [
            # Budget thresholds
            AlertThreshold(
                threshold_id="daily_budget_80",
                name="Daily Budget Warning",
                description="Daily budget usage exceeds 80%",
                alert_type=AlertType.BUDGET_THRESHOLD,
                severity=AlertSeverity.WARNING,
                metric_name="budget_usage_percentage",
                operator=">=",
                value=80.0,
                time_window=timedelta(minutes=5),
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.LOG_ONLY]
            ),
            AlertThreshold(
                threshold_id="daily_budget_95",
                name="Daily Budget Critical",
                description="Daily budget usage exceeds 95%",
                alert_type=AlertType.BUDGET_THRESHOLD,
                severity=AlertSeverity.CRITICAL,
                metric_name="budget_usage_percentage",
                operator=">=",
                value=95.0,
                time_window=timedelta(minutes=2),
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                escalation_after=timedelta(minutes=10),
                escalation_channels=[NotificationChannel.SMS, NotificationChannel.PAGERDUTY]
            ),
            
            # Error rate thresholds
            AlertThreshold(
                threshold_id="error_rate_high",
                name="High Error Rate",
                description="Error rate exceeds 10%",
                alert_type=AlertType.ERROR_RATE,
                severity=AlertSeverity.ERROR,
                metric_name="error_rate",
                operator=">",
                value=10.0,
                time_window=timedelta(minutes=5),
                consecutive_breaches=3,
                notification_channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL]
            ),
            AlertThreshold(
                threshold_id="error_rate_critical",
                name="Critical Error Rate",
                description="Error rate exceeds 25%",
                alert_type=AlertType.ERROR_RATE,
                severity=AlertSeverity.CRITICAL,
                metric_name="error_rate",
                operator=">",
                value=25.0,
                time_window=timedelta(minutes=2),
                consecutive_breaches=2,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.SMS]
            ),
            
            # Latency thresholds
            AlertThreshold(
                threshold_id="latency_high",
                name="High Latency",
                description="Average latency exceeds 5 seconds",
                alert_type=AlertType.LATENCY_SPIKE,
                severity=AlertSeverity.WARNING,
                metric_name="avg_latency",
                operator=">",
                value=5000.0,  # milliseconds
                time_window=timedelta(minutes=10),
                consecutive_breaches=3,
                notification_channels=[NotificationChannel.LOG_ONLY]
            ),
            
            # Cost anomaly detection
            AlertThreshold(
                threshold_id="cost_spike",
                name="Cost Spike Detected",
                description="Cost increase exceeds 200% of normal",
                alert_type=AlertType.COST_ANOMALY,
                severity=AlertSeverity.ERROR,
                metric_name="cost_anomaly_percentage",
                operator=">",
                value=200.0,
                time_window=timedelta(minutes=15),
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
            ),
            
            # Usage spike detection
            AlertThreshold(
                threshold_id="usage_spike",
                name="Usage Spike Detected", 
                description="Request volume exceeds 300% of normal",
                alert_type=AlertType.USAGE_SPIKE,
                severity=AlertSeverity.WARNING,
                metric_name="request_volume_percentage",
                operator=">",
                value=300.0,
                time_window=timedelta(minutes=10),
                notification_channels=[NotificationChannel.EMAIL]
            ),
        ]
        
        for threshold in default_thresholds:
            self.thresholds[threshold.threshold_id] = threshold
    
    def _initialize_default_notifications(self) -> None:
        """Initialize default notification configurations."""
        # Email configuration (would be loaded from environment/config)
        self.notification_configs[NotificationChannel.EMAIL] = NotificationConfig(
            channel=NotificationChannel.EMAIL,
            email_host="smtp.gmail.com",
            email_port=587,
            email_from="alerts@mdmai-assistant.com",
            email_to=["admin@mdmai-assistant.com"],
            email_use_tls=True
        )
        
        # Slack configuration
        self.notification_configs[NotificationChannel.SLACK] = NotificationConfig(
            channel=NotificationChannel.SLACK,
            slack_channel="#alerts"
        )
        
        # Webhook configuration
        self.notification_configs[NotificationChannel.WEBHOOK] = NotificationConfig(
            channel=NotificationChannel.WEBHOOK,
            webhook_method="POST",
            webhook_headers={"Content-Type": "application/json"}
        )
        
        # Log-only configuration
        self.notification_configs[NotificationChannel.LOG_ONLY] = NotificationConfig(
            channel=NotificationChannel.LOG_ONLY,
            enabled=True
        )
    
    async def _start_background_processes(self) -> None:
        """Start background alert evaluation and processing tasks."""
        try:
            # Start threshold evaluation
            self._evaluation_task = asyncio.create_task(self._run_threshold_evaluation())
            
            # Start escalation monitoring
            self._escalation_task = asyncio.create_task(self._run_escalation_monitoring())
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._run_alert_cleanup())
            
            logger.info("Alert background processes started")
            
        except Exception as e:
            logger.error("Failed to start alert background processes", error=str(e))
    
    async def _run_threshold_evaluation(self) -> None:
        """Continuously evaluate alert thresholds."""
        while True:
            try:
                await self._evaluate_all_thresholds()
                
                # Wait before next evaluation cycle
                await asyncio.sleep(30)  # Every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in threshold evaluation", error=str(e))
                await asyncio.sleep(60)
    
    async def _evaluate_all_thresholds(self) -> None:
        """Evaluate all enabled alert thresholds."""
        current_time = datetime.now(timezone.utc)
        
        for threshold in self.thresholds.values():
            if not threshold.enabled:
                continue
            
            try:
                await self._evaluate_threshold(threshold, current_time)
            except Exception as e:
                logger.error("Failed to evaluate threshold", 
                           threshold_id=threshold.threshold_id, error=str(e))
        
        self.alert_stats["last_evaluation"] = current_time.isoformat()
    
    async def _evaluate_threshold(self, threshold: AlertThreshold, current_time: datetime) -> None:
        """Evaluate a specific threshold."""
        # Get current metric value
        metric_value = await self._get_current_metric_value(threshold.metric_name, threshold.time_window)
        
        if metric_value is None:
            return  # No data available
        
        # Check if threshold is breached
        breached = self._check_threshold_breach(metric_value, threshold.operator, threshold.value)
        
        # Get or initialize threshold state
        state_key = threshold.threshold_id
        if state_key not in self.threshold_states:
            self.threshold_states[state_key] = {
                "consecutive_breaches": 0,
                "last_breach_time": None,
                "last_alert_time": None,
                "in_cooldown": False
            }
        
        state = self.threshold_states[state_key]
        
        if breached:
            state["consecutive_breaches"] += 1
            state["last_breach_time"] = current_time
            
            # Check if we should trigger an alert
            should_alert = (
                state["consecutive_breaches"] >= threshold.consecutive_breaches and
                not state["in_cooldown"] and
                (state["last_alert_time"] is None or 
                 current_time - state["last_alert_time"] >= threshold.cooldown_period)
            )
            
            if should_alert:
                await self._trigger_alert(threshold, metric_value, current_time)
                state["last_alert_time"] = current_time
                state["in_cooldown"] = True
                state["consecutive_breaches"] = 0  # Reset after alerting
        else:
            # Reset breach count if not breached
            state["consecutive_breaches"] = 0
            
            # Check if cooldown should end
            if (state["in_cooldown"] and state["last_alert_time"] and
                current_time - state["last_alert_time"] >= threshold.cooldown_period):
                state["in_cooldown"] = False
    
    async def _get_current_metric_value(self, metric_name: str, time_window: timedelta) -> Optional[float]:
        """Get current value for a metric."""
        current_time = datetime.now(timezone.utc)
        start_time = current_time - time_window
        
        # Special handling for budget metrics
        if metric_name == "budget_usage_percentage":
            return await self._calculate_budget_usage_percentage()
        elif metric_name == "cost_anomaly_percentage":
            return await self._calculate_cost_anomaly_percentage()
        elif metric_name == "request_volume_percentage":
            return await self._calculate_request_volume_percentage()
        
        # Get metric from metrics collector
        try:
            # This would need to be implemented to query the metrics collector
            # For now, simulate with usage tracker data
            if metric_name == "error_rate":
                return await self._calculate_current_error_rate()
            elif metric_name == "avg_latency":
                return await self._calculate_current_avg_latency()
            else:
                # Default to checking metric buffers
                buffer = self.metrics_collector.metric_buffers.get(metric_name)
                if buffer:
                    # Get recent values within time window
                    recent_values = [
                        snapshot.metric_value for snapshot in buffer
                        if current_time - snapshot.timestamp <= time_window
                    ]
                    if recent_values:
                        return sum(recent_values) / len(recent_values)  # Average
                
        except Exception as e:
            logger.error("Failed to get metric value", metric=metric_name, error=str(e))
        
        return None
    
    async def _calculate_budget_usage_percentage(self) -> float:
        """Calculate current budget usage percentage."""
        # Get total daily usage across all users
        total_usage = 0.0
        total_limit = 0.0
        
        for user_id, limits in self.usage_tracker.user_limits.items():
            if limits.enabled and limits.daily_limit:
                usage = self.usage_tracker.get_user_daily_usage(user_id)
                total_usage += usage
                total_limit += limits.daily_limit
        
        if total_limit > 0:
            return (total_usage / total_limit) * 100
        
        return 0.0
    
    async def _calculate_cost_anomaly_percentage(self) -> float:
        """Calculate cost anomaly percentage (simplified)."""
        # This would implement anomaly detection
        # For now, return a placeholder
        return 0.0
    
    async def _calculate_request_volume_percentage(self) -> float:
        """Calculate request volume anomaly percentage."""
        # This would compare current volume to historical baseline
        # For now, return a placeholder
        return 0.0
    
    async def _calculate_current_error_rate(self) -> float:
        """Calculate current error rate."""
        # Get recent usage data and calculate error rate
        total_requests = 0
        failed_requests = 0
        
        current_time = datetime.now()
        today = current_time.date().isoformat()
        
        for user_daily in self.usage_tracker.daily_usage.values():
            if today in user_daily:
                agg = user_daily[today]
                total_requests += agg.total_requests
                failed_requests += agg.failed_requests
        
        if total_requests > 0:
            return (failed_requests / total_requests) * 100
        
        return 0.0
    
    async def _calculate_current_avg_latency(self) -> float:
        """Calculate current average latency."""
        # Get recent usage data and calculate average latency
        total_latency = 0.0
        request_count = 0
        
        current_time = datetime.now()
        today = current_time.date().isoformat()
        
        for user_daily in self.usage_tracker.daily_usage.values():
            if today in user_daily:
                agg = user_daily[today]
                total_latency += agg.avg_latency_ms * agg.total_requests
                request_count += agg.total_requests
        
        if request_count > 0:
            return total_latency / request_count
        
        return 0.0
    
    def _check_threshold_breach(self, value: float, operator: str, threshold: float) -> bool:
        """Check if a value breaches the threshold."""
        if operator == ">":
            return value > threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<":
            return value < threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return abs(value - threshold) < 0.001  # Float comparison
        elif operator == "!=":
            return abs(value - threshold) >= 0.001
        else:
            logger.warning("Unknown operator", operator=operator)
            return False
    
    async def _trigger_alert(self, threshold: AlertThreshold, current_value: float, timestamp: datetime) -> None:
        """Trigger a new alert."""
        try:
            # Generate alert ID
            alert_id = f"{threshold.threshold_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Create alert
            alert = Alert(
                alert_id=alert_id,
                threshold_id=threshold.threshold_id,
                alert_type=threshold.alert_type,
                severity=threshold.severity,
                status=AlertStatus.OPEN,
                title=threshold.name,
                description=f"{threshold.description} (Current: {current_value:.2f}, Threshold: {threshold.value:.2f})",
                metric_name=threshold.metric_name,
                current_value=current_value,
                threshold_value=threshold.value,
                first_occurrence=timestamp,
                last_occurrence=timestamp,
                tags=threshold.tags.copy()
            )
            
            # Add to active alerts
            self.active_alerts[alert_id] = alert
            
            # Send notifications
            await self._send_alert_notifications(alert, threshold.notification_channels)
            
            # Update statistics
            self.alert_stats["total_alerts"] += 1
            self.alert_stats["alerts_by_severity"][alert.severity.value] += 1
            self.alert_stats["alerts_by_type"][alert.alert_type.value] += 1
            
            # Save alert to storage
            await self._save_alert(alert)
            
            logger.warning("Alert triggered",
                         alert_id=alert_id,
                         threshold_id=threshold.threshold_id,
                         severity=threshold.severity.value,
                         current_value=current_value)
            
        except Exception as e:
            logger.error("Failed to trigger alert", threshold_id=threshold.threshold_id, error=str(e))
    
    async def _send_alert_notifications(self, alert: Alert, channels: List[NotificationChannel]) -> None:
        """Send alert notifications to specified channels."""
        for channel in channels:
            try:
                if channel in self.notification_configs:
                    config = self.notification_configs[channel]
                    if config.enabled and self._check_rate_limit(channel, config):
                        await self._send_notification(alert, channel, config)
                        alert.notifications_sent.append(channel.value)
                        self.alert_stats["notifications_sent"] += 1
            except Exception as e:
                logger.error("Failed to send notification", 
                           alert_id=alert.alert_id, 
                           channel=channel.value, 
                           error=str(e))
    
    def _check_rate_limit(self, channel: NotificationChannel, config: NotificationConfig) -> bool:
        """Check if notification channel is within rate limits."""
        current_time = datetime.now(timezone.utc)
        channel_key = channel.value
        
        if channel_key not in self.notification_rate_limits:
            self.notification_rate_limits[channel_key] = deque()
        
        rate_queue = self.notification_rate_limits[channel_key]
        
        # Remove old entries (older than 1 hour)
        cutoff_time = current_time - timedelta(hours=1)
        while rate_queue and rate_queue[0] < cutoff_time:
            rate_queue.popleft()
        
        # Check hourly limit
        if len(rate_queue) >= config.rate_limit_per_hour:
            logger.warning("Notification rate limit exceeded", 
                         channel=channel.value, 
                         limit=config.rate_limit_per_hour)
            return False
        
        # Add current notification time
        rate_queue.append(current_time)
        
        return True
    
    async def _send_notification(self, alert: Alert, channel: NotificationChannel, config: NotificationConfig) -> None:
        """Send notification through specific channel."""
        if channel == NotificationChannel.EMAIL:
            await self._send_email_notification(alert, config)
        elif channel == NotificationChannel.SLACK:
            await self._send_slack_notification(alert, config)
        elif channel == NotificationChannel.WEBHOOK:
            await self._send_webhook_notification(alert, config)
        elif channel == NotificationChannel.SMS:
            await self._send_sms_notification(alert, config)
        elif channel == NotificationChannel.LOG_ONLY:
            self._send_log_notification(alert)
        else:
            logger.warning("Unsupported notification channel", channel=channel.value)
    
    async def _send_email_notification(self, alert: Alert, config: NotificationConfig) -> None:
        """Send email notification using async SMTP."""
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = config.email_from
            msg['To'] = ', '.join(config.email_to)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Email body
            body = f"""
Alert Details:
- Alert ID: {alert.alert_id}
- Severity: {alert.severity.value.upper()}
- Type: {alert.alert_type.value}
- Description: {alert.description}
- Current Value: {alert.current_value:.2f}
- Threshold: {alert.threshold_value:.2f}
- First Occurrence: {alert.first_occurrence.isoformat()}
- Last Occurrence: {alert.last_occurrence.isoformat()}

Additional Context:
{json.dumps(alert.context, indent=2) if alert.context else 'None'}

This alert was generated by the MDMAI TTRPG Assistant monitoring system.
"""
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email using async SMTP
            if config.email_host and config.email_username and config.email_password:
                # Use aiosmtplib for async email sending
                await aiosmtplib.send(
                    msg,
                    hostname=config.email_host,
                    port=config.email_port,
                    username=config.email_username,
                    password=config.email_password,
                    use_tls=config.email_use_tls,
                )
                
                logger.debug("Email notification sent", alert_id=alert.alert_id)
            else:
                logger.warning("Email configuration incomplete", alert_id=alert.alert_id)
                
        except Exception as e:
            logger.error("Failed to send email notification", alert_id=alert.alert_id, error=str(e))
    
    async def _send_slack_notification(self, alert: Alert, config: NotificationConfig) -> None:
        """Send Slack notification."""
        try:
            if not config.slack_webhook_url:
                logger.warning("Slack webhook URL not configured")
                return
            
            # Create Slack message
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning", 
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "danger"
            }
            
            payload = {
                "channel": config.slack_channel or "#alerts",
                "username": "MDMAI Alert System",
                "attachments": [{
                    "color": color_map.get(alert.severity, "danger"),
                    "title": f"{alert.severity.value.upper()}: {alert.title}",
                    "text": alert.description,
                    "fields": [
                        {"title": "Alert ID", "value": alert.alert_id, "short": True},
                        {"title": "Type", "value": alert.alert_type.value, "short": True},
                        {"title": "Current Value", "value": f"{alert.current_value:.2f}", "short": True},
                        {"title": "Threshold", "value": f"{alert.threshold_value:.2f}", "short": True},
                        {"title": "First Seen", "value": alert.first_occurrence.strftime("%Y-%m-%d %H:%M:%S UTC"), "short": True},
                        {"title": "Last Seen", "value": alert.last_occurrence.strftime("%Y-%m-%d %H:%M:%S UTC"), "short": True},
                    ],
                    "footer": "MDMAI Alert System",
                    "ts": int(alert.first_occurrence.timestamp())
                }]
            }
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(config.slack_webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.debug("Slack notification sent", alert_id=alert.alert_id)
                    else:
                        logger.error("Slack notification failed", 
                                   alert_id=alert.alert_id, 
                                   status=response.status)
                        
        except Exception as e:
            logger.error("Failed to send Slack notification", alert_id=alert.alert_id, error=str(e))
    
    async def _send_webhook_notification(self, alert: Alert, config: NotificationConfig) -> None:
        """Send webhook notification."""
        try:
            if not config.webhook_url:
                logger.warning("Webhook URL not configured")
                return
            
            # Create payload
            payload = {
                "alert": alert.to_dict(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system": "mdmai-ttrpg-assistant"
            }
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                kwargs = {
                    "json" if config.webhook_method.upper() == "POST" else "params": payload,
                    "headers": config.webhook_headers
                }
                
                if config.webhook_auth:
                    kwargs["auth"] = aiohttp.BasicAuth(
                        config.webhook_auth.get("username", ""),
                        config.webhook_auth.get("password", "")
                    )
                
                async with session.request(config.webhook_method, config.webhook_url, **kwargs) as response:
                    if response.status < 400:
                        logger.debug("Webhook notification sent", alert_id=alert.alert_id)
                    else:
                        logger.error("Webhook notification failed",
                                   alert_id=alert.alert_id,
                                   status=response.status)
                        
        except Exception as e:
            logger.error("Failed to send webhook notification", alert_id=alert.alert_id, error=str(e))
    
    async def _send_sms_notification(self, alert: Alert, config: NotificationConfig) -> None:
        """Send SMS notification (placeholder - would integrate with SMS provider)."""
        # This would integrate with Twilio, AWS SNS, or other SMS providers
        logger.info("SMS notification would be sent", alert_id=alert.alert_id)
    
    def _send_log_notification(self, alert: Alert) -> None:
        """Send log-only notification."""
        logger.warning("ALERT",
                      alert_id=alert.alert_id,
                      severity=alert.severity.value,
                      title=alert.title,
                      description=alert.description,
                      current_value=alert.current_value,
                      threshold_value=alert.threshold_value)
    
    async def _run_escalation_monitoring(self) -> None:
        """Monitor alerts for escalation."""
        while True:
            try:
                await self._check_escalations()
                await asyncio.sleep(300)  # Every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in escalation monitoring", error=str(e))
                await asyncio.sleep(600)
    
    async def _check_escalations(self) -> None:
        """Check for alerts that need escalation."""
        current_time = datetime.now(timezone.utc)
        
        for alert in self.active_alerts.values():
            if alert.status not in [AlertStatus.OPEN, AlertStatus.ACKNOWLEDGED]:
                continue
            
            threshold = self.thresholds.get(alert.threshold_id)
            if not threshold or not threshold.escalation_after:
                continue
            
            # Check if escalation time has passed
            time_since_first = current_time - alert.first_occurrence
            if time_since_first >= threshold.escalation_after and alert.escalation_level == 0:
                await self._escalate_alert(alert, threshold)
    
    async def _escalate_alert(self, alert: Alert, threshold: AlertThreshold) -> None:
        """Escalate an alert to higher notification channels."""
        try:
            alert.escalation_level += 1
            alert.escalated_at = datetime.now(timezone.utc)
            
            # Send escalation notifications
            if threshold.escalation_channels:
                await self._send_alert_notifications(alert, threshold.escalation_channels)
            
            self.alert_stats["escalations_performed"] += 1
            
            logger.warning("Alert escalated",
                         alert_id=alert.alert_id,
                         escalation_level=alert.escalation_level)
            
        except Exception as e:
            logger.error("Failed to escalate alert", alert_id=alert.alert_id, error=str(e))
    
    async def _run_alert_cleanup(self) -> None:
        """Clean up old resolved alerts."""
        while True:
            try:
                await self._cleanup_old_alerts()
                await asyncio.sleep(3600)  # Every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in alert cleanup", error=str(e))
                await asyncio.sleep(3600)
    
    async def _cleanup_old_alerts(self) -> None:
        """Clean up resolved alerts older than retention period."""
        current_time = datetime.now(timezone.utc)
        retention_period = timedelta(days=30)  # Keep resolved alerts for 30 days
        cutoff_time = current_time - retention_period
        
        alerts_to_remove = []
        
        for alert_id, alert in self.active_alerts.items():
            if (alert.status in [AlertStatus.RESOLVED, AlertStatus.CLOSED] and
                alert.resolved_at and alert.resolved_at < cutoff_time):
                alerts_to_remove.append(alert_id)
        
        for alert_id in alerts_to_remove:
            alert = self.active_alerts.pop(alert_id)
            self.alert_history.append(alert)
        
        # Limit history size
        if len(self.alert_history) > 10000:
            self.alert_history = self.alert_history[-5000:]  # Keep last 5000
        
        if alerts_to_remove:
            logger.debug("Cleaned up old alerts", count=len(alerts_to_remove))
    
    async def _save_alert(self, alert: Alert) -> None:
        """Save alert to persistent storage using append-only format for efficiency."""
        try:
            # Use JSON Lines format (.jsonl) for efficient appending
            alerts_file = self.storage_path / f"alerts_{datetime.now().strftime('%Y%m')}.jsonl"
            
            # Append the alert as a single JSON line (much more efficient)
            alert_json = json.dumps(alert.to_dict(), separators=(',', ':'))
            
            async with aiofiles.open(alerts_file, 'a') as f:
                await f.write(alert_json + '\n')
                
        except Exception as e:
            logger.error("Failed to save alert", alert_id=alert.alert_id, error=str(e))
    
    # Public API methods
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an active alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        if alert.status == AlertStatus.OPEN:
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now(timezone.utc)
            alert.context["acknowledged_by"] = acknowledged_by
            
            logger.info("Alert acknowledged", alert_id=alert_id, acknowledged_by=acknowledged_by)
            return True
        
        return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = "system", resolution_notes: str = "") -> bool:
        """Resolve an active alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now(timezone.utc)
        alert.context["resolved_by"] = resolved_by
        if resolution_notes:
            alert.context["resolution_notes"] = resolution_notes
        
        # Update resolution time statistics
        if alert.resolved_at and alert.first_occurrence:
            resolution_time = (alert.resolved_at - alert.first_occurrence).total_seconds()
            current_avg = self.alert_stats["avg_resolution_time"]
            total_alerts = self.alert_stats["total_alerts"]
            
            if total_alerts > 0:
                self.alert_stats["avg_resolution_time"] = (
                    (current_avg * (total_alerts - 1) + resolution_time) / total_alerts
                )
        
        logger.info("Alert resolved", alert_id=alert_id, resolved_by=resolved_by)
        return True
    
    def add_threshold(self, threshold: AlertThreshold) -> None:
        """Add a new alert threshold."""
        self.thresholds[threshold.threshold_id] = threshold
        logger.info("Alert threshold added", threshold_id=threshold.threshold_id)
    
    def remove_threshold(self, threshold_id: str) -> bool:
        """Remove an alert threshold."""
        if threshold_id in self.thresholds:
            del self.thresholds[threshold_id]
            if threshold_id in self.threshold_states:
                del self.threshold_states[threshold_id]
            logger.info("Alert threshold removed", threshold_id=threshold_id)
            return True
        return False
    
    def update_notification_config(self, channel: NotificationChannel, config: NotificationConfig) -> None:
        """Update notification configuration for a channel."""
        self.notification_configs[channel] = config
        logger.info("Notification config updated", channel=channel.value)
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None, alert_type: Optional[AlertType] = None) -> List[Alert]:
        """Get list of active alerts with optional filtering."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        return sorted(alerts, key=lambda a: a.first_occurrence, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert system statistics."""
        active_count = len(self.active_alerts)
        active_by_severity = defaultdict(int)
        active_by_status = defaultdict(int)
        
        for alert in self.active_alerts.values():
            active_by_severity[alert.severity.value] += 1
            active_by_status[alert.status.value] += 1
        
        return {
            **self.alert_stats,
            "active_alerts": {
                "total": active_count,
                "by_severity": dict(active_by_severity),
                "by_status": dict(active_by_status)
            },
            "thresholds": {
                "total": len(self.thresholds),
                "enabled": len([t for t in self.thresholds.values() if t.enabled])
            },
            "notification_channels": {
                "configured": len(self.notification_configs),
                "enabled": len([c for c in self.notification_configs.values() if c.enabled])
            }
        }
    
    async def test_notification_channel(self, channel: NotificationChannel) -> bool:
        """Test a notification channel by sending a test alert."""
        try:
            # Create test alert
            test_alert = Alert(
                alert_id="test_alert_" + datetime.now().strftime('%Y%m%d_%H%M%S'),
                threshold_id="test_threshold",
                alert_type=AlertType.SYSTEM_HEALTH,
                severity=AlertSeverity.INFO,
                status=AlertStatus.OPEN,
                title="Test Alert",
                description="This is a test alert to verify notification channel functionality",
                metric_name="test_metric",
                current_value=1.0,
                threshold_value=0.0
            )
            
            if channel in self.notification_configs:
                config = self.notification_configs[channel]
                await self._send_notification(test_alert, channel, config)
                return True
            
            return False
            
        except Exception as e:
            logger.error("Test notification failed", channel=channel.value, error=str(e))
            return False
    
    async def cleanup(self) -> None:
        """Clean up resources and stop background tasks."""
        # Cancel background tasks
        for task in [self._evaluation_task, self._escalation_task, self._cleanup_task]:
            if task and not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        tasks = [t for t in [self._evaluation_task, self._escalation_task, self._cleanup_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Alert system cleanup completed")