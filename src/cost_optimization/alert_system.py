"""
Sophisticated Alerting System with Trend Analysis and Predictive Notifications.

This module provides intelligent alerting for cost management:
- Multi-channel notification delivery (email, webhook, in-app, SMS)
- Severity-based alert escalation
- Trend analysis and anomaly detection
- Predictive alerts based on spending patterns
- Custom alert rules and conditions
- Alert rate limiting and deduplication
- Smart notification scheduling
"""

import asyncio
import hashlib
import json
import smtplib
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from statistics import mean, stdev
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import aiohttp
from structlog import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    """Alert delivery channels."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    IN_APP = "in_app"
    SMS = "sms"
    SLACK = "slack"
    DISCORD = "discord"


class AlertType(Enum):
    """Types of alerts."""
    BUDGET_THRESHOLD = "budget_threshold"
    BUDGET_EXCEEDED = "budget_exceeded"
    VELOCITY_WARNING = "velocity_warning"
    COST_ANOMALY = "cost_anomaly"
    PROVIDER_ERROR = "provider_error"
    QUOTA_WARNING = "quota_warning"
    EMERGENCY_BRAKE = "emergency_brake"
    CIRCUIT_BREAKER = "circuit_breaker"
    FORECAST_WARNING = "forecast_warning"
    PATTERN_CHANGE = "pattern_change"


class TrendDirection(Enum):
    """Trend directions for analysis."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    user_id: str
    timestamp: datetime
    metadata: Dict[str, Any]
    channels: List[AlertChannel]
    acknowledged: bool = False
    resolved: bool = False
    escalated: bool = False


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    alert_type: AlertType
    condition: Dict[str, Any]  # Condition parameters
    severity: AlertSeverity
    channels: List[AlertChannel]
    enabled: bool = True
    rate_limit_minutes: int = 60  # Minimum time between similar alerts
    escalation_delay_minutes: int = 30
    suppress_similar: bool = True


class TrendAnalyzer:
    """Analyze spending trends and patterns."""
    
    def __init__(self, window_hours: int = 24):
        self.window_hours = window_hours
        self.data_points = deque(maxlen=1000)
        self.trend_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def add_data_point(self, timestamp: datetime, value: float, metadata: Dict[str, Any] = None) -> None:
        """Add data point for trend analysis."""
        self.data_points.append({
            'timestamp': timestamp,
            'value': value,
            'metadata': metadata or {}
        })
        
        # Clear cache when new data arrives
        self.trend_cache.clear()
    
    def get_trend_direction(self, hours_back: int = 24) -> TrendDirection:
        """Analyze trend direction over specified time period."""
        cache_key = f"trend_direction_{hours_back}"
        
        if cache_key in self.trend_cache:
            cached_result, timestamp = self.trend_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_result
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        recent_points = [
            point for point in self.data_points
            if point['timestamp'] >= cutoff_time
        ]
        
        if len(recent_points) < 3:
            return TrendDirection.STABLE
        
        # Calculate trend using linear regression
        x_values = [(point['timestamp'] - cutoff_time).total_seconds() for point in recent_points]
        y_values = [point['value'] for point in recent_points]
        
        # Simple linear regression
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = (n * sum_x2 - sum_x * sum_x)
        if denominator == 0:
            return TrendDirection.STABLE
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Calculate coefficient of variation for volatility
        if len(y_values) > 1:
            std_dev = stdev(y_values)
            mean_val = mean(y_values)
            coefficient_of_variation = std_dev / mean_val if mean_val != 0 else 0
        else:
            coefficient_of_variation = 0
        
        # Determine trend
        if coefficient_of_variation > 0.5:  # High volatility
            direction = TrendDirection.VOLATILE
        elif slope > 0.1:
            direction = TrendDirection.INCREASING
        elif slope < -0.1:
            direction = TrendDirection.DECREASING
        else:
            direction = TrendDirection.STABLE
        
        # Cache result
        self.trend_cache[cache_key] = (direction, time.time())
        return direction
    
    def detect_anomalies(self, sensitivity: float = 2.0) -> List[Dict[str, Any]]:
        """Detect cost anomalies using statistical analysis."""
        if len(self.data_points) < 10:
            return []
        
        values = [point['value'] for point in self.data_points]
        mean_val = mean(values)
        std_val = stdev(values) if len(values) > 1 else 0
        
        anomalies = []
        threshold = mean_val + (sensitivity * std_val)
        
        for point in list(self.data_points)[-20:]:  # Check last 20 points
            if point['value'] > threshold:
                anomalies.append({
                    'timestamp': point['timestamp'],
                    'value': point['value'],
                    'expected_max': threshold,
                    'deviation': point['value'] - threshold,
                    'severity': 'high' if point['value'] > threshold * 1.5 else 'medium',
                    'metadata': point['metadata']
                })
        
        return anomalies
    
    def calculate_forecast_accuracy(self, actual_values: List[float], predicted_values: List[float]) -> float:
        """Calculate forecast accuracy using MAPE (Mean Absolute Percentage Error)."""
        if len(actual_values) != len(predicted_values) or len(actual_values) == 0:
            return 0.0
        
        errors = []
        for actual, predicted in zip(actual_values, predicted_values):
            if actual != 0:
                error = abs((actual - predicted) / actual)
                errors.append(error)
        
        return (1.0 - mean(errors)) * 100 if errors else 0.0


class NotificationChannel:
    """Base class for notification channels."""
    
    async def send(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send alert notification. Returns True if successful."""
        raise NotImplementedError


class EmailChannel(NotificationChannel):
    """Email notification channel."""
    
    async def send(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send email notification."""
        try:
            smtp_server = config.get('smtp_server', 'localhost')
            smtp_port = config.get('smtp_port', 587)
            username = config.get('username')
            password = config.get('password')
            from_email = config.get('from_email')
            to_emails = config.get('to_emails', [])
            
            if not to_emails or not from_email:
                logger.warning("Email configuration incomplete")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Email body
            body = f"""
Alert: {alert.title}
Severity: {alert.severity.value.upper()}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
User: {alert.user_id}

Message:
{alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}

Alert ID: {alert.alert_id}
            """.strip()
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if username and password:
                    server.starttls()
                    server.login(username, password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent for {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class WebhookChannel(NotificationChannel):
    """Webhook notification channel."""
    
    async def send(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send webhook notification."""
        try:
            url = config.get('url')
            headers = config.get('headers', {})
            timeout = config.get('timeout', 30)
            
            if not url:
                logger.warning("Webhook URL not configured")
                return False
            
            payload = {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type.value,
                'severity': alert.severity.value,
                'title': alert.title,
                'message': alert.message,
                'user_id': alert.user_id,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status < 400:
                        logger.info(f"Webhook alert sent for {alert.alert_id}")
                        return True
                    else:
                        logger.warning(f"Webhook returned status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class InAppChannel(NotificationChannel):
    """In-app notification channel."""
    
    def __init__(self):
        self.notifications = {}  # user_id -> list of notifications
        self.max_per_user = 100
    
    async def send(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Store in-app notification."""
        try:
            user_id = alert.user_id
            
            if user_id not in self.notifications:
                self.notifications[user_id] = deque(maxlen=self.max_per_user)
            
            notification = {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type.value,
                'severity': alert.severity.value,
                'title': alert.title,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'acknowledged': False,
                'metadata': alert.metadata
            }
            
            self.notifications[user_id].appendleft(notification)
            logger.info(f"In-app alert stored for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store in-app alert: {e}")
            return False
    
    def get_notifications(self, user_id: str, unread_only: bool = False) -> List[Dict[str, Any]]:
        """Get notifications for a user."""
        if user_id not in self.notifications:
            return []
        
        notifications = list(self.notifications[user_id])
        
        if unread_only:
            notifications = [n for n in notifications if not n['acknowledged']]
        
        return notifications
    
    def acknowledge_notification(self, user_id: str, alert_id: str) -> bool:
        """Mark notification as acknowledged."""
        if user_id not in self.notifications:
            return False
        
        for notification in self.notifications[user_id]:
            if notification['alert_id'] == alert_id:
                notification['acknowledged'] = True
                return True
        
        return False


class AlertSystem:
    """Sophisticated alerting system with trend analysis."""
    
    def __init__(self):
        self.alert_rules = {}  # rule_id -> AlertRule
        self.active_alerts = {}  # alert_id -> Alert
        self.alert_history = deque(maxlen=10000)
        self.trend_analyzers = {}  # metric_name -> TrendAnalyzer
        
        # Notification channels
        self.channels = {
            AlertChannel.EMAIL: EmailChannel(),
            AlertChannel.WEBHOOK: WebhookChannel(),
            AlertChannel.IN_APP: InAppChannel(),
        }
        
        # Channel configurations
        self.channel_configs = {}  # channel -> config dict
        
        # Rate limiting
        self.alert_counts = defaultdict(lambda: deque(maxlen=100))  # (rule_id, user_id) -> timestamps
        self.suppressed_alerts = set()  # Set of suppressed alert signatures
        
        # Escalation tracking
        self.escalation_tasks = {}  # alert_id -> asyncio.Task
        
        logger.info("Alert System initialized")
    
    def configure_channel(self, channel: AlertChannel, config: Dict[str, Any]) -> None:
        """Configure notification channel."""
        self.channel_configs[channel] = config
        logger.info(f"Configured {channel.value} notification channel")
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False
    
    def get_trend_analyzer(self, metric_name: str) -> TrendAnalyzer:
        """Get or create trend analyzer for a metric."""
        if metric_name not in self.trend_analyzers:
            self.trend_analyzers[metric_name] = TrendAnalyzer()
        return self.trend_analyzers[metric_name]
    
    def create_alert_signature(self, rule_id: str, user_id: str, metadata: Dict[str, Any]) -> str:
        """Create unique signature for alert deduplication."""
        signature_data = f"{rule_id}:{user_id}:{json.dumps(metadata, sort_keys=True)}"
        return hashlib.md5(signature_data.encode()).hexdigest()
    
    def is_rate_limited(self, rule_id: str, user_id: str, rate_limit_minutes: int) -> bool:
        """Check if alert is rate limited."""
        key = (rule_id, user_id)
        now = time.time()
        cutoff = now - (rate_limit_minutes * 60)
        
        # Clean old entries
        while self.alert_counts[key] and self.alert_counts[key][0] < cutoff:
            self.alert_counts[key].popleft()
        
        # Check if we can send another alert
        return len(self.alert_counts[key]) > 0
    
    def record_alert_sent(self, rule_id: str, user_id: str) -> None:
        """Record that an alert was sent for rate limiting."""
        key = (rule_id, user_id)
        self.alert_counts[key].append(time.time())
    
    async def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        user_id: str,
        metadata: Dict[str, Any] = None,
        rule_id: Optional[str] = None
    ) -> Optional[Alert]:
        """Create and process alert."""
        
        alert_id = f"{alert_type.value}_{user_id}_{int(time.time())}"
        metadata = metadata or {}
        
        # Find applicable rule or create default
        applicable_rule = None
        if rule_id:
            applicable_rule = self.alert_rules.get(rule_id)
        else:
            # Find first matching rule
            for rule in self.alert_rules.values():
                if rule.alert_type == alert_type and rule.enabled:
                    applicable_rule = rule
                    break
        
        if not applicable_rule:
            # Create default rule
            channels = [AlertChannel.IN_APP]
            if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                channels.append(AlertChannel.EMAIL)
            
            applicable_rule = AlertRule(
                rule_id=f"default_{alert_type.value}",
                name=f"Default {alert_type.value} rule",
                alert_type=alert_type,
                condition={},
                severity=severity,
                channels=channels
            )
        
        # Check rate limiting
        if self.is_rate_limited(applicable_rule.rule_id, user_id, applicable_rule.rate_limit_minutes):
            logger.debug(f"Alert rate limited: {alert_type.value} for user {user_id}")
            return None
        
        # Check suppression
        signature = self.create_alert_signature(applicable_rule.rule_id, user_id, metadata)
        if applicable_rule.suppress_similar and signature in self.suppressed_alerts:
            logger.debug(f"Alert suppressed: {alert_type.value} for user {user_id}")
            return None
        
        # Create alert
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            metadata=metadata,
            channels=applicable_rule.channels
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        await self._send_notifications(alert, applicable_rule)
        
        # Record for rate limiting
        self.record_alert_sent(applicable_rule.rule_id, user_id)
        
        # Add to suppression set
        if applicable_rule.suppress_similar:
            self.suppressed_alerts.add(signature)
            
            # Remove from suppression after some time
            asyncio.create_task(self._remove_suppression_later(signature, 3600))  # 1 hour
        
        # Schedule escalation if needed
        if applicable_rule.escalation_delay_minutes > 0:
            self.escalation_tasks[alert_id] = asyncio.create_task(
                self._schedule_escalation(alert, applicable_rule)
            )
        
        logger.info(f"Created alert {alert_id}: {title}")
        return alert
    
    async def _send_notifications(self, alert: Alert, rule: AlertRule) -> None:
        """Send notifications through configured channels."""
        for channel in alert.channels:
            if channel in self.channels and channel in self.channel_configs:
                try:
                    config = self.channel_configs[channel]
                    success = await self.channels[channel].send(alert, config)
                    if not success:
                        logger.warning(f"Failed to send {channel.value} notification for {alert.alert_id}")
                except Exception as e:
                    logger.error(f"Error sending {channel.value} notification: {e}")
    
    async def _remove_suppression_later(self, signature: str, delay_seconds: int) -> None:
        """Remove alert from suppression set after delay."""
        await asyncio.sleep(delay_seconds)
        self.suppressed_alerts.discard(signature)
    
    async def _schedule_escalation(self, alert: Alert, rule: AlertRule) -> None:
        """Schedule alert escalation."""
        await asyncio.sleep(rule.escalation_delay_minutes * 60)
        
        # Check if alert is still active and unacknowledged
        if alert.alert_id in self.active_alerts and not alert.acknowledged:
            alert.escalated = True
            escalated_alert = Alert(
                alert_id=f"escalated_{alert.alert_id}",
                alert_type=alert.alert_type,
                severity=AlertSeverity.CRITICAL,  # Escalate severity
                title=f"ESCALATED: {alert.title}",
                message=f"This alert has been escalated due to lack of acknowledgment.\n\nOriginal message:\n{alert.message}",
                user_id=alert.user_id,
                timestamp=datetime.utcnow(),
                metadata={**alert.metadata, 'escalated_from': alert.alert_id},
                channels=[AlertChannel.EMAIL, AlertChannel.WEBHOOK]  # Use high-priority channels
            )
            
            await self._send_notifications(escalated_alert, rule)
            logger.warning(f"Escalated alert {alert.alert_id}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = None) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.metadata['acknowledged_by'] = acknowledged_by
            alert.metadata['acknowledged_at'] = datetime.utcnow().isoformat()
            
            # Cancel escalation task
            if alert_id in self.escalation_tasks:
                self.escalation_tasks[alert_id].cancel()
                del self.escalation_tasks[alert_id]
            
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = None) -> bool:
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.metadata['resolved_by'] = resolved_by
            alert.metadata['resolved_at'] = datetime.utcnow().isoformat()
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            # Cancel escalation task
            if alert_id in self.escalation_tasks:
                self.escalation_tasks[alert_id].cancel()
                del self.escalation_tasks[alert_id]
            
            logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            return True
        
        return False
    
    async def check_budget_threshold_alert(
        self,
        user_id: str,
        budget_name: str,
        current_spending: Decimal,
        budget_limit: Decimal,
        threshold_percentage: float
    ) -> None:
        """Check and create budget threshold alert."""
        usage_percentage = float((current_spending / budget_limit) * 100) if budget_limit > 0 else 0
        
        if usage_percentage >= threshold_percentage:
            severity = AlertSeverity.WARNING
            if usage_percentage >= 90:
                severity = AlertSeverity.CRITICAL
            elif usage_percentage >= 95:
                severity = AlertSeverity.EMERGENCY
            
            await self.create_alert(
                alert_type=AlertType.BUDGET_THRESHOLD,
                severity=severity,
                title=f"Budget Threshold Alert: {budget_name}",
                message=f"Budget '{budget_name}' is at {usage_percentage:.1f}% utilization (${current_spending} of ${budget_limit})",
                user_id=user_id,
                metadata={
                    'budget_name': budget_name,
                    'current_spending': float(current_spending),
                    'budget_limit': float(budget_limit),
                    'usage_percentage': usage_percentage,
                    'threshold': threshold_percentage
                }
            )
    
    async def check_velocity_warning_alert(
        self,
        user_id: str,
        current_velocity: float,
        predicted_daily_spend: float,
        daily_budget: float
    ) -> None:
        """Check and create velocity warning alert."""
        if predicted_daily_spend > daily_budget * 0.8:  # 80% of daily budget
            severity = AlertSeverity.WARNING
            if predicted_daily_spend > daily_budget:
                severity = AlertSeverity.CRITICAL
            
            await self.create_alert(
                alert_type=AlertType.VELOCITY_WARNING,
                severity=severity,
                title="High Spending Velocity Detected",
                message=f"Current spending velocity predicts ${predicted_daily_spend:.2f} daily spend, exceeding budget of ${daily_budget:.2f}",
                user_id=user_id,
                metadata={
                    'current_velocity': current_velocity,
                    'predicted_daily_spend': predicted_daily_spend,
                    'daily_budget': daily_budget,
                    'velocity_ratio': predicted_daily_spend / daily_budget if daily_budget > 0 else 0
                }
            )
    
    async def check_cost_anomaly_alert(
        self,
        user_id: str,
        metric_name: str,
        current_value: float,
        expected_range: Tuple[float, float]
    ) -> None:
        """Check and create cost anomaly alert."""
        min_expected, max_expected = expected_range
        
        if current_value > max_expected:
            deviation_percentage = ((current_value - max_expected) / max_expected) * 100
            
            severity = AlertSeverity.WARNING
            if deviation_percentage > 100:  # More than 100% above expected
                severity = AlertSeverity.CRITICAL
            
            await self.create_alert(
                alert_type=AlertType.COST_ANOMALY,
                severity=severity,
                title=f"Cost Anomaly Detected: {metric_name}",
                message=f"{metric_name} is {deviation_percentage:.1f}% above expected range (${current_value:.2f} vs ${max_expected:.2f} expected)",
                user_id=user_id,
                metadata={
                    'metric_name': metric_name,
                    'current_value': current_value,
                    'expected_min': min_expected,
                    'expected_max': max_expected,
                    'deviation_percentage': deviation_percentage
                }
            )
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics."""
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)
        
        # Count alerts by time period
        alerts_24h = [a for a in self.alert_history if a.timestamp >= last_24h]
        alerts_7d = [a for a in self.alert_history if a.timestamp >= last_7d]
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in alerts_24h:
            severity_counts[alert.severity.value] += 1
        
        # Count by type
        type_counts = defaultdict(int)
        for alert in alerts_24h:
            type_counts[alert.alert_type.value] += 1
        
        return {
            'active_alerts': len(self.active_alerts),
            'total_alerts': len(self.alert_history),
            'alerts_last_24h': len(alerts_24h),
            'alerts_last_7d': len(alerts_7d),
            'severity_distribution_24h': dict(severity_counts),
            'type_distribution_24h': dict(type_counts),
            'suppressed_alerts': len(self.suppressed_alerts),
            'escalated_alerts': len(self.escalation_tasks),
            'alert_rules': len(self.alert_rules)
        }
    
    def get_user_notifications(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get notifications for a user."""
        # Get in-app notifications
        in_app_channel = self.channels.get(AlertChannel.IN_APP)
        if isinstance(in_app_channel, InAppChannel):
            return in_app_channel.get_notifications(user_id)[:limit]
        
        return []
    
    def acknowledge_user_notification(self, user_id: str, alert_id: str) -> bool:
        """Acknowledge user notification."""
        # Acknowledge in alert system
        acknowledged = self.acknowledge_alert(alert_id, user_id)
        
        # Acknowledge in in-app channel
        in_app_channel = self.channels.get(AlertChannel.IN_APP)
        if isinstance(in_app_channel, InAppChannel):
            in_app_channel.acknowledge_notification(user_id, alert_id)
        
        return acknowledged