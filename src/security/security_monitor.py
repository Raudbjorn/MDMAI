"""Security monitoring and threat detection system."""

import asyncio
import json
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from config.logging_config import get_logger
from src.security.models import SecurityMetrics, SessionStatus
from src.security.security_audit import SecurityEventType, SecuritySeverity

logger = get_logger(__name__)


class ThreatLevel(Enum):
    """Threat level classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatIndicator(Enum):
    """Types of threat indicators."""

    BRUTE_FORCE = "brute_force"
    RATE_LIMIT_ABUSE = "rate_limit_abuse"
    SQL_INJECTION = "sql_injection"
    XSS_ATTEMPT = "xss_attempt"
    PATH_TRAVERSAL = "path_traversal"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    ACCOUNT_TAKEOVER = "account_takeover"
    SESSION_HIJACK = "session_hijack"
    API_ABUSE = "api_abuse"
    DATA_EXFILTRATION = "data_exfiltration"
    DDOS_ATTEMPT = "ddos_attempt"


class SecurityAlert(BaseModel):
    """Security alert model."""

    alert_id: str
    threat_level: ThreatLevel
    threat_indicator: ThreatIndicator
    timestamp: datetime
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)
    actions_taken: List[str] = Field(default_factory=list)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


class MonitoringRule(BaseModel):
    """Rule for monitoring security events."""

    rule_id: str
    name: str
    description: str
    enabled: bool = True
    
    # Trigger conditions
    event_types: List[SecurityEventType] = Field(default_factory=list)
    severity_threshold: SecuritySeverity = SecuritySeverity.WARNING
    
    # Thresholds
    count_threshold: int = Field(default=5, description="Number of events to trigger")
    time_window_seconds: int = Field(default=300, description="Time window for counting")
    
    # Pattern matching
    patterns: List[str] = Field(default_factory=list)
    
    # Response
    threat_level: ThreatLevel = ThreatLevel.MEDIUM
    threat_indicator: ThreatIndicator = ThreatIndicator.SUSPICIOUS_PATTERN
    auto_block: bool = False
    notify_admins: bool = True
    
    # Rate limiting
    cooldown_seconds: int = Field(default=3600, description="Cooldown between alerts")


class SecurityMonitor:
    """Real-time security monitoring and threat detection."""

    def __init__(self):
        """Initialize security monitor."""
        self._event_buffer: Deque[Tuple[SecurityEventType, datetime, Dict[str, Any]]] = deque(maxlen=10000)
        self._alerts: Dict[str, SecurityAlert] = {}
        self._blocked_ips: Set[str] = set()
        self._blocked_users: Set[str] = set()
        self._suspicious_sessions: Set[str] = set()
        self._rules: Dict[str, MonitoringRule] = self._initialize_rules()
        self._rule_triggers: Dict[str, datetime] = {}
        self._metrics_history: Deque[SecurityMetrics] = deque(maxlen=288)  # 24 hours at 5-min intervals
        self._anomaly_baseline: Optional[Dict[str, float]] = None
        
        # Event counters for rate detection
        self._ip_events: Dict[str, Deque[datetime]] = defaultdict(lambda: deque(maxlen=100))
        self._user_events: Dict[str, Deque[datetime]] = defaultdict(lambda: deque(maxlen=100))
        
        # Start monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None

    def _initialize_rules(self) -> Dict[str, MonitoringRule]:
        """Initialize default monitoring rules."""
        rules = [
            MonitoringRule(
                rule_id="brute_force",
                name="Brute Force Detection",
                description="Detect brute force login attempts",
                event_types=[SecurityEventType.AUTH_LOGIN_FAILURE],
                count_threshold=5,
                time_window_seconds=60,
                threat_level=ThreatLevel.HIGH,
                threat_indicator=ThreatIndicator.BRUTE_FORCE,
                auto_block=True,
            ),
            MonitoringRule(
                rule_id="rate_limit_abuse",
                name="Rate Limit Abuse",
                description="Detect excessive rate limit violations",
                event_types=[SecurityEventType.RATE_LIMIT_EXCEEDED],
                count_threshold=10,
                time_window_seconds=300,
                threat_level=ThreatLevel.MEDIUM,
                threat_indicator=ThreatIndicator.RATE_LIMIT_ABUSE,
            ),
            MonitoringRule(
                rule_id="injection_attempts",
                name="Injection Attack Detection",
                description="Detect SQL injection and XSS attempts",
                event_types=[SecurityEventType.INJECTION_ATTEMPT],
                count_threshold=3,
                time_window_seconds=60,
                threat_level=ThreatLevel.CRITICAL,
                threat_indicator=ThreatIndicator.SQL_INJECTION,
                auto_block=True,
            ),
            MonitoringRule(
                rule_id="path_traversal",
                name="Path Traversal Detection",
                description="Detect path traversal attempts",
                event_types=[SecurityEventType.PATH_TRAVERSAL_ATTEMPT],
                count_threshold=2,
                time_window_seconds=60,
                threat_level=ThreatLevel.HIGH,
                threat_indicator=ThreatIndicator.PATH_TRAVERSAL,
                auto_block=True,
            ),
            MonitoringRule(
                rule_id="privilege_escalation",
                name="Privilege Escalation Detection",
                description="Detect unauthorized privilege changes",
                event_types=[
                    SecurityEventType.PERMISSION_ELEVATED,
                    SecurityEventType.ACCESS_DENIED,
                ],
                count_threshold=5,
                time_window_seconds=300,
                threat_level=ThreatLevel.HIGH,
                threat_indicator=ThreatIndicator.PRIVILEGE_ESCALATION,
            ),
            MonitoringRule(
                rule_id="session_anomaly",
                name="Session Anomaly Detection",
                description="Detect suspicious session behavior",
                event_types=[
                    SecurityEventType.SESSION_CREATED,
                    SecurityEventType.SESSION_EXPIRED,
                ],
                count_threshold=10,
                time_window_seconds=60,
                threat_level=ThreatLevel.MEDIUM,
                threat_indicator=ThreatIndicator.SESSION_HIJACK,
            ),
        ]
        
        return {rule.rule_id: rule for rule in rules}

    async def start_monitoring(self) -> None:
        """Start background monitoring task."""
        if not self._monitoring_task or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Security monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring task."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Security monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                # Process event buffer
                await self._process_events()
                
                # Check for anomalies
                await self._detect_anomalies()
                
                # Update metrics
                await self._update_metrics()
                
                # Cleanup old data
                await self._cleanup()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")

    def record_event(
        self,
        event_type: SecurityEventType,
        ip_address: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record security event for monitoring."""
        now = datetime.utcnow()
        
        # Add to buffer
        self._event_buffer.append((
            event_type,
            now,
            {
                "ip_address": ip_address,
                "user_id": user_id,
                "session_id": session_id,
                "details": details or {},
            }
        ))
        
        # Track by IP and user
        if ip_address:
            self._ip_events[ip_address].append(now)
        if user_id:
            self._user_events[user_id].append(now)

    async def _process_events(self) -> None:
        """Process events and check rules."""
        now = datetime.utcnow()
        
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            
            # Check cooldown
            last_trigger = self._rule_triggers.get(rule.rule_id)
            if last_trigger and (now - last_trigger).total_seconds() < rule.cooldown_seconds:
                continue
            
            # Count matching events in time window
            window_start = now - timedelta(seconds=rule.time_window_seconds)
            matching_events = []
            
            for event_type, timestamp, context in self._event_buffer:
                if timestamp < window_start:
                    continue
                
                if event_type in rule.event_types:
                    matching_events.append((event_type, timestamp, context))
            
            # Check threshold
            if len(matching_events) >= rule.count_threshold:
                await self._trigger_rule(rule, matching_events)

    async def _trigger_rule(
        self,
        rule: MonitoringRule,
        events: List[Tuple[SecurityEventType, datetime, Dict[str, Any]]],
    ) -> None:
        """Trigger monitoring rule and create alert."""
        # Update trigger time
        self._rule_triggers[rule.rule_id] = datetime.utcnow()
        
        # Extract common attributes
        ips = {e[2].get("ip_address") for e in events if e[2].get("ip_address")}
        users = {e[2].get("user_id") for e in events if e[2].get("user_id")}
        sessions = {e[2].get("session_id") for e in events if e[2].get("session_id")}
        
        # Create alert
        alert = SecurityAlert(
            alert_id=f"{rule.rule_id}_{datetime.utcnow().timestamp()}",
            threat_level=rule.threat_level,
            threat_indicator=rule.threat_indicator,
            timestamp=datetime.utcnow(),
            source_ip=list(ips)[0] if len(ips) == 1 else None,
            user_id=list(users)[0] if len(users) == 1 else None,
            session_id=list(sessions)[0] if len(sessions) == 1 else None,
            description=f"{rule.name}: {rule.description}",
            details={
                "rule_id": rule.rule_id,
                "event_count": len(events),
                "affected_ips": list(ips),
                "affected_users": list(users),
                "affected_sessions": list(sessions),
            },
        )
        
        # Take automated actions
        if rule.auto_block:
            for ip in ips:
                self.block_ip(ip)
                alert.actions_taken.append(f"Blocked IP: {ip}")
            
            for user_id in users:
                self.block_user(user_id)
                alert.actions_taken.append(f"Blocked user: {user_id}")
        
        # Mark sessions as suspicious
        for session_id in sessions:
            self._suspicious_sessions.add(session_id)
            alert.actions_taken.append(f"Flagged session: {session_id}")
        
        # Store alert
        self._alerts[alert.alert_id] = alert
        
        # Log alert
        logger.warning(
            f"Security alert: {alert.description}",
            threat_level=alert.threat_level.value,
            threat_indicator=alert.threat_indicator.value,
            details=alert.details,
        )
        
        # Notify admins if configured
        if rule.notify_admins:
            await self._notify_admins(alert)

    async def _detect_anomalies(self) -> None:
        """Detect anomalous behavior patterns."""
        now = datetime.utcnow()
        
        # Check for rapid IP rotation (potential session hijacking)
        for user_id, events in self._user_events.items():
            recent_events = [e for e in events if (now - e).total_seconds() < 300]
            if len(recent_events) > 5:
                # Check for multiple IPs
                ips = set()
                for _, _, context in self._event_buffer:
                    if context.get("user_id") == user_id:
                        ip = context.get("ip_address")
                        if ip:
                            ips.add(ip)
                
                if len(ips) > 3:  # More than 3 different IPs in 5 minutes
                    alert = SecurityAlert(
                        alert_id=f"ip_rotation_{user_id}_{now.timestamp()}",
                        threat_level=ThreatLevel.HIGH,
                        threat_indicator=ThreatIndicator.SESSION_HIJACK,
                        timestamp=now,
                        user_id=user_id,
                        description="Rapid IP rotation detected",
                        details={
                            "ip_count": len(ips),
                            "ips": list(ips),
                        },
                    )
                    self._alerts[alert.alert_id] = alert

    async def _update_metrics(self) -> None:
        """Update security metrics."""
        now = datetime.utcnow()
        period_start = now - timedelta(minutes=5)
        
        # Calculate metrics for the last 5 minutes
        metrics = SecurityMetrics(
            period_start=period_start,
            period_end=now,
            active_sessions=0,  # Would get from session store
            failed_login_attempts=sum(
                1 for e in self._event_buffer
                if e[0] == SecurityEventType.AUTH_LOGIN_FAILURE and e[1] >= period_start
            ),
            successful_logins=sum(
                1 for e in self._event_buffer
                if e[0] == SecurityEventType.AUTH_LOGIN_SUCCESS and e[1] >= period_start
            ),
            rate_limit_hits=sum(
                1 for e in self._event_buffer
                if e[0] == SecurityEventType.RATE_LIMIT_EXCEEDED and e[1] >= period_start
            ),
            blocked_ips=list(self._blocked_ips),
            suspicious_activities=len([
                a for a in self._alerts.values()
                if not a.resolved and a.timestamp >= period_start
            ]),
        )
        
        self._metrics_history.append(metrics)

    async def _cleanup(self) -> None:
        """Clean up old data."""
        now = datetime.utcnow()
        
        # Clean up old events from IP/user trackers
        cutoff = now - timedelta(hours=1)
        
        for ip, events in list(self._ip_events.items()):
            self._ip_events[ip] = deque(
                (e for e in events if e > cutoff),
                maxlen=100,
            )
            if not self._ip_events[ip]:
                del self._ip_events[ip]
        
        for user_id, events in list(self._user_events.items()):
            self._user_events[user_id] = deque(
                (e for e in events if e > cutoff),
                maxlen=100,
            )
            if not self._user_events[user_id]:
                del self._user_events[user_id]

    async def _notify_admins(self, alert: SecurityAlert) -> None:
        """Notify administrators about security alert."""
        # In production, this would send emails, SMS, or push notifications
        logger.info(f"Admin notification would be sent for alert: {alert.alert_id}")

    def block_ip(self, ip_address: str, duration_hours: int = 24) -> None:
        """Block IP address."""
        self._blocked_ips.add(ip_address)
        logger.warning(f"Blocked IP address: {ip_address}")

    def unblock_ip(self, ip_address: str) -> None:
        """Unblock IP address."""
        self._blocked_ips.discard(ip_address)
        logger.info(f"Unblocked IP address: {ip_address}")

    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked."""
        return ip_address in self._blocked_ips

    def block_user(self, user_id: str, duration_hours: int = 24) -> None:
        """Block user."""
        self._blocked_users.add(user_id)
        logger.warning(f"Blocked user: {user_id}")

    def unblock_user(self, user_id: str) -> None:
        """Unblock user."""
        self._blocked_users.discard(user_id)
        logger.info(f"Unblocked user: {user_id}")

    def is_user_blocked(self, user_id: str) -> bool:
        """Check if user is blocked."""
        return user_id in self._blocked_users

    def is_session_suspicious(self, session_id: str) -> bool:
        """Check if session is flagged as suspicious."""
        return session_id in self._suspicious_sessions

    def get_active_alerts(self) -> List[SecurityAlert]:
        """Get active security alerts."""
        return [a for a in self._alerts.values() if not a.resolved]

    def resolve_alert(
        self,
        alert_id: str,
        resolved_by: Optional[str] = None,
    ) -> bool:
        """Resolve security alert."""
        if alert_id in self._alerts:
            alert = self._alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            alert.resolved_by = resolved_by
            return True
        return False

    def get_metrics(self, hours: int = 24) -> List[SecurityMetrics]:
        """Get security metrics for the specified time range."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            m for m in self._metrics_history
            if m.period_end >= cutoff
        ]

    def get_threat_summary(self) -> Dict[str, Any]:
        """Get current threat summary."""
        active_alerts = self.get_active_alerts()
        
        threat_counts = defaultdict(int)
        for alert in active_alerts:
            threat_counts[alert.threat_level.value] += 1
        
        return {
            "overall_threat_level": self._calculate_overall_threat_level(),
            "active_alerts": len(active_alerts),
            "threat_counts": dict(threat_counts),
            "blocked_ips": len(self._blocked_ips),
            "blocked_users": len(self._blocked_users),
            "suspicious_sessions": len(self._suspicious_sessions),
            "recent_attacks": [
                {
                    "timestamp": alert.timestamp.isoformat(),
                    "type": alert.threat_indicator.value,
                    "level": alert.threat_level.value,
                    "description": alert.description,
                }
                for alert in sorted(active_alerts, key=lambda a: a.timestamp, reverse=True)[:5]
            ],
        }

    def _calculate_overall_threat_level(self) -> str:
        """Calculate overall threat level based on active alerts."""
        active_alerts = self.get_active_alerts()
        
        if any(a.threat_level == ThreatLevel.CRITICAL for a in active_alerts):
            return ThreatLevel.CRITICAL.value
        elif any(a.threat_level == ThreatLevel.HIGH for a in active_alerts):
            return ThreatLevel.HIGH.value
        elif any(a.threat_level == ThreatLevel.MEDIUM for a in active_alerts):
            return ThreatLevel.MEDIUM.value
        elif active_alerts:
            return ThreatLevel.LOW.value
        else:
            return "none"

    def export_alerts(self, format: str = "json") -> str:
        """Export alerts in specified format."""
        alerts_data = [
            alert.model_dump(mode="json")
            for alert in self._alerts.values()
        ]
        
        if format == "json":
            return json.dumps(alerts_data, indent=2, default=str)
        else:
            # Could add CSV, XML, etc.
            return json.dumps(alerts_data, default=str)