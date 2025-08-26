"""Security audit trail and logging system."""

import json
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class SecurityEventType(Enum):
    """Types of security events to audit."""

    # Authentication events
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILURE = "auth.login.failure"
    AUTH_LOGOUT = "auth.logout"
    AUTH_SESSION_EXPIRED = "auth.session.expired"

    # Authorization events
    ACCESS_GRANTED = "access.granted"
    ACCESS_DENIED = "access.denied"
    PERMISSION_ELEVATED = "permission.elevated"
    PERMISSION_REVOKED = "permission.revoked"

    # Validation events
    INPUT_VALIDATION_FAILED = "input.validation.failed"
    INJECTION_ATTEMPT = "injection.attempt"
    PATH_TRAVERSAL_ATTEMPT = "path.traversal.attempt"
    XSS_ATTEMPT = "xss.attempt"

    # Rate limiting events
    RATE_LIMIT_EXCEEDED = "rate.limit.exceeded"
    RATE_LIMIT_RESET = "rate.limit.reset"

    # Data access events
    DATA_ACCESS = "data.access"
    DATA_MODIFICATION = "data.modification"
    DATA_DELETION = "data.deletion"

    # Campaign events
    CAMPAIGN_CREATED = "campaign.created"
    CAMPAIGN_ACCESSED = "campaign.accessed"
    CAMPAIGN_MODIFIED = "campaign.modified"
    CAMPAIGN_DELETED = "campaign.deleted"

    # Source events
    SOURCE_ADDED = "source.added"
    SOURCE_ACCESSED = "source.accessed"
    SOURCE_DELETED = "source.deleted"

    # System events
    SYSTEM_CONFIG_CHANGED = "system.config.changed"
    CACHE_CLEARED = "cache.cleared"
    INDEX_UPDATED = "index.updated"

    # Security events
    SECURITY_SCAN = "security.scan"
    SECURITY_VIOLATION = "security.violation"
    SECURITY_CONFIG_CHANGED = "security.config.changed"


class SecuritySeverity(Enum):
    """Severity levels for security events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SecurityEvent(BaseModel):
    """Security event model."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: SecurityEventType
    severity: SecuritySeverity
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None
    result: str  # success, failure, blocked
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    trace_id: Optional[str] = None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        use_enum_values = True


class SecurityReport(BaseModel):
    """Security report model."""

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    period_start: datetime
    period_end: datetime
    total_events: int = 0
    events_by_type: Dict[str, int] = Field(default_factory=dict)
    events_by_severity: Dict[str, int] = Field(default_factory=dict)
    top_users: List[Dict[str, Any]] = Field(default_factory=list)
    suspicious_activities: List[Dict[str, Any]] = Field(default_factory=list)
    security_violations: List[SecurityEvent] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class SecurityAuditTrail:
    """Manages security audit trail and compliance logging."""

    def __init__(
        self,
        enable_audit: bool = True,
        audit_file_path: Optional[Path] = None,
        retention_days: int = 90,
        max_events_in_memory: int = 10000,
    ):
        """
        Initialize security audit trail.

        Args:
            enable_audit: Whether to enable audit logging
            audit_file_path: Path to audit log file
            retention_days: Number of days to retain audit logs
            max_events_in_memory: Maximum events to keep in memory
        """
        self.enabled = enable_audit
        self.retention_days = retention_days
        self.max_events_in_memory = max_events_in_memory

        # Set audit file path
        if audit_file_path:
            self.audit_file = audit_file_path
        else:
            audit_dir = Path(settings.cache_dir) / "security" / "audit"
            audit_dir.mkdir(parents=True, exist_ok=True)
            self.audit_file = audit_dir / f"audit_{datetime.utcnow().strftime('%Y%m')}.jsonl"

        # In-memory storage for recent events
        self.recent_events: List[SecurityEvent] = []

        # Tracking for suspicious activity detection
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.suspicious_ips: Set[str] = set()
        self.violation_count: Dict[str, int] = {}

        # Initialize audit file
        if self.enabled:
            self._initialize_audit_file()

        logger.info(
            "Security audit trail initialized",
            enabled=self.enabled,
            audit_file=str(self.audit_file),
            retention_days=retention_days,
        )

    def _initialize_audit_file(self) -> None:
        """Initialize audit file with header if needed."""
        if not self.audit_file.exists():
            self.audit_file.parent.mkdir(parents=True, exist_ok=True)
            self.audit_file.touch()
            logger.info(f"Created audit file: {self.audit_file}")

    def log_event(
        self,
        event_type: SecurityEventType,
        severity: SecuritySeverity,
        message: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        result: str = "success",
        details: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> SecurityEvent:
        """
        Log a security event.

        Args:
            event_type: Type of security event
            severity: Event severity
            message: Event message
            user_id: User ID if applicable
            session_id: Session ID if applicable
            ip_address: Client IP address
            user_agent: Client user agent
            resource_type: Type of resource accessed
            resource_id: ID of resource accessed
            action: Action performed
            result: Result of the action
            details: Additional event details
            trace_id: Trace ID for correlation

        Returns:
            The logged security event
        """
        if not self.enabled:
            return SecurityEvent(
                event_type=event_type,
                severity=severity,
                message=message,
                result=result,
            )

        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            result=result,
            details=details or {},
            trace_id=trace_id,
        )

        # Add to recent events
        self.recent_events.append(event)
        if len(self.recent_events) > self.max_events_in_memory:
            self.recent_events.pop(0)

        # Write to audit file
        self._write_event_to_file(event)

        # Track suspicious activity
        self._track_suspicious_activity(event)

        # Log to standard logger based on severity
        log_method = getattr(logger, severity.value, logger.info)
        log_method(
            f"Security event: {event_type.value}",
            user_id=user_id,
            ip_address=ip_address,
            result=result,
            details=details,
        )

        return event

    def _write_event_to_file(self, event: SecurityEvent) -> None:
        """Write event to audit file."""
        try:
            with open(self.audit_file, "a") as f:
                event_dict = event.model_dump()
                # Convert datetime to ISO format
                event_dict["timestamp"] = event.timestamp.isoformat()
                f.write(json.dumps(event_dict) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit event to file: {e}")

    def _track_suspicious_activity(self, event: SecurityEvent) -> None:
        """Track suspicious activity patterns."""
        # Track failed authentication attempts
        if event.event_type in [
            SecurityEventType.AUTH_LOGIN_FAILURE,
            SecurityEventType.ACCESS_DENIED,
        ]:
            key = event.ip_address or event.user_id or "unknown"
            if key not in self.failed_attempts:
                self.failed_attempts[key] = []
            self.failed_attempts[key].append(event.timestamp)

            # Check for brute force attempts (5 failures in 5 minutes)
            recent_failures = [
                t
                for t in self.failed_attempts[key]
                if t > event.timestamp - timedelta(minutes=5)
            ]
            if len(recent_failures) >= 5:
                self.suspicious_ips.add(event.ip_address) if event.ip_address else None
                self.log_event(
                    SecurityEventType.SECURITY_VIOLATION,
                    SecuritySeverity.WARNING,
                    f"Possible brute force attempt detected from {key}",
                    ip_address=event.ip_address,
                    details={"failure_count": len(recent_failures)},
                )

        # Track injection attempts
        if event.event_type in [
            SecurityEventType.INJECTION_ATTEMPT,
            SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
            SecurityEventType.XSS_ATTEMPT,
        ]:
            key = event.ip_address or event.user_id or "unknown"
            self.violation_count[key] = self.violation_count.get(key, 0) + 1
            if self.violation_count[key] >= 3:
                self.suspicious_ips.add(event.ip_address) if event.ip_address else None

        # Clean old tracking data
        self._cleanup_tracking_data()

    def _cleanup_tracking_data(self) -> None:
        """Clean up old tracking data."""
        cutoff = datetime.utcnow() - timedelta(hours=1)

        # Clean failed attempts
        for key in list(self.failed_attempts.keys()):
            self.failed_attempts[key] = [
                t for t in self.failed_attempts[key] if t > cutoff
            ]
            if not self.failed_attempts[key]:
                del self.failed_attempts[key]

    def get_recent_events(
        self,
        limit: int = 100,
        event_type: Optional[SecurityEventType] = None,
        severity: Optional[SecuritySeverity] = None,
        user_id: Optional[str] = None,
    ) -> List[SecurityEvent]:
        """
        Get recent security events.

        Args:
            limit: Maximum number of events to return
            event_type: Filter by event type
            severity: Filter by severity
            user_id: Filter by user ID

        Returns:
            List of recent security events
        """
        events = self.recent_events.copy()

        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if severity:
            events = [e for e in events if e.severity == severity]
        if user_id:
            events = [e for e in events if e.user_id == user_id]

        # Return most recent events
        return events[-limit:][::-1]

    def generate_security_report(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> SecurityReport:
        """
        Generate a security report for the specified period.

        Args:
            period_start: Start of reporting period
            period_end: End of reporting period

        Returns:
            Security report
        """
        if not period_end:
            period_end = datetime.utcnow()
        if not period_start:
            period_start = period_end - timedelta(days=7)

        report = SecurityReport(
            period_start=period_start,
            period_end=period_end,
        )

        # Filter events in the period
        period_events = [
            e
            for e in self.recent_events
            if period_start <= e.timestamp <= period_end
        ]

        report.total_events = len(period_events)

        # Count events by type
        for event in period_events:
            event_type = event.event_type.value if isinstance(event.event_type, Enum) else event.event_type
            report.events_by_type[event_type] = (
                report.events_by_type.get(event_type, 0) + 1
            )

        # Count events by severity
        for event in period_events:
            severity = event.severity.value if isinstance(event.severity, Enum) else event.severity
            report.events_by_severity[severity] = (
                report.events_by_severity.get(severity, 0) + 1
            )

        # Find top users by activity
        user_activity = {}
        for event in period_events:
            if event.user_id:
                user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1

        report.top_users = [
            {"user_id": user_id, "event_count": count}
            for user_id, count in sorted(
                user_activity.items(), key=lambda x: x[1], reverse=True
            )[:10]
        ]

        # Collect security violations
        report.security_violations = [
            e
            for e in period_events
            if e.severity
            in [SecuritySeverity.WARNING, SecuritySeverity.ERROR, SecuritySeverity.CRITICAL]
        ]

        # Identify suspicious activities
        suspicious = []
        for ip in self.suspicious_ips:
            ip_events = [e for e in period_events if e.ip_address == ip]
            if ip_events:
                suspicious.append(
                    {
                        "ip_address": ip,
                        "event_count": len(ip_events),
                        "event_types": list(set(e.event_type.value if isinstance(e.event_type, Enum) else e.event_type for e in ip_events)),
                    }
                )
        report.suspicious_activities = suspicious

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def _generate_recommendations(self, report: SecurityReport) -> List[str]:
        """Generate security recommendations based on report data."""
        recommendations = []

        # Check for high failure rates
        auth_failures = report.events_by_type.get(
            SecurityEventType.AUTH_LOGIN_FAILURE.value, 0
        )
        if auth_failures > 50:
            recommendations.append(
                "High number of authentication failures detected. "
                "Consider implementing stronger authentication mechanisms."
            )

        # Check for injection attempts
        injection_events = sum(
            report.events_by_type.get(event_type.value, 0)
            for event_type in [
                SecurityEventType.INJECTION_ATTEMPT,
                SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
                SecurityEventType.XSS_ATTEMPT,
            ]
        )
        if injection_events > 10:
            recommendations.append(
                "Multiple injection attempts detected. "
                "Review input validation and consider implementing a WAF."
            )

        # Check for rate limiting issues
        rate_limit_events = report.events_by_type.get(
            SecurityEventType.RATE_LIMIT_EXCEEDED.value, 0
        )
        if rate_limit_events > 100:
            recommendations.append(
                "Frequent rate limit violations. "
                "Consider adjusting rate limits or investigating suspicious traffic."
            )

        # Check severity distribution
        critical_events = report.events_by_severity.get(SecuritySeverity.CRITICAL.value, 0)
        if critical_events > 0:
            recommendations.append(
                f"{critical_events} critical security events detected. "
                "Immediate investigation recommended."
            )

        # Check for suspicious IPs
        if report.suspicious_activities:
            recommendations.append(
                f"{len(report.suspicious_activities)} suspicious IP addresses identified. "
                "Consider blocking or rate limiting these addresses."
            )

        return recommendations

    def search_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[SecurityEventType]] = None,
        severities: Optional[List[SecuritySeverity]] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        resource_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[SecurityEvent]:
        """
        Search audit events with filters.

        Args:
            start_date: Start date for search
            end_date: End date for search
            event_types: List of event types to filter
            severities: List of severities to filter
            user_id: User ID to filter
            ip_address: IP address to filter
            resource_id: Resource ID to filter
            limit: Maximum results to return

        Returns:
            List of matching security events
        """
        # For now, search in memory events
        # In production, this would query a database
        events = self.recent_events.copy()

        # Apply filters
        if start_date:
            events = [e for e in events if e.timestamp >= start_date]
        if end_date:
            events = [e for e in events if e.timestamp <= end_date]
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        if severities:
            events = [e for e in events if e.severity in severities]
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if ip_address:
            events = [e for e in events if e.ip_address == ip_address]
        if resource_id:
            events = [e for e in events if e.resource_id == resource_id]

        return events[:limit]

    def cleanup_old_logs(self) -> int:
        """
        Clean up old audit logs based on retention policy.

        Returns:
            Number of files cleaned up
        """
        if not self.audit_file.parent.exists():
            return 0

        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        cleaned = 0

        for audit_file in self.audit_file.parent.glob("audit_*.jsonl"):
            try:
                # Parse date from filename
                date_str = audit_file.stem.replace("audit_", "")
                file_date = datetime.strptime(date_str, "%Y%m")

                if file_date < cutoff_date:
                    audit_file.unlink()
                    cleaned += 1
                    logger.info(f"Deleted old audit file: {audit_file}")

            except Exception as e:
                logger.error(f"Error processing audit file {audit_file}: {e}")

        return cleaned

    def export_compliance_report(
        self,
        output_path: Path,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> None:
        """
        Export compliance report for regulatory requirements.

        Args:
            output_path: Path to export report
            period_start: Start of reporting period
            period_end: End of reporting period
        """
        report = self.generate_security_report(period_start, period_end)

        # Convert to compliance format
        compliance_data = {
            "report_id": report.report_id,
            "generated_at": report.generated_at.isoformat(),
            "period": {
                "start": report.period_start.isoformat(),
                "end": report.period_end.isoformat(),
            },
            "summary": {
                "total_events": report.total_events,
                "security_violations": len(report.security_violations),
                "suspicious_activities": len(report.suspicious_activities),
            },
            "events_by_type": report.events_by_type,
            "events_by_severity": report.events_by_severity,
            "violations": [
                {
                    "event_id": v.event_id,
                    "timestamp": v.timestamp.isoformat(),
                    "type": v.event_type.value if isinstance(v.event_type, Enum) else v.event_type,
                    "severity": v.severity.value if isinstance(v.severity, Enum) else v.severity,
                    "message": v.message,
                    "user_id": v.user_id,
                    "ip_address": v.ip_address,
                }
                for v in report.security_violations
            ],
            "recommendations": report.recommendations,
        }

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(compliance_data, f, indent=2)

        logger.info(f"Exported compliance report to: {output_path}")