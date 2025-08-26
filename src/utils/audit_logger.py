"""
Audit logging system for MDMAI TTRPG Assistant.

This module provides comprehensive audit trail logging for security,
compliance, and debugging purposes.
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""

    # Authentication events
    LOGIN_SUCCESS = auto()
    LOGIN_FAILURE = auto()
    LOGOUT = auto()
    SESSION_EXPIRED = auto()
    PASSWORD_CHANGED = auto()
    
    # Authorization events
    PERMISSION_GRANTED = auto()
    PERMISSION_DENIED = auto()
    ROLE_CHANGED = auto()
    
    # Data access events
    DATA_READ = auto()
    DATA_CREATED = auto()
    DATA_UPDATED = auto()
    DATA_DELETED = auto()
    DATA_EXPORTED = auto()
    DATA_IMPORTED = auto()
    
    # System events
    SYSTEM_START = auto()
    SYSTEM_STOP = auto()
    CONFIG_CHANGED = auto()
    SERVICE_ERROR = auto()
    
    # Security events
    SECURITY_VIOLATION = auto()
    SUSPICIOUS_ACTIVITY = auto()
    RATE_LIMIT_EXCEEDED = auto()
    
    # MCP events
    MCP_TOOL_CALLED = auto()
    MCP_TOOL_SUCCESS = auto()
    MCP_TOOL_FAILURE = auto()
    
    # Game events
    CHARACTER_ACTION = auto()
    CAMPAIGN_EVENT = auto()
    COMBAT_ACTION = auto()
    RULES_QUERY = auto()


class AuditSeverity(Enum):
    """Severity levels for audit events."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Represents an audit event."""

    event_type: AuditEventType
    severity: AuditSeverity
    user_id: Optional[str]
    session_id: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    action: str
    outcome: str
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Generate event ID if not provided."""
        if not self.event_id:
            self.event_id = self._generate_event_id()
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        data = f"{self.timestamp.isoformat()}{self.event_type.name}{self.user_id}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "outcome": self.outcome,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
        }
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """Main audit logging system."""

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        encryption_key: Optional[bytes] = None,
        retention_days: int = 90,
        enable_console: bool = False,
    ) -> None:
        """
        Initialize audit logger.

        Args:
            log_dir: Directory for audit logs (defaults to ~/.mdmai/audit or ./logs/audit)
            encryption_key: Optional encryption key for sensitive data
            retention_days: Number of days to retain logs
            enable_console: Enable console output
        """
        if log_dir is None:
            # Try user home directory first, fall back to local directory
            home_audit_dir = Path.home() / ".mdmai" / "audit"
            local_audit_dir = Path("./logs/audit")
            
            # Use home directory if writable, otherwise use local directory
            try:
                home_audit_dir.mkdir(parents=True, exist_ok=True)
                self.log_dir = home_audit_dir
            except (OSError, PermissionError):
                local_audit_dir.mkdir(parents=True, exist_ok=True)
                self.log_dir = local_audit_dir
        else:
            self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.retention_days = retention_days
        self.enable_console = enable_console
        
        # Setup encryption if key provided
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            self.cipher = None
        
        # Setup logger
        self.logger = self._setup_logger()
        
        # Event buffer for batch writing
        self.event_buffer: List[AuditEvent] = []
        self.buffer_size = 100
        self._lock = asyncio.Lock()
        
        # Statistics
        self.stats: Dict[str, int] = {}

    def _setup_logger(self) -> logging.Logger:
        """Setup audit logger configuration."""
        logger = logging.getLogger("audit")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10,
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(message)s")
        )
        logger.addHandler(file_handler)
        
        # Console handler if enabled
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - AUDIT - %(message)s")
            )
            logger.addHandler(console_handler)
        
        return logger

    async def log_event(self, event: AuditEvent) -> None:
        """
        Log an audit event.

        Args:
            event: Audit event to log
        """
        async with self._lock:
            # Add to buffer
            self.event_buffer.append(event)
            
            # Update statistics
            event_type = event.event_type.name
            self.stats[event_type] = self.stats.get(event_type, 0) + 1
            
            # Flush if buffer is full
            if len(self.event_buffer) >= self.buffer_size:
                await self._flush_buffer()

    async def _flush_buffer(self) -> None:
        """Flush event buffer to log file."""
        if not self.event_buffer:
            return
        
        for event in self.event_buffer:
            # Encrypt sensitive data if cipher available
            log_data = event.to_dict()
            if self.cipher and event.details.get("sensitive"):
                log_data["details"] = self._encrypt_data(log_data["details"])
            
            # Log the event
            self.logger.info(json.dumps(log_data))
        
        self.event_buffer.clear()

    def _encrypt_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive data fields."""
        encrypted = {}
        for key, value in data.items():
            if key in ["password", "token", "secret", "key"]:
                encrypted[key] = self.cipher.encrypt(
                    json.dumps(value).encode()
                ).decode()
            else:
                encrypted[key] = value
        return encrypted

    async def log_authentication(
        self,
        user_id: str,
        action: str,
        success: bool,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log authentication event.

        Args:
            user_id: User identifier
            action: Authentication action
            success: Whether action succeeded
            ip_address: Client IP address
            details: Additional details
        """
        event_type = AuditEventType.LOGIN_SUCCESS if success else AuditEventType.LOGIN_FAILURE
        severity = AuditSeverity.INFO if success else AuditSeverity.WARNING
        
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            action=action,
            outcome="success" if success else "failure",
            ip_address=ip_address,
            details=details or {},
        )
        
        await self.log_event(event)

    async def log_data_access(
        self,
        user_id: str,
        operation: str,
        resource_type: str,
        resource_id: str,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log data access event.

        Args:
            user_id: User identifier
            operation: Operation performed (read/write/delete)
            resource_type: Type of resource accessed
            resource_id: Resource identifier
            success: Whether operation succeeded
            details: Additional details
        """
        event_map = {
            "read": AuditEventType.DATA_READ,
            "create": AuditEventType.DATA_CREATED,
            "update": AuditEventType.DATA_UPDATED,
            "delete": AuditEventType.DATA_DELETED,
        }
        
        event = AuditEvent(
            event_type=event_map.get(operation, AuditEventType.DATA_READ),
            severity=AuditSeverity.INFO,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=operation,
            outcome="success" if success else "failure",
            details=details or {},
        )
        
        await self.log_event(event)

    async def log_mcp_tool(
        self,
        tool_name: str,
        user_id: Optional[str],
        parameters: Dict[str, Any],
        result: Any,
        success: bool,
        duration_ms: float,
    ) -> None:
        """
        Log MCP tool execution.

        Args:
            tool_name: Name of MCP tool
            user_id: User identifier
            parameters: Tool parameters
            result: Tool result
            success: Whether execution succeeded
            duration_ms: Execution duration
        """
        event = AuditEvent(
            event_type=AuditEventType.MCP_TOOL_SUCCESS if success else AuditEventType.MCP_TOOL_FAILURE,
            severity=AuditSeverity.INFO if success else AuditSeverity.ERROR,
            user_id=user_id,
            action=f"execute_tool:{tool_name}",
            outcome="success" if success else "failure",
            details={
                "tool": tool_name,
                "parameters": parameters,
                "result": str(result)[:500] if success else str(result),
                "duration_ms": duration_ms,
            },
        )
        
        await self.log_event(event)

    async def log_security_event(
        self,
        event_type: str,
        severity: AuditSeverity,
        user_id: Optional[str],
        description: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log security event.

        Args:
            event_type: Type of security event
            severity: Event severity
            user_id: User identifier if applicable
            description: Event description
            details: Additional details
        """
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_VIOLATION,
            severity=severity,
            user_id=user_id,
            action=event_type,
            outcome="detected",
            details={
                "description": description,
                **(details or {}),
            },
        )
        
        await self.log_event(event)

    async def search_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        resource_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Search audit events.

        Args:
            start_time: Start time for search
            end_time: End time for search
            user_id: Filter by user ID
            event_type: Filter by event type
            resource_type: Filter by resource type
            limit: Maximum results

        Returns:
            List of matching events
        """
        results = []
        
        # Flush buffer first
        async with self._lock:
            await self._flush_buffer()
        
        # Search log files
        for log_file in sorted(self.log_dir.glob("audit_*.log"), reverse=True):
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        # More robust parsing - try multiple formats
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Try to parse as pure JSON first
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            # Try to extract JSON from log format (timestamp - JSON)
                            # Look for the first '{' character to find JSON start
                            json_start = line.find('{')
                            if json_start != -1:
                                event = json.loads(line[json_start:])
                            else:
                                continue
                        
                        # Apply filters
                        if start_time and datetime.fromisoformat(event["timestamp"]) < start_time:
                            continue
                        if end_time and datetime.fromisoformat(event["timestamp"]) > end_time:
                            continue
                        if user_id and event.get("user_id") != user_id:
                            continue
                        if event_type and event.get("event_type") != event_type.name:
                            continue
                        if resource_type and event.get("resource_type") != resource_type:
                            continue
                        
                        results.append(event)
                        
                        if len(results) >= limit:
                            return results
                            
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        # Log parsing errors for debugging but continue processing
                        logger.debug(f"Failed to parse log line: {e}")
                        continue
        
        return results

    async def get_statistics(
        self,
        time_range: Optional[timedelta] = None,
    ) -> Dict[str, Any]:
        """
        Get audit statistics.

        Args:
            time_range: Time range for statistics

        Returns:
            Statistics dictionary
        """
        start_time = None
        if time_range:
            start_time = datetime.utcnow() - time_range
        
        events = await self.search_events(start_time=start_time)
        
        # Calculate statistics
        stats = {
            "total_events": len(events),
            "events_by_type": {},
            "events_by_severity": {},
            "events_by_user": {},
            "events_by_hour": {},
        }
        
        for event in events:
            # By type
            event_type = event.get("event_type", "unknown")
            stats["events_by_type"][event_type] = stats["events_by_type"].get(event_type, 0) + 1
            
            # By severity
            severity = event.get("severity", "unknown")
            stats["events_by_severity"][severity] = stats["events_by_severity"].get(severity, 0) + 1
            
            # By user
            user = event.get("user_id", "anonymous")
            stats["events_by_user"][user] = stats["events_by_user"].get(user, 0) + 1
            
            # By hour
            timestamp = datetime.fromisoformat(event["timestamp"])
            hour = timestamp.strftime("%Y-%m-%d %H:00")
            stats["events_by_hour"][hour] = stats["events_by_hour"].get(hour, 0) + 1
        
        return stats

    async def cleanup_old_logs(self) -> int:
        """
        Clean up old audit logs.

        Returns:
            Number of files deleted
        """
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        deleted = 0
        
        for log_file in self.log_dir.glob("audit_*.log"):
            # Parse date from filename
            try:
                date_str = log_file.stem.split("_")[1]
                file_date = datetime.strptime(date_str, "%Y%m%d")
                
                if file_date < cutoff_date:
                    log_file.unlink()
                    deleted += 1
                    self.logger.info(f"Deleted old audit log: {log_file}")
                    
            except (ValueError, IndexError):
                continue
        
        return deleted


class ComplianceReporter:
    """Generate compliance reports from audit logs."""

    def __init__(self, audit_logger: AuditLogger) -> None:
        """
        Initialize compliance reporter.

        Args:
            audit_logger: Audit logger instance
        """
        self.audit_logger = audit_logger

    async def generate_access_report(
        self,
        start_date: datetime,
        end_date: datetime,
        resource_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate data access report.

        Args:
            start_date: Report start date
            end_date: Report end date
            resource_type: Optional resource type filter

        Returns:
            Access report
        """
        events = await self.audit_logger.search_events(
            start_time=start_date,
            end_time=end_date,
            resource_type=resource_type,
        )
        
        report = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "total_accesses": 0,
            "unique_users": set(),
            "access_by_type": {},
            "access_by_user": {},
            "sensitive_access": [],
        }
        
        for event in events:
            if "DATA_" in event.get("event_type", ""):
                report["total_accesses"] += 1
                report["unique_users"].add(event.get("user_id", "unknown"))
                
                # By type
                resource = event.get("resource_type", "unknown")
                report["access_by_type"][resource] = report["access_by_type"].get(resource, 0) + 1
                
                # By user
                user = event.get("user_id", "unknown")
                report["access_by_user"][user] = report["access_by_user"].get(user, 0) + 1
                
                # Check for sensitive access
                if event.get("details", {}).get("sensitive"):
                    report["sensitive_access"].append({
                        "timestamp": event["timestamp"],
                        "user": user,
                        "resource": f"{resource}:{event.get('resource_id')}",
                        "action": event.get("action"),
                    })
        
        report["unique_users"] = len(report["unique_users"])
        return report

    async def generate_security_report(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """
        Generate security report.

        Args:
            start_date: Report start date
            end_date: Report end date

        Returns:
            Security report
        """
        events = await self.audit_logger.search_events(
            start_time=start_date,
            end_time=end_date,
        )
        
        report = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "security_events": [],
            "failed_logins": [],
            "permission_denials": [],
            "suspicious_activities": [],
            "summary": {
                "total_security_events": 0,
                "critical_events": 0,
                "failed_login_attempts": 0,
                "unique_threat_actors": set(),
            },
        }
        
        for event in events:
            event_type = event.get("event_type", "")
            
            # Security violations
            if "SECURITY" in event_type or "SUSPICIOUS" in event_type:
                report["security_events"].append(event)
                report["summary"]["total_security_events"] += 1
                
                if event.get("severity") == "critical":
                    report["summary"]["critical_events"] += 1
            
            # Failed logins
            elif event_type == "LOGIN_FAILURE":
                report["failed_logins"].append({
                    "timestamp": event["timestamp"],
                    "user": event.get("user_id"),
                    "ip": event.get("ip_address"),
                })
                report["summary"]["failed_login_attempts"] += 1
                report["summary"]["unique_threat_actors"].add(
                    event.get("ip_address", "unknown")
                )
            
            # Permission denials
            elif event_type == "PERMISSION_DENIED":
                report["permission_denials"].append({
                    "timestamp": event["timestamp"],
                    "user": event.get("user_id"),
                    "resource": event.get("resource_type"),
                    "action": event.get("action"),
                })
        
        report["summary"]["unique_threat_actors"] = len(
            report["summary"]["unique_threat_actors"]
        )
        
        return report


# Global audit logger instance
audit_logger = AuditLogger()