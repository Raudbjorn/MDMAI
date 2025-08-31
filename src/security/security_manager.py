"""Main security manager coordinating all security components."""

import functools
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel

from config.logging_config import get_logger
from config.settings import settings
from src.security.access_control import (
    AccessControlManager,
    AccessLevel,
    Permission,
    ResourceType,
    User,
)
from src.security.input_validator import (
    CampaignParameters,
    FilePathParameters,
    InputValidator,
    SearchParameters,
    ValidationResult,
)
from src.security.rate_limiter import (
    OperationType,
    RateLimiter,
    RateLimitConfig,
    RateLimitStatus,
)
from src.security.security_audit import (
    SecurityAuditTrail,
    SecurityEventType,
    SecurityReport,
    SecuritySeverity,
)

logger = get_logger(__name__)


class SecurityConfig:
    """Security configuration."""

    def __init__(
        self,
        enable_authentication: bool = False,
        enable_rate_limiting: bool = True,
        enable_audit: bool = True,
        enable_input_validation: bool = True,
        session_timeout_minutes: int = 60,
        audit_retention_days: int = 90,
        allowed_directories: Optional[List[Path]] = None,
        custom_rate_limits: Optional[Dict[OperationType, RateLimitConfig]] = None,
    ):
        """Initialize security configuration."""
        self.enable_authentication = enable_authentication
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_audit = enable_audit
        self.enable_input_validation = enable_input_validation
        self.session_timeout_minutes = session_timeout_minutes
        self.audit_retention_days = audit_retention_days
        self.allowed_directories = allowed_directories or [
            Path(settings.chroma_db_path),
            Path(settings.cache_dir),
            Path("/tmp"),
        ]
        self.custom_rate_limits = custom_rate_limits


class SecurityManager:
    """Manages and coordinates all security components."""

    def __init__(self, config: Optional[SecurityConfig] = None):
        """
        Initialize security manager.

        Args:
            config: Security configuration
        """
        self.config = config or SecurityConfig()

        # Initialize components
        self.access_control = AccessControlManager(
            enable_auth=self.config.enable_authentication,
            session_timeout_minutes=self.config.session_timeout_minutes,
        )

        self.input_validator = InputValidator(
            strict_mode=self.config.enable_input_validation
        )

        self.rate_limiter = RateLimiter(
            custom_limits=self.config.custom_rate_limits,
            enable_rate_limiting=self.config.enable_rate_limiting,
        )

        self.audit_trail = SecurityAuditTrail(
            enable_audit=self.config.enable_audit,
            retention_days=self.config.audit_retention_days,
        )

        # Track active sessions/users
        self.active_sessions: Dict[str, User] = {}

        logger.info(
            "Security manager initialized",
            auth_enabled=self.config.enable_authentication,
            rate_limiting_enabled=self.config.enable_rate_limiting,
            audit_enabled=self.config.enable_audit,
            validation_enabled=self.config.enable_input_validation,
        )

    def authenticate(
        self, username: str, password: str, ip_address: Optional[str] = None
    ) -> Optional[str]:
        """
        Authenticate user and return session token.

        Args:
            username: Username
            password: Password
            ip_address: Client IP address

        Returns:
            Session token if successful
        """
        session = self.access_control.authenticate_user(username, password, ip_address)

        if session:
            self.audit_trail.log_event(
                SecurityEventType.AUTH_LOGIN_SUCCESS,
                SecuritySeverity.INFO,
                f"User {username} logged in successfully",
                user_id=session.user_id,
                session_id=session.session_id,
                ip_address=ip_address,
            )
            return session.token
        else:
            self.audit_trail.log_event(
                SecurityEventType.AUTH_LOGIN_FAILURE,
                SecuritySeverity.WARNING,
                f"Failed login attempt for user {username}",
                ip_address=ip_address,
                details={"username": username},
            )
            return None

    def validate_session(self, session_token: str) -> Optional[User]:
        """
        Validate session and return user.

        Args:
            session_token: Session token

        Returns:
            User if session is valid
        """
        return self.access_control.validate_session(session_token)

    def check_permission(
        self,
        user: Optional[User],
        permission: Permission,
        resource_type: Optional[ResourceType] = None,
        resource_id: Optional[str] = None,
    ) -> bool:
        """
        Check if user has permission.

        Args:
            user: User to check (None for anonymous)
            permission: Required permission
            resource_type: Type of resource
            resource_id: Resource ID

        Returns:
            True if permission granted
        """
        if not self.config.enable_authentication:
            # Get default admin user when auth is disabled
            user = self.access_control.users.get("default_admin")

        if not user:
            self.audit_trail.log_event(
                SecurityEventType.ACCESS_DENIED,
                SecuritySeverity.WARNING,
                f"Anonymous access denied for permission {permission.value}",
                resource_type=resource_type.value if resource_type else None,
                resource_id=resource_id,
                result="failure",
            )
            return False

        allowed = self.access_control.check_permission(
            user, permission, resource_type, resource_id
        )

        if allowed:
            self.audit_trail.log_event(
                SecurityEventType.ACCESS_GRANTED,
                SecuritySeverity.DEBUG,
                f"Access granted for {permission.value}",
                user_id=user.user_id,
                resource_type=resource_type.value if resource_type else None,
                resource_id=resource_id,
                result="success",
            )
        else:
            self.audit_trail.log_event(
                SecurityEventType.ACCESS_DENIED,
                SecuritySeverity.WARNING,
                f"Access denied for {permission.value}",
                user_id=user.user_id,
                resource_type=resource_type.value if resource_type else None,
                resource_id=resource_id,
                result="failure",
            )

        return allowed

    def check_rate_limit(
        self,
        client_id: str,
        operation: OperationType,
        consume: bool = True,
    ) -> RateLimitStatus:
        """
        Check rate limit for an operation.

        Args:
            client_id: Client identifier
            operation: Operation type
            consume: Whether to consume from limit

        Returns:
            Rate limit status
        """
        status = self.rate_limiter.check_rate_limit(client_id, operation, consume)

        if not status.allowed:
            self.audit_trail.log_event(
                SecurityEventType.RATE_LIMIT_EXCEEDED,
                SecuritySeverity.WARNING,
                f"Rate limit exceeded for {operation.value}",
                details={
                    "client_id": client_id,
                    "operation": operation.value,
                    "retry_after": status.retry_after,
                },
            )

        return status

    def validate_input(
        self,
        value: Any,
        input_type: str = "general",
        max_length: Optional[int] = None,
        model: Optional[Type[BaseModel]] = None,
    ) -> ValidationResult:
        """
        Validate and sanitize input.

        Args:
            value: Input value
            input_type: Type of input
            max_length: Maximum length
            model: Pydantic model for validation

        Returns:
            Validation result
        """
        # Use Pydantic model if provided
        if model:
            result = self.input_validator.validate_data_type(value, dict, model)
        else:
            result = self.input_validator.validate_input(value, input_type, max_length)

        # Log validation failures
        if not result.is_valid:
            # Check for injection attempts
            if any(
                keyword in str(err).lower()
                for err in result.errors
                for keyword in ["injection", "xss", "traversal"]
            ):
                event_type = SecurityEventType.INJECTION_ATTEMPT
                severity = SecuritySeverity.ERROR
            else:
                event_type = SecurityEventType.INPUT_VALIDATION_FAILED
                severity = SecuritySeverity.WARNING

            self.audit_trail.log_event(
                event_type,
                severity,
                f"Input validation failed for {input_type}",
                details={
                    "input_type": input_type,
                    "errors": result.errors,
                    "value_preview": str(value)[:100] if value else None,
                },
            )

        return result

    def validate_file_path(
        self,
        path: str,
        must_exist: bool = False,
        allowed_extensions: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        Validate file path for security.

        Args:
            path: File path
            must_exist: Whether file must exist
            allowed_extensions: Allowed file extensions

        Returns:
            Validation result
        """
        result = self.input_validator.validate_file_path(
            path,
            allowed_dirs=self.config.allowed_directories,
            must_exist=must_exist,
            allowed_extensions=allowed_extensions,
        )

        if not result.is_valid:
            # Check if it's a path traversal attempt
            if "traversal" in str(result.errors).lower():
                self.audit_trail.log_event(
                    SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
                    SecuritySeverity.CRITICAL,
                    f"Path traversal attempt detected: {path[:100]}",
                    details={"path": path[:500], "errors": result.errors},
                )

        return result

    def secure_tool(
        self,
        permission: Optional[Permission] = None,
        operation_type: Optional[OperationType] = None,
        validate_params: Optional[Dict[str, Type[BaseModel]]] = None,
        resource_type: Optional[ResourceType] = None,
        audit_event: Optional[SecurityEventType] = None,
    ) -> Callable:
        """
        Decorator to secure MCP tools with comprehensive security checks.

        Args:
            permission: Required permission
            operation_type: Operation type for rate limiting
            validate_params: Parameter validation models
            resource_type: Resource type for access control
            audit_event: Security event to log

        Returns:
            Decorated function
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                # Get session/user info (would come from MCP context in production)
                session_token = kwargs.pop("_session_token", None)
                client_id = kwargs.pop("_client_id", "default")
                ip_address = kwargs.pop("_ip_address", None)

                user = None
                if session_token:
                    user = self.validate_session(session_token)
                elif not self.config.enable_authentication:
                    user = self.access_control.users.get("default_admin")

                # Check permission
                if permission and self.config.enable_authentication:
                    resource_id = kwargs.get("campaign_id") or kwargs.get("resource_id")
                    if not self.check_permission(user, permission, resource_type, resource_id):
                        return {
                            "status": "error",
                            "error": "Permission denied",
                            "required_permission": permission.value,
                        }

                # Check rate limit
                if operation_type and self.config.enable_rate_limiting:
                    rate_status = self.check_rate_limit(client_id, operation_type)
                    if not rate_status.allowed:
                        return {
                            "status": "error",
                            "error": "Rate limit exceeded",
                            "retry_after": rate_status.retry_after,
                            "message": rate_status.message,
                        }

                # Validate parameters
                if validate_params and self.config.enable_input_validation:
                    for param_name, model_class in validate_params.items():
                        if param_name in kwargs:
                            validation = self.validate_input(
                                kwargs[param_name],
                                input_type=param_name,
                                model=model_class,
                            )
                            if not validation.is_valid:
                                return {
                                    "status": "error",
                                    "error": "Invalid parameters",
                                    "validation_errors": validation.errors,
                                }
                            # Replace with sanitized value
                            if validation.value is not None:
                                kwargs[param_name] = validation.value

                # Log audit event for operation start
                if audit_event and self.config.enable_audit:
                    self.audit_trail.log_event(
                        audit_event,
                        SecuritySeverity.INFO,
                        f"Executing {func.__name__}",
                        user_id=user.user_id if user else None,
                        ip_address=ip_address,
                        action=func.__name__,
                        details={"parameters": list(kwargs.keys())},
                    )

                try:
                    # Execute the function
                    result = await func(*args, **kwargs)

                    # Log successful operation
                    if audit_event and self.config.enable_audit:
                        self.audit_trail.log_event(
                            audit_event,
                            SecuritySeverity.INFO,
                            f"Successfully completed {func.__name__}",
                            user_id=user.user_id if user else None,
                            ip_address=ip_address,
                            action=func.__name__,
                            result="success",
                        )

                    return result

                except Exception as e:
                    # Log operation failure
                    if self.config.enable_audit:
                        self.audit_trail.log_event(
                            SecurityEventType.SECURITY_VIOLATION,
                            SecuritySeverity.ERROR,
                            f"Error in {func.__name__}: {str(e)}",
                            user_id=user.user_id if user else None,
                            ip_address=ip_address,
                            action=func.__name__,
                            result="failure",
                            details={"error": str(e)},
                        )
                    raise

            return wrapper

        return decorator

    def validate_search_params(self, **params) -> ValidationResult:
        """Validate search parameters."""
        try:
            validated = SearchParameters(**params)
            return ValidationResult(True, validated.model_dump())
        except Exception as e:
            return ValidationResult(False, None, [str(e)])

    def validate_campaign_params(self, **params) -> ValidationResult:
        """Validate campaign parameters."""
        try:
            validated = CampaignParameters(**params)
            return ValidationResult(True, validated.model_dump())
        except Exception as e:
            return ValidationResult(False, None, [str(e)])

    def validate_file_params(self, **params) -> ValidationResult:
        """Validate file operation parameters."""
        try:
            validated = FilePathParameters(**params)
            return ValidationResult(True, validated.model_dump())
        except Exception as e:
            return ValidationResult(False, None, [str(e)])

    def grant_campaign_access(
        self, user_id: str, campaign_id: str, access_level: AccessLevel
    ) -> bool:
        """
        Grant user access to a campaign.

        Args:
            user_id: User ID
            campaign_id: Campaign ID
            access_level: Access level

        Returns:
            True if successful
        """
        success = self.access_control.grant_campaign_access(
            user_id, campaign_id, access_level
        )

        if success:
            self.audit_trail.log_event(
                SecurityEventType.PERMISSION_ELEVATED,
                SecuritySeverity.INFO,
                f"Granted {access_level.name} access to campaign",
                user_id=user_id,
                resource_id=campaign_id,
                resource_type="campaign",
            )

        return success

    def revoke_campaign_access(self, user_id: str, campaign_id: str) -> bool:
        """
        Revoke user access to a campaign.

        Args:
            user_id: User ID
            campaign_id: Campaign ID

        Returns:
            True if successful
        """
        success = self.access_control.revoke_campaign_access(user_id, campaign_id)

        if success:
            self.audit_trail.log_event(
                SecurityEventType.PERMISSION_REVOKED,
                SecuritySeverity.INFO,
                "Revoked access to campaign",
                user_id=user_id,
                resource_id=campaign_id,
                resource_type="campaign",
            )

        return success

    def set_campaign_owner(self, campaign_id: str, user_id: str) -> None:
        """
        Set campaign owner.

        Args:
            campaign_id: Campaign ID
            user_id: User ID
        """
        self.access_control.set_campaign_owner(campaign_id, user_id)

        self.audit_trail.log_event(
            SecurityEventType.CAMPAIGN_CREATED,
            SecuritySeverity.INFO,
            "Campaign ownership set",
            user_id=user_id,
            resource_id=campaign_id,
            resource_type="campaign",
        )

    def get_security_report(
        self,
        period_start: Optional[Any] = None,
        period_end: Optional[Any] = None,
    ) -> SecurityReport:
        """
        Get security report.

        Args:
            period_start: Start of period
            period_end: End of period

        Returns:
            Security report
        """
        return self.audit_trail.generate_security_report(period_start, period_end)

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned
        """
        count = self.access_control.cleanup_expired_sessions()
        if count > 0:
            logger.info(f"Cleaned up {count} expired sessions")
        return count

    def cleanup_old_audit_logs(self) -> int:
        """
        Clean up old audit logs.

        Returns:
            Number of files cleaned
        """
        count = self.audit_trail.cleanup_old_logs()
        if count > 0:
            logger.info(f"Cleaned up {count} old audit log files")
        return count

    def cleanup_rate_limit_entries(self) -> int:
        """
        Clean up old rate limit entries.

        Returns:
            Number of entries cleaned
        """
        count = self.rate_limiter.cleanup_old_entries()
        if count > 0:
            logger.info(f"Cleaned up {count} old rate limit entries")
        return count

    def perform_security_maintenance(self) -> Dict[str, int]:
        """
        Perform all security maintenance tasks.

        Returns:
            Dictionary with cleanup counts
        """
        return {
            "sessions": self.cleanup_expired_sessions(),
            "audit_logs": self.cleanup_old_audit_logs(),
            "rate_limits": self.cleanup_rate_limit_entries(),
        }


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


def initialize_security(config: Optional[SecurityConfig] = None) -> SecurityManager:
    """
    Initialize the global security manager.

    Args:
        config: Security configuration

    Returns:
        Security manager instance
    """
    global _security_manager
    _security_manager = SecurityManager(config)
    return _security_manager


def get_security_manager() -> SecurityManager:
    """
    Get the global security manager instance.

    Returns:
        Security manager instance
    """
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


def secure_mcp_tool(
    permission: Optional[Permission] = None,
    operation_type: Optional[OperationType] = None,
    validate_params: Optional[Dict[str, Type[BaseModel]]] = None,
    resource_type: Optional[ResourceType] = None,
    audit_event: Optional[SecurityEventType] = None,
) -> Callable:
    """
    Decorator to secure MCP tools.

    Args:
        permission: Required permission
        operation_type: Operation type for rate limiting
        validate_params: Parameter validation models
        resource_type: Resource type
        audit_event: Audit event type

    Returns:
        Decorator function
    """
    manager = get_security_manager()
    return manager.secure_tool(
        permission=permission,
        operation_type=operation_type,
        validate_params=validate_params,
        resource_type=resource_type,
        audit_event=audit_event,
    )