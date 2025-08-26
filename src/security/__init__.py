"""Security module for TTRPG Assistant MCP Server."""

from src.security.access_control import (
    AccessControlManager,
    AccessLevel,
    Permission,
    ResourceType,
    Role,
    Session,
    User,
)
from src.security.input_validator import (
    CampaignParameters,
    DataTypeValidationError,
    FilePathParameters,
    InjectionAttackError,
    InputValidator,
    PathTraversalError,
    SearchParameters,
    SecurityValidationError,
    ValidationResult,
    validate_and_sanitize,
)
from src.security.rate_limiter import (
    OperationType,
    RateLimiter,
    RateLimitConfig,
    RateLimitStatus,
    RateLimitStrategy,
)
from src.security.security_audit import (
    SecurityAuditTrail,
    SecurityEvent,
    SecurityEventType,
    SecurityReport,
    SecuritySeverity,
)
from src.security.security_manager import (
    SecurityConfig,
    SecurityManager,
    get_security_manager,
    initialize_security,
    secure_mcp_tool,
)

__all__ = [
    # Access Control
    "AccessControlManager",
    "AccessLevel",
    "Permission",
    "ResourceType",
    "Role",
    "Session",
    "User",
    # Input Validation
    "InputValidator",
    "ValidationResult",
    "SecurityValidationError",
    "InjectionAttackError",
    "PathTraversalError",
    "DataTypeValidationError",
    "SearchParameters",
    "CampaignParameters",
    "FilePathParameters",
    "validate_and_sanitize",
    # Rate Limiting
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitStatus",
    "RateLimitStrategy",
    "OperationType",
    # Security Audit
    "SecurityAuditTrail",
    "SecurityEvent",
    "SecurityEventType",
    "SecurityReport",
    "SecuritySeverity",
    # Security Manager
    "SecurityManager",
    "SecurityConfig",
    "initialize_security",
    "get_security_manager",
    "secure_mcp_tool",
]