# Security Module Documentation

## Overview

The security module provides comprehensive security features for the TTRPG Assistant MCP Server, implementing Phase 12 requirements for security and validation. This module ensures data integrity, prevents malicious attacks, and provides access control and audit capabilities.

## Components

### 1. Input Validator (`input_validator.py`)

Provides comprehensive input validation and sanitization to prevent injection attacks.

**Features:**
- SQL injection detection and prevention
- XSS (Cross-Site Scripting) detection
- Path traversal prevention
- Command injection detection
- LDAP injection detection
- Data type validation using Pydantic models
- Parameter range and format validation
- File path security validation

**Usage:**
```python
from src.security import InputValidator, ValidationResult

validator = InputValidator(strict_mode=True)

# Validate general input
result = validator.validate_input(
    value="user input",
    input_type="query",
    max_length=1000
)

if result.is_valid:
    sanitized_value = result.value
else:
    errors = result.errors
```

### 2. Access Control (`access_control.py`)

Manages user authentication, authorization, and campaign-level isolation.

**Features:**
- User and role management
- Permission-based access control
- Campaign-level isolation
- Session management
- Resource-level access control
- Hierarchical permission system

**Permissions:**
- Campaign operations (create, read, update, delete, rollback)
- Source management (add, read, delete, update)
- Search operations (basic, advanced, analytics)
- Character operations (create, read, update, delete)
- Session management (create, read, update, delete)
- Personality management (create, read, update, delete, apply)
- System operations (admin, config, monitor)
- Cache operations (read, clear, config)

**Usage:**
```python
from src.security import AccessControlManager, Permission, AccessLevel

acm = AccessControlManager(enable_auth=True)

# Create user
user = acm.create_user("username", email="user@example.com")

# Check permission
if acm.check_permission(user, Permission.CAMPAIGN_CREATE):
    # User has permission
    pass

# Grant campaign access
acm.grant_campaign_access(user.user_id, campaign_id, AccessLevel.READ)
```

### 3. Rate Limiter (`rate_limiter.py`)

Implements sophisticated rate limiting to prevent abuse and ensure fair resource usage.

**Features:**
- Multiple rate limiting strategies:
  - Sliding window
  - Token bucket
  - Fixed window
  - Leaky bucket
- Per-operation rate limits
- Global rate limiting
- Client-specific tracking
- Configurable limits per operation type

**Default Limits:**
- Search basic: 100 requests/minute
- Search advanced: 50 requests/minute
- Campaign operations: 30-200 requests/minute depending on operation
- Source addition: 5 per 5 minutes (token bucket)
- Character generation: 20 per minute (token bucket)
- Cache clear: 5 per 5 minutes
- Index update: 2 per 5 minutes

**Usage:**
```python
from src.security import RateLimiter, OperationType, RateLimitConfig

limiter = RateLimiter(enable_rate_limiting=True)

# Check rate limit
status = limiter.check_rate_limit(
    client_id="user123",
    operation=OperationType.SEARCH_BASIC
)

if status.allowed:
    # Process request
    remaining = status.remaining_requests
else:
    # Rate limit exceeded
    retry_after = status.retry_after
```

### 4. Security Audit (`security_audit.py`)

Provides comprehensive audit trail and security event logging for compliance and monitoring.

**Features:**
- Security event logging
- Audit trail for all security-relevant events
- Suspicious activity detection
- Security report generation
- Compliance report export
- Event search and filtering
- Automatic log retention management

**Event Types:**
- Authentication events (login, logout, session expiry)
- Authorization events (access granted/denied, permission changes)
- Validation events (validation failures, injection attempts)
- Rate limiting events
- Data access events (access, modification, deletion)
- Campaign and source events
- System configuration changes

**Usage:**
```python
from src.security import SecurityAuditTrail, SecurityEventType, SecuritySeverity

audit = SecurityAuditTrail(enable_audit=True)

# Log security event
audit.log_event(
    SecurityEventType.AUTH_LOGIN_SUCCESS,
    SecuritySeverity.INFO,
    "User logged in successfully",
    user_id="user123",
    ip_address="192.168.1.1"
)

# Generate security report
report = audit.generate_security_report()

# Search events
events = audit.search_events(
    event_types=[SecurityEventType.INJECTION_ATTEMPT],
    severities=[SecuritySeverity.CRITICAL]
)
```

### 5. Security Manager (`security_manager.py`)

Central coordinator for all security components, providing a unified interface.

**Features:**
- Centralized security configuration
- Tool decoration for automatic security
- Integration with MCP tools
- Security maintenance tasks
- Unified security API

**Usage:**
```python
from src.security import SecurityManager, SecurityConfig, secure_mcp_tool

# Initialize security
config = SecurityConfig(
    enable_authentication=True,
    enable_rate_limiting=True,
    enable_audit=True,
    enable_input_validation=True
)
security_manager = SecurityManager(config)

# Secure an MCP tool
@secure_mcp_tool(
    permission=Permission.SEARCH_BASIC,
    operation_type=OperationType.SEARCH_BASIC,
    audit_event=SecurityEventType.DATA_ACCESS
)
async def search_tool(query: str, max_results: int = 5):
    # Tool implementation
    pass
```

## Configuration

Security settings can be configured via environment variables or the settings file:

```bash
# Authentication
ENABLE_AUTHENTICATION=false  # Enable user authentication
SESSION_TIMEOUT_MINUTES=60   # Session timeout

# Rate Limiting
ENABLE_RATE_LIMITING=true    # Enable rate limiting

# Audit
ENABLE_AUDIT=true            # Enable security audit
AUDIT_RETENTION_DAYS=90      # Audit log retention period

# Input Validation
ENABLE_INPUT_VALIDATION=true # Enable input validation

# Security Log
SECURITY_LOG_FILE=/path/to/security.log  # Optional separate security log
```

## Security Best Practices

### 1. Input Validation
- Always validate and sanitize user inputs
- Use Pydantic models for structured data validation
- Apply context-specific sanitization (query, path, metadata)
- Validate file paths against allowed directories

### 2. Access Control
- Enable authentication for production environments
- Use role-based access control (RBAC)
- Implement campaign-level isolation
- Regularly review and audit permissions

### 3. Rate Limiting
- Configure appropriate limits for each operation
- Use token bucket for burst-prone operations
- Monitor rate limit violations for suspicious activity
- Adjust limits based on usage patterns

### 4. Security Monitoring
- Enable audit logging for compliance
- Regularly review security reports
- Monitor for suspicious activity patterns
- Export compliance reports as needed

### 5. Maintenance
- Schedule regular security maintenance
- Clean up expired sessions
- Rotate audit logs
- Update security configurations as needed

## Integration with MCP Tools

The security module seamlessly integrates with MCP tools through decorators:

```python
@mcp.tool()
@secure_mcp_tool(
    permission=Permission.CAMPAIGN_CREATE,
    operation_type=OperationType.CAMPAIGN_WRITE,
    validate_params={"name": CampaignParameters},
    resource_type=ResourceType.CAMPAIGN,
    audit_event=SecurityEventType.CAMPAIGN_CREATED
)
async def create_campaign(name: str, system: str, **kwargs):
    # Implementation
    pass
```

This ensures:
- Permission checking before execution
- Rate limiting enforcement
- Input validation and sanitization
- Audit logging of operations
- Consistent security across all tools

## Testing

The module includes comprehensive tests in `tests/test_security.py`:

```bash
# Run security tests
pytest tests/test_security.py -v

# Run specific test classes
pytest tests/test_security.py::TestInputValidator -v
pytest tests/test_security.py::TestAccessControl -v
pytest tests/test_security.py::TestRateLimiter -v
pytest tests/test_security.py::TestSecurityAudit -v
pytest tests/test_security.py::TestSecurityManager -v
```

## Security Incident Response

If a security incident is detected:

1. **Immediate Response:**
   - Review audit logs for the incident
   - Check recent security events
   - Identify affected users/resources

2. **Investigation:**
   - Generate security report for the period
   - Search for related security events
   - Analyze suspicious activity patterns

3. **Mitigation:**
   - Block suspicious IP addresses
   - Revoke compromised sessions
   - Adjust rate limits if needed
   - Update security rules

4. **Recovery:**
   - Reset affected user credentials
   - Review and update permissions
   - Export compliance report
   - Document incident and response

## Compliance

The security module supports compliance requirements through:

- Comprehensive audit trail
- Security event logging
- Compliance report generation
- Data retention policies
- Access control and authorization
- Input validation and sanitization

Export compliance reports using:
```python
audit.export_compliance_report(
    output_path=Path("compliance_report.json"),
    period_start=start_date,
    period_end=end_date
)
```

## Performance Considerations

The security module is designed for minimal performance impact:

- Efficient regex compilation and caching
- Thread-safe operations with minimal locking
- Configurable in-memory event limits
- Automatic cleanup of old data
- Asynchronous security operations where possible

## Future Enhancements

Potential future improvements:

1. **Advanced Authentication:**
   - OAuth2/JWT integration
   - Multi-factor authentication
   - API key management

2. **Enhanced Monitoring:**
   - Real-time security dashboards
   - Alert system for critical events
   - Machine learning for anomaly detection

3. **Extended Protection:**
   - DDoS protection
   - Advanced threat detection
   - Security scanning tools

4. **Compliance Features:**
   - GDPR compliance tools
   - Data encryption at rest
   - Privacy controls

## Support

For security-related issues or questions:

1. Check the security logs in `data/security/audit/`
2. Review the security report using `security_status()` tool
3. Consult the test suite for usage examples
4. Enable debug logging for detailed security information

## License

This security module is part of the TTRPG Assistant MCP Server and follows the same license terms.