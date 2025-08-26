# Security Integration Guide

## Overview
The security module has been successfully integrated into the TTRPG Assistant MCP Server. This document describes the integration and how to configure security features.

## Integration Points

### 1. Security Initialization
Security is initialized early in the `main()` function with configurable options:

```python
security_config = SecurityConfig(
    enable_authentication=False,  # Can be enabled via settings
    enable_rate_limiting=True,     # Rate limiting enabled by default
    enable_audit=True,             # Security audit logging enabled
    enable_input_validation=True,  # Input validation enabled
    session_timeout_minutes=60,
    audit_retention_days=90,
)
security_manager = initialize_security(security_config)
```

### 2. Secured MCP Tools
All MCP tools are now protected with the `@secure_mcp_tool` decorator that provides:
- **Permission checking** - Ensures users have required permissions
- **Rate limiting** - Prevents abuse and DoS attacks  
- **Input validation** - Validates and sanitizes parameters
- **Audit logging** - Tracks all security-relevant events

Example:
```python
@mcp.tool()
@secure_mcp_tool(
    permission=Permission.READ,
    operation_type=OperationType.SEARCH,
    validate_params={"query": SearchParameters},
    resource_type=ResourceType.CONTENT,
    audit_event=SecurityEventType.DATA_ACCESS,
)
async def search(...):
```

### 3. Security Tools
Two new tools have been added for security management:

#### `security_status`
- Returns current security status and statistics
- Shows active sessions, rate limits, and audit summary
- Requires READ permission

#### `security_maintenance`
- Performs security maintenance tasks
- Cleans up expired sessions and old logs
- Requires ADMIN permission

### 4. Cleanup and Shutdown
Security cleanup is integrated into the shutdown process:
- Automatic cleanup on normal shutdown
- Cleanup on exceptions and interrupts
- Final maintenance tasks performed

## Configuration

Security features can be configured via settings or environment variables:

```python
# In settings.py or environment
security_enable_authentication = False  # Enable/disable authentication
security_enable_rate_limiting = True    # Enable/disable rate limiting
security_enable_audit = True           # Enable/disable audit logging
security_enable_input_validation = True # Enable/disable input validation
security_session_timeout_minutes = 60   # Session timeout in minutes
security_audit_retention_days = 90      # Audit log retention period
```

## Security Features by Tool

| Tool | Permission | Rate Limit | Validation | Audit Event |
|------|------------|------------|------------|-------------|
| search | READ | SEARCH | SearchParameters | DATA_ACCESS |
| add_source | WRITE | CREATE | FilePathParameters | CAMPAIGN_CREATED |
| list_sources | READ | READ | - | DATA_ACCESS |
| search_analytics | READ | READ | - | DATA_ACCESS |
| clear_search_cache | WRITE | UPDATE | - | CONFIG_CHANGED |
| update_search_indices | ADMIN | UPDATE | - | CONFIG_CHANGED |
| server_info | READ | READ | - | DATA_ACCESS |
| create_personality_profile | WRITE | CREATE | - | CAMPAIGN_CREATED |
| list_personality_profiles | READ | READ | - | DATA_ACCESS |
| set_active_personality | WRITE | UPDATE | - | CONFIG_CHANGED |
| apply_personality | READ | READ | - | DATA_ACCESS |
| security_status | READ | READ | - | DATA_ACCESS |
| security_maintenance | ADMIN | UPDATE | - | CONFIG_CHANGED |

## Backward Compatibility

The security integration maintains full backward compatibility:
- Security features can be enabled/disabled via configuration
- Default configuration has authentication disabled
- Existing functionality remains unchanged
- No breaking changes to the API

## Testing

Run the security syntax check:
```bash
python test_security_syntax.py
```

This verifies:
- Correct imports and syntax
- Security decorator application
- Security manager initialization
- Security cleanup integration
- Security tool registration

## Future Enhancements

Potential future security enhancements:
1. OAuth2/JWT authentication support
2. Role-based access control (RBAC) 
3. API key management
4. Security dashboards and monitoring
5. Integration with external security services
6. Advanced threat detection
7. Security compliance reporting