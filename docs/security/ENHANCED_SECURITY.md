# Enhanced Security and Authentication System

## Overview

Phase 16 introduces comprehensive security enhancements for the TTRPG Assistant, building upon the existing security infrastructure from Phase 12. This implementation provides enterprise-grade authentication, authorization, and security monitoring specifically designed for web UI integration.

## Features

### 1. Authentication System

#### OAuth2 Integration
- **Supported Providers**: Google, GitHub, Microsoft, Discord
- **PKCE Support**: Enhanced security for public clients
- **State Management**: CSRF protection with secure state tokens
- **Automatic User Creation**: Seamless onboarding from OAuth providers

```python
# Example: OAuth2 login
from src.security.enhanced_security_manager import EnhancedSecurityManager
from src.security.auth_providers import OAuthProvider

manager = EnhancedSecurityManager()
login_url = manager.get_oauth_login_url(OAuthProvider.GOOGLE)
```

#### JWT Token System
- **RS256 Algorithm**: Asymmetric signing for enhanced security
- **Token Pairs**: Short-lived access tokens with refresh tokens
- **Token Revocation**: Blacklist support for immediate invalidation
- **Custom Claims**: Extensible token payload

```python
# Example: JWT token creation
token_pair = manager.jwt_manager.create_token_pair(
    subject=user.user_id,
    claims={"roles": ["admin"], "permissions": ["read", "write"]}
)
```

#### API Key Authentication
- **Service Accounts**: Dedicated authentication for services
- **Permission Scoping**: Fine-grained access control
- **Rate Limiting**: Per-key rate limits
- **IP Restrictions**: Whitelist IP addresses per key

### 2. Session Management

#### Redis-Backed Sessions
- **Distributed Storage**: Scalable session storage with Redis
- **Session Clustering**: Support for multi-node deployments
- **Automatic Expiration**: TTL-based session cleanup
- **Device Tracking**: Monitor sessions across devices

```python
# Example: Session creation
session, tokens = await manager.create_web_session(
    user=user,
    ip_address="192.168.1.1",
    user_agent="Mozilla/5.0...",
    device_info={"type": "desktop", "browser": "Chrome"}
)
```

#### Session Security
- **CSRF Protection**: Token-based CSRF prevention
- **Session Hijacking Detection**: IP rotation monitoring
- **Idle Timeout**: Automatic session expiration
- **Concurrent Session Limits**: Maximum sessions per user

### 3. Authorization Framework

#### Role-Based Access Control (RBAC)
- **Predefined Roles**: Admin, Moderator, User, Guest, Service
- **Custom Permissions**: Extensible permission system
- **Resource-Level Permissions**: Per-resource access control
- **Permission Inheritance**: Hierarchical permission model

#### Enhanced Rate Limiting
- **Multi-Level Limits**: Per-minute, per-hour, per-day
- **Operation-Specific**: Different limits for different operations
- **User-Based**: Personalized rate limits
- **Adaptive Throttling**: Dynamic rate adjustment

### 4. Process Isolation

#### Sandboxed Execution
- **Security Policies**: Strict, Moderate, Relaxed, Custom
- **Resource Limits**: CPU, memory, disk, network
- **Filesystem Restrictions**: Path whitelisting/blacklisting
- **Network Isolation**: Control network access

```python
# Example: Sandboxed Python execution
result = await manager.execute_python_sandboxed(
    code="print('Hello, World!')",
    policy=SandboxPolicy.STRICT,
    timeout=30
)
```

#### Supported Isolation Methods
- **Linux Namespaces**: Process isolation
- **Firejail**: Enhanced sandboxing (Linux)
- **Docker Containers**: Full isolation
- **Resource Limits**: System resource constraints

### 5. Security Monitoring

#### Real-Time Threat Detection
- **Attack Pattern Recognition**: SQL injection, XSS, path traversal
- **Behavioral Analysis**: Anomaly detection
- **Automated Response**: Auto-blocking threats
- **Alert System**: Real-time security alerts

#### Security Dashboard
- **Threat Overview**: Current threat level assessment
- **Active Alerts**: Unresolved security incidents
- **Blocked Entities**: IPs and users currently blocked
- **Metrics Visualization**: Security metrics over time

```python
# Example: Security dashboard
dashboard = manager.get_security_dashboard()
# Returns:
# {
#     "threat_summary": {...},
#     "active_alerts": [...],
#     "blocked_ips": [...],
#     "metrics": [...]
# }
```

## Configuration

### Environment Variables

```bash
# OAuth2 Configuration
OAUTH_GOOGLE_CLIENT_ID=your-google-client-id
OAUTH_GOOGLE_CLIENT_SECRET=your-google-secret
OAUTH_GITHUB_CLIENT_ID=your-github-client-id
OAUTH_GITHUB_CLIENT_SECRET=your-github-secret

# JWT Configuration
JWT_SECRET_KEY=your-secret-key  # For HMAC algorithms
JWT_ACCESS_TOKEN_TTL=900  # 15 minutes
JWT_REFRESH_TOKEN_TTL=2592000  # 30 days

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
SESSION_TTL_HOURS=24
MAX_SESSIONS_PER_USER=10

# Security Settings
ENABLE_SANDBOXING=true
SANDBOX_POLICY=strict
ENABLE_MONITORING=true
AUTO_BLOCK_THREATS=true
```

### Python Configuration

```python
from src.security.enhanced_security_manager import (
    EnhancedSecurityManager,
    EnhancedSecurityConfig
)

config = EnhancedSecurityConfig(
    # OAuth2 providers
    oauth_providers={
        "google": {
            "client_id": "...",
            "client_secret": "..."
        },
        "github": {
            "client_id": "...",
            "client_secret": "..."
        }
    },
    
    # JWT settings
    jwt_algorithm="RS256",
    jwt_access_ttl=900,
    jwt_refresh_ttl=2592000,
    
    # Redis settings
    redis_url="redis://localhost:6379/0",
    session_ttl_hours=24,
    max_sessions_per_user=10,
    
    # Security settings
    enable_sandboxing=True,
    sandbox_policy=SandboxPolicy.STRICT,
    enable_monitoring=True,
    auto_block_threats=True
)

manager = EnhancedSecurityManager(config)
```

## Web Integration

### FastAPI Integration

```python
from fastapi import FastAPI
from src.security.web_integration import (
    create_security_routes,
    SecurityMiddleware,
    SecurityDependencies
)

app = FastAPI()

# Initialize security manager
security_manager = EnhancedSecurityManager()

# Add security middleware
app.add_middleware(SecurityMiddleware(security_manager))

# Create security routes
create_security_routes(app, security_manager)

# Use security dependencies
deps = SecurityDependencies(security_manager)

@app.get("/protected")
async def protected_route(
    user = Depends(deps.get_current_user)
):
    return {"message": f"Hello, {user.username}"}

@app.get("/admin")
async def admin_route(
    user = Depends(deps.require_role(SecurityRole.ADMIN))
):
    return {"message": "Admin access granted"}
```

### Authentication Flow

1. **Local Authentication**:
   ```
   POST /auth/login
   {
     "username": "user@example.com",
     "password": "secure_password"
   }
   ```

2. **OAuth2 Authentication**:
   ```
   GET /auth/oauth/google
   -> Redirects to Google OAuth
   -> Callback to /auth/callback/google
   -> Returns JWT tokens
   ```

3. **Token Refresh**:
   ```
   POST /auth/refresh
   {
     "refresh_token": "..."
   }
   ```

## Security Best Practices

### 1. Token Management
- Store tokens securely (HttpOnly cookies for web)
- Implement token rotation
- Use short-lived access tokens
- Invalidate tokens on logout

### 2. Session Security
- Use HTTPS in production
- Implement CSRF protection
- Monitor for session anomalies
- Enforce session limits

### 3. API Security
- Rotate API keys regularly
- Use IP whitelisting
- Implement rate limiting
- Monitor API usage

### 4. Process Isolation
- Use strictest policy possible
- Validate all user input
- Monitor resource usage
- Log security events

### 5. Monitoring
- Review security alerts regularly
- Investigate anomalies promptly
- Update security rules based on threats
- Maintain audit logs

## Testing

### Run Security Tests

```bash
# Run all security tests
pytest tests/test_enhanced_security.py -v

# Test specific components
pytest tests/test_enhanced_security.py::TestJWTManager -v
pytest tests/test_enhanced_security.py::TestSessionStore -v
pytest tests/test_enhanced_security.py::TestProcessSandbox -v
pytest tests/test_enhanced_security.py::TestSecurityMonitor -v
```

### Security Validation

```python
# Validate security configuration
from src.security.enhanced_security_manager import EnhancedSecurityManager

manager = EnhancedSecurityManager()

# Test OAuth2
assert manager.get_oauth_login_url(OAuthProvider.GOOGLE) is not None

# Test JWT
token_pair = manager.jwt_manager.create_token_pair("test_user")
assert token_pair.access_token is not None

# Test sandboxing
result = await manager.execute_sandboxed(["echo", "test"])
assert result["success"] is True

# Test monitoring
dashboard = manager.get_security_dashboard()
assert "threat_summary" in dashboard
```

## Performance Considerations

### Session Store
- Redis connection pooling: 50 connections default
- Local cache with 5-minute TTL
- Automatic cleanup of expired sessions
- Batch operations for bulk updates

### JWT Performance
- Token caching for validation
- Async token operations
- Minimal database lookups
- Efficient blacklist implementation

### Monitoring Impact
- Event buffering (10,000 events)
- 5-second processing intervals
- Automatic data cleanup
- Metrics aggregation

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**:
   - Check Redis server is running
   - Verify connection URL
   - Falls back to in-memory storage

2. **OAuth2 Not Working**:
   - Verify client credentials
   - Check redirect URLs
   - Ensure provider is configured

3. **JWT Validation Errors**:
   - Check token expiration
   - Verify signing keys
   - Ensure algorithm matches

4. **Sandbox Failures**:
   - Check system dependencies (firejail/docker)
   - Verify resource limits
   - Review security policies

5. **High Threat Alerts**:
   - Review monitoring rules
   - Check for false positives
   - Adjust thresholds if needed

## Migration Guide

### From Phase 12 Security

1. **Update Configuration**:
   ```python
   # Old configuration
   from src.security.security_manager import SecurityManager
   
   # New configuration
   from src.security.enhanced_security_manager import EnhancedSecurityManager
   ```

2. **Update Authentication**:
   ```python
   # Old: Basic authentication
   session = manager.authenticate(username, password)
   
   # New: Enhanced authentication with JWT
   user, tokens = await manager.create_web_session(user, ip_address)
   ```

3. **Update Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Security Compliance

### Standards Supported
- OAuth 2.0 (RFC 6749)
- OpenID Connect 1.0
- JWT (RFC 7519)
- PKCE (RFC 7636)

### Security Features
- CSRF Protection
- XSS Prevention
- SQL Injection Prevention
- Path Traversal Protection
- Rate Limiting
- Session Management
- Audit Logging

## Future Enhancements

- [ ] WebAuthn/FIDO2 support
- [ ] Hardware token authentication
- [ ] Advanced threat intelligence
- [ ] Machine learning-based anomaly detection
- [ ] Distributed rate limiting
- [ ] Zero-trust architecture
- [ ] Compliance reporting (GDPR, SOC2)