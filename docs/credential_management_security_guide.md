# TTRPG Assistant - Secure Credential Management System

## Security Architecture and Best Practices Guide

### Overview

The TTRPG Assistant MCP Server includes a comprehensive secure credential management system designed to protect AI provider API keys using enterprise-grade security measures. This system implements defense-in-depth principles with multiple layers of security.

## 🔒 Security Components

### 1. Encryption Layer (AES-256-GCM)

**Implementation**: `src/security/credential_encryption.py`

- **Algorithm**: AES-256 in GCM mode for authenticated encryption
- **Key Derivation**: PBKDF2-HMAC-SHA256 with 600,000+ iterations (OWASP recommended)
- **Salt Management**: User-specific salts (256-bit) for key isolation
- **Memory Safety**: Secure memory wiping with multiple overwrite passes

**Security Properties**:
- ✅ Semantic security (same plaintext produces different ciphertexts)
- ✅ Authentication (tamper detection via GCM tag)
- ✅ Forward secrecy (compromised keys don't affect past encryptions)
- ✅ User isolation (users cannot decrypt each other's credentials)

### 2. Storage Layer

**Implementation**: `src/security/credential_storage.py`

**JSON Backend**:
- Atomic file operations (write-to-temp-then-move)
- Restrictive file permissions (0600 for files, 0700 for directories)
- Automatic backup rotation with configurable retention
- Integrity verification on restore operations

**ChromaDB Backend**:
- Encrypted metadata storage
- Transaction safety
- Built-in data validation
- Vector database benefits for future search capabilities

### 3. Validation Layer

**Implementation**: `src/security/credential_validator.py`

**Pre-storage Validation**:
- Format validation (provider-specific patterns)
- Functional testing (API endpoint connectivity)
- Account status verification
- Rate limit detection

**Supported Providers**:
- ✅ Anthropic (Claude) - `sk-ant-` prefix validation
- ✅ OpenAI (GPT) - `sk-` prefix validation  
- ✅ Google AI - Standard API key format validation

### 4. Rotation Layer

**Implementation**: `src/security/credential_rotation.py`

**Automated Rotation**:
- Age-based rotation (configurable maximum age)
- Usage-based rotation (configurable usage limits)
- Security event triggers
- Scheduled maintenance windows

**Manual Rotation**:
- On-demand rotation with audit trails
- Rollback capabilities within time windows
- Validation of new keys before activation

## 🛡️ Security Best Practices

### Master Password Security

```python
# ✅ GOOD: Strong master password
master_password = "MySecure_TTRPG_Master_Password_2024!@#"

# ❌ BAD: Weak master password  
master_password = "password123"
```

**Requirements**:
- Minimum 12 characters (enforced)
- Mix of uppercase, lowercase, numbers, symbols
- Not based on dictionary words or personal information
- Stored securely (consider using system keychain)

### Configuration Security

```python
# ✅ GOOD: Secure configuration
config = CredentialManagerConfig(
    master_password=os.environ.get("TTRPG_MASTER_PASSWORD"),  # From environment
    storage_path="~/.ttrpg_assistant/credentials",            # User-specific path
    encryption_config=EncryptionConfig(
        pbkdf2_iterations=600_000,  # Strong key derivation
        memory_wipe_passes=3        # Multiple wipe passes
    ),
    enable_validation=True,         # Always validate keys
    enable_rotation=True           # Enable automatic rotation
)

# ❌ BAD: Insecure configuration
config = CredentialManagerConfig(
    master_password="hardcoded_password",  # Never hardcode passwords
    storage_path="/tmp/credentials",       # Insecure location
    enable_validation=False               # Skip validation
)
```

### Storage Location Security

**Recommended Paths**:
- Linux/macOS: `~/.ttrpg_assistant/credentials/`
- Windows: `%USERPROFILE%\.ttrpg_assistant\credentials\`

**Security Measures**:
- User-only access (no group/world permissions)
- Outside of application directory
- Regular backup to secure locations
- Not in cloud-synced directories by default

### Key Rotation Policy

```python
# ✅ GOOD: Comprehensive rotation policy
rotation_policy = RotationPolicy(
    max_age_days=90,                    # Rotate quarterly
    rotation_warning_days=7,            # Give advance warning
    enable_scheduled_rotation=True,     # Enable automation
    rotation_schedule_hour=2,           # Off-peak hours
    auto_rotate_on_validation_failure=True,
    keep_previous_versions=3            # Allow rollback
)
```

### Monitoring and Auditing

```python
# Enable comprehensive logging
config.storage_config.enable_audit_log = True
config.storage_config.audit_log_path = "~/.ttrpg_assistant/audit.log"

# Regular health checks
async def security_maintenance():
    health = await credential_manager.health_check()
    if not health["overall_healthy"]:
        # Alert administrator
        await send_security_alert(health)
```

## 🔍 Threat Model

### Threats Addressed

1. **Credential Theft**
   - **Mitigation**: AES-256 encryption, user-specific salts
   - **Detection**: Access logging, validation failures

2. **Memory Dumps**
   - **Mitigation**: Secure memory wiping, minimal exposure time
   - **Prevention**: Immediate cleanup after use

3. **File System Access**
   - **Mitigation**: Restrictive permissions, encrypted storage
   - **Monitoring**: File integrity checks

4. **Replay Attacks**
   - **Mitigation**: Time-based validation, rotation policies
   - **Detection**: Usage pattern analysis

5. **Insider Threats**
   - **Mitigation**: User isolation, audit trails
   - **Detection**: Access monitoring, behavior analysis

### Residual Risks

1. **Master Password Compromise**
   - **Impact**: All credentials for the system could be compromised
   - **Mitigation**: Strong password policy, rotation, secure storage

2. **Memory Attacks**
   - **Impact**: Limited exposure during credential use
   - **Mitigation**: Minimal exposure time, secure cleanup

3. **Side-Channel Attacks**
   - **Impact**: Timing attacks on validation
   - **Mitigation**: Constant-time operations where possible

## 📋 Implementation Checklist

### Initial Setup
- [ ] Generate strong master password
- [ ] Configure secure storage location
- [ ] Set restrictive file permissions
- [ ] Enable audit logging
- [ ] Test backup/restore procedures

### Operational Security
- [ ] Regular credential validation
- [ ] Monitor rotation schedules
- [ ] Review audit logs
- [ ] Update rotation policies
- [ ] Test incident response procedures

### Maintenance
- [ ] Backup encryption keys
- [ ] Update key derivation parameters
- [ ] Clean up expired credentials
- [ ] Review access patterns
- [ ] Update threat model

## 🚨 Incident Response

### Suspected Credential Compromise

1. **Immediate Actions**:
   ```python
   # Immediately rotate affected credentials
   rotation_result = await credential_manager.rotate_credential(
       credential_id=compromised_id,
       user_id=user_id,
       reason=RotationReason.COMPROMISE_SUSPECTED
   )
   
   # Revoke old credentials at provider
   await provider_client.revoke_api_key(old_key)
   ```

2. **Investigation**:
   - Review audit logs for unauthorized access
   - Check validation failures and unusual patterns
   - Verify file system integrity
   - Examine network traffic logs

3. **Recovery**:
   - Generate new credentials with providers
   - Update stored credentials
   - Verify new credentials functionality
   - Document incident and lessons learned

### System Compromise

1. **Containment**:
   - Stop credential manager services
   - Backup current state
   - Isolate affected systems

2. **Eradication**:
   - Rotate all stored credentials
   - Change master password
   - Regenerate encryption salts
   - Clean affected storage

3. **Recovery**:
   - Restore from clean backups
   - Re-initialize with new secrets
   - Verify system integrity
   - Resume operations with monitoring

## 📊 Performance Considerations

### Encryption Performance

- **PBKDF2 Iterations**: 600,000 iterations ≈ 100-200ms on modern hardware
- **Memory Usage**: ~32MB peak during key derivation
- **Storage Overhead**: ~200 bytes per credential (metadata + encryption overhead)

### Scaling Guidelines

| Credentials | Recommended Backend | Expected Performance |
|-------------|-------------------|---------------------|
| < 100       | JSON             | < 50ms operations   |
| 100-1000    | JSON or ChromaDB | < 100ms operations  |
| > 1000      | ChromaDB         | < 200ms operations  |

### Optimization Tips

```python
# Cache frequently accessed credentials
config.encryption_config.cache_derived_keys = True

# Batch operations when possible
credentials = await manager.list_credentials(user_id)
for cred in credentials:
    await manager.validate_credential(cred.credential_id, user_id)

# Use background rotation for large deployments
config.rotation_policy.rotation_schedule_hour = 2  # Off-peak hours
```

## 🔧 Troubleshooting

### Common Issues

1. **Initialization Failures**
   ```
   Error: "Master key not set"
   Solution: Ensure master password is provided and valid
   ```

2. **Permission Errors**
   ```
   Error: "Permission denied accessing credential storage"
   Solution: Check file/directory permissions (should be 0600/0700)
   ```

3. **Validation Failures**
   ```
   Error: "API key validation failed"
   Solution: Check network connectivity and API key validity
   ```

4. **Rotation Failures**
   ```
   Error: "Rotation deadline passed"
   Solution: Provide new API key within rotation timeout window
   ```

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger("src.security").setLevel(logging.DEBUG)

# Health check for diagnostics
health = await credential_manager.health_check()
print(f"System Health: {health}")

# Component-specific diagnostics
encryption_health = credential_manager.encryption_service.health_check()
```

## 📚 References

- [OWASP Cryptographic Storage Cheat Sheet](https://owasp.org/www-project-cheat-sheets/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html)
- [NIST Special Publication 800-132](https://csrc.nist.gov/publications/detail/sp/800-132/final)
- [RFC 7517 - JSON Web Key (JWK)](https://tools.ietf.org/html/rfc7517)
- [CVE-2021-44228](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-44228) - Log4j vulnerability (logging security)

---

## Support

For security-related questions or to report vulnerabilities:

- **Documentation**: Check the examples in `examples/credential_management_examples.py`
- **Testing**: Run the test suite in `tests/test_secure_credential_management.py`
- **Issues**: Create issues in the project repository with security-related tags

**Remember**: This system is designed for local-first deployments. For production server environments, consider additional measures such as HSMs, key management services, and network security controls.