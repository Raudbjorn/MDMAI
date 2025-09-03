# TTRPG Assistant - Secure Credential Management Guide

## Overview

This guide covers the secure credential management system implemented in Task 25.1, which provides enterprise-grade security for storing and managing AI provider API keys with AES-256 encryption, user isolation, and automated key rotation.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Security Features](#security-features)
3. [Quick Start Guide](#quick-start-guide)
4. [Configuration](#configuration)
5. [API Reference](#api-reference)
6. [Security Best Practices](#security-best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Topics](#advanced-topics)

## Architecture Overview

The credential management system consists of five main components:

```
┌─────────────────────────────────────┐
│        Credential Manager           │
│     (Main Interface)                │
├─────────────────────────────────────┤
│  Encryption  │  Storage │ Validator │
│   Service    │ Backends │  Service  │
├─────────────────────────────────────┤
│          Rotation Service           │
│       (Automated & Manual)         │
└─────────────────────────────────────┘
```

### Components

1. **Credential Manager** - Main interface for all credential operations
2. **Encryption Service** - AES-256-GCM encryption with PBKDF2 key derivation
3. **Storage Backends** - JSON file and ChromaDB storage options
4. **Validator Service** - Pre-storage API key validation
5. **Rotation Service** - Automated and manual key rotation

## Security Features

### ✅ Encryption
- **AES-256-GCM**: Authenticated encryption preventing tampering
- **PBKDF2**: 600,000+ iterations (OWASP 2023 compliant)
- **User Isolation**: Each user has unique salts and keys
- **Forward Secrecy**: Compromised keys don't affect past encryptions

### ✅ Storage Security
- **Atomic Operations**: Write-to-temp-then-move for integrity
- **File Permissions**: 0600 for files, 0700 for directories
- **Backup & Recovery**: Automated backups with retention policies
- **Audit Logging**: Complete trail of all operations

### ✅ Validation & Rotation
- **Pre-storage Validation**: Prevents storing invalid keys
- **Automated Rotation**: Age-based and usage-based policies
- **Manual Rotation**: On-demand with rollback capabilities
- **Emergency Response**: Immediate rotation on compromise detection

## Quick Start Guide

### 1. Basic Setup

```python
from src.security.credential_manager import SecureCredentialManager, CredentialManagerConfig

# Initialize with default configuration
config = CredentialManagerConfig(
    master_password="your_secure_master_password",
    storage_backend="json",  # or "chromadb"
    enable_validation=True,
    enable_rotation=True
)

manager = SecureCredentialManager(config)
await manager.initialize()
```

### 2. Store a Credential

```python
from src.ai_providers.models import ProviderType

# Store an Anthropic API key
result = await manager.store_credential(
    api_key="sk-ant-api03-your-key-here",
    provider_type=ProviderType.ANTHROPIC,
    user_id="user123",
    metadata={"purpose": "main_account", "project": "ttrpg_assistant"}
)

if result.is_success():
    credential_id = result.unwrap()
    print(f"Stored credential: {credential_id}")
```

### 3. Retrieve and Use Credentials

```python
# Retrieve for use
api_key_result = await manager.retrieve_credential(credential_id, "user123")
if api_key_result.is_success():
    api_key = api_key_result.unwrap()
    
    # Use with AI provider
    provider_config = await manager.get_provider_config(
        ProviderType.ANTHROPIC, 
        "user123"
    )
```

### 4. List and Search Credentials

```python
# List all credentials for a user
credentials = await manager.list_user_credentials("user123")

# Search by metadata
search_result = await manager.search_credentials(
    user_id="user123",
    provider_type=ProviderType.ANTHROPIC,
    metadata_filters={"project": "ttrpg_assistant"}
)
```

## Configuration

### CredentialManagerConfig Options

```python
@dataclass
class CredentialManagerConfig:
    # Master password for encryption
    master_password: str = ""
    
    # Storage backend ("json" or "chromadb")
    storage_backend: str = "json"
    
    # Enable features
    enable_validation: bool = True
    enable_rotation: bool = True
    enable_backup: bool = True
    
    # Paths
    storage_path: str = "~/.ttrpg_assistant/credentials"
    backup_path: str = "~/.ttrpg_assistant/backups"
    
    # Security settings
    pbkdf2_iterations: int = 600000
    backup_retention_days: int = 30
    validation_timeout_seconds: int = 10
    
    # Rotation policies
    default_rotation_days: int = 90
    high_usage_rotation_days: int = 30
    
    # Performance
    max_concurrent_operations: int = 10
    cache_size: int = 100
```

### Environment Variables

```bash
# Optional environment variables
export TTRPG_MASTER_PASSWORD="your_master_password"
export TTRPG_CREDENTIAL_PATH="/custom/path/to/credentials"
export TTRPG_STORAGE_BACKEND="chromadb"
export TTRPG_ENABLE_ROTATION="true"
```

## API Reference

### Core Methods

#### `store_credential(api_key, provider_type, user_id, metadata=None)`
Store an encrypted credential with validation.

**Parameters:**
- `api_key` (str): The API key to store
- `provider_type` (ProviderType): AI provider type
- `user_id` (str): User identifier
- `metadata` (dict, optional): Additional metadata

**Returns:** `Result[str, str]` - Success with credential ID or failure with error

#### `retrieve_credential(credential_id, user_id)`
Retrieve and decrypt a credential.

**Parameters:**
- `credential_id` (str): The credential identifier
- `user_id` (str): User identifier for access control

**Returns:** `Result[str, str]` - Success with decrypted API key or failure

#### `delete_credential(credential_id, user_id)`
Securely delete a credential.

**Parameters:**
- `credential_id` (str): The credential identifier
- `user_id` (str): User identifier for access control

**Returns:** `Result[bool, str]` - Success status or error

#### `rotate_credential(credential_id, new_api_key, user_id)`
Rotate a credential to a new API key.

**Parameters:**
- `credential_id` (str): Existing credential ID
- `new_api_key` (str): New API key
- `user_id` (str): User identifier

**Returns:** `Result[str, str]` - Success with new credential ID or error

### Provider Integration Methods

#### `get_provider_config(provider_type, user_id)`
Get provider configuration for AI system integration.

**Parameters:**
- `provider_type` (ProviderType): AI provider type
- `user_id` (str): User identifier

**Returns:** `Result[ProviderConfig, str]` - Provider configuration or error

#### `get_all_provider_configs(user_id)`
Get configurations for all providers a user has credentials for.

**Parameters:**
- `user_id` (str): User identifier

**Returns:** `Result[List[ProviderConfig], str]` - List of configurations

### Advanced Methods

#### `backup_credentials(user_id=None)`
Create encrypted backup of credentials.

#### `restore_credentials(backup_path, user_id=None)`
Restore credentials from encrypted backup.

#### `audit_credentials(user_id=None)`
Generate security audit report.

#### `health_check()`
Perform comprehensive system health check.

## Security Best Practices

### 1. Master Password Security

```python
# ✅ Good: Strong, unique password
config.master_password = "Tr0ub4dor&3-ComplexMasterKey2024!"

# ❌ Bad: Weak or reused password
config.master_password = "password123"
```

### 2. User ID Guidelines

```python
# ✅ Good: Use stable, non-PII identifiers
user_id = "user_uuid_12345678-1234-5678-9abc-123456789def"

# ❌ Bad: Use email or personal info
user_id = "john.doe@example.com"  # PII in logs
```

### 3. Metadata Security

```python
# ✅ Good: Non-sensitive metadata
metadata = {
    "project": "ttrpg_assistant",
    "purpose": "main_account",
    "created_date": "2024-01-15"
}

# ❌ Bad: Sensitive data in metadata
metadata = {
    "real_name": "John Doe",  # PII
    "credit_card": "1234-5678"  # Sensitive
}
```

### 4. Error Handling

```python
# ✅ Good: Proper error handling
result = await manager.store_credential(api_key, provider, user_id)
if result.is_failure():
    logger.error("Failed to store credential", error=result.failure())
    return None
    
credential_id = result.unwrap()

# ❌ Bad: Ignoring errors
credential_id = await manager.store_credential(api_key, provider, user_id)
```

### 5. Cleanup and Lifecycle

```python
# ✅ Good: Proper cleanup
try:
    await manager.initialize()
    # ... use manager
finally:
    await manager.cleanup()

# Or use context manager
async with SecureCredentialManager(config) as manager:
    # ... use manager
    pass  # Automatic cleanup
```

## Troubleshooting

### Common Issues

#### 1. "Master password not set" Error

```python
# Solution: Set master password before initialization
config.master_password = "your_secure_password"
await manager.initialize()
```

#### 2. "Permission denied" Error

```bash
# Solution: Check file permissions
chmod 700 ~/.ttrpg_assistant
chmod 600 ~/.ttrpg_assistant/credentials/*
```

#### 3. "Credential not found" Error

```python
# Solution: Verify user_id matches storage
credentials = await manager.list_user_credentials(user_id)
print("Available credentials:", [c.id for c in credentials])
```

#### 4. "Validation failed" Error

```python
# Solution: Test API key manually first
config.enable_validation = False  # Temporarily disable
# Or check network connectivity and provider status
```

### Debugging

#### Enable Debug Logging

```python
import structlog
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = structlog.get_logger()
```

#### Health Check

```python
# Run comprehensive health check
health_result = await manager.health_check()
if health_result.is_success():
    health_data = health_result.unwrap()
    print("System status:", health_data["status"])
    print("Issues:", health_data.get("issues", []))
```

#### Audit Report

```python
# Generate security audit
audit_result = await manager.audit_credentials(user_id)
if audit_result.is_success():
    audit_data = audit_result.unwrap()
    print("Security status:", audit_data["overall_status"])
    print("Recommendations:", audit_data["recommendations"])
```

## Advanced Topics

### Custom Storage Backend

```python
from src.security.credential_storage import CredentialStorageBackend

class CustomStorageBackend(CredentialStorageBackend):
    async def store_credential(self, encrypted_cred, metadata):
        # Custom implementation
        pass
        
    async def retrieve_credential(self, credential_id):
        # Custom implementation
        pass

# Use custom backend
storage_manager = CredentialStorageManager(
    backend=CustomStorageBackend(),
    config=storage_config
)
```

### Custom Validation Rules

```python
from src.security.credential_validator import ValidationRule

class CustomProviderRule(ValidationRule):
    async def validate(self, api_key: str) -> ValidationResult:
        # Custom validation logic
        if not api_key.startswith("custom-"):
            return ValidationResult(
                is_valid=False,
                error="Invalid custom provider key format"
            )
        
        # Test API key
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.customprovider.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            
            return ValidationResult(
                is_valid=response.status_code == 200,
                error=None if response.status_code == 200 else "API key invalid"
            )
```

### Performance Optimization

#### Batch Operations

```python
# Batch credential storage
credentials_to_store = [
    ("api_key_1", ProviderType.ANTHROPIC, "user1"),
    ("api_key_2", ProviderType.OPENAI, "user2"),
    # ... more credentials
]

results = await manager.batch_store_credentials(credentials_to_store)
```

#### Caching

```python
# Enable aggressive caching for read-heavy workloads
config = CredentialManagerConfig(
    cache_size=1000,  # Increase cache size
    cache_ttl_seconds=3600,  # Cache for 1 hour
    enable_memory_cache=True
)
```

### Integration with Existing Systems

#### AI Provider Integration

```python
from src.ai_providers.manager import AIProviderManager

# Get all provider configs for a user
provider_configs = await credential_manager.get_all_provider_configs(user_id)

# Initialize AI provider manager
ai_manager = AIProviderManager()
await ai_manager.initialize(provider_configs.unwrap())

# Use AI providers as normal
response = await ai_manager.generate_text(
    provider_type=ProviderType.ANTHROPIC,
    prompt="Generate a character backstory..."
)
```

#### MCP Tool Integration

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("TTRPG")

@mcp.tool()
async def manage_api_keys(
    action: str,
    provider: str,
    api_key: str = None,
    user_id: str = None
) -> dict:
    """MCP tool for managing API credentials."""
    
    if action == "store":
        result = await credential_manager.store_credential(
            api_key=api_key,
            provider_type=ProviderType(provider),
            user_id=user_id
        )
        
        return {
            "success": result.is_success(),
            "credential_id": result.unwrap() if result.is_success() else None,
            "error": result.failure() if result.is_failure() else None
        }
    
    elif action == "list":
        credentials = await credential_manager.list_user_credentials(user_id)
        return {
            "success": True,
            "credentials": [
                {
                    "id": cred.id,
                    "provider": cred.provider_type.value,
                    "created_at": cred.created_at.isoformat()
                }
                for cred in credentials.unwrap()
            ]
        }
```

### Monitoring and Alerting

```python
# Set up monitoring
from src.security.credential_manager import SecurityEventType

async def security_event_handler(event_type: SecurityEventType, details: dict):
    """Handle security events for monitoring."""
    
    if event_type == SecurityEventType.CREDENTIAL_COMPROMISED:
        # Send alert
        await send_security_alert(
            f"Credential compromised: {details['credential_id']}"
        )
        
        # Trigger emergency rotation
        await credential_manager.emergency_rotate_credential(
            details['credential_id'],
            details['user_id']
        )
    
    elif event_type == SecurityEventType.SUSPICIOUS_ACCESS:
        # Log for investigation
        logger.warning("Suspicious credential access", extra=details)

# Register event handler
credential_manager.register_security_event_handler(security_event_handler)
```

---

## Support

For issues or questions:
- **Documentation**: See project README and inline code documentation
- **Issues**: Open a GitHub issue with debug logs and configuration details
- **Security Issues**: Report privately following security.md guidelines

## Contributing

When contributing to credential management:
1. Follow existing security patterns and error handling
2. Add comprehensive tests including security property tests
3. Update documentation for any API changes
4. Consider backwards compatibility for existing stored credentials

---

*This credential management system provides enterprise-grade security suitable for production deployments while maintaining ease of use for development environments.*