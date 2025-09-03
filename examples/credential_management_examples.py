"""
Examples of using the secure credential management system.

This module demonstrates how to integrate and use the secure credential
management system in your TTRPG Assistant MCP Server application.
"""

import asyncio
from datetime import datetime
from pathlib import Path

# Import the credential management system
from src.security.credential_manager import (
    SecureCredentialManager,
    CredentialManagerConfig
)
from src.security.credential_encryption import EncryptionConfig
from src.security.credential_storage import StorageConfig
from src.security.credential_rotation import RotationPolicy, RotationReason
from src.ai_providers.models import ProviderType


async def basic_usage_example():
    """
    Basic example of storing and retrieving credentials.
    """
    print("=== Basic Usage Example ===")
    
    # Configure the credential manager
    config = CredentialManagerConfig(
        master_password="your_secure_master_password_here",
        storage_backend="json",  # Use JSON file storage
        storage_path="~/.ttrpg_assistant/credentials",
        enable_validation=True,  # Validate API keys before storing
        enable_rotation=True     # Enable automatic rotation
    )
    
    # Create and initialize the credential manager
    manager = SecureCredentialManager(config)
    
    try:
        # Initialize the manager
        init_result = await manager.initialize()
        if not init_result.is_success():
            print(f"Failed to initialize: {init_result.failure()}")
            return
        
        print("✓ Credential manager initialized")
        
        # Store API keys for different providers
        user_id = "user_123"
        
        # Store Anthropic API key
        anthropic_key = "sk-ant-your-anthropic-api-key-here"
        store_result = await manager.store_credential(
            api_key=anthropic_key,
            provider_type=ProviderType.ANTHROPIC,
            user_id=user_id,
            display_name="My Anthropic Key"
        )
        
        if store_result.is_success():
            anthropic_cred_id = store_result.unwrap()
            print(f"✓ Stored Anthropic credential: {anthropic_cred_id}")
        else:
            print(f"✗ Failed to store Anthropic key: {store_result.failure()}")
        
        # Store OpenAI API key
        openai_key = "sk-your-openai-api-key-here"
        store_result = await manager.store_credential(
            api_key=openai_key,
            provider_type=ProviderType.OPENAI,
            user_id=user_id,
            display_name="My OpenAI Key"
        )
        
        if store_result.is_success():
            openai_cred_id = store_result.unwrap()
            print(f"✓ Stored OpenAI credential: {openai_cred_id}")
        
        # List all credentials for the user
        list_result = await manager.list_credentials(user_id=user_id)
        if list_result.is_success():
            credentials = list_result.unwrap()
            print(f"\n📋 User has {len(credentials)} stored credentials:")
            for cred in credentials:
                print(f"  - {cred.display_name} ({cred.provider_type.value})")
                print(f"    ID: {cred.credential_id}")
                print(f"    Created: {cred.created_at}")
                print(f"    Status: {cred.validation_status}")
        
        # Retrieve a credential for use
        if 'anthropic_cred_id' in locals():
            retrieve_result = await manager.retrieve_credential(anthropic_cred_id, user_id)
            if retrieve_result.is_success():
                # Use the decrypted API key
                decrypted_key = retrieve_result.unwrap()
                print(f"✓ Retrieved Anthropic key: {decrypted_key[:10]}...")
                
                # Key is automatically securely deleted from memory after use
    
    finally:
        # Always shutdown the manager
        await manager.shutdown()
        print("✓ Credential manager shut down")


async def advanced_usage_example():
    """
    Advanced example with custom configuration and rotation.
    """
    print("\n=== Advanced Usage Example ===")
    
    # Create custom configurations
    encryption_config = EncryptionConfig(
        pbkdf2_iterations=600_000,  # Strong key derivation
        salt_length=32,             # 256-bit salt
        key_length=32,              # AES-256
        master_key_rotation_days=365
    )
    
    storage_config = StorageConfig(
        json_storage_path="~/.ttrpg_assistant/secure_credentials",
        backup_count=10,            # Keep 10 backups
        backup_interval_hours=12,   # Backup every 12 hours
        enable_compression=True     # Compress storage
    )
    
    rotation_policy = RotationPolicy(
        max_age_days=90,                    # Rotate after 90 days
        rotation_warning_days=7,            # Warn 7 days before
        enable_scheduled_rotation=True,     # Enable automatic rotation
        rotation_schedule_hour=2,           # Rotate at 2 AM
        auto_rotate_on_validation_failure=True
    )
    
    config = CredentialManagerConfig(
        master_password="highly_secure_master_password_2024!",
        storage_backend="json",
        encryption_config=encryption_config,
        storage_config=storage_config,
        rotation_policy=rotation_policy,
        enable_validation=True,
        enable_rotation=True
    )
    
    manager = SecureCredentialManager(config)
    
    try:
        await manager.initialize()
        print("✓ Advanced credential manager initialized")
        
        user_id = "advanced_user"
        
        # Store credential with validation
        api_key = "sk-test-api-key-for-demo"
        store_result = await manager.store_credential(
            api_key=api_key,
            provider_type=ProviderType.ANTHROPIC,
            user_id=user_id,
            display_name="Production Anthropic Key",
            validate_before_storage=True  # Will validate key before storing
        )
        
        if store_result.is_success():
            cred_id = store_result.unwrap()
            print(f"✓ Stored and validated credential: {cred_id}")
            
            # Manually validate a credential
            validation_result = await manager.validate_credential(cred_id, user_id)
            if validation_result.is_success():
                validation = validation_result.unwrap()
                print(f"📊 Validation: {validation.get_summary()}")
                if validation.account_info:
                    print(f"   Account info: {validation.account_info}")
            
            # Manual rotation example
            print("\n🔄 Initiating manual credential rotation...")
            rotation_result = await manager.rotate_credential(
                credential_id=cred_id,
                user_id=user_id,
                new_api_key=None,  # Will create pending rotation
                reason=RotationReason.MANUAL
            )
            
            if rotation_result.is_success():
                rotation_record = rotation_result.unwrap()
                print(f"✓ Rotation initiated: {rotation_record.rotation_id}")
                print(f"   Status: {rotation_record.status.value}")
                print(f"   Reason: {rotation_record.reason.value}")
        
        # Get system health status
        health = await manager.health_check()
        print(f"\n🏥 System Health: {'✓ Healthy' if health['overall_healthy'] else '✗ Issues'}")
        for component, status in health['components'].items():
            health_icon = "✓" if status.get('healthy', False) else "✗"
            print(f"   {component}: {health_icon}")
    
    finally:
        await manager.shutdown()


async def integration_with_ai_providers_example():
    """
    Example of integrating with the existing AI provider system.
    """
    print("\n=== AI Provider Integration Example ===")
    
    # Initialize credential manager
    config = CredentialManagerConfig(
        master_password="integration_test_password",
        storage_backend="json"
    )
    
    manager = SecureCredentialManager(config)
    
    try:
        await manager.initialize()
        
        user_id = "integration_user"
        
        # Store credentials for multiple providers
        providers_and_keys = [
            (ProviderType.ANTHROPIC, "sk-ant-your-key-here"),
            (ProviderType.OPENAI, "sk-your-openai-key-here"),
            (ProviderType.GOOGLE, "your-google-api-key-here")
        ]
        
        for provider_type, api_key in providers_and_keys:
            await manager.store_credential(
                api_key=api_key,
                provider_type=provider_type,
                user_id=user_id,
                validate_before_storage=False  # Skip validation for demo
            )
        
        print("✓ Stored credentials for all providers")
        
        # Get provider configurations for use with AI provider manager
        for provider_type in [ProviderType.ANTHROPIC, ProviderType.OPENAI, ProviderType.GOOGLE]:
            config_result = await manager.get_provider_config(provider_type, user_id)
            
            if config_result.is_success():
                provider_config = config_result.unwrap()
                print(f"✓ Got config for {provider_type.value}:")
                print(f"   Enabled: {provider_config.enabled}")
                print(f"   API Key: {provider_config.api_key[:10]}...")
                
                # Here you would pass provider_config to your AIProviderManager
                # ai_provider_manager.initialize([provider_config])
            else:
                print(f"✗ Failed to get config for {provider_type.value}")
    
    finally:
        await manager.shutdown()


async def backup_and_recovery_example():
    """
    Example of backup and recovery operations.
    """
    print("\n=== Backup and Recovery Example ===")
    
    config = CredentialManagerConfig(
        master_password="backup_test_password",
        storage_backend="json"
    )
    
    manager = SecureCredentialManager(config)
    
    try:
        await manager.initialize()
        
        user_id = "backup_user"
        
        # Store some test credentials
        await manager.store_credential(
            api_key="sk-test-key-1",
            provider_type=ProviderType.ANTHROPIC,
            user_id=user_id,
            validate_before_storage=False
        )
        
        await manager.store_credential(
            api_key="sk-test-key-2",
            provider_type=ProviderType.OPENAI,
            user_id=user_id,
            validate_before_storage=False
        )
        
        print("✓ Created test credentials")
        
        # Create backup
        backup_result = await manager.storage_manager.backup_data()
        if backup_result.is_success():
            backup_path = backup_result.unwrap()
            print(f"✓ Created backup: {backup_path}")
            
            # In a real scenario, you might copy this backup to secure storage
            print("   Backup file can be copied to secure remote storage")
        
        # Get storage statistics
        stats = manager.storage_manager.get_storage_stats()
        print(f"\n📊 Storage Statistics:")
        print(f"   Backend: {stats['backend_type']}")
        if 'file_size_bytes' in stats:
            print(f"   File size: {stats['file_size_bytes']} bytes")
        if 'credential_count' in stats:
            print(f"   Credentials: {stats['credential_count']}")
    
    finally:
        await manager.shutdown()


async def chromadb_storage_example():
    """
    Example using ChromaDB as the storage backend.
    """
    print("\n=== ChromaDB Storage Example ===")
    
    config = CredentialManagerConfig(
        master_password="chromadb_test_password",
        storage_backend="chromadb",  # Use ChromaDB instead of JSON
        storage_path="~/.ttrpg_assistant/chromadb_credentials"
    )
    
    manager = SecureCredentialManager(config)
    
    try:
        await manager.initialize()
        print("✓ ChromaDB credential manager initialized")
        
        user_id = "chromadb_user"
        
        # Store credentials (same API as JSON backend)
        store_result = await manager.store_credential(
            api_key="sk-chromadb-test-key",
            provider_type=ProviderType.ANTHROPIC,
            user_id=user_id,
            display_name="ChromaDB Test Key",
            validate_before_storage=False
        )
        
        if store_result.is_success():
            cred_id = store_result.unwrap()
            print(f"✓ Stored credential in ChromaDB: {cred_id}")
            
            # Retrieve credential
            retrieve_result = await manager.retrieve_credential(cred_id, user_id)
            if retrieve_result.is_success():
                print("✓ Successfully retrieved credential from ChromaDB")
        
        # The API is identical regardless of storage backend
        # ChromaDB provides better search and metadata capabilities
        
    finally:
        await manager.shutdown()


async def monitoring_and_maintenance_example():
    """
    Example of monitoring and maintenance operations.
    """
    print("\n=== Monitoring and Maintenance Example ===")
    
    config = CredentialManagerConfig(
        master_password="monitoring_password",
        storage_backend="json"
    )
    
    manager = SecureCredentialManager(config)
    
    try:
        await manager.initialize()
        
        # Get comprehensive system status
        status = manager.get_system_status()
        print("📊 System Status:")
        print(f"   Initialized: {status['initialized']}")
        print(f"   Storage Backend: {status['components']['storage']['backend']}")
        print(f"   Cached Credentials: {status['credentials']['cached_count']}")
        print(f"   Validation Enabled: {status['components']['validation']['enabled']}")
        print(f"   Rotation Enabled: {status['components']['rotation']['enabled']}")
        
        # Perform health check
        health = await manager.health_check()
        print(f"\n🏥 Health Check (Overall: {'Healthy' if health['overall_healthy'] else 'Issues'}):")
        
        for component, health_info in health['components'].items():
            component_healthy = health_info.get('healthy', False)
            status_icon = "✓" if component_healthy else "✗"
            print(f"   {component}: {status_icon}")
            
            if not component_healthy and 'error' in health_info:
                print(f"      Error: {health_info['error']}")
        
        # If rotation is enabled, get rotation statistics
        if manager.rotation_service:
            rotation_stats = manager.rotation_service.get_rotation_statistics()
            print(f"\n🔄 Rotation Statistics:")
            print(f"   Total Rotations: {rotation_stats['total_rotations']}")
            print(f"   Active Rotations: {rotation_stats['active_rotations']}")
            print(f"   Scheduler Running: {rotation_stats['scheduler_running']}")
            
            if rotation_stats['reason_distribution']:
                print("   Rotation Reasons:")
                for reason, count in rotation_stats['reason_distribution'].items():
                    print(f"     {reason}: {count}")
    
    finally:
        await manager.shutdown()


def setup_logging():
    """
    Set up structured logging for the examples.
    """
    import logging
    import structlog
    
    # Configure structured logging
    logging.basicConfig(
        format="%(message)s",
        stream=None,
        level=logging.INFO,
    )
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


async def main():
    """
    Run all examples.
    """
    print("🔐 TTRPG Assistant - Secure Credential Management Examples")
    print("=" * 60)
    
    # Set up logging
    setup_logging()
    
    try:
        # Run examples
        await basic_usage_example()
        await advanced_usage_example()
        await integration_with_ai_providers_example()
        await backup_and_recovery_example()
        await chromadb_storage_example()
        await monitoring_and_maintenance_example()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())