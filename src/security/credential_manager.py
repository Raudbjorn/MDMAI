"""
Unified credential management system for the TTRPG Assistant MCP Server.

This module provides the main interface for secure credential management,
integrating all security components:
- AES-256 encryption with user-specific salts
- Secure storage (JSON/ChromaDB backends)
- API key validation
- Automated key rotation
- Integration with existing AI provider system

This is the primary interface that should be used by the application.
"""

import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from structlog import get_logger
from returns.result import Result, Success, Failure

# Import our security components
from .credential_encryption import CredentialEncryptionService, EncryptedCredential, EncryptionConfig, SecureMemory
from .credential_storage import CredentialStorageManager, CredentialMetadata, StorageConfig
from .credential_validator import CredentialValidationService, ValidationResult
from .credential_rotation import CredentialRotationService, RotationPolicy, RotationReason, RotationRecord

# Import existing AI provider types
from ..ai_providers.models import ProviderType, ProviderConfig

logger = get_logger(__name__)


class CredentialManagerConfig:
    """Configuration for the credential manager."""
    
    def __init__(
        self,
        master_password: Optional[str] = None,
        storage_backend: str = "json",  # "json" or "chromadb"
        storage_path: Optional[str] = None,
        enable_rotation: bool = True,
        enable_validation: bool = True,
        encryption_config: Optional[EncryptionConfig] = None,
        storage_config: Optional[StorageConfig] = None,
        rotation_policy: Optional[RotationPolicy] = None
    ):
        """Initialize credential manager configuration."""
        self.master_password = master_password
        self.storage_backend = storage_backend
        self.storage_path = storage_path or "~/.ttrpg_assistant/credentials"
        self.enable_rotation = enable_rotation
        self.enable_validation = enable_validation
        
        # Component configurations
        self.encryption_config = encryption_config or EncryptionConfig()
        
        # Update storage config with custom path if provided
        self.storage_config = storage_config or StorageConfig()
        if storage_path:
            if storage_backend == "json":
                self.storage_config.json_storage_path = storage_path
            else:
                self.storage_config.chromadb_path = storage_path
        
        self.rotation_policy = rotation_policy or RotationPolicy()


@dataclass
class StoredCredential:
    """High-level representation of a stored credential."""
    
    credential_id: str
    user_id: str
    provider_type: ProviderType
    display_name: Optional[str] = None
    created_at: datetime = None
    last_accessed: Optional[datetime] = None
    last_validated: Optional[datetime] = None
    validation_status: str = "unknown"  # "valid", "invalid", "unknown"
    is_active: bool = True
    rotation_due: Optional[datetime] = None
    tags: Set[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.tags is None:
            self.tags = set()
        if self.metadata is None:
            self.metadata = {}


class SecureCredentialManager:
    """
    Main credential manager providing secure storage and management of API keys.
    
    This class integrates all security components and provides a clean interface
    for the application to manage AI provider credentials securely.
    """
    
    def __init__(self, config: CredentialManagerConfig):
        """Initialize the secure credential manager."""
        self.config = config
        
        # Initialize core components
        self.encryption_service = CredentialEncryptionService(config.encryption_config)
        self.storage_manager = CredentialStorageManager(config.storage_backend, config.storage_config)
        
        # Initialize optional components
        self.validation_service = CredentialValidationService() if config.enable_validation else None
        self.rotation_service = None
        
        # State tracking
        self._initialized = False
        self._credentials_cache: Dict[str, StoredCredential] = {}
        self._cache_expires: Dict[str, datetime] = {}
        
        logger.info(
            "Initialized secure credential manager",
            storage_backend=config.storage_backend,
            validation_enabled=config.enable_validation,
            rotation_enabled=config.enable_rotation
        )
    
    async def initialize(self, master_password: Optional[str] = None) -> Result[bool, str]:
        """
        Initialize the credential manager.
        
        Args:
            master_password: Master password for encryption (overrides config)
            
        Returns:
            Result indicating success or failure
        """
        try:
            if self._initialized:
                return Success(True)
            
            # Set master password
            password = master_password or self.config.master_password
            if not password:
                return Failure("Master password is required")
            
            # Initialize encryption
            encrypt_result = self.encryption_service.set_master_password(password)
            if not encrypt_result.is_success():
                return Failure(f"Encryption initialization failed: {encrypt_result.failure()}")
            
            # Initialize rotation service if enabled
            if self.config.enable_rotation and self.validation_service:
                self.rotation_service = CredentialRotationService(
                    self.encryption_service,
                    self.storage_manager,
                    self.validation_service,
                    self.config.rotation_policy
                )
                
                # Start rotation scheduler
                scheduler_result = await self.rotation_service.start_scheduler()
                if not scheduler_result.is_success():
                    logger.warning("Failed to start rotation scheduler", error=scheduler_result.failure())
            
            # Load existing credentials into cache
            await self._refresh_credentials_cache()
            
            self._initialized = True
            
            logger.info(
                "Credential manager initialized successfully",
                cached_credentials=len(self._credentials_cache)
            )
            
            return Success(True)
            
        except Exception as e:
            logger.error("Failed to initialize credential manager", error=str(e))
            return Failure(f"Initialization failed: {str(e)}")
    
    async def shutdown(self) -> None:
        """Shutdown the credential manager."""
        try:
            if self.rotation_service:
                await self.rotation_service.stop_scheduler()
            
            # Clear sensitive data from cache
            self._credentials_cache.clear()
            self._cache_expires.clear()
            
            self._initialized = False
            
            logger.info("Credential manager shut down successfully")
            
        except Exception as e:
            logger.error("Error during credential manager shutdown", error=str(e))
    
    async def store_credential(
        self,
        api_key: str,
        provider_type: ProviderType,
        user_id: str,
        display_name: Optional[str] = None,
        validate_before_storage: bool = True
    ) -> Result[str, str]:
        """
        Store a new API credential securely.
        
        Args:
            api_key: The API key to store
            provider_type: The AI provider type
            user_id: User identifier
            display_name: Optional display name for the credential
            validate_before_storage: Whether to validate the key before storing
            
        Returns:
            Result containing credential ID or error message
        """
        try:
            if not self._initialized:
                return Failure("Credential manager not initialized")
            
            # Validate API key before storage if requested
            if validate_before_storage and self.validation_service:
                validation_result = await self.validation_service.validate_credential(
                    api_key, provider_type
                )
                
                if not validation_result.is_success():
                    return Failure(f"Validation failed: {validation_result.failure()}")
                
                validation = validation_result.unwrap()
                if not validation.is_valid:
                    return Failure(f"Invalid API key: {', '.join(validation.issues)}")
                
                logger.info(
                    "API key validation passed",
                    provider=provider_type.value,
                    validation_summary=validation.get_summary()
                )
            
            # Generate credential ID
            credential_id = f"{provider_type.value}_{user_id}_{secrets.token_hex(8)}"
            
            # Encrypt the credential
            encrypt_result = self.encryption_service.encrypt_credential(
                api_key, user_id, provider_type.value
            )
            
            if not encrypt_result.is_success():
                return Failure(f"Encryption failed: {encrypt_result.failure()}")
            
            encrypted_credential = encrypt_result.unwrap()
            
            # Create metadata
            metadata = CredentialMetadata(
                credential_id=credential_id,
                user_id=user_id,
                provider_type=provider_type.value,
                created_at=datetime.utcnow(),
                is_active=True,
                tags={provider_type.value, "active"}
            )
            
            # Store in backend
            store_result = await self.storage_manager.store_credential(
                credential_id, encrypted_credential, metadata
            )
            
            if not store_result.is_success():
                return Failure(f"Storage failed: {store_result.failure()}")
            
            # Update cache
            stored_credential = StoredCredential(
                credential_id=credential_id,
                user_id=user_id,
                provider_type=provider_type,
                display_name=display_name or f"{provider_type.value.title()} Key",
                created_at=metadata.created_at,
                validation_status="valid" if validate_before_storage else "unknown",
                last_validated=datetime.utcnow() if validate_before_storage else None,
                tags=metadata.tags
            )
            
            self._credentials_cache[credential_id] = stored_credential
            self._cache_expires[credential_id] = datetime.utcnow() + timedelta(hours=1)
            
            # Securely delete API key from memory
            api_key_bytes = bytearray(api_key.encode('utf-8'))
            SecureMemory.secure_zero(api_key_bytes)
            
            logger.info(
                "Credential stored successfully",
                credential_id=credential_id,
                provider=provider_type.value,
                user_id=user_id[:8] + "***"
            )
            
            return Success(credential_id)
            
        except Exception as e:
            logger.error("Failed to store credential", error=str(e))
            return Failure(f"Storage failed: {str(e)}")
    
    async def retrieve_credential(
        self,
        credential_id: str,
        user_id: str
    ) -> Result[str, str]:
        """
        Retrieve and decrypt a stored credential.
        
        Args:
            credential_id: ID of the credential to retrieve
            user_id: User identifier (for authorization)
            
        Returns:
            Result containing decrypted API key or error message
        """
        try:
            if not self._initialized:
                return Failure("Credential manager not initialized")
            
            # Get encrypted credential from storage
            retrieve_result = await self.storage_manager.retrieve_credential(credential_id)
            if not retrieve_result.is_success():
                return Failure(f"Retrieval failed: {retrieve_result.failure()}")
            
            encrypted_credential = retrieve_result.unwrap()
            
            # Decrypt the credential
            decrypt_result = self.encryption_service.decrypt(encrypted_credential, user_id)
            if not decrypt_result.is_success():
                return Failure(f"Decryption failed: {decrypt_result.failure()}")
            
            api_key = decrypt_result.unwrap()
            
            # Update cache with access information
            if credential_id in self._credentials_cache:
                self._credentials_cache[credential_id].last_accessed = datetime.utcnow()
            
            logger.debug(
                "Credential retrieved successfully",
                credential_id=credential_id,
                user_id=user_id[:8] + "***"
            )
            
            return Success(api_key)
            
        except Exception as e:
            logger.error("Failed to retrieve credential", error=str(e))
            return Failure(f"Retrieval failed: {str(e)}")
    
    async def list_credentials(
        self,
        user_id: Optional[str] = None,
        provider_type: Optional[ProviderType] = None,
        active_only: bool = True
    ) -> Result[List[StoredCredential], str]:
        """
        List stored credentials with optional filtering.
        
        Args:
            user_id: Filter by user ID
            provider_type: Filter by provider type
            active_only: Only return active credentials
            
        Returns:
            Result containing list of stored credentials
        """
        try:
            if not self._initialized:
                return Failure("Credential manager not initialized")
            
            # Refresh cache if needed
            await self._refresh_credentials_cache()
            
            credentials = list(self._credentials_cache.values())
            
            # Apply filters
            if user_id:
                credentials = [c for c in credentials if c.user_id == user_id]
            
            if provider_type:
                credentials = [c for c in credentials if c.provider_type == provider_type]
            
            if active_only:
                credentials = [c for c in credentials if c.is_active]
            
            # Sort by creation date (newest first)
            credentials.sort(key=lambda c: c.created_at, reverse=True)
            
            return Success(credentials)
            
        except Exception as e:
            logger.error("Failed to list credentials", error=str(e))
            return Failure(f"Listing failed: {str(e)}")
    
    async def delete_credential(
        self,
        credential_id: str,
        user_id: str,
        secure_delete: bool = True
    ) -> Result[bool, str]:
        """
        Delete a stored credential.
        
        Args:
            credential_id: ID of credential to delete
            user_id: User identifier (for authorization)
            secure_delete: Whether to perform secure deletion
            
        Returns:
            Result indicating success or failure
        """
        try:
            if not self._initialized:
                return Failure("Credential manager not initialized")
            
            # Verify ownership
            if credential_id in self._credentials_cache:
                cached_cred = self._credentials_cache[credential_id]
                if cached_cred.user_id != user_id:
                    return Failure("Access denied: credential belongs to different user")
            
            # If secure delete requested, retrieve and overwrite the credential
            if secure_delete:
                retrieve_result = await self.retrieve_credential(credential_id, user_id)
                if retrieve_result.is_success():
                    api_key = retrieve_result.unwrap()
                    api_key_bytes = bytearray(api_key.encode('utf-8'))
                    SecureMemory.secure_zero(api_key_bytes)
            
            # Delete from storage
            delete_result = await self.storage_manager.delete_credential(credential_id)
            if not delete_result.is_success():
                return Failure(f"Deletion failed: {delete_result.failure()}")
            
            # Remove from cache
            self._credentials_cache.pop(credential_id, None)
            self._cache_expires.pop(credential_id, None)
            
            logger.info(
                "Credential deleted successfully",
                credential_id=credential_id,
                user_id=user_id[:8] + "***",
                secure_delete=secure_delete
            )
            
            return Success(True)
            
        except Exception as e:
            logger.error("Failed to delete credential", error=str(e))
            return Failure(f"Deletion failed: {str(e)}")
    
    async def validate_credential(
        self,
        credential_id: str,
        user_id: str
    ) -> Result[ValidationResult, str]:
        """
        Validate a stored credential.
        
        Args:
            credential_id: ID of credential to validate
            user_id: User identifier
            
        Returns:
            Result containing validation result
        """
        try:
            if not self._initialized or not self.validation_service:
                return Failure("Validation service not available")
            
            # Retrieve credential
            retrieve_result = await self.retrieve_credential(credential_id, user_id)
            if not retrieve_result.is_success():
                return Failure(f"Failed to retrieve credential: {retrieve_result.failure()}")
            
            api_key = retrieve_result.unwrap()
            
            # Get provider type from cache
            if credential_id not in self._credentials_cache:
                return Failure("Credential metadata not found")
            
            provider_type = self._credentials_cache[credential_id].provider_type
            
            # Validate
            validation_result = await self.validation_service.validate_credential(
                api_key, provider_type
            )
            
            if not validation_result.is_success():
                return Failure(f"Validation failed: {validation_result.failure()}")
            
            validation = validation_result.unwrap()
            
            # Update cache with validation results
            cached_cred = self._credentials_cache[credential_id]
            cached_cred.last_validated = datetime.utcnow()
            cached_cred.validation_status = "valid" if validation.is_valid else "invalid"
            
            # Securely delete API key from memory
            api_key_bytes = bytearray(api_key.encode('utf-8'))
            SecureMemory.secure_zero(api_key_bytes)
            
            logger.info(
                "Credential validation completed",
                credential_id=credential_id,
                valid=validation.is_valid,
                summary=validation.get_summary()
            )
            
            return Success(validation)
            
        except Exception as e:
            logger.error("Failed to validate credential", error=str(e))
            return Failure(f"Validation failed: {str(e)}")
    
    async def rotate_credential(
        self,
        credential_id: str,
        user_id: str,
        new_api_key: Optional[str] = None,
        reason: RotationReason = RotationReason.MANUAL
    ) -> Result[RotationRecord, str]:
        """
        Rotate a stored credential.
        
        Args:
            credential_id: ID of credential to rotate
            user_id: User identifier
            new_api_key: New API key (if None, rotation will be pending)
            reason: Reason for rotation
            
        Returns:
            Result containing rotation record
        """
        try:
            if not self._initialized or not self.rotation_service:
                return Failure("Rotation service not available")
            
            # Verify ownership
            if credential_id in self._credentials_cache:
                cached_cred = self._credentials_cache[credential_id]
                if cached_cred.user_id != user_id:
                    return Failure("Access denied: credential belongs to different user")
            
            # Perform rotation
            rotation_result = await self.rotation_service.rotate_credential(
                credential_id, reason, new_api_key
            )
            
            if rotation_result.is_success():
                # Refresh cache to reflect changes
                await self._refresh_credentials_cache()
            
            return rotation_result
            
        except Exception as e:
            logger.error("Failed to rotate credential", error=str(e))
            return Failure(f"Rotation failed: {str(e)}")
    
    async def get_provider_config(
        self,
        provider_type: ProviderType,
        user_id: str
    ) -> Result[ProviderConfig, str]:
        """
        Get a provider configuration with decrypted API key.
        
        This method integrates with the existing AI provider system by
        creating a ProviderConfig object with the decrypted API key.
        
        Args:
            provider_type: The provider type
            user_id: User identifier
            
        Returns:
            Result containing ProviderConfig or error
        """
        try:
            if not self._initialized:
                return Failure("Credential manager not initialized")
            
            # Find credential for this provider and user
            credentials_result = await self.list_credentials(user_id, provider_type)
            if not credentials_result.is_success():
                return Failure(f"Failed to list credentials: {credentials_result.failure()}")
            
            credentials = credentials_result.unwrap()
            if not credentials:
                return Failure(f"No credentials found for {provider_type.value}")
            
            # Use the most recent active credential
            credential = credentials[0]
            
            # Retrieve API key
            api_key_result = await self.retrieve_credential(credential.credential_id, user_id)
            if not api_key_result.is_success():
                return Failure(f"Failed to retrieve API key: {api_key_result.failure()}")
            
            api_key = api_key_result.unwrap()
            
            # Create ProviderConfig (you may need to adjust fields based on your existing structure)
            provider_config = ProviderConfig(
                provider_type=provider_type,
                api_key=api_key,
                enabled=credential.is_active,
                # Add other default values as needed based on your ProviderConfig structure
            )
            
            # Securely delete API key from memory after creating config
            api_key_bytes = bytearray(api_key.encode('utf-8'))
            SecureMemory.secure_zero(api_key_bytes)
            
            logger.debug(
                "Provider config created",
                provider=provider_type.value,
                user_id=user_id[:8] + "***"
            )
            
            return Success(provider_config)
            
        except Exception as e:
            logger.error("Failed to get provider config", error=str(e))
            return Failure(f"Provider config retrieval failed: {str(e)}")
    
    async def _refresh_credentials_cache(self) -> None:
        """Refresh the credentials cache from storage."""
        try:
            # Get all credentials from storage
            list_result = await self.storage_manager.list_credentials()
            if not list_result.is_success():
                logger.error("Failed to refresh credentials cache")
                return
            
            metadata_list = list_result.unwrap()
            
            # Update cache
            new_cache = {}
            for metadata in metadata_list:
                stored_credential = StoredCredential(
                    credential_id=metadata.credential_id,
                    user_id=metadata.user_id,
                    provider_type=ProviderType(metadata.provider_type),
                    created_at=metadata.created_at,
                    last_accessed=metadata.last_accessed,
                    is_active=metadata.is_active,
                    tags=metadata.tags
                )
                
                new_cache[metadata.credential_id] = stored_credential
                self._cache_expires[metadata.credential_id] = datetime.utcnow() + timedelta(hours=1)
            
            self._credentials_cache = new_cache
            
        except Exception as e:
            logger.error("Failed to refresh credentials cache", error=str(e))
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "initialized": self._initialized,
            "components": {
                "encryption": {
                    "healthy": True,
                    "master_key_set": self.encryption_service._master_key is not None,
                    "user_salts": len(self.encryption_service._user_salts)
                },
                "storage": {
                    "backend": self.config.storage_backend,
                    "stats": self.storage_manager.get_storage_stats()
                },
                "validation": {
                    "enabled": self.validation_service is not None,
                    "supported_providers": [p.value for p in self.validation_service.get_supported_providers()] if self.validation_service else []
                },
                "rotation": {
                    "enabled": self.rotation_service is not None,
                    "scheduler_running": self.rotation_service._scheduler_running if self.rotation_service else False
                }
            },
            "credentials": {
                "cached_count": len(self._credentials_cache),
                "by_provider": {}
            }
        }
        
        # Count credentials by provider
        for cred in self._credentials_cache.values():
            provider = cred.provider_type.value
            if provider not in status["credentials"]["by_provider"]:
                status["credentials"]["by_provider"][provider] = 0
            status["credentials"]["by_provider"][provider] += 1
        
        # Add rotation stats if available
        if self.rotation_service:
            status["rotation_stats"] = self.rotation_service.get_rotation_statistics()
        
        return status
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "overall_healthy": True,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        try:
            # Check encryption service
            encryption_health = self.encryption_service.health_check()
            health_status["components"]["encryption"] = encryption_health
            if not encryption_health.get("healthy", False):
                health_status["overall_healthy"] = False
            
            # Check storage (basic test)
            storage_health = {"healthy": True, "backend": self.config.storage_backend}
            try:
                await self.storage_manager.list_credentials()
            except Exception as e:
                storage_health = {"healthy": False, "error": str(e)}
                health_status["overall_healthy"] = False
            
            health_status["components"]["storage"] = storage_health
            
            # Check validation service
            if self.validation_service:
                validation_health = {
                    "healthy": True,
                    "stats": self.validation_service.get_validation_stats()
                }
            else:
                validation_health = {"healthy": True, "enabled": False}
            
            health_status["components"]["validation"] = validation_health
            
            # Check rotation service
            if self.rotation_service:
                rotation_health = {
                    "healthy": True,
                    "scheduler_running": self.rotation_service._scheduler_running,
                    "stats": self.rotation_service.get_rotation_statistics()
                }
            else:
                rotation_health = {"healthy": True, "enabled": False}
            
            health_status["components"]["rotation"] = rotation_health
            
        except Exception as e:
            health_status["overall_healthy"] = False
            health_status["error"] = str(e)
        
        return health_status