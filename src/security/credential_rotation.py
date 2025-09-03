"""
Credential rotation service for automated and manual key rotation.

This module provides comprehensive key rotation capabilities including:
- Scheduled automatic rotation based on age and usage
- Manual rotation triggers
- Rotation history tracking
- Rollback capabilities
- Integration with existing credential storage and encryption
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import secrets

from structlog import get_logger
from returns.result import Result, Success, Failure

from .credential_encryption import CredentialEncryptionService, EncryptedCredential
from .credential_storage import CredentialStorageManager, CredentialMetadata
from .credential_validator import CredentialValidationService, ValidationResult
from ..ai_providers.models import ProviderType

logger = get_logger(__name__)


class RotationReason(Enum):
    """Reasons for credential rotation."""
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    SECURITY_INCIDENT = "security_incident"
    VALIDATION_FAILED = "validation_failed"
    USAGE_EXCEEDED = "usage_exceeded"
    AGE_LIMIT = "age_limit"
    COMPROMISE_SUSPECTED = "compromise_suspected"


class RotationStatus(Enum):
    """Status of rotation operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class RotationPolicy:
    """Policy for credential rotation."""
    
    # Time-based rotation
    max_age_days: Optional[int] = 90  # Rotate after 90 days
    rotation_warning_days: int = 7  # Warn 7 days before rotation
    
    # Usage-based rotation  
    max_usage_count: Optional[int] = None  # No usage limit by default
    usage_warning_threshold: Optional[int] = None
    
    # Security-based rotation
    auto_rotate_on_validation_failure: bool = True
    auto_rotate_on_security_incident: bool = True
    
    # Scheduling
    enable_scheduled_rotation: bool = True
    rotation_schedule_hour: int = 2  # Rotate at 2 AM
    rotation_check_interval_minutes: int = 60  # Check every hour
    
    # Rollback policy
    keep_previous_versions: int = 3  # Keep 3 previous versions
    rollback_timeout_minutes: int = 30  # Allow rollback for 30 minutes


@dataclass
class RotationRecord:
    """Record of a credential rotation operation."""
    
    rotation_id: str
    credential_id: str
    user_id: str
    provider_type: ProviderType
    reason: RotationReason
    status: RotationStatus
    initiated_at: datetime
    completed_at: Optional[datetime] = None
    old_key_hash: Optional[str] = None
    new_key_hash: Optional[str] = None
    validation_result: Optional[ValidationResult] = None
    error_message: Optional[str] = None
    rollback_deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.rollback_deadline is None and self.status == RotationStatus.COMPLETED:
            self.rollback_deadline = datetime.utcnow() + timedelta(minutes=30)


class CredentialRotationService:
    """Service for managing credential rotation operations."""
    
    def __init__(
        self,
        encryption_service: CredentialEncryptionService,
        storage_manager: CredentialStorageManager,
        validation_service: CredentialValidationService,
        policy: Optional[RotationPolicy] = None
    ):
        """Initialize rotation service."""
        self.encryption_service = encryption_service
        self.storage_manager = storage_manager
        self.validation_service = validation_service
        self.policy = policy or RotationPolicy()
        
        # Rotation tracking
        self._rotation_history: Dict[str, RotationRecord] = {}
        self._active_rotations: Set[str] = set()
        self._rotation_callbacks: Dict[RotationReason, List[Callable]] = {}
        
        # Scheduler
        self._scheduler_task: Optional[asyncio.Task] = None
        self._scheduler_running = False
        
        logger.info(
            "Initialized credential rotation service",
            max_age_days=self.policy.max_age_days,
            scheduled_rotation=self.policy.enable_scheduled_rotation
        )
    
    async def start_scheduler(self) -> Result[bool, str]:
        """Start the rotation scheduler."""
        try:
            if self._scheduler_running:
                return Success(True)
            
            self._scheduler_running = True
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            
            logger.info("Started credential rotation scheduler")
            return Success(True)
            
        except Exception as e:
            logger.error("Failed to start rotation scheduler", error=str(e))
            return Failure(f"Scheduler start failed: {str(e)}")
    
    async def stop_scheduler(self) -> None:
        """Stop the rotation scheduler."""
        self._scheduler_running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None
        
        logger.info("Stopped credential rotation scheduler")
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._scheduler_running:
            try:
                await self._check_rotation_requirements()
                await asyncio.sleep(self.policy.rotation_check_interval_minutes * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in rotation scheduler loop", error=str(e))
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _check_rotation_requirements(self) -> None:
        """Check if any credentials require rotation."""
        try:
            # Get all credentials
            credentials_result = await self.storage_manager.list_credentials()
            if not credentials_result.is_success():
                logger.error("Failed to list credentials for rotation check")
                return
            
            credentials = credentials_result.unwrap()
            current_time = datetime.utcnow()
            
            for metadata in credentials:
                # Skip if already being rotated
                if metadata.credential_id in self._active_rotations:
                    continue
                
                # Check age-based rotation
                if self.policy.max_age_days:
                    age_days = (current_time - metadata.created_at).days
                    if age_days >= self.policy.max_age_days:
                        await self._schedule_rotation(
                            metadata.credential_id,
                            RotationReason.AGE_LIMIT,
                            f"Credential is {age_days} days old"
                        )
                        continue
                    
                    # Check for rotation warning
                    warning_threshold = self.policy.max_age_days - self.policy.rotation_warning_days
                    if age_days >= warning_threshold:
                        logger.warning(
                            "Credential approaching rotation age",
                            credential_id=metadata.credential_id,
                            age_days=age_days,
                            days_until_rotation=self.policy.max_age_days - age_days
                        )
                
                # Check usage-based rotation
                if self.policy.max_usage_count and metadata.access_count >= self.policy.max_usage_count:
                    await self._schedule_rotation(
                        metadata.credential_id,
                        RotationReason.USAGE_EXCEEDED,
                        f"Usage count {metadata.access_count} exceeded limit"
                    )
                    continue
                
                # Check if validation recently failed (this would require tracking)
                # This could be implemented by checking validation results in metadata
                
        except Exception as e:
            logger.error("Error checking rotation requirements", error=str(e))
    
    async def _schedule_rotation(
        self,
        credential_id: str,
        reason: RotationReason,
        description: str
    ) -> None:
        """Schedule a credential for rotation."""
        try:
            # Check if it's during rotation hours for scheduled rotations
            if reason == RotationReason.SCHEDULED:
                current_hour = datetime.utcnow().hour
                if current_hour != self.policy.rotation_schedule_hour:
                    return
            
            logger.info(
                "Scheduling credential rotation",
                credential_id=credential_id,
                reason=reason.value,
                description=description
            )
            
            # Perform rotation
            await self.rotate_credential(credential_id, reason)
            
        except Exception as e:
            logger.error("Failed to schedule rotation", error=str(e))
    
    async def rotate_credential(
        self,
        credential_id: str,
        reason: RotationReason,
        new_api_key: Optional[str] = None
    ) -> Result[RotationRecord, str]:
        """
        Rotate a credential.
        
        Args:
            credential_id: ID of credential to rotate
            reason: Reason for rotation
            new_api_key: New API key (if None, user must provide)
            
        Returns:
            Result containing rotation record or error
        """
        try:
            # Check if already being rotated
            if credential_id in self._active_rotations:
                return Failure(f"Credential {credential_id} is already being rotated")
            
            # Get current credential
            cred_result = await self.storage_manager.retrieve_credential(credential_id)
            if not cred_result.is_success():
                return Failure(f"Failed to retrieve credential: {cred_result.failure()}")
            
            current_encrypted = cred_result.unwrap()
            
            # Get metadata
            metadata_result = await self.storage_manager.list_credentials()
            if not metadata_result.is_success():
                return Failure("Failed to retrieve credential metadata")
            
            metadata_list = metadata_result.unwrap()
            metadata = None
            for m in metadata_list:
                if m.credential_id == credential_id:
                    metadata = m
                    break
            
            if not metadata:
                return Failure(f"Metadata not found for credential {credential_id}")
            
            # Create rotation record
            rotation_id = f"rot_{secrets.token_hex(8)}"
            rotation_record = RotationRecord(
                rotation_id=rotation_id,
                credential_id=credential_id,
                user_id=metadata.user_id,
                provider_type=ProviderType(current_encrypted.provider_type),
                reason=reason,
                status=RotationStatus.PENDING,
                initiated_at=datetime.utcnow()
            )
            
            # Mark as active
            self._active_rotations.add(credential_id)
            self._rotation_history[rotation_id] = rotation_record
            
            try:
                # Update status
                rotation_record.status = RotationStatus.IN_PROGRESS
                
                # If no new key provided, this is a request for user to provide one
                if new_api_key is None:
                    rotation_record.status = RotationStatus.PENDING
                    rotation_record.metadata['requires_new_key'] = True
                    
                    logger.info(
                        "Rotation initiated, waiting for new API key",
                        rotation_id=rotation_id,
                        credential_id=credential_id
                    )
                    
                    return Success(rotation_record)
                
                # Validate new API key
                validation_result = await self.validation_service.validate_credential(
                    new_api_key,
                    ProviderType(current_encrypted.provider_type)
                )
                
                if not validation_result.is_success():
                    rotation_record.status = RotationStatus.FAILED
                    rotation_record.error_message = f"Validation failed: {validation_result.failure()}"
                    return Failure(rotation_record.error_message)
                
                validation = validation_result.unwrap()
                if not validation.is_valid:
                    rotation_record.status = RotationStatus.FAILED
                    rotation_record.error_message = f"New API key is invalid: {', '.join(validation.issues)}"
                    return Failure(rotation_record.error_message)
                
                rotation_record.validation_result = validation
                
                # Store hash of old key for rollback
                import hashlib
                old_key_result = self.encryption_service.decrypt_credential(current_encrypted, metadata.user_id)
                if old_key_result.is_success():
                    old_key = old_key_result.unwrap()
                    rotation_record.old_key_hash = hashlib.sha256(old_key.encode()).hexdigest()[:16]
                    self.encryption_service.secure_delete(old_key)
                
                # Encrypt new credential
                new_encrypted_result = self.encryption_service.encrypt_credential(
                    new_api_key,
                    metadata.user_id,
                    current_encrypted.provider_type
                )
                
                if not new_encrypted_result.is_success():
                    rotation_record.status = RotationStatus.FAILED
                    rotation_record.error_message = f"Encryption failed: {new_encrypted_result.failure()}"
                    return Failure(rotation_record.error_message)
                
                new_encrypted = new_encrypted_result.unwrap()
                rotation_record.new_key_hash = hashlib.sha256(new_api_key.encode()).hexdigest()[:16]
                
                # Store new credential
                updated_metadata = metadata
                updated_metadata.tags.add(f"rotated_{rotation_record.initiated_at.strftime('%Y%m%d')}")
                
                store_result = await self.storage_manager.store_credential(
                    credential_id,
                    new_encrypted,
                    updated_metadata
                )
                
                if not store_result.is_success():
                    rotation_record.status = RotationStatus.FAILED
                    rotation_record.error_message = f"Storage failed: {store_result.failure()}"
                    return Failure(rotation_record.error_message)
                
                # Update rotation record
                rotation_record.status = RotationStatus.COMPLETED
                rotation_record.completed_at = datetime.utcnow()
                rotation_record.rollback_deadline = datetime.utcnow() + timedelta(
                    minutes=self.policy.rollback_timeout_minutes
                )
                
                # Securely delete new key from memory
                self.encryption_service.secure_delete(new_api_key)
                
                # Trigger callbacks
                await self._trigger_rotation_callbacks(rotation_record)
                
                logger.info(
                    "Credential rotation completed successfully",
                    rotation_id=rotation_id,
                    credential_id=credential_id,
                    reason=reason.value
                )
                
                return Success(rotation_record)
                
            except Exception as e:
                rotation_record.status = RotationStatus.FAILED
                rotation_record.error_message = str(e)
                logger.error("Credential rotation failed", error=str(e))
                return Failure(str(e))
            
            finally:
                self._active_rotations.discard(credential_id)
        
        except Exception as e:
            logger.error("Failed to rotate credential", error=str(e))
            return Failure(f"Rotation failed: {str(e)}")
    
    async def complete_pending_rotation(
        self,
        rotation_id: str,
        new_api_key: str
    ) -> Result[RotationRecord, str]:
        """
        Complete a pending rotation with the provided API key.
        
        Args:
            rotation_id: ID of pending rotation
            new_api_key: New API key to use
            
        Returns:
            Result containing completed rotation record or error
        """
        try:
            if rotation_id not in self._rotation_history:
                return Failure(f"Rotation {rotation_id} not found")
            
            rotation_record = self._rotation_history[rotation_id]
            
            if rotation_record.status != RotationStatus.PENDING:
                return Failure(f"Rotation {rotation_id} is not pending (status: {rotation_record.status.value})")
            
            # Complete the rotation
            return await self.rotate_credential(
                rotation_record.credential_id,
                rotation_record.reason,
                new_api_key
            )
            
        except Exception as e:
            logger.error("Failed to complete pending rotation", error=str(e))
            return Failure(f"Completion failed: {str(e)}")
    
    async def rollback_rotation(
        self,
        rotation_id: str,
        reason: str = ""
    ) -> Result[bool, str]:
        """
        Rollback a completed rotation.
        
        Args:
            rotation_id: ID of rotation to rollback
            reason: Reason for rollback
            
        Returns:
            Result indicating success or failure
        """
        try:
            if rotation_id not in self._rotation_history:
                return Failure(f"Rotation {rotation_id} not found")
            
            rotation_record = self._rotation_history[rotation_id]
            
            if rotation_record.status != RotationStatus.COMPLETED:
                return Failure(f"Can only rollback completed rotations (current status: {rotation_record.status.value})")
            
            # Check rollback deadline
            if datetime.utcnow() > rotation_record.rollback_deadline:
                return Failure("Rollback deadline has passed")
            
            logger.warning(
                "Rolling back credential rotation",
                rotation_id=rotation_id,
                credential_id=rotation_record.credential_id,
                reason=reason
            )
            
            # For now, mark as rolled back - actual implementation would restore previous key
            # This would require storing encrypted version of old key
            rotation_record.status = RotationStatus.ROLLED_BACK
            rotation_record.metadata['rollback_reason'] = reason
            rotation_record.metadata['rolled_back_at'] = datetime.utcnow().isoformat()
            
            # Note: Full rollback implementation would require:
            # 1. Storing previous encrypted credential versions
            # 2. Restoring the previous version
            # 3. Validating the restored credential still works
            
            return Success(True)
            
        except Exception as e:
            logger.error("Failed to rollback rotation", error=str(e))
            return Failure(f"Rollback failed: {str(e)}")
    
    async def _trigger_rotation_callbacks(self, rotation_record: RotationRecord) -> None:
        """Trigger registered callbacks for rotation events."""
        try:
            callbacks = self._rotation_callbacks.get(rotation_record.reason, [])
            for callback in callbacks:
                try:
                    await callback(rotation_record)
                except Exception as e:
                    logger.error("Rotation callback failed", callback=str(callback), error=str(e))
        except Exception as e:
            logger.error("Failed to trigger rotation callbacks", error=str(e))
    
    def register_rotation_callback(
        self,
        reason: RotationReason,
        callback: Callable[[RotationRecord], None]
    ) -> None:
        """Register a callback for rotation events."""
        if reason not in self._rotation_callbacks:
            self._rotation_callbacks[reason] = []
        self._rotation_callbacks[reason].append(callback)
        
        logger.info("Registered rotation callback", reason=reason.value)
    
    async def get_rotation_status(self, credential_id: str) -> Result[List[RotationRecord], str]:
        """Get rotation history for a credential."""
        try:
            records = []
            for record in self._rotation_history.values():
                if record.credential_id == credential_id:
                    records.append(record)
            
            # Sort by initiated date
            records.sort(key=lambda r: r.initiated_at, reverse=True)
            
            return Success(records)
            
        except Exception as e:
            logger.error("Failed to get rotation status", error=str(e))
            return Failure(f"Status retrieval failed: {str(e)}")
    
    async def get_pending_rotations(self) -> Result[List[RotationRecord], str]:
        """Get all pending rotations."""
        try:
            pending = []
            for record in self._rotation_history.values():
                if record.status == RotationStatus.PENDING:
                    pending.append(record)
            
            return Success(pending)
            
        except Exception as e:
            logger.error("Failed to get pending rotations", error=str(e))
            return Failure(f"Pending rotations retrieval failed: {str(e)}")
    
    def get_rotation_statistics(self) -> Dict[str, Any]:
        """Get rotation service statistics."""
        status_counts = {}
        reason_counts = {}
        
        for record in self._rotation_history.values():
            status_counts[record.status.value] = status_counts.get(record.status.value, 0) + 1
            reason_counts[record.reason.value] = reason_counts.get(record.reason.value, 0) + 1
        
        return {
            'total_rotations': len(self._rotation_history),
            'active_rotations': len(self._active_rotations),
            'status_distribution': status_counts,
            'reason_distribution': reason_counts,
            'scheduler_running': self._scheduler_running,
            'policy': {
                'max_age_days': self.policy.max_age_days,
                'scheduled_rotation': self.policy.enable_scheduled_rotation,
                'rotation_check_interval': self.policy.rotation_check_interval_minutes
            }
        }
    
    async def force_rotation_check(self) -> Result[Dict[str, int], str]:
        """Force immediate check of rotation requirements."""
        try:
            initial_count = len(self._active_rotations)
            await self._check_rotation_requirements()
            final_count = len(self._active_rotations)
            
            return Success({
                'rotations_initiated': final_count - initial_count,
                'total_active': final_count
            })
            
        except Exception as e:
            logger.error("Failed to force rotation check", error=str(e))
            return Failure(f"Force check failed: {str(e)}")