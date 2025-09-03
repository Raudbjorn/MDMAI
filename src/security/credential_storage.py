"""
Secure credential storage backend with JSON and ChromaDB support.

This module provides secure, persistent storage for encrypted credentials
with the following features:
- Dual backend support (JSON file and ChromaDB)
- Atomic operations for data integrity
- Backup and recovery mechanisms
- Audit logging for all operations
- Local-first architecture suitable for desktop and server deployments
"""

import json
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any
from dataclasses import dataclass, asdict

import chromadb
from chromadb.config import Settings
from structlog import get_logger
from returns.result import Result, Success, Failure

from .credential_encryption import EncryptedCredential

logger = get_logger(__name__)


@dataclass
class StorageConfig:
    """Configuration for credential storage."""
    
    # JSON storage settings
    json_storage_path: str = "~/.ttrpg_assistant/credentials"
    json_filename: str = "encrypted_credentials.json"
    backup_count: int = 5
    backup_interval_hours: int = 24
    
    # ChromaDB settings
    chromadb_path: str = "~/.ttrpg_assistant/chromadb"
    chromadb_collection: str = "encrypted_credentials"
    chromadb_persist_directory: Optional[str] = None
    
    # Security settings
    file_permissions: int = 0o600  # Read/write for owner only
    directory_permissions: int = 0o700  # Read/write/execute for owner only
    enable_audit_log: bool = True
    audit_log_path: str = "~/.ttrpg_assistant/audit.log"
    
    # Performance settings
    enable_compression: bool = False
    max_storage_size_mb: int = 100
    cleanup_expired_after_days: int = 90


@dataclass 
class CredentialMetadata:
    """Metadata for stored credentials."""
    
    credential_id: str
    user_id: str
    provider_type: str
    created_at: datetime
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    is_active: bool = True
    tags: Set[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = set()


class CredentialStorageBackend(ABC):
    """Abstract base class for credential storage backends."""
    
    @abstractmethod
    async def store_credential(
        self, 
        credential_id: str,
        encrypted_credential: EncryptedCredential,
        metadata: CredentialMetadata
    ) -> Result[bool, str]:
        """Store an encrypted credential."""
        pass
    
    @abstractmethod
    async def retrieve_credential(
        self, 
        credential_id: str
    ) -> Result[EncryptedCredential, str]:
        """Retrieve an encrypted credential."""
        pass
    
    @abstractmethod
    async def delete_credential(
        self, 
        credential_id: str
    ) -> Result[bool, str]:
        """Delete a credential."""
        pass
    
    @abstractmethod
    async def list_credentials(
        self, 
        user_id: Optional[str] = None,
        provider_type: Optional[str] = None
    ) -> Result[List[CredentialMetadata], str]:
        """List credentials with optional filtering."""
        pass
    
    @abstractmethod
    async def update_metadata(
        self, 
        credential_id: str,
        metadata: CredentialMetadata
    ) -> Result[bool, str]:
        """Update credential metadata."""
        pass
    
    @abstractmethod
    async def backup_data(
        self, 
        backup_path: Optional[str] = None
    ) -> Result[str, str]:
        """Create backup of credential data."""
        pass
    
    @abstractmethod
    async def restore_data(
        self, 
        backup_path: str
    ) -> Result[bool, str]:
        """Restore credential data from backup."""
        pass


class JSONCredentialStorage(CredentialStorageBackend):
    """JSON file-based credential storage backend."""
    
    def __init__(self, config: StorageConfig):
        """Initialize JSON storage backend."""
        self.config = config
        self.storage_dir = Path(config.json_storage_path).expanduser()
        self.storage_file = self.storage_dir / config.json_filename
        self.backup_dir = self.storage_dir / "backups"
        
        # Ensure directories exist with proper permissions
        self._ensure_directories()
        
        logger.info(
            "Initialized JSON credential storage",
            storage_path=str(self.storage_file),
            backup_dir=str(self.backup_dir)
        )
    
    def _ensure_directories(self) -> None:
        """Ensure storage directories exist with proper permissions."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Set restrictive permissions
        os.chmod(self.storage_dir, self.config.directory_permissions)
        os.chmod(self.backup_dir, self.config.directory_permissions)
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data from JSON file."""
        if not self.storage_file.exists():
            return {"credentials": {}, "metadata": {}, "version": "1.0"}
        
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Failed to load credentials file", error=str(e))
            return {"credentials": {}, "metadata": {}, "version": "1.0"}
    
    def _save_data(self, data: Dict[str, Any]) -> Result[bool, str]:
        """Atomically save data to JSON file."""
        try:
            # Write to temporary file first for atomic operation
            temp_file = self.storage_file.with_suffix('.tmp')
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Set file permissions
            os.chmod(temp_file, self.config.file_permissions)
            
            # Atomic move
            shutil.move(str(temp_file), str(self.storage_file))
            
            return Success(True)
            
        except Exception as e:
            logger.error("Failed to save credentials file", error=str(e))
            return Failure(f"Failed to save: {str(e)}")
    
    async def store_credential(
        self,
        credential_id: str,
        encrypted_credential: EncryptedCredential,
        metadata: CredentialMetadata
    ) -> Result[bool, str]:
        """Store an encrypted credential."""
        try:
            data = self._load_data()
            
            # Store encrypted credential
            data["credentials"][credential_id] = encrypted_credential.to_dict()
            
            # Store metadata
            metadata_dict = asdict(metadata)
            metadata_dict['tags'] = list(metadata_dict['tags'])  # Convert set to list
            data["metadata"][credential_id] = metadata_dict
            
            # Save atomically
            save_result = self._save_data(data)
            if not save_result.is_success():
                return save_result
            
            logger.info(
                "Stored credential",
                credential_id=credential_id,
                provider_type=encrypted_credential.provider_type
            )
            
            return Success(True)
            
        except Exception as e:
            logger.error("Failed to store credential", error=str(e))
            return Failure(f"Storage failed: {str(e)}")
    
    async def retrieve_credential(self, credential_id: str) -> Result[EncryptedCredential, str]:
        """Retrieve an encrypted credential."""
        try:
            data = self._load_data()
            
            if credential_id not in data["credentials"]:
                return Failure(f"Credential {credential_id} not found")
            
            cred_data = data["credentials"][credential_id]
            encrypted_credential = EncryptedCredential.from_dict(cred_data)
            
            # Update access metadata
            if credential_id in data["metadata"]:
                metadata = data["metadata"][credential_id]
                metadata["last_accessed"] = datetime.utcnow().isoformat()
                metadata["access_count"] = metadata.get("access_count", 0) + 1
                
                # Save updated metadata
                self._save_data(data)
            
            logger.debug("Retrieved credential", credential_id=credential_id)
            
            return Success(encrypted_credential)
            
        except Exception as e:
            logger.error("Failed to retrieve credential", error=str(e))
            return Failure(f"Retrieval failed: {str(e)}")
    
    async def delete_credential(self, credential_id: str) -> Result[bool, str]:
        """Delete a credential."""
        try:
            data = self._load_data()
            
            if credential_id not in data["credentials"]:
                return Failure(f"Credential {credential_id} not found")
            
            # Remove credential and metadata
            del data["credentials"][credential_id]
            if credential_id in data["metadata"]:
                del data["metadata"][credential_id]
            
            # Save changes
            save_result = self._save_data(data)
            if not save_result.is_success():
                return save_result
            
            logger.info("Deleted credential", credential_id=credential_id)
            
            return Success(True)
            
        except Exception as e:
            logger.error("Failed to delete credential", error=str(e))
            return Failure(f"Deletion failed: {str(e)}")
    
    async def list_credentials(
        self,
        user_id: Optional[str] = None,
        provider_type: Optional[str] = None
    ) -> Result[List[CredentialMetadata], str]:
        """List credentials with optional filtering."""
        try:
            data = self._load_data()
            credentials = []
            
            for cred_id, metadata_dict in data["metadata"].items():
                # Convert back to CredentialMetadata
                metadata_dict['tags'] = set(metadata_dict.get('tags', []))
                metadata = CredentialMetadata(**metadata_dict)
                
                # Apply filters
                if user_id and metadata.user_id != user_id:
                    continue
                if provider_type and metadata.provider_type != provider_type:
                    continue
                
                credentials.append(metadata)
            
            return Success(credentials)
            
        except Exception as e:
            logger.error("Failed to list credentials", error=str(e))
            return Failure(f"Listing failed: {str(e)}")
    
    async def update_metadata(
        self,
        credential_id: str,
        metadata: CredentialMetadata
    ) -> Result[bool, str]:
        """Update credential metadata."""
        try:
            data = self._load_data()
            
            if credential_id not in data["credentials"]:
                return Failure(f"Credential {credential_id} not found")
            
            # Update metadata
            metadata_dict = asdict(metadata)
            metadata_dict['tags'] = list(metadata_dict['tags'])
            data["metadata"][credential_id] = metadata_dict
            
            # Save changes
            save_result = self._save_data(data)
            if not save_result.is_success():
                return save_result
            
            logger.debug("Updated credential metadata", credential_id=credential_id)
            
            return Success(True)
            
        except Exception as e:
            logger.error("Failed to update metadata", error=str(e))
            return Failure(f"Update failed: {str(e)}")
    
    async def backup_data(self, backup_path: Optional[str] = None) -> Result[str, str]:
        """Create backup of credential data."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            if backup_path:
                backup_file = Path(backup_path)
            else:
                backup_file = self.backup_dir / f"credentials_backup_{timestamp}.json"
            
            # Copy current storage file
            if self.storage_file.exists():
                shutil.copy2(self.storage_file, backup_file)
                os.chmod(backup_file, self.config.file_permissions)
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            logger.info("Created credential backup", backup_path=str(backup_file))
            
            return Success(str(backup_file))
            
        except Exception as e:
            logger.error("Failed to create backup", error=str(e))
            return Failure(f"Backup failed: {str(e)}")
    
    def _cleanup_old_backups(self) -> None:
        """Remove old backup files beyond retention limit."""
        try:
            backup_files = list(self.backup_dir.glob("credentials_backup_*.json"))
            backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Remove files beyond backup count
            for old_backup in backup_files[self.config.backup_count:]:
                old_backup.unlink()
                logger.debug("Removed old backup", backup_file=str(old_backup))
                
        except Exception as e:
            logger.warning("Failed to cleanup old backups", error=str(e))
    
    async def restore_data(self, backup_path: str) -> Result[bool, str]:
        """Restore credential data from backup."""
        try:
            backup_file = Path(backup_path)
            
            if not backup_file.exists():
                return Failure(f"Backup file not found: {backup_path}")
            
            # Validate backup file
            try:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
                
                # Basic validation
                required_keys = ["credentials", "metadata", "version"]
                if not all(key in backup_data for key in required_keys):
                    return Failure("Invalid backup file format")
                    
            except json.JSONDecodeError:
                return Failure("Backup file is corrupted")
            
            # Create backup of current file
            if self.storage_file.exists():
                current_backup = self.storage_file.with_suffix('.pre_restore.bak')
                shutil.copy2(self.storage_file, current_backup)
            
            # Restore from backup
            shutil.copy2(backup_file, self.storage_file)
            os.chmod(self.storage_file, self.config.file_permissions)
            
            logger.info("Restored credentials from backup", backup_path=backup_path)
            
            return Success(True)
            
        except Exception as e:
            logger.error("Failed to restore from backup", error=str(e))
            return Failure(f"Restore failed: {str(e)}")


class ChromaDBCredentialStorage(CredentialStorageBackend):
    """ChromaDB-based credential storage backend."""
    
    def __init__(self, config: StorageConfig):
        """Initialize ChromaDB storage backend."""
        self.config = config
        self.persist_dir = Path(config.chromadb_path).expanduser()
        self.collection_name = config.chromadb_collection
        
        # Ensure directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.persist_dir, config.directory_permissions)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except ValueError:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Encrypted credentials storage"}
            )
        
        logger.info(
            "Initialized ChromaDB credential storage",
            persist_dir=str(self.persist_dir),
            collection=self.collection_name
        )
    
    async def store_credential(
        self,
        credential_id: str,
        encrypted_credential: EncryptedCredential,
        metadata: CredentialMetadata
    ) -> Result[bool, str]:
        """Store an encrypted credential."""
        try:
            # Prepare data for ChromaDB
            documents = [json.dumps(encrypted_credential.to_dict())]
            metadatas = [{
                "user_id": metadata.user_id,
                "provider_type": metadata.provider_type,
                "created_at": metadata.created_at.isoformat(),
                "last_accessed": metadata.last_accessed.isoformat() if metadata.last_accessed else None,
                "access_count": metadata.access_count,
                "is_active": metadata.is_active,
                "tags": ",".join(metadata.tags)
            }]
            ids = [credential_id]
            
            # Check if credential already exists
            try:
                existing = self.collection.get(ids=[credential_id])
                if existing['ids']:
                    # Update existing
                    self.collection.update(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                    )
                    logger.debug("Updated existing credential", credential_id=credential_id)
                else:
                    raise ValueError("Not found")  # Will fall through to add
            except ValueError:
                # Add new credential
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                logger.info("Added new credential", credential_id=credential_id)
            
            return Success(True)
            
        except Exception as e:
            logger.error("Failed to store credential", error=str(e))
            return Failure(f"ChromaDB storage failed: {str(e)}")
    
    async def retrieve_credential(self, credential_id: str) -> Result[EncryptedCredential, str]:
        """Retrieve an encrypted credential."""
        try:
            result = self.collection.get(
                ids=[credential_id],
                include=["documents", "metadatas"]
            )
            
            if not result['ids']:
                return Failure(f"Credential {credential_id} not found")
            
            # Parse encrypted credential
            cred_data = json.loads(result['documents'][0])
            encrypted_credential = EncryptedCredential.from_dict(cred_data)
            
            # Update access metadata
            metadata = result['metadatas'][0]
            metadata['last_accessed'] = datetime.utcnow().isoformat()
            metadata['access_count'] = int(metadata.get('access_count', 0)) + 1
            
            # Update in database
            self.collection.update(
                ids=[credential_id],
                metadatas=[metadata]
            )
            
            logger.debug("Retrieved credential", credential_id=credential_id)
            
            return Success(encrypted_credential)
            
        except Exception as e:
            logger.error("Failed to retrieve credential", error=str(e))
            return Failure(f"ChromaDB retrieval failed: {str(e)}")
    
    async def delete_credential(self, credential_id: str) -> Result[bool, str]:
        """Delete a credential."""
        try:
            # Check if exists
            result = self.collection.get(ids=[credential_id])
            if not result['ids']:
                return Failure(f"Credential {credential_id} not found")
            
            # Delete from collection
            self.collection.delete(ids=[credential_id])
            
            logger.info("Deleted credential", credential_id=credential_id)
            
            return Success(True)
            
        except Exception as e:
            logger.error("Failed to delete credential", error=str(e))
            return Failure(f"ChromaDB deletion failed: {str(e)}")
    
    async def list_credentials(
        self,
        user_id: Optional[str] = None,
        provider_type: Optional[str] = None
    ) -> Result[List[CredentialMetadata], str]:
        """List credentials with optional filtering."""
        try:
            # Build where clause for filtering
            where_clause = {}
            if user_id:
                where_clause["user_id"] = user_id
            if provider_type:
                where_clause["provider_type"] = provider_type
            
            # Get matching credentials
            result = self.collection.get(
                where=where_clause if where_clause else None,
                include=["metadatas"]
            )
            
            credentials = []
            for i, cred_id in enumerate(result['ids']):
                metadata_dict = result['metadatas'][i]
                
                # Convert metadata back to CredentialMetadata
                metadata = CredentialMetadata(
                    credential_id=cred_id,
                    user_id=metadata_dict['user_id'],
                    provider_type=metadata_dict['provider_type'],
                    created_at=datetime.fromisoformat(metadata_dict['created_at']),
                    last_accessed=datetime.fromisoformat(metadata_dict['last_accessed']) if metadata_dict.get('last_accessed') else None,
                    access_count=int(metadata_dict.get('access_count', 0)),
                    is_active=bool(metadata_dict.get('is_active', True)),
                    tags=set(metadata_dict.get('tags', '').split(',')) if metadata_dict.get('tags') else set()
                )
                
                credentials.append(metadata)
            
            return Success(credentials)
            
        except Exception as e:
            logger.error("Failed to list credentials", error=str(e))
            return Failure(f"ChromaDB listing failed: {str(e)}")
    
    async def update_metadata(
        self,
        credential_id: str,
        metadata: CredentialMetadata
    ) -> Result[bool, str]:
        """Update credential metadata."""
        try:
            # Check if credential exists
            result = self.collection.get(ids=[credential_id])
            if not result['ids']:
                return Failure(f"Credential {credential_id} not found")
            
            # Update metadata
            metadata_dict = {
                "user_id": metadata.user_id,
                "provider_type": metadata.provider_type,
                "created_at": metadata.created_at.isoformat(),
                "last_accessed": metadata.last_accessed.isoformat() if metadata.last_accessed else None,
                "access_count": metadata.access_count,
                "is_active": metadata.is_active,
                "tags": ",".join(metadata.tags)
            }
            
            self.collection.update(
                ids=[credential_id],
                metadatas=[metadata_dict]
            )
            
            logger.debug("Updated credential metadata", credential_id=credential_id)
            
            return Success(True)
            
        except Exception as e:
            logger.error("Failed to update metadata", error=str(e))
            return Failure(f"ChromaDB update failed: {str(e)}")
    
    async def backup_data(self, backup_path: Optional[str] = None) -> Result[str, str]:
        """Create backup of credential data."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            if backup_path:
                backup_file = Path(backup_path)
            else:
                backup_dir = self.persist_dir / "backups"
                backup_dir.mkdir(exist_ok=True)
                backup_file = backup_dir / f"chromadb_backup_{timestamp}.json"
            
            # Get all data from collection
            result = self.collection.get(include=["documents", "metadatas"])
            
            backup_data = {
                "version": "1.0",
                "timestamp": timestamp,
                "collection_name": self.collection_name,
                "data": {
                    "ids": result['ids'],
                    "documents": result['documents'],
                    "metadatas": result['metadatas']
                }
            }
            
            # Write backup file
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            os.chmod(backup_file, self.config.file_permissions)
            
            logger.info("Created ChromaDB backup", backup_path=str(backup_file))
            
            return Success(str(backup_file))
            
        except Exception as e:
            logger.error("Failed to create backup", error=str(e))
            return Failure(f"ChromaDB backup failed: {str(e)}")
    
    async def restore_data(self, backup_path: str) -> Result[bool, str]:
        """Restore credential data from backup."""
        try:
            backup_file = Path(backup_path)
            
            if not backup_file.exists():
                return Failure(f"Backup file not found: {backup_path}")
            
            # Load backup data
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # Validate backup structure
            if not all(key in backup_data for key in ["version", "data"]):
                return Failure("Invalid backup file format")
            
            data = backup_data["data"]
            if not all(key in data for key in ["ids", "documents", "metadatas"]):
                return Failure("Invalid backup data structure")
            
            # Clear existing collection (after creating backup)
            # Note: ChromaDB doesn't have a clear method, so we delete all items
            try:
                existing = self.collection.get()
                if existing['ids']:
                    self.collection.delete(ids=existing['ids'])
            except Exception:
                pass  # Collection might be empty
            
            # Restore data
            if data['ids']:
                self.collection.add(
                    ids=data['ids'],
                    documents=data['documents'],
                    metadatas=data['metadatas']
                )
            
            logger.info("Restored ChromaDB from backup", backup_path=backup_path)
            
            return Success(True)
            
        except Exception as e:
            logger.error("Failed to restore from backup", error=str(e))
            return Failure(f"ChromaDB restore failed: {str(e)}")


class CredentialStorageManager:
    """Manager for credential storage with multiple backend support."""
    
    def __init__(
        self,
        backend_type: str = "json",
        config: Optional[StorageConfig] = None
    ):
        """Initialize storage manager."""
        self.config = config or StorageConfig()
        self.backend_type = backend_type
        
        # Initialize backend
        if backend_type == "json":
            self.backend = JSONCredentialStorage(self.config)
        elif backend_type == "chromadb":
            self.backend = ChromaDBCredentialStorage(self.config)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
        
        logger.info(
            "Initialized credential storage manager",
            backend=backend_type,
            config=str(self.config)
        )
    
    async def store_credential(
        self,
        credential_id: str,
        encrypted_credential: EncryptedCredential,
        metadata: CredentialMetadata
    ) -> Result[bool, str]:
        """Store an encrypted credential."""
        return await self.backend.store_credential(credential_id, encrypted_credential, metadata)
    
    async def retrieve_credential(self, credential_id: str) -> Result[EncryptedCredential, str]:
        """Retrieve an encrypted credential."""
        return await self.backend.retrieve_credential(credential_id)
    
    async def delete_credential(self, credential_id: str) -> Result[bool, str]:
        """Delete a credential."""
        return await self.backend.delete_credential(credential_id)
    
    async def list_credentials(
        self,
        user_id: Optional[str] = None,
        provider_type: Optional[str] = None
    ) -> Result[List[CredentialMetadata], str]:
        """List credentials with optional filtering."""
        return await self.backend.list_credentials(user_id, provider_type)
    
    async def update_metadata(
        self,
        credential_id: str,
        metadata: CredentialMetadata
    ) -> Result[bool, str]:
        """Update credential metadata."""
        return await self.backend.update_metadata(credential_id, metadata)
    
    async def backup_data(self, backup_path: Optional[str] = None) -> Result[str, str]:
        """Create backup of credential data."""
        return await self.backend.backup_data(backup_path)
    
    async def restore_data(self, backup_path: str) -> Result[bool, str]:
        """Restore credential data from backup."""
        return await self.backend.restore_data(backup_path)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            if self.backend_type == "json":
                json_backend = self.backend
                storage_file = json_backend.storage_file
                
                if storage_file.exists():
                    stat = storage_file.stat()
                    return {
                        "backend_type": self.backend_type,
                        "storage_file": str(storage_file),
                        "file_size_bytes": stat.st_size,
                        "last_modified": datetime.fromtimestamp(stat.st_mtime),
                        "backup_count": len(list(json_backend.backup_dir.glob("*.json")))
                    }
                else:
                    return {
                        "backend_type": self.backend_type,
                        "storage_file": str(storage_file),
                        "exists": False
                    }
            
            elif self.backend_type == "chromadb":
                chromadb_backend = self.backend
                collection = chromadb_backend.collection
                
                # Get collection stats
                result = collection.get()
                count = len(result['ids']) if result['ids'] else 0
                
                return {
                    "backend_type": self.backend_type,
                    "persist_dir": str(chromadb_backend.persist_dir),
                    "collection_name": chromadb_backend.collection_name,
                    "credential_count": count
                }
                
        except Exception as e:
            logger.error("Failed to get storage stats", error=str(e))
            return {
                "backend_type": self.backend_type,
                "error": str(e)
            }