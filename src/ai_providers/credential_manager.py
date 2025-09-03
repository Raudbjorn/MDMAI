"""
Secure credential management for MDMAI TTRPG Assistant.

This module provides secure storage and management of API keys using AES-256 encryption
with local filesystem storage. No external databases are required.
"""

from cryptography.fernet import Fernet, InvalidToken
import os
import logging
import json
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class CredentialManager:
    """
    Manages secure storage and retrieval of API credentials.
    
    Uses AES-256 encryption via the cryptography library's Fernet implementation
    for secure local storage of API keys. Each user's credentials are encrypted
    with a namespace prefix to prevent cross-user access.
    """
    
    def __init__(self, storage_path: str = "./data/credentials"):
        """
        Initialize the credential manager.
        
        Args:
            storage_path: Path to store encrypted credentials
            
        Raises:
            ValueError: If MDMAI_ENCRYPTION_KEY is not set or invalid
        """
        # The encryption key MUST be loaded from a secure, persistent source.
        # Generating a key on the fly is insecure as it makes previously encrypted data unrecoverable on restart.
        key = os.environ.get('MDMAI_ENCRYPTION_KEY')
        if not key:
            raise ValueError(
                "MDMAI_ENCRYPTION_KEY environment variable must be set for secure operation. "
                "Generate a valid key using: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode('utf-8'))\""
            )
        
        # Validate the key format before creating Fernet cipher
        # Fernet requires a 32-byte URL-safe base64-encoded key
        try:
            # Ensure the key is properly encoded
            if isinstance(key, str):
                key_bytes = key.encode('utf-8')
            else:
                key_bytes = key
                
            # Validate key format by attempting to create Fernet instance
            # This will raise ValueError if the key format is invalid
            self.cipher = Fernet(key_bytes)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid MDMAI_ENCRYPTION_KEY format. Must be a 32-byte URL-safe base64-encoded key. "
                f"Generate a valid key using: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode('utf-8'))\". "
                f"Error: {e}"
            )
        
        # Use local filesystem for credential storage - no external databases
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.credentials_file = self.storage_path / "encrypted_credentials.json"
        
        # Load existing credentials from local storage
        self.stored_credentials = self._load_credentials()
        
        logger.info(f"CredentialManager initialized with storage at {self.storage_path}")
    
    def encrypt_api_key(self, api_key: str, user_id: str) -> str:
        """
        Encrypt API key for secure storage.
        
        Args:
            api_key: The API key to encrypt
            user_id: User identifier for namespacing
            
        Returns:
            str: Base64-encoded encrypted API key
            
        Raises:
            ValueError: If api_key or user_id is empty
        """
        if not api_key or not user_id:
            raise ValueError("Both api_key and user_id must be provided")
        
        # Add user-specific identifier (not a cryptographic salt, but user context for namespacing)
        # This concatenates user_id with the API key to ensure keys are unique per user
        salted_key = f"{user_id}:{api_key}"
        
        try:
            encrypted = self.cipher.encrypt(salted_key.encode()).decode()
            
            # Store encrypted key locally
            self.stored_credentials[user_id] = encrypted
            self._save_credentials()
            
            logger.info(f"API key encrypted and stored for user {user_id}")
            return encrypted
            
        except Exception as e:
            logger.error(f"Failed to encrypt API key for user {user_id}: {e}")
            raise
    
    def decrypt_api_key(self, encrypted_key: str, user_id: str) -> Optional[str]:
        """
        Decrypt API key for use.
        
        Args:
            encrypted_key: The encrypted API key (optional if stored)
            user_id: User identifier for validation
            
        Returns:
            Optional[str]: The decrypted API key, or None if decryption fails
        """
        if not user_id:
            logger.warning("User ID is required for API key decryption")
            return None
        
        try:
            # Try to use provided key or fall back to stored credential
            key_to_decrypt = encrypted_key
            if not key_to_decrypt and user_id in self.stored_credentials:
                key_to_decrypt = self.stored_credentials[user_id]
            
            if not key_to_decrypt:
                logger.warning(f"No encrypted API key found for user {user_id}")
                return None
            
            # Decrypt and validate
            decrypted = self.cipher.decrypt(key_to_decrypt.encode()).decode()
            stored_id, api_key = decrypted.split(':', 1)
            
            if stored_id == user_id:
                logger.debug(f"API key successfully decrypted for user {user_id}")
                return api_key
            else:
                logger.warning(f"User ID mismatch in encrypted key for {user_id}")
                return None
                
        except InvalidToken:
            logger.warning(f"API key decryption failed for user {user_id}: Invalid token")
        except TypeError as e:
            logger.warning(f"API key decryption failed for user {user_id}: Type error - {type(e).__name__}")
        except ValueError as e:
            logger.warning(f"API key decryption failed for user {user_id}: Value error - {str(e)[:100]}")
        except Exception as e:
            logger.error(f"Unexpected error during decryption for user {user_id}: {e}")
        
        return None
    
    def delete_credentials(self, user_id: str) -> bool:
        """
        Delete stored credentials for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            bool: True if credentials were deleted, False if not found
        """
        if user_id in self.stored_credentials:
            del self.stored_credentials[user_id]
            self._save_credentials()
            logger.info(f"Credentials deleted for user {user_id}")
            return True
        
        logger.warning(f"No credentials found to delete for user {user_id}")
        return False
    
    def has_credentials(self, user_id: str) -> bool:
        """
        Check if credentials exist for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            bool: True if credentials exist
        """
        return user_id in self.stored_credentials
    
    def list_users_with_credentials(self) -> list[str]:
        """
        Get list of users with stored credentials.
        
        Returns:
            list[str]: List of user IDs with credentials
        """
        return list(self.stored_credentials.keys())
    
    def rotate_encryption_key(self, new_key: str) -> bool:
        """
        Rotate the encryption key while preserving existing data.
        
        Args:
            new_key: New encryption key
            
        Returns:
            bool: True if rotation was successful
            
        Raises:
            ValueError: If new key is invalid
        """
        # Validate new key format
        try:
            new_cipher = Fernet(new_key.encode('utf-8'))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid new encryption key format: {e}")
        
        # Decrypt all existing credentials with old key and re-encrypt with new key
        try:
            decrypted_data = {}
            
            # Decrypt all existing data
            for user_id, encrypted_key in self.stored_credentials.items():
                decrypted = self.cipher.decrypt(encrypted_key.encode()).decode()
                stored_id, api_key = decrypted.split(':', 1)
                if stored_id == user_id:
                    decrypted_data[user_id] = api_key
            
            # Re-encrypt with new key
            self.cipher = new_cipher
            new_credentials = {}
            
            for user_id, api_key in decrypted_data.items():
                salted_key = f"{user_id}:{api_key}"
                encrypted = self.cipher.encrypt(salted_key.encode()).decode()
                new_credentials[user_id] = encrypted
            
            # Update stored credentials and save
            self.stored_credentials = new_credentials
            self._save_credentials()
            
            logger.info(f"Successfully rotated encryption key for {len(new_credentials)} users")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rotate encryption key: {e}")
            return False
    
    def _load_credentials(self) -> Dict[str, str]:
        """
        Load encrypted credentials from local JSON file.
        
        Returns:
            Dict[str, str]: Dictionary of user_id -> encrypted_key
        """
        if self.credentials_file.exists():
            try:
                with open(self.credentials_file, 'r') as f:
                    data = json.load(f)
                    logger.debug(f"Loaded credentials for {len(data)} users")
                    return data
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load credentials file: {e}")
                # Create backup of corrupted file
                backup_file = self.credentials_file.with_suffix('.json.backup')
                try:
                    self.credentials_file.rename(backup_file)
                    logger.info(f"Corrupted credentials file backed up to {backup_file}")
                except Exception:
                    pass
        
        return {}
    
    def _save_credentials(self) -> None:
        """
        Save encrypted credentials to local JSON file.
        
        Raises:
            IOError: If file cannot be written
        """
        try:
            # Create temporary file first for atomic write
            temp_file = self.credentials_file.with_suffix('.json.tmp')
            
            with open(temp_file, 'w') as f:
                json.dump(self.stored_credentials, f, indent=2)
            
            # Atomic move to final location
            temp_file.replace(self.credentials_file)
            
            logger.debug(f"Saved credentials for {len(self.stored_credentials)} users")
            
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            # Clean up temp file if it exists
            temp_file = self.credentials_file.with_suffix('.json.tmp')
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            raise
    
    def backup_credentials(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the credentials file.
        
        Args:
            backup_path: Optional custom backup path
            
        Returns:
            str: Path to the backup file
        """
        if backup_path:
            backup_file = Path(backup_path)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.credentials_file.with_name(f"credentials_backup_{timestamp}.json")
        
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        
        if self.credentials_file.exists():
            import shutil
            shutil.copy2(self.credentials_file, backup_file)
            logger.info(f"Credentials backed up to {backup_file}")
        else:
            # Create empty backup file
            backup_file.write_text('{}')
            logger.info(f"Empty credentials backup created at {backup_file}")
        
        return str(backup_file)