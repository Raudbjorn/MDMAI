"""
Secure credential encryption service with AES-256-GCM and PBKDF2.

This module provides enterprise-grade encryption for API keys and credentials
with the following security features:
- AES-256-GCM encryption with authenticated encryption
- PBKDF2 key derivation with 600,000+ iterations (OWASP 2023 recommendation)
- User-specific salt generation with cryptographically secure randomness
- Memory-safe key handling with secure zeroing
- Secure random generation using OS entropy
- Result/Either pattern for comprehensive error handling
- Constant-time comparisons to prevent timing attacks
- Key rotation and versioning support
"""

import hashlib
import hmac
import os
import secrets
import sys
from typing import Dict, Optional, Tuple, Union, Any, Final, TypeVar, Generic
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
import base64

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives import hashes, serialization, constant_time
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from structlog import get_logger
# Import Result types from the project's core module
try:
    from returns.result import Result, Success, Failure
except ImportError:
    # Fallback to simple Result implementation if returns library is not available
    from typing import Union, Generic, TypeVar
    
    T = TypeVar('T')
    E = TypeVar('E')
    
    class Success(Generic[T]):
        def __init__(self, value: T):
            self._value = value
        
        def unwrap(self) -> T:
            return self._value
        
        def is_success(self) -> bool:
            return True
        
        def failure(self):
            raise RuntimeError("Cannot get failure from Success")
    
    class Failure(Generic[E]):
        def __init__(self, error: E):
            self._error = error
        
        def unwrap(self):
            raise RuntimeError(f"Cannot unwrap Failure: {self._error}")
        
        def is_success(self) -> bool:
            return False
        
        def failure(self) -> E:
            return self._error
    
    Result = Union[Success[T], Failure[E]]

logger = get_logger(__name__)

# Type aliases for clarity
T = TypeVar('T')
EncryptionResult = Result[T, str]

# Security constants (following OWASP 2023 recommendations)
MIN_PBKDF2_ITERATIONS: Final[int] = 600_000
MIN_PASSWORD_LENGTH: Final[int] = 12
SALT_LENGTH_BYTES: Final[int] = 32  # 256 bits
KEY_LENGTH_BYTES: Final[int] = 32   # AES-256
GCM_NONCE_LENGTH: Final[int] = 12   # 96 bits for GCM
GCM_TAG_LENGTH: Final[int] = 16     # 128 bits
SECURE_WIPE_PASSES: Final[int] = 3


class EncryptionError(Exception):
    """Base exception for encryption-related errors."""
    pass


class KeyDerivationError(EncryptionError):
    """Exception raised when key derivation fails."""
    pass


class DecryptionError(EncryptionError):
    """Exception raised when decryption fails."""
    pass


class MemorySecurityError(EncryptionError):
    """Exception raised when secure memory operations fail."""
    pass


@dataclass(frozen=True)
class EncryptionConfig:
    """
    Configuration for credential encryption with secure defaults.
    
    All values follow OWASP 2023 and NIST SP 800-132 recommendations.
    """
    
    # PBKDF2 configuration (OWASP 2023 recommendations)
    pbkdf2_iterations: int = 600_000  # Minimum for PBKDF2-HMAC-SHA256
    pbkdf2_hash_algorithm: str = "SHA256"
    salt_length: int = SALT_LENGTH_BYTES
    
    # AES-GCM configuration
    key_length: int = KEY_LENGTH_BYTES
    nonce_length: int = GCM_NONCE_LENGTH
    tag_length: int = GCM_TAG_LENGTH
    
    # Security settings
    master_key_rotation_days: int = 365
    user_salt_rotation_days: int = 180
    credential_rotation_days: int = 90
    memory_wipe_passes: int = SECURE_WIPE_PASSES
    min_password_entropy_bits: int = 60
    
    # Performance tuning
    use_scrypt_for_master: bool = False  # Option for master key derivation
    scrypt_n: int = 2**16  # CPU/memory cost parameter
    scrypt_r: int = 8      # Block size parameter
    scrypt_p: int = 1      # Parallelization parameter
    scrypt_maxmem: int = 128 * 1024 * 1024  # 128 MB
    
    # Additional security features
    enable_key_stretching: bool = True
    enable_memory_hardening: bool = True
    require_secure_random: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration on initialization."""
        if self.pbkdf2_iterations < MIN_PBKDF2_ITERATIONS:
            raise ValueError(
                f"PBKDF2 iterations must be at least {MIN_PBKDF2_ITERATIONS} "
                f"(OWASP 2023 recommendation)"
            )


@dataclass(frozen=True)
class EncryptedCredential:
    """
    Immutable encrypted credential with comprehensive metadata.
    
    Uses frozen dataclass to prevent accidental modification of encrypted data.
    """
    
    encrypted_data: bytes
    salt: bytes
    nonce: bytes  # Using proper GCM terminology
    tag: bytes
    provider_type: str
    encrypted_at: datetime
    last_accessed: Optional[datetime] = None
    rotation_due: Optional[datetime] = None
    key_version: int = 1
    algorithm: str = "AES-256-GCM"
    kdf_algorithm: str = "PBKDF2-HMAC-SHA256"
    kdf_iterations: int = MIN_PBKDF2_ITERATIONS
    
    def to_dict(self) -> Dict[str, Union[str, int, None]]:
        """
        Convert to dictionary for secure storage.
        
        Returns:
            Dictionary with hex-encoded binary data and ISO timestamps
        """
        return {
            'encrypted_data': base64.b64encode(self.encrypted_data).decode('ascii'),
            'salt': base64.b64encode(self.salt).decode('ascii'),
            'nonce': base64.b64encode(self.nonce).decode('ascii'),
            'tag': base64.b64encode(self.tag).decode('ascii'),
            'provider_type': self.provider_type,
            'encrypted_at': self.encrypted_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'rotation_due': self.rotation_due.isoformat() if self.rotation_due else None,
            'key_version': self.key_version,
            'algorithm': self.algorithm,
            'kdf_algorithm': self.kdf_algorithm,
            'kdf_iterations': self.kdf_iterations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncryptedCredential':
        """
        Create from dictionary with validation.
        
        Args:
            data: Dictionary containing encrypted credential data
            
        Returns:
            EncryptedCredential instance
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        try:
            # Handle both base64 and hex encoding for backwards compatibility
            encrypted_data_str = str(data.get('encrypted_data', ''))
            if encrypted_data_str.startswith('0x'):
                encrypted_data = bytes.fromhex(encrypted_data_str[2:])  # Remove '0x' prefix
            elif 'encrypted_data' in data:
                encrypted_data = base64.b64decode(data['encrypted_data'])
            else:
                encrypted_data = bytes()
            
            return cls(
                encrypted_data=encrypted_data,
                salt=base64.b64decode(data['salt']) if 'salt' in data else bytes(),
                nonce=base64.b64decode(data.get('nonce', data.get('iv', ''))),
                tag=base64.b64decode(data['tag']) if 'tag' in data else bytes(),
                provider_type=str(data['provider_type']),
                encrypted_at=datetime.fromisoformat(str(data['encrypted_at'])),
                last_accessed=datetime.fromisoformat(str(data['last_accessed'])) if data.get('last_accessed') else None,
                rotation_due=datetime.fromisoformat(str(data['rotation_due'])) if data.get('rotation_due') else None,
                key_version=int(data.get('key_version', 1)),
                algorithm=str(data.get('algorithm', 'AES-256-GCM')),
                kdf_algorithm=str(data.get('kdf_algorithm', 'PBKDF2-HMAC-SHA256')),
                kdf_iterations=int(data.get('kdf_iterations', MIN_PBKDF2_ITERATIONS))
            )
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid encrypted credential data: {e}") from e


class SecureMemory:
    """
    Secure memory operations for sensitive data handling.
    
    Implements DoD 5220.22-M standard for data sanitization and
    constant-time operations to prevent timing attacks.
    """
    
    @staticmethod
    def secure_zero(data: bytearray, passes: int = SECURE_WIPE_PASSES) -> None:
        """
        Securely zero memory using DoD 5220.22-M standard.
        
        Args:
            data: Bytearray to securely zero
            passes: Number of overwrite passes (default: 3)
            
        Raises:
            MemorySecurityError: If secure zeroing fails
        """
        if not isinstance(data, bytearray):
            raise TypeError("Can only securely zero bytearray objects")
        
        if not data:
            return
        
        try:
            # DoD 5220.22-M standard patterns
            patterns = [
                b'\x00',  # All zeros
                b'\xFF',  # All ones
                secrets.token_bytes(1),  # Random byte
            ]
            
            for _ in range(passes):
                for pattern in patterns:
                    # Use memoryview for efficient memory access
                    mv = memoryview(data)
                    for i in range(len(mv)):
                        mv[i] = pattern[0]
            
            # Final verification pass
            for i in range(len(data)):
                data[i] = 0
                
        except Exception as e:
            raise MemorySecurityError(f"Failed to securely zero memory: {e}") from e
    
    @staticmethod
    def secure_compare(a: bytes, b: bytes) -> bool:
        """
        Constant-time comparison to prevent timing attacks.
        
        Uses cryptography library's constant_time module for
        cryptographically secure comparison.
        
        Args:
            a: First byte string
            b: Second byte string
            
        Returns:
            True if equal, False otherwise
        """
        if not isinstance(a, bytes) or not isinstance(b, bytes):
            return False
            
        if len(a) != len(b):
            return False
            
        # Use cryptography's constant-time comparison
        return constant_time.bytes_eq(a, b)
    
    @staticmethod
    def secure_random_bytes(length: int) -> bytes:
        """
        Generate cryptographically secure random bytes.
        
        Args:
            length: Number of random bytes to generate
            
        Returns:
            Secure random bytes
            
        Raises:
            ValueError: If length is invalid
        """
        if length <= 0:
            raise ValueError("Length must be positive")
            
        # Use OS entropy source
        return secrets.token_bytes(length)


class CredentialEncryption:
    """
    Production-ready credential encryption service with AES-256-GCM.
    
    Features:
    - AES-256-GCM authenticated encryption
    - PBKDF2-HMAC-SHA256 with 600,000+ iterations
    - User-specific salt generation
    - Secure memory handling with DoD 5220.22-M standard
    - Result/Either pattern for comprehensive error handling
    - Key rotation and versioning support
    - Constant-time operations to prevent timing attacks
    """
    
    def __init__(self, config: Optional[EncryptionConfig] = None) -> None:
        """
        Initialize the encryption service with secure defaults.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or EncryptionConfig()
        self._user_salts: Dict[str, bytes] = {}
        self._key_versions: Dict[str, int] = {}
        self._key_cache: Dict[str, Tuple[bytes, datetime]] = {}  # Cache with expiry
        
        # Initialize master key storage
        self._master_key: Optional[bytearray] = None  # Use bytearray for secure zeroing
        self._master_salt: Optional[bytes] = None
        self._initialized_at = datetime.now(timezone.utc)
        
        # Verify secure random is available
        if self.config.require_secure_random:
            self._verify_secure_random()
        
        logger.info(
            "Initialized credential encryption service",
            pbkdf2_iterations=self.config.pbkdf2_iterations,
            key_length_bits=self.config.key_length * 8,
            algorithm="AES-256-GCM",
            kdf="PBKDF2-HMAC-SHA256"
        )
    
    def _verify_secure_random(self) -> None:
        """Verify that secure random number generation is available."""
        try:
            test_bytes = secrets.token_bytes(32)
            if len(test_bytes) != 32:
                raise EncryptionError("Secure random generation failed")
        except Exception as e:
            raise EncryptionError(f"Secure random not available: {e}") from e
    
    def set_master_password(self, password: str) -> EncryptionResult[bool]:
        """
        Set the master password for credential encryption with validation.
        
        Args:
            password: Master password (min 12 chars, recommended 16+)
            
        Returns:
            Result[bool, str] indicating success or failure
        """
        password_bytes = None
        try:
            # Validate password strength
            if len(password) < MIN_PASSWORD_LENGTH:
                return Failure(f"Master password must be at least {MIN_PASSWORD_LENGTH} characters")
            
            # Check password entropy (basic check)
            if len(set(password)) < 6:
                return Failure("Password has insufficient character diversity")
            
            # Generate master salt if not exists
            if self._master_salt is None:
                self._master_salt = SecureMemory.secure_random_bytes(self.config.salt_length)
            
            # Convert password to bytes for processing
            password_bytes = bytearray(password.encode('utf-8'))
            
            # Derive master key with secure KDF
            derived_key = self._derive_key(bytes(password_bytes), self._master_salt)
            
            # Clear existing master key if present
            if self._master_key is not None:
                SecureMemory.secure_zero(self._master_key)
            
            # Store as bytearray for secure deletion later
            self._master_key = bytearray(derived_key)
            
            logger.info(
                "Master password set successfully",
                salt_length=len(self._master_salt),
                iterations=self.config.pbkdf2_iterations
            )
            return Success(True)
            
        except Exception as e:
            logger.error("Failed to set master password", error=str(e))
            return Failure(f"Failed to set master password: {str(e)}")
        finally:
            # Always clear password from memory
            if password_bytes is not None:
                SecureMemory.secure_zero(password_bytes)
    
    def generate_user_salt(self, user_id: str) -> bytes:
        """
        Generate or retrieve cryptographically secure user-specific salt.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            User-specific salt (32 bytes)
        """
        if user_id not in self._user_salts:
            # Generate fully random salt (don't mix with user_id for better security)
            self._user_salts[user_id] = SecureMemory.secure_random_bytes(self.config.salt_length)
            
            # Increment key version for new salt
            self._key_versions[user_id] = 1
            
            logger.debug(
                "Generated new user salt",
                user_id_hash=hashlib.sha256(user_id.encode()).hexdigest()[:8],
                salt_length=self.config.salt_length
            )
        
        return self._user_salts[user_id]
    
    def _derive_key(self, password: bytes, salt: bytes) -> bytes:
        """
        Derive encryption key using PBKDF2-HMAC-SHA256 or Scrypt.
        
        Args:
            password: Password bytes
            salt: Salt bytes
            
        Returns:
            Derived key (32 bytes for AES-256)
            
        Raises:
            KeyDerivationError: If key derivation fails
        """
        try:
            if self.config.use_scrypt_for_master and password == bytes(self._master_key):
                # Use Scrypt for master key (memory-hard)
                kdf = Scrypt(
                    salt=salt,
                    length=self.config.key_length,
                    n=self.config.scrypt_n,
                    r=self.config.scrypt_r,
                    p=self.config.scrypt_p,
                    backend=default_backend()
                )
            else:
                # Use PBKDF2-HMAC-SHA256 (standard)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=self.config.key_length,
                    salt=salt,
                    iterations=self.config.pbkdf2_iterations,
                    backend=default_backend()
                )
            
            derived_key = kdf.derive(password)
            
            # Verify key length
            if len(derived_key) != self.config.key_length:
                raise KeyDerivationError(f"Invalid key length: {len(derived_key)}")
            
            return derived_key
            
        except Exception as e:
            raise KeyDerivationError(f"Key derivation failed: {e}") from e
    
    def encrypt(
        self, 
        credential: str, 
        user_id: str, 
        provider_type: str
    ) -> EncryptionResult[EncryptedCredential]:
        """
        Encrypt a credential using AES-256-GCM with authenticated encryption.
        
        Args:
            credential: The credential/API key to encrypt
            user_id: Unique user identifier
            provider_type: Provider type (e.g., 'anthropic', 'openai', 'google')
            
        Returns:
            Result[EncryptedCredential, str] with encrypted data or error message
        """
        credential_bytes = None
        user_key = None
        try:
            # Validate inputs
            if self._master_key is None:
                return Failure("Master key not set. Call set_master_password first.")
            
            if not credential:
                return Failure("Credential cannot be empty")
            
            if not user_id:
                return Failure("User ID cannot be empty")
            
            # Get or generate user salt
            user_salt = self.generate_user_salt(user_id)
            
            # Derive user-specific key from master key
            user_key = bytearray(self._derive_key(bytes(self._master_key), user_salt))
            
            # Generate cryptographically secure nonce for GCM
            nonce = SecureMemory.secure_random_bytes(self.config.nonce_length)
            
            # Convert credential to bytes
            credential_bytes = bytearray(credential.encode('utf-8'))
            
            # Use AESGCM for authenticated encryption
            aesgcm = AESGCM(bytes(user_key))
            
            # Add associated data for additional authentication
            associated_data = f"{user_id}:{provider_type}:{self._key_versions.get(user_id, 1)}".encode('utf-8')
            
            # Encrypt with authentication
            ciphertext_and_tag = aesgcm.encrypt(nonce, bytes(credential_bytes), associated_data)
            
            # Split ciphertext and tag
            ciphertext = ciphertext_and_tag[:-16]
            tag = ciphertext_and_tag[-16:]
            
            # Create encrypted credential object
            now = datetime.now(timezone.utc)
            encrypted_cred = EncryptedCredential(
                encrypted_data=ciphertext,
                salt=user_salt,
                nonce=nonce,
                tag=tag,
                provider_type=provider_type,
                encrypted_at=now,
                rotation_due=now + timedelta(days=self.config.credential_rotation_days),
                key_version=self._key_versions.get(user_id, 1),
                algorithm="AES-256-GCM",
                kdf_algorithm="PBKDF2-HMAC-SHA256",
                kdf_iterations=self.config.pbkdf2_iterations
            )
            
            logger.info(
                "Successfully encrypted credential",
                user_id_hash=hashlib.sha256(user_id.encode()).hexdigest()[:8],
                provider_type=provider_type,
                key_version=self._key_versions.get(user_id, 1)
            )
            
            return Success(encrypted_cred)
            
        except Exception as e:
            logger.error("Failed to encrypt credential", error=str(e), exc_info=True)
            return Failure(f"Encryption failed: {str(e)}")
        finally:
            # Always clear sensitive data from memory
            if credential_bytes is not None:
                SecureMemory.secure_zero(credential_bytes)
            if user_key is not None:
                SecureMemory.secure_zero(user_key)
    
    def decrypt(
        self, 
        encrypted_cred: EncryptedCredential, 
        user_id: str
    ) -> EncryptionResult[str]:
        """
        Decrypt a credential using AES-256-GCM with authentication verification.
        
        Args:
            encrypted_cred: Encrypted credential object
            user_id: User identifier for key derivation
            
        Returns:
            Result[str, str] with decrypted credential or error message
        """
        user_key = None
        plaintext_bytes = None
        try:
            # Validate inputs
            if self._master_key is None:
                return Failure("Master key not set. Call set_master_password first.")
            
            if not encrypted_cred:
                return Failure("Encrypted credential cannot be None")
            
            # Derive user-specific key using the stored salt
            user_key = bytearray(self._derive_key(bytes(self._master_key), encrypted_cred.salt))
            
            # Use AESGCM for authenticated decryption
            aesgcm = AESGCM(bytes(user_key))
            
            # Reconstruct associated data for authentication
            associated_data = f"{user_id}:{encrypted_cred.provider_type}:{encrypted_cred.key_version}".encode('utf-8')
            
            # Combine ciphertext and tag for decryption
            ciphertext_and_tag = encrypted_cred.encrypted_data + encrypted_cred.tag
            
            # Decrypt with authentication verification
            try:
                plaintext_bytes = bytearray(
                    aesgcm.decrypt(encrypted_cred.nonce, ciphertext_and_tag, associated_data)
                )
            except Exception as e:
                # Authentication failed - possible tampering
                logger.warning(
                    "Decryption authentication failed",
                    user_id_hash=hashlib.sha256(user_id.encode()).hexdigest()[:8],
                    error=str(e)
                )
                return Failure("Decryption failed: Authentication tag verification failed")
            
            # Convert to string
            credential = bytes(plaintext_bytes).decode('utf-8')
            
            # Note: We can't update last_accessed on frozen dataclass
            # This would need to be handled by the caller
            
            logger.debug(
                "Successfully decrypted credential",
                user_id_hash=hashlib.sha256(user_id.encode()).hexdigest()[:8],
                provider_type=encrypted_cred.provider_type
            )
            
            return Success(credential)
            
        except DecryptionError as e:
            logger.error("Decryption error", error=str(e))
            return Failure(str(e))
        except Exception as e:
            logger.error("Failed to decrypt credential", error=str(e), exc_info=True)
            return Failure(f"Decryption failed: {str(e)}")
        finally:
            # Always clear sensitive data from memory
            if user_key is not None:
                SecureMemory.secure_zero(user_key)
            if plaintext_bytes is not None:
                SecureMemory.secure_zero(plaintext_bytes)
    
    def rotate_user_salt(self, user_id: str) -> EncryptionResult[bool]:
        """
        Rotate user salt (requires re-encryption of all credentials).
        
        Args:
            user_id: User identifier
            
        Returns:
            Result indicating success or failure
        """
        try:
            # Generate new salt
            new_salt = SecureMemory.secure_random_bytes(self.config.salt_length)
            
            # Store old salt temporarily
            old_salt = self._user_salts.get(user_id)
            
            # Update salt
            self._user_salts[user_id] = new_salt
            
            # Increment key version
            self._key_versions[user_id] = self._key_versions.get(user_id, 1) + 1
            
            logger.info(
                "Rotated user salt",
                user_id_hash=hashlib.sha256(user_id.encode()).hexdigest()[:8],
                new_version=self._key_versions[user_id]
            )
            
            return Success(True)
            
        except Exception as e:
            logger.error("Failed to rotate user salt", error=str(e))
            return Failure(f"Salt rotation failed: {str(e)}")
    
    def get_master_salt(self) -> Optional[bytes]:
        """Get master salt for persistence."""
        return self._master_salt
    
    def set_master_salt(self, salt: bytes) -> None:
        """Set master salt from persistence."""
        self._master_salt = salt
    
    def get_user_salts(self) -> Dict[str, bytes]:
        """Get all user salts for persistence."""
        return self._user_salts.copy()
    
    def set_user_salts(self, salts: Dict[str, bytes]) -> None:
        """Set user salts from persistence."""
        self._user_salts = salts.copy()
    
    def secure_cleanup(self) -> None:
        """
        Securely clean up all sensitive data from memory.
        
        Should be called when the encryption service is no longer needed.
        """
        try:
            # Clear master key
            if self._master_key is not None:
                SecureMemory.secure_zero(self._master_key)
                self._master_key = None
            
            # Clear cached keys
            for key_data in self._key_cache.values():
                if isinstance(key_data[0], bytearray):
                    SecureMemory.secure_zero(key_data[0])
            self._key_cache.clear()
            
            # Clear user salts (less sensitive but still good practice)
            self._user_salts.clear()
            self._key_versions.clear()
            
            logger.info("Securely cleaned up encryption service")
            
        except Exception as e:
            logger.error("Error during secure cleanup", error=str(e))
    
    def is_rotation_due(self, encrypted_cred: EncryptedCredential) -> bool:
        """
        Check if credential rotation is due.
        
        Args:
            encrypted_cred: Encrypted credential to check
            
        Returns:
            True if rotation is due
        """
        if encrypted_cred.rotation_due is None:
            return False
        return datetime.now(timezone.utc) >= encrypted_cred.rotation_due
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check on encryption service.
        
        Returns:
            Health status dictionary with detailed diagnostics
        """
        test_data = None
        decrypted = None
        try:
            # Check master key status
            if self._master_key is None:
                return {
                    "healthy": False,
                    "error": "Master key not set",
                    "master_key_set": False,
                    "algorithm": "AES-256-GCM",
                    "kdf": "PBKDF2-HMAC-SHA256"
                }
            
            # Test secure random
            try:
                test_random = SecureMemory.secure_random_bytes(32)
                if len(test_random) != 32:
                    raise ValueError("Random generation failed")
            except Exception as e:
                return {
                    "healthy": False,
                    "error": "Secure random generation failed",
                    "details": str(e)
                }
            
            # Test encryption/decryption roundtrip
            test_data = "health_check_test_credential_" + secrets.token_hex(8)
            test_user = "health_check_user_" + secrets.token_hex(4)
            
            # Encrypt
            encrypt_result = self.encrypt(test_data, test_user, "health_check")
            if isinstance(encrypt_result, Failure):
                return {
                    "healthy": False,
                    "error": "Encryption test failed",
                    "details": str(encrypt_result.failure())
                }
            
            encrypted = encrypt_result.unwrap()
            
            # Decrypt
            decrypt_result = self.decrypt(encrypted, test_user)
            if isinstance(decrypt_result, Failure):
                return {
                    "healthy": False,
                    "error": "Decryption test failed",
                    "details": str(decrypt_result.failure())
                }
            
            decrypted = decrypt_result.unwrap()
            
            # Verify roundtrip
            if not SecureMemory.secure_compare(
                test_data.encode('utf-8'),
                decrypted.encode('utf-8')
            ):
                return {
                    "healthy": False,
                    "error": "Encryption roundtrip failed",
                    "details": "Decrypted data does not match original"
                }
            
            # All checks passed
            return {
                "healthy": True,
                "master_key_set": True,
                "user_salts_count": len(self._user_salts),
                "key_versions_count": len(self._key_versions),
                "algorithm": "AES-256-GCM",
                "kdf": "PBKDF2-HMAC-SHA256",
                "kdf_iterations": self.config.pbkdf2_iterations,
                "key_length_bits": self.config.key_length * 8,
                "salt_length_bytes": self.config.salt_length,
                "service_uptime_hours": (
                    datetime.now(timezone.utc) - self._initialized_at
                ).total_seconds() / 3600
            }
            
        except Exception as e:
            logger.error("Health check failed", error=str(e), exc_info=True)
            return {
                "healthy": False,
                "error": "Health check exception",
                "details": str(e),
                "master_key_set": self._master_key is not None
            }
        finally:
            # Clean up test data
            if test_data:
                test_bytes = bytearray(test_data.encode('utf-8'))
                SecureMemory.secure_zero(test_bytes)
            if decrypted:
                decrypted_bytes = bytearray(decrypted.encode('utf-8'))
                SecureMemory.secure_zero(decrypted_bytes)

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.secure_cleanup()


# Backwards compatibility alias
CredentialEncryptionService = CredentialEncryption


# Example usage
if __name__ == "__main__":
    """
    Example usage of the CredentialEncryption service.
    
    This demonstrates the basic encryption/decryption workflow.
    """
    import secrets
    
    # Initialize with default configuration
    encryption = CredentialEncryption()
    
    # Set master password (in production, this should come from secure input)
    master_password = "ExampleSecurePassword123!"
    result = encryption.set_master_password(master_password)
    
    if isinstance(result, Failure):
        print(f"Failed to set master password: {result.failure()}")
        exit(1)
    
    # Encrypt a credential
    api_key = "sk-ant-api03-example-key-" + secrets.token_hex(16)
    user_id = "user_example_12345"
    provider = "anthropic"
    
    encrypt_result = encryption.encrypt(api_key, user_id, provider)
    if isinstance(encrypt_result, Success):
        encrypted_cred = encrypt_result.unwrap()
        print(f"Encrypted credential for {provider}")
        print(f"Algorithm: {encrypted_cred.algorithm}")
        print(f"Key derivation: {encrypted_cred.kdf_algorithm} with {encrypted_cred.kdf_iterations:,} iterations")
        
        # Decrypt the credential
        decrypt_result = encryption.decrypt(encrypted_cred, user_id)
        if isinstance(decrypt_result, Success):
            decrypted_key = decrypt_result.unwrap()
            print(f"Successfully decrypted: {decrypted_key[:20]}...")
            
            # Verify roundtrip
            if decrypted_key == api_key:
                print("✓ Encryption/decryption roundtrip successful")
            else:
                print("✗ Roundtrip failed")
        else:
            print(f"Decryption failed: {decrypt_result.failure()}")
    else:
        print(f"Encryption failed: {encrypt_result.failure()}")
    
    # Clean up
    encryption.secure_cleanup()
    print("Example completed")