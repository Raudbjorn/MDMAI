"""
Tests for credential management functionality.

Tests cover encryption/decryption, storage, key rotation, and security features.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from cryptography.fernet import Fernet

from src.ai_providers.credential_manager import CredentialManager


class TestCredentialManager:
    """Test cases for CredentialManager."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def encryption_key(self):
        """Generate test encryption key."""
        return Fernet.generate_key().decode('utf-8')
    
    @pytest.fixture
    def credential_manager(self, temp_storage, encryption_key):
        """Create CredentialManager instance for testing."""
        # Set environment variable for encryption key
        os.environ['MDMAI_ENCRYPTION_KEY'] = encryption_key
        
        manager = CredentialManager(storage_path=temp_storage)
        
        yield manager
        
        # Cleanup
        if 'MDMAI_ENCRYPTION_KEY' in os.environ:
            del os.environ['MDMAI_ENCRYPTION_KEY']
    
    def test_init_requires_encryption_key(self, temp_storage):
        """Test that initialization requires encryption key."""
        # Remove key if it exists
        if 'MDMAI_ENCRYPTION_KEY' in os.environ:
            del os.environ['MDMAI_ENCRYPTION_KEY']
        
        with pytest.raises(ValueError, match="MDMAI_ENCRYPTION_KEY environment variable must be set"):
            CredentialManager(storage_path=temp_storage)
    
    def test_init_invalid_encryption_key(self, temp_storage):
        """Test that initialization fails with invalid key."""
        os.environ['MDMAI_ENCRYPTION_KEY'] = "invalid_key"
        
        with pytest.raises(ValueError, match="Invalid MDMAI_ENCRYPTION_KEY format"):
            CredentialManager(storage_path=temp_storage)
        
        # Cleanup
        del os.environ['MDMAI_ENCRYPTION_KEY']
    
    def test_encrypt_decrypt_api_key(self, credential_manager):
        """Test API key encryption and decryption."""
        user_id = "test_user"
        api_key = "sk-test-key-12345"
        
        # Encrypt the key
        encrypted_key = credential_manager.encrypt_api_key(api_key, user_id)
        
        # Verify it's encrypted (not plain text)
        assert encrypted_key != api_key
        assert len(encrypted_key) > len(api_key)
        
        # Decrypt and verify
        decrypted_key = credential_manager.decrypt_api_key(encrypted_key, user_id)
        assert decrypted_key == api_key
    
    def test_encrypt_empty_values(self, credential_manager):
        """Test encryption with empty values."""
        with pytest.raises(ValueError, match="Both api_key and user_id must be provided"):
            credential_manager.encrypt_api_key("", "user_id")
        
        with pytest.raises(ValueError, match="Both api_key and user_id must be provided"):
            credential_manager.encrypt_api_key("api_key", "")
    
    def test_decrypt_with_wrong_user_id(self, credential_manager):
        """Test decryption with wrong user ID."""
        user_id = "test_user"
        wrong_user_id = "wrong_user"
        api_key = "sk-test-key-12345"
        
        encrypted_key = credential_manager.encrypt_api_key(api_key, user_id)
        
        # Try to decrypt with wrong user ID
        decrypted_key = credential_manager.decrypt_api_key(encrypted_key, wrong_user_id)
        assert decrypted_key is None
    
    def test_decrypt_invalid_token(self, credential_manager):
        """Test decryption with invalid encrypted token."""
        user_id = "test_user"
        invalid_token = "invalid_encrypted_token"
        
        decrypted_key = credential_manager.decrypt_api_key(invalid_token, user_id)
        assert decrypted_key is None
    
    def test_storage_persistence(self, credential_manager, temp_storage):
        """Test that credentials persist to storage."""
        user_id = "test_user"
        api_key = "sk-test-key-12345"
        
        # Encrypt and store
        credential_manager.encrypt_api_key(api_key, user_id)
        
        # Verify file was created
        credentials_file = Path(temp_storage) / "encrypted_credentials.json"
        assert credentials_file.exists()
        
        # Verify content is stored
        with open(credentials_file, 'r') as f:
            stored_data = json.load(f)
        
        assert user_id in stored_data
        assert stored_data[user_id] != api_key  # Should be encrypted
    
    def test_load_existing_credentials(self, temp_storage, encryption_key):
        """Test loading existing credentials from storage."""
        user_id = "test_user"
        api_key = "sk-test-key-12345"
        
        # Create first manager and store credentials
        os.environ['MDMAI_ENCRYPTION_KEY'] = encryption_key
        manager1 = CredentialManager(storage_path=temp_storage)
        encrypted_key = manager1.encrypt_api_key(api_key, user_id)
        
        # Create second manager and verify it loads existing credentials
        manager2 = CredentialManager(storage_path=temp_storage)
        
        # Decrypt using stored credential (empty encrypted_key parameter)
        decrypted_key = manager2.decrypt_api_key("", user_id)
        assert decrypted_key == api_key
        
        # Cleanup
        del os.environ['MDMAI_ENCRYPTION_KEY']
    
    def test_delete_credentials(self, credential_manager):
        """Test credential deletion."""
        user_id = "test_user"
        api_key = "sk-test-key-12345"
        
        # Store credentials
        credential_manager.encrypt_api_key(api_key, user_id)
        
        # Verify stored
        assert credential_manager.has_credentials(user_id)
        
        # Delete
        result = credential_manager.delete_credentials(user_id)
        assert result is True
        
        # Verify deleted
        assert not credential_manager.has_credentials(user_id)
        
        # Try to delete non-existent
        result = credential_manager.delete_credentials("non_existent_user")
        assert result is False
    
    def test_has_credentials(self, credential_manager):
        """Test checking if credentials exist."""
        user_id = "test_user"
        api_key = "sk-test-key-12345"
        
        # Initially no credentials
        assert not credential_manager.has_credentials(user_id)
        
        # Store credentials
        credential_manager.encrypt_api_key(api_key, user_id)
        
        # Now should have credentials
        assert credential_manager.has_credentials(user_id)
    
    def test_list_users_with_credentials(self, credential_manager):
        """Test listing users with credentials."""
        users = ["user1", "user2", "user3"]
        api_key = "sk-test-key-12345"
        
        # Initially empty
        assert credential_manager.list_users_with_credentials() == []
        
        # Add credentials for users
        for user in users:
            credential_manager.encrypt_api_key(api_key, user)
        
        # Verify list
        stored_users = credential_manager.list_users_with_credentials()
        assert set(stored_users) == set(users)
    
    def test_key_rotation(self, credential_manager, encryption_key):
        """Test encryption key rotation."""
        user_id = "test_user"
        api_key = "sk-test-key-12345"
        
        # Store credentials with original key
        credential_manager.encrypt_api_key(api_key, user_id)
        
        # Generate new key
        new_key = Fernet.generate_key().decode('utf-8')
        
        # Rotate key
        result = credential_manager.rotate_encryption_key(new_key)
        assert result is True
        
        # Verify we can still decrypt with new key
        decrypted_key = credential_manager.decrypt_api_key("", user_id)
        assert decrypted_key == api_key
    
    def test_key_rotation_invalid_key(self, credential_manager):
        """Test key rotation with invalid key."""
        with pytest.raises(ValueError, match="Invalid new encryption key format"):
            credential_manager.rotate_encryption_key("invalid_key")
    
    def test_backup_credentials(self, credential_manager, temp_storage):
        """Test credential backup functionality."""
        user_id = "test_user"
        api_key = "sk-test-key-12345"
        
        # Store credentials
        credential_manager.encrypt_api_key(api_key, user_id)
        
        # Create backup
        backup_path = credential_manager.backup_credentials()
        
        # Verify backup exists
        assert Path(backup_path).exists()
        
        # Verify backup contains data
        with open(backup_path, 'r') as f:
            backup_data = json.load(f)
        
        assert user_id in backup_data
    
    def test_backup_credentials_custom_path(self, credential_manager, temp_storage):
        """Test credential backup with custom path."""
        user_id = "test_user"
        api_key = "sk-test-key-12345"
        
        # Store credentials
        credential_manager.encrypt_api_key(api_key, user_id)
        
        # Create backup with custom path
        custom_backup_path = str(Path(temp_storage) / "custom_backup.json")
        backup_path = credential_manager.backup_credentials(custom_backup_path)
        
        assert backup_path == custom_backup_path
        assert Path(backup_path).exists()
    
    def test_corrupted_credentials_file(self, temp_storage, encryption_key):
        """Test handling of corrupted credentials file."""
        credentials_file = Path(temp_storage) / "encrypted_credentials.json"
        credentials_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create corrupted file
        with open(credentials_file, 'w') as f:
            f.write("invalid json content")
        
        # Initialize manager - should handle corruption gracefully
        os.environ['MDMAI_ENCRYPTION_KEY'] = encryption_key
        manager = CredentialManager(storage_path=temp_storage)
        
        # Should start with empty credentials
        assert manager.list_users_with_credentials() == []
        
        # Backup file should be created
        backup_file = credentials_file.with_suffix('.json.backup')
        assert backup_file.exists()
        
        # Cleanup
        del os.environ['MDMAI_ENCRYPTION_KEY']
    
    def test_multiple_users_same_key(self, credential_manager):
        """Test multiple users can store the same API key."""
        users = ["user1", "user2", "user3"]
        api_key = "sk-shared-key-12345"
        
        # Store same key for multiple users
        for user in users:
            credential_manager.encrypt_api_key(api_key, user)
        
        # Verify each user can decrypt their own key
        for user in users:
            decrypted_key = credential_manager.decrypt_api_key("", user)
            assert decrypted_key == api_key
    
    def test_user_isolation(self, credential_manager):
        """Test that users' credentials are isolated."""
        user1 = "user1"
        user2 = "user2"
        api_key1 = "sk-user1-key-12345"
        api_key2 = "sk-user2-key-67890"
        
        # Store different keys for each user
        encrypted1 = credential_manager.encrypt_api_key(api_key1, user1)
        encrypted2 = credential_manager.encrypt_api_key(api_key2, user2)
        
        # Verify keys are different when encrypted
        assert encrypted1 != encrypted2
        
        # Verify each user gets their own key
        assert credential_manager.decrypt_api_key("", user1) == api_key1
        assert credential_manager.decrypt_api_key("", user2) == api_key2
        
        # Verify cross-user access fails
        assert credential_manager.decrypt_api_key(encrypted1, user2) is None
        assert credential_manager.decrypt_api_key(encrypted2, user1) is None
    
    def test_special_characters_in_api_key(self, credential_manager):
        """Test API keys with special characters."""
        user_id = "test_user"
        api_keys = [
            "sk-key-with-dashes",
            "sk_key_with_underscores",
            "sk.key.with.dots",
            "sk+key+with+plus",
            "sk=key=with=equals",
            "sk key with spaces",
            "sk-key-with-unicode-éñ",
            "sk-key-with-symbols-!@#$%^&*()"
        ]
        
        for api_key in api_keys:
            # Encrypt
            encrypted_key = credential_manager.encrypt_api_key(api_key, f"{user_id}_{len(api_key)}")
            
            # Decrypt and verify
            decrypted_key = credential_manager.decrypt_api_key(encrypted_key, f"{user_id}_{len(api_key)}")
            assert decrypted_key == api_key, f"Failed for API key: {api_key}"
    
    def test_large_api_key(self, credential_manager):
        """Test very large API keys."""
        user_id = "test_user"
        # Create a large API key (1KB)
        large_api_key = "sk-" + "x" * 1020
        
        encrypted_key = credential_manager.encrypt_api_key(large_api_key, user_id)
        decrypted_key = credential_manager.decrypt_api_key(encrypted_key, user_id)
        
        assert decrypted_key == large_api_key
    
    def test_concurrent_access(self, temp_storage, encryption_key):
        """Test concurrent access to credentials (simulation)."""
        import threading
        import time
        
        os.environ['MDMAI_ENCRYPTION_KEY'] = encryption_key
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                manager = CredentialManager(storage_path=temp_storage)
                user_id = f"user_{worker_id}"
                api_key = f"sk-key-{worker_id}"
                
                # Store credential
                encrypted = manager.encrypt_api_key(api_key, user_id)
                
                # Small delay to simulate real usage
                time.sleep(0.01)
                
                # Retrieve credential
                decrypted = manager.decrypt_api_key("", user_id)
                
                if decrypted == api_key:
                    results.append(worker_id)
                else:
                    errors.append(f"Worker {worker_id}: expected {api_key}, got {decrypted}")
                    
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 10, f"Expected 10 successful operations, got {len(results)}"
        
        # Cleanup
        del os.environ['MDMAI_ENCRYPTION_KEY']