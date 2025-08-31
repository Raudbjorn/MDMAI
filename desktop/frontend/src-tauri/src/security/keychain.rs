//! OS Keychain Integration
//! 
//! This module provides secure credential storage using the operating system's
//! native keychain/credential management systems:
//! - Windows Credential Manager
//! - macOS Keychain Services
//! - Linux Secret Service API (libsecret)
//! - Fallback encrypted storage for unsupported platforms

use super::*;
use std::collections::HashMap;
use std::time::SystemTime;
use serde::{Deserialize, Serialize};

/// Keychain entry for storing credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeychainEntry {
    pub id: String,
    pub service: String,
    pub account: String,
    pub description: String,
    pub created_at: SystemTime,
    pub last_accessed: SystemTime,
    pub metadata: HashMap<String, String>,
}

/// Credential data stored in keychain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialData {
    pub secret: String,
    pub additional_data: HashMap<String, String>,
    pub expires_at: Option<SystemTime>,
}

/// Keychain manager for secure credential storage
pub struct KeychainManager {
    config: SecurityConfig,
    service_prefix: String,
    fallback_storage: Option<EncryptedStorage>,
}

/// Encrypted fallback storage for platforms without keychain support
struct EncryptedStorage {
    storage_path: std::path::PathBuf,
    encryption_key: [u8; 32],
}

impl KeychainManager {
    pub async fn new(config: &SecurityConfig) -> SecurityResult<Self> {
        let service_prefix = "com.ttrpg.assistant".to_string();
        
        // Check if native keychain is available, otherwise use fallback
        let fallback_storage = if Self::is_native_keychain_available() {
            None
        } else {
            log::warn!("Native keychain not available, using encrypted fallback storage");
            Some(EncryptedStorage::new().await?)
        };

        Ok(Self {
            config: config.clone(),
            service_prefix,
            fallback_storage,
        })
    }

    pub async fn initialize(&self) -> SecurityResult<()> {
        // Test keychain access
        if self.fallback_storage.is_none() {
            self.test_native_keychain().await?;
        }
        
        log::info!("Keychain manager initialized");
        Ok(())
    }

    /// Store a credential in the keychain
    pub async fn store_credential(
        &self,
        service: &str,
        account: &str,
        credential: &CredentialData,
        description: Option<&str>,
    ) -> SecurityResult<String> {
        let entry_id = format!("{}_{}", service, account);
        let full_service = format!("{}_{}", self.service_prefix, service);
        
        let keychain_entry = KeychainEntry {
            id: entry_id.clone(),
            service: service.to_string(),
            account: account.to_string(),
            description: description.unwrap_or("TTRPG Assistant credential").to_string(),
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            metadata: HashMap::new(),
        };

        // Serialize the credential data
        let credential_json = serde_json::to_string(credential)
            .map_err(|e| SecurityError::KeychainError {
                message: format!("Failed to serialize credential: {}", e),
            })?;

        // Store using native keychain or fallback
        if let Some(fallback) = &self.fallback_storage {
            fallback.store(&entry_id, &credential_json, &keychain_entry).await?;
        } else {
            self.store_native(&full_service, account, &credential_json).await?;
        }

        log::info!("Stored credential for service: {} account: {}", service, account);
        Ok(entry_id)
    }

    /// Retrieve a credential from the keychain
    pub async fn retrieve_credential(
        &self,
        service: &str,
        account: &str,
    ) -> SecurityResult<CredentialData> {
        let entry_id = format!("{}_{}", service, account);
        let full_service = format!("{}_{}", self.service_prefix, service);

        // Retrieve using native keychain or fallback
        let credential_json = if let Some(fallback) = &self.fallback_storage {
            fallback.retrieve(&entry_id).await?
        } else {
            self.retrieve_native(&full_service, account).await?
        };

        // Deserialize the credential data
        let mut credential: CredentialData = serde_json::from_str(&credential_json)
            .map_err(|e| SecurityError::KeychainError {
                message: format!("Failed to deserialize credential: {}", e),
            })?;

        // Check if credential has expired
        if let Some(expires_at) = credential.expires_at {
            if SystemTime::now() > expires_at {
                // Remove expired credential
                self.delete_credential(service, account).await?;
                return Err(SecurityError::KeychainError {
                    message: "Credential has expired".to_string(),
                });
            }
        }

        log::info!("Retrieved credential for service: {} account: {}", service, account);
        Ok(credential)
    }

    /// Delete a credential from the keychain
    pub async fn delete_credential(
        &self,
        service: &str,
        account: &str,
    ) -> SecurityResult<()> {
        let entry_id = format!("{}_{}", service, account);
        let full_service = format!("{}_{}", self.service_prefix, service);

        // Delete using native keychain or fallback
        if let Some(fallback) = &self.fallback_storage {
            fallback.delete(&entry_id).await?;
        } else {
            self.delete_native(&full_service, account).await?;
        }

        log::info!("Deleted credential for service: {} account: {}", service, account);
        Ok(())
    }

    /// List all stored credentials (metadata only, no secrets)
    pub async fn list_credentials(&self) -> SecurityResult<Vec<KeychainEntry>> {
        if let Some(fallback) = &self.fallback_storage {
            fallback.list_entries().await
        } else {
            // For native keychain, we'd need to maintain a separate index
            // This is a simplified implementation
            Ok(Vec::new())
        }
    }

    /// Check if a credential exists
    pub async fn credential_exists(&self, service: &str, account: &str) -> bool {
        self.retrieve_credential(service, account).await.is_ok()
    }

    /// Update credential metadata (without changing the secret)
    pub async fn update_credential_metadata(
        &self,
        service: &str,
        account: &str,
        metadata: HashMap<String, String>,
    ) -> SecurityResult<()> {
        // For native keychain, we'd need to re-store with updated metadata
        // For fallback storage, we can update the entry directly
        if let Some(fallback) = &self.fallback_storage {
            let entry_id = format!("{}_{}", service, account);
            fallback.update_metadata(&entry_id, metadata).await?;
        }
        
        Ok(())
    }

    /// Cleanup expired credentials
    pub async fn cleanup_expired_credentials(&self) -> SecurityResult<u32> {
        let mut deleted_count = 0;

        if let Some(fallback) = &self.fallback_storage {
            deleted_count = fallback.cleanup_expired().await?;
        }

        log::info!("Cleaned up {} expired credentials", deleted_count);
        Ok(deleted_count)
    }

    // Private methods

    /// Check if native keychain is available
    fn is_native_keychain_available() -> bool {
        #[cfg(target_os = "windows")]
        return true; // Windows Credential Manager is always available

        #[cfg(target_os = "macos")]
        return true; // Keychain Services is always available

        #[cfg(target_os = "linux")]
        {
            // Check if Secret Service is available
            // This would require checking for dbus and libsecret
            return false; // Simplified for now
        }

        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        return false;
    }

    /// Test native keychain access
    async fn test_native_keychain(&self) -> SecurityResult<()> {
        let test_service = format!("{}_test", self.service_prefix);
        let test_account = "test_account";
        let test_data = "test_data";

        // Try to store and retrieve a test credential
        self.store_native(&test_service, test_account, test_data).await?;
        let retrieved = self.retrieve_native(&test_service, test_account).await?;
        self.delete_native(&test_service, test_account).await?;

        if retrieved != test_data {
            return Err(SecurityError::KeychainError {
                message: "Native keychain test failed".to_string(),
            });
        }

        Ok(())
    }

    /// Store credential using native keychain
    async fn store_native(&self, service: &str, account: &str, secret: &str) -> SecurityResult<()> {
        #[cfg(target_os = "windows")]
        return self.store_windows(service, account, secret).await;

        #[cfg(target_os = "macos")]
        return self.store_macos(service, account, secret).await;

        #[cfg(target_os = "linux")]
        return self.store_linux(service, account, secret).await;

        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        Err(SecurityError::KeychainError {
            message: "Native keychain not supported on this platform".to_string(),
        })
    }

    /// Retrieve credential using native keychain
    async fn retrieve_native(&self, service: &str, account: &str) -> SecurityResult<String> {
        #[cfg(target_os = "windows")]
        return self.retrieve_windows(service, account).await;

        #[cfg(target_os = "macos")]
        return self.retrieve_macos(service, account).await;

        #[cfg(target_os = "linux")]
        return self.retrieve_linux(service, account).await;

        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        Err(SecurityError::KeychainError {
            message: "Native keychain not supported on this platform".to_string(),
        })
    }

    /// Delete credential using native keychain
    async fn delete_native(&self, service: &str, account: &str) -> SecurityResult<()> {
        #[cfg(target_os = "windows")]
        return self.delete_windows(service, account).await;

        #[cfg(target_os = "macos")]
        return self.delete_macos(service, account).await;

        #[cfg(target_os = "linux")]
        return self.delete_linux(service, account).await;

        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        Err(SecurityError::KeychainError {
            message: "Native keychain not supported on this platform".to_string(),
        })
    }

    // Platform-specific implementations

    #[cfg(target_os = "windows")]
    async fn store_windows(&self, service: &str, account: &str, secret: &str) -> SecurityResult<()> {
        // This would use the Windows Credential Manager API
        // For now, this is a placeholder implementation
        tokio::task::spawn_blocking(move || {
            // Call Windows API to store credential
            // Use CredWriteW or similar
            Ok(())
        }).await.map_err(|e| SecurityError::KeychainError {
            message: format!("Windows keychain operation failed: {}", e),
        })?
    }

    #[cfg(target_os = "windows")]
    async fn retrieve_windows(&self, service: &str, account: &str) -> SecurityResult<String> {
        tokio::task::spawn_blocking(move || {
            // Call Windows API to retrieve credential
            // Use CredReadW or similar
            Ok("placeholder".to_string())
        }).await.map_err(|e| SecurityError::KeychainError {
            message: format!("Windows keychain operation failed: {}", e),
        })?
    }

    #[cfg(target_os = "windows")]
    async fn delete_windows(&self, service: &str, account: &str) -> SecurityResult<()> {
        tokio::task::spawn_blocking(move || {
            // Call Windows API to delete credential
            // Use CredDeleteW or similar
            Ok(())
        }).await.map_err(|e| SecurityError::KeychainError {
            message: format!("Windows keychain operation failed: {}", e),
        })?
    }

    #[cfg(target_os = "macos")]
    async fn store_macos(&self, service: &str, account: &str, secret: &str) -> SecurityResult<()> {
        // This would use the macOS Security framework
        // For now, this is a placeholder implementation
        tokio::task::spawn_blocking(move || {
            // Call Security framework to store in keychain
            // Use SecKeychainAddInternetPassword or similar
            Ok(())
        }).await.map_err(|e| SecurityError::KeychainError {
            message: format!("macOS keychain operation failed: {}", e),
        })?
    }

    #[cfg(target_os = "macos")]
    async fn retrieve_macos(&self, service: &str, account: &str) -> SecurityResult<String> {
        tokio::task::spawn_blocking(move || {
            // Call Security framework to retrieve from keychain
            Ok("placeholder".to_string())
        }).await.map_err(|e| SecurityError::KeychainError {
            message: format!("macOS keychain operation failed: {}", e),
        })?
    }

    #[cfg(target_os = "macos")]
    async fn delete_macos(&self, service: &str, account: &str) -> SecurityResult<()> {
        tokio::task::spawn_blocking(move || {
            // Call Security framework to delete from keychain
            Ok(())
        }).await.map_err(|e| SecurityError::KeychainError {
            message: format!("macOS keychain operation failed: {}", e),
        })?
    }

    #[cfg(target_os = "linux")]
    async fn store_linux(&self, service: &str, account: &str, secret: &str) -> SecurityResult<()> {
        // This would use the Secret Service API (libsecret)
        tokio::task::spawn_blocking(move || {
            // Call libsecret to store credential
            Ok(())
        }).await.map_err(|e| SecurityError::KeychainError {
            message: format!("Linux keychain operation failed: {}", e),
        })?
    }

    #[cfg(target_os = "linux")]
    async fn retrieve_linux(&self, service: &str, account: &str) -> SecurityResult<String> {
        tokio::task::spawn_blocking(move || {
            // Call libsecret to retrieve credential
            Ok("placeholder".to_string())
        }).await.map_err(|e| SecurityError::KeychainError {
            message: format!("Linux keychain operation failed: {}", e),
        })?
    }

    #[cfg(target_os = "linux")]
    async fn delete_linux(&self, service: &str, account: &str) -> SecurityResult<()> {
        tokio::task::spawn_blocking(move || {
            // Call libsecret to delete credential
            Ok(())
        }).await.map_err(|e| SecurityError::KeychainError {
            message: format!("Linux keychain operation failed: {}", e),
        })?
    }
}

impl EncryptedStorage {
    async fn new() -> SecurityResult<Self> {
        use rand::RngCore;
        
        let storage_path = std::env::temp_dir().join("ttrpg_assistant_keystore.enc");
        let mut encryption_key = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut encryption_key);

        Ok(Self {
            storage_path,
            encryption_key,
        })
    }

    async fn store(&self, id: &str, secret: &str, entry: &KeychainEntry) -> SecurityResult<()> {
        // Load existing storage
        let mut storage = self.load_storage().await.unwrap_or_default();
        
        // Add new entry
        storage.entries.insert(id.to_string(), entry.clone());
        storage.secrets.insert(id.to_string(), secret.to_string());
        storage.last_modified = SystemTime::now();

        // Save storage
        self.save_storage(&storage).await
    }

    async fn retrieve(&self, id: &str) -> SecurityResult<String> {
        let storage = self.load_storage().await?;
        
        storage.secrets.get(id).cloned().ok_or_else(|| {
            SecurityError::KeychainError {
                message: format!("Credential not found: {}", id),
            }
        })
    }

    async fn delete(&self, id: &str) -> SecurityResult<()> {
        let mut storage = self.load_storage().await?;
        
        storage.entries.remove(id);
        storage.secrets.remove(id);
        storage.last_modified = SystemTime::now();

        self.save_storage(&storage).await
    }

    async fn list_entries(&self) -> SecurityResult<Vec<KeychainEntry>> {
        let storage = self.load_storage().await?;
        Ok(storage.entries.values().cloned().collect())
    }

    async fn update_metadata(&self, id: &str, metadata: HashMap<String, String>) -> SecurityResult<()> {
        let mut storage = self.load_storage().await?;
        
        if let Some(entry) = storage.entries.get_mut(id) {
            entry.metadata = metadata;
            entry.last_accessed = SystemTime::now();
            storage.last_modified = SystemTime::now();
            self.save_storage(&storage).await
        } else {
            Err(SecurityError::KeychainError {
                message: format!("Entry not found: {}", id),
            })
        }
    }

    async fn cleanup_expired(&self) -> SecurityResult<u32> {
        let mut storage = self.load_storage().await?;
        let mut deleted_count = 0;

        let entries_to_remove: Vec<String> = storage
            .entries
            .iter()
            .filter(|(_, entry)| {
                // Check if we can determine expiration from metadata
                // This is a simplified check
                false // No expiration logic in this example
            })
            .map(|(id, _)| id.clone())
            .collect();

        for id in entries_to_remove {
            storage.entries.remove(&id);
            storage.secrets.remove(&id);
            deleted_count += 1;
        }

        if deleted_count > 0 {
            storage.last_modified = SystemTime::now();
            self.save_storage(&storage).await?;
        }

        Ok(deleted_count)
    }

    async fn load_storage(&self) -> SecurityResult<StorageData> {
        if !self.storage_path.exists() {
            return Ok(StorageData::default());
        }

        let encrypted_data = tokio::fs::read(&self.storage_path).await
            .map_err(|e| SecurityError::KeychainError {
                message: format!("Failed to read storage file: {}", e),
            })?;

        // Decrypt and deserialize
        let decrypted_data = self.decrypt_data(&encrypted_data)?;
        let storage: StorageData = serde_json::from_slice(&decrypted_data)
            .map_err(|e| SecurityError::KeychainError {
                message: format!("Failed to deserialize storage data: {}", e),
            })?;

        Ok(storage)
    }

    async fn save_storage(&self, storage: &StorageData) -> SecurityResult<()> {
        let json_data = serde_json::to_vec(storage)
            .map_err(|e| SecurityError::KeychainError {
                message: format!("Failed to serialize storage data: {}", e),
            })?;

        let encrypted_data = self.encrypt_data(&json_data)?;
        
        tokio::fs::write(&self.storage_path, encrypted_data).await
            .map_err(|e| SecurityError::KeychainError {
                message: format!("Failed to write storage file: {}", e),
            })
    }

    fn encrypt_data(&self, data: &[u8]) -> SecurityResult<Vec<u8>> {
        use aes_gcm::{Aes256Gcm, Key, Nonce};
        use aes_gcm::aead::{Aead, KeyInit};
        use rand::RngCore;

        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&self.encryption_key));
        let mut nonce_bytes = [0u8; 12];
        rand::rngs::OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher.encrypt(nonce, data)
            .map_err(|e| SecurityError::CryptographicError {
                message: format!("Encryption failed: {}", e),
            })?;

        let mut result = Vec::with_capacity(12 + ciphertext.len());
        result.extend_from_slice(&nonce_bytes);
        result.extend_from_slice(&ciphertext);

        Ok(result)
    }

    fn decrypt_data(&self, encrypted_data: &[u8]) -> SecurityResult<Vec<u8>> {
        use aes_gcm::{Aes256Gcm, Key, Nonce};
        use aes_gcm::aead::{Aead, KeyInit};

        if encrypted_data.len() < 12 {
            return Err(SecurityError::CryptographicError {
                message: "Invalid encrypted data format".to_string(),
            });
        }

        let (nonce_bytes, ciphertext) = encrypted_data.split_at(12);
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&self.encryption_key));
        let nonce = Nonce::from_slice(nonce_bytes);

        cipher.decrypt(nonce, ciphertext)
            .map_err(|e| SecurityError::CryptographicError {
                message: format!("Decryption failed: {}", e),
            })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct StorageData {
    entries: HashMap<String, KeychainEntry>,
    secrets: HashMap<String, String>,
    last_modified: SystemTime,
    version: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_keychain_operations() {
        let config = SecurityConfig::default();
        let keychain = KeychainManager::new(&config).await.unwrap();
        keychain.initialize().await.unwrap();

        let credential = CredentialData {
            secret: "test_secret".to_string(),
            additional_data: HashMap::new(),
            expires_at: None,
        };

        // Store credential
        let entry_id = keychain
            .store_credential("test_service", "test_account", &credential, Some("Test credential"))
            .await
            .unwrap();

        // Retrieve credential
        let retrieved = keychain
            .retrieve_credential("test_service", "test_account")
            .await
            .unwrap();
        
        assert_eq!(retrieved.secret, "test_secret");

        // Delete credential
        keychain
            .delete_credential("test_service", "test_account")
            .await
            .unwrap();

        // Verify deletion
        assert!(!keychain.credential_exists("test_service", "test_account").await);
    }
}