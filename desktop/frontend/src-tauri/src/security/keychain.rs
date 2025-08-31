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
use sha2::{Sha256, Digest};

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

    /// Generate a collision-resistant entry ID using SHA-256 hashing
    /// This prevents collisions from service/account combinations containing separators
    fn generate_entry_id(&self, service: &str, account: &str) -> String {
        let mut hasher = Sha256::new();
        
        // Hash the service prefix, service, and account with length prefixes
        // This ensures that even if service="a" account="bc" and service="ab" account="c"
        // they will produce different hashes due to length encoding
        hasher.update(b"service_prefix:");
        hasher.update((self.service_prefix.len() as u32).to_be_bytes());
        hasher.update(self.service_prefix.as_bytes());
        
        hasher.update(b"service:");
        hasher.update((service.len() as u32).to_be_bytes());
        hasher.update(service.as_bytes());
        
        hasher.update(b"account:");
        hasher.update((account.len() as u32).to_be_bytes());
        hasher.update(account.as_bytes());
        
        let result = hasher.finalize();
        
        // Use hex encoding for readability in logs and debugging
        // Prefix with a version identifier for future compatibility
        format!("v1_{}", hex::encode(result))
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
        let entry_id = self.generate_entry_id(service, account);
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
        let entry_id = self.generate_entry_id(service, account);
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
        let entry_id = self.generate_entry_id(service, account);
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
            let entry_id = self.generate_entry_id(service, account);
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
        let service = service.to_string();
        let account = account.to_string();
        let secret = secret.to_string();
        
        tokio::task::spawn_blocking(move || {
            use keyring::Entry;
            
            let entry = Entry::new(&service, &account)
                .map_err(|e| SecurityError::KeychainError {
                    message: format!("Failed to create Windows keychain entry: {}", e),
                })?;
            
            entry.set_password(&secret)
                .map_err(|e| SecurityError::KeychainError {
                    message: format!("Failed to store Windows credential: {}", e),
                })?;
            
            Ok(())
        }).await.map_err(|e| SecurityError::KeychainError {
            message: format!("Windows keychain operation failed: {}", e),
        })?
    }

    #[cfg(target_os = "windows")]
    async fn retrieve_windows(&self, service: &str, account: &str) -> SecurityResult<String> {
        let service = service.to_string();
        let account = account.to_string();
        
        tokio::task::spawn_blocking(move || {
            use keyring::Entry;
            
            let entry = Entry::new(&service, &account)
                .map_err(|e| SecurityError::KeychainError {
                    message: format!("Failed to create Windows keychain entry: {}", e),
                })?;
            
            entry.get_password()
                .map_err(|e| SecurityError::KeychainError {
                    message: format!("Failed to retrieve Windows credential: {}", e),
                })
        }).await.map_err(|e| SecurityError::KeychainError {
            message: format!("Windows keychain operation failed: {}", e),
        })?
    }

    #[cfg(target_os = "windows")]
    async fn delete_windows(&self, service: &str, account: &str) -> SecurityResult<()> {
        let service = service.to_string();
        let account = account.to_string();
        
        tokio::task::spawn_blocking(move || {
            use keyring::Entry;
            
            let entry = Entry::new(&service, &account)
                .map_err(|e| SecurityError::KeychainError {
                    message: format!("Failed to create Windows keychain entry: {}", e),
                })?;
            
            entry.delete_password()
                .map_err(|e| SecurityError::KeychainError {
                    message: format!("Failed to delete Windows credential: {}", e),
                })?;
            
            Ok(())
        }).await.map_err(|e| SecurityError::KeychainError {
            message: format!("Windows keychain operation failed: {}", e),
        })?
    }

    #[cfg(target_os = "macos")]
    async fn store_macos(&self, service: &str, account: &str, secret: &str) -> SecurityResult<()> {
        let service = service.to_string();
        let account = account.to_string();
        let secret = secret.to_string();
        
        tokio::task::spawn_blocking(move || {
            use keyring::Entry;
            
            let entry = Entry::new(&service, &account)
                .map_err(|e| SecurityError::KeychainError {
                    message: format!("Failed to create macOS keychain entry: {}", e),
                })?;
            
            entry.set_password(&secret)
                .map_err(|e| SecurityError::KeychainError {
                    message: format!("Failed to store macOS credential: {}", e),
                })?;
            
            Ok(())
        }).await.map_err(|e| SecurityError::KeychainError {
            message: format!("macOS keychain operation failed: {}", e),
        })?
    }

    #[cfg(target_os = "macos")]
    async fn retrieve_macos(&self, service: &str, account: &str) -> SecurityResult<String> {
        let service = service.to_string();
        let account = account.to_string();
        
        tokio::task::spawn_blocking(move || {
            use keyring::Entry;
            
            let entry = Entry::new(&service, &account)
                .map_err(|e| SecurityError::KeychainError {
                    message: format!("Failed to create macOS keychain entry: {}", e),
                })?;
            
            entry.get_password()
                .map_err(|e| SecurityError::KeychainError {
                    message: format!("Failed to retrieve macOS credential: {}", e),
                })
        }).await.map_err(|e| SecurityError::KeychainError {
            message: format!("macOS keychain operation failed: {}", e),
        })?
    }

    #[cfg(target_os = "macos")]
    async fn delete_macos(&self, service: &str, account: &str) -> SecurityResult<()> {
        let service = service.to_string();
        let account = account.to_string();
        
        tokio::task::spawn_blocking(move || {
            use keyring::Entry;
            
            let entry = Entry::new(&service, &account)
                .map_err(|e| SecurityError::KeychainError {
                    message: format!("Failed to create macOS keychain entry: {}", e),
                })?;
            
            entry.delete_password()
                .map_err(|e| SecurityError::KeychainError {
                    message: format!("Failed to delete macOS credential: {}", e),
                })?;
            
            Ok(())
        }).await.map_err(|e| SecurityError::KeychainError {
            message: format!("macOS keychain operation failed: {}", e),
        })?
    }

    #[cfg(target_os = "linux")]
    async fn store_linux(&self, service: &str, account: &str, secret: &str) -> SecurityResult<()> {
        let service = service.to_string();
        let account = account.to_string();
        let secret = secret.to_string();
        
        tokio::task::spawn_blocking(move || {
            use keyring::Entry;
            
            let entry = Entry::new(&service, &account)
                .map_err(|e| SecurityError::KeychainError {
                    message: format!("Failed to create Linux keychain entry: {}", e),
                })?;
            
            entry.set_password(&secret)
                .map_err(|e| SecurityError::KeychainError {
                    message: format!("Failed to store Linux credential: {}", e),
                })?;
            
            Ok(())
        }).await.map_err(|e| SecurityError::KeychainError {
            message: format!("Linux keychain operation failed: {}", e),
        })?
    }

    #[cfg(target_os = "linux")]
    async fn retrieve_linux(&self, service: &str, account: &str) -> SecurityResult<String> {
        let service = service.to_string();
        let account = account.to_string();
        
        tokio::task::spawn_blocking(move || {
            use keyring::Entry;
            
            let entry = Entry::new(&service, &account)
                .map_err(|e| SecurityError::KeychainError {
                    message: format!("Failed to create Linux keychain entry: {}", e),
                })?;
            
            entry.get_password()
                .map_err(|e| SecurityError::KeychainError {
                    message: format!("Failed to retrieve Linux credential: {}", e),
                })
        }).await.map_err(|e| SecurityError::KeychainError {
            message: format!("Linux keychain operation failed: {}", e),
        })?
    }

    #[cfg(target_os = "linux")]
    async fn delete_linux(&self, service: &str, account: &str) -> SecurityResult<()> {
        let service = service.to_string();
        let account = account.to_string();
        
        tokio::task::spawn_blocking(move || {
            use keyring::Entry;
            
            let entry = Entry::new(&service, &account)
                .map_err(|e| SecurityError::KeychainError {
                    message: format!("Failed to create Linux keychain entry: {}", e),
                })?;
            
            entry.delete_password()
                .map_err(|e| SecurityError::KeychainError {
                    message: format!("Failed to delete Linux credential: {}", e),
                })?;
            
            Ok(())
        }).await.map_err(|e| SecurityError::KeychainError {
            message: format!("Linux keychain operation failed: {}", e),
        })?
    }
}

impl EncryptedStorage {
    async fn new() -> SecurityResult<Self> {
        let storage_path = std::env::temp_dir().join("ttrpg_assistant_keystore.enc");
        let encryption_key = Self::get_or_create_master_key().await?;

        Ok(Self {
            storage_path,
            encryption_key,
        })
    }

    /// Get or create a persistent master encryption key
    async fn get_or_create_master_key() -> SecurityResult<[u8; 32]> {
        use keyring::Entry;
        use sha2::{Sha256, Digest};
        use rand::RngCore;
        
        let service = "com.ttrpg.assistant.masterkey";
        let account = "encryption";
        
        // Try to retrieve existing key from native keychain
        let entry = Entry::new(service, account)
            .map_err(|e| SecurityError::KeychainError {
                message: format!("Failed to create keychain entry for master key: {}", e),
            })?;

        match entry.get_password() {
            Ok(key_string) => {
                // Decode the existing key
                let key_bytes = hex::decode(&key_string)
                    .map_err(|_| SecurityError::KeychainError {
                        message: "Invalid master key format in keychain".to_string(),
                    })?;
                
                if key_bytes.len() != 32 {
                    return Err(SecurityError::KeychainError {
                        message: "Master key has incorrect length".to_string(),
                    });
                }
                
                let mut key = [0u8; 32];
                key.copy_from_slice(&key_bytes);
                log::debug!("Retrieved existing master encryption key from keychain");
                Ok(key)
            }
            Err(_) => {
                // Generate a new key
                let mut key = [0u8; 32];
                rand::rngs::OsRng.fill_bytes(&mut key);
                
                // Store the key in native keychain
                let key_string = hex::encode(&key);
                entry.set_password(&key_string)
                    .map_err(|e| SecurityError::KeychainError {
                        message: format!("Failed to store master key in keychain: {}", e),
                    })?;
                
                log::info!("Generated and stored new master encryption key in keychain");
                Ok(key)
            }
        }
    }

    /// Derive a storage-specific key from the master key
    fn derive_storage_key(&self, storage_id: &str) -> [u8; 32] {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        
        type HmacSha256 = Hmac<Sha256>;
        
        let mut mac = HmacSha256::new_from_slice(&self.encryption_key)
            .expect("HMAC can take keys of any size");
        mac.update(b"storage_key_derivation");
        mac.update(storage_id.as_bytes());
        
        let result = mac.finalize();
        let mut key = [0u8; 32];
        key.copy_from_slice(&result.into_bytes()[..32]);
        key
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

        // Derive a storage-specific encryption key
        let storage_key = self.derive_storage_key("fallback_storage");
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&storage_key));
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
        // Use the same derived key for decryption
        let storage_key = self.derive_storage_key("fallback_storage");
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&storage_key));
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

    #[tokio::test]
    async fn test_entry_id_collision_resistance() {
        let config = SecurityConfig::default();
        let keychain = KeychainManager::new(&config).await.unwrap();

        // Test cases that would collide with simple underscore concatenation
        let test_cases = vec![
            ("my_app", "user"),
            ("my", "app_user"),
            ("service_with_underscores", "account"),
            ("service", "with_underscores_account"),
            ("a_b_c", "d_e_f"),
            ("a_b", "c_d_e_f"),
        ];

        let mut entry_ids = HashSet::new();
        
        for (service, account) in test_cases {
            let entry_id = keychain.generate_entry_id(service, account);
            
            // Verify each entry ID is unique
            assert!(!entry_ids.contains(&entry_id), 
                "Entry ID collision detected for service='{}' account='{}': {}", 
                service, account, entry_id);
            
            // Verify entry ID format
            assert!(entry_id.starts_with("v1_"), "Entry ID should start with version prefix");
            assert_eq!(entry_id.len(), 67, "Entry ID should be 67 characters (v1_ + 64 hex chars)");
            
            entry_ids.insert(entry_id);
        }
        
        // Verify reproducibility - same inputs should produce same ID
        let id1 = keychain.generate_entry_id("test_service", "test_account");
        let id2 = keychain.generate_entry_id("test_service", "test_account");
        assert_eq!(id1, id2, "Entry ID generation should be deterministic");
        
        // Verify different inputs produce different IDs
        let id3 = keychain.generate_entry_id("different_service", "test_account");
        assert_ne!(id1, id3, "Different service should produce different entry ID");
        
        let id4 = keychain.generate_entry_id("test_service", "different_account");
        assert_ne!(id1, id4, "Different account should produce different entry ID");
    }

    #[test]
    fn test_generate_entry_id_security() {
        use crate::security::SecurityConfig;
        
        // Create a test instance (not async since we're just testing the hash function)
        let config = SecurityConfig::default();
        let service_prefix = "com.ttrpg.assistant".to_string();
        let keychain_data = super::KeychainManager {
            config,
            service_prefix,
            fallback_storage: None,
        };

        // Test edge cases and potential attack vectors
        let edge_cases = vec![
            ("", ""),                          // Empty strings
            ("service", ""),                   // Empty account
            ("", "account"),                   // Empty service
            ("service\0null", "account"),      // Null bytes
            ("service\nnewline", "account"),   // Newlines
            ("service\ttab", "account"),       // Tabs
            ("🔐service", "🔑account"),        // Unicode characters
            ("very_long_service_name_that_exceeds_normal_limits_and_tests_boundary_conditions", "account"),
            ("service", "very_long_account_name_that_exceeds_normal_limits_and_tests_boundary_conditions"),
        ];

        let mut ids = HashSet::new();
        for (service, account) in edge_cases {
            let id = keychain_data.generate_entry_id(service, account);
            
            // Verify uniqueness
            assert!(!ids.contains(&id), "Collision detected for edge case: '{}', '{}'", service, account);
            
            // Verify format consistency
            assert!(id.starts_with("v1_"), "Entry ID should start with v1_ prefix");
            assert!(id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'), 
                "Entry ID should only contain alphanumeric characters and underscores");
            
            ids.insert(id);
        }
    }
}