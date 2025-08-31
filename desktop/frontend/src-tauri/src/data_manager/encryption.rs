//! Encryption management for sensitive data
//! 
//! This module provides comprehensive encryption services including:
//! - AES-GCM encryption for data at rest
//! - Key derivation and management
//! - Secure key storage and rotation
//! - Password-based encryption with Argon2

use super::*;
use aes_gcm::{
    aead::{Aead, AeadCore, KeyInit, OsRng, rand_core::RngCore},
    Aes256Gcm, Key, Nonce,
};
use argon2::{Argon2, PasswordHasher, password_hash::{
    rand_core::OsRng as Argon2OsRng, PasswordHash, SaltString
}};
use sha2::{Sha256, Digest};
use std::fs;
use std::path::{Path, PathBuf};
use tokio::sync::RwLock;

const KEY_SIZE: usize = 32; // 256 bits for AES-256
const NONCE_SIZE: usize = 12; // 96 bits for GCM
const SALT_SIZE: usize = 32;

/// Inner state of the encryption manager (protected by RwLock)
#[derive(Debug)]
struct EncryptionState {
    master_key: Option<[u8; KEY_SIZE]>,
    initialized: bool,
}

/// Encryption manager for handling all cryptographic operations
/// Uses interior mutability to allow initialization after creation while maintaining thread safety
pub struct EncryptionManager {
    state: RwLock<EncryptionState>,
    key_file_path: PathBuf,
    config: DataManagerConfig,
}

impl EncryptionManager {
    /// Create new encryption manager
    pub fn new(config: &DataManagerConfig) -> DataResult<Self> {
        let key_file_path = config.data_dir.join(".encryption_key");
        
        let manager = Self {
            state: RwLock::new(EncryptionState {
                master_key: None,
                initialized: false,
            }),
            key_file_path,
            config: config.clone(),
        };
        
        Ok(manager)
    }
    
    /// Initialize encryption with a password
    /// This method can be called safely even after the EncryptionManager is shared in an Arc
    pub async fn initialize_with_password(&self, password: &str) -> DataResult<()> {
        if !self.config.encryption_enabled {
            let mut state = self.state.write().await;
            state.initialized = true;
            return Ok(());
        }
        
        // Check if key file already exists
        if self.key_file_path.exists() {
            // Load existing key
            self.load_key_from_password(password).await?;
        } else {
            // Generate new key and save it
            self.generate_and_save_key(password).await?;
        }
        
        let mut state = self.state.write().await;
        state.initialized = true;
        
        log::info!("Encryption initialized successfully");
        Ok(())
    }
    
    /// Generate a new master key and save it encrypted with password
    async fn generate_and_save_key(&self, password: &str) -> DataResult<()> {
        // Generate random master key
        let mut master_key = [0u8; KEY_SIZE];
        use rand::RngCore;
        OsRng.fill_bytes(&mut master_key);
        
        // Derive key encryption key from password
        let salt = SaltString::generate(&mut Argon2OsRng);
        let argon2 = Argon2::default();
        let password_hash = argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to hash password: {}", e),
            })?;
        
        // Extract key from hash
        let kek = self.derive_key_from_hash(&password_hash)?;
        
        // Encrypt master key with KEK
        let cipher = Aes256Gcm::new(&kek);
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let encrypted_key = cipher.encrypt(&nonce, master_key.as_ref())
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to encrypt master key: {}", e),
            })?;
        
        // Create key file data
        let key_data = KeyFileData {
            salt: salt.to_string(),
            nonce: hex::encode(nonce),
            encrypted_key: hex::encode(encrypted_key),
            version: 1,
            algorithm: "AES-256-GCM".to_string(),
            kdf: "Argon2id".to_string(),
            created_at: Utc::now(),
        };
        
        // Save key file
        let key_json = serde_json::to_string_pretty(&key_data)
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to serialize key data: {}", e),
            })?;
        
        fs::write(&self.key_file_path, key_json)
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to write key file: {}", e),
            })?;
        
        // Set restrictive permissions on key file
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&self.key_file_path, fs::Permissions::from_mode(0o600))
                .map_err(|e| DataError::Encryption {
                    message: format!("Failed to set key file permissions: {}", e),
                })?;
        }
        
        // Store the master key in our protected state
        let mut state = self.state.write().await;
        state.master_key = Some(master_key);
        Ok(())
    }
    
    /// Load existing key using password
    async fn load_key_from_password(&self, password: &str) -> DataResult<()> {
        // Read key file
        let key_json = fs::read_to_string(&self.key_file_path)
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to read key file: {}", e),
            })?;
        
        let key_data: KeyFileData = serde_json::from_str(&key_json)
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to parse key file: {}", e),
            })?;
        
        // Verify password and derive KEK
        let salt = SaltString::new(&key_data.salt)
            .map_err(|e| DataError::Encryption {
                message: format!("Invalid salt in key file: {}", e),
            })?;
        
        let argon2 = Argon2::default();
        let password_hash = argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to hash password: {}", e),
            })?;
        
        let kek = self.derive_key_from_hash(&password_hash)?;
        
        // Decrypt master key
        let cipher = Aes256Gcm::new(&kek);
        let nonce_bytes = hex::decode(&key_data.nonce)
            .map_err(|e| DataError::Encryption {
                message: format!("Invalid nonce in key file: {}", e),
            })?;
        let nonce = Nonce::from_slice(&nonce_bytes);
        
        let encrypted_key = hex::decode(&key_data.encrypted_key)
            .map_err(|e| DataError::Encryption {
                message: format!("Invalid encrypted key in key file: {}", e),
            })?;
        
        let master_key_bytes = cipher.decrypt(nonce, encrypted_key.as_ref())
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to decrypt master key (wrong password?): {}", e),
            })?;
        
        if master_key_bytes.len() != KEY_SIZE {
            return Err(DataError::Encryption {
                message: "Invalid master key size".to_string(),
            });
        }
        
        let mut master_key = [0u8; KEY_SIZE];
        master_key.copy_from_slice(&master_key_bytes);
        
        // Store the master key in our protected state
        let mut state = self.state.write().await;
        state.master_key = Some(master_key);
        
        Ok(())
    }
    
    /// Derive encryption key from password hash
    fn derive_key_from_hash(&self, password_hash: &PasswordHash) -> DataResult<Key<Aes256Gcm>> {
        let hash = password_hash.hash
            .ok_or_else(|| DataError::Encryption {
                message: "No hash in password hash".to_string(),
            })?;
        let hash_bytes = hash.as_bytes();
        
        // Use SHA-256 to derive a 256-bit key from the hash
        let mut hasher = Sha256::new();
        hasher.update(hash_bytes);
        let key_bytes = hasher.finalize();
        
        Ok(Key::<Aes256Gcm>::from_slice(&key_bytes).clone())
    }
    
    /// Encrypt a string
    pub async fn encrypt_string(&self, plaintext: &str) -> DataResult<String> {
        if !self.config.encryption_enabled {
            return Ok(plaintext.to_string());
        }
        
        let state = self.state.read().await;
        let master_key = state.master_key.ok_or_else(|| DataError::Encryption {
            message: "Encryption not initialized".to_string(),
        })?;
        
        let cipher = Aes256Gcm::new(aes_gcm::Key::<Aes256Gcm>::from_slice(&master_key));
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        
        let ciphertext = cipher.encrypt(&nonce, plaintext.as_bytes())
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to encrypt string: {}", e),
            })?;
        
        // Combine nonce and ciphertext
        let mut combined = Vec::new();
        combined.extend_from_slice(nonce.as_slice());
        combined.extend_from_slice(&ciphertext);
        
        Ok(hex::encode(combined))
    }
    
    /// Decrypt a string
    pub async fn decrypt_string(&self, ciphertext_hex: &str) -> DataResult<String> {
        if !self.config.encryption_enabled {
            return Ok(ciphertext_hex.to_string());
        }
        
        let state = self.state.read().await;
        let master_key = state.master_key.ok_or_else(|| DataError::Encryption {
            message: "Encryption not initialized".to_string(),
        })?;
        
        let combined = hex::decode(ciphertext_hex)
            .map_err(|e| DataError::Encryption {
                message: format!("Invalid hex encoding: {}", e),
            })?;
        
        if combined.len() < NONCE_SIZE {
            return Err(DataError::Encryption {
                message: "Ciphertext too short".to_string(),
            });
        }
        
        let (nonce_bytes, ciphertext) = combined.split_at(NONCE_SIZE);
        let nonce = Nonce::from_slice(nonce_bytes);
        
        let cipher = Aes256Gcm::new(aes_gcm::Key::<Aes256Gcm>::from_slice(&master_key));
        let plaintext = cipher.decrypt(nonce, ciphertext)
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to decrypt string: {}", e),
            })?;
        
        String::from_utf8(plaintext)
            .map_err(|e| DataError::Encryption {
                message: format!("Invalid UTF-8 in decrypted data: {}", e),
            })
    }
    
    /// Encrypt JSON data
    pub async fn encrypt_json(&self, data: &serde_json::Value) -> DataResult<String> {
        let json_string = serde_json::to_string(data)
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to serialize JSON: {}", e),
            })?;
        
        self.encrypt_string(&json_string).await
    }
    
    /// Decrypt JSON data
    pub async fn decrypt_json(&self, ciphertext_hex: &str) -> DataResult<serde_json::Value> {
        let json_string = self.decrypt_string(ciphertext_hex).await?;
        
        serde_json::from_str(&json_string)
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to parse decrypted JSON: {}", e),
            })
    }
    
    /// Encrypt file contents
    pub async fn encrypt_file(&self, file_path: &Path) -> DataResult<Vec<u8>> {
        if !self.config.encryption_enabled {
            return fs::read(file_path)
                .map_err(|e| DataError::FileSystem {
                    message: format!("Failed to read file: {}", e),
                });
        }
        
        let plaintext = fs::read(file_path)
            .map_err(|e| DataError::FileSystem {
                message: format!("Failed to read file: {}", e),
            })?;
        
        self.encrypt_bytes(&plaintext).await
    }
    
    /// Encrypt bytes directly (public for async usage)
    pub async fn encrypt_bytes(&self, plaintext: &[u8]) -> DataResult<Vec<u8>> {
        if !self.config.encryption_enabled {
            return Ok(plaintext.to_vec());
        }
        
        let state = self.state.read().await;
        let master_key = state.master_key.ok_or_else(|| DataError::Encryption {
            message: "Encryption not initialized".to_string(),
        })?;
        
        let cipher = Aes256Gcm::new(aes_gcm::Key::<Aes256Gcm>::from_slice(&master_key));
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        
        let ciphertext = cipher.encrypt(&nonce, plaintext.as_ref())
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to encrypt data: {}", e),
            })?;
        
        // Combine nonce and ciphertext
        let mut combined = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
        combined.extend_from_slice(nonce.as_slice());
        combined.extend_from_slice(&ciphertext);
        
        Ok(combined)
    }
    
    /// Decrypt file contents
    pub async fn decrypt_file_contents(&self, encrypted_data: &[u8]) -> DataResult<Vec<u8>> {
        if !self.config.encryption_enabled {
            return Ok(encrypted_data.to_vec());
        }
        
        let state = self.state.read().await;
        let master_key = state.master_key.ok_or_else(|| DataError::Encryption {
            message: "Encryption not initialized".to_string(),
        })?;
        
        if encrypted_data.len() < NONCE_SIZE {
            return Err(DataError::Encryption {
                message: "Encrypted data too short".to_string(),
            });
        }
        
        let (nonce_bytes, ciphertext) = encrypted_data.split_at(NONCE_SIZE);
        let nonce = Nonce::from_slice(nonce_bytes);
        
        let cipher = Aes256Gcm::new(aes_gcm::Key::<Aes256Gcm>::from_slice(&master_key));
        cipher.decrypt(nonce, ciphertext)
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to decrypt file contents: {}", e),
            })
    }
    
    /// Generate hash for integrity checking
    pub fn generate_hash(&self, data: &[u8]) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(data);
        hex::encode(hasher.finalize().as_bytes())
    }
    
    /// Generate hash from file using streaming I/O to avoid loading entire file into memory
    /// Uses 64KB chunks for optimal performance with large files
    pub async fn generate_hash_streaming<P: AsRef<std::path::Path>>(&self, path: P) -> Result<String, std::io::Error> {
        const CHUNK_SIZE: usize = 64 * 1024; // 64KB chunks
        
        use tokio::fs::File;
        use tokio::io::{AsyncReadExt, BufReader};
        
        let file = File::open(path.as_ref()).await?;
        let mut reader = BufReader::with_capacity(CHUNK_SIZE, file);
        let mut hasher = blake3::Hasher::new();
        let mut buffer = vec![0u8; CHUNK_SIZE];
        
        loop {
            let bytes_read = reader.read(&mut buffer).await?;
            if bytes_read == 0 {
                break; // End of file
            }
            
            // Only hash the actual bytes read, not the entire buffer
            hasher.update(&buffer[..bytes_read]);
        }
        
        Ok(hex::encode(hasher.finalize().as_bytes()))
    }
    
    /// Verify hash
    pub fn verify_hash(&self, data: &[u8], expected_hash: &str) -> bool {
        let actual_hash = self.generate_hash(data);
        actual_hash == expected_hash
    }
    
    /// Rotate master key (re-encrypt with new key)
    pub async fn rotate_key(&self, old_password: &str, new_password: &str) -> DataResult<()> {
        if !self.config.encryption_enabled {
            return Ok(());
        }
        
        // Verify old password and load current key
        self.load_key_from_password(old_password).await?;
        
        // Generate new key
        let mut new_master_key = [0u8; KEY_SIZE];
        OsRng.fill_bytes(&mut new_master_key);
        
        // Save new key with new password
        let salt = SaltString::generate(&mut Argon2OsRng);
        let argon2 = Argon2::default();
        let password_hash = argon2
            .hash_password(new_password.as_bytes(), &salt)
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to hash new password: {}", e),
            })?;
        
        let kek = self.derive_key_from_hash(&password_hash)?;
        let cipher = Aes256Gcm::new(&kek);
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        
        let encrypted_key = cipher.encrypt(&nonce, new_master_key.as_ref())
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to encrypt new master key: {}", e),
            })?;
        
        // Create new key file data
        let key_data = KeyFileData {
            salt: salt.to_string(),
            nonce: hex::encode(nonce),
            encrypted_key: hex::encode(encrypted_key),
            version: 1,
            algorithm: "AES-256-GCM".to_string(),
            kdf: "Argon2id".to_string(),
            created_at: Utc::now(),
        };
        
        // Backup old key file
        let backup_path = self.key_file_path.with_extension("key.backup");
        fs::copy(&self.key_file_path, &backup_path)
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to backup old key file: {}", e),
            })?;
        
        // Save new key file
        let key_json = serde_json::to_string_pretty(&key_data)
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to serialize new key data: {}", e),
            })?;
        
        fs::write(&self.key_file_path, key_json)
            .map_err(|e| DataError::Encryption {
                message: format!("Failed to write new key file: {}", e),
            })?;
        
        // Set permissions
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&self.key_file_path, fs::Permissions::from_mode(0o600))
                .map_err(|e| DataError::Encryption {
                    message: format!("Failed to set key file permissions: {}", e),
                })?;
        }
        
        // Store the new master key in our protected state
        let mut state = self.state.write().await;
        state.master_key = Some(new_master_key);
        
        log::info!("Master key rotated successfully");
        Ok(())
    }
    
    /// Check if encryption is initialized
    pub async fn is_initialized(&self) -> bool {
        let state = self.state.read().await;
        state.initialized && state.master_key.is_some()
    }
    
    /// Secure memory cleanup
    pub async fn cleanup(&self) {
        let mut state = self.state.write().await;
        if let Some(ref mut key) = state.master_key {
            // Zero out the key in memory
            key.fill(0);
        }
        state.master_key = None;
        state.initialized = false;
    }
}

impl Drop for EncryptionManager {
    fn drop(&mut self) {
        // Since we can't use async in Drop, we'll do a blocking cleanup
        // This is a last resort cleanup - proper cleanup should use the async cleanup method
        if let Ok(mut state) = self.state.try_write() {
            if let Some(ref mut key) = state.master_key {
                // Zero out the key in memory
                key.fill(0);
            }
            state.master_key = None;
            state.initialized = false;
        }
    }
}

/// Key file data structure
#[derive(Debug, Serialize, Deserialize)]
struct KeyFileData {
    salt: String,
    nonce: String,
    encrypted_key: String,
    version: u32,
    algorithm: String,
    kdf: String,
    created_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_encryption_roundtrip() {
        let temp_dir = tempdir().unwrap();
        let config = DataManagerConfig {
            data_dir: temp_dir.path().to_path_buf(),
            encryption_enabled: true,
            ..Default::default()
        };
        
        let encryption = EncryptionManager::new(&config).unwrap();
        encryption.initialize_with_password("test_password").await.unwrap();
        
        let original_text = "Hello, World! This is a test string with special characters: 🎲🐉";
        let encrypted = encryption.encrypt_string(original_text).await.unwrap();
        let decrypted = encryption.decrypt_string(&encrypted).await.unwrap();
        
        assert_eq!(original_text, decrypted);
        assert_ne!(original_text, encrypted);
    }
    
    #[tokio::test]
    async fn test_json_encryption() {
        let temp_dir = tempdir().unwrap();
        let config = DataManagerConfig {
            data_dir: temp_dir.path().to_path_buf(),
            encryption_enabled: true,
            ..Default::default()
        };
        
        let encryption = EncryptionManager::new(&config).unwrap();
        encryption.initialize_with_password("test_password").await.unwrap();
        
        let original_json = serde_json::json!({
            "name": "Test Campaign",
            "description": "A test campaign",
            "players": ["Alice", "Bob", "Charlie"],
            "stats": {
                "level": 5,
                "experience": 12500
            }
        });
        
        let encrypted = encryption.encrypt_json(&original_json).await.unwrap();
        let decrypted = encryption.decrypt_json(&encrypted).await.unwrap();
        
        assert_eq!(original_json, decrypted);
    }
}