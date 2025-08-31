use anyhow::{Result, anyhow};
use aes_gcm::{Aes256Gcm, Key, Nonce, KeyInit};
use aes_gcm::aead::{Aead, OsRng, rand_core::RngCore};
use argon2::{Argon2, password_hash::{PasswordHasher, SaltString, rand_core::OsRng as ArgonRng}};
use sha2::{Sha256, Digest};
use std::convert::TryInto;

/// Thread-safe encryption manager that handles AES-256-GCM encryption
/// with Argon2 key derivation from passwords
pub struct EncryptionManager {
    cipher: Option<Aes256Gcm>,
    key_hash: Option<[u8; 32]>,
    initialized: bool,
}

impl EncryptionManager {
    /// Create a new uninitialized encryption manager
    pub fn new() -> Self {
        Self {
            cipher: None,
            key_hash: None,
            initialized: false,
        }
    }

    /// Initialize with a password using Argon2 key derivation
    pub fn initialize_with_password(&mut self, password: &str, salt: &[u8]) -> Result<()> {
        if password.is_empty() {
            return Err(anyhow!("Password cannot be empty"));
        }

        if salt.len() < 16 {
            return Err(anyhow!("Salt must be at least 16 bytes"));
        }

        // Use Argon2 for key derivation
        let argon2 = Argon2::default();
        let salt_string = SaltString::from_b64(&base64::encode(salt))
            .map_err(|e| anyhow!("Invalid salt format: {}", e))?;

        let password_hash = argon2
            .hash_password(password.as_bytes(), &salt_string)
            .map_err(|e| anyhow!("Failed to hash password: {}", e))?;

        // Extract the raw hash bytes for the key
        let key_bytes = password_hash.hash
            .ok_or_else(|| anyhow!("Failed to extract hash from password"))?
            .as_bytes();

        // Use SHA-256 to ensure we have exactly 32 bytes for AES-256
        let mut hasher = Sha256::new();
        hasher.update(key_bytes);
        let key_hash = hasher.finalize();

        // Create the AES cipher
        let key = Key::<Aes256Gcm>::from_slice(&key_hash);
        let cipher = Aes256Gcm::new(key);

        self.cipher = Some(cipher);
        self.key_hash = Some(key_hash.into());
        self.initialized = true;

        log::debug!("Encryption manager initialized with Argon2 key derivation");
        Ok(())
    }

    /// Initialize with a raw 256-bit key
    pub fn initialize_with_key(&mut self, key: &[u8; 32]) -> Result<()> {
        let aes_key = Key::<Aes256Gcm>::from_slice(key);
        let cipher = Aes256Gcm::new(aes_key);

        self.cipher = Some(cipher);
        self.key_hash = Some(*key);
        self.initialized = true;

        log::debug!("Encryption manager initialized with raw key");
        Ok(())
    }

    /// Check if the manager is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Encrypt data using AES-256-GCM
    /// Returns: [12-byte nonce][encrypted_data][16-byte tag]
    pub fn encrypt_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        let cipher = self.cipher.as_ref()
            .ok_or_else(|| anyhow!("Encryption manager not initialized"))?;

        if data.is_empty() {
            return Err(anyhow!("Cannot encrypt empty data"));
        }

        // Generate a random 96-bit nonce
        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Encrypt the data
        let ciphertext = cipher.encrypt(nonce, data)
            .map_err(|e| anyhow!("Encryption failed: {}", e))?;

        // Combine nonce + encrypted data (which includes the authentication tag)
        let mut result = Vec::with_capacity(12 + ciphertext.len());
        result.extend_from_slice(&nonce_bytes);
        result.extend_from_slice(&ciphertext);

        Ok(result)
    }

    /// Decrypt data using AES-256-GCM
    /// Expects: [12-byte nonce][encrypted_data][16-byte tag]
    pub fn decrypt_data(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        let cipher = self.cipher.as_ref()
            .ok_or_else(|| anyhow!("Encryption manager not initialized"))?;

        if encrypted_data.len() < 28 { // 12 (nonce) + 16 (minimum tag)
            return Err(anyhow!("Encrypted data too short"));
        }

        // Extract nonce and ciphertext
        let (nonce_bytes, ciphertext) = encrypted_data.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        // Decrypt the data
        let plaintext = cipher.decrypt(nonce, ciphertext)
            .map_err(|e| anyhow!("Decryption failed: {}", e))?;

        Ok(plaintext)
    }

    /// Encrypt a string and return base64-encoded result
    pub fn encrypt_string(&self, data: &str) -> Result<String> {
        let encrypted = self.encrypt_data(data.as_bytes())?;
        Ok(base64::encode(encrypted))
    }

    /// Decrypt base64-encoded string
    pub fn decrypt_string(&self, encrypted_data: &str) -> Result<String> {
        let encrypted_bytes = base64::decode(encrypted_data)
            .map_err(|e| anyhow!("Invalid base64 encoding: {}", e))?;
        
        let decrypted = self.decrypt_data(&encrypted_bytes)?;
        String::from_utf8(decrypted)
            .map_err(|e| anyhow!("Decrypted data is not valid UTF-8: {}", e))
    }

    /// Get a hash of the current encryption key (for verification)
    pub fn get_key_hash(&self) -> Option<[u8; 32]> {
        self.key_hash
    }

    /// Securely clear the encryption manager
    pub fn clear(&mut self) {
        // Zero out sensitive data
        if let Some(key_hash) = &mut self.key_hash {
            key_hash.fill(0);
        }
        
        self.cipher = None;
        self.key_hash = None;
        self.initialized = false;

        log::debug!("Encryption manager cleared");
    }

    /// Generate a secure random salt
    pub fn generate_salt() -> [u8; 32] {
        let mut salt = [0u8; 32];
        OsRng.fill_bytes(&mut salt);
        salt
    }

    /// Derive a key from password and salt without storing it
    pub fn derive_key_from_password(password: &str, salt: &[u8]) -> Result<[u8; 32]> {
        if password.is_empty() {
            return Err(anyhow!("Password cannot be empty"));
        }

        if salt.len() < 16 {
            return Err(anyhow!("Salt must be at least 16 bytes"));
        }

        let argon2 = Argon2::default();
        let salt_string = SaltString::from_b64(&base64::encode(salt))
            .map_err(|e| anyhow!("Invalid salt format: {}", e))?;

        let password_hash = argon2
            .hash_password(password.as_bytes(), &salt_string)
            .map_err(|e| anyhow!("Failed to hash password: {}", e))?;

        let key_bytes = password_hash.hash
            .ok_or_else(|| anyhow!("Failed to extract hash from password"))?
            .as_bytes();

        let mut hasher = Sha256::new();
        hasher.update(key_bytes);
        let key_hash = hasher.finalize();

        Ok(key_hash.into())
    }

    /// Verify a password against the current key
    pub fn verify_password(&self, password: &str, salt: &[u8]) -> Result<bool> {
        if !self.initialized {
            return Ok(false);
        }

        let current_hash = self.key_hash
            .ok_or_else(|| anyhow!("No key hash available"))?;

        let derived_key = Self::derive_key_from_password(password, salt)?;
        
        // Use constant-time comparison to prevent timing attacks
        Ok(constant_time_eq::constant_time_eq(&current_hash, &derived_key))
    }
}

impl Drop for EncryptionManager {
    fn drop(&mut self) {
        self.clear();
    }
}

// Ensure we don't accidentally clone sensitive data
impl std::fmt::Debug for EncryptionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptionManager")
            .field("initialized", &self.initialized)
            .field("cipher", &self.cipher.is_some())
            .field("key_hash", &"[REDACTED]")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encryption_manager_lifecycle() {
        let mut manager = EncryptionManager::new();
        assert!(!manager.is_initialized());

        let salt = EncryptionManager::generate_salt();
        manager.initialize_with_password("test_password", &salt).unwrap();
        assert!(manager.is_initialized());

        let plaintext = b"Hello, world!";
        let encrypted = manager.encrypt_data(plaintext).unwrap();
        let decrypted = manager.decrypt_data(&encrypted).unwrap();

        assert_eq!(plaintext, decrypted.as_slice());
        assert!(manager.verify_password("test_password", &salt).unwrap());
        assert!(!manager.verify_password("wrong_password", &salt).unwrap());
    }

    #[test]
    fn test_string_encryption() {
        let mut manager = EncryptionManager::new();
        let salt = EncryptionManager::generate_salt();
        manager.initialize_with_password("test_password", &salt).unwrap();

        let plaintext = "Hello, 世界! 🌍";
        let encrypted = manager.encrypt_string(plaintext).unwrap();
        let decrypted = manager.decrypt_string(&encrypted).unwrap();

        assert_eq!(plaintext, decrypted);
    }

    #[test]
    fn test_key_derivation() {
        let password = "test_password";
        let salt = EncryptionManager::generate_salt();
        
        let key1 = EncryptionManager::derive_key_from_password(password, &salt).unwrap();
        let key2 = EncryptionManager::derive_key_from_password(password, &salt).unwrap();
        
        assert_eq!(key1, key2);
        
        let different_salt = EncryptionManager::generate_salt();
        let key3 = EncryptionManager::derive_key_from_password(password, &different_salt).unwrap();
        assert_ne!(key1, key3);
    }
}