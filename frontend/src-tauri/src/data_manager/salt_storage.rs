use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tauri::api::path::app_data_dir;
use tauri::Config;

/// Salt storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaltStorageConfig {
    /// Application identifier for secure storage
    pub app_id: String,
    /// Salt file name
    pub salt_file: String,
    /// Whether to use OS keychain (when available)
    pub use_keychain: bool,
    /// Backup salt to multiple locations
    pub use_backup: bool,
}

impl Default for SaltStorageConfig {
    fn default() -> Self {
        Self {
            app_id: "com.mdmai.desktop".to_string(),
            salt_file: "encryption.salt".to_string(),
            use_keychain: true,
            use_backup: true,
        }
    }
}

/// Salt metadata for validation and corruption detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaltMetadata {
    /// SHA-256 hash of the salt for integrity verification
    pub checksum: [u8; 32],
    /// Creation timestamp (Unix timestamp)
    pub created_at: u64,
    /// Application version that created this salt
    pub app_version: String,
    /// Salt length in bytes
    pub salt_length: usize,
}

impl SaltMetadata {
    pub fn new(salt: &[u8], app_version: String) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(salt);
        let checksum = hasher.finalize().into();

        Self {
            checksum,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            app_version,
            salt_length: salt.len(),
        }
    }

    pub fn verify_salt(&self, salt: &[u8]) -> bool {
        if salt.len() != self.salt_length {
            return false;
        }

        let mut hasher = Sha256::new();
        hasher.update(salt);
        let computed_checksum: [u8; 32] = hasher.finalize().into();
        
        constant_time_eq::constant_time_eq(&self.checksum, &computed_checksum)
    }
}

/// Stored salt data with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredSalt {
    pub salt: Vec<u8>,
    pub metadata: SaltMetadata,
}

/// Secure salt storage manager with cross-platform support
pub struct SaltStorage {
    pub config: SaltStorageConfig,
    pub app_data_dir: PathBuf,
}

impl SaltStorage {
    /// Create a new salt storage instance
    pub fn new(config: SaltStorageConfig, tauri_config: &Config) -> Result<Self> {
        let app_data_dir = app_data_dir(tauri_config)
            .ok_or_else(|| anyhow!("Failed to get app data directory"))?;

        // Ensure the app data directory exists
        if !app_data_dir.exists() {
            fs::create_dir_all(&app_data_dir)
                .map_err(|e| anyhow!("Failed to create app data directory: {}", e))?;
            log::info!("Created app data directory: {:?}", app_data_dir);
        }

        Ok(Self {
            config,
            app_data_dir,
        })
    }

    /// Get the path for the salt file
    fn get_salt_file_path(&self) -> PathBuf {
        self.app_data_dir.join(&self.config.salt_file)
    }

    /// Get the path for the backup salt file
    fn get_backup_salt_file_path(&self) -> PathBuf {
        self.app_data_dir.join(format!("{}.backup", &self.config.salt_file))
    }

    /// Generate a cryptographically secure salt
    pub fn generate_salt(&self, length: usize) -> Vec<u8> {
        use rand::RngCore;
        let mut salt = vec![0u8; length];
        rand::rngs::OsRng.fill_bytes(&mut salt);
        log::debug!("Generated new salt with {} bytes", length);
        salt
    }

    /// Store salt securely to filesystem with metadata
    pub fn store_salt(&self, salt: &[u8]) -> Result<()> {
        if salt.is_empty() {
            return Err(anyhow!("Cannot store empty salt"));
        }

        if salt.len() < 16 {
            return Err(anyhow!("Salt must be at least 16 bytes"));
        }

        let app_version = std::env::var("CARGO_PKG_VERSION")
            .unwrap_or_else(|_| "unknown".to_string());
        let metadata = SaltMetadata::new(salt, app_version);
        
        let stored_salt = StoredSalt {
            salt: salt.to_vec(),
            metadata,
        };

        // Serialize the salt data
        let serialized = bincode::serialize(&stored_salt)
            .map_err(|e| anyhow!("Failed to serialize salt data: {}", e))?;

        // Write to primary location
        let salt_path = self.get_salt_file_path();
        let mut file = File::create(&salt_path)
            .map_err(|e| anyhow!("Failed to create salt file: {}", e))?;
        
        file.write_all(&serialized)
            .map_err(|e| anyhow!("Failed to write salt data: {}", e))?;
        
        file.sync_all()
            .map_err(|e| anyhow!("Failed to sync salt file: {}", e))?;

        log::info!("Stored salt to: {:?}", salt_path);

        // Create backup if enabled
        if self.config.use_backup {
            let backup_path = self.get_backup_salt_file_path();
            if let Err(e) = fs::copy(&salt_path, &backup_path) {
                log::warn!("Failed to create salt backup: {}", e);
            } else {
                log::debug!("Created salt backup at: {:?}", backup_path);
            }
        }

        // Try to store in keychain if available and enabled
        if self.config.use_keychain {
            if let Err(e) = self.store_salt_in_keychain(salt) {
                log::warn!("Failed to store salt in keychain: {}", e);
                // Don't fail the operation if keychain storage fails
            }
        }

        Ok(())
    }

    /// Load salt from secure storage with validation
    pub fn load_salt(&self) -> Result<Vec<u8>> {
        // Try to load from primary location first
        let salt_result = self.load_salt_from_file(&self.get_salt_file_path());
        
        match salt_result {
            Ok(salt) => {
                log::debug!("Loaded salt from primary location");
                return Ok(salt);
            }
            Err(e) => {
                log::warn!("Failed to load salt from primary location: {}", e);
            }
        }

        // Try backup if primary fails and backup is enabled
        if self.config.use_backup {
            let backup_result = self.load_salt_from_file(&self.get_backup_salt_file_path());
            match backup_result {
                Ok(salt) => {
                    log::warn!("Loaded salt from backup location");
                    // Try to restore primary file from backup
                    if let Err(e) = self.store_salt(&salt) {
                        log::error!("Failed to restore primary salt file from backup: {}", e);
                    }
                    return Ok(salt);
                }
                Err(e) => {
                    log::warn!("Failed to load salt from backup location: {}", e);
                }
            }
        }

        // Try keychain as last resort if available
        if self.config.use_keychain {
            match self.load_salt_from_keychain() {
                Ok(salt) => {
                    log::warn!("Loaded salt from keychain as fallback");
                    // Try to restore file from keychain
                    if let Err(e) = self.store_salt(&salt) {
                        log::error!("Failed to restore salt file from keychain: {}", e);
                    }
                    return Ok(salt);
                }
                Err(e) => {
                    log::warn!("Failed to load salt from keychain: {}", e);
                }
            }
        }

        Err(anyhow!("No valid salt found in any storage location"))
    }

    /// Load and validate salt from a specific file
    fn load_salt_from_file(&self, path: &Path) -> Result<Vec<u8>> {
        if !path.exists() {
            return Err(anyhow!("Salt file does not exist: {:?}", path));
        }

        let mut file = File::open(path)
            .map_err(|e| anyhow!("Failed to open salt file: {}", e))?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| anyhow!("Failed to read salt file: {}", e))?;

        if data.is_empty() {
            return Err(anyhow!("Salt file is empty"));
        }

        // Deserialize the salt data
        let stored_salt: StoredSalt = bincode::deserialize(&data)
            .map_err(|e| anyhow!("Failed to deserialize salt data: {}", e))?;

        // Validate the salt integrity
        if !stored_salt.metadata.verify_salt(&stored_salt.salt) {
            return Err(anyhow!("Salt integrity verification failed - possible corruption"));
        }

        log::debug!("Successfully validated salt from: {:?}", path);
        Ok(stored_salt.salt)
    }

    /// Check if salt exists in storage
    pub fn salt_exists(&self) -> bool {
        let primary_exists = self.get_salt_file_path().exists();
        let backup_exists = if self.config.use_backup {
            self.get_backup_salt_file_path().exists()
        } else {
            false
        };

        primary_exists || backup_exists
    }

    /// Delete salt from all storage locations (for cleanup)
    pub fn delete_salt(&self) -> Result<()> {
        let mut errors = Vec::new();

        // Remove primary salt file
        let salt_path = self.get_salt_file_path();
        if salt_path.exists() {
            if let Err(e) = fs::remove_file(&salt_path) {
                errors.push(format!("Failed to remove primary salt file: {}", e));
            } else {
                log::info!("Removed primary salt file");
            }
        }

        // Remove backup salt file
        if self.config.use_backup {
            let backup_path = self.get_backup_salt_file_path();
            if backup_path.exists() {
                if let Err(e) = fs::remove_file(&backup_path) {
                    errors.push(format!("Failed to remove backup salt file: {}", e));
                } else {
                    log::info!("Removed backup salt file");
                }
            }
        }

        // Remove from keychain
        if self.config.use_keychain {
            if let Err(e) = self.delete_salt_from_keychain() {
                errors.push(format!("Failed to remove salt from keychain: {}", e));
            }
        }

        if !errors.is_empty() {
            return Err(anyhow!("Salt deletion errors: {}", errors.join("; ")));
        }

        Ok(())
    }

    /// Store salt in OS keychain (platform-specific implementations would go here)
    fn store_salt_in_keychain(&self, _salt: &[u8]) -> Result<()> {
        // For now, this is a placeholder. In a real implementation, you would use:
        // - Windows Credential Manager
        // - macOS Keychain
        // - Linux Secret Service API / gnome-keyring
        log::debug!("Keychain storage not implemented yet");
        Err(anyhow!("Keychain storage not implemented"))
    }

    /// Load salt from OS keychain
    fn load_salt_from_keychain(&self) -> Result<Vec<u8>> {
        // Placeholder for keychain loading
        log::debug!("Keychain loading not implemented yet");
        Err(anyhow!("Keychain loading not implemented"))
    }

    /// Delete salt from OS keychain
    fn delete_salt_from_keychain(&self) -> Result<()> {
        // Placeholder for keychain deletion
        log::debug!("Keychain deletion not implemented yet");
        Err(anyhow!("Keychain deletion not implemented"))
    }

    /// Get or create a salt - the main interface for salt management
    pub fn get_or_create_salt(&self, length: usize) -> Result<Vec<u8>> {
        // First, try to load existing salt
        match self.load_salt() {
            Ok(salt) => {
                if salt.len() >= length {
                    log::debug!("Using existing salt");
                    return Ok(salt);
                } else {
                    log::warn!("Existing salt is too short ({} bytes), generating new one", salt.len());
                }
            }
            Err(e) => {
                log::info!("No existing salt found, generating new one: {}", e);
            }
        }

        // Generate new salt if none exists or existing is invalid
        let new_salt = self.generate_salt(length);
        self.store_salt(&new_salt)?;
        
        log::info!("Created and stored new salt with {} bytes", length);
        Ok(new_salt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_config() -> tauri::Config {
        tauri::Config {
            app_url: None,
            build: tauri::config::BuildConfig::default(),
            tauri: tauri::config::TauriConfig::default(),
            package: tauri::config::PackageConfig::default(),
        }
    }

    fn create_test_storage() -> (SaltStorage, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = SaltStorageConfig::default();
        config.use_keychain = false; // Disable for testing
        
        let mut tauri_config = create_test_config();
        // Override app data dir for testing
        let storage = SaltStorage {
            config,
            app_data_dir: temp_dir.path().to_path_buf(),
        };
        
        (storage, temp_dir)
    }

    #[test]
    fn test_salt_generation() {
        let (storage, _temp_dir) = create_test_storage();
        
        let salt1 = storage.generate_salt(32);
        let salt2 = storage.generate_salt(32);
        
        assert_eq!(salt1.len(), 32);
        assert_eq!(salt2.len(), 32);
        assert_ne!(salt1, salt2); // Should be different
    }

    #[test]
    fn test_salt_storage_and_loading() {
        let (storage, _temp_dir) = create_test_storage();
        
        let original_salt = storage.generate_salt(32);
        storage.store_salt(&original_salt).unwrap();
        
        let loaded_salt = storage.load_salt().unwrap();
        assert_eq!(original_salt, loaded_salt);
    }

    #[test]
    fn test_get_or_create_salt() {
        let (storage, _temp_dir) = create_test_storage();
        
        // First call should create new salt
        let salt1 = storage.get_or_create_salt(32).unwrap();
        assert_eq!(salt1.len(), 32);
        
        // Second call should return the same salt
        let salt2 = storage.get_or_create_salt(32).unwrap();
        assert_eq!(salt1, salt2);
    }

    #[test]
    fn test_salt_metadata_validation() {
        let salt = b"test_salt_data_12345678901234567890";
        let metadata = SaltMetadata::new(salt, "1.0.0".to_string());
        
        assert!(metadata.verify_salt(salt));
        assert!(!metadata.verify_salt(b"different_salt"));
        assert!(!metadata.verify_salt(&salt[..10])); // Wrong length
    }

    #[test]
    fn test_corrupted_salt_detection() {
        let (storage, _temp_dir) = create_test_storage();
        
        let original_salt = storage.generate_salt(32);
        storage.store_salt(&original_salt).unwrap();
        
        // Corrupt the salt file
        let salt_path = storage.get_salt_file_path();
        fs::write(&salt_path, b"corrupted_data").unwrap();
        
        // Loading should fail due to corruption
        assert!(storage.load_salt().is_err());
    }
}