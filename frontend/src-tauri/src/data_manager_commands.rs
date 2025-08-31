use std::sync::Arc;
use parking_lot::RwLock;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tauri::{State, Manager};
use crate::data_manager::{
    encryption::EncryptionManager,
    cache::CacheManager,
    integrity::IntegrityChecker,
    file_manager::FileManager,
    backup::BackupManager,
    salt_storage::{SaltStorage, SaltStorageConfig},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataManagerConfig {
    pub cache_size_mb: usize,
    pub encryption_key_iterations: u32,
    pub backup_retention_days: u32,
}

impl Default for DataManagerConfig {
    fn default() -> Self {
        Self {
            cache_size_mb: 256,
            encryption_key_iterations: 100_000,
            backup_retention_days: 30,
        }
    }
}

/// Thread-safe wrapper for EncryptionManager with interior mutability
pub struct ThreadSafeEncryptionManager {
    inner: Arc<RwLock<EncryptionManager>>,
}

impl ThreadSafeEncryptionManager {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(EncryptionManager::new())),
        }
    }

    pub fn initialize_with_password(&self, password: &str, salt: &[u8]) -> Result<()> {
        let mut manager = self.inner.write();
        manager.initialize_with_password(password, salt)
    }

    pub fn encrypt_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        let manager = self.inner.read();
        manager.encrypt_data(data)
    }

    pub fn decrypt_data(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        let manager = self.inner.read();
        manager.decrypt_data(encrypted_data)
    }

    pub fn is_initialized(&self) -> bool {
        let manager = self.inner.read();
        manager.is_initialized()
    }
}

impl Clone for ThreadSafeEncryptionManager {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Main data manager state that coordinates all data operations
pub struct DataManagerState {
    pub encryption: ThreadSafeEncryptionManager,
    pub cache: Arc<CacheManager>,
    pub integrity: Arc<IntegrityChecker>,
    pub file_manager: Arc<FileManager>,
    pub backup: Arc<BackupManager>,
    pub salt_storage: Arc<SaltStorage>,
    pub config: DataManagerConfig,
}

impl DataManagerState {
    pub fn new(config: DataManagerConfig, tauri_config: &tauri::Config) -> Result<Self> {
        let cache = Arc::new(CacheManager::new(config.cache_size_mb * 1024 * 1024)?);
        let encryption = ThreadSafeEncryptionManager::new();
        let integrity = Arc::new(IntegrityChecker::new());
        let file_manager = Arc::new(FileManager::new()?);
        let backup = Arc::new(BackupManager::new(config.backup_retention_days)?);
        
        // Initialize salt storage with secure configuration
        let salt_config = SaltStorageConfig::default();
        let salt_storage = Arc::new(SaltStorage::new(salt_config, tauri_config)?);

        Ok(Self {
            encryption,
            cache,
            integrity,
            file_manager,
            backup,
            salt_storage,
            config,
        })
    }
}

/// Initialize the data manager with a password
#[tauri::command]
pub async fn initialize_data_manager(
    password: String,
    salt: Option<Vec<u8>>,
    state: State<'_, DataManagerState>,
) -> Result<(), String> {
    if password.is_empty() {
        return Err("Password cannot be empty".to_string());
    }

    // Handle salt securely - either use provided salt or load/create from secure storage
    let salt = match salt {
        Some(provided_salt) => {
            if provided_salt.len() < 16 {
                return Err("Provided salt must be at least 16 bytes".to_string());
            }
            log::debug!("Using provided salt for initialization");
            
            // If salt is provided, we should still store it for future use
            if let Err(e) = state.salt_storage.store_salt(&provided_salt) {
                log::warn!("Failed to store provided salt: {}", e);
                // Don't fail initialization if storage fails - continue with provided salt
            }
            
            provided_salt
        }
        None => {
            // Use secure salt management - load existing or create new
            match state.salt_storage.get_or_create_salt(32) {
                Ok(stored_salt) => {
                    log::info!("Successfully loaded or created salt from secure storage");
                    stored_salt
                }
                Err(e) => {
                    log::error!("Critical error: Failed to manage salt storage: {}", e);
                    return Err(format!("Salt storage error: {}", e));
                }
            }
        }
    };

    // Initialize encryption with the secure salt
    if let Err(e) = state.encryption.initialize_with_password(&password, &salt) {
        log::error!("Failed to initialize encryption: {}", e);
        return Err(format!("Encryption initialization failed: {}", e));
    }

    // Initialize other components
    state.cache.clear().await;
    
    log::info!("Data manager initialized successfully with secure salt management");
    Ok(())
}

/// Check if the data manager is initialized
#[tauri::command]
pub async fn is_data_manager_initialized(
    state: State<'_, DataManagerState>,
) -> Result<bool, String> {
    Ok(state.encryption.is_initialized())
}

/// Encrypt and store data with caching
#[tauri::command]
pub async fn store_encrypted_data(
    key: String,
    data: Vec<u8>,
    use_cache: Option<bool>,
    state: State<'_, DataManagerState>,
) -> Result<String, String> {
    let use_cache = use_cache.unwrap_or(true);
    
    // Encrypt the data
    let encrypted_data = state.encryption.encrypt_data(&data)
        .map_err(|e| format!("Encryption failed: {}", e))?;
    
    // Generate a unique identifier
    let data_id = uuid::Uuid::new_v4().to_string();
    
    // Store in file system
    state.file_manager.store_file(&data_id, &encrypted_data).await
        .map_err(|e| format!("File storage failed: {}", e))?;
    
    // Update cache if requested
    if use_cache {
        state.cache.put(key.clone(), encrypted_data.clone()).await;
    }
    
    // Update integrity records
    state.integrity.add_file_record(&data_id, &encrypted_data).await
        .map_err(|e| format!("Integrity check failed: {}", e))?;
    
    log::debug!("Stored encrypted data with ID: {}", data_id);
    Ok(data_id)
}

/// Retrieve and decrypt data with caching
#[tauri::command]
pub async fn retrieve_encrypted_data(
    data_id: String,
    use_cache: Option<bool>,
    state: State<'_, DataManagerState>,
) -> Result<Vec<u8>, String> {
    let use_cache = use_cache.unwrap_or(true);
    
    // Try cache first if enabled
    let encrypted_data = if use_cache {
        if let Some(cached_data) = state.cache.get(&data_id).await {
            cached_data
        } else {
            // Load from file system
            let file_data = state.file_manager.load_file(&data_id).await
                .map_err(|e| format!("File loading failed: {}", e))?;
            
            // Update cache
            state.cache.put(data_id.clone(), file_data.clone()).await;
            file_data
        }
    } else {
        // Load directly from file system
        state.file_manager.load_file(&data_id).await
            .map_err(|e| format!("File loading failed: {}", e))?
    };
    
    // Verify integrity
    state.integrity.verify_file(&data_id, &encrypted_data).await
        .map_err(|e| format!("Integrity verification failed: {}", e))?;
    
    // Decrypt the data
    let decrypted_data = state.encryption.decrypt_data(&encrypted_data)
        .map_err(|e| format!("Decryption failed: {}", e))?;
    
    log::debug!("Retrieved and decrypted data with ID: {}", data_id);
    Ok(decrypted_data)
}

/// Delete encrypted data and clean up all references
#[tauri::command]
pub async fn delete_encrypted_data(
    data_id: String,
    state: State<'_, DataManagerState>,
) -> Result<(), String> {
    // Remove from file system
    state.file_manager.delete_file(&data_id).await
        .map_err(|e| format!("File deletion failed: {}", e))?;
    
    // Remove from cache
    state.cache.remove(&data_id).await;
    
    // Remove integrity records
    state.integrity.remove_file_record(&data_id).await
        .map_err(|e| format!("Integrity cleanup failed: {}", e))?;
    
    log::debug!("Deleted encrypted data with ID: {}", data_id);
    Ok(())
}

/// Get cache statistics
#[tauri::command]
pub async fn get_cache_stats(
    state: State<'_, DataManagerState>,
) -> Result<serde_json::Value, String> {
    let stats = state.cache.get_stats().await;
    Ok(serde_json::to_value(stats)
        .map_err(|e| format!("Failed to serialize stats: {}", e))?)
}

/// Clear cache
#[tauri::command]
pub async fn clear_cache(
    state: State<'_, DataManagerState>,
) -> Result<(), String> {
    state.cache.clear().await;
    log::info!("Cache cleared");
    Ok(())
}

/// Create a backup
#[tauri::command]
pub async fn create_backup(
    backup_name: Option<String>,
    state: State<'_, DataManagerState>,
) -> Result<String, String> {
    let backup_id = state.backup.create_backup(backup_name).await
        .map_err(|e| format!("Backup creation failed: {}", e))?;
    
    log::info!("Created backup with ID: {}", backup_id);
    Ok(backup_id)
}

/// Restore from backup
#[tauri::command]
pub async fn restore_backup(
    backup_id: String,
    state: State<'_, DataManagerState>,
) -> Result<(), String> {
    state.backup.restore_backup(&backup_id).await
        .map_err(|e| format!("Backup restoration failed: {}", e))?;
    
    // Clear cache after restore to ensure consistency
    state.cache.clear().await;
    
    log::info!("Restored from backup ID: {}", backup_id);
    Ok(())
}

/// List available backups
#[tauri::command]
pub async fn list_backups(
    state: State<'_, DataManagerState>,
) -> Result<Vec<serde_json::Value>, String> {
    let backups = state.backup.list_backups().await
        .map_err(|e| format!("Failed to list backups: {}", e))?;
    
    Ok(backups.into_iter()
        .map(|backup| serde_json::to_value(backup).unwrap_or(serde_json::Value::Null))
        .collect())
}

/// Verify data integrity
#[tauri::command]
pub async fn verify_data_integrity(
    data_id: Option<String>,
    state: State<'_, DataManagerState>,
) -> Result<serde_json::Value, String> {
    let result = if let Some(id) = data_id {
        // Verify specific file
        let file_data = state.file_manager.load_file(&id).await
            .map_err(|e| format!("File loading failed: {}", e))?;
        
        state.integrity.verify_file(&id, &file_data).await
            .map_err(|e| format!("Integrity verification failed: {}", e))?;
        
        serde_json::json!({
            "status": "valid",
            "file_id": id,
            "message": "File integrity verified"
        })
    } else {
        // Verify all files
        let verification_result = state.integrity.verify_all_files().await
            .map_err(|e| format!("Full integrity check failed: {}", e))?;
        
        serde_json::to_value(verification_result)
            .map_err(|e| format!("Failed to serialize verification result: {}", e))?
    };
    
    Ok(result)
}

/// Check if salt exists in secure storage
#[tauri::command]
pub async fn check_salt_exists(
    state: State<'_, DataManagerState>,
) -> Result<bool, String> {
    Ok(state.salt_storage.salt_exists())
}

/// Regenerate salt (WARNING: This will make existing encrypted data inaccessible!)
#[tauri::command]
pub async fn regenerate_salt(
    confirm_data_loss: bool,
    state: State<'_, DataManagerState>,
) -> Result<(), String> {
    if !confirm_data_loss {
        return Err("Salt regeneration requires explicit confirmation of data loss risk".to_string());
    }

    log::warn!("CRITICAL: Regenerating salt - existing encrypted data will become inaccessible!");
    
    // Delete existing salt
    if let Err(e) = state.salt_storage.delete_salt() {
        log::warn!("Failed to delete old salt: {}", e);
    }

    // Generate new salt
    let new_salt = state.salt_storage.get_or_create_salt(32)
        .map_err(|e| format!("Failed to generate new salt: {}", e))?;

    log::warn!("New salt generated - {} bytes. Previous encrypted data is now inaccessible.", new_salt.len());
    
    // Clear cache to prevent inconsistency
    state.cache.clear().await;
    
    Ok(())
}

/// Get salt storage status for debugging
#[tauri::command]
pub async fn get_salt_storage_status(
    state: State<'_, DataManagerState>,
) -> Result<serde_json::Value, String> {
    let salt_exists = state.salt_storage.salt_exists();
    let app_data_dir = state.salt_storage.app_data_dir.clone();
    
    Ok(serde_json::json!({
        "salt_exists": salt_exists,
        "storage_path": app_data_dir.to_string_lossy(),
        "config": {
            "use_keychain": state.salt_storage.config.use_keychain,
            "use_backup": state.salt_storage.config.use_backup,
            "app_id": state.salt_storage.config.app_id,
            "salt_file": state.salt_storage.config.salt_file,
        }
    }))
}