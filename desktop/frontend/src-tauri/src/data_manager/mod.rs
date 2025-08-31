//! Data Management System for TTRPG Assistant
//! 
//! This module provides comprehensive local data management including:
//! - Encrypted SQLite storage for structured data
//! - File system management for documents and assets
//! - Backup and restore with versioning
//! - Data migration and integrity validation
//! - Cross-platform data portability

pub mod models;
pub mod storage;
pub mod encryption;
pub mod backup;
pub mod migration;
pub mod integrity;
pub mod file_manager;
pub mod cache;

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

pub use models::*;
pub use storage::*;
pub use encryption::*;
pub use backup::*;
pub use migration::*;
pub use integrity::*;
pub use file_manager::*;
pub use cache::*;

/// Configuration for the data management system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataManagerConfig {
    /// Base directory for all data storage
    pub data_dir: PathBuf,
    /// Database file path
    pub database_path: PathBuf,
    /// Files directory for assets and documents
    pub files_dir: PathBuf,
    /// Backup directory
    pub backup_dir: PathBuf,
    /// Cache directory
    pub cache_dir: PathBuf,
    /// Enable data encryption
    pub encryption_enabled: bool,
    /// Auto-backup interval in minutes (0 = disabled)
    pub auto_backup_interval: u64,
    /// Maximum number of backups to keep
    pub max_backup_count: u32,
    /// Cache size limit in MB
    pub cache_size_limit_mb: u64,
    /// Enable integrity checking
    pub integrity_checking_enabled: bool,
    /// Integrity check interval in hours
    pub integrity_check_interval_hours: u64,
}

impl Default for DataManagerConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("ttrpg_data"),
            database_path: PathBuf::from("ttrpg_data/app.db"),
            files_dir: PathBuf::from("ttrpg_data/files"),
            backup_dir: PathBuf::from("ttrpg_data/backups"),
            cache_dir: PathBuf::from("ttrpg_data/cache"),
            encryption_enabled: true,
            auto_backup_interval: 60, // 1 hour
            max_backup_count: 10,
            cache_size_limit_mb: 500,
            integrity_checking_enabled: true,
            integrity_check_interval_hours: 24,
        }
    }
}

/// Result type for data operations
pub type DataResult<T> = Result<T, DataError>;

/// Comprehensive error type for data operations
#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
pub enum DataError {
    #[error("Database error: {message}")]
    Database { message: String },
    
    #[error("Encryption error: {message}")]
    Encryption { message: String },
    
    #[error("File system error: {message}")]
    FileSystem { message: String },
    
    #[error("Backup error: {message}")]
    Backup { message: String },
    
    #[error("Migration error: {message}")]
    Migration { message: String },
    
    #[error("Integrity error: {message}")]
    Integrity { message: String },
    
    #[error("Validation error: {field}: {message}")]
    Validation { field: String, message: String },
    
    #[error("Not found: {resource}")]
    NotFound { resource: String },
    
    #[error("Permission denied: {operation}")]
    PermissionDenied { operation: String },
    
    #[error("Configuration error: {message}")]
    Configuration { message: String },
    
    #[error("Cache error: {message}")]
    Cache { message: String },
}

/// Main data manager state
#[derive(Clone)]
pub struct DataManagerState {
    config: DataManagerConfig,
    storage: Arc<RwLock<DataStorage>>,
    encryption: Arc<EncryptionManager>,
    backup: Arc<BackupManager>,
    migration: Arc<MigrationManager>,
    integrity: Arc<IntegrityChecker>,
    file_manager: Arc<FileManager>,
    cache: Arc<RwLock<CacheManager>>,
}

impl DataManagerState {
    /// Create a new data manager with default configuration
    pub async fn new() -> DataResult<Self> {
        Self::with_config(DataManagerConfig::default()).await
    }
    
    /// Create a new data manager with custom configuration
    pub async fn with_config(config: DataManagerConfig) -> DataResult<Self> {
        // Initialize encryption manager
        let encryption = Arc::new(EncryptionManager::new(&config)?);
        
        // Initialize storage
        let storage = Arc::new(RwLock::new(DataStorage::new(&config, &encryption).await?));
        
        // Initialize backup manager
        let backup = Arc::new(BackupManager::new(&config, &encryption)?);
        
        // Initialize migration manager
        let migration = Arc::new(MigrationManager::new(&config)?);
        
        // Initialize integrity checker
        let integrity = Arc::new(IntegrityChecker::new(&config)?);
        
        // Initialize file manager
        let file_manager = Arc::new(FileManager::new(&config, &encryption)?);
        
        // Initialize cache manager
        let cache = Arc::new(RwLock::new(CacheManager::new(&config)?));
        
        Ok(Self {
            config,
            storage,
            encryption,
            backup,
            migration,
            integrity,
            file_manager,
            cache,
        })
    }
    
    /// Initialize the data manager (create directories, run migrations, etc.)
    pub async fn initialize(&self) -> DataResult<()> {
        // Create necessary directories
        self.create_directories()?;
        
        // Run any pending migrations
        self.migration.run_pending_migrations(&self.storage).await?;
        
        // Perform initial integrity check
        if self.config.integrity_checking_enabled {
            self.integrity.perform_initial_check(&self.storage).await?;
        }
        
        // Start background tasks
        self.start_background_tasks().await?;
        
        log::info!("Data manager initialized successfully");
        Ok(())
    }
    
    /// Create necessary directories
    fn create_directories(&self) -> DataResult<()> {
        let dirs = [
            &self.config.data_dir,
            &self.config.files_dir,
            &self.config.backup_dir,
            &self.config.cache_dir,
        ];
        
        for dir in &dirs {
            if let Some(parent) = dir.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| DataError::FileSystem {
                        message: format!("Failed to create directory {}: {}", parent.display(), e),
                    })?;
            }
            std::fs::create_dir_all(dir)
                .map_err(|e| DataError::FileSystem {
                    message: format!("Failed to create directory {}: {}", dir.display(), e),
                })?;
        }
        
        Ok(())
    }
    
    /// Start background tasks (auto-backup, cache cleanup, integrity checks)
    async fn start_background_tasks(&self) -> DataResult<()> {
        // Auto-backup task
        if self.config.auto_backup_interval > 0 {
            let backup_manager = self.backup.clone();
            let storage = self.storage.clone();
            let file_manager = self.file_manager.clone();
            let interval = self.config.auto_backup_interval;
            
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(
                    std::time::Duration::from_secs(interval * 60)
                );
                
                loop {
                    interval.tick().await;
                    if let Err(e) = backup_manager.create_auto_backup(&storage, &file_manager).await {
                        log::error!("Auto-backup failed: {}", e);
                    }
                }
            });
        }
        
        // Periodic integrity checks
        if self.config.integrity_checking_enabled && self.config.integrity_check_interval_hours > 0 {
            let integrity_checker = self.integrity.clone();
            let storage = self.storage.clone();
            let interval = self.config.integrity_check_interval_hours;
            
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(
                    std::time::Duration::from_secs(interval * 3600)
                );
                
                loop {
                    interval.tick().await;
                    if let Err(e) = integrity_checker.perform_scheduled_check(&storage).await {
                        log::error!("Scheduled integrity check failed: {}", e);
                    }
                }
            });
        }
        
        // Cache cleanup task
        let cache = self.cache.clone();
        let config = self.config.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600)); // 1 hour
            
            loop {
                interval.tick().await;
                let mut cache_guard = cache.write().await;
                if let Err(e) = cache_guard.cleanup_expired().await {
                    log::error!("Cache cleanup failed: {}", e);
                }
                if let Err(e) = cache_guard.enforce_size_limit(config.cache_size_limit_mb * 1024 * 1024).await {
                    log::error!("Cache size limit enforcement failed: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Get reference to storage
    pub fn storage(&self) -> &Arc<RwLock<DataStorage>> {
        &self.storage
    }
    
    /// Get reference to encryption manager
    pub fn encryption(&self) -> &Arc<EncryptionManager> {
        &self.encryption
    }
    
    /// Get reference to backup manager
    pub fn backup(&self) -> &Arc<BackupManager> {
        &self.backup
    }
    
    /// Get reference to migration manager
    pub fn migration(&self) -> &Arc<MigrationManager> {
        &self.migration
    }
    
    /// Get reference to integrity checker
    pub fn integrity(&self) -> &Arc<IntegrityChecker> {
        &self.integrity
    }
    
    /// Get reference to file manager
    pub fn file_manager(&self) -> &Arc<FileManager> {
        &self.file_manager
    }
    
    /// Get reference to cache manager
    pub fn cache(&self) -> &Arc<RwLock<CacheManager>> {
        &self.cache
    }
    
    /// Get current configuration
    pub fn config(&self) -> &DataManagerConfig {
        &self.config
    }
    
    /// Update configuration
    pub async fn update_config(&mut self, new_config: DataManagerConfig) -> DataResult<()> {
        // Validate new configuration
        self.validate_config(&new_config)?;
        
        // Update configuration
        self.config = new_config;
        
        // Recreate directories if needed
        self.create_directories()?;
        
        log::info!("Data manager configuration updated");
        Ok(())
    }
    
    /// Validate configuration
    fn validate_config(&self, config: &DataManagerConfig) -> DataResult<()> {
        if config.max_backup_count == 0 {
            return Err(DataError::Configuration {
                message: "max_backup_count must be greater than 0".to_string(),
            });
        }
        
        if config.cache_size_limit_mb == 0 {
            return Err(DataError::Configuration {
                message: "cache_size_limit_mb must be greater than 0".to_string(),
            });
        }
        
        Ok(())
    }
    
    /// Perform graceful shutdown
    pub async fn shutdown(&self) -> DataResult<()> {
        log::info!("Shutting down data manager");
        
        // Perform final backup if enabled
        if self.config.auto_backup_interval > 0 {
            let _ = self.backup.create_shutdown_backup(&self.storage, &self.file_manager).await;
        }
        
        // Close storage connections
        let storage = self.storage.write().await;
        storage.close().await?;
        
        // Clear cache
        let mut cache = self.cache.write().await;
        cache.clear().await?;
        
        log::info!("Data manager shutdown complete");
        Ok(())
    }
}