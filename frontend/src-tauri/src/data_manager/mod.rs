//! Data management module for MDMAI desktop application
//! 
//! This module provides comprehensive data management functionality including:
//! - Thread-safe encryption with AES-256-GCM
//! - High-performance caching with O(1) operations
//! - Streaming file processing for memory efficiency
//! - Data integrity verification
//! - Type-safe database backup and restore
//! 
//! All components are designed to handle large files and datasets efficiently
//! without causing memory issues or performance degradation.

pub mod encryption;
pub mod cache;
pub mod integrity;
pub mod file_manager;
pub mod backup;

pub use encryption::EncryptionManager;
pub use cache::{CacheManager, CacheStats};
pub use integrity::{IntegrityChecker, FileIntegrityRecord, VerificationResult};
pub use file_manager::{FileManager, FileMetadata, DuplicateGroup, FileOperationStats};
pub use backup::{BackupManager, BackupMetadata, DatabaseExport, SqlColumnType, TypedColumnValue};

use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::sync::Arc;

/// Configuration for the entire data management system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataManagerConfig {
    /// Maximum cache size in megabytes
    pub cache_size_mb: usize,
    
    /// Number of iterations for key derivation (Argon2)
    pub encryption_key_iterations: u32,
    
    /// Number of days to retain backups
    pub backup_retention_days: u32,
    
    /// Base directory for file storage
    pub storage_base_path: Option<String>,
    
    /// Base directory for backup storage
    pub backup_base_path: Option<String>,
    
    /// Enable integrity checking by default
    pub enable_integrity_checking: bool,
    
    /// Maximum number of concurrent file operations
    pub max_concurrent_operations: usize,
}

impl Default for DataManagerConfig {
    fn default() -> Self {
        Self {
            cache_size_mb: 256,
            encryption_key_iterations: 100_000,
            backup_retention_days: 30,
            storage_base_path: None,
            backup_base_path: None,
            enable_integrity_checking: true,
            max_concurrent_operations: 8,
        }
    }
}

/// System-wide statistics for data management operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataManagerStats {
    pub cache_stats: CacheStats,
    pub file_stats: std::collections::HashMap<String, serde_json::Value>,
    pub integrity_stats: std::collections::HashMap<String, serde_json::Value>,
    pub backup_count: usize,
    pub total_storage_bytes: u64,
    pub system_health_score: f64,
}

/// Initialize logging for the data management system
pub fn init_logging() -> Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp_secs()
        .init();
    
    log::info!("Data manager logging initialized");
    Ok(())
}

/// Validate configuration parameters
pub fn validate_config(config: &DataManagerConfig) -> Result<()> {
    if config.cache_size_mb == 0 {
        return Err(anyhow::anyhow!("Cache size must be greater than 0"));
    }
    
    if config.encryption_key_iterations < 10_000 {
        return Err(anyhow::anyhow!("Key iterations must be at least 10,000 for security"));
    }
    
    if config.backup_retention_days == 0 {
        return Err(anyhow::anyhow!("Backup retention must be at least 1 day"));
    }
    
    if config.max_concurrent_operations == 0 || config.max_concurrent_operations > 64 {
        return Err(anyhow::anyhow!("Max concurrent operations must be between 1 and 64"));
    }
    
    log::debug!("Configuration validated successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DataManagerConfig::default();
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let mut config = DataManagerConfig::default();
        config.cache_size_mb = 0;
        assert!(validate_config(&config).is_err());

        config.cache_size_mb = 256;
        config.encryption_key_iterations = 1000;
        assert!(validate_config(&config).is_err());
    }
}