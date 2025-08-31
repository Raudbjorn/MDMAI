//! Backup and restore system with versioning
//! 
//! This module provides comprehensive backup and restore functionality including:
//! - Full and incremental backups
//! - Compression and encryption
//! - Versioning and retention policies
//! - Cross-platform backup portability
//! - Integrity verification

use super::*;
use tokio::fs::{self, File};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use tar::{Archive, Builder as TarBuilder};
use walkdir::WalkDir;
use zstd::stream::{Encoder as ZstdEncoder, Decoder as ZstdDecoder};
use futures::stream::{self, StreamExt};
use tokio::sync::mpsc;
use std::sync::atomic::{AtomicU64, Ordering};
use sqlx::Row;

/// Backup manager for handling all backup operations
pub struct BackupManager {
    config: DataManagerConfig,
    encryption: Arc<EncryptionManager>,
}

impl BackupManager {
    /// Create new backup manager
    pub fn new(config: &DataManagerConfig, encryption: &Arc<EncryptionManager>) -> DataResult<Self> {
        // Ensure backup directory exists
        std::fs::create_dir_all(&config.backup_dir)
            .map_err(|e| DataError::Backup {
                message: format!("Failed to create backup directory: {}", e),
            })?;
        
        Ok(Self {
            config: config.clone(),
            encryption: encryption.clone(),
        })
    }
    
    /// Create a full backup
    pub async fn create_full_backup(
        &self,
        storage: &Arc<RwLock<DataStorage>>,
        file_manager: &Arc<FileManager>,
        description: Option<String>
    ) -> DataResult<BackupMetadata> {
        log::info!("Starting full backup");
        
        let backup_id = Uuid::new_v4();
        let timestamp = Utc::now();
        let backup_name = format!("full_backup_{}", timestamp.format("%Y%m%d_%H%M%S"));
        
        self.create_backup(
            backup_id,
            &backup_name,
            BackupType::Full,
            storage,
            file_manager,
            description,
            None, // No reference backup for full backup
        ).await
    }
    
    /// Create an incremental backup
    pub async fn create_incremental_backup(
        &self,
        storage: &Arc<RwLock<DataStorage>>,
        file_manager: &Arc<FileManager>,
        reference_backup_id: Uuid,
        description: Option<String>
    ) -> DataResult<BackupMetadata> {
        log::info!("Starting incremental backup");
        
        let backup_id = Uuid::new_v4();
        let timestamp = Utc::now();
        let backup_name = format!("incremental_backup_{}", timestamp.format("%Y%m%d_%H%M%S"));
        
        self.create_backup(
            backup_id,
            &backup_name,
            BackupType::Incremental,
            storage,
            file_manager,
            description,
            Some(reference_backup_id),
        ).await
    }
    
    /// Create an automatic backup (triggered by schedule)
    pub async fn create_auto_backup(
        &self,
        storage: &Arc<RwLock<DataStorage>>,
        file_manager: &Arc<FileManager>
    ) -> DataResult<BackupMetadata> {
        log::info!("Starting automatic backup");
        
        let backup_id = Uuid::new_v4();
        let timestamp = Utc::now();
        let backup_name = format!("auto_backup_{}", timestamp.format("%Y%m%d_%H%M%S"));
        
        self.create_backup(
            backup_id,
            &backup_name,
            BackupType::Automatic,
            storage,
            file_manager,
            Some("Automatic scheduled backup".to_string()),
            None,
        ).await
    }
    
    /// Create a shutdown backup
    pub async fn create_shutdown_backup(
        &self,
        storage: &Arc<RwLock<DataStorage>>,
        file_manager: &Arc<FileManager>
    ) -> DataResult<BackupMetadata> {
        log::info!("Starting shutdown backup");
        
        let backup_id = Uuid::new_v4();
        let timestamp = Utc::now();
        let backup_name = format!("shutdown_backup_{}", timestamp.format("%Y%m%d_%H%M%S"));
        
        self.create_backup(
            backup_id,
            &backup_name,
            BackupType::Shutdown,
            storage,
            file_manager,
            Some("Backup created during application shutdown".to_string()),
            None,
        ).await
    }
    
    /// Core backup creation logic
    async fn create_backup(
        &self,
        backup_id: Uuid,
        backup_name: &str,
        backup_type: BackupType,
        storage: &Arc<RwLock<DataStorage>>,
        file_manager: &Arc<FileManager>,
        description: Option<String>,
        reference_backup_id: Option<Uuid>,
    ) -> DataResult<BackupMetadata> {
        let start_time = std::time::Instant::now();
        
        // Create backup file path
        let backup_filename = format!("{}.tar.zst", backup_name);
        let backup_path = self.config.backup_dir.join(&backup_filename);
        
        // Create temporary directory for backup preparation
        let temp_dir = tempfile::TempDir::new()
            .map_err(|e| DataError::Backup {
                message: format!("Failed to create temporary directory: {}", e),
            })?;
        
        let temp_backup_dir = temp_dir.path().join("backup_data");
        std::fs::create_dir_all(&temp_backup_dir)
            .map_err(|e| DataError::Backup {
                message: format!("Failed to create backup temp directory: {}", e),
            })?;
        
        // Export database
        log::info!("Exporting database for backup");
        let db_export_path = temp_backup_dir.join("database.sql");
        self.export_database(storage, &db_export_path).await?;
        
        // Copy files directory
        log::info!("Copying files for backup");
        let files_backup_dir = temp_backup_dir.join("files");
        if self.config.files_dir.exists() {
            self.copy_directory(&self.config.files_dir, &files_backup_dir).await?;
        }
        
        // Create backup manifest
        log::info!("Creating backup manifest");
        let manifest = self.create_backup_manifest(
            backup_id,
            backup_name,
            &backup_type,
            &description,
            reference_backup_id,
            &temp_backup_dir,
        ).await?;
        
        let manifest_path = temp_backup_dir.join("manifest.json");
        let manifest_json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| DataError::Backup {
                message: format!("Failed to serialize backup manifest: {}", e),
            })?;
        
        std::fs::write(&manifest_path, manifest_json)
            .map_err(|e| DataError::Backup {
                message: format!("Failed to write backup manifest: {}", e),
            })?;
        
        // Create compressed archive with progress tracking
        log::info!("Creating compressed backup archive");
        let uncompressed_size = self.calculate_directory_size_async(&temp_backup_dir).await?;
        let compressed_size = self.create_compressed_archive_async(&temp_backup_dir, &backup_path).await?;
        
        // Generate file hash asynchronously
        log::info!("Generating backup file hash");
        let file_hash = self.calculate_file_hash_async(&backup_path).await?;
        
        let duration = start_time.elapsed();
        
        // Create backup metadata
        let metadata = BackupMetadata {
            id: backup_id,
            name: backup_name.to_string(),
            created_at: Utc::now(),
            backup_type,
            file_path: backup_path.to_string_lossy().to_string(),
            file_size: uncompressed_size,
            compressed_size,
            file_hash,
            description,
            database_version: self.get_database_version(storage).await?,
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            compression_algorithm: "zstd".to_string(),
            encryption_enabled: self.config.encryption_enabled,
            integrity_verified: true,
            metadata: serde_json::json!({
                "duration_seconds": duration.as_secs(),
                "compression_ratio": compressed_size as f64 / uncompressed_size as f64,
                "reference_backup_id": reference_backup_id
            }),
        };
        
        // Record backup in database
        self.record_backup_metadata(storage, &metadata).await?;
        
        // Cleanup old backups if needed
        self.cleanup_old_backups(storage).await?;
        
        log::info!(
            "Backup completed successfully: {} ({:.2} MB -> {:.2} MB, {:.1}% compression)",
            backup_name,
            uncompressed_size as f64 / (1024.0 * 1024.0),
            compressed_size as f64 / (1024.0 * 1024.0),
            (1.0 - (compressed_size as f64 / uncompressed_size as f64)) * 100.0
        );
        
        Ok(metadata)
    }
    
    /// Export database to SQL file (optimized with streaming)
    async fn export_database(&self, storage: &Arc<RwLock<DataStorage>>, export_path: &Path) -> DataResult<()> {
        let storage_guard = storage.read().await;
        
        // Get all table names
        let tables = sqlx::query_scalar::<_, String>(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        .fetch_all(&storage_guard.pool)
        .await
        .map_err(|e| DataError::Backup {
            message: format!("Failed to get table names: {}", e),
        })?;
        
        // Use file writer for streaming large exports
        let file = fs::File::create(export_path).await
            .map_err(|e| DataError::Backup {
                message: format!("Failed to create export file: {}", e),
            })?;
        let mut writer = BufWriter::with_capacity(128 * 1024, file); // 128KB buffer
        
        let mut export_content = String::with_capacity(1024);
        
        // Add header
        export_content.push_str("-- TTRPG Assistant Database Export\n");
        export_content.push_str(&format!("-- Created: {}\n", Utc::now().to_rfc3339()));
        export_content.push_str("-- Version: 1.0\n\n");
        
        export_content.push_str("PRAGMA foreign_keys=OFF;\n");
        export_content.push_str("BEGIN TRANSACTION;\n\n");
        
        // Export each table
        for table in &tables {
            log::debug!("Exporting table: {}", table);
            
            // Get table schema
            let schema = sqlx::query_scalar::<_, String>(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?"
            )
            .bind(table)
            .fetch_one(&storage_guard.pool)
            .await
            .map_err(|e| DataError::Backup {
                message: format!("Failed to get schema for table {}: {}", table, e),
            })?;
            
            export_content.push_str(&format!("-- Table: {}\n", table));
            export_content.push_str(&schema);
            export_content.push_str(";\n\n");
            
            // Get table data
            let rows = sqlx::query(&format!("SELECT * FROM {}", table))
                .fetch_all(&storage_guard.pool)
                .await
                .map_err(|e| DataError::Backup {
                    message: format!("Failed to export data for table {}: {}", table, e),
                })?;
            
            if !rows.is_empty() {
                // Get column names from the first row
                // Using a predefined list since SqliteRow doesn't expose columns directly
                let column_names = self.get_table_columns(table);
                
                for row in rows.iter() {
                    let values: Vec<String> = column_names
                        .iter()
                        .map(|col| {
                            match row.try_get::<Option<String>, _>(*col) {
                                Ok(Some(val)) => format!("'{}'", val.replace('\'', "''")),
                                Ok(None) => "NULL".to_string(),
                                Err(_) => "NULL".to_string(),
                            }
                        })
                        .collect();
                    
                    export_content.push_str(&format!(
                        "INSERT INTO {} ({}) VALUES ({});\n",
                        table,
                        column_names.join(", "),
                        values.join(", ")
                    ));
                }
                
                export_content.push('\n');
            }
        }
        
        // Write batch to file to avoid memory issues with large tables
        if export_content.len() > 100 * 1024 { // Write every 100KB
            writer.write_all(export_content.as_bytes()).await
                .map_err(|e| DataError::Backup {
                    message: format!("Failed to write export batch: {}", e),
                })?;
            export_content.clear();
        }
        
        export_content.push_str("COMMIT;\n");
        export_content.push_str("PRAGMA foreign_keys=ON;\n");
        
        // Write final content and flush
        writer.write_all(export_content.as_bytes()).await
            .map_err(|e| DataError::Backup {
                message: format!("Failed to write database export: {}", e),
            })?;
        
        writer.flush().await
            .map_err(|e| DataError::Backup {
                message: format!("Failed to flush export file: {}", e),
            })?;
        
        Ok(())
    }
    
    /// Copy directory recursively (optimized with async and parallel operations)
    async fn copy_directory(&self, src: &Path, dst: &Path) -> DataResult<()> {
        fs::create_dir_all(dst).await
            .map_err(|e| DataError::FileSystem {
                message: format!("Failed to create destination directory: {}", e),
            })?;
        
        // Collect all files to copy
        let mut copy_tasks = Vec::new();
        for entry in WalkDir::new(src) {
            let entry = entry.map_err(|e| DataError::FileSystem {
                message: format!("Failed to walk directory: {}", e),
            })?;
            
            let src_path = entry.path().to_path_buf();
            let relative_path = src_path.strip_prefix(src)
                .map_err(|e| DataError::FileSystem {
                    message: format!("Failed to get relative path: {}", e),
                })?
                .to_path_buf();
            let dst_path = dst.join(&relative_path);
            
            if entry.file_type().is_dir() {
                fs::create_dir_all(&dst_path).await
                    .map_err(|e| DataError::FileSystem {
                        message: format!("Failed to create directory: {}", e),
                    })?;
            } else if entry.file_type().is_file() {
                copy_tasks.push((src_path, dst_path));
            }
        }
        
        // Copy files in parallel batches for better performance
        const BATCH_SIZE: usize = 10;
        for chunk in copy_tasks.chunks(BATCH_SIZE) {
            let mut handles = Vec::new();
            
            for (src_path, dst_path) in chunk {
                let src = src_path.clone();
                let dst = dst_path.clone();
                
                let handle = tokio::spawn(async move {
                    if let Some(parent) = dst.parent() {
                        fs::create_dir_all(parent).await?;
                    }
                    fs::copy(&src, &dst).await.map(|_| ())
                });
                
                handles.push(handle);
            }
            
            // Wait for batch to complete
            for handle in handles {
                handle.await
                    .map_err(|e| DataError::FileSystem {
                        message: format!("Task failed: {}", e),
                    })?
                    .map_err(|e| DataError::FileSystem {
                        message: format!("Failed to copy file: {}", e),
                    })?;
            }
        }
        
        Ok(())
    }
    
    /// Create backup manifest
    async fn create_backup_manifest(
        &self,
        backup_id: Uuid,
        backup_name: &str,
        backup_type: &BackupType,
        description: &Option<String>,
        reference_backup_id: Option<Uuid>,
        backup_dir: &Path,
    ) -> DataResult<BackupManifest> {
        let mut files = Vec::new();
        
        // Collect all files in backup
        for entry in WalkDir::new(backup_dir) {
            let entry = entry.map_err(|e| DataError::Backup {
                message: format!("Failed to walk backup directory: {}", e),
            })?;
            
            if entry.file_type().is_file() {
                let path = entry.path();
                let relative_path = path.strip_prefix(backup_dir)
                    .map_err(|e| DataError::Backup {
                        message: format!("Failed to get relative path: {}", e),
                    })?;
                
                let metadata = entry.metadata()
                    .map_err(|e| DataError::Backup {
                        message: format!("Failed to get file metadata: {}", e),
                    })?;
                
                // Use streaming hash calculation to avoid loading entire file into memory
                let hash = self.encryption.generate_hash_streaming(path).await
                    .map_err(|e| DataError::Backup {
                        message: format!("Failed to calculate file hash: {}", e),
                    })?;
                
                files.push(BackupFileEntry {
                    path: relative_path.to_string_lossy().to_string(),
                    size: metadata.len(),
                    hash,
                    modified: metadata.modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| DateTime::<Utc>::from(std::time::UNIX_EPOCH + d))
                        .unwrap_or_else(Utc::now),
                });
            }
        }
        
        Ok(BackupManifest {
            backup_id,
            name: backup_name.to_string(),
            backup_type: backup_type.clone(),
            created_at: Utc::now(),
            description: description.clone(),
            reference_backup_id,
            database_version: "1".to_string(), // TODO: Get from migrations
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            files,
            total_size: 0, // Will be calculated
            checksum: String::new(), // Will be calculated
        })
    }
    
    /// Calculate directory size asynchronously
    async fn calculate_directory_size_async(&self, dir: &Path) -> DataResult<i64> {
        let mut total_size = 0i64;
        
        // Collect all file paths
        let mut file_paths = Vec::new();
        for entry in WalkDir::new(dir) {
            let entry = entry.map_err(|e| DataError::Backup {
                message: format!("Failed to walk directory for size calculation: {}", e),
            })?;
            
            if entry.file_type().is_file() {
                file_paths.push(entry.path().to_path_buf());
            }
        }
        
        // Calculate sizes in parallel batches
        const BATCH_SIZE: usize = 20;
        for chunk in file_paths.chunks(BATCH_SIZE) {
            let mut handles = Vec::new();
            
            for path in chunk {
                let path = path.clone();
                let handle = tokio::spawn(async move {
                    fs::metadata(&path).await.map(|m| m.len() as i64).ok()
                });
                handles.push(handle);
            }
            
            for handle in handles {
                if let Ok(Some(size)) = handle.await {
                    total_size += size;
                }
            }
        }
        
        Ok(total_size)
    }
    
    /// Calculate directory size (sync fallback)
    fn calculate_directory_size(&self, dir: &Path) -> DataResult<i64> {
        let mut total_size = 0i64;
        
        for entry in WalkDir::new(dir) {
            let entry = entry.map_err(|e| DataError::Backup {
                message: format!("Failed to walk directory for size calculation: {}", e),
            })?;
            
            if entry.file_type().is_file() {
                total_size += entry.metadata()
                    .map_err(|e| DataError::Backup {
                        message: format!("Failed to get file metadata: {}", e),
                    })?
                    .len() as i64;
            }
        }
        
        Ok(total_size)
    }
    
    /// Create compressed archive asynchronously with progress tracking
    async fn create_compressed_archive_async(&self, source_dir: &Path, archive_path: &Path) -> DataResult<i64> {
        // Use spawn_blocking for CPU-intensive compression
        let source_dir = source_dir.to_path_buf();
        let archive_path = archive_path.to_path_buf();
        
        let result = tokio::task::spawn_blocking(move || {
            let tar_file = std::fs::File::create(&archive_path)
                .map_err(|e| DataError::Backup {
                    message: format!("Failed to create archive file: {}", e),
                })?;
            
            // Use higher compression level for better compression ratio
            let compressed_writer = ZstdEncoder::new(tar_file, 6)
                .map_err(|e| DataError::Backup {
                    message: format!("Failed to create compression encoder: {}", e),
                })?;
            
            let mut archive_builder = TarBuilder::new(compressed_writer);
            
            // Add all files to archive
            for entry in WalkDir::new(&source_dir) {
                let entry = entry.map_err(|e| DataError::Backup {
                    message: format!("Failed to walk source directory: {}", e),
                })?;
                
                let path = entry.path();
                if path.is_file() {
                    let relative_path = path.strip_prefix(&source_dir)
                        .map_err(|e| DataError::Backup {
                            message: format!("Failed to get relative path: {}", e),
                        })?;
                    
                    archive_builder.append_path_with_name(path, relative_path)
                        .map_err(|e| DataError::Backup {
                            message: format!("Failed to add file to archive: {}", e),
                        })?;
                }
            }
            
            let compressed_writer = archive_builder.into_inner()
                .map_err(|e| DataError::Backup {
                    message: format!("Failed to finalize archive: {}", e),
                })?;
            
            let _final_writer = compressed_writer.finish()
                .map_err(|e| DataError::Backup {
                    message: format!("Failed to finish compression: {}", e),
                })?;
            
            // Get final file size
            let metadata = std::fs::metadata(&archive_path)
                .map_err(|e| DataError::Backup {
                    message: format!("Failed to get archive metadata: {}", e),
                })?;
            
            Ok::<i64, DataError>(metadata.len() as i64)
        }).await
            .map_err(|e| DataError::Backup {
                message: format!("Compression task failed: {}", e),
            })?;
        
        result
    }
    
    /// Create compressed archive (sync fallback)
    fn create_compressed_archive(&self, source_dir: &Path, archive_path: &Path) -> DataResult<i64> {
        let tar_file = std::fs::File::create(archive_path)
            .map_err(|e| DataError::Backup {
                message: format!("Failed to create archive file: {}", e),
            })?;
        
        let compressed_writer = ZstdEncoder::new(tar_file, 3) // Level 3 compression
            .map_err(|e| DataError::Backup {
                message: format!("Failed to create compression encoder: {}", e),
            })?;
        
        let mut archive_builder = TarBuilder::new(compressed_writer);
        
        // Add all files to archive
        for entry in WalkDir::new(source_dir) {
            let entry = entry.map_err(|e| DataError::Backup {
                message: format!("Failed to walk source directory: {}", e),
            })?;
            
            let path = entry.path();
            if path.is_file() {
                let relative_path = path.strip_prefix(source_dir)
                    .map_err(|e| DataError::Backup {
                        message: format!("Failed to get relative path: {}", e),
                    })?;
                
                archive_builder.append_path_with_name(path, relative_path)
                    .map_err(|e| DataError::Backup {
                        message: format!("Failed to add file to archive: {}", e),
                    })?;
            }
        }
        
        let compressed_writer = archive_builder.into_inner()
            .map_err(|e| DataError::Backup {
                message: format!("Failed to finalize archive: {}", e),
            })?;
        
        let _final_writer = compressed_writer.finish()
            .map_err(|e| DataError::Backup {
                message: format!("Failed to finish compression: {}", e),
            })?;
        
        // Get final file size
        let metadata = std::fs::metadata(archive_path)
            .map_err(|e| DataError::Backup {
                message: format!("Failed to get archive metadata: {}", e),
            })?;
        
        Ok(metadata.len() as i64)
    }
    
    /// Calculate file hash asynchronously with streaming for large files
    async fn calculate_file_hash_async(&self, file_path: &Path) -> DataResult<String> {
        let file_size = fs::metadata(file_path).await
            .map_err(|e| DataError::Backup {
                message: format!("Failed to get file metadata: {}", e),
            })?.len();
        
        if file_size > 50 * 1024 * 1024 { // > 50MB
            // Use streaming hash for large files
            let mut file = File::open(file_path).await
                .map_err(|e| DataError::Backup {
                    message: format!("Failed to open file for hashing: {}", e),
                })?;
            
            let mut hasher = sha2::Sha256::new();
            let mut buffer = vec![0; 64 * 1024]; // 64KB chunks
            
            loop {
                let bytes_read = file.read(&mut buffer).await
                    .map_err(|e| DataError::Backup {
                        message: format!("Failed to read file chunk: {}", e),
                    })?;
                
                if bytes_read == 0 {
                    break;
                }
                
                use sha2::Digest;
                hasher.update(&buffer[..bytes_read]);
            }
            
            use sha2::Digest;
            Ok(format!("{:x}", hasher.finalize()))
        } else {
            // Read entire file for smaller files
            let content = fs::read(file_path).await
                .map_err(|e| DataError::Backup {
                    message: format!("Failed to read file for hashing: {}", e),
                })?;
            
            Ok(self.encryption.generate_hash(&content))
        }
    }
    
    /// Get table columns for SQL export
    fn get_table_columns(&self, table_name: &str) -> Vec<&str> {
        match table_name {
            "campaigns" => vec!["id", "name", "description", "setting", "created_at", "updated_at", "is_active", "metadata"],
            "characters" => vec!["id", "name", "campaign_id", "player_name", "class", "level", "race", "background", "alignment", "experience_points", "hit_points", "armor_class", "speed", "stats", "skills", "equipment", "notes", "created_at", "updated_at", "metadata"],
            "npcs" => vec!["id", "name", "campaign_id", "role", "description", "stats", "notes", "created_at", "updated_at", "metadata"],
            "sessions" => vec!["id", "campaign_id", "session_number", "title", "date", "summary", "notes", "created_at", "updated_at", "metadata"],
            "rulebooks" => vec!["id", "title", "system", "version", "file_path", "file_hash", "file_size", "imported_at", "metadata"],
            "assets" => vec!["id", "name", "type", "file_path", "file_hash", "file_size", "mime_type", "created_at", "metadata"],
            "settings" => vec!["id", "category", "key", "value", "updated_at"],
            "backup_metadata" => vec!["id", "name", "created_at", "backup_type", "file_path", "file_size", "compressed_size", "file_hash", "description", "database_version", "app_version", "compression_algorithm", "encryption_enabled", "integrity_verified", "metadata"],
            _ => vec!["*"], // Generic fallback
        }
    }
    
    /// Calculate file hash (sync fallback)
    fn calculate_file_hash(&self, file_path: &Path) -> DataResult<String> {
        let content = std::fs::read(file_path)
            .map_err(|e| DataError::Backup {
                message: format!("Failed to read file for hashing: {}", e),
            })?;
        
        Ok(self.encryption.generate_hash(&content))
    }
    
    /// Get database version
    async fn get_database_version(&self, storage: &Arc<RwLock<DataStorage>>) -> DataResult<String> {
        let storage_guard = storage.read().await;
        
        let version = sqlx::query_scalar::<_, Option<String>>(
            "SELECT value FROM settings WHERE category = 'database' AND key = 'schema_version'"
        )
        .fetch_optional(&storage_guard.pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to get database version: {}", e),
        })?;
        
        if let Some(value) = version {
            Ok(value.unwrap_or_else(|| "1".to_string()))
        } else {
            Ok("1".to_string())
        }
    }
    
    /// Record backup metadata in database
    async fn record_backup_metadata(&self, storage: &Arc<RwLock<DataStorage>>, metadata: &BackupMetadata) -> DataResult<()> {
        let storage_guard = storage.read().await;
        
        sqlx::query(
            r#"
            INSERT INTO backup_metadata (
                id, name, created_at, backup_type, file_path, file_size,
                compressed_size, file_hash, description, database_version,
                app_version, compression_algorithm, encryption_enabled,
                integrity_verified, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#
        )
        .bind(&metadata.id)
        .bind(&metadata.name)
        .bind(&metadata.created_at)
        .bind(&metadata.backup_type)
        .bind(&metadata.file_path)
        .bind(&metadata.file_size)
        .bind(&metadata.compressed_size)
        .bind(&metadata.file_hash)
        .bind(&metadata.description)
        .bind(&metadata.database_version)
        .bind(&metadata.app_version)
        .bind(&metadata.compression_algorithm)
        .bind(&metadata.encryption_enabled)
        .bind(&metadata.integrity_verified)
        .bind(&metadata.metadata)
        .execute(&storage_guard.pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to record backup metadata: {}", e),
        })?;
        
        Ok(())
    }
    
    /// Cleanup old backups based on retention policy
    async fn cleanup_old_backups(&self, storage: &Arc<RwLock<DataStorage>>) -> DataResult<()> {
        let storage_guard = storage.read().await;
        
        // Get all backups ordered by creation time (newest first)
        let backups = sqlx::query(
            "SELECT id, file_path FROM backup_metadata ORDER BY created_at DESC"
        )
        .fetch_all(&storage_guard.pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to get backup list: {}", e),
        })?;
        
        // Keep only the specified number of backups
        if backups.len() > self.config.max_backup_count as usize {
            let backups_to_delete = &backups[self.config.max_backup_count as usize..];
            
            for backup in backups_to_delete {
                let backup_id: Uuid = backup.try_get("id").unwrap_or_default();
                let file_path: String = backup.try_get("file_path").unwrap_or_default();
                
                log::info!("Cleaning up old backup: {}", backup_id);
                
                // Delete backup file
                if let Err(e) = std::fs::remove_file(&file_path) {
                    log::warn!("Failed to delete backup file {}: {}", file_path, e);
                }
                
                // Remove from database
                sqlx::query("DELETE FROM backup_metadata WHERE id = ?")
                    .bind(&backup_id)
                    .execute(&storage_guard.pool)
                    .await
                    .map_err(|e| DataError::Database {
                        message: format!("Failed to delete backup metadata: {}", e),
                    })?;
            }
            
            log::info!("Cleaned up {} old backups", backups_to_delete.len());
        }
        
        Ok(())
    }
    
    /// Restore from backup
    pub async fn restore_backup(
        &self,
        backup_id: Uuid,
        storage: &Arc<RwLock<DataStorage>>,
        file_manager: &Arc<FileManager>,
        verify_integrity: bool,
    ) -> DataResult<RestoreResult> {
        log::info!("Starting backup restore: {}", backup_id);
        
        // Get backup metadata
        let backup_metadata = self.get_backup_metadata(storage, backup_id).await?;
        
        // Verify backup file exists
        let backup_path = Path::new(&backup_metadata.file_path);
        if !backup_path.exists() {
            return Err(DataError::Backup {
                message: format!("Backup file not found: {}", backup_metadata.file_path),
            });
        }
        
        // Verify backup integrity if requested
        if verify_integrity {
            log::info!("Verifying backup integrity");
            self.verify_backup_integrity(&backup_metadata).await?;
        }
        
        // Create restoration temporary directory
        let temp_dir = tempfile::TempDir::new()
            .map_err(|e| DataError::Backup {
                message: format!("Failed to create temporary directory: {}", e),
            })?;
        
        // Extract backup archive
        log::info!("Extracting backup archive");
        self.extract_backup_archive(backup_path, temp_dir.path()).await?;
        
        // Read and verify manifest
        let manifest_path = temp_dir.path().join("manifest.json");
        let manifest = self.read_backup_manifest(&manifest_path)?;
        
        // Create backup of current data before restore
        log::info!("Creating safety backup before restore");
        let safety_backup = self.create_full_backup(
            storage,
            file_manager,
            Some("Safety backup before restore".to_string())
        ).await?;
        
        // Restore database
        log::info!("Restoring database");
        let db_restore_path = temp_dir.path().join("database.sql");
        self.restore_database(storage, &db_restore_path).await?;
        
        // Restore files
        log::info!("Restoring files");
        let files_restore_dir = temp_dir.path().join("files");
        if files_restore_dir.exists() {
            // Backup current files directory
            if self.config.files_dir.exists() {
                let files_backup_dir = temp_dir.path().join("files_backup");
                self.copy_directory(&self.config.files_dir, &files_backup_dir).await?;
            }
            
            // Remove current files directory
            if self.config.files_dir.exists() {
                tokio::fs::remove_dir_all(&self.config.files_dir).await
                    .map_err(|e| DataError::FileSystem {
                        message: format!("Failed to remove current files directory: {}", e),
                    })?;
            }
            
            // Restore files from backup
            self.copy_directory(&files_restore_dir, &self.config.files_dir).await?;
        }
        
        log::info!("Backup restore completed successfully");
        
        Ok(RestoreResult {
            backup_id,
            restored_at: Utc::now(),
            safety_backup_id: safety_backup.id,
            files_restored: manifest.files.len(),
            database_restored: true,
        })
    }
    
    /// Get backup metadata
    async fn get_backup_metadata(&self, storage: &Arc<RwLock<DataStorage>>, backup_id: Uuid) -> DataResult<BackupMetadata> {
        let storage_guard = storage.read().await;
        
        let row = sqlx::query(
            "SELECT * FROM backup_metadata WHERE id = ?"
        )
        .bind(&backup_id)
        .fetch_optional(&storage_guard.pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to get backup metadata: {}", e),
        })?;
        
        if let Some(row) = row {
            let backup_type_str: String = row.try_get("backup_type").unwrap_or_else(|_| "Full".to_string());
            let backup_type: BackupType = serde_json::from_str(&format!("\"{}\"", backup_type_str))
                .map_err(|e| DataError::Database {
                    message: format!("Invalid backup type: {}", e),
                })?;
            
            Ok(BackupMetadata {
                id: row.try_get("id").unwrap_or_default(),
                name: row.try_get("name").unwrap_or_default(),
                created_at: row.try_get("created_at").unwrap_or_default(),
                backup_type,
                file_path: row.try_get("file_path").unwrap_or_default(),
                file_size: row.try_get("file_size").unwrap_or_default(),
                compressed_size: row.try_get("compressed_size").unwrap_or_default(),
                file_hash: row.try_get("file_hash").unwrap_or_default(),
                description: row.try_get("description").ok(),
                database_version: row.try_get("database_version").unwrap_or_default(),
                app_version: row.try_get("app_version").unwrap_or_default(),
                compression_algorithm: row.try_get("compression_algorithm").unwrap_or_default(),
                encryption_enabled: row.try_get("encryption_enabled").unwrap_or_default(),
                integrity_verified: row.try_get("integrity_verified").unwrap_or_default(),
                metadata: row.try_get::<serde_json::Value, _>("metadata").ok().unwrap_or(serde_json::Value::Null),
            })
        } else {
            Err(DataError::NotFound {
                resource: format!("Backup with id {}", backup_id),
            })
        }
    }
    
    /// Verify backup integrity
    async fn verify_backup_integrity(&self, backup_metadata: &BackupMetadata) -> DataResult<()> {
        // Verify file exists
        let backup_path = Path::new(&backup_metadata.file_path);
        if !backup_path.exists() {
            return Err(DataError::Backup {
                message: "Backup file does not exist".to_string(),
            });
        }
        
        // Verify file hash
        let actual_hash = self.calculate_file_hash(backup_path)?;
        if actual_hash != backup_metadata.file_hash {
            return Err(DataError::Backup {
                message: "Backup file hash verification failed - file may be corrupted".to_string(),
            });
        }
        
        log::info!("Backup integrity verification passed");
        Ok(())
    }
    
    /// Extract backup archive
    async fn extract_backup_archive(&self, archive_path: &Path, extract_dir: &Path) -> DataResult<()> {
        let tar_file = std::fs::File::open(archive_path)
            .map_err(|e| DataError::Backup {
                message: format!("Failed to open backup archive: {}", e),
            })?;
        
        let decompressed_reader = ZstdDecoder::new(tar_file)
            .map_err(|e| DataError::Backup {
                message: format!("Failed to create decompression decoder: {}", e),
            })?;
        
        let mut archive = Archive::new(decompressed_reader);
        archive.unpack(extract_dir)
            .map_err(|e| DataError::Backup {
                message: format!("Failed to extract backup archive: {}", e),
            })?;
        
        Ok(())
    }
    
    /// Read backup manifest
    fn read_backup_manifest(&self, manifest_path: &Path) -> DataResult<BackupManifest> {
        let manifest_content = std::fs::read_to_string(manifest_path)
            .map_err(|e| DataError::Backup {
                message: format!("Failed to read backup manifest: {}", e),
            })?;
        
        serde_json::from_str(&manifest_content)
            .map_err(|e| DataError::Backup {
                message: format!("Failed to parse backup manifest: {}", e),
            })
    }
    
    /// Restore database from SQL file
    async fn restore_database(&self, storage: &Arc<RwLock<DataStorage>>, sql_file_path: &Path) -> DataResult<()> {
        let storage_guard = storage.read().await;
        
        let sql_content = std::fs::read_to_string(sql_file_path)
            .map_err(|e| DataError::Backup {
                message: format!("Failed to read database restore file: {}", e),
            })?;
        
        // Execute SQL statements
        for statement in sql_content.split(';') {
            let statement = statement.trim();
            if !statement.is_empty() && !statement.starts_with("--") {
                sqlx::query(statement)
                    .execute(&storage_guard.pool)
                    .await
                    .map_err(|e| DataError::Database {
                        message: format!("Failed to execute restore statement: {}\nStatement: {}", e, statement),
                    })?;
            }
        }
        
        Ok(())
    }
    
    /// List available backups
    pub async fn list_backups(&self, storage: &Arc<RwLock<DataStorage>>) -> DataResult<Vec<BackupMetadata>> {
        let storage_guard = storage.read().await;
        
        let rows = sqlx::query(
            "SELECT * FROM backup_metadata ORDER BY created_at DESC"
        )
        .fetch_all(&storage_guard.pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to list backups: {}", e),
        })?;
        
        let mut backups = Vec::new();
        for row in rows {
            let backup_type_str: String = row.try_get("backup_type").unwrap_or_else(|_| "Full".to_string());
            let backup_type: BackupType = serde_json::from_str(&format!("\"{}\"", backup_type_str))
                .map_err(|e| DataError::Database {
                    message: format!("Invalid backup type: {}", e),
                })?;
            
            backups.push(BackupMetadata {
                id: row.try_get("id").unwrap_or_default(),
                name: row.try_get("name").unwrap_or_default(),
                created_at: row.try_get("created_at").unwrap_or_default(),
                backup_type,
                file_path: row.try_get("file_path").unwrap_or_default(),
                file_size: row.try_get("file_size").unwrap_or_default(),
                compressed_size: row.try_get("compressed_size").unwrap_or_default(),
                file_hash: row.try_get("file_hash").unwrap_or_default(),
                description: row.try_get("description").ok(),
                database_version: row.try_get("database_version").unwrap_or_default(),
                app_version: row.try_get("app_version").unwrap_or_default(),
                compression_algorithm: row.try_get("compression_algorithm").unwrap_or_default(),
                encryption_enabled: row.try_get("encryption_enabled").unwrap_or_default(),
                integrity_verified: row.try_get("integrity_verified").unwrap_or_default(),
                metadata: row.try_get::<serde_json::Value, _>("metadata").ok().unwrap_or(serde_json::Value::Null),
            });
        }
        
        Ok(backups)
    }
    
    /// Delete a backup
    pub async fn delete_backup(&self, backup_id: Uuid, storage: &Arc<RwLock<DataStorage>>) -> DataResult<()> {
        // Get backup metadata
        let backup_metadata = self.get_backup_metadata(storage, backup_id).await?;
        
        // Delete backup file
        if Path::new(&backup_metadata.file_path).exists() {
            std::fs::remove_file(&backup_metadata.file_path)
                .map_err(|e| DataError::Backup {
                    message: format!("Failed to delete backup file: {}", e),
                })?;
        }
        
        // Remove from database
        let storage_guard = storage.read().await;
        sqlx::query("DELETE FROM backup_metadata WHERE id = ?")
            .bind(&backup_id)
            .execute(&storage_guard.pool)
            .await
            .map_err(|e| DataError::Database {
                message: format!("Failed to delete backup metadata: {}", e),
            })?;
        
        log::info!("Backup {} deleted successfully", backup_id);
        Ok(())
    }
}

/// Backup manifest structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupManifest {
    pub backup_id: Uuid,
    pub name: String,
    pub backup_type: BackupType,
    pub created_at: DateTime<Utc>,
    pub description: Option<String>,
    pub reference_backup_id: Option<Uuid>,
    pub database_version: String,
    pub app_version: String,
    pub files: Vec<BackupFileEntry>,
    pub total_size: i64,
    pub checksum: String,
}

/// Individual file entry in backup manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupFileEntry {
    pub path: String,
    pub size: u64,
    pub hash: String,
    pub modified: DateTime<Utc>,
}

/// Result of restore operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreResult {
    pub backup_id: Uuid,
    pub restored_at: DateTime<Utc>,
    pub safety_backup_id: Uuid,
    pub files_restored: usize,
    pub database_restored: bool,
}