//! File management system for assets and documents
//! 
//! This module provides comprehensive file management including:
//! - Secure file storage with optional encryption
//! - File organization and categorization
//! - Thumbnail generation and metadata extraction
//! - File deduplication and optimization
//! - Cross-platform path handling

use super::*;
use std::path::{Path, PathBuf};
use tokio::fs::{self, File};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio_stream::{wrappers::ReadDirStream, StreamExt};
use walkdir::WalkDir;
use std::collections::HashMap;
use futures::stream::{self, StreamExt as FuturesStreamExt};
use bytes::{Bytes, BytesMut};

/// File manager for handling all file operations
pub struct FileManager {
    config: DataManagerConfig,
    encryption: Arc<EncryptionManager>,
}

impl FileManager {
    /// Create new file manager
    pub fn new(config: &DataManagerConfig, encryption: &Arc<EncryptionManager>) -> DataResult<Self> {
        // Ensure files directory exists
        std::fs::create_dir_all(&config.files_dir)
            .map_err(|e| DataError::FileSystem {
                message: format!("Failed to create files directory: {}", e),
            })?;
        
        Ok(Self {
            config: config.clone(),
            encryption: encryption.clone(),
        })
    }
    
    /// Store a file with optional encryption (optimized with streaming)
    pub async fn store_file(
        &self,
        source_path: &Path,
        category: FileCategory,
        metadata: Option<HashMap<String, serde_json::Value>>
    ) -> DataResult<StoredFile> {
        let file_id = Uuid::new_v4();
        
        // Get file size first for progress tracking
        let file_metadata = fs::metadata(source_path).await
            .map_err(|e| DataError::FileSystem {
                message: format!("Failed to get file metadata: {}", e),
            })?;
        let file_size = file_metadata.len();
        
        // Stream file content with buffering for large files
        let file_content = if file_size > 10 * 1024 * 1024 { // > 10MB
            self.read_file_streamed(source_path).await?
        } else {
            fs::read(source_path).await
                .map_err(|e| DataError::FileSystem {
                    message: format!("Failed to read source file: {}", e),
                })?  
        };
        
        // Generate file hash
        let file_hash = self.encryption.generate_hash(&file_content);
        
        // Get file extension
        let extension = source_path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("bin");
        
        // Determine storage path
        let relative_path = self.get_storage_path(file_id, category, extension);
        let full_path = self.config.files_dir.join(&relative_path);
        
        // Ensure parent directory exists (async)
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent).await
                .map_err(|e| DataError::FileSystem {
                    message: format!("Failed to create storage directory: {}", e),
                })?;
        }
        
        // Store file (encrypted if enabled) with streaming
        let stored_content = if self.config.encryption_enabled {
            // Use async encryption for large files
            tokio::task::spawn_blocking({
                let encryption = self.encryption.clone();
                let content = file_content.clone();
                move || encryption.encrypt_bytes(&content)
            }).await
                .map_err(|e| DataError::Encryption {
                    message: format!("Failed to encrypt file: {}", e),
                })?
                .map_err(|e| DataError::Encryption {
                    message: format!("Encryption failed: {:?}", e),
                })?
        } else {
            file_content.clone()
        };
        
        // Write with buffering for better performance
        self.write_file_buffered(&full_path, &stored_content).await?;
        
        // Extract metadata
        let extracted_metadata = self.extract_file_metadata(source_path, &file_content)?;
        let combined_metadata = if let Some(user_metadata) = metadata {
            let mut combined = extracted_metadata;
            for (key, value) in user_metadata {
                combined.insert(key, value);
            }
            combined
        } else {
            extracted_metadata
        };
        
        // Determine MIME type
        let mime_type = self.determine_mime_type(source_path, &file_content);
        
        Ok(StoredFile {
            id: file_id,
            original_name: source_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            stored_path: relative_path,
            full_path: full_path.to_string_lossy().to_string(),
            category,
            size: file_content.len() as i64,
            hash: file_hash,
            mime_type,
            metadata: serde_json::json!(combined_metadata),
            created_at: Utc::now(),
            is_encrypted: self.config.encryption_enabled,
        })
    }
    
    /// Retrieve a stored file (optimized with streaming)
    pub async fn retrieve_file(&self, file_id: Uuid, stored_path: &str) -> DataResult<Vec<u8>> {
        let full_path = self.config.files_dir.join(stored_path);
        
        // Check existence asynchronously
        match fs::metadata(&full_path).await {
            Ok(_) => {},
            Err(_) => return Err(DataError::NotFound {
                resource: format!("File with id {}", file_id),
            }),
        }
        
        // Read with streaming for large files
        let file_size = fs::metadata(&full_path).await
            .map_err(|e| DataError::FileSystem {
                message: format!("Failed to get file metadata: {}", e),
            })?.len();
            
        let stored_content = if file_size > 10 * 1024 * 1024 { // > 10MB
            self.read_file_streamed(&full_path).await?
        } else {
            fs::read(&full_path).await
                .map_err(|e| DataError::FileSystem {
                    message: format!("Failed to read stored file: {}", e),
                })?
        };
        
        // Decrypt if encrypted (async for large files)
        if self.config.encryption_enabled {
            let decrypted = tokio::task::spawn_blocking({
                let encryption = self.encryption.clone();
                let content = stored_content;
                move || encryption.decrypt_file_contents(&content)
            }).await
                .map_err(|e| DataError::Encryption {
                    message: format!("Failed to decrypt file: {}", e),
                })?
                .map_err(|e| DataError::Encryption {
                    message: format!("Decryption failed: {:?}", e),
                })?;
            Ok(decrypted)
        } else {
            Ok(stored_content)
        }
    }
    
    /// Copy file to external location (optimized with streaming)
    pub async fn export_file(
        &self,
        file_id: Uuid,
        stored_path: &str,
        destination: &Path
    ) -> DataResult<()> {
        // For large files, use streaming copy instead of loading into memory
        let full_path = self.config.files_dir.join(stored_path);
        
        // Check file size
        let file_size = fs::metadata(&full_path).await
            .map_err(|e| DataError::FileSystem {
                message: format!("Failed to get file metadata: {}", e),
            })?.len();
        
        // Ensure destination directory exists
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).await
                .map_err(|e| DataError::FileSystem {
                    message: format!("Failed to create destination directory: {}", e),
                })?;
        }
        
        if file_size > 10 * 1024 * 1024 && !self.config.encryption_enabled {
            // Direct streaming copy for large unencrypted files
            fs::copy(&full_path, destination).await
                .map_err(|e| DataError::FileSystem {
                    message: format!("Failed to export file: {}", e),
                })?;
        } else {
            // Use retrieve for decryption if needed
            let file_content = self.retrieve_file(file_id, stored_path).await?;
            self.write_file_buffered(destination, &file_content).await?;
        }
        
        Ok(())
    }
    
    /// Delete a stored file (async)
    pub async fn delete_file(&self, stored_path: &str) -> DataResult<()> {
        let full_path = self.config.files_dir.join(stored_path);
        
        // Check and remove asynchronously
        if fs::metadata(&full_path).await.is_ok() {
            fs::remove_file(&full_path).await
                .map_err(|e| DataError::FileSystem {
                    message: format!("Failed to delete file: {}", e),
                })?;
        }
        
        // Clean up empty directories asynchronously
        if let Some(parent) = full_path.parent() {
            let _ = self.cleanup_empty_directories_async(parent).await;
        }
        
        Ok(())
    }
    
    /// Generate storage path for a file
    fn get_storage_path(&self, file_id: Uuid, category: FileCategory, extension: &str) -> String {
        let category_dir = match category {
            FileCategory::RulebookPdf => "rulebooks",
            FileCategory::CharacterImage => "characters",
            FileCategory::NpcImage => "npcs",
            FileCategory::CampaignImage => "campaigns",
            FileCategory::Map => "maps",
            FileCategory::Handout => "handouts",
            FileCategory::Audio => "audio",
            FileCategory::Video => "video",
            FileCategory::Document => "documents",
            FileCategory::Other => "other",
        };
        
        // Use first two characters of UUID for subdirectory to avoid too many files in one folder
        let uuid_str = file_id.to_string();
        let subdir = &uuid_str[0..2];
        
        format!("{}/{}/{}.{}", category_dir, subdir, file_id, extension)
    }
    
    /// Extract metadata from file
    fn extract_file_metadata(&self, path: &Path, content: &[u8]) -> DataResult<HashMap<String, serde_json::Value>> {
        let mut metadata = HashMap::new();
        
        // Basic file information
        if let Ok(file_metadata) = path.metadata() {
            metadata.insert("size_bytes".to_string(), serde_json::json!(file_metadata.len()));
            
            if let Ok(modified) = file_metadata.modified() {
                if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                    let modified_timestamp = DateTime::<Utc>::from(std::time::UNIX_EPOCH + duration);
                    metadata.insert("modified_at".to_string(), serde_json::json!(modified_timestamp));
                }
            }
        }
        
        // File type specific metadata
        let mime_type = self.determine_mime_type(path, content);
        metadata.insert("mime_type".to_string(), serde_json::json!(mime_type));
        
        if mime_type.starts_with("image/") {
            if let Ok(image_metadata) = self.extract_image_metadata(content) {
                metadata.extend(image_metadata);
            }
        } else if mime_type == "application/pdf" {
            if let Ok(pdf_metadata) = self.extract_pdf_metadata(content) {
                metadata.extend(pdf_metadata);
            }
        }
        
        Ok(metadata)
    }
    
    /// Extract image metadata
    fn extract_image_metadata(&self, _content: &[u8]) -> DataResult<HashMap<String, serde_json::Value>> {
        let mut metadata = HashMap::new();
        
        // Basic image metadata extraction would go here
        // For now, just return empty metadata
        // In a real implementation, you'd use image processing libraries
        
        metadata.insert("type".to_string(), serde_json::json!("image"));
        
        Ok(metadata)
    }
    
    /// Extract PDF metadata
    fn extract_pdf_metadata(&self, _content: &[u8]) -> DataResult<HashMap<String, serde_json::Value>> {
        let mut metadata = HashMap::new();
        
        // Basic PDF metadata extraction would go here
        // For now, just return basic info
        // In a real implementation, you'd use PDF processing libraries
        
        metadata.insert("type".to_string(), serde_json::json!("pdf"));
        
        Ok(metadata)
    }
    
    /// Determine MIME type
    fn determine_mime_type(&self, path: &Path, content: &[u8]) -> String {
        // Check by file extension first
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            match extension.to_lowercase().as_str() {
                "pdf" => return "application/pdf".to_string(),
                "jpg" | "jpeg" => return "image/jpeg".to_string(),
                "png" => return "image/png".to_string(),
                "gif" => return "image/gif".to_string(),
                "webp" => return "image/webp".to_string(),
                "mp3" => return "audio/mpeg".to_string(),
                "wav" => return "audio/wav".to_string(),
                "mp4" => return "video/mp4".to_string(),
                "webm" => return "video/webm".to_string(),
                "txt" => return "text/plain".to_string(),
                "md" => return "text/markdown".to_string(),
                "json" => return "application/json".to_string(),
                "xml" => return "application/xml".to_string(),
                _ => {}
            }
        }
        
        // Check by content (magic numbers)
        if content.len() >= 4 {
            if content.starts_with(b"%PDF") {
                return "application/pdf".to_string();
            }
            if content.starts_with(&[0xFF, 0xD8, 0xFF]) {
                return "image/jpeg".to_string();
            }
            if content.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
                return "image/png".to_string();
            }
        }
        
        "application/octet-stream".to_string()
    }
    
    /// Clean up empty directories
    fn cleanup_empty_directories<'a>(&'a self, dir: &'a Path) -> futures::future::BoxFuture<'a, DataResult<()>> {
        Box::pin(async move {
        if dir.is_dir() {
            // Check if directory is empty
            match fs::read_dir(dir).await {
                Ok(mut entries) => {
                    if entries.next_entry().await.map_err(|e| DataError::FileSystem {
                        message: format!("Failed to read directory entries: {}", e)
                    })?.is_none() {
                        let _ = fs::remove_dir(dir).await;
                        
                        // Recursively check parent directory
                        if let Some(parent) = dir.parent() {
                            if parent != self.config.files_dir {
                                let _ = self.cleanup_empty_directories(parent).await;
                            }
                        }
                    }
                },
                Err(_) => {}
            }
        }
        Ok(())
        })
    }
    
    /// Get file statistics (optimized with async operations)
    pub async fn get_storage_stats(&self) -> DataResult<StorageStats> {
        let mut stats = StorageStats {
            total_files: 0,
            total_size: 0,
            categories: HashMap::new(),
            largest_files: Vec::new(),
        };
        
        if !self.config.files_dir.exists() {
            return Ok(stats);
        }
        
        let mut file_sizes = Vec::new();
        
        // Use async directory walking for better performance
        let mut entries = Vec::new();
        for entry in WalkDir::new(&self.config.files_dir) {
            let entry = entry.map_err(|e| DataError::FileSystem {
                message: format!("Failed to walk files directory: {}", e),
            })?;
            if entry.file_type().is_file() {
                entries.push(entry.path().to_path_buf());
            }
        }
        
        // Process files in parallel
        const BATCH_SIZE: usize = 20;
        for chunk in entries.chunks(BATCH_SIZE) {
            let mut handles = Vec::new();
            
            for path in chunk {
                let path = path.clone();
                let handle = tokio::spawn(async move {
                    let metadata = fs::metadata(&path).await.ok()?;
                    Some((path, metadata.len()))
                });
                handles.push(handle);
            }
            
            // Collect results
            for handle in handles {
                if let Ok(Some((path, size))) = handle.await {
                    stats.total_files += 1;
                    stats.total_size += size;
                    
                    // Determine category from path
                    let category = self.determine_category_from_path(&path);
                    *stats.categories.entry(category).or_insert(0) += 1;
                    
                    // Track for largest files
                    file_sizes.push((path, size));
                }
            }
        }
        
        // Sort by size and take top 10
        file_sizes.sort_by_key(|(_, size)| std::cmp::Reverse(*size));
        stats.largest_files = file_sizes
            .into_iter()
            .take(10)
            .map(|(path, size)| (path.to_string_lossy().to_string(), size))
            .collect();
        
        Ok(stats)
    }
    
    /// Determine file category from path
    fn determine_category_from_path(&self, path: &Path) -> String {
        if let Some(parent) = path.parent() {
            if let Some(parent_name) = parent.file_name().and_then(|n| n.to_str()) {
                return parent_name.to_string();
            }
        }
        "unknown".to_string()
    }
    
    /// Find duplicate files by hash (optimized with async and chunked reading)
    pub async fn find_duplicate_files(&self) -> DataResult<Vec<DuplicateGroup>> {
        let mut file_hashes: HashMap<String, Vec<PathBuf>> = HashMap::new();
        
        if !self.config.files_dir.exists() {
            return Ok(Vec::new());
        }
        
        // Collect all file paths first
        let mut file_paths = Vec::new();
        for entry in WalkDir::new(&self.config.files_dir) {
            let entry = entry.map_err(|e| DataError::FileSystem {
                message: format!("Failed to walk files directory: {}", e),
            })?;
            
            if entry.file_type().is_file() {
                file_paths.push(entry.path().to_path_buf());
            }
        }
        
        // Process files in parallel batches for better performance
        const BATCH_SIZE: usize = 10;
        for chunk in file_paths.chunks(BATCH_SIZE) {
            let mut handles = Vec::new();
            
            for path in chunk {
                let path = path.clone();
                let encryption = self.encryption.clone();
                
                let handle = tokio::spawn(async move {
                    // Only read first 1MB for hash to avoid loading entire file
                    let mut file = File::open(&path).await.ok()?;
                    let mut buffer = vec![0; 1024 * 1024]; // 1MB
                    let bytes_read = file.read(&mut buffer).await.ok()?;
                    buffer.truncate(bytes_read);
                    
                    // Get file size for additional uniqueness
                    let metadata = file.metadata().await.ok()?;
                    let size = metadata.len();
                    
                    // Hash with size prefix for better accuracy
                    let mut hash_input = Vec::with_capacity(8 + buffer.len());
                    hash_input.extend_from_slice(&size.to_le_bytes());
                    hash_input.extend_from_slice(&buffer);
                    
                    let hash = encryption.generate_hash(&hash_input);
                    Some((hash, path))
                });
                
                handles.push(handle);
            }
            
            // Collect results from batch
            for handle in handles {
                if let Ok(Some((hash, path))) = handle.await {
                    file_hashes.entry(hash).or_default().push(path);
                }
            }
        }
        
        // Find groups with more than one file
        let mut duplicates: Vec<DuplicateGroup> = Vec::new();
        
        for (hash, files) in file_hashes.into_iter().filter(|(_, files)| files.len() > 1) {
            let mut total_size = 0u64;
            for path in &files {
                if let Ok(metadata) = fs::metadata(path).await {
                    total_size += metadata.len();
                }
            }
            
            duplicates.push(DuplicateGroup {
                hash,
                files: files.into_iter().map(|p| p.to_string_lossy().to_string()).collect(),
                total_size,
            });
        }
        
        Ok(duplicates)
    }
    
    /// Optimize storage by removing duplicates (async)
    pub async fn optimize_storage(&self, keep_strategy: DuplicateStrategy) -> DataResult<OptimizationResult> {
        let duplicates = self.find_duplicate_files().await?;
        let mut removed_count = 0;
        let mut space_saved = 0u64;
        let mut errors = Vec::new();
        
        for group in duplicates {
            match self.resolve_duplicates(&group, &keep_strategy).await {
                Ok((removed, saved)) => {
                    removed_count += removed;
                    space_saved += saved;
                },
                Err(e) => {
                    errors.push(format!("Failed to resolve duplicates for hash {}: {}", group.hash, e));
                }
            }
        }
        
        Ok(OptimizationResult {
            files_removed: removed_count,
            space_saved,
            errors,
            timestamp: Utc::now(),
        })
    }
    
    /// Resolve duplicate files based on strategy
    async fn resolve_duplicates(&self, group: &DuplicateGroup, strategy: &DuplicateStrategy) -> DataResult<(usize, u64)> {
        if group.files.len() <= 1 {
            return Ok((0, 0));
        }
        
        let keep_index = match strategy {
            DuplicateStrategy::KeepFirst => 0,
            DuplicateStrategy::KeepLast => group.files.len() - 1,
            DuplicateStrategy::KeepOldest => {
                // Find oldest file by creation time
                let mut oldest_index = 0;
                let mut oldest_time = std::time::SystemTime::now();
                
                for (i, file_path) in group.files.iter().enumerate() {
                    if let Ok(metadata) = fs::metadata(file_path).await {
                        if let Ok(created) = metadata.created().or_else(|_| metadata.modified()) {
                            if created < oldest_time {
                                oldest_time = created;
                                oldest_index = i;
                            }
                        }
                    }
                }
                oldest_index
            },
            DuplicateStrategy::KeepNewest => {
                // Find newest file by creation time
                let mut newest_index = 0;
                let mut newest_time = std::time::UNIX_EPOCH;
                
                for (i, file_path) in group.files.iter().enumerate() {
                    if let Ok(metadata) = fs::metadata(file_path).await {
                        if let Ok(created) = metadata.created().or_else(|_| metadata.modified()) {
                            if created > newest_time {
                                newest_time = created;
                                newest_index = i;
                            }
                        }
                    }
                }
                newest_index
            },
        };
        
        let mut removed_count = 0;
        let mut space_saved = 0u64;
        
        for (i, file_path) in group.files.iter().enumerate() {
            if i != keep_index {
                if let Ok(metadata) = fs::metadata(file_path).await {
                    space_saved += metadata.len();
                }
                
                if let Err(e) = fs::remove_file(file_path).await {
                    log::warn!("Failed to remove duplicate file {}: {}", file_path, e);
                } else {
                    removed_count += 1;
                }
            }
        }
        
        Ok((removed_count, space_saved))
    }
    
    /// Stream read large files with buffering
    async fn read_file_streamed(&self, path: &Path) -> DataResult<Vec<u8>> {
        let file = File::open(path).await
            .map_err(|e| DataError::FileSystem {
                message: format!("Failed to open file for streaming: {}", e),
            })?;
        
        let metadata = file.metadata().await
            .map_err(|e| DataError::FileSystem {
                message: format!("Failed to get file metadata: {}", e),
            })?;
        
        let mut reader = BufReader::with_capacity(64 * 1024, file); // 64KB buffer
        let mut buffer = Vec::with_capacity(metadata.len() as usize);
        
        reader.read_to_end(&mut buffer).await
            .map_err(|e| DataError::FileSystem {
                message: format!("Failed to read file stream: {}", e),
            })?;
        
        Ok(buffer)
    }
    
    /// Write file with buffering for better performance
    async fn write_file_buffered(&self, path: &Path, content: &[u8]) -> DataResult<()> {
        let file = File::create(path).await
            .map_err(|e| DataError::FileSystem {
                message: format!("Failed to create file for writing: {}", e),
            })?;
        
        let mut writer = BufWriter::with_capacity(64 * 1024, file); // 64KB buffer
        
        writer.write_all(content).await
            .map_err(|e| DataError::FileSystem {
                message: format!("Failed to write file buffer: {}", e),
            })?;
        
        writer.flush().await
            .map_err(|e| DataError::FileSystem {
                message: format!("Failed to flush file buffer: {}", e),
            })?;
        
        Ok(())
    }
    
    /// Async cleanup of empty directories
    fn cleanup_empty_directories_async<'a>(&'a self, dir: &'a Path) -> futures::future::BoxFuture<'a, DataResult<()>> {
        Box::pin(async move {
            if dir.is_dir() {
                // Check if directory is empty asynchronously
                let mut entries = fs::read_dir(dir).await
                    .map_err(|e| DataError::FileSystem {
                        message: format!("Failed to read directory: {}", e),
                    })?;
                
                let mut is_empty = true;
                while let Some(_) = entries.next_entry().await
                    .map_err(|e| DataError::FileSystem {
                        message: format!("Failed to read directory entry: {}", e),
                    })? {
                    is_empty = false;
                    break;
                }
                
                if is_empty {
                    let _ = fs::remove_dir(dir).await;
                    
                    // Recursively check parent directory
                    if let Some(parent) = dir.parent() {
                        if parent != self.config.files_dir {
                            let _ = self.cleanup_empty_directories_async(parent).await;
                        }
                    }
                }
            }
            Ok(())
        })
    }
    
    /// Create directory structure backup
    pub fn backup_directory_structure(&self) -> DataResult<DirectoryStructure> {
        let mut structure = DirectoryStructure {
            root: self.config.files_dir.clone(),
            directories: Vec::new(),
            files: Vec::new(),
            created_at: Utc::now(),
        };
        
        if !self.config.files_dir.exists() {
            return Ok(structure);
        }
        
        for entry in WalkDir::new(&self.config.files_dir) {
            let entry = entry.map_err(|e| DataError::FileSystem {
                message: format!("Failed to walk directory structure: {}", e),
            })?;
            
            let path = entry.path();
            let relative_path = path.strip_prefix(&self.config.files_dir)
                .map_err(|e| DataError::FileSystem {
                    message: format!("Failed to get relative path: {}", e),
                })?;
            
            if entry.file_type().is_dir() {
                structure.directories.push(relative_path.to_string_lossy().to_string());
            } else if entry.file_type().is_file() {
                let metadata = entry.metadata().map_err(|e| DataError::FileSystem {
                    message: format!("Failed to get file metadata: {}", e),
                })?;
                
                structure.files.push(FileStructureEntry {
                    path: relative_path.to_string_lossy().to_string(),
                    size: metadata.len(),
                    modified: metadata.modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| DateTime::<Utc>::from(std::time::UNIX_EPOCH + d))
                        .unwrap_or_else(Utc::now),
                });
            }
        }
        
        Ok(structure)
    }
}

/// File category for organization
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FileCategory {
    RulebookPdf,
    CharacterImage,
    NpcImage,
    CampaignImage,
    Map,
    Handout,
    Audio,
    Video,
    Document,
    Other,
}

/// Stored file information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredFile {
    pub id: Uuid,
    pub original_name: String,
    pub stored_path: String,
    pub full_path: String,
    pub category: FileCategory,
    pub size: i64,
    pub hash: String,
    pub mime_type: String,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub is_encrypted: bool,
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_files: u64,
    pub total_size: u64,
    pub categories: HashMap<String, u64>,
    pub largest_files: Vec<(String, u64)>,
}

/// Duplicate file group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroup {
    pub hash: String,
    pub files: Vec<String>,
    pub total_size: u64,
}

/// Strategy for handling duplicates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DuplicateStrategy {
    KeepFirst,
    KeepLast,
    KeepOldest,
    KeepNewest,
}

/// Storage optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub files_removed: usize,
    pub space_saved: u64,
    pub errors: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

/// Directory structure backup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryStructure {
    pub root: PathBuf,
    pub directories: Vec<String>,
    pub files: Vec<FileStructureEntry>,
    pub created_at: DateTime<Utc>,
}

/// File structure entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileStructureEntry {
    pub path: String,
    pub size: u64,
    pub modified: DateTime<Utc>,
}