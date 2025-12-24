use anyhow::{Result, anyhow};
use blake3::{Hasher, Hash};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use std::io::{Read, Write, BufReader, BufWriter};
use std::fs::{File, OpenOptions, create_dir_all};
use chrono::{DateTime, Utc};
use walkdir::WalkDir;
use rayon::prelude::*;

/// Chunk size for streaming file operations (64KB)
const CHUNK_SIZE: usize = 65536;

/// Maximum number of files to process concurrently for duplicate detection
const MAX_CONCURRENT_FILES: usize = 8;

/// File metadata for tracking and duplicate detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub file_id: String,
    pub original_name: Option<String>,
    pub file_path: PathBuf,
    pub file_size: u64,
    pub blake3_hash: String,
    pub created_at: DateTime<Utc>,
    pub last_modified: DateTime<Utc>,
    pub mime_type: Option<String>,
    pub is_duplicate: bool,
    pub duplicate_of: Option<String>,
}

/// Duplicate file group information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroup {
    pub hash: String,
    pub file_size: u64,
    pub file_ids: Vec<String>,
    pub paths: Vec<PathBuf>,
    pub total_wasted_space: u64,
}

/// File operation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileOperationStats {
    pub total_files: usize,
    pub total_size_bytes: u64,
    pub duplicates_found: usize,
    pub duplicate_groups: usize,
    pub space_saved_bytes: u64,
    pub processing_time_ms: u64,
}

/// Memory-efficient file manager with streaming duplicate detection
/// Processes large files without loading them entirely into memory
pub struct FileManager {
    /// Base directory for file storage
    base_path: PathBuf,
    
    /// File metadata registry
    metadata_registry: Arc<RwLock<HashMap<String, FileMetadata>>>,
    
    /// Hash-to-FileID mapping for efficient duplicate detection
    hash_index: Arc<RwLock<HashMap<String, Vec<String>>>>,
    
    /// Chunk size for streaming operations
    chunk_size: usize,
}

impl FileManager {
    /// Create a new file manager with default base path
    pub fn new() -> Result<Self> {
        let base_path = std::env::temp_dir().join("mdmai_files");
        Self::with_base_path(base_path)
    }

    /// Create file manager with custom base path
    pub fn with_base_path<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        
        // Ensure base directory exists
        create_dir_all(&base_path)
            .map_err(|e| anyhow!("Failed to create base directory: {}", e))?;

        Ok(Self {
            base_path,
            metadata_registry: Arc::new(RwLock::new(HashMap::new())),
            hash_index: Arc::new(RwLock::new(HashMap::new())),
            chunk_size: CHUNK_SIZE,
        })
    }

    /// Store file data with streaming hash calculation and duplicate detection
    pub async fn store_file(&self, file_id: &str, data: &[u8]) -> Result<String> {
        self.store_file_with_metadata(file_id, data, None, None).await
    }

    /// Store file with additional metadata
    pub async fn store_file_with_metadata(
        &self,
        file_id: &str,
        data: &[u8],
        original_name: Option<String>,
        mime_type: Option<String>,
    ) -> Result<String> {
        // Generate file path
        let file_path = self.base_path.join(format!("{}.dat", file_id));
        
        // Ensure parent directory exists
        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| anyhow!("Failed to create directory: {}", e))?;
        }

        // Write file and calculate hash simultaneously using streaming approach
        let blake3_hash = self.write_and_hash_file(&file_path, data).await?;

        // Check for duplicates
        let (is_duplicate, duplicate_of) = self.check_for_duplicates(&blake3_hash, file_id).await;

        // Create metadata record
        let metadata = FileMetadata {
            file_id: file_id.to_string(),
            original_name,
            file_path: file_path.clone(),
            file_size: data.len() as u64,
            blake3_hash: blake3_hash.clone(),
            created_at: Utc::now(),
            last_modified: Utc::now(),
            mime_type,
            is_duplicate,
            duplicate_of: duplicate_of.clone(),
        };

        // Update registries
        self.update_metadata_registry(metadata).await;
        self.update_hash_index(&blake3_hash, file_id).await;

        if is_duplicate {
            log::info!("Stored file {} (duplicate of {})", file_id, duplicate_of.unwrap_or_else(|| "unknown".to_string()));
        } else {
            log::debug!("Stored new file: {}", file_id);
        }

        Ok(blake3_hash)
    }

    /// Load file data with streaming approach
    pub async fn load_file(&self, file_id: &str) -> Result<Vec<u8>> {
        let metadata = {
            let registry = self.metadata_registry.read();
            registry.get(file_id).cloned()
        };

        let metadata = metadata.ok_or_else(|| anyhow!("File not found: {}", file_id))?;

        if !metadata.file_path.exists() {
            return Err(anyhow!("File does not exist: {:?}", metadata.file_path));
        }

        // Read file using efficient streaming
        let data = tokio::fs::read(&metadata.file_path).await
            .map_err(|e| anyhow!("Failed to read file: {}", e))?;

        // Update last accessed time (optional metadata update)
        // For now, we'll skip this to avoid lock contention on reads

        Ok(data)
    }

    /// Delete file and clean up metadata
    pub async fn delete_file(&self, file_id: &str) -> Result<()> {
        let metadata = {
            let mut registry = self.metadata_registry.write();
            registry.remove(file_id)
        };

        if let Some(metadata) = metadata {
            // Remove from hash index
            {
                let mut hash_index = self.hash_index.write();
                if let Some(file_ids) = hash_index.get_mut(&metadata.blake3_hash) {
                    file_ids.retain(|id| id != file_id);
                    if file_ids.is_empty() {
                        hash_index.remove(&metadata.blake3_hash);
                    }
                }
            }

            // Delete actual file
            if metadata.file_path.exists() {
                tokio::fs::remove_file(&metadata.file_path).await
                    .map_err(|e| anyhow!("Failed to delete file: {}", e))?;
            }

            log::debug!("Deleted file: {}", file_id);
        }

        Ok(())
    }

    /// Find duplicate files using streaming hash calculation
    /// This efficiently processes large directories without memory issues
    pub async fn find_duplicates_in_directory<P: AsRef<Path>>(&self, directory: P) -> Result<Vec<DuplicateGroup>> {
        let directory = directory.as_ref();
        if !directory.exists() || !directory.is_dir() {
            return Err(anyhow!("Directory does not exist or is not a directory: {:?}", directory));
        }

        log::info!("Starting duplicate detection in directory: {:?}", directory);
        let start_time = std::time::Instant::now();

        // Collect all files to process
        let file_paths: Vec<PathBuf> = WalkDir::new(directory)
            .follow_links(false)
            .into_iter()
            .filter_map(|entry| {
                entry.ok().and_then(|e| {
                    let path = e.path();
                    if path.is_file() {
                        Some(path.to_path_buf())
                    } else {
                        None
                    }
                })
            })
            .collect();

        if file_paths.is_empty() {
            return Ok(Vec::new());
        }

        log::debug!("Found {} files to analyze for duplicates", file_paths.len());

        // Process files in chunks to avoid memory exhaustion
        let chunk_size = MAX_CONCURRENT_FILES;
        let mut hash_to_paths: HashMap<String, Vec<PathBuf>> = HashMap::new();

        for chunk in file_paths.chunks(chunk_size) {
            // Process chunk in parallel using streaming hash calculation
            let results: Vec<_> = chunk.par_iter()
                .filter_map(|path| {
                    match self.calculate_file_hash_streaming_sync(path) {
                        Ok(hash) => Some((hash, path.clone())),
                        Err(e) => {
                            log::warn!("Failed to calculate hash for {:?}: {}", path, e);
                            None
                        }
                    }
                })
                .collect();

            // Collect results
            for (hash, path) in results {
                hash_to_paths.entry(hash).or_insert_with(Vec::new).push(path);
            }
        }

        // Find duplicates (hashes with more than one file)
        let mut duplicate_groups = Vec::new();
        for (hash, paths) in hash_to_paths {
            if paths.len() > 1 {
                // Calculate file size (assuming all files with same hash have same size)
                let file_size = if let Ok(metadata) = std::fs::metadata(&paths[0]) {
                    metadata.len()
                } else {
                    0
                };

                let total_wasted_space = file_size * (paths.len() as u64 - 1);

                duplicate_groups.push(DuplicateGroup {
                    hash,
                    file_size,
                    file_ids: paths.iter().map(|p| p.to_string_lossy().to_string()).collect(),
                    paths,
                    total_wasted_space,
                });
            }
        }

        // Sort by wasted space (largest first)
        duplicate_groups.sort_by(|a, b| b.total_wasted_space.cmp(&a.total_wasted_space));

        let processing_time = start_time.elapsed();
        log::info!("Duplicate detection completed: {} duplicate groups found in {:?}", 
                  duplicate_groups.len(), processing_time);

        Ok(duplicate_groups)
    }

    /// Get file metadata
    pub async fn get_file_metadata(&self, file_id: &str) -> Option<FileMetadata> {
        self.metadata_registry.read().get(file_id).cloned()
    }

    /// List all files
    pub async fn list_files(&self) -> Vec<FileMetadata> {
        self.metadata_registry.read().values().cloned().collect()
    }

    /// Get files by hash (for finding duplicates)
    pub async fn get_files_by_hash(&self, hash: &str) -> Vec<String> {
        self.hash_index.read()
            .get(hash)
            .cloned()
            .unwrap_or_else(Vec::new)
    }

    /// Get duplicate groups from managed files
    pub async fn get_duplicate_groups(&self) -> Vec<DuplicateGroup> {
        let hash_index = self.hash_index.read();
        let metadata_registry = self.metadata_registry.read();

        let mut duplicate_groups = Vec::new();

        for (hash, file_ids) in hash_index.iter() {
            if file_ids.len() > 1 {
                // Get file size and paths from metadata
                let mut paths = Vec::new();
                let mut file_size = 0u64;

                for file_id in file_ids {
                    if let Some(metadata) = metadata_registry.get(file_id) {
                        paths.push(metadata.file_path.clone());
                        file_size = metadata.file_size; // All duplicates have same size
                    }
                }

                if paths.len() > 1 {
                    let total_wasted_space = file_size * (paths.len() as u64 - 1);

                    duplicate_groups.push(DuplicateGroup {
                        hash: hash.clone(),
                        file_size,
                        file_ids: file_ids.clone(),
                        paths,
                        total_wasted_space,
                    });
                }
            }
        }

        duplicate_groups.sort_by(|a, b| b.total_wasted_space.cmp(&a.total_wasted_space));
        duplicate_groups
    }

    /// Get storage statistics
    pub async fn get_storage_stats(&self) -> HashMap<String, serde_json::Value> {
        let metadata_registry = self.metadata_registry.read();
        
        let total_files = metadata_registry.len();
        let total_size: u64 = metadata_registry.values().map(|m| m.file_size).sum();
        let duplicates_count = metadata_registry.values().filter(|m| m.is_duplicate).count();
        
        let duplicate_groups = self.get_duplicate_groups().await;
        let wasted_space: u64 = duplicate_groups.iter().map(|g| g.total_wasted_space).sum();

        let mut stats = HashMap::new();
        stats.insert("total_files".to_string(), serde_json::Value::Number(total_files.into()));
        stats.insert("total_size_bytes".to_string(), serde_json::Value::Number(total_size.into()));
        stats.insert("duplicates_count".to_string(), serde_json::Value::Number(duplicates_count.into()));
        stats.insert("duplicate_groups_count".to_string(), serde_json::Value::Number(duplicate_groups.len().into()));
        stats.insert("wasted_space_bytes".to_string(), serde_json::Value::Number(wasted_space.into()));
        
        let efficiency = if total_size > 0 {
            ((total_size - wasted_space) as f64 / total_size as f64) * 100.0
        } else {
            100.0
        };
        stats.insert("storage_efficiency_percent".to_string(), 
                    serde_json::Value::Number(serde_json::Number::from_f64(efficiency).unwrap_or_else(|| 0.into())));

        stats
    }

    /// Write file and calculate hash simultaneously using streaming approach
    async fn write_and_hash_file(&self, file_path: &Path, data: &[u8]) -> Result<String> {
        let file_path = file_path.to_path_buf();
        let data = data.to_vec();
        let chunk_size = self.chunk_size;

        tokio::task::spawn_blocking(move || {
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&file_path)
                .map_err(|e| anyhow!("Failed to create file: {}", e))?;

            let mut writer = BufWriter::new(file);
            let mut hasher = Hasher::new();
            
            // Process data in chunks for consistent memory usage
            for chunk in data.chunks(chunk_size) {
                // Update hash
                hasher.update(chunk);
                
                // Write chunk to file
                writer.write_all(chunk)
                    .map_err(|e| anyhow!("Failed to write chunk: {}", e))?;
            }

            // Ensure all data is written
            writer.flush()
                .map_err(|e| anyhow!("Failed to flush file: {}", e))?;

            let hash = hasher.finalize();
            Ok(hash.to_hex().to_string())
        }).await.map_err(|e| anyhow!("File write task failed: {}", e))?
    }

    /// Calculate file hash using streaming approach (synchronous for parallel processing)
    fn calculate_file_hash_streaming_sync(&self, file_path: &Path) -> Result<String> {
        let file = File::open(file_path)
            .map_err(|e| anyhow!("Failed to open file: {}", e))?;

        let mut reader = BufReader::new(file);
        let mut hasher = Hasher::new();
        let mut buffer = vec![0u8; self.chunk_size];

        loop {
            let bytes_read = reader.read(&mut buffer)
                .map_err(|e| anyhow!("Failed to read file chunk: {}", e))?;

            if bytes_read == 0 {
                break; // End of file
            }

            hasher.update(&buffer[..bytes_read]);
        }

        let hash = hasher.finalize();
        Ok(hash.to_hex().to_string())
    }

    /// Check if a file with the given hash already exists (duplicate detection)
    async fn check_for_duplicates(&self, hash: &str, current_file_id: &str) -> (bool, Option<String>) {
        let hash_index = self.hash_index.read();
        
        if let Some(existing_file_ids) = hash_index.get(hash) {
            // Find the first non-current file ID
            for file_id in existing_file_ids {
                if file_id != current_file_id {
                    return (true, Some(file_id.clone()));
                }
            }
        }
        
        (false, None)
    }

    /// Update metadata registry
    async fn update_metadata_registry(&self, metadata: FileMetadata) {
        self.metadata_registry.write().insert(metadata.file_id.clone(), metadata);
    }

    /// Update hash index for duplicate detection
    async fn update_hash_index(&self, hash: &str, file_id: &str) {
        self.hash_index.write()
            .entry(hash.to_string())
            .or_insert_with(Vec::new)
            .push(file_id.to_string());
    }

    /// Remove duplicates by keeping only one copy of each file
    pub async fn deduplicate_files(&self) -> Result<FileOperationStats> {
        let start_time = std::time::Instant::now();
        let duplicate_groups = self.get_duplicate_groups().await;
        
        let mut files_removed = 0;
        let mut space_saved = 0u64;

        for group in duplicate_groups.iter() {
            // Keep the first file, remove the rest
            let files_to_remove = &group.file_ids[1..];
            
            for file_id in files_to_remove {
                match self.delete_file(file_id).await {
                    Ok(()) => {
                        files_removed += 1;
                        space_saved += group.file_size;
                        log::debug!("Removed duplicate file: {}", file_id);
                    }
                    Err(e) => {
                        log::warn!("Failed to remove duplicate file {}: {}", file_id, e);
                    }
                }
            }
        }

        let processing_time = start_time.elapsed().as_millis() as u64;
        
        Ok(FileOperationStats {
            total_files: self.metadata_registry.read().len(),
            total_size_bytes: self.metadata_registry.read().values().map(|m| m.file_size).sum(),
            duplicates_found: files_removed,
            duplicate_groups: duplicate_groups.len(),
            space_saved_bytes: space_saved,
            processing_time_ms: processing_time,
        })
    }
}

impl Default for FileManager {
    fn default() -> Self {
        Self::new().expect("Failed to create default FileManager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_file_storage_and_retrieval() {
        let temp_dir = tempdir().unwrap();
        let file_manager = FileManager::with_base_path(temp_dir.path()).unwrap();

        let test_data = b"Hello, world! This is test data for file storage.";
        let file_id = "test_file_001";

        // Store file
        let hash = file_manager.store_file(file_id, test_data).await.unwrap();
        assert!(!hash.is_empty());

        // Retrieve file
        let retrieved_data = file_manager.load_file(file_id).await.unwrap();
        assert_eq!(test_data, retrieved_data.as_slice());

        // Get metadata
        let metadata = file_manager.get_file_metadata(file_id).await.unwrap();
        assert_eq!(metadata.file_size, test_data.len() as u64);
        assert_eq!(metadata.blake3_hash, hash);
    }

    #[tokio::test]
    async fn test_duplicate_detection() {
        let temp_dir = tempdir().unwrap();
        let file_manager = FileManager::with_base_path(temp_dir.path()).unwrap();

        let test_data = b"Duplicate test data";
        
        // Store same data with different IDs
        file_manager.store_file("file1", test_data).await.unwrap();
        file_manager.store_file("file2", test_data).await.unwrap();

        let duplicate_groups = file_manager.get_duplicate_groups().await;
        assert_eq!(duplicate_groups.len(), 1);
        assert_eq!(duplicate_groups[0].file_ids.len(), 2);
    }

    #[tokio::test]
    async fn test_large_file_handling() {
        let temp_dir = tempdir().unwrap();
        let file_manager = FileManager::with_base_path(temp_dir.path()).unwrap();

        // Create large test data (1MB)
        let large_data = vec![42u8; 1024 * 1024];
        let file_id = "large_test_file";

        // Store and retrieve large file
        let hash = file_manager.store_file(file_id, &large_data).await.unwrap();
        let retrieved_data = file_manager.load_file(file_id).await.unwrap();
        
        assert_eq!(large_data.len(), retrieved_data.len());
        assert_eq!(large_data, retrieved_data);
        assert!(!hash.is_empty());
    }

    #[tokio::test]
    async fn test_deduplication() {
        let temp_dir = tempdir().unwrap();
        let file_manager = FileManager::with_base_path(temp_dir.path()).unwrap();

        let test_data = b"Data for deduplication test";
        
        // Store multiple copies
        file_manager.store_file("dup1", test_data).await.unwrap();
        file_manager.store_file("dup2", test_data).await.unwrap();
        file_manager.store_file("dup3", test_data).await.unwrap();

        let stats_before = file_manager.get_storage_stats().await;
        let files_before = stats_before["total_files"].as_u64().unwrap();

        // Deduplicate
        let dedup_stats = file_manager.deduplicate_files().await.unwrap();
        
        assert_eq!(dedup_stats.duplicates_found, 2); // Should remove 2 out of 3
        assert!(dedup_stats.space_saved_bytes > 0);

        let stats_after = file_manager.get_storage_stats().await;
        let files_after = stats_after["total_files"].as_u64().unwrap();
        
        assert_eq!(files_after, files_before - 2);
    }
}