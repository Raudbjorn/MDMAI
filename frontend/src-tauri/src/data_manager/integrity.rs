use anyhow::{Result, anyhow};
use blake3::{Hasher, Hash};
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use std::io::{Read, BufReader, Seek, SeekFrom};
use std::fs::{File, OpenOptions};
use chrono::{DateTime, Utc};
use rayon::prelude::*;

/// Chunk size for streaming file processing (64KB)
const CHUNK_SIZE: usize = 65536;

/// Maximum number of concurrent integrity checks
const MAX_CONCURRENT_CHECKS: usize = 4;

/// File integrity record with comprehensive metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileIntegrityRecord {
    pub file_id: String,
    pub file_path: Option<PathBuf>,
    pub blake3_hash: String,
    pub sha256_hash: String,
    pub file_size: u64,
    pub chunk_count: usize,
    pub created_at: DateTime<Utc>,
    pub last_verified: DateTime<Utc>,
    pub verification_count: u64,
    pub is_corrupted: bool,
}

/// Verification result for integrity checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub total_files: usize,
    pub verified_files: usize,
    pub corrupted_files: usize,
    pub failed_files: usize,
    pub total_size_bytes: u64,
    pub verification_time_ms: u64,
    pub corrupted_file_ids: Vec<String>,
    pub failed_file_ids: Vec<String>,
}

/// Streaming integrity checker that processes large files efficiently
/// without loading entire files into memory
pub struct IntegrityChecker {
    /// In-memory storage for file integrity records
    records: Arc<RwLock<HashMap<String, FileIntegrityRecord>>>,
    
    /// Base directory for file storage
    base_path: PathBuf,
    
    /// Chunk size for streaming operations
    chunk_size: usize,
}

impl IntegrityChecker {
    /// Create a new integrity checker
    pub fn new() -> Self {
        let base_path = std::env::temp_dir().join("mdmai_data");
        
        Self {
            records: Arc::new(RwLock::new(HashMap::new())),
            base_path,
            chunk_size: CHUNK_SIZE,
        }
    }

    /// Create integrity checker with custom base path
    pub fn with_base_path<P: AsRef<Path>>(base_path: P) -> Self {
        Self {
            records: Arc::new(RwLock::new(HashMap::new())),
            base_path: base_path.as_ref().to_path_buf(),
            chunk_size: CHUNK_SIZE,
        }
    }

    /// Add a new file record using streaming hash calculation
    /// This prevents memory issues with large files
    pub async fn add_file_record(&self, file_id: &str, data: &[u8]) -> Result<()> {
        let file_path = self.base_path.join(format!("{}.dat", file_id));
        
        // Ensure parent directory exists
        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| anyhow!("Failed to create directory: {}", e))?;
        }

        // Write data to file
        tokio::fs::write(&file_path, data).await
            .map_err(|e| anyhow!("Failed to write file: {}", e))?;

        // Calculate hashes using streaming approach
        let (blake3_hash, sha256_hash, chunk_count) = self.calculate_streaming_hashes(&file_path).await?;

        // Create integrity record
        let record = FileIntegrityRecord {
            file_id: file_id.to_string(),
            file_path: Some(file_path),
            blake3_hash,
            sha256_hash,
            file_size: data.len() as u64,
            chunk_count,
            created_at: Utc::now(),
            last_verified: Utc::now(),
            verification_count: 0,
            is_corrupted: false,
        };

        // Store record
        self.records.write().insert(file_id.to_string(), record);
        
        log::debug!("Added integrity record for file: {}", file_id);
        Ok(())
    }

    /// Verify file integrity using streaming hash calculation
    pub async fn verify_file(&self, file_id: &str, expected_data: &[u8]) -> Result<bool> {
        let record = {
            let records = self.records.read();
            records.get(file_id).cloned()
        };

        let mut record = record.ok_or_else(|| anyhow!("File record not found: {}", file_id))?;

        let file_path = record.file_path.as_ref()
            .ok_or_else(|| anyhow!("File path not available for: {}", file_id))?;

        // Check if file exists
        if !file_path.exists() {
            record.is_corrupted = true;
            self.update_record(&record).await;
            return Err(anyhow!("File does not exist: {:?}", file_path));
        }

        // Get file size
        let file_size = tokio::fs::metadata(file_path).await
            .map_err(|e| anyhow!("Failed to get file metadata: {}", e))?
            .len();

        // Quick size check
        if file_size != expected_data.len() as u64 {
            record.is_corrupted = true;
            self.update_record(&record).await;
            return Ok(false);
        }

        // Calculate current hashes using streaming approach
        let (current_blake3, current_sha256, _) = self.calculate_streaming_hashes(file_path).await?;

        // Verify against stored hashes
        let is_valid = current_blake3 == record.blake3_hash && current_sha256 == record.sha256_hash;

        // Update record
        record.last_verified = Utc::now();
        record.verification_count += 1;
        record.is_corrupted = !is_valid;

        self.update_record(&record).await;

        if !is_valid {
            log::warn!("File integrity check failed for: {}", file_id);
        }

        Ok(is_valid)
    }

    /// Verify all files in parallel using streaming processing
    pub async fn verify_all_files(&self) -> Result<VerificationResult> {
        let start_time = std::time::Instant::now();
        let file_ids: Vec<String> = {
            let records = self.records.read();
            records.keys().cloned().collect()
        };

        if file_ids.is_empty() {
            return Ok(VerificationResult {
                total_files: 0,
                verified_files: 0,
                corrupted_files: 0,
                failed_files: 0,
                total_size_bytes: 0,
                verification_time_ms: start_time.elapsed().as_millis() as u64,
                corrupted_file_ids: Vec::new(),
                failed_file_ids: Vec::new(),
            });
        }

        log::info!("Starting integrity verification for {} files", file_ids.len());

        // Process files in parallel chunks to limit resource usage
        let chunk_size = MAX_CONCURRENT_CHECKS;
        let mut verified_files = 0;
        let mut corrupted_files = 0;
        let mut failed_files = 0;
        let mut total_size_bytes = 0u64;
        let mut corrupted_file_ids = Vec::new();
        let mut failed_file_ids = Vec::new();

        for chunk in file_ids.chunks(chunk_size) {
            // Process chunk in parallel
            let results: Vec<_> = chunk.par_iter()
                .map(|file_id| {
                    // Use sync version for parallel processing
                    self.verify_file_sync(file_id)
                })
                .collect();

            // Collect results
            for (file_id, result) in chunk.iter().zip(results.iter()) {
                match result {
                    Ok((is_valid, file_size)) => {
                        total_size_bytes += file_size;
                        if *is_valid {
                            verified_files += 1;
                        } else {
                            corrupted_files += 1;
                            corrupted_file_ids.push(file_id.clone());
                        }
                    },
                    Err(e) => {
                        failed_files += 1;
                        failed_file_ids.push(file_id.clone());
                        log::error!("Failed to verify file {}: {}", file_id, e);
                    }
                }
            }
        }

        let verification_time_ms = start_time.elapsed().as_millis() as u64;

        log::info!("Integrity verification completed: {}/{} files verified, {} corrupted, {} failed in {}ms",
                  verified_files, file_ids.len(), corrupted_files, failed_files, verification_time_ms);

        Ok(VerificationResult {
            total_files: file_ids.len(),
            verified_files,
            corrupted_files,
            failed_files,
            total_size_bytes,
            verification_time_ms,
            corrupted_file_ids,
            failed_file_ids,
        })
    }

    /// Remove a file record and clean up associated data
    pub async fn remove_file_record(&self, file_id: &str) -> Result<()> {
        let record = {
            let mut records = self.records.write();
            records.remove(file_id)
        };

        if let Some(record) = record {
            // Clean up the actual file if it exists
            if let Some(file_path) = &record.file_path {
                if file_path.exists() {
                    tokio::fs::remove_file(file_path).await
                        .map_err(|e| anyhow!("Failed to remove file: {}", e))?;
                }
            }
            log::debug!("Removed integrity record and file for: {}", file_id);
        }

        Ok(())
    }

    /// Get integrity record for a file
    pub async fn get_file_record(&self, file_id: &str) -> Option<FileIntegrityRecord> {
        self.records.read().get(file_id).cloned()
    }

    /// Get all integrity records
    pub async fn get_all_records(&self) -> Vec<FileIntegrityRecord> {
        self.records.read().values().cloned().collect()
    }

    /// Get statistics about integrity records
    pub async fn get_integrity_stats(&self) -> HashMap<String, serde_json::Value> {
        let records = self.records.read();
        let total_files = records.len();
        let corrupted_files = records.values().filter(|r| r.is_corrupted).count();
        let total_size: u64 = records.values().map(|r| r.file_size).sum();
        let avg_verification_count = if total_files > 0 {
            records.values().map(|r| r.verification_count).sum::<u64>() as f64 / total_files as f64
        } else {
            0.0
        };

        let mut stats = HashMap::new();
        stats.insert("total_files".to_string(), serde_json::Value::Number(total_files.into()));
        stats.insert("corrupted_files".to_string(), serde_json::Value::Number(corrupted_files.into()));
        stats.insert("healthy_files".to_string(), serde_json::Value::Number((total_files - corrupted_files).into()));
        stats.insert("total_size_bytes".to_string(), serde_json::Value::Number(total_size.into()));
        stats.insert("average_verification_count".to_string(), 
                    serde_json::Value::Number(serde_json::Number::from_f64(avg_verification_count).unwrap_or_else(|| 0.into())));

        stats
    }

    /// Calculate streaming hashes for a file without loading it entirely into memory
    async fn calculate_streaming_hashes(&self, file_path: &Path) -> Result<(String, String, usize)> {
        let file_path = file_path.to_path_buf();
        let chunk_size = self.chunk_size;

        tokio::task::spawn_blocking(move || {
            let file = File::open(&file_path)
                .map_err(|e| anyhow!("Failed to open file: {}", e))?;

            let mut reader = BufReader::new(file);
            let mut blake3_hasher = Hasher::new();
            let mut sha256_hasher = Sha256::new();
            let mut buffer = vec![0u8; chunk_size];
            let mut chunk_count = 0;

            loop {
                let bytes_read = reader.read(&mut buffer)
                    .map_err(|e| anyhow!("Failed to read file chunk: {}", e))?;

                if bytes_read == 0 {
                    break; // End of file
                }

                // Update both hashers with the chunk
                let chunk = &buffer[..bytes_read];
                blake3_hasher.update(chunk);
                sha256_hasher.update(chunk);
                chunk_count += 1;
            }

            // Finalize hashes
            let blake3_hash = blake3_hasher.finalize();
            let sha256_hash = sha256_hasher.finalize();

            Ok((
                blake3_hash.to_hex().to_string(),
                format!("{:x}", sha256_hash),
                chunk_count,
            ))
        }).await.map_err(|e| anyhow!("Hash calculation task failed: {}", e))?
    }

    /// Synchronous version of verify_file for parallel processing
    fn verify_file_sync(&self, file_id: &str) -> Result<(bool, u64)> {
        let record = {
            let records = self.records.read();
            records.get(file_id).cloned()
        };

        let mut record = record.ok_or_else(|| anyhow!("File record not found: {}", file_id))?;

        let file_path = record.file_path.as_ref()
            .ok_or_else(|| anyhow!("File path not available for: {}", file_id))?;

        // Check if file exists
        if !file_path.exists() {
            record.is_corrupted = true;
            // Note: We can't await in sync context, so we'll mark it corrupted
            // and update later if needed
            return Err(anyhow!("File does not exist: {:?}", file_path));
        }

        // Get file size
        let file_size = std::fs::metadata(file_path)
            .map_err(|e| anyhow!("Failed to get file metadata: {}", e))?
            .len();

        // Quick size check
        if file_size != record.file_size {
            record.is_corrupted = true;
            return Ok((false, file_size));
        }

        // Calculate current hashes using streaming approach (sync version)
        let (current_blake3, current_sha256, _) = self.calculate_streaming_hashes_sync(file_path)?;

        // Verify against stored hashes
        let is_valid = current_blake3 == record.blake3_hash && current_sha256 == record.sha256_hash;

        if !is_valid {
            log::warn!("File integrity check failed for: {}", file_id);
        }

        Ok((is_valid, file_size))
    }

    /// Synchronous streaming hash calculation
    fn calculate_streaming_hashes_sync(&self, file_path: &Path) -> Result<(String, String, usize)> {
        let file = File::open(file_path)
            .map_err(|e| anyhow!("Failed to open file: {}", e))?;

        let mut reader = BufReader::new(file);
        let mut blake3_hasher = Hasher::new();
        let mut sha256_hasher = Sha256::new();
        let mut buffer = vec![0u8; self.chunk_size];
        let mut chunk_count = 0;

        loop {
            let bytes_read = reader.read(&mut buffer)
                .map_err(|e| anyhow!("Failed to read file chunk: {}", e))?;

            if bytes_read == 0 {
                break; // End of file
            }

            // Update both hashers with the chunk
            let chunk = &buffer[..bytes_read];
            blake3_hasher.update(chunk);
            sha256_hasher.update(chunk);
            chunk_count += 1;
        }

        // Finalize hashes
        let blake3_hash = blake3_hasher.finalize();
        let sha256_hash = sha256_hasher.finalize();

        Ok((
            blake3_hash.to_hex().to_string(),
            format!("{:x}", sha256_hash),
            chunk_count,
        ))
    }

    /// Update a record in storage
    async fn update_record(&self, record: &FileIntegrityRecord) {
        self.records.write().insert(record.file_id.clone(), record.clone());
    }

    /// Repair corrupted file by recalculating its integrity record
    pub async fn repair_file(&self, file_id: &str) -> Result<bool> {
        let file_path = {
            let records = self.records.read();
            records.get(file_id)
                .and_then(|r| r.file_path.clone())
                .ok_or_else(|| anyhow!("File record not found: {}", file_id))?
        };

        if !file_path.exists() {
            return Err(anyhow!("Cannot repair non-existent file: {:?}", file_path));
        }

        // Recalculate hashes
        let (blake3_hash, sha256_hash, chunk_count) = self.calculate_streaming_hashes(&file_path).await?;

        // Get file size
        let file_size = tokio::fs::metadata(&file_path).await
            .map_err(|e| anyhow!("Failed to get file metadata: {}", e))?
            .len();

        // Update record with new values
        let mut records = self.records.write();
        if let Some(record) = records.get_mut(file_id) {
            record.blake3_hash = blake3_hash;
            record.sha256_hash = sha256_hash;
            record.file_size = file_size;
            record.chunk_count = chunk_count;
            record.last_verified = Utc::now();
            record.verification_count += 1;
            record.is_corrupted = false;
            
            log::info!("Repaired integrity record for file: {}", file_id);
            return Ok(true);
        }

        Err(anyhow!("File record disappeared during repair: {}", file_id))
    }
}

impl Default for IntegrityChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_streaming_hash_calculation() {
        let temp_dir = tempdir().unwrap();
        let checker = IntegrityChecker::with_base_path(temp_dir.path());

        let test_data = b"Hello, world! This is a test for streaming hash calculation.";
        
        checker.add_file_record("test_file", test_data).await.unwrap();
        
        let record = checker.get_file_record("test_file").await.unwrap();
        assert!(!record.blake3_hash.is_empty());
        assert!(!record.sha256_hash.is_empty());
        assert_eq!(record.file_size, test_data.len() as u64);
    }

    #[tokio::test]
    async fn test_integrity_verification() {
        let temp_dir = tempdir().unwrap();
        let checker = IntegrityChecker::with_base_path(temp_dir.path());

        let test_data = b"Test data for integrity verification";
        
        checker.add_file_record("test_file", test_data).await.unwrap();
        
        // Verify with correct data
        let is_valid = checker.verify_file("test_file", test_data).await.unwrap();
        assert!(is_valid);

        // Verify with incorrect data
        let wrong_data = b"Wrong data";
        let is_valid = checker.verify_file("test_file", wrong_data).await.unwrap();
        assert!(!is_valid);
    }

    #[tokio::test]
    async fn test_large_file_streaming() {
        let temp_dir = tempdir().unwrap();
        let checker = IntegrityChecker::with_base_path(temp_dir.path());

        // Create a large test file (1MB of repeated pattern)
        let pattern = b"This is a test pattern for large file handling. ";
        let repetitions = 1024 * 1024 / pattern.len();
        let large_data: Vec<u8> = pattern.iter().cycle().take(repetitions * pattern.len()).cloned().collect();

        checker.add_file_record("large_test_file", &large_data).await.unwrap();
        
        let is_valid = checker.verify_file("large_test_file", &large_data).await.unwrap();
        assert!(is_valid);

        let record = checker.get_file_record("large_test_file").await.unwrap();
        assert_eq!(record.file_size, large_data.len() as u64);
        assert!(record.chunk_count > 1); // Should be processed in multiple chunks
    }
}