//! Cache management for performance optimization
//! 
//! This module provides intelligent caching including:
//! - Multi-level cache hierarchy (memory, disk)
//! - LRU eviction policies
//! - TTL-based expiration
//! - Cache warming and preloading
//! - Compression for large items
//! - Thread-safe access patterns

use super::*;
use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use std::path::PathBuf;
use std::fs::{self, File};
use std::io::{Read, Write};
use zstd::stream::{Encoder as ZstdEncoder, Decoder as ZstdDecoder};
use std::io::{Error as IoError, Result as IoResult};

/// Cache manager for performance optimization
pub struct CacheManager {
    config: DataManagerConfig,
    memory_cache: HashMap<String, CacheEntry>,
    access_times: BTreeMap<Instant, String>,
    cache_stats: CacheStats,
    max_memory_size: usize,
    current_memory_size: usize,
}

impl CacheManager {
    /// Create new cache manager
    pub fn new(config: &DataManagerConfig) -> DataResult<Self> {
        // Ensure cache directory exists
        std::fs::create_dir_all(&config.cache_dir)
            .map_err(|e| DataError::Cache {
                message: format!("Failed to create cache directory: {}", e),
            })?;
        
        Ok(Self {
            config: config.clone(),
            memory_cache: HashMap::new(),
            access_times: BTreeMap::new(),
            cache_stats: CacheStats::default(),
            max_memory_size: (config.cache_size_limit_mb * 1024 * 1024) as usize,
            current_memory_size: 0,
        })
    }
    
    /// Get item from cache
    pub async fn get<T>(&mut self, key: &str) -> DataResult<Option<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        self.cache_stats.total_requests += 1;
        
        // Check memory cache first
        if let Some(entry) = self.memory_cache.get(key) {
            // Check if expired
            if let Some(ttl) = entry.expires_at {
                if Instant::now() > ttl {
                    self.remove_from_memory(key);
                    self.cache_stats.expiration_count += 1;
                } else {
                    // Update access time
                    self.update_access_time(key);
                    self.cache_stats.memory_hits += 1;
                    
                    let data = serde_json::from_slice(&entry.data)
                        .map_err(|e| DataError::Cache {
                            message: format!("Failed to deserialize cached data: {}", e),
                        })?;
                    return Ok(Some(data));
                }
            } else {
                // No expiration, update access time
                self.update_access_time(key);
                self.cache_stats.memory_hits += 1;
                
                let data = serde_json::from_slice(&entry.data)
                    .map_err(|e| DataError::Cache {
                        message: format!("Failed to deserialize cached data: {}", e),
                    })?;
                return Ok(Some(data));
            }
        }
        
        // Check disk cache
        let disk_path = self.get_disk_cache_path(key);
        if disk_path.exists() {
            match self.load_from_disk(key) {
                Ok(Some(entry)) => {
                    // Check if expired
                    if let Some(ttl) = entry.expires_at {
                        if Instant::now() > ttl {
                            let _ = fs::remove_file(&disk_path);
                            self.cache_stats.expiration_count += 1;
                        } else {
                            // Promote to memory cache if there's room
                            if self.current_memory_size + entry.size <= self.max_memory_size {
                                self.add_to_memory(key.to_string(), entry.clone());
                            }
                            
                            self.cache_stats.disk_hits += 1;
                            
                            let data = serde_json::from_slice(&entry.data)
                                .map_err(|e| DataError::Cache {
                                    message: format!("Failed to deserialize cached data: {}", e),
                                })?;
                            return Ok(Some(data));
                        }
                    } else {
                        // No expiration, promote to memory if there's room
                        if self.current_memory_size + entry.size <= self.max_memory_size {
                            self.add_to_memory(key.to_string(), entry.clone());
                        }
                        
                        self.cache_stats.disk_hits += 1;
                        
                        let data = serde_json::from_slice(&entry.data)
                            .map_err(|e| DataError::Cache {
                                message: format!("Failed to deserialize cached data: {}", e),
                            })?;
                        return Ok(Some(data));
                    }
                },
                Ok(None) => {
                    // File exists but couldn't be loaded, remove it
                    let _ = fs::remove_file(&disk_path);
                },
                Err(e) => {
                    log::warn!("Failed to load from disk cache: {}", e);
                    let _ = fs::remove_file(&disk_path);
                }
            }
        }
        
        self.cache_stats.misses += 1;
        Ok(None)
    }
    
    /// Put item in cache
    pub async fn put<T>(&mut self, key: &str, value: &T, ttl: Option<Duration>) -> DataResult<()>
    where
        T: Serialize,
    {
        let data = serde_json::to_vec(value)
            .map_err(|e| DataError::Cache {
                message: format!("Failed to serialize cache data: {}", e),
            })?;
        
        let expires_at = ttl.map(|t| Instant::now() + t);
        let size = data.len();
        
        let entry = CacheEntry {
            data,
            size,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 1,
            expires_at,
            is_compressed: false,
        };
        
        // Try to add to memory cache first
        if size <= self.max_memory_size {
            // Make room if necessary
            while self.current_memory_size + size > self.max_memory_size {
                if !self.evict_lru() {
                    break; // No more items to evict
                }
            }
            
            if self.current_memory_size + size <= self.max_memory_size {
                self.add_to_memory(key.to_string(), entry);
                self.cache_stats.memory_writes += 1;
                return Ok(());
            }
        }
        
        // Fall back to disk cache
        self.save_to_disk(key, &entry).await?;
        self.cache_stats.disk_writes += 1;
        
        Ok(())
    }
    
    /// Remove item from cache
    pub async fn remove(&mut self, key: &str) -> DataResult<bool> {
        let mut removed = false;
        
        // Remove from memory cache
        if self.remove_from_memory(key) {
            removed = true;
        }
        
        // Remove from disk cache
        let disk_path = self.get_disk_cache_path(key);
        if disk_path.exists() {
            fs::remove_file(&disk_path)
                .map_err(|e| DataError::Cache {
                    message: format!("Failed to remove disk cache file: {}", e),
                })?;
            removed = true;
        }
        
        if removed {
            self.cache_stats.removals += 1;
        }
        
        Ok(removed)
    }
    
    /// Clear all cached items
    pub async fn clear(&mut self) -> DataResult<()> {
        // Clear memory cache
        self.memory_cache.clear();
        self.access_times.clear();
        self.current_memory_size = 0;
        
        // Clear disk cache
        if self.config.cache_dir.exists() {
            for entry in fs::read_dir(&self.config.cache_dir)
                .map_err(|e| DataError::Cache {
                    message: format!("Failed to read cache directory: {}", e),
                })?
            {
                let entry = entry.map_err(|e| DataError::Cache {
                    message: format!("Failed to read cache entry: {}", e),
                })?;
                
                if entry.file_type().map_err(|e| DataError::Cache {
                    message: format!("Failed to get file type: {}", e),
                })?.is_file() {
                    fs::remove_file(entry.path())
                        .map_err(|e| DataError::Cache {
                            message: format!("Failed to remove cache file: {}", e),
                        })?;
                }
            }
        }
        
        // Reset stats
        self.cache_stats = CacheStats::default();
        
        log::info!("Cache cleared successfully");
        Ok(())
    }
    
    /// Cleanup expired entries
    pub async fn cleanup_expired(&mut self) -> DataResult<()> {
        let now = Instant::now();
        let mut expired_keys = Vec::new();
        
        // Find expired memory cache entries
        for (key, entry) in &self.memory_cache {
            if let Some(expires_at) = entry.expires_at {
                if now > expires_at {
                    expired_keys.push(key.clone());
                }
            }
        }
        
        // Remove expired memory cache entries
        for key in expired_keys {
            self.remove_from_memory(&key);
            self.cache_stats.expiration_count += 1;
        }
        
        // Cleanup expired disk cache files
        if self.config.cache_dir.exists() {
            for entry in fs::read_dir(&self.config.cache_dir)
                .map_err(|e| DataError::Cache {
                    message: format!("Failed to read cache directory: {}", e),
                })?
            {
                let entry = entry.map_err(|e| DataError::Cache {
                    message: format!("Failed to read cache entry: {}", e),
                })?;
                let path = entry.path();
                
                if path.is_file() {
                    // Try to load and check expiration
                    let key = path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("");
                    
                    if let Ok(Some(cache_entry)) = self.load_from_disk(key) {
                        if let Some(expires_at) = cache_entry.expires_at {
                            if now > expires_at {
                                let _ = fs::remove_file(&path);
                                self.cache_stats.expiration_count += 1;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Enforce cache size limits
    pub async fn enforce_size_limit(&mut self, max_size_bytes: u64) -> DataResult<()> {
        let max_size = max_size_bytes as usize;
        
        // Enforce memory cache size limit
        while self.current_memory_size > max_size {
            if !self.evict_lru() {
                break; // No more items to evict
            }
        }
        
        // Check disk cache size
        let disk_size = self.calculate_disk_cache_size()?;
        if disk_size > max_size_bytes {
            self.cleanup_disk_cache_by_size(max_size_bytes).await?;
        }
        
        Ok(())
    }
    
    /// Add item to memory cache
    fn add_to_memory(&mut self, key: String, entry: CacheEntry) {
        let size = entry.size;
        let access_time = entry.last_accessed;
        
        // Remove old entry if exists
        if self.memory_cache.contains_key(&key) {
            self.remove_from_memory(&key);
        }
        
        self.memory_cache.insert(key.clone(), entry);
        self.access_times.insert(access_time, key);
        self.current_memory_size += size;
    }
    
    /// Remove item from memory cache
    fn remove_from_memory(&mut self, key: &str) -> bool {
        if let Some(entry) = self.memory_cache.remove(key) {
            self.current_memory_size -= entry.size;
            
            // Remove from access times (find by value since access time might have changed)
            let access_time_to_remove = self.access_times
                .iter()
                .find_map(|(time, k)| if k == key { Some(*time) } else { None });
            
            if let Some(time) = access_time_to_remove {
                self.access_times.remove(&time);
            }
            
            return true;
        }
        false
    }
    
    /// Update access time for cache entry
    fn update_access_time(&mut self, key: &str) {
        if let Some(entry) = self.memory_cache.get_mut(key) {
            let old_access_time = entry.last_accessed;
            let new_access_time = Instant::now();
            
            entry.last_accessed = new_access_time;
            entry.access_count += 1;
            
            // Update access times map
            self.access_times.remove(&old_access_time);
            self.access_times.insert(new_access_time, key.to_string());
        }
    }
    
    /// Evict least recently used item
    fn evict_lru(&mut self) -> bool {
        if let Some((_, key)) = self.access_times.iter().next() {
            let key = key.clone();
            self.remove_from_memory(&key);
            self.cache_stats.evictions += 1;
            return true;
        }
        false
    }
    
    /// Get disk cache path for key
    fn get_disk_cache_path(&self, key: &str) -> PathBuf {
        let safe_key = key.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
        self.config.cache_dir.join(format!("{}.cache", safe_key))
    }
    
    /// Save entry to disk
    async fn save_to_disk(&self, key: &str, entry: &CacheEntry) -> DataResult<()> {
        let disk_path = self.get_disk_cache_path(key);
        
        let cache_file = CacheFile {
            data: entry.data.clone(),
            size: entry.size,
            created_at: SystemTime::now(),
            expires_at: entry.expires_at.map(|instant| {
                SystemTime::now() + instant.duration_since(Instant::now())
            }),
            access_count: entry.access_count,
            is_compressed: false,
        };
        
        // Compress if data is large
        let final_cache_file = if entry.size > 1024 {
            // Compress the data
            let mut encoder = ZstdEncoder::new(Vec::new(), 3)
                .map_err(|e| DataError::Cache {
                    message: format!("Failed to create compression encoder: {}", e),
                })?;
            encoder.write_all(&entry.data)
                .map_err(|e| DataError::Cache {
                    message: format!("Failed to write to compression encoder: {}", e),
                })?;
            let compressed_data = encoder.finish()
                .map_err(|e| DataError::Cache {
                    message: format!("Failed to finish compression: {}", e),
                })?;
            
            CacheFile {
                data: compressed_data,
                is_compressed: true,
                ..cache_file
            }
        } else {
            cache_file
        };
        
        let serialized = serde_json::to_vec(&final_cache_file)
            .map_err(|e| DataError::Cache {
                message: format!("Failed to serialize cache file: {}", e),
            })?;
        
        fs::write(&disk_path, serialized)
            .map_err(|e| DataError::Cache {
                message: format!("Failed to write cache file: {}", e),
            })?;
        
        Ok(())
    }
    
    /// Load entry from disk
    fn load_from_disk(&self, key: &str) -> DataResult<Option<CacheEntry>> {
        let disk_path = self.get_disk_cache_path(key);
        
        if !disk_path.exists() {
            return Ok(None);
        }
        
        let file_content = fs::read(&disk_path)
            .map_err(|e| DataError::Cache {
                message: format!("Failed to read cache file: {}", e),
            })?;
        
        let cache_file: CacheFile = serde_json::from_slice(&file_content)
            .map_err(|e| DataError::Cache {
                message: format!("Failed to deserialize cache file: {}", e),
            })?;
        
        // Decompress if necessary
        let data = if cache_file.is_compressed {
            let mut decoder = ZstdDecoder::new(&cache_file.data[..])
                .map_err(|e| DataError::Cache {
                    message: format!("Failed to create decompression decoder: {}", e),
                })?;
            let mut decompressed_data = Vec::new();
            decoder.read_to_end(&mut decompressed_data)
                .map_err(|e| DataError::Cache {
                    message: format!("Failed to decompress data: {}", e),
                })?;
            decompressed_data
        } else {
            cache_file.data
        };
        
        let expires_at = cache_file.expires_at.map(|system_time| {
            let now = SystemTime::now();
            if system_time > now {
                Instant::now() + system_time.duration_since(now).unwrap_or_default()
            } else {
                Instant::now() // Already expired
            }
        });
        
        Ok(Some(CacheEntry {
            data,
            size: cache_file.size,
            created_at: Instant::now(), // Reset creation time for memory cache
            last_accessed: Instant::now(),
            access_count: cache_file.access_count,
            expires_at,
            is_compressed: false, // Decompressed for memory cache
        }))
    }
    
    /// Calculate disk cache size
    fn calculate_disk_cache_size(&self) -> DataResult<u64> {
        let mut total_size = 0u64;
        
        if !self.config.cache_dir.exists() {
            return Ok(0);
        }
        
        for entry in fs::read_dir(&self.config.cache_dir)
            .map_err(|e| DataError::Cache {
                message: format!("Failed to read cache directory: {}", e),
            })?
        {
            let entry = entry.map_err(|e| DataError::Cache {
                message: format!("Failed to read cache entry: {}", e),
            })?;
            
            if entry.file_type().map_err(|e| DataError::Cache {
                message: format!("Failed to get file type: {}", e),
            })?.is_file() {
                let metadata = entry.metadata().map_err(|e| DataError::Cache {
                    message: format!("Failed to get file metadata: {}", e),
                })?;
                total_size += metadata.len();
            }
        }
        
        Ok(total_size)
    }
    
    /// Cleanup disk cache by size (LRU)
    async fn cleanup_disk_cache_by_size(&self, max_size: u64) -> DataResult<()> {
        let mut files_with_times = Vec::new();
        
        if !self.config.cache_dir.exists() {
            return Ok(());
        }
        
        // Collect all cache files with their access times
        for entry in fs::read_dir(&self.config.cache_dir)
            .map_err(|e| DataError::Cache {
                message: format!("Failed to read cache directory: {}", e),
            })?
        {
            let entry = entry.map_err(|e| DataError::Cache {
                message: format!("Failed to read cache entry: {}", e),
            })?;
            let path = entry.path();
            
            if path.is_file() {
                let metadata = entry.metadata().map_err(|e| DataError::Cache {
                    message: format!("Failed to get file metadata: {}", e),
                })?;
                
                let access_time = metadata.accessed()
                    .or_else(|_| metadata.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                
                files_with_times.push((path, metadata.len(), access_time));
            }
        }
        
        // Sort by access time (oldest first)
        files_with_times.sort_by_key(|(_, _, access_time)| *access_time);
        
        let mut current_size: u64 = files_with_times.iter().map(|(_, size, _)| size).sum();
        
        // Remove oldest files until under size limit
        for (path, size, _) in files_with_times {
            if current_size <= max_size {
                break;
            }
            
            if let Err(e) = fs::remove_file(&path) {
                log::warn!("Failed to remove cache file {}: {}", path.display(), e);
            } else {
                current_size -= size;
            }
        }
        
        Ok(())
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let mut stats = self.cache_stats.clone();
        stats.memory_entries = self.memory_cache.len();
        stats.memory_size_bytes = self.current_memory_size;
        stats.hit_rate = if stats.total_requests > 0 {
            ((stats.memory_hits + stats.disk_hits) as f64 / stats.total_requests as f64) * 100.0
        } else {
            0.0
        };
        stats
    }
    
    /// Warm cache with frequently accessed data
    pub async fn warm_cache(&mut self, warm_data: Vec<(String, serde_json::Value)>) -> DataResult<()> {
        for (key, value) in warm_data {
            self.put(&key, &value, None).await?;
        }
        log::info!("Cache warmed with {} entries", self.memory_cache.len());
        Ok(())
    }
}

/// Cache entry stored in memory
#[derive(Debug, Clone)]
struct CacheEntry {
    data: Vec<u8>,
    size: usize,
    created_at: Instant,
    last_accessed: Instant,
    access_count: u64,
    expires_at: Option<Instant>,
    is_compressed: bool,
}

/// Cache entry stored on disk
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheFile {
    data: Vec<u8>,
    size: usize,
    created_at: SystemTime,
    expires_at: Option<SystemTime>,
    access_count: u64,
    is_compressed: bool,
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_requests: u64,
    pub memory_hits: u64,
    pub disk_hits: u64,
    pub misses: u64,
    pub memory_writes: u64,
    pub disk_writes: u64,
    pub evictions: u64,
    pub removals: u64,
    pub expiration_count: u64,
    pub memory_entries: usize,
    pub memory_size_bytes: usize,
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_cache_get_put() {
        let temp_dir = tempdir().unwrap();
        let config = DataManagerConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            cache_size_limit_mb: 10,
            ..Default::default()
        };
        
        let mut cache = CacheManager::new(&config).unwrap();
        
        // Test put and get
        let test_data = serde_json::json!({"test": "data", "number": 42});
        cache.put("test_key", &test_data, None).await.unwrap();
        
        let retrieved: Option<serde_json::Value> = cache.get("test_key").await.unwrap();
        assert_eq!(retrieved, Some(test_data));
        
        // Test miss
        let missing: Option<serde_json::Value> = cache.get("nonexistent_key").await.unwrap();
        assert_eq!(missing, None);
    }
    
    #[tokio::test]
    async fn test_cache_expiration() {
        let temp_dir = tempdir().unwrap();
        let config = DataManagerConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            cache_size_limit_mb: 10,
            ..Default::default()
        };
        
        let mut cache = CacheManager::new(&config).unwrap();
        
        // Put item with short TTL
        let test_data = serde_json::json!({"expires": "soon"});
        cache.put("expire_test", &test_data, Some(Duration::from_millis(50))).await.unwrap();
        
        // Should be available immediately
        let retrieved: Option<serde_json::Value> = cache.get("expire_test").await.unwrap();
        assert_eq!(retrieved, Some(test_data));
        
        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(60)).await;
        
        // Should be expired
        let expired: Option<serde_json::Value> = cache.get("expire_test").await.unwrap();
        assert_eq!(expired, None);
    }
}