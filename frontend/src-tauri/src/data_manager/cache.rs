use anyhow::Result;
use std::collections::{HashMap, BTreeMap};
use std::time::{Instant, Duration};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use dashmap::DashMap;
use thiserror::Error;

/// Cache-specific error types
#[derive(Debug, Error)]
pub enum CacheError {
    #[error("Cache size must be greater than 0")]
    InvalidSize,
    #[error("Cache operation failed: {0}")]
    Operation(String),
    #[error("Entry not found: {0}")]
    EntryNotFound(String),
}

/// Cache entry with metadata and efficient access tracking
#[derive(Debug, Clone)]
struct CacheEntry {
    data: Arc<Vec<u8>>, // Use Arc to reduce cloning overhead
    created_at: Instant,
    last_accessed: Instant,
    access_count: u64,
    size: usize,
}

impl CacheEntry {
    fn new(data: Vec<u8>) -> Self {
        let size = data.len();
        let now = Instant::now();
        Self {
            data: Arc::new(data),
            created_at: now,
            last_accessed: now,
            access_count: 1,
            size,
        }
    }

    /// Update access time and count
    fn touch(&mut self) -> Instant {
        let old_access = self.last_accessed;
        self.last_accessed = Instant::now();
        self.access_count += 1;
        old_access
    }

    /// Get data without cloning the entire vector
    fn get_data(&self) -> Arc<Vec<u8>> {
        Arc::clone(&self.data)
    }

    /// Check if entry is stale based on age
    fn is_stale(&self, max_age: Duration) -> bool {
        self.last_accessed.elapsed() > max_age
    }
}

/// Configuration for cache behavior
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub max_size_bytes: usize,
    pub max_entry_age: Option<Duration>,
    pub eviction_batch_size: usize,
}

impl CacheConfig {
    pub fn new(max_size_bytes: usize) -> Result<Self, CacheError> {
        if max_size_bytes == 0 {
            return Err(CacheError::InvalidSize);
        }
        
        Ok(Self {
            max_size_bytes,
            max_entry_age: None,
            eviction_batch_size: max_size_bytes / 10, // Evict 10% by default
        })
    }

    pub fn with_max_age(mut self, max_age: Duration) -> Self {
        self.max_entry_age = Some(max_age);
        self
    }

    pub fn with_eviction_batch_size(mut self, batch_size: usize) -> Self {
        self.eviction_batch_size = batch_size;
        self
    }
}

/// Statistics about cache performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_size_bytes: usize,
    pub max_size_bytes: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
    pub hit_ratio: f64,
    pub average_entry_size: usize,
    pub oldest_entry_age_seconds: Option<u64>,
    pub utilization_percent: f64,
}

impl CacheStats {
    /// Calculate hit ratio
    pub fn calculate_hit_ratio(hits: u64, misses: u64) -> f64 {
        let total = hits + misses;
        if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Calculate utilization percentage
    pub fn calculate_utilization(current_size: usize, max_size: usize) -> f64 {
        if max_size > 0 {
            (current_size as f64 / max_size as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Efficient LRU cache with simplified design and better performance
/// Uses DashMap for concurrent access and simplified tracking
pub struct CacheManager {
    /// Main cache storage with concurrent access
    entries: DashMap<String, CacheEntry>,
    
    /// Configuration
    config: CacheConfig,
    
    /// Performance statistics
    stats: Arc<RwLock<CacheStatistics>>,
}

/// Internal statistics tracking
#[derive(Debug, Default)]
struct CacheStatistics {
    current_size_bytes: usize,
    hit_count: u64,
    miss_count: u64,
    eviction_count: u64,
}

impl CacheManager {
    /// Create a new cache manager with specified configuration
    pub fn new(max_size_bytes: usize) -> Result<Self, CacheError> {
        let config = CacheConfig::new(max_size_bytes)?;
        Self::with_config(config)
    }

    /// Create cache manager with custom configuration
    pub fn with_config(config: CacheConfig) -> Result<Self, CacheError> {
        Ok(Self {
            entries: DashMap::new(),
            config,
            stats: Arc::new(RwLock::new(CacheStatistics::default())),
        })
    }

    /// Store data in cache with efficient operations and automatic eviction
    pub async fn put(&self, key: String, data: Vec<u8>) -> Result<(), CacheError> {
        let entry_size = data.len() + key.len();
        
        // Remove existing entry if present
        if let Some((_, old_entry)) = self.entries.remove(&key) {
            self.update_size_after_removal(&old_entry, &key);
        }

        // Create and insert new entry
        let entry = CacheEntry::new(data);
        self.entries.insert(key.clone(), entry);
        
        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.current_size_bytes += entry_size;
        }

        // Trigger eviction if needed
        self.evict_if_needed().await;
        Ok(())
    }

    /// Retrieve data from cache with efficient access tracking
    pub async fn get(&self, key: &str) -> Option<Arc<Vec<u8>>> {
        match self.entries.get_mut(key) {
            Some(mut entry) => {
                // Check if entry is stale
                if let Some(max_age) = self.config.max_entry_age {
                    if entry.is_stale(max_age) {
                        // Remove stale entry
                        drop(entry);
                        self.remove(key).await;
                        self.stats.write().miss_count += 1;
                        return None;
                    }
                }

                // Update access time and get data
                entry.touch();
                let data = entry.get_data();
                
                // Update statistics
                self.stats.write().hit_count += 1;
                Some(data)
            },
            None => {
                self.stats.write().miss_count += 1;
                None
            }
        }
    }

    /// Retrieve data and convert to Vec<u8> for backward compatibility
    pub async fn get_vec(&self, key: &str) -> Option<Vec<u8>> {
        self.get(key).await.map(|arc| (*arc).clone())
    }

    /// Check if key exists in cache
    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Remove entry from cache
    pub async fn remove(&self, key: &str) -> Option<Arc<Vec<u8>>> {
        self.entries.remove(key).map(|(_, entry)| {
            self.update_size_after_removal(&entry, key);
            entry.get_data()
        })
    }

    /// Remove and return as Vec<u8> for backward compatibility
    pub async fn remove_vec(&self, key: &str) -> Option<Vec<u8>> {
        self.remove(key).await.map(|arc| (*arc).clone())
    }

    /// Clear all entries from cache
    pub fn clear(&self) {
        self.entries.clear();
        let mut stats = self.stats.write();
        stats.current_size_bytes = 0;
    }

    /// Get current cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let total_entries = self.entries.len();
        let stats = self.stats.read();
        
        let hit_ratio = CacheStats::calculate_hit_ratio(stats.hit_count, stats.miss_count);
        let utilization_percent = CacheStats::calculate_utilization(
            stats.current_size_bytes, 
            self.config.max_size_bytes
        );
        
        let average_entry_size = if total_entries > 0 {
            stats.current_size_bytes / total_entries
        } else {
            0
        };
        
        // Find oldest entry age by scanning entries
        let oldest_entry_age_seconds = self.entries
            .iter()
            .map(|entry| entry.created_at.elapsed().as_secs())
            .max();
        
        CacheStats {
            total_entries,
            total_size_bytes: stats.current_size_bytes,
            max_size_bytes: self.config.max_size_bytes,
            hit_count: stats.hit_count,
            miss_count: stats.miss_count,
            eviction_count: stats.eviction_count,
            hit_ratio,
            average_entry_size,
            oldest_entry_age_seconds,
            utilization_percent,
        }
    }

    /// Get all keys in cache
    pub fn get_keys(&self) -> Vec<String> {
        self.entries.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Get cache utilization as a percentage
    pub fn get_utilization(&self) -> f64 {
        let stats = self.stats.read();
        CacheStats::calculate_utilization(stats.current_size_bytes, self.config.max_size_bytes)
    }

    /// Get keys sorted by access time (most recent first)
    pub fn get_keys_by_access_time(&self) -> Vec<String> {
        let mut entries: Vec<_> = self.entries
            .iter()
            .map(|entry| (entry.key().clone(), entry.last_accessed))
            .collect();
        
        entries.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by access time, most recent first
        entries.into_iter().map(|(key, _)| key).collect()
    }

    /// Manually trigger cache eviction to free space
    pub async fn evict_entries(&self, target_size_bytes: usize) -> usize {
        let mut evicted_count = 0;
        
        while self.stats.read().current_size_bytes > target_size_bytes {
            // Find the least recently used entry
            let lru_key = self.find_lru_key();
            
            if let Some(key) = lru_key {
                if self.evict_entry(&key) {
                    evicted_count += 1;
                } else {
                    break; // Failed to evict, exit loop
                }
            } else {
                break; // No more entries to evict
            }
        }
        
        evicted_count
    }

    /// Find the least recently used key (O(n) operation)
    fn find_lru_key(&self) -> Option<String> {
        self.entries
            .iter()
            .min_by_key(|entry| entry.last_accessed)
            .map(|entry| entry.key().clone())
    }

    /// Evict a specific entry
    fn evict_entry(&self, key: &str) -> bool {
        if let Some((_, entry)) = self.entries.remove(key) {
            self.update_size_after_removal(&entry, key);
            self.stats.write().eviction_count += 1;
            log::debug!("Evicted cache entry: {}", key);
            true
        } else {
            false
        }
    }

    /// Check if eviction is needed and perform it
    async fn evict_if_needed(&self) {
        let current_size = self.stats.read().current_size_bytes;
        
        if current_size > self.config.max_size_bytes {
            // Calculate target size (80% of max to provide buffer)
            let target_size = (self.config.max_size_bytes as f64 * 0.8) as usize;
            let evicted = self.evict_entries(target_size).await;
            
            if evicted > 0 {
                log::debug!("Cache eviction: removed {} entries", evicted);
            }
        }
    }

}

impl Drop for CacheManager {
    fn drop(&mut self) {
        let stats = self.get_stats();
        log::info!(
            "Cache manager dropped. Final stats: hit_ratio={:.2}%, entries={}, size={}MB",
            stats.hit_ratio * 100.0,
            stats.total_entries,
            stats.total_size_bytes / (1024 * 1024)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio;

    #[tokio::test]
    async fn test_cache_basic_operations() {
        let cache = CacheManager::new(1024).unwrap();
        
        // Test put and get
        cache.put("key1".to_string(), b"data1".to_vec()).await.unwrap();
        let retrieved = cache.get_vec("key1").await;
        assert_eq!(retrieved, Some(b"data1".to_vec()));
        
        // Test contains
        assert!(cache.contains_key("key1"));
        assert!(!cache.contains_key("nonexistent"));
        
        // Test remove
        let removed = cache.remove_vec("key1").await;
        assert_eq!(removed, Some(b"data1".to_vec()));
        assert!(!cache.contains_key("key1"));
    }

    #[tokio::test]
    async fn test_cache_arc_operations() {
        let cache = CacheManager::new(1024).unwrap();
        
        // Test Arc-based operations for efficiency
        cache.put("key1".to_string(), b"data1".to_vec()).await.unwrap();
        let retrieved_arc = cache.get("key1").await;
        assert!(retrieved_arc.is_some());
        
        let data_arc = retrieved_arc.unwrap();
        assert_eq!(*data_arc, b"data1".to_vec());
        
        // Test that Arc allows sharing without cloning
        let data_arc2 = cache.get("key1").await.unwrap();
        assert!(Arc::ptr_eq(&data_arc, &data_arc2));
    }

    #[tokio::test]
    async fn test_cache_config() {
        let config = CacheConfig::new(1024)
            .unwrap()
            .with_max_age(Duration::from_secs(60))
            .with_eviction_batch_size(100);
        
        let cache = CacheManager::with_config(config).unwrap();
        
        // Test configuration is applied
        assert_eq!(cache.config.max_size_bytes, 1024);
        assert_eq!(cache.config.max_entry_age, Some(Duration::from_secs(60)));
        assert_eq!(cache.config.eviction_batch_size, 100);
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let cache = CacheManager::new(100).unwrap(); // Small cache for testing
        
        // Fill cache beyond capacity
        for i in 0..10 {
            let key = format!("key{}", i);
            let data = vec![0u8; 15]; // 15 bytes per entry + key overhead
            cache.put(key, data).await.unwrap();
        }
        
        let stats = cache.get_stats();
        assert!(stats.eviction_count > 0);
        assert!(stats.total_size_bytes <= stats.max_size_bytes);
    }

    #[tokio::test]
    async fn test_cache_lru_behavior() {
        let cache = CacheManager::new(120).unwrap();
        
        // Add entries
        cache.put("key1".to_string(), vec![0u8; 20]).await.unwrap();
        cache.put("key2".to_string(), vec![0u8; 20]).await.unwrap();
        cache.put("key3".to_string(), vec![0u8; 20]).await.unwrap();
        
        // Access key1 to make it recently used
        cache.get("key1").await;
        
        // Add more entries to trigger eviction
        cache.put("key4".to_string(), vec![0u8; 40]).await.unwrap();
        
        // key1 should still exist (recently accessed)
        assert!(cache.contains_key("key1"));
        
        // Some entries should be evicted
        let remaining_keys = cache.get_keys();
        assert!(remaining_keys.len() < 4);
        assert!(remaining_keys.contains(&"key1".to_string()));
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let cache = CacheManager::new(1024).unwrap();
        
        // Add some data and access it
        cache.put("key1".to_string(), b"data1".to_vec()).await.unwrap();
        cache.get("key1").await; // Hit
        cache.get("nonexistent").await; // Miss
        
        let stats = cache.get_stats();
        assert_eq!(stats.hit_count, 1);
        assert_eq!(stats.miss_count, 1);
        assert_eq!(stats.hit_ratio, 0.5);
        assert_eq!(stats.total_entries, 1);
        assert!(stats.total_size_bytes > 0);
        assert!(stats.utilization_percent < 100.0);
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let cache = CacheManager::new(1024).unwrap();
        
        // Add some data
        for i in 0..5 {
            cache.put(format!("key{}", i), vec![0u8; 10]).await.unwrap();
        }
        
        assert_eq!(cache.get_keys().len(), 5);
        
        // Clear cache
        cache.clear();
        
        assert_eq!(cache.get_keys().len(), 0);
        let stats = cache.get_stats();
        assert_eq!(stats.total_size_bytes, 0);
    }

    #[tokio::test]
    async fn test_manual_eviction() {
        let cache = CacheManager::new(1000).unwrap();
        
        // Add entries
        for i in 0..10 {
            cache.put(format!("key{}", i), vec![0u8; 50]).await.unwrap();
        }
        
        let initial_count = cache.get_keys().len();
        
        // Manually evict to target size
        let evicted = cache.evict_entries(200).await;
        
        assert!(evicted > 0);
        assert!(cache.get_keys().len() < initial_count);
        assert!(cache.get_stats().total_size_bytes <= 200);
    }

    #[tokio::test]
    async fn test_keys_by_access_time() {
        let cache = CacheManager::new(1024).unwrap();
        
        // Add entries with some delay to ensure different access times
        cache.put("old".to_string(), vec![1]).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        cache.put("new".to_string(), vec![2]).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        // Access old to update its access time
        cache.get("old").await;
        
        let keys = cache.get_keys_by_access_time();
        // Most recently accessed should be first
        assert_eq!(keys[0], "old");
    }

    #[test]
    fn test_cache_config_validation() {
        // Test invalid config
        assert!(CacheConfig::new(0).is_err());
        
        // Test valid config
        let config = CacheConfig::new(1024).unwrap();
        assert_eq!(config.max_size_bytes, 1024);
    }

    #[test]
    fn test_cache_stats_calculations() {
        assert_eq!(CacheStats::calculate_hit_ratio(0, 0), 0.0);
        assert_eq!(CacheStats::calculate_hit_ratio(5, 5), 0.5);
        assert_eq!(CacheStats::calculate_hit_ratio(10, 0), 1.0);
        
        assert_eq!(CacheStats::calculate_utilization(0, 100), 0.0);
        assert_eq!(CacheStats::calculate_utilization(50, 100), 50.0);
        assert_eq!(CacheStats::calculate_utilization(100, 100), 100.0);
    }
}