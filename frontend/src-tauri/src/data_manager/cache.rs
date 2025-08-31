use anyhow::{Result, anyhow};
use std::collections::{HashMap, BTreeMap};
use std::time::{Instant, Duration};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use dashmap::DashMap;

/// Cache entry with metadata for efficient management
#[derive(Debug, Clone)]
struct CacheEntry {
    data: Vec<u8>,
    created_at: Instant,
    last_accessed: Instant,
    access_count: u64,
    size: usize,
}

impl CacheEntry {
    fn new(data: Vec<u8>) -> Self {
        let now = Instant::now();
        let size = data.len();
        Self {
            data,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            size,
        }
    }

    fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
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
}

/// Efficient LRU cache with O(1) operations using DashMap and dual HashMap approach
/// Fixes the O(N) lookup issue in access_times BTreeMap
pub struct CacheManager {
    /// Main cache storage - DashMap provides concurrent access with high performance
    entries: DashMap<String, CacheEntry>,
    
    /// O(1) lookup for access times using HashMap instead of BTreeMap linear scan
    access_times: Arc<RwLock<HashMap<String, Instant>>>,
    
    /// Age-ordered entries for efficient LRU eviction - BTreeMap<Instant, Vec<String>>
    /// Groups entries by access time to handle duplicate timestamps
    age_index: Arc<RwLock<BTreeMap<Instant, Vec<String>>>>,
    
    /// Configuration and statistics
    max_size_bytes: usize,
    current_size_bytes: Arc<RwLock<usize>>,
    hit_count: Arc<RwLock<u64>>,
    miss_count: Arc<RwLock<u64>>,
    eviction_count: Arc<RwLock<u64>>,
}

impl CacheManager {
    /// Create a new cache manager with specified maximum size in bytes
    pub fn new(max_size_bytes: usize) -> Result<Self> {
        if max_size_bytes == 0 {
            return Err(anyhow!("Cache size must be greater than 0"));
        }

        Ok(Self {
            entries: DashMap::new(),
            access_times: Arc::new(RwLock::new(HashMap::new())),
            age_index: Arc::new(RwLock::new(BTreeMap::new())),
            max_size_bytes,
            current_size_bytes: Arc::new(RwLock::new(0)),
            hit_count: Arc::new(RwLock::new(0)),
            miss_count: Arc::new(RwLock::new(0)),
            eviction_count: Arc::new(RwLock::new(0)),
        })
    }

    /// Store data in cache with efficient O(1) operations
    pub async fn put(&self, key: String, data: Vec<u8>) {
        let entry_size = data.len() + key.len(); // Approximate memory usage
        let now = Instant::now();
        
        // Check if key already exists and remove old entry
        if let Some((_, old_entry)) = self.entries.remove(&key) {
            self.remove_from_indices(&key, old_entry.last_accessed);
            let mut current_size = self.current_size_bytes.write();
            *current_size = current_size.saturating_sub(old_entry.size + key.len());
        }

        // Create new entry
        let entry = CacheEntry::new(data);
        
        // Update size tracking
        {
            let mut current_size = self.current_size_bytes.write();
            *current_size += entry_size;
        }

        // Add to indices with O(1) operations
        self.add_to_indices(&key, now);
        
        // Insert the entry
        self.entries.insert(key, entry);
        
        // Evict entries if necessary
        self.evict_if_needed().await;
    }

    /// Retrieve data from cache with O(1) lookup
    pub async fn get(&self, key: &str) -> Option<Vec<u8>> {
        if let Some(mut entry) = self.entries.get_mut(key) {
            // Update access tracking
            let old_access_time = entry.last_accessed;
            entry.touch();
            let new_access_time = entry.last_accessed;
            
            // Update indices efficiently
            self.update_access_indices(key, old_access_time, new_access_time);
            
            // Update hit count
            {
                let mut hits = self.hit_count.write();
                *hits += 1;
            }
            
            Some(entry.data.clone())
        } else {
            // Update miss count
            {
                let mut misses = self.miss_count.write();
                *misses += 1;
            }
            None
        }
    }

    /// Check if key exists in cache (O(1) operation)
    pub async fn contains_key(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Remove entry from cache with O(1) cleanup
    pub async fn remove(&self, key: &str) -> Option<Vec<u8>> {
        if let Some((_, entry)) = self.entries.remove(key) {
            // Clean up indices
            self.remove_from_indices(key, entry.last_accessed);
            
            // Update size tracking
            {
                let mut current_size = self.current_size_bytes.write();
                *current_size = current_size.saturating_sub(entry.size + key.len());
            }
            
            Some(entry.data)
        } else {
            None
        }
    }

    /// Clear all entries from cache
    pub async fn clear(&self) {
        self.entries.clear();
        self.access_times.write().clear();
        self.age_index.write().clear();
        *self.current_size_bytes.write() = 0;
    }

    /// Get current cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let total_entries = self.entries.len();
        let total_size_bytes = *self.current_size_bytes.read();
        let hit_count = *self.hit_count.read();
        let miss_count = *self.miss_count.read();
        let eviction_count = *self.eviction_count.read();
        
        let hit_ratio = if hit_count + miss_count > 0 {
            hit_count as f64 / (hit_count + miss_count) as f64
        } else {
            0.0
        };
        
        let average_entry_size = if total_entries > 0 {
            total_size_bytes / total_entries
        } else {
            0
        };
        
        // Find oldest entry age
        let oldest_entry_age_seconds = {
            let age_index = self.age_index.read();
            age_index.keys().next().map(|oldest_time| {
                oldest_time.elapsed().as_secs()
            })
        };
        
        CacheStats {
            total_entries,
            total_size_bytes,
            max_size_bytes: self.max_size_bytes,
            hit_count,
            miss_count,
            eviction_count,
            hit_ratio,
            average_entry_size,
            oldest_entry_age_seconds,
        }
    }

    /// Get all keys in cache (sorted by access time)
    pub async fn get_keys(&self) -> Vec<String> {
        let mut keys = Vec::new();
        let age_index = self.age_index.read();
        
        // Collect keys in LRU order (oldest first)
        for key_list in age_index.values() {
            keys.extend(key_list.iter().cloned());
        }
        
        keys
    }

    /// Get cache utilization as a percentage
    pub async fn get_utilization(&self) -> f64 {
        let current_size = *self.current_size_bytes.read();
        (current_size as f64 / self.max_size_bytes as f64) * 100.0
    }

    /// Manually trigger cache eviction to free space
    pub async fn evict_entries(&self, target_size_bytes: usize) -> usize {
        let mut evicted_count = 0;
        let mut current_size = *self.current_size_bytes.read();
        
        while current_size > target_size_bytes {
            if let Some(key_to_evict) = self.find_lru_key() {
                if self.evict_entry(&key_to_evict) {
                    evicted_count += 1;
                    current_size = *self.current_size_bytes.read();
                } else {
                    break; // No more entries to evict
                }
            } else {
                break; // No entries found
            }
        }
        
        evicted_count
    }

    /// Add entry to tracking indices with O(1) operations
    fn add_to_indices(&self, key: &str, access_time: Instant) {
        // Update access_times HashMap for O(1) lookups
        self.access_times.write().insert(key.to_string(), access_time);
        
        // Update age_index for efficient LRU eviction
        let mut age_index = self.age_index.write();
        age_index.entry(access_time)
            .or_insert_with(Vec::new)
            .push(key.to_string());
    }

    /// Remove entry from tracking indices with O(1) operations
    fn remove_from_indices(&self, key: &str, access_time: Instant) {
        // Remove from access_times HashMap - O(1)
        self.access_times.write().remove(key);
        
        // Remove from age_index - O(log N) for BTreeMap access, but efficient cleanup
        let mut age_index = self.age_index.write();
        if let Some(key_list) = age_index.get_mut(&access_time) {
            key_list.retain(|k| k != key);
            
            // Clean up empty time buckets
            if key_list.is_empty() {
                age_index.remove(&access_time);
            }
        }
    }

    /// Update access indices when entry is accessed - efficient O(1) operations
    fn update_access_indices(&self, key: &str, old_time: Instant, new_time: Instant) {
        // Remove from old time bucket
        self.remove_from_indices(key, old_time);
        
        // Add to new time bucket
        self.add_to_indices(key, new_time);
    }

    /// Find the least recently used key for eviction - O(1) amortized
    fn find_lru_key(&self) -> Option<String> {
        let age_index = self.age_index.read();
        age_index.iter().next().and_then(|(_, keys)| keys.first().cloned())
    }

    /// Evict a specific entry and return success status
    fn evict_entry(&self, key: &str) -> bool {
        if let Some((_, entry)) = self.entries.remove(key) {
            self.remove_from_indices(key, entry.last_accessed);
            
            // Update size tracking
            {
                let mut current_size = self.current_size_bytes.write();
                *current_size = current_size.saturating_sub(entry.size + key.len());
            }
            
            // Update eviction count
            {
                let mut evictions = self.eviction_count.write();
                *evictions += 1;
            }
            
            log::debug!("Evicted cache entry: {}", key);
            true
        } else {
            false
        }
    }

    /// Evict entries if cache size exceeds maximum
    async fn evict_if_needed(&self) {
        let current_size = *self.current_size_bytes.read();
        
        if current_size > self.max_size_bytes {
            // Evict 20% of max size to provide some buffer
            let target_size = (self.max_size_bytes as f64 * 0.8) as usize;
            let evicted = self.evict_entries(target_size).await;
            
            if evicted > 0 {
                log::debug!("Evicted {} entries to free space", evicted);
            }
        }
    }
}

impl Drop for CacheManager {
    fn drop(&mut self) {
        // Clean up resources
        let stats = futures::executor::block_on(self.get_stats());
        log::info!("Cache manager dropped. Final stats: hit_ratio={:.2}%, entries={}, size={}MB", 
                  stats.hit_ratio * 100.0, stats.total_entries, stats.total_size_bytes / 1024 / 1024);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_cache_basic_operations() {
        let cache = CacheManager::new(1024).unwrap();
        
        // Test put and get
        cache.put("key1".to_string(), b"data1".to_vec()).await;
        let retrieved = cache.get("key1").await;
        assert_eq!(retrieved, Some(b"data1".to_vec()));
        
        // Test contains
        assert!(cache.contains_key("key1").await);
        assert!(!cache.contains_key("nonexistent").await);
        
        // Test remove
        let removed = cache.remove("key1").await;
        assert_eq!(removed, Some(b"data1".to_vec()));
        assert!(!cache.contains_key("key1").await);
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let cache = CacheManager::new(100).unwrap(); // Small cache for testing
        
        // Fill cache beyond capacity
        for i in 0..10 {
            let key = format!("key{}", i);
            let data = vec![0u8; 20]; // 20 bytes per entry
            cache.put(key, data).await;
        }
        
        let stats = cache.get_stats().await;
        assert!(stats.eviction_count > 0);
        assert!(stats.total_size_bytes <= cache.max_size_bytes);
    }

    #[tokio::test]
    async fn test_cache_lru_behavior() {
        let cache = CacheManager::new(100).unwrap();
        
        // Add entries
        cache.put("key1".to_string(), vec![0u8; 20]).await;
        cache.put("key2".to_string(), vec![0u8; 20]).await;
        cache.put("key3".to_string(), vec![0u8; 20]).await;
        
        // Access key1 to make it recently used
        cache.get("key1").await;
        
        // Add more entries to trigger eviction
        cache.put("key4".to_string(), vec![0u8; 50]).await;
        
        // key1 should still exist (recently accessed)
        assert!(cache.contains_key("key1").await);
        
        // key2 or key3 should be evicted (least recently used)
        let key2_exists = cache.contains_key("key2").await;
        let key3_exists = cache.contains_key("key3").await;
        assert!(!(key2_exists && key3_exists)); // At least one should be evicted
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let cache = CacheManager::new(1024).unwrap();
        
        // Add some data and access it
        cache.put("key1".to_string(), b"data1".to_vec()).await;
        cache.get("key1").await; // Hit
        cache.get("nonexistent").await; // Miss
        
        let stats = cache.get_stats().await;
        assert_eq!(stats.hit_count, 1);
        assert_eq!(stats.miss_count, 1);
        assert_eq!(stats.hit_ratio, 0.5);
        assert_eq!(stats.total_entries, 1);
        assert!(stats.total_size_bytes > 0);
    }
}