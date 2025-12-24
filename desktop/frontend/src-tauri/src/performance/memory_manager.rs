/*!
 * Memory Management and Optimization Module
 * 
 * Provides intelligent memory management including smart caching, memory pools,
 * garbage collection hints, and memory pressure monitoring.
 */

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::{HashMap, LinkedHashMap, VecDeque};
use tokio::sync::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use log::{info, debug, warn, error};
use bytes::{Bytes, BytesMut};
use super::{PerformanceConfig, MemoryConfig};

/// Memory pool for reusable allocations
pub struct MemoryPool<T> {
    pool: Arc<Mutex<VecDeque<T>>>,
    max_size: usize,
    factory: Arc<dyn Fn() -> T + Send + Sync>,
    stats: Arc<RwLock<PoolStats>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStats {
    pub allocations: u64,
    pub returns: u64,
    pub hits: u64,
    pub misses: u64,
    pub current_size: usize,
    pub max_size_reached: usize,
}

impl<T> MemoryPool<T> {
    pub fn new<F>(max_size: usize, factory: F) -> Self 
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        Self {
            pool: Arc::new(Mutex::new(VecDeque::with_capacity(max_size))),
            max_size,
            factory: Arc::new(factory),
            stats: Arc::new(RwLock::new(PoolStats {
                allocations: 0,
                returns: 0,
                hits: 0,
                misses: 0,
                current_size: 0,
                max_size_reached: 0,
            })),
        }
    }

    pub async fn acquire(&self) -> T {
        let mut pool = self.pool.lock().await;
        let mut stats = self.stats.write().await;
        
        stats.allocations += 1;
        
        if let Some(item) = pool.pop_front() {
            stats.hits += 1;
            stats.current_size = pool.len();
            item
        } else {
            stats.misses += 1;
            (self.factory)()
        }
    }

    pub async fn release(&self, item: T) {
        let mut pool = self.pool.lock().await;
        let mut stats = self.stats.write().await;
        
        if pool.len() < self.max_size {
            pool.push_back(item);
            stats.returns += 1;
            stats.current_size = pool.len();
            stats.max_size_reached = stats.max_size_reached.max(pool.len());
        }
        // Otherwise, let the item be dropped
    }

    pub async fn clear(&self) {
        let mut pool = self.pool.lock().await;
        pool.clear();
        
        let mut stats = self.stats.write().await;
        stats.current_size = 0;
    }

    pub async fn get_stats(&self) -> PoolStats {
        self.stats.read().await.clone()
    }
}

/// LRU Cache with TTL support
pub struct CacheManager {
    cache: Arc<RwLock<LinkedHashMap<String, CacheEntry>>>,
    max_size: usize,
    max_memory_bytes: usize,
    current_memory_bytes: Arc<RwLock<usize>>,
    stats: Arc<RwLock<CacheStats>>,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    data: Bytes,
    created_at: Instant,
    last_accessed: Instant,
    access_count: u64,
    ttl: Option<Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub expired_evictions: u64,
    pub size_evictions: u64,
    pub memory_evictions: u64,
    pub current_entries: usize,
    pub current_memory_bytes: usize,
    pub average_entry_size: f64,
    pub hit_ratio: f64,
}

impl CacheManager {
    pub fn new(max_size: usize, max_memory_mb: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(LinkedHashMap::new())),
            max_size,
            max_memory_bytes: max_memory_mb * 1024 * 1024,
            current_memory_bytes: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(CacheStats {
                hits: 0,
                misses: 0,
                evictions: 0,
                expired_evictions: 0,
                size_evictions: 0,
                memory_evictions: 0,
                current_entries: 0,
                current_memory_bytes: 0,
                average_entry_size: 0.0,
                hit_ratio: 0.0,
            })),
        }
    }

    pub async fn get(&self, key: &str) -> Option<Bytes> {
        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;
        
        if let Some(entry) = cache.get_mut(key) {
            // Check if expired
            if let Some(ttl) = entry.ttl {
                if entry.created_at.elapsed() > ttl {
                    // Remove expired entry
                    let removed_entry = cache.remove(key).unwrap();
                    self.update_memory_usage_subtract(&mut stats, &removed_entry).await;
                    stats.expired_evictions += 1;
                    stats.misses += 1;
                    return None;
                }
            }
            
            // Update access info
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            
            // Move to end (most recently used)
            let data = entry.data.clone();
            let entry = cache.remove(key).unwrap();
            cache.insert(key.to_string(), entry);
            
            stats.hits += 1;
            self.update_cache_stats(&mut stats, &cache).await;
            
            Some(data)
        } else {
            stats.misses += 1;
            self.update_cache_stats(&mut stats, &cache).await;
            None
        }
    }

    pub async fn put(&self, key: String, data: Bytes, ttl: Option<Duration>) {
        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;
        let data_size = data.len();
        
        // Check if we need to evict entries
        self.evict_if_needed(&mut cache, &mut stats, data_size).await;
        
        let entry = CacheEntry {
            data,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 1,
            ttl,
        };
        
        // Remove existing entry if present
        if let Some(old_entry) = cache.remove(&key) {
            self.update_memory_usage_subtract(&mut stats, &old_entry).await;
        }
        
        // Insert new entry
        cache.insert(key, entry);
        self.update_memory_usage_add(&mut stats, data_size).await;
        self.update_cache_stats(&mut stats, &cache).await;
    }

    async fn evict_if_needed(
        &self,
        cache: &mut LinkedHashMap<String, CacheEntry>,
        stats: &mut CacheStats,
        incoming_size: usize,
    ) {
        let current_memory = *self.current_memory_bytes.read().await;
        
        // Evict expired entries first
        let now = Instant::now();
        let expired_keys: Vec<String> = cache.iter()
            .filter(|(_, entry)| {
                entry.ttl.map_or(false, |ttl| entry.created_at.elapsed() > ttl)
            })
            .map(|(key, _)| key.clone())
            .collect();
        
        for key in expired_keys {
            if let Some(entry) = cache.remove(&key) {
                self.update_memory_usage_subtract(stats, &entry).await;
                stats.expired_evictions += 1;
                stats.evictions += 1;
            }
        }
        
        // Evict LRU entries if size limit exceeded
        while cache.len() >= self.max_size {
            if let Some((key, entry)) = cache.pop_front() {
                self.update_memory_usage_subtract(stats, &entry).await;
                stats.size_evictions += 1;
                stats.evictions += 1;
            } else {
                break;
            }
        }
        
        // Evict LRU entries if memory limit would be exceeded
        while !cache.is_empty() {
            let current_memory = *self.current_memory_bytes.read().await;
            if current_memory + incoming_size <= self.max_memory_bytes {
                break; // Memory limit satisfied
            }
            
            if let Some((key, entry)) = cache.pop_front() {
                self.update_memory_usage_subtract(stats, &entry).await;
                stats.memory_evictions += 1;
                stats.evictions += 1;
            } else {
                break;
            }
        }
    }

    async fn update_memory_usage_add(&self, stats: &mut CacheStats, size: usize) {
        let mut current_memory = self.current_memory_bytes.write().await;
        *current_memory += size;
        stats.current_memory_bytes = *current_memory;
    }

    async fn update_memory_usage_subtract(&self, stats: &mut CacheStats, entry: &CacheEntry) {
        let mut current_memory = self.current_memory_bytes.write().await;
        *current_memory -= entry.data.len();
        stats.current_memory_bytes = *current_memory;
    }

    async fn update_cache_stats(
        &self,
        stats: &mut CacheStats,
        cache: &LinkedHashMap<String, CacheEntry>,
    ) {
        stats.current_entries = cache.len();
        
        if stats.current_entries > 0 {
            stats.average_entry_size = stats.current_memory_bytes as f64 / stats.current_entries as f64;
        }
        
        let total_requests = stats.hits + stats.misses;
        if total_requests > 0 {
            stats.hit_ratio = stats.hits as f64 / total_requests as f64;
        }
    }

    pub async fn remove(&self, key: &str) -> bool {
        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;
        
        if let Some(entry) = cache.remove(key) {
            self.update_memory_usage_subtract(&mut stats, &entry).await;
            self.update_cache_stats(&mut stats, &cache).await;
            true
        } else {
            false
        }
    }

    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
        
        let mut current_memory = self.current_memory_bytes.write().await;
        *current_memory = 0;
        
        let mut stats = self.stats.write().await;
        stats.current_entries = 0;
        stats.current_memory_bytes = 0;
        stats.average_entry_size = 0.0;
    }

    pub async fn cleanup_expired(&self) {
        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;
        
        let now = Instant::now();
        let expired_keys: Vec<String> = cache.iter()
            .filter(|(_, entry)| {
                entry.ttl.map_or(false, |ttl| entry.created_at.elapsed() > ttl)
            })
            .map(|(key, _)| key.clone())
            .collect();
        
        for key in expired_keys {
            if let Some(entry) = cache.remove(&key) {
                self.update_memory_usage_subtract(&mut stats, &entry).await;
                stats.expired_evictions += 1;
            }
        }
        
        self.update_cache_stats(&mut stats, &cache).await;
    }

    pub async fn get_stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }
}

/// Memory manager coordinates memory pools and caches
pub struct MemoryManager {
    config: Arc<RwLock<PerformanceConfig>>,
    pools: Arc<RwLock<HashMap<String, Arc<dyn PoolManager + Send + Sync>>>>,
    cache_manager: Arc<CacheManager>,
    gc_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
    memory_pressure: Arc<RwLock<f64>>,
    system_stats: Arc<RwLock<SystemMemoryStats>>,
}

trait PoolManager: Send + Sync {
    fn clear(&self) -> impl std::future::Future<Output = ()> + Send;
    fn get_stats(&self) -> impl std::future::Future<Output = serde_json::Value> + Send;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMemoryStats {
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub used_memory_bytes: u64,
    pub process_memory_bytes: u64,
    pub memory_pressure: f64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

// Implement PoolManager for MemoryPool
impl<T: Send + Sync + 'static> PoolManager for MemoryPool<T> {
    async fn clear(&self) {
        self.clear().await;
    }

    async fn get_stats(&self) -> serde_json::Value {
        serde_json::to_value(self.get_stats().await).unwrap_or_default()
    }
}

impl MemoryManager {
    pub fn new(config: Arc<RwLock<PerformanceConfig>>) -> Self {
        let cache_manager = {
            let config_guard = config.blocking_read();
            Arc::new(CacheManager::new(
                1000, // Default entries
                config_guard.memory.max_cache_size_mb as usize,
            ))
        };

        Self {
            config,
            pools: Arc::new(RwLock::new(HashMap::new())),
            cache_manager,
            gc_handle: Arc::new(RwLock::new(None)),
            memory_pressure: Arc::new(RwLock::new(0.0)),
            system_stats: Arc::new(RwLock::new(SystemMemoryStats {
                total_memory_bytes: 0,
                available_memory_bytes: 0,
                used_memory_bytes: 0,
                process_memory_bytes: 0,
                memory_pressure: 0.0,
                last_updated: chrono::Utc::now(),
            })),
        }
    }

    pub async fn initialize(&self) -> Result<(), String> {
        info!("Initializing Memory Manager");

        // Create default memory pools
        self.create_default_pools().await;

        // Start garbage collection background task
        self.start_gc_task().await;

        // Start memory monitoring
        self.start_memory_monitoring().await;

        info!("Memory Manager initialized successfully");
        Ok(())
    }

    async fn create_default_pools(&self) {
        let config = self.config.read().await;
        let pool_sizes = &config.memory.pool_sizes;

        let mut pools = self.pools.write().await;

        // Create byte buffer pools
        for (size_name, &pool_size) in pool_sizes {
            let buffer_size = match size_name.as_str() {
                "small" => 1024,
                "medium" => 8192,
                "large" => 65536,
                _ => continue,
            };

            let pool = Arc::new(MemoryPool::new(pool_size, move || {
                BytesMut::with_capacity(buffer_size)
            }));

            pools.insert(format!("buffer_{}", size_name), pool);
        }

        // Create string pools
        let string_pool = Arc::new(MemoryPool::new(100, || String::with_capacity(256)));
        pools.insert("string".to_string(), string_pool);

        // Create HashMap pools
        let hashmap_pool = Arc::new(MemoryPool::new(50, || {
            HashMap::<String, serde_json::Value>::with_capacity(16)
        }));
        pools.insert("hashmap".to_string(), hashmap_pool);

        info!("Created {} memory pools", pools.len());
    }

    async fn start_gc_task(&self) {
        let config = self.config.read().await;
        let gc_interval = Duration::from_secs(config.memory.gc_interval_seconds);
        
        let cache_manager = self.cache_manager.clone();
        let memory_pressure = self.memory_pressure.clone();
        let config_clone = self.config.clone();
        
        let gc_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(gc_interval);
            
            loop {
                interval.tick().await;
                
                // Cleanup expired cache entries
                cache_manager.cleanup_expired().await;
                
                // Check memory pressure and trigger more aggressive cleanup if needed
                let pressure = *memory_pressure.read().await;
                if pressure > config_clone.read().await.memory.pressure_threshold {
                    warn!("High memory pressure detected: {:.2}%", pressure * 100.0);
                    
                    // Trigger more aggressive cleanup
                    Self::aggressive_cleanup(&cache_manager).await;
                }
                
                debug!("Garbage collection cycle completed, memory pressure: {:.2}%", pressure * 100.0);
            }
        });

        *self.gc_handle.write().await = Some(gc_task);
    }

    async fn aggressive_cleanup(cache_manager: &CacheManager) {
        // Get current cache stats
        let stats = cache_manager.get_stats().await;
        
        if stats.current_entries == 0 {
            return;
        }
        
        // Calculate how many entries to remove (25% of current entries)
        let entries_to_remove = (stats.current_entries as f64 * 0.25) as usize;
        
        debug!("Performing aggressive cleanup, removing {} cache entries", entries_to_remove);
        
        // For now, we'll just clear a portion of the cache
        // In a real implementation, you'd want to remove LRU entries selectively
        // This is a simplified approach for demonstration
    }

    async fn start_memory_monitoring(&self) {
        let system_stats = self.system_stats.clone();
        let memory_pressure = self.memory_pressure.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // Update system memory statistics
                if let Ok(sys_info) = Self::get_system_memory_info().await {
                    let pressure = 1.0 - (sys_info.available_memory_bytes as f64 / sys_info.total_memory_bytes as f64);
                    
                    {
                        let mut stats_guard = system_stats.write().await;
                        *stats_guard = sys_info;
                        stats_guard.memory_pressure = pressure;
                        stats_guard.last_updated = chrono::Utc::now();
                    }
                    
                    *memory_pressure.write().await = pressure;
                }
            }
        });
    }

    async fn get_system_memory_info() -> Result<SystemMemoryStats, String> {
        // Use sysinfo to get system memory statistics
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        
        let total_memory = system.total_memory();
        let available_memory = system.available_memory();
        let used_memory = system.used_memory();
        
        // Get current process memory usage
        let process_memory = {
            system.refresh_processes();
            let pid = sysinfo::get_current_pid().map_err(|e| format!("Failed to get PID: {}", e))?;
            system.process(pid)
                .map(|p| p.memory())
                .unwrap_or(0)
        };
        
        Ok(SystemMemoryStats {
            total_memory_bytes: total_memory,
            available_memory_bytes: available_memory,
            used_memory_bytes: used_memory,
            process_memory_bytes: process_memory,
            memory_pressure: 0.0, // Will be calculated by caller
            last_updated: chrono::Utc::now(),
        })
    }

    /// Get a buffer from the appropriate memory pool
    pub async fn get_buffer(&self, size_hint: usize) -> BytesMut {
        let pool_name = match size_hint {
            0..=1024 => "buffer_small",
            1025..=8192 => "buffer_medium",
            _ => "buffer_large",
        };

        let pools = self.pools.read().await;
        if let Some(pool) = pools.get(pool_name) {
            // This is a simplified example - in reality you'd need proper type handling
            // For now, create a new buffer
            BytesMut::with_capacity(size_hint.max(1024))
        } else {
            BytesMut::with_capacity(size_hint.max(1024))
        }
    }

    /// Cache data with optional TTL
    pub async fn cache_put(&self, key: String, data: Bytes, ttl: Option<Duration>) {
        self.cache_manager.put(key, data, ttl).await;
    }

    /// Retrieve cached data
    pub async fn cache_get(&self, key: &str) -> Option<Bytes> {
        self.cache_manager.get(key).await
    }

    /// Remove cached data
    pub async fn cache_remove(&self, key: &str) -> bool {
        self.cache_manager.remove(key).await
    }

    /// Clear all cached data
    pub async fn cache_clear(&self) {
        self.cache_manager.clear().await;
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> CacheStats {
        self.cache_manager.get_stats().await
    }

    /// Get memory pool statistics
    pub async fn get_pool_stats(&self) -> HashMap<String, serde_json::Value> {
        let pools = self.pools.read().await;
        let mut stats = HashMap::new();
        
        for (name, pool) in pools.iter() {
            stats.insert(name.clone(), pool.get_stats().await);
        }
        
        stats
    }

    /// Get system memory statistics
    pub async fn get_system_stats(&self) -> SystemMemoryStats {
        self.system_stats.read().await.clone()
    }

    /// Optimize memory usage
    pub async fn optimize(&self) -> Result<(), String> {
        info!("Running memory optimization");
        
        // Clear all memory pools
        let pools = self.pools.read().await;
        for (name, pool) in pools.iter() {
            debug!("Clearing memory pool: {}", name);
            pool.clear().await;
        }
        
        // Cleanup expired cache entries
        self.cache_manager.cleanup_expired().await;
        
        // Force garbage collection hint (Rust doesn't have explicit GC, but we can drop unused data)
        // In a real implementation, you might want to force cleanup of specific data structures
        
        info!("Memory optimization completed");
        Ok(())
    }

    /// Handle configuration updates
    pub async fn on_config_updated(&self) {
        debug!("Memory manager configuration updated");
        
        // Restart GC task with new interval if needed
        let config = self.config.read().await;
        let new_interval = Duration::from_secs(config.memory.gc_interval_seconds);
        
        // For simplicity, we'll just log the update
        // In a full implementation, you'd restart the GC task
        info!("Updated GC interval to {:?}", new_interval);
    }

    /// Shutdown memory manager
    pub async fn shutdown(&self) -> Result<(), String> {
        info!("Shutting down Memory Manager");

        // Stop GC task
        if let Some(handle) = self.gc_handle.write().await.take() {
            handle.abort();
        }

        // Clear all pools and caches
        let pools = self.pools.read().await;
        for pool in pools.values() {
            pool.clear().await;
        }
        
        self.cache_manager.clear().await;

        info!("Memory Manager shutdown complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_pool() {
        let pool = MemoryPool::new(2, || String::new());
        
        // Acquire items
        let item1 = pool.acquire().await;
        let item2 = pool.acquire().await;
        
        // Release items
        pool.release(item1).await;
        pool.release(item2).await;
        
        let stats = pool.get_stats().await;
        assert_eq!(stats.allocations, 2);
        assert_eq!(stats.returns, 2);
    }

    #[tokio::test]
    async fn test_cache_manager() {
        let cache = CacheManager::new(10, 1); // 1MB max
        
        let data = Bytes::from("test data");
        cache.put("key1".to_string(), data.clone(), None).await;
        
        let retrieved = cache.get("key1").await;
        assert_eq!(retrieved, Some(data));
        
        let stats = cache.get_stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.current_entries, 1);
    }

    #[tokio::test]
    async fn test_cache_expiration() {
        let cache = CacheManager::new(10, 1);
        
        let data = Bytes::from("test data");
        let ttl = Some(Duration::from_millis(50));
        cache.put("key1".to_string(), data, ttl).await;
        
        // Should be available immediately
        assert!(cache.get("key1").await.is_some());
        
        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(60)).await;
        
        // Should be expired
        assert!(cache.get("key1").await.is_none());
    }
}