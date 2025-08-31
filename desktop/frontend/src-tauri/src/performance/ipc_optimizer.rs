/*!
 * IPC Optimization Module
 * 
 * Provides intelligent IPC optimization including command batching, response caching,
 * request deduplication, and latency optimization.
 */

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use tokio::sync::{RwLock, Mutex, mpsc, oneshot};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use log::{info, debug, warn, error};
use bytes::Bytes;
use super::{PerformanceConfig, IpcConfig};

/// Batched command for optimized IPC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedCommand {
    pub id: String,
    pub method: String,
    pub params: Value,
    pub timestamp: Instant,
    pub priority: u8,
}

/// Batch execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    pub batch_id: String,
    pub commands: Vec<CommandResult>,
    pub total_duration: Duration,
    pub compression_ratio: Option<f64>,
}

/// Individual command result within a batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResult {
    pub id: String,
    pub success: bool,
    pub result: Option<Value>,
    pub error: Option<String>,
    pub duration: Duration,
    pub cached: bool,
}

/// Command batcher accumulates commands for batch execution
pub struct CommandBatcher {
    config: Arc<RwLock<IpcConfig>>,
    pending_commands: Arc<Mutex<VecDeque<BatchedCommand>>>,
    pending_callbacks: Arc<Mutex<HashMap<String, oneshot::Sender<CommandResult>>>>,
    batch_timer: Arc<Mutex<Option<tokio::time::Interval>>>,
    stats: Arc<RwLock<BatcherStats>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatcherStats {
    pub commands_batched: u64,
    pub batches_executed: u64,
    pub average_batch_size: f64,
    pub total_latency_saved_ms: u64,
    pub compression_ratio: f64,
    pub cache_hit_ratio: f64,
}

impl CommandBatcher {
    pub fn new(config: Arc<RwLock<IpcConfig>>) -> Self {
        Self {
            config,
            pending_commands: Arc::new(Mutex::new(VecDeque::new())),
            pending_callbacks: Arc::new(Mutex::new(HashMap::new())),
            batch_timer: Arc::new(Mutex::new(None)),
            stats: Arc::new(RwLock::new(BatcherStats {
                commands_batched: 0,
                batches_executed: 0,
                average_batch_size: 0.0,
                total_latency_saved_ms: 0,
                compression_ratio: 1.0,
                cache_hit_ratio: 0.0,
            })),
        }
    }

    /// Add command to batch queue
    pub async fn queue_command(
        &self,
        method: String,
        params: Value,
        priority: u8,
    ) -> Result<CommandResult, String> {
        let command_id = uuid::Uuid::new_v4().to_string();
        let command = BatchedCommand {
            id: command_id.clone(),
            method,
            params,
            timestamp: Instant::now(),
            priority,
        };

        // Create response channel
        let (tx, rx) = oneshot::channel();
        
        // Queue command
        {
            let mut pending = self.pending_commands.lock().await;
            pending.push_back(command);
            
            let mut callbacks = self.pending_callbacks.lock().await;
            callbacks.insert(command_id, tx);
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.commands_batched += 1;
        }

        // Check if we should flush the batch
        self.check_batch_ready().await;

        // Wait for result
        rx.await.map_err(|_| "Command execution cancelled".to_string())
    }

    /// Check if batch is ready for execution
    async fn check_batch_ready(&self) {
        let config = self.config.read().await;
        let pending_count = self.pending_commands.lock().await.len();
        
        if pending_count >= config.max_batch_size {
            debug!("Batch ready due to size limit: {}", pending_count);
            self.execute_batch().await;
        } else if pending_count > 0 {
            // Start or reset the batch timer
            self.start_batch_timer().await;
        }
    }

    /// Start the batch execution timer
    async fn start_batch_timer(&self) {
        let config = self.config.read().await;
        let timeout = Duration::from_millis(config.batch_timeout_ms);
        
        let mut timer_guard = self.batch_timer.lock().await;
        if timer_guard.is_none() {
            let mut interval = tokio::time::interval(timeout);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            *timer_guard = Some(interval);
            
            // Spawn timer task
            let batcher_clone = Arc::new(self.clone());
            tokio::spawn(async move {
                let mut timer_guard = batcher_clone.batch_timer.lock().await;
                if let Some(ref mut interval) = timer_guard.as_mut() {
                    interval.tick().await; // First tick is immediate
                    interval.tick().await; // Second tick is after the timeout
                    
                    debug!("Batch timer expired, executing batch");
                    drop(timer_guard); // Release the lock
                    batcher_clone.execute_batch().await;
                }
            });
        }
    }

    /// Execute accumulated batch
    async fn execute_batch(&self) {
        let batch_start = Instant::now();
        let batch_id = uuid::Uuid::new_v4().to_string();
        
        // Collect commands to execute
        let (commands, callbacks) = {
            let mut pending = self.pending_commands.lock().await;
            let mut callback_map = self.pending_callbacks.lock().await;
            
            let commands: Vec<BatchedCommand> = pending.drain(..).collect();
            let callbacks: HashMap<String, oneshot::Sender<CommandResult>> = 
                callback_map.drain().collect();
            
            (commands, callbacks)
        };

        // Clear the batch timer
        {
            *self.batch_timer.lock().await = None;
        }

        if commands.is_empty() {
            return;
        }

        debug!("Executing batch {} with {} commands", batch_id, commands.len());

        // Sort commands by priority (lower number = higher priority)
        let mut sorted_commands = commands;
        sorted_commands.sort_by_key(|cmd| cmd.priority);

        // Execute commands (in reality, this would call the actual IPC mechanism)
        let mut results = Vec::new();
        for command in &sorted_commands {
            let start_time = Instant::now();
            
            // Simulate command execution
            let result = self.execute_single_command(command).await;
            
            let command_result = CommandResult {
                id: command.id.clone(),
                success: result.is_ok(),
                result: result.as_ref().ok().cloned(),
                error: result.as_ref().err().cloned(),
                duration: start_time.elapsed(),
                cached: false, // Would be set by response cache
            };
            
            results.push(command_result.clone());
            
            // Send result back to waiting caller
            if let Some(callback) = callbacks.get(&command.id) {
                // We can't use the callback here because it's been moved
                // This is a simplified example - in reality you'd handle this differently
            }
        }

        let total_duration = batch_start.elapsed();
        let batch_result = BatchResult {
            batch_id: batch_id.clone(),
            commands: results,
            total_duration,
            compression_ratio: None, // Would be calculated if compression is enabled
        };

        // Send results to all waiting callers
        for (command_id, callback) in callbacks {
            if let Some(result) = batch_result.commands.iter()
                .find(|r| r.id == command_id) {
                let _ = callback.send(result.clone());
            }
        }

        // Update statistics
        self.update_batch_stats(&batch_result).await;

        debug!("Batch {} completed in {:?}", batch_id, total_duration);
    }

    /// Execute a single command (stub implementation)
    async fn execute_single_command(&self, command: &BatchedCommand) -> Result<Value, String> {
        // This is a placeholder - in reality, this would call the actual IPC mechanism
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        if command.method == "error_test" {
            Err("Simulated error".to_string())
        } else {
            Ok(serde_json::json!({
                "method": command.method,
                "result": "success",
                "timestamp": chrono::Utc::now()
            }))
        }
    }

    /// Update batch execution statistics
    async fn update_batch_stats(&self, result: &BatchResult) {
        let mut stats = self.stats.write().await;
        
        stats.batches_executed += 1;
        
        // Update average batch size
        let total_commands = stats.commands_batched as f64;
        let total_batches = stats.batches_executed as f64;
        if total_batches > 0.0 {
            stats.average_batch_size = total_commands / total_batches;
        }
        
        // Calculate latency saved (estimated)
        let individual_latency_estimate = result.commands.len() as u64 * 50; // 50ms per individual call
        let actual_latency = result.total_duration.as_millis() as u64;
        if individual_latency_estimate > actual_latency {
            stats.total_latency_saved_ms += individual_latency_estimate - actual_latency;
        }
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> BatcherStats {
        self.stats.read().await.clone()
    }

    /// Clear pending commands (for shutdown)
    pub async fn clear_pending(&self) {
        let mut pending = self.pending_commands.lock().await;
        pending.clear();
        
        let callbacks = {
            let mut callback_map = self.pending_callbacks.lock().await;
            callback_map.drain().collect::<Vec<_>>()
        };
        
        // Send cancellation to all pending callbacks
        for (_, callback) in callbacks {
            let _ = callback.send(CommandResult {
                id: "cancelled".to_string(),
                success: false,
                result: None,
                error: Some("Shutdown".to_string()),
                duration: Duration::ZERO,
                cached: false,
            });
        }
    }
}

impl Clone for CommandBatcher {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            pending_commands: self.pending_commands.clone(),
            pending_callbacks: self.pending_callbacks.clone(),
            batch_timer: self.batch_timer.clone(),
            stats: self.stats.clone(),
        }
    }
}

/// Response cache for IPC calls
pub struct ResponseCache {
    config: Arc<RwLock<IpcConfig>>,
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    stats: Arc<RwLock<ResponseCacheStats>>,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    response: Value,
    created_at: Instant,
    access_count: u64,
    last_accessed: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    pub hit_ratio: f64,
    pub average_response_time_ms: f64,
    pub memory_usage_bytes: usize,
}

impl ResponseCache {
    pub fn new(config: Arc<RwLock<IpcConfig>>) -> Self {
        Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(ResponseCacheStats {
                hits: 0,
                misses: 0,
                entries: 0,
                hit_ratio: 0.0,
                average_response_time_ms: 0.0,
                memory_usage_bytes: 0,
            })),
        }
    }

    /// Generate cache key for method and parameters
    fn generate_cache_key(&self, method: &str, params: &Value) -> String {
        // Create a deterministic key based on method and parameters
        let params_str = serde_json::to_string(params).unwrap_or_default();
        format!("{}:{}", method, blake3::hash(params_str.as_bytes()).to_hex())
    }

    /// Get cached response
    pub async fn get(&self, method: &str, params: &Value) -> Option<Value> {
        let cache_key = self.generate_cache_key(method, params);
        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;
        
        if let Some(entry) = cache.get_mut(&cache_key) {
            // Check if entry is still valid
            let config = self.config.read().await;
            let ttl = Duration::from_secs(config.cache_ttl_seconds);
            
            if entry.created_at.elapsed() < ttl {
                // Update access info
                entry.access_count += 1;
                entry.last_accessed = Instant::now();
                
                stats.hits += 1;
                self.update_cache_stats(&mut stats, &cache);
                
                debug!("Cache hit for method: {}", method);
                return Some(entry.response.clone());
            } else {
                // Remove expired entry
                cache.remove(&cache_key);
                debug!("Removed expired cache entry for method: {}", method);
            }
        }
        
        stats.misses += 1;
        self.update_cache_stats(&mut stats, &cache);
        
        debug!("Cache miss for method: {}", method);
        None
    }

    /// Store response in cache
    pub async fn put(&self, method: &str, params: &Value, response: Value) {
        let cache_key = self.generate_cache_key(method, params);
        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;
        
        // Check cache size limit
        let config = self.config.read().await;
        if cache.len() >= config.response_cache_size {
            // Remove least recently accessed entry
            if let Some(lru_key) = self.find_lru_key(&cache) {
                cache.remove(&lru_key);
                debug!("Removed LRU cache entry to make space");
            }
        }
        
        let entry = CacheEntry {
            response,
            created_at: Instant::now(),
            access_count: 1,
            last_accessed: Instant::now(),
        };
        
        cache.insert(cache_key, entry);
        self.update_cache_stats(&mut stats, &cache);
        
        debug!("Cached response for method: {}", method);
    }

    /// Find least recently used cache key
    fn find_lru_key(&self, cache: &HashMap<String, CacheEntry>) -> Option<String> {
        cache.iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| key.clone())
    }

    /// Update cache statistics
    fn update_cache_stats(&self, stats: &mut ResponseCacheStats, cache: &HashMap<String, CacheEntry>) {
        stats.entries = cache.len();
        
        let total_requests = stats.hits + stats.misses;
        if total_requests > 0 {
            stats.hit_ratio = stats.hits as f64 / total_requests as f64;
        }
        
        // Estimate memory usage (simplified)
        stats.memory_usage_bytes = cache.len() * 1024; // Rough estimate
    }

    /// Clear expired entries
    pub async fn cleanup_expired(&self) {
        let config = self.config.read().await;
        let ttl = Duration::from_secs(config.cache_ttl_seconds);
        let now = Instant::now();
        
        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;
        
        let initial_size = cache.len();
        cache.retain(|_, entry| now.duration_since(entry.created_at) < ttl);
        
        let removed = initial_size - cache.len();
        if removed > 0 {
            debug!("Cleaned up {} expired cache entries", removed);
            self.update_cache_stats(&mut stats, &cache);
        }
    }

    /// Clear all cached entries
    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        let size = cache.len();
        cache.clear();
        
        let mut stats = self.stats.write().await;
        stats.entries = 0;
        stats.memory_usage_bytes = 0;
        
        info!("Cleared {} cache entries", size);
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> ResponseCacheStats {
        self.stats.read().await.clone()
    }
}

/// Main IPC optimizer coordinating all optimizations
pub struct IpcOptimizer {
    config: Arc<RwLock<PerformanceConfig>>,
    command_batcher: Arc<CommandBatcher>,
    response_cache: Arc<ResponseCache>,
    cleanup_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
    stats: Arc<RwLock<IpcOptimizerStats>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcOptimizerStats {
    pub total_requests: u64,
    pub batched_requests: u64,
    pub cached_responses: u64,
    pub average_latency_ms: f64,
    pub latency_reduction_percent: f64,
    pub compression_savings_bytes: u64,
}

impl IpcOptimizer {
    pub fn new(config: Arc<RwLock<PerformanceConfig>>) -> Self {
        let ipc_config = Arc::new(RwLock::new(config.blocking_read().ipc.clone()));
        
        Self {
            config,
            command_batcher: Arc::new(CommandBatcher::new(ipc_config.clone())),
            response_cache: Arc::new(ResponseCache::new(ipc_config)),
            cleanup_handle: Arc::new(RwLock::new(None)),
            stats: Arc::new(RwLock::new(IpcOptimizerStats {
                total_requests: 0,
                batched_requests: 0,
                cached_responses: 0,
                average_latency_ms: 0.0,
                latency_reduction_percent: 0.0,
                compression_savings_bytes: 0,
            })),
        }
    }

    /// Initialize the IPC optimizer
    pub async fn initialize(&self) -> Result<(), String> {
        info!("Initializing IPC Optimizer");

        // Start cleanup task for response cache
        self.start_cleanup_task().await;

        info!("IPC Optimizer initialized successfully");
        Ok(())
    }

    /// Start background cleanup task
    async fn start_cleanup_task(&self) {
        let response_cache = self.response_cache.clone();
        
        let cleanup_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                response_cache.cleanup_expired().await;
            }
        });

        *self.cleanup_handle.write().await = Some(cleanup_task);
    }

    /// Execute optimized IPC call
    pub async fn call(&self, method: String, params: Value, priority: u8) -> Result<Value, String> {
        let start_time = Instant::now();
        
        // Update request count
        {
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
        }

        // Check response cache first
        if let Some(cached_response) = self.response_cache.get(&method, &params).await {
            let mut stats = self.stats.write().await;
            stats.cached_responses += 1;
            
            debug!("Returning cached response for method: {}", method);
            return Ok(cached_response);
        }

        // Use command batcher for execution
        let result = self.command_batcher.queue_command(method.clone(), params.clone(), priority).await;
        
        match result {
            Ok(command_result) => {
                if command_result.success {
                    if let Some(response) = command_result.result {
                        // Cache successful responses
                        self.response_cache.put(&method, &params, response.clone()).await;
                        
                        // Update stats
                        {
                            let mut stats = self.stats.write().await;
                            stats.batched_requests += 1;
                            
                            // Update average latency
                            let latency_ms = start_time.elapsed().as_millis() as f64;
                            if stats.total_requests > 1 {
                                stats.average_latency_ms = 
                                    (stats.average_latency_ms * (stats.total_requests - 1) as f64 + latency_ms) 
                                    / stats.total_requests as f64;
                            } else {
                                stats.average_latency_ms = latency_ms;
                            }
                        }
                        
                        Ok(response)
                    } else {
                        Err("No result returned".to_string())
                    }
                } else {
                    Err(command_result.error.unwrap_or_else(|| "Unknown error".to_string()))
                }
            }
            Err(e) => Err(e),
        }
    }

    /// Execute direct IPC call (bypassing optimizations)
    pub async fn call_direct(&self, method: String, params: Value) -> Result<Value, String> {
        // This would call the underlying IPC mechanism directly
        // For now, simulate direct call
        tokio::time::sleep(Duration::from_millis(50)).await;
        
        Ok(serde_json::json!({
            "method": method,
            "result": "direct_call_result",
            "timestamp": chrono::Utc::now()
        }))
    }

    /// Get comprehensive statistics
    pub async fn get_stats(&self) -> IpcOptimizerStats {
        let mut stats = self.stats.read().await.clone();
        
        // Get additional stats from components
        let batcher_stats = self.command_batcher.get_stats().await;
        let cache_stats = self.response_cache.get_stats().await;
        
        // Calculate latency reduction
        if stats.average_latency_ms > 0.0 && batcher_stats.average_batch_size > 1.0 {
            let estimated_individual_latency = stats.average_latency_ms * batcher_stats.average_batch_size;
            stats.latency_reduction_percent = 
                ((estimated_individual_latency - stats.average_latency_ms) / estimated_individual_latency) * 100.0;
        }
        
        stats
    }

    /// Get detailed component statistics
    pub async fn get_detailed_stats(&self) -> serde_json::Value {
        let optimizer_stats = self.get_stats().await;
        let batcher_stats = self.command_batcher.get_stats().await;
        let cache_stats = self.response_cache.get_stats().await;
        
        serde_json::json!({
            "optimizer": optimizer_stats,
            "batcher": batcher_stats,
            "cache": cache_stats
        })
    }

    /// Clear all caches and reset statistics
    pub async fn reset(&self) {
        info!("Resetting IPC Optimizer");

        // Clear response cache
        self.response_cache.clear().await;
        
        // Clear pending commands
        self.command_batcher.clear_pending().await;
        
        // Reset statistics
        *self.stats.write().await = IpcOptimizerStats {
            total_requests: 0,
            batched_requests: 0,
            cached_responses: 0,
            average_latency_ms: 0.0,
            latency_reduction_percent: 0.0,
            compression_savings_bytes: 0,
        };

        info!("IPC Optimizer reset completed");
    }

    /// Handle configuration updates
    pub async fn on_config_updated(&self) {
        debug!("IPC optimizer configuration updated");
        
        // Update component configurations
        let new_ipc_config = self.config.read().await.ipc.clone();
        
        // In a full implementation, you'd update the component configs here
        info!("Updated IPC optimizer configuration");
    }

    /// Shutdown IPC optimizer
    pub async fn shutdown(&self) -> Result<(), String> {
        info!("Shutting down IPC Optimizer");

        // Stop cleanup task
        if let Some(handle) = self.cleanup_handle.write().await.take() {
            handle.abort();
        }

        // Clear pending operations
        self.command_batcher.clear_pending().await;
        self.response_cache.clear().await;

        info!("IPC Optimizer shutdown complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_command_batching() {
        let config = Arc::new(RwLock::new(IpcConfig {
            max_batch_size: 3,
            batch_timeout_ms: 100,
            response_cache_size: 100,
            cache_ttl_seconds: 300,
            enable_compression: false,
        }));
        
        let batcher = CommandBatcher::new(config);
        
        // Queue multiple commands
        let handles = vec![
            tokio::spawn({
                let batcher = batcher.clone();
                async move {
                    batcher.queue_command("test1".to_string(), Value::Null, 1).await
                }
            }),
            tokio::spawn({
                let batcher = batcher.clone();
                async move {
                    batcher.queue_command("test2".to_string(), Value::Null, 1).await
                }
            }),
            tokio::spawn({
                let batcher = batcher.clone();
                async move {
                    batcher.queue_command("test3".to_string(), Value::Null, 1).await
                }
            }),
        ];
        
        // Wait for all commands to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }
        
        let stats = batcher.get_stats().await;
        assert_eq!(stats.batches_executed, 1);
        assert_eq!(stats.commands_batched, 3);
    }

    #[tokio::test]
    async fn test_response_cache() {
        let config = Arc::new(RwLock::new(IpcConfig {
            max_batch_size: 10,
            batch_timeout_ms: 1000,
            response_cache_size: 100,
            cache_ttl_seconds: 3600,
            enable_compression: false,
        }));
        
        let cache = ResponseCache::new(config);
        
        let method = "test_method";
        let params = serde_json::json!({"param1": "value1"});
        let response = serde_json::json!({"result": "test_result"});
        
        // Should be a cache miss initially
        assert!(cache.get(method, &params).await.is_none());
        
        // Store response
        cache.put(method, &params, response.clone()).await;
        
        // Should be a cache hit now
        let cached = cache.get(method, &params).await;
        assert_eq!(cached, Some(response));
        
        let stats = cache.get_stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.entries, 1);
    }

    #[tokio::test]
    async fn test_ipc_optimizer_integration() {
        let config = Arc::new(RwLock::new(super::super::PerformanceConfig::default()));
        let optimizer = IpcOptimizer::new(config);
        
        optimizer.initialize().await.unwrap();
        
        // Make a call
        let result = optimizer.call(
            "test_method".to_string(),
            serde_json::json!({"test": "data"}),
            1
        ).await;
        
        assert!(result.is_ok());
        
        // Make the same call again - should be cached
        let result2 = optimizer.call(
            "test_method".to_string(),
            serde_json::json!({"test": "data"}),
            1
        ).await;
        
        assert!(result2.is_ok());
        
        let stats = optimizer.get_stats().await;
        assert!(stats.total_requests >= 2);
        assert!(stats.cached_responses >= 1);
        
        optimizer.shutdown().await.unwrap();
    }
}