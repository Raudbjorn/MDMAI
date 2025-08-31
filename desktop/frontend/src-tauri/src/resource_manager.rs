/**
 * Resource Manager for Thread-Safe Operation and Cleanup
 * 
 * This module provides centralized resource management, ensuring proper cleanup
 * and thread-safe operation across the entire application.
 */

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock, Semaphore};
use tokio::time::timeout;
use serde::{Deserialize, Serialize};
use log::{info, error, debug};

// Resource types for tracking
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResourceType {
    Process,
    NetworkConnection,
    FileHandle,
    Channel,
    Task,
    Stream,
    Timer,
    Memory,
}

// Resource information
#[derive(Debug, Clone)]
pub struct ResourceInfo {
    pub id: String,
    pub resource_type: ResourceType,
    pub created_at: Instant,
    pub size_bytes: Option<u64>,
    pub description: String,
    pub is_critical: bool,
}

// Resource cleanup callback
pub type CleanupCallback = Box<dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send>> + Send + Sync>;

// Resource entry with cleanup
struct ResourceEntry {
    info: ResourceInfo,
    cleanup: Option<CleanupCallback>,
    is_cleaned: AtomicBool,
}

// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStats {
    pub total_resources: u64,
    pub active_resources: u64,
    pub cleaned_resources: u64,
    pub failed_cleanups: u64,
    pub memory_usage_mb: f64,
    pub process_count: u32,
    pub network_connections: u32,
    pub file_handles: u32,
    pub active_tasks: u32,
    pub last_update: u64,
}

// Resource limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_memory_mb: u64,
    pub max_processes: u32,
    pub max_connections: u32,
    pub max_file_handles: u32,
    pub max_concurrent_tasks: u32,
    pub cleanup_timeout_ms: u64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        ResourceLimits {
            max_memory_mb: 2048, // 2GB
            max_processes: 10,
            max_connections: 100,
            max_file_handles: 1000,
            max_concurrent_tasks: 50,
            cleanup_timeout_ms: 10000, // 10 seconds
        }
    }
}

// Thread-safe resource manager
pub struct ResourceManager {
    // Resource tracking
    resources: Arc<RwLock<std::collections::HashMap<String, Arc<ResourceEntry>>>>,
    resource_counter: AtomicU64,
    
    // Limits and semaphores for resource control
    limits: Arc<RwLock<ResourceLimits>>,
    process_semaphore: Arc<Semaphore>,
    connection_semaphore: Arc<Semaphore>,
    file_semaphore: Arc<Semaphore>,
    task_semaphore: Arc<Semaphore>,
    
    // State tracking
    is_shutting_down: Arc<AtomicBool>,
    cleanup_in_progress: Arc<AtomicBool>,
    
    // Statistics
    stats: Arc<RwLock<ResourceStats>>,
    
    // Background tasks
    monitor_task: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    cleanup_task: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl ResourceManager {
    /// Create a new resource manager
    pub fn new() -> Self {
        let limits = ResourceLimits::default();
        
        ResourceManager {
            resources: Arc::new(RwLock::new(std::collections::HashMap::new())),
            resource_counter: AtomicU64::new(0),
            limits: Arc::new(RwLock::new(limits.clone())),
            process_semaphore: Arc::new(Semaphore::new(limits.max_processes as usize)),
            connection_semaphore: Arc::new(Semaphore::new(limits.max_connections as usize)),
            file_semaphore: Arc::new(Semaphore::new(limits.max_file_handles as usize)),
            task_semaphore: Arc::new(Semaphore::new(limits.max_concurrent_tasks as usize)),
            is_shutting_down: Arc::new(AtomicBool::new(false)),
            cleanup_in_progress: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(RwLock::new(ResourceStats::default())),
            monitor_task: Arc::new(Mutex::new(None)),
            cleanup_task: Arc::new(Mutex::new(None)),
        }
    }
    
    /// Create with custom limits
    pub fn with_limits(limits: ResourceLimits) -> Self {
        ResourceManager {
            resources: Arc::new(RwLock::new(std::collections::HashMap::new())),
            resource_counter: AtomicU64::new(0),
            process_semaphore: Arc::new(Semaphore::new(limits.max_processes as usize)),
            connection_semaphore: Arc::new(Semaphore::new(limits.max_connections as usize)),
            file_semaphore: Arc::new(Semaphore::new(limits.max_file_handles as usize)),
            task_semaphore: Arc::new(Semaphore::new(limits.max_concurrent_tasks as usize)),
            limits: Arc::new(RwLock::new(limits)),
            is_shutting_down: Arc::new(AtomicBool::new(false)),
            cleanup_in_progress: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(RwLock::new(ResourceStats::default())),
            monitor_task: Arc::new(Mutex::new(None)),
            cleanup_task: Arc::new(Mutex::new(None)),
        }
    }
    
    /// Start monitoring and cleanup tasks
    pub async fn start_monitoring(&self) {
        info!("Starting resource monitoring");
        
        // Start resource monitor
        let resources_clone = self.resources.clone();
        let stats_clone = self.stats.clone();
        let is_shutting_down = self.is_shutting_down.clone();
        
        let monitor_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            
            while !is_shutting_down.load(Ordering::Relaxed) {
                interval.tick().await;
                
                // Update statistics
                let resources = resources_clone.read().await;
                let resource_count = resources.len() as u64;
                let active_count = resources.values()
                    .filter(|r| !r.is_cleaned.load(Ordering::Relaxed))
                    .count() as u64;
                
                // Count by type
                let mut process_count = 0u32;
                let mut connection_count = 0u32;
                let mut file_count = 0u32;
                let mut task_count = 0u32;
                let mut memory_usage = 0u64;
                
                for resource in resources.values() {
                    if !resource.is_cleaned.load(Ordering::Relaxed) {
                        match resource.info.resource_type {
                            ResourceType::Process => process_count += 1,
                            ResourceType::NetworkConnection => connection_count += 1,
                            ResourceType::FileHandle => file_count += 1,
                            ResourceType::Task => task_count += 1,
                            ResourceType::Memory => {
                                if let Some(size) = resource.info.size_bytes {
                                    memory_usage += size;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                drop(resources);
                
                // Update stats
                let mut stats = stats_clone.write().await;
                stats.total_resources = resource_count;
                stats.active_resources = active_count;
                stats.memory_usage_mb = memory_usage as f64 / 1024.0 / 1024.0;
                stats.process_count = process_count;
                stats.network_connections = connection_count;
                stats.file_handles = file_count;
                stats.active_tasks = task_count;
                stats.last_update = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                
                debug!("Resource stats: {} active, {:.1} MB memory, {} processes", 
                       active_count, stats.memory_usage_mb, process_count);
            }
        });
        
        *self.monitor_task.lock().await = Some(monitor_handle);
        
        // Start cleanup task
        self.start_cleanup_task().await;
    }
    
    /// Start background cleanup task
    async fn start_cleanup_task(&self) {
        let resources_clone = self.resources.clone();
        let is_shutting_down = self.is_shutting_down.clone();
        let cleanup_in_progress = self.cleanup_in_progress.clone();
        let limits_clone = self.limits.clone();
        
        let cleanup_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            
            while !is_shutting_down.load(Ordering::Relaxed) {
                interval.tick().await;
                
                if cleanup_in_progress.load(Ordering::Relaxed) {
                    continue;
                }
                
                // Find resources that should be cleaned up
                let mut to_cleanup = Vec::new();
                {
                    let resources = resources_clone.read().await;
                    let now = Instant::now();
                    
                    for (id, resource) in resources.iter() {
                        // Skip already cleaned resources
                        if resource.is_cleaned.load(Ordering::Relaxed) {
                            continue;
                        }
                        
                        // Check for stale resources (older than 1 hour without being critical)
                        if !resource.info.is_critical && 
                           now.duration_since(resource.info.created_at) > Duration::from_secs(3600) {
                            to_cleanup.push((id.clone(), resource.clone()));
                        }
                    }
                }
                
                // Clean up stale resources
                if !to_cleanup.is_empty() {
                    info!("Cleaning up {} stale resources", to_cleanup.len());
                    
                    cleanup_in_progress.store(true, Ordering::Relaxed);
                    
                    for (id, resource) in to_cleanup {
                        if let Some(ref cleanup_fn) = resource.cleanup {
                            let cleanup_timeout = {
                                let limits = limits_clone.read().await;
                                Duration::from_millis(limits.cleanup_timeout_ms)
                            };
                            
                            let cleanup_future = cleanup_fn();
                            match timeout(cleanup_timeout, cleanup_future).await {
                                Ok(Ok(_)) => {
                                    resource.is_cleaned.store(true, Ordering::Relaxed);
                                    debug!("Cleaned up stale resource: {}", id);
                                }
                                Ok(Err(e)) => {
                                    error!("Failed to cleanup resource {}: {}", id, e);
                                }
                                Err(_) => {
                                    error!("Timeout cleaning up resource: {}", id);
                                }
                            }
                        } else {
                            // No cleanup function, just mark as cleaned
                            resource.is_cleaned.store(true, Ordering::Relaxed);
                        }
                    }
                    
                    cleanup_in_progress.store(false, Ordering::Relaxed);
                }
            }
        });
        
        *self.cleanup_task.lock().await = Some(cleanup_handle);
    }
    
    /// Register a new resource
    pub async fn register_resource<F>(&self, 
        resource_type: ResourceType,
        description: String,
        size_bytes: Option<u64>,
        is_critical: bool,
        cleanup: Option<F>,
    ) -> Result<String, String> 
    where
        F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send>> + Send + Sync + 'static,
    {
        if self.is_shutting_down.load(Ordering::Relaxed) {
            return Err("Resource manager is shutting down".to_string());
        }
        
        // Check resource limits
        self.check_resource_limits(&resource_type).await?;
        
        // Acquire semaphore for resource type
        let _permit = self.acquire_permit(&resource_type).await?;
        
        // Generate unique ID
        let id = format!("{}_{}", 
            resource_type_to_string(&resource_type),
            self.resource_counter.fetch_add(1, Ordering::Relaxed)
        );
        
        let info = ResourceInfo {
            id: id.clone(),
            resource_type,
            created_at: Instant::now(),
            size_bytes,
            description: description.clone(),
            is_critical,
        };
        
        let cleanup_fn = cleanup.map(|f| -> CleanupCallback {
            Box::new(move || Box::pin(f()))
        });
        
        let entry = Arc::new(ResourceEntry {
            info,
            cleanup: cleanup_fn,
            is_cleaned: AtomicBool::new(false),
        });
        
        // Register resource
        self.resources.write().await.insert(id.clone(), entry);
        
        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_resources += 1;
            stats.active_resources += 1;
        }
        
        debug!("Registered resource: {} ({})", id, description);
        Ok(id)
    }
    
    /// Unregister and cleanup a resource
    pub async fn unregister_resource(&self, id: &str) -> Result<(), String> {
        let resource = {
            let resources = self.resources.read().await;
            resources.get(id).cloned()
        };
        
        if let Some(resource) = resource {
            // Perform cleanup if not already done
            if !resource.is_cleaned.load(Ordering::Relaxed) {
                if let Some(ref cleanup_fn) = resource.cleanup {
                    let limits = self.limits.read().await;
                    let cleanup_timeout = Duration::from_millis(limits.cleanup_timeout_ms);
                    drop(limits);
                    
                    let cleanup_future = cleanup_fn();
                    match timeout(cleanup_timeout, cleanup_future).await {
                        Ok(Ok(_)) => {
                            resource.is_cleaned.store(true, Ordering::Relaxed);
                            debug!("Cleaned up resource: {}", id);
                        }
                        Ok(Err(e)) => {
                            error!("Failed to cleanup resource {}: {}", id, e);
                            let mut stats = self.stats.write().await;
                            stats.failed_cleanups += 1;
                        }
                        Err(_) => {
                            error!("Timeout cleaning up resource: {}", id);
                            let mut stats = self.stats.write().await;
                            stats.failed_cleanups += 1;
                        }
                    }
                } else {
                    resource.is_cleaned.store(true, Ordering::Relaxed);
                }
            }
            
            // Remove from registry
            self.resources.write().await.remove(id);
            
            // Update stats
            {
                let mut stats = self.stats.write().await;
                stats.active_resources = stats.active_resources.saturating_sub(1);
                stats.cleaned_resources += 1;
            }
            
            debug!("Unregistered resource: {}", id);
            Ok(())
        } else {
            Err(format!("Resource not found: {}", id))
        }
    }
    
    /// Check resource limits
    async fn check_resource_limits(&self, resource_type: &ResourceType) -> Result<(), String> {
        let limits = self.limits.read().await;
        let stats = self.stats.read().await;
        
        match resource_type {
            ResourceType::Process => {
                if stats.process_count >= limits.max_processes {
                    return Err(format!("Process limit reached: {}", limits.max_processes));
                }
            }
            ResourceType::NetworkConnection => {
                if stats.network_connections >= limits.max_connections {
                    return Err(format!("Connection limit reached: {}", limits.max_connections));
                }
            }
            ResourceType::FileHandle => {
                if stats.file_handles >= limits.max_file_handles {
                    return Err(format!("File handle limit reached: {}", limits.max_file_handles));
                }
            }
            ResourceType::Task => {
                if stats.active_tasks >= limits.max_concurrent_tasks {
                    return Err(format!("Task limit reached: {}", limits.max_concurrent_tasks));
                }
            }
            ResourceType::Memory => {
                if stats.memory_usage_mb > limits.max_memory_mb as f64 {
                    return Err(format!("Memory limit reached: {} MB", limits.max_memory_mb));
                }
            }
            _ => {} // No limits for other types
        }
        
        Ok(())
    }
    
    /// Acquire semaphore permit for resource type
    async fn acquire_permit(&self, resource_type: &ResourceType) -> Result<tokio::sync::SemaphorePermit, String> {
        let semaphore = match resource_type {
            ResourceType::Process => &self.process_semaphore,
            ResourceType::NetworkConnection => &self.connection_semaphore,
            ResourceType::FileHandle => &self.file_semaphore,
            ResourceType::Task => &self.task_semaphore,
            _ => return Ok(self.task_semaphore.acquire().await.map_err(|e| e.to_string())?), // Default
        };
        
        semaphore.acquire().await.map_err(|e| e.to_string())
    }
    
    /// Get current resource statistics
    pub async fn get_stats(&self) -> ResourceStats {
        self.stats.read().await.clone()
    }
    
    /// Update resource limits
    pub async fn update_limits(&self, limits: ResourceLimits) -> Result<(), String> {
        *self.limits.write().await = limits;
        info!("Resource limits updated");
        Ok(())
    }
    
    /// Force cleanup of all non-critical resources
    pub async fn force_cleanup(&self) -> Result<u32, String> {
        if self.cleanup_in_progress.load(Ordering::Relaxed) {
            return Err("Cleanup already in progress".to_string());
        }
        
        info!("Starting force cleanup of non-critical resources");
        self.cleanup_in_progress.store(true, Ordering::Relaxed);
        
        let mut cleaned_count = 0u32;
        let resources: Vec<_> = {
            let resources = self.resources.read().await;
            resources.iter()
                .filter(|(_, r)| !r.info.is_critical && !r.is_cleaned.load(Ordering::Relaxed))
                .map(|(id, r)| (id.clone(), r.clone()))
                .collect()
        };
        
        for (id, resource) in resources {
            if let Some(ref cleanup_fn) = resource.cleanup {
                let limits = self.limits.read().await;
                let cleanup_timeout = Duration::from_millis(limits.cleanup_timeout_ms);
                drop(limits);
                
                let cleanup_future = cleanup_fn();
                match timeout(cleanup_timeout, cleanup_future).await {
                    Ok(Ok(_)) => {
                        resource.is_cleaned.store(true, Ordering::Relaxed);
                        cleaned_count += 1;
                        debug!("Force cleaned resource: {}", id);
                    }
                    Ok(Err(e)) => {
                        error!("Failed to force cleanup resource {}: {}", id, e);
                    }
                    Err(_) => {
                        error!("Timeout during force cleanup of resource: {}", id);
                    }
                }
            } else {
                resource.is_cleaned.store(true, Ordering::Relaxed);
                cleaned_count += 1;
            }
        }
        
        self.cleanup_in_progress.store(false, Ordering::Relaxed);
        info!("Force cleanup completed: {} resources cleaned", cleaned_count);
        
        Ok(cleaned_count)
    }
    
    /// Graceful shutdown - cleanup all resources
    pub async fn shutdown(&self) -> Result<(), String> {
        info!("Starting resource manager shutdown");
        self.is_shutting_down.store(true, Ordering::Relaxed);
        
        // Stop monitoring tasks
        if let Some(handle) = self.monitor_task.lock().await.take() {
            handle.abort();
        }
        if let Some(handle) = self.cleanup_task.lock().await.take() {
            handle.abort();
        }
        
        // Wait for any ongoing cleanup to finish
        let mut wait_count = 0;
        while self.cleanup_in_progress.load(Ordering::Relaxed) && wait_count < 10 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            wait_count += 1;
        }
        
        // Force cleanup all remaining resources
        info!("Cleaning up all remaining resources");
        let resources: Vec<_> = {
            let resources = self.resources.read().await;
            resources.iter()
                .filter(|(_, r)| !r.is_cleaned.load(Ordering::Relaxed))
                .map(|(id, r)| (id.clone(), r.clone()))
                .collect()
        };
        
        let mut cleaned_count = 0;
        let mut failed_count = 0;
        
        for (id, resource) in resources {
            if let Some(ref cleanup_fn) = resource.cleanup {
                let cleanup_future = cleanup_fn();
                match timeout(Duration::from_secs(5), cleanup_future).await {
                    Ok(Ok(_)) => {
                        resource.is_cleaned.store(true, Ordering::Relaxed);
                        cleaned_count += 1;
                    }
                    Ok(Err(e)) => {
                        error!("Failed to cleanup resource {} during shutdown: {}", id, e);
                        failed_count += 1;
                    }
                    Err(_) => {
                        error!("Timeout cleaning up resource {} during shutdown", id);
                        failed_count += 1;
                    }
                }
            } else {
                resource.is_cleaned.store(true, Ordering::Relaxed);
                cleaned_count += 1;
            }
        }
        
        // Clear all resources
        self.resources.write().await.clear();
        
        info!("Resource manager shutdown complete: {} cleaned, {} failed", cleaned_count, failed_count);
        
        if failed_count > 0 {
            Err(format!("Failed to cleanup {} resources", failed_count))
        } else {
            Ok(())
        }
    }
}

impl Default for ResourceStats {
    fn default() -> Self {
        ResourceStats {
            total_resources: 0,
            active_resources: 0,
            cleaned_resources: 0,
            failed_cleanups: 0,
            memory_usage_mb: 0.0,
            process_count: 0,
            network_connections: 0,
            file_handles: 0,
            active_tasks: 0,
            last_update: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
}

// Helper function to convert resource type to string
fn resource_type_to_string(resource_type: &ResourceType) -> &'static str {
    match resource_type {
        ResourceType::Process => "process",
        ResourceType::NetworkConnection => "connection",
        ResourceType::FileHandle => "file",
        ResourceType::Channel => "channel",
        ResourceType::Task => "task",
        ResourceType::Stream => "stream",
        ResourceType::Timer => "timer",
        ResourceType::Memory => "memory",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    
    #[tokio::test]
    async fn test_resource_registration() {
        let manager = ResourceManager::new();
        
        let cleanup_called = Arc::new(Arc::new(AtomicBool::new(false)));
        let cleanup_called_clone = cleanup_called.clone();
        
        let id = manager.register_resource(
            ResourceType::Task,
            "Test resource".to_string(),
            Some(1024),
            false,
            Some(move || {
                let cleanup_called = cleanup_called_clone.clone();
                Box::pin(async move {
                    cleanup_called.store(true, Ordering::Relaxed);
                    Ok(())
                })
            }),
        ).await.unwrap();
        
        assert!(!id.is_empty());
        
        let stats = manager.get_stats().await;
        assert_eq!(stats.active_resources, 1);
        
        // Unregister should call cleanup
        manager.unregister_resource(&id).await.unwrap();
        assert!(cleanup_called.load(Ordering::Relaxed));
        
        let stats = manager.get_stats().await;
        assert_eq!(stats.active_resources, 0);
        assert_eq!(stats.cleaned_resources, 1);
    }
    
    #[tokio::test]
    async fn test_resource_limits() {
        let limits = ResourceLimits {
            max_processes: 1,
            ..Default::default()
        };
        let manager = ResourceManager::with_limits(limits);
        
        // First registration should succeed
        let id1 = manager.register_resource(
            ResourceType::Process,
            "Process 1".to_string(),
            None,
            false,
            None::<fn() -> _>,
        ).await.unwrap();
        
        // Second registration should fail due to limit
        let result = manager.register_resource(
            ResourceType::Process,
            "Process 2".to_string(),
            None,
            false,
            None::<fn() -> _>,
        ).await;
        
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("limit reached"));
        
        // After unregistering, should be able to register again
        manager.unregister_resource(&id1).await.unwrap();
        
        let id2 = manager.register_resource(
            ResourceType::Process,
            "Process 3".to_string(),
            None,
            false,
            None::<fn() -> _>,
        ).await.unwrap();
        
        assert!(!id2.is_empty());
    }
    
    #[tokio::test]
    async fn test_force_cleanup() {
        let manager = ResourceManager::new();
        
        // Register some non-critical resources
        for i in 0..5 {
            manager.register_resource(
                ResourceType::Task,
                format!("Task {}", i),
                None,
                false,
                None::<fn() -> _>,
            ).await.unwrap();
        }
        
        // Register a critical resource
        manager.register_resource(
            ResourceType::Task,
            "Critical task".to_string(),
            None,
            true,
            None::<fn() -> _>,
        ).await.unwrap();
        
        let stats = manager.get_stats().await;
        assert_eq!(stats.active_resources, 6);
        
        // Force cleanup should clean non-critical resources only
        let cleaned = manager.force_cleanup().await.unwrap();
        assert_eq!(cleaned, 5);
        
        let stats = manager.get_stats().await;
        assert_eq!(stats.active_resources, 1); // Only critical resource remains
    }
    
    #[tokio::test]
    async fn test_shutdown() {
        let manager = ResourceManager::new();
        
        // Register some resources
        for i in 0..3 {
            manager.register_resource(
                ResourceType::Task,
                format!("Task {}", i),
                None,
                i == 0, // First one is critical
                None::<fn() -> _>,
            ).await.unwrap();
        }
        
        let stats = manager.get_stats().await;
        assert_eq!(stats.active_resources, 3);
        
        // Shutdown should clean all resources
        manager.shutdown().await.unwrap();
        
        let stats = manager.get_stats().await;
        assert_eq!(stats.active_resources, 0);
    }
}