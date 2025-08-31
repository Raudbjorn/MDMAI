/*!
 * Performance Optimization Module
 * 
 * This module provides comprehensive performance optimization for the TTRPG Assistant,
 * including startup optimization, memory management, IPC optimization, and monitoring.
 */

use std::sync::Arc;
use std::time::{Instant, Duration};
use std::collections::HashMap;
use tokio::sync::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use log::{info, debug, warn, error};

pub mod startup_optimizer;
pub mod memory_manager;
pub mod ipc_optimizer;
pub mod lazy_loader;
pub mod benchmarking;
pub mod metrics;
pub mod resource_monitor;

pub use startup_optimizer::StartupOptimizer;
pub use memory_manager::{MemoryManager, MemoryPool, CacheManager};
pub use ipc_optimizer::{IpcOptimizer, CommandBatcher, ResponseCache};
pub use lazy_loader::{LazyLoader, LazyComponent};
pub use benchmarking::{BenchmarkSuite, PerformanceTest, BenchmarkResult};
pub use metrics::{MetricsCollector, PerformanceMetrics, ResourceMetrics};
pub use resource_monitor::{ResourceMonitor, ResourceAlert, ResourceThresholds};

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Startup optimization settings
    pub startup: StartupConfig,
    /// Memory management settings
    pub memory: MemoryConfig,
    /// IPC optimization settings
    pub ipc: IpcConfig,
    /// Lazy loading configuration
    pub lazy_loading: LazyLoadingConfig,
    /// Resource monitoring settings
    pub monitoring: MonitoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartupConfig {
    /// Enable parallel initialization
    pub parallel_init: bool,
    /// Maximum concurrent initialization tasks
    pub max_concurrent_tasks: usize,
    /// Startup timeout in seconds
    pub timeout_seconds: u64,
    /// Enable startup cache
    pub enable_cache: bool,
    /// Cache duration in seconds
    pub cache_duration: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum cache size in MB
    pub max_cache_size_mb: u64,
    /// Memory pool sizes
    pub pool_sizes: HashMap<String, usize>,
    /// Garbage collection interval in seconds
    pub gc_interval_seconds: u64,
    /// Memory pressure threshold (0.0 - 1.0)
    pub pressure_threshold: f64,
    /// Enable memory compaction
    pub enable_compaction: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcConfig {
    /// Maximum batch size for commands
    pub max_batch_size: usize,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Response cache size
    pub response_cache_size: usize,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Enable compression
    pub enable_compression: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LazyLoadingConfig {
    /// Components to lazy load
    pub lazy_components: Vec<String>,
    /// Preload threshold in milliseconds
    pub preload_threshold_ms: u64,
    /// Enable background preloading
    pub background_preload: bool,
    /// Preload priority order
    pub preload_order: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Metrics collection interval in seconds
    pub collection_interval_seconds: u64,
    /// Resource alert thresholds
    pub alert_thresholds: ResourceThresholds,
    /// Enable performance regression detection
    pub regression_detection: bool,
    /// Benchmark suite configuration
    pub benchmarks: BenchmarkConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Enable automatic benchmarking
    pub auto_benchmark: bool,
    /// Benchmark frequency in hours
    pub frequency_hours: u64,
    /// Warmup iterations
    pub warmup_iterations: u32,
    /// Test iterations
    pub test_iterations: u32,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            startup: StartupConfig {
                parallel_init: true,
                max_concurrent_tasks: 4,
                timeout_seconds: 10,
                enable_cache: true,
                cache_duration: 3600, // 1 hour
            },
            memory: MemoryConfig {
                max_cache_size_mb: 50,
                pool_sizes: {
                    let mut sizes = HashMap::new();
                    sizes.insert("small".to_string(), 1024);    // 1KB objects
                    sizes.insert("medium".to_string(), 8192);   // 8KB objects
                    sizes.insert("large".to_string(), 65536);   // 64KB objects
                    sizes
                },
                gc_interval_seconds: 300, // 5 minutes
                pressure_threshold: 0.85,
                enable_compaction: true,
            },
            ipc: IpcConfig {
                max_batch_size: 10,
                batch_timeout_ms: 100,
                response_cache_size: 1000,
                cache_ttl_seconds: 300,
                enable_compression: true,
            },
            lazy_loading: LazyLoadingConfig {
                lazy_components: vec![
                    "security".to_string(),
                    "data_manager".to_string(),
                    "advanced_features".to_string(),
                ],
                preload_threshold_ms: 5000,
                background_preload: true,
                preload_order: vec![
                    "core".to_string(),
                    "mcp_bridge".to_string(),
                    "data_manager".to_string(),
                    "security".to_string(),
                ],
            },
            monitoring: MonitoringConfig {
                collection_interval_seconds: 30,
                alert_thresholds: ResourceThresholds::default(),
                regression_detection: true,
                benchmarks: BenchmarkConfig {
                    auto_benchmark: true,
                    frequency_hours: 24,
                    warmup_iterations: 3,
                    test_iterations: 10,
                },
            },
        }
    }
}

/// Performance optimization manager
pub struct PerformanceManager {
    config: Arc<RwLock<PerformanceConfig>>,
    startup_optimizer: Arc<StartupOptimizer>,
    memory_manager: Arc<MemoryManager>,
    ipc_optimizer: Arc<IpcOptimizer>,
    lazy_loader: Arc<LazyLoader>,
    metrics_collector: Arc<MetricsCollector>,
    resource_monitor: Arc<ResourceMonitor>,
    benchmark_suite: Arc<BenchmarkSuite>,
    startup_time: Arc<RwLock<Option<Duration>>>,
    initialization_tasks: Arc<RwLock<HashMap<String, Instant>>>,
}

impl PerformanceManager {
    pub fn new() -> Self {
        let config = Arc::new(RwLock::new(PerformanceConfig::default()));
        
        Self {
            startup_optimizer: Arc::new(StartupOptimizer::new(config.clone())),
            memory_manager: Arc::new(MemoryManager::new(config.clone())),
            ipc_optimizer: Arc::new(IpcOptimizer::new(config.clone())),
            lazy_loader: Arc::new(LazyLoader::new(config.clone())),
            metrics_collector: Arc::new(MetricsCollector::new()),
            resource_monitor: Arc::new(ResourceMonitor::new(config.clone())),
            benchmark_suite: Arc::new(BenchmarkSuite::new()),
            config,
            startup_time: Arc::new(RwLock::new(None)),
            initialization_tasks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Initialize the performance manager
    pub async fn initialize(&self) -> Result<(), String> {
        let start = Instant::now();
        info!("Initializing Performance Manager");

        // Start resource monitoring
        self.resource_monitor.start_monitoring().await;

        // Start metrics collection
        self.metrics_collector.start_collection().await;

        // Initialize memory manager
        self.memory_manager.initialize().await?;

        // Initialize IPC optimizer
        self.ipc_optimizer.initialize().await?;

        // Initialize lazy loader
        self.lazy_loader.initialize().await?;

        let duration = start.elapsed();
        info!("Performance Manager initialized in {:?}", duration);

        Ok(())
    }

    /// Start application startup optimization
    pub async fn begin_startup(&self) -> StartupContext {
        let start = Instant::now();
        info!("Beginning optimized startup sequence");

        // Record startup begin
        self.metrics_collector.record_startup_begin().await;

        StartupContext {
            start_time: start,
            manager: Arc::new(self.clone()),
            tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Complete application startup optimization
    pub async fn complete_startup(&self, context: StartupContext) {
        let total_duration = context.start_time.elapsed();
        *self.startup_time.write().await = Some(total_duration);

        info!("Startup completed in {:?}", total_duration);

        // Record metrics
        self.metrics_collector.record_startup_complete(total_duration).await;

        // Run startup benchmark if enabled
        let config = self.config.read().await;
        if config.monitoring.benchmarks.auto_benchmark {
            tokio::spawn({
                let benchmark_suite = self.benchmark_suite.clone();
                async move {
                    if let Err(e) = benchmark_suite.run_startup_benchmark().await {
                        error!("Failed to run startup benchmark: {}", e);
                    }
                }
            });
        }
    }

    /// Get current performance metrics
    pub async fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics_collector.get_current_metrics().await
    }

    /// Update performance configuration
    pub async fn update_config(&self, new_config: PerformanceConfig) -> Result<(), String> {
        info!("Updating performance configuration");

        // Validate configuration
        self.validate_config(&new_config)?;

        // Update configuration
        *self.config.write().await = new_config;

        // Notify components of configuration change
        self.startup_optimizer.on_config_updated().await;
        self.memory_manager.on_config_updated().await;
        self.ipc_optimizer.on_config_updated().await;
        self.lazy_loader.on_config_updated().await;
        self.resource_monitor.on_config_updated().await;

        info!("Performance configuration updated successfully");
        Ok(())
    }

    /// Run performance benchmarks
    pub async fn run_benchmarks(&self) -> Result<Vec<BenchmarkResult>, String> {
        info!("Running performance benchmarks");
        self.benchmark_suite.run_full_suite().await
    }

    /// Get resource usage statistics
    pub async fn get_resource_stats(&self) -> ResourceMetrics {
        self.resource_monitor.get_current_stats().await
    }

    /// Optimize memory usage
    pub async fn optimize_memory(&self) -> Result<(), String> {
        info!("Running memory optimization");
        self.memory_manager.optimize().await
    }

    /// Clean up resources and shutdown
    pub async fn shutdown(&self) -> Result<(), String> {
        info!("Shutting down Performance Manager");

        // Stop monitoring
        self.resource_monitor.stop_monitoring().await;
        self.metrics_collector.stop_collection().await;

        // Shutdown components
        self.memory_manager.shutdown().await?;
        self.ipc_optimizer.shutdown().await?;
        self.lazy_loader.shutdown().await?;

        info!("Performance Manager shutdown complete");
        Ok(())
    }

    fn validate_config(&self, config: &PerformanceConfig) -> Result<(), String> {
        // Validate startup config
        if config.startup.max_concurrent_tasks == 0 {
            return Err("max_concurrent_tasks must be greater than 0".to_string());
        }

        if config.startup.timeout_seconds == 0 {
            return Err("timeout_seconds must be greater than 0".to_string());
        }

        // Validate memory config
        if config.memory.max_cache_size_mb == 0 {
            return Err("max_cache_size_mb must be greater than 0".to_string());
        }

        if config.memory.pressure_threshold < 0.0 || config.memory.pressure_threshold > 1.0 {
            return Err("pressure_threshold must be between 0.0 and 1.0".to_string());
        }

        // Validate IPC config
        if config.ipc.max_batch_size == 0 {
            return Err("max_batch_size must be greater than 0".to_string());
        }

        Ok(())
    }
}

impl Clone for PerformanceManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            startup_optimizer: self.startup_optimizer.clone(),
            memory_manager: self.memory_manager.clone(),
            ipc_optimizer: self.ipc_optimizer.clone(),
            lazy_loader: self.lazy_loader.clone(),
            metrics_collector: self.metrics_collector.clone(),
            resource_monitor: self.resource_monitor.clone(),
            benchmark_suite: self.benchmark_suite.clone(),
            startup_time: self.startup_time.clone(),
            initialization_tasks: self.initialization_tasks.clone(),
        }
    }
}

/// Startup optimization context
pub struct StartupContext {
    start_time: Instant,
    manager: Arc<PerformanceManager>,
    tasks: Arc<Mutex<HashMap<String, Instant>>>,
}

impl StartupContext {
    /// Begin a startup task
    pub async fn begin_task(&self, task_name: &str) -> TaskContext {
        let start = Instant::now();
        self.tasks.lock().await.insert(task_name.to_string(), start);
        
        debug!("Starting startup task: {}", task_name);
        
        TaskContext {
            task_name: task_name.to_string(),
            start_time: start,
            context: self.manager.clone(),
        }
    }

    /// Get elapsed startup time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Task execution context
pub struct TaskContext {
    task_name: String,
    start_time: Instant,
    context: Arc<PerformanceManager>,
}

impl TaskContext {
    /// Complete the task
    pub async fn complete(self) {
        let duration = self.start_time.elapsed();
        debug!("Completed startup task '{}' in {:?}", self.task_name, duration);
        
        self.context.metrics_collector
            .record_task_duration(&self.task_name, duration).await;
    }

    /// Complete the task with error
    pub async fn complete_with_error(self, error: &str) {
        let duration = self.start_time.elapsed();
        warn!("Failed startup task '{}' after {:?}: {}", 
              self.task_name, duration, error);
        
        self.context.metrics_collector
            .record_task_error(&self.task_name, duration, error).await;
    }
}

/// Performance optimization error types
#[derive(Debug, thiserror::Error)]
pub enum PerformanceError {
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Memory error: {0}")]
    Memory(String),
    
    #[error("IPC optimization error: {0}")]
    Ipc(String),
    
    #[error("Lazy loading error: {0}")]
    LazyLoading(String),
    
    #[error("Benchmarking error: {0}")]
    Benchmarking(String),
    
    #[error("Resource monitoring error: {0}")]
    ResourceMonitoring(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<PerformanceError> for String {
    fn from(error: PerformanceError) -> Self {
        error.to_string()
    }
}

/// Performance optimization results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub startup_time_ms: u64,
    pub memory_usage_mb: u64,
    pub memory_saved_mb: u64,
    pub ipc_latency_ms: f64,
    pub cache_hit_ratio: f64,
    pub optimizations_applied: Vec<String>,
    pub warnings: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl OptimizationResult {
    pub fn new() -> Self {
        Self {
            startup_time_ms: 0,
            memory_usage_mb: 0,
            memory_saved_mb: 0,
            ipc_latency_ms: 0.0,
            cache_hit_ratio: 0.0,
            optimizations_applied: Vec::new(),
            warnings: Vec::new(),
            timestamp: chrono::Utc::now(),
        }
    }
}