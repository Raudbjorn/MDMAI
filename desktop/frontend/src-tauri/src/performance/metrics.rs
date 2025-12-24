/*!
 * Performance Metrics Collection Module
 * 
 * Provides comprehensive metrics collection including real-time performance monitoring,
 * resource usage tracking, and performance trend analysis.
 */

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use std::collections::{HashMap, VecDeque};
use tokio::sync::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use log::{info, debug, warn, error};
use sysinfo::{System, SystemExt, ProcessExt, CpuExt};

/// Performance metrics aggregated over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub startup_metrics: StartupMetrics,
    pub runtime_metrics: RuntimeMetrics,
    pub resource_metrics: ResourceMetrics,
    pub ipc_metrics: IpcMetrics,
    pub error_metrics: ErrorMetrics,
    pub user_interaction_metrics: UserInteractionMetrics,
}

/// Startup-specific performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartupMetrics {
    pub total_startup_time_ms: u64,
    pub component_load_times: HashMap<String, u64>,
    pub initialization_phases: Vec<PhaseMetrics>,
    pub memory_at_startup_mb: u32,
    pub time_to_first_interaction_ms: u64,
    pub critical_path_duration_ms: u64,
}

/// Runtime performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeMetrics {
    pub uptime_seconds: u64,
    pub average_cpu_usage_percent: f32,
    pub peak_cpu_usage_percent: f32,
    pub average_memory_usage_mb: u32,
    pub peak_memory_usage_mb: u32,
    pub gc_collections: u32,
    pub gc_total_time_ms: u64,
    pub background_task_performance: HashMap<String, TaskMetrics>,
}

/// Resource usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub memory_usage: MemoryUsageMetrics,
    pub cpu_usage: CpuUsageMetrics,
    pub disk_usage: DiskUsageMetrics,
    pub network_usage: NetworkUsageMetrics,
    pub thread_metrics: ThreadMetrics,
    pub handle_metrics: HandleMetrics,
}

/// IPC communication metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub requests_per_second: f64,
    pub batching_efficiency: f64,
    pub cache_hit_ratio: f64,
}

/// Error and failure metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    pub total_errors: u64,
    pub error_rate_per_hour: f64,
    pub critical_errors: u64,
    pub warning_count: u64,
    pub error_categories: HashMap<String, u64>,
    pub recent_errors: Vec<ErrorSample>,
    pub mean_time_between_failures_hours: f64,
}

/// User interaction performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInteractionMetrics {
    pub ui_response_time_ms: f64,
    pub input_latency_ms: f64,
    pub frame_rate_fps: f64,
    pub ui_thread_utilization_percent: f32,
    pub user_actions_per_minute: f64,
    pub session_duration_minutes: u64,
}

/// Individual phase metrics during startup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseMetrics {
    pub name: String,
    pub duration_ms: u64,
    pub memory_delta_mb: i32,
    pub success: bool,
    pub parallel_tasks: u32,
}

/// Background task performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetrics {
    pub executions: u64,
    pub average_duration_ms: f64,
    pub success_rate: f64,
    pub last_execution: chrono::DateTime<chrono::Utc>,
    pub queue_depth: u32,
}

/// Memory usage detailed metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageMetrics {
    pub total_allocated_mb: u32,
    pub heap_usage_mb: u32,
    pub stack_usage_mb: u32,
    pub cache_usage_mb: u32,
    pub buffer_usage_mb: u32,
    pub fragmentation_percent: f32,
    pub allocation_rate_mb_per_sec: f32,
    pub deallocation_rate_mb_per_sec: f32,
}

/// CPU usage detailed metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuUsageMetrics {
    pub total_usage_percent: f32,
    pub user_time_percent: f32,
    pub system_time_percent: f32,
    pub idle_time_percent: f32,
    pub context_switches_per_sec: u32,
    pub instructions_per_cycle: f32,
    pub cache_miss_ratio: f32,
}

/// Disk usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskUsageMetrics {
    pub read_bytes_per_sec: u64,
    pub write_bytes_per_sec: u64,
    pub read_operations_per_sec: u32,
    pub write_operations_per_sec: u32,
    pub average_seek_time_ms: f32,
    pub disk_queue_depth: u32,
}

/// Network usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkUsageMetrics {
    pub bytes_sent_per_sec: u64,
    pub bytes_received_per_sec: u64,
    pub packets_sent_per_sec: u32,
    pub packets_received_per_sec: u32,
    pub connection_count: u32,
    pub network_latency_ms: f32,
}

/// Thread performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadMetrics {
    pub active_threads: u32,
    pub thread_pool_utilization_percent: f32,
    pub average_task_wait_time_ms: f32,
    pub thread_contention_events: u32,
    pub deadlock_detections: u32,
}

/// System handle metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandleMetrics {
    pub open_file_handles: u32,
    pub open_network_handles: u32,
    pub open_registry_handles: u32,
    pub memory_mapped_files: u32,
    pub handle_leaks_detected: u32,
}

/// Error sample for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSample {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub error_type: String,
    pub message: String,
    pub component: String,
    pub stack_trace: Option<String>,
}

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub collection_interval_ms: u64,
    pub history_retention_hours: u64,
    pub enable_detailed_profiling: bool,
    pub enable_cpu_profiling: bool,
    pub enable_memory_profiling: bool,
    pub enable_network_monitoring: bool,
    pub alert_thresholds: AlertThresholds,
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub high_cpu_percent: f32,
    pub high_memory_mb: u32,
    pub high_latency_ms: f64,
    pub high_error_rate_per_hour: f64,
    pub low_disk_space_mb: u64,
    pub response_time_ms: f64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            collection_interval_ms: 30000, // 30 seconds
            history_retention_hours: 24,   // 24 hours
            enable_detailed_profiling: true,
            enable_cpu_profiling: true,
            enable_memory_profiling: true,
            enable_network_monitoring: false, // Disabled by default for privacy
            alert_thresholds: AlertThresholds {
                high_cpu_percent: 80.0,
                high_memory_mb: 500,
                high_latency_ms: 100.0,
                high_error_rate_per_hour: 10.0,
                low_disk_space_mb: 1000,
                response_time_ms: 50.0,
            },
        }
    }
}

/// Time-series data point for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsDataPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub cpu_percent: f32,
    pub memory_mb: u32,
    pub latency_ms: f64,
    pub request_rate: f64,
    pub error_count: u64,
}

/// Metrics collector managing all performance data
pub struct MetricsCollector {
    config: Arc<RwLock<MetricsConfig>>,
    current_metrics: Arc<RwLock<PerformanceMetrics>>,
    history: Arc<RwLock<VecDeque<MetricsDataPoint>>>,
    task_metrics: Arc<RwLock<HashMap<String, TaskMetrics>>>,
    error_samples: Arc<RwLock<VecDeque<ErrorSample>>>,
    collection_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
    startup_time: Arc<RwLock<Option<Instant>>>,
    component_timings: Arc<RwLock<HashMap<String, (Instant, Option<Duration>)>>>,
    ipc_latencies: Arc<RwLock<VecDeque<Duration>>>,
    system_info: Arc<RwLock<System>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            config: Arc::new(RwLock::new(MetricsConfig::default())),
            current_metrics: Arc::new(RwLock::new(Self::default_metrics())),
            history: Arc::new(RwLock::new(VecDeque::new())),
            task_metrics: Arc::new(RwLock::new(HashMap::new())),
            error_samples: Arc::new(RwLock::new(VecDeque::new())),
            collection_handle: Arc::new(RwLock::new(None)),
            startup_time: Arc::new(RwLock::new(None)),
            component_timings: Arc::new(RwLock::new(HashMap::new())),
            ipc_latencies: Arc::new(RwLock::new(VecDeque::new())),
            system_info: Arc::new(RwLock::new(System::new())),
        }
    }

    /// Initialize metrics collection
    pub async fn start_collection(&self) {
        info!("Starting performance metrics collection");
        
        // Initialize system info
        {
            let mut system = self.system_info.write().await;
            system.refresh_all();
        }

        // Start background collection task
        self.start_collection_task().await;
    }

    /// Stop metrics collection
    pub async fn stop_collection(&self) {
        info!("Stopping performance metrics collection");
        
        if let Some(handle) = self.collection_handle.write().await.take() {
            handle.abort();
        }
    }

    /// Start background metrics collection task
    async fn start_collection_task(&self) {
        let config = self.config.read().await.clone();
        let interval_ms = config.collection_interval_ms;
        
        let collector = self.clone();
        let collection_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));
            
            loop {
                interval.tick().await;
                
                if let Err(e) = collector.collect_metrics().await {
                    error!("Failed to collect metrics: {}", e);
                }
            }
        });

        *self.collection_handle.write().await = Some(collection_task);
    }

    /// Collect current performance metrics
    async fn collect_metrics(&self) -> Result<(), String> {
        let start_time = Instant::now();
        
        // Update system information
        {
            let mut system = self.system_info.write().await;
            system.refresh_all();
        }

        // Collect all metric categories
        let startup_metrics = self.collect_startup_metrics().await;
        let runtime_metrics = self.collect_runtime_metrics().await;
        let resource_metrics = self.collect_resource_metrics().await;
        let ipc_metrics = self.collect_ipc_metrics().await;
        let error_metrics = self.collect_error_metrics().await;
        let user_interaction_metrics = self.collect_user_interaction_metrics().await;

        // Create current metrics snapshot
        let metrics = PerformanceMetrics {
            timestamp: chrono::Utc::now(),
            startup_metrics,
            runtime_metrics,
            resource_metrics,
            ipc_metrics,
            error_metrics,
            user_interaction_metrics,
        };

        // Update current metrics
        *self.current_metrics.write().await = metrics.clone();

        // Add to history
        self.add_to_history(&metrics).await;

        // Clean old data
        self.cleanup_old_data().await;

        let collection_duration = start_time.elapsed();
        if collection_duration > Duration::from_millis(100) {
            warn!("Metrics collection took {:?} - consider reducing collection frequency", collection_duration);
        }

        Ok(())
    }

    /// Collect startup-specific metrics
    async fn collect_startup_metrics(&self) -> StartupMetrics {
        let startup_time = self.startup_time.read().await;
        let component_timings = self.component_timings.read().await;
        
        let total_startup_time_ms = startup_time
            .map(|start| start.elapsed().as_millis() as u64)
            .unwrap_or(0);

        let component_load_times = component_timings.iter()
            .filter_map(|(name, (start, duration))| {
                duration.map(|d| (name.clone(), d.as_millis() as u64))
            })
            .collect();

        // Get current process memory
        let memory_at_startup_mb = self.get_process_memory_usage().await;

        StartupMetrics {
            total_startup_time_ms,
            component_load_times,
            initialization_phases: vec![], // Would be populated with actual phase data
            memory_at_startup_mb,
            time_to_first_interaction_ms: 0, // Would track actual user interaction
            critical_path_duration_ms: 0,    // Would calculate critical path
        }
    }

    /// Collect runtime performance metrics
    async fn collect_runtime_metrics(&self) -> RuntimeMetrics {
        let uptime_seconds = self.startup_time.read().await
            .map(|start| start.elapsed().as_secs())
            .unwrap_or(0);

        let (avg_cpu, peak_cpu) = self.get_cpu_metrics().await;
        let (avg_memory, peak_memory) = self.get_memory_metrics().await;
        let task_metrics = self.task_metrics.read().await.clone();

        RuntimeMetrics {
            uptime_seconds,
            average_cpu_usage_percent: avg_cpu,
            peak_cpu_usage_percent: peak_cpu,
            average_memory_usage_mb: avg_memory,
            peak_memory_usage_mb: peak_memory,
            gc_collections: 0,        // Rust doesn't have GC, but could track other cleanup
            gc_total_time_ms: 0,
            background_task_performance: task_metrics,
        }
    }

    /// Collect resource usage metrics
    async fn collect_resource_metrics(&self) -> ResourceMetrics {
        let system = self.system_info.read().await;
        let process_memory = self.get_process_memory_usage().await;
        let cpu_usage = self.get_current_cpu_usage().await;

        ResourceMetrics {
            memory_usage: MemoryUsageMetrics {
                total_allocated_mb: process_memory,
                heap_usage_mb: process_memory, // Simplified
                stack_usage_mb: 1, // Estimated
                cache_usage_mb: 0,  // Would need cache manager integration
                buffer_usage_mb: 0, // Would need buffer pool integration
                fragmentation_percent: 0.0,
                allocation_rate_mb_per_sec: 0.0,
                deallocation_rate_mb_per_sec: 0.0,
            },
            cpu_usage: CpuUsageMetrics {
                total_usage_percent: cpu_usage,
                user_time_percent: cpu_usage * 0.8, // Estimated split
                system_time_percent: cpu_usage * 0.2,
                idle_time_percent: 100.0 - cpu_usage,
                context_switches_per_sec: 0,
                instructions_per_cycle: 0.0,
                cache_miss_ratio: 0.0,
            },
            disk_usage: DiskUsageMetrics {
                read_bytes_per_sec: 0,
                write_bytes_per_sec: 0,
                read_operations_per_sec: 0,
                write_operations_per_sec: 0,
                average_seek_time_ms: 0.0,
                disk_queue_depth: 0,
            },
            network_usage: NetworkUsageMetrics {
                bytes_sent_per_sec: 0,
                bytes_received_per_sec: 0,
                packets_sent_per_sec: 0,
                packets_received_per_sec: 0,
                connection_count: 0,
                network_latency_ms: 0.0,
            },
            thread_metrics: ThreadMetrics {
                active_threads: 0, // Would need actual thread counting
                thread_pool_utilization_percent: 0.0,
                average_task_wait_time_ms: 0.0,
                thread_contention_events: 0,
                deadlock_detections: 0,
            },
            handle_metrics: HandleMetrics {
                open_file_handles: 0,
                open_network_handles: 0,
                open_registry_handles: 0,
                memory_mapped_files: 0,
                handle_leaks_detected: 0,
            },
        }
    }

    /// Collect IPC performance metrics
    async fn collect_ipc_metrics(&self) -> IpcMetrics {
        let latencies = self.ipc_latencies.read().await;
        
        if latencies.is_empty() {
            return IpcMetrics {
                total_requests: 0,
                successful_requests: 0,
                failed_requests: 0,
                average_latency_ms: 0.0,
                p95_latency_ms: 0.0,
                p99_latency_ms: 0.0,
                requests_per_second: 0.0,
                batching_efficiency: 0.0,
                cache_hit_ratio: 0.0,
            };
        }

        let mut sorted_latencies: Vec<Duration> = latencies.iter().cloned().collect();
        sorted_latencies.sort();

        let total_requests = sorted_latencies.len() as u64;
        let avg_latency_ms = sorted_latencies.iter()
            .map(|d| d.as_millis() as f64)
            .sum::<f64>() / total_requests as f64;

        let p95_index = ((total_requests as f64) * 0.95) as usize;
        let p95_latency_ms = sorted_latencies.get(p95_index)
            .map(|d| d.as_millis() as f64)
            .unwrap_or(0.0);

        let p99_index = ((total_requests as f64) * 0.99) as usize;
        let p99_latency_ms = sorted_latencies.get(p99_index)
            .map(|d| d.as_millis() as f64)
            .unwrap_or(0.0);

        IpcMetrics {
            total_requests,
            successful_requests: total_requests, // Simplified - would track actual success/failure
            failed_requests: 0,
            average_latency_ms: avg_latency_ms,
            p95_latency_ms,
            p99_latency_ms,
            requests_per_second: 0.0, // Would calculate based on time window
            batching_efficiency: 0.0, // Would integrate with IPC optimizer
            cache_hit_ratio: 0.0,     // Would integrate with response cache
        }
    }

    /// Collect error and failure metrics
    async fn collect_error_metrics(&self) -> ErrorMetrics {
        let error_samples = self.error_samples.read().await;
        
        ErrorMetrics {
            total_errors: error_samples.len() as u64,
            error_rate_per_hour: 0.0, // Would calculate based on time window
            critical_errors: error_samples.iter()
                .filter(|e| e.error_type == "Critical")
                .count() as u64,
            warning_count: error_samples.iter()
                .filter(|e| e.error_type == "Warning")
                .count() as u64,
            error_categories: HashMap::new(), // Would categorize errors
            recent_errors: error_samples.iter()
                .rev()
                .take(10)
                .cloned()
                .collect(),
            mean_time_between_failures_hours: 0.0,
        }
    }

    /// Collect user interaction metrics
    async fn collect_user_interaction_metrics(&self) -> UserInteractionMetrics {
        UserInteractionMetrics {
            ui_response_time_ms: 0.0,
            input_latency_ms: 0.0,
            frame_rate_fps: 60.0, // Assumption for native app
            ui_thread_utilization_percent: 0.0,
            user_actions_per_minute: 0.0,
            session_duration_minutes: self.startup_time.read().await
                .map(|start| start.elapsed().as_secs() / 60)
                .unwrap_or(0),
        }
    }

    /// Get current CPU usage for the process
    async fn get_current_cpu_usage(&self) -> f32 {
        let system = self.system_info.read().await;
        
        if let Ok(current_pid) = sysinfo::get_current_pid() {
            if let Some(process) = system.process(current_pid) {
                return process.cpu_usage();
            }
        }
        
        0.0
    }

    /// Get current memory usage for the process in MB
    async fn get_process_memory_usage(&self) -> u32 {
        let system = self.system_info.read().await;
        
        if let Ok(current_pid) = sysinfo::get_current_pid() {
            if let Some(process) = system.process(current_pid) {
                return (process.memory() / 1024 / 1024) as u32;
            }
        }
        
        0
    }

    /// Get CPU metrics (average and peak)
    async fn get_cpu_metrics(&self) -> (f32, f32) {
        // Would track over time window
        let current = self.get_current_cpu_usage().await;
        (current, current * 1.2) // Simplified - peak is estimated
    }

    /// Get memory metrics (average and peak)
    async fn get_memory_metrics(&self) -> (u32, u32) {
        // Would track over time window
        let current = self.get_process_memory_usage().await;
        (current, current + 10) // Simplified - peak is estimated
    }

    /// Add metrics to historical data
    async fn add_to_history(&self, metrics: &PerformanceMetrics) {
        let data_point = MetricsDataPoint {
            timestamp: metrics.timestamp,
            cpu_percent: metrics.runtime_metrics.average_cpu_usage_percent,
            memory_mb: metrics.runtime_metrics.average_memory_usage_mb,
            latency_ms: metrics.ipc_metrics.average_latency_ms,
            request_rate: metrics.ipc_metrics.requests_per_second,
            error_count: metrics.error_metrics.total_errors,
        };

        let mut history = self.history.write().await;
        history.push_back(data_point);

        // Limit history size based on retention policy
        let config = self.config.read().await;
        let max_points = (config.history_retention_hours * 3600) / (config.collection_interval_ms / 1000);
        
        while history.len() > max_points as usize {
            history.pop_front();
        }
    }

    /// Clean up old data to prevent memory leaks
    async fn cleanup_old_data(&self) {
        let config = self.config.read().await;
        let retention_duration = chrono::Duration::hours(config.history_retention_hours as i64);
        let cutoff_time = chrono::Utc::now() - retention_duration;

        // Clean up error samples
        {
            let mut error_samples = self.error_samples.write().await;
            error_samples.retain(|sample| sample.timestamp > cutoff_time);
        }

        // Clean up IPC latency samples
        {
            let mut latencies = self.ipc_latencies.write().await;
            if latencies.len() > 1000 { // Keep last 1000 samples
                while latencies.len() > 1000 {
                    latencies.pop_front();
                }
            }
        }
    }

    /// Record startup beginning
    pub async fn record_startup_begin(&self) {
        *self.startup_time.write().await = Some(Instant::now());
        info!("Startup metrics recording began");
    }

    /// Record startup completion
    pub async fn record_startup_complete(&self, duration: Duration) {
        info!("Startup completed in {:?}", duration);
        // Startup metrics are automatically collected in collect_startup_metrics
    }

    /// Record task duration
    pub async fn record_task_duration(&self, task_name: &str, duration: Duration) {
        let mut component_timings = self.component_timings.write().await;
        if let Some((start, _)) = component_timings.get_mut(task_name) {
            component_timings.insert(task_name.to_string(), (*start, Some(duration)));
        } else {
            component_timings.insert(task_name.to_string(), (Instant::now(), Some(duration)));
        }
    }

    /// Record task error
    pub async fn record_task_error(&self, task_name: &str, duration: Duration, error: &str) {
        warn!("Task '{}' failed after {:?}: {}", task_name, duration, error);
        
        let error_sample = ErrorSample {
            timestamp: chrono::Utc::now(),
            error_type: "Task Error".to_string(),
            message: error.to_string(),
            component: task_name.to_string(),
            stack_trace: None,
        };

        let mut error_samples = self.error_samples.write().await;
        error_samples.push_back(error_sample);

        // Limit error sample size
        if error_samples.len() > 1000 {
            error_samples.pop_front();
        }
    }

    /// Record IPC latency
    pub async fn record_ipc_latency(&self, latency: Duration) {
        let mut latencies = self.ipc_latencies.write().await;
        latencies.push_back(latency);
        
        // Limit size
        if latencies.len() > 1000 {
            latencies.pop_front();
        }
    }

    /// Record background task execution
    pub async fn record_background_task(&self, task_name: &str, duration: Duration, success: bool) {
        let mut task_metrics = self.task_metrics.write().await;
        
        let metrics = task_metrics.entry(task_name.to_string()).or_insert(TaskMetrics {
            executions: 0,
            average_duration_ms: 0.0,
            success_rate: 0.0,
            last_execution: chrono::Utc::now(),
            queue_depth: 0,
        });

        metrics.executions += 1;
        metrics.last_execution = chrono::Utc::now();
        
        // Update rolling average
        let new_duration_ms = duration.as_millis() as f64;
        metrics.average_duration_ms = 
            (metrics.average_duration_ms * (metrics.executions - 1) as f64 + new_duration_ms) 
            / metrics.executions as f64;

        // Update success rate (simplified rolling calculation)
        let success_value = if success { 1.0 } else { 0.0 };
        metrics.success_rate = 
            (metrics.success_rate * (metrics.executions - 1) as f64 + success_value) 
            / metrics.executions as f64;
    }

    /// Get current performance metrics
    pub async fn get_current_metrics(&self) -> PerformanceMetrics {
        self.current_metrics.read().await.clone()
    }

    /// Get metrics history
    pub async fn get_history(&self) -> Vec<MetricsDataPoint> {
        self.history.read().await.iter().cloned().collect()
    }

    /// Get metrics for a specific time range
    pub async fn get_metrics_range(
        &self, 
        start: chrono::DateTime<chrono::Utc>, 
        end: chrono::DateTime<chrono::Utc>
    ) -> Vec<MetricsDataPoint> {
        self.history.read().await.iter()
            .filter(|point| point.timestamp >= start && point.timestamp <= end)
            .cloned()
            .collect()
    }

    /// Update configuration
    pub async fn update_config(&self, new_config: MetricsConfig) {
        *self.config.write().await = new_config;
        
        // Restart collection with new configuration
        self.stop_collection().await;
        self.start_collection().await;
        
        info!("Metrics collector configuration updated");
    }

    /// Create default metrics structure
    fn default_metrics() -> PerformanceMetrics {
        PerformanceMetrics {
            timestamp: chrono::Utc::now(),
            startup_metrics: StartupMetrics {
                total_startup_time_ms: 0,
                component_load_times: HashMap::new(),
                initialization_phases: Vec::new(),
                memory_at_startup_mb: 0,
                time_to_first_interaction_ms: 0,
                critical_path_duration_ms: 0,
            },
            runtime_metrics: RuntimeMetrics {
                uptime_seconds: 0,
                average_cpu_usage_percent: 0.0,
                peak_cpu_usage_percent: 0.0,
                average_memory_usage_mb: 0,
                peak_memory_usage_mb: 0,
                gc_collections: 0,
                gc_total_time_ms: 0,
                background_task_performance: HashMap::new(),
            },
            resource_metrics: ResourceMetrics {
                memory_usage: MemoryUsageMetrics {
                    total_allocated_mb: 0,
                    heap_usage_mb: 0,
                    stack_usage_mb: 0,
                    cache_usage_mb: 0,
                    buffer_usage_mb: 0,
                    fragmentation_percent: 0.0,
                    allocation_rate_mb_per_sec: 0.0,
                    deallocation_rate_mb_per_sec: 0.0,
                },
                cpu_usage: CpuUsageMetrics {
                    total_usage_percent: 0.0,
                    user_time_percent: 0.0,
                    system_time_percent: 0.0,
                    idle_time_percent: 100.0,
                    context_switches_per_sec: 0,
                    instructions_per_cycle: 0.0,
                    cache_miss_ratio: 0.0,
                },
                disk_usage: DiskUsageMetrics {
                    read_bytes_per_sec: 0,
                    write_bytes_per_sec: 0,
                    read_operations_per_sec: 0,
                    write_operations_per_sec: 0,
                    average_seek_time_ms: 0.0,
                    disk_queue_depth: 0,
                },
                network_usage: NetworkUsageMetrics {
                    bytes_sent_per_sec: 0,
                    bytes_received_per_sec: 0,
                    packets_sent_per_sec: 0,
                    packets_received_per_sec: 0,
                    connection_count: 0,
                    network_latency_ms: 0.0,
                },
                thread_metrics: ThreadMetrics {
                    active_threads: 0,
                    thread_pool_utilization_percent: 0.0,
                    average_task_wait_time_ms: 0.0,
                    thread_contention_events: 0,
                    deadlock_detections: 0,
                },
                handle_metrics: HandleMetrics {
                    open_file_handles: 0,
                    open_network_handles: 0,
                    open_registry_handles: 0,
                    memory_mapped_files: 0,
                    handle_leaks_detected: 0,
                },
            },
            ipc_metrics: IpcMetrics {
                total_requests: 0,
                successful_requests: 0,
                failed_requests: 0,
                average_latency_ms: 0.0,
                p95_latency_ms: 0.0,
                p99_latency_ms: 0.0,
                requests_per_second: 0.0,
                batching_efficiency: 0.0,
                cache_hit_ratio: 0.0,
            },
            error_metrics: ErrorMetrics {
                total_errors: 0,
                error_rate_per_hour: 0.0,
                critical_errors: 0,
                warning_count: 0,
                error_categories: HashMap::new(),
                recent_errors: Vec::new(),
                mean_time_between_failures_hours: 0.0,
            },
            user_interaction_metrics: UserInteractionMetrics {
                ui_response_time_ms: 0.0,
                input_latency_ms: 0.0,
                frame_rate_fps: 60.0,
                ui_thread_utilization_percent: 0.0,
                user_actions_per_minute: 0.0,
                session_duration_minutes: 0,
            },
        }
    }
}

impl Clone for MetricsCollector {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            current_metrics: self.current_metrics.clone(),
            history: self.history.clone(),
            task_metrics: self.task_metrics.clone(),
            error_samples: self.error_samples.clone(),
            collection_handle: self.collection_handle.clone(),
            startup_time: self.startup_time.clone(),
            component_timings: self.component_timings.clone(),
            ipc_latencies: self.ipc_latencies.clone(),
            system_info: self.system_info.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_collection() {
        let collector = MetricsCollector::new();
        
        collector.record_startup_begin().await;
        tokio::time::sleep(Duration::from_millis(10)).await;
        collector.record_startup_complete(Duration::from_millis(10)).await;
        
        collector.record_task_duration("test_task", Duration::from_millis(50)).await;
        collector.record_ipc_latency(Duration::from_millis(5)).await;
        
        let metrics = collector.get_current_metrics().await;
        assert!(metrics.startup_metrics.total_startup_time_ms >= 10);
    }

    #[tokio::test]
    async fn test_background_task_metrics() {
        let collector = MetricsCollector::new();
        
        collector.record_background_task("test_task", Duration::from_millis(100), true).await;
        collector.record_background_task("test_task", Duration::from_millis(150), true).await;
        collector.record_background_task("test_task", Duration::from_millis(75), false).await;
        
        let task_metrics = collector.task_metrics.read().await;
        let metrics = task_metrics.get("test_task").unwrap();
        
        assert_eq!(metrics.executions, 3);
        assert!(metrics.average_duration_ms > 0.0);
        assert!(metrics.success_rate < 1.0); // One failure
    }

    #[tokio::test]
    async fn test_error_recording() {
        let collector = MetricsCollector::new();
        
        collector.record_task_error("test_component", Duration::from_millis(10), "Test error").await;
        
        let metrics = collector.get_current_metrics().await;
        assert_eq!(metrics.error_metrics.total_errors, 1);
        assert!(!metrics.error_metrics.recent_errors.is_empty());
        assert_eq!(metrics.error_metrics.recent_errors[0].component, "test_component");
    }
}