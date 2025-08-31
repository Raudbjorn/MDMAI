/*!
 * Resource Monitoring Module
 * 
 * Provides comprehensive system resource monitoring including CPU, memory, disk,
 * network usage tracking, alert generation, and resource optimization recommendations.
 */

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use tokio::sync::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use log::{info, debug, warn, error};
use sysinfo::{System, SystemExt, ProcessExt, CpuExt, DiskExt, NetworkExt};
use super::{PerformanceConfig, MonitoringConfig};

/// Resource usage thresholds for alerting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceThresholds {
    pub cpu_warning_percent: f32,
    pub cpu_critical_percent: f32,
    pub memory_warning_percent: f32,
    pub memory_critical_percent: f32,
    pub disk_warning_percent: f32,
    pub disk_critical_percent: f32,
    pub network_warning_mbps: f32,
    pub network_critical_mbps: f32,
    pub response_time_warning_ms: f64,
    pub response_time_critical_ms: f64,
}

impl Default for ResourceThresholds {
    fn default() -> Self {
        Self {
            cpu_warning_percent: 75.0,
            cpu_critical_percent: 90.0,
            memory_warning_percent: 80.0,
            memory_critical_percent: 95.0,
            disk_warning_percent: 85.0,
            disk_critical_percent: 95.0,
            network_warning_mbps: 100.0,
            network_critical_mbps: 1000.0,
            response_time_warning_ms: 100.0,
            response_time_critical_ms: 500.0,
        }
    }
}

/// Resource alert levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
}

/// Resource alert information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAlert {
    pub id: String,
    pub level: AlertLevel,
    pub resource_type: ResourceType,
    pub message: String,
    pub current_value: f64,
    pub threshold_value: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub resolved: bool,
    pub recommendations: Vec<String>,
}

/// Types of system resources being monitored
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResourceType {
    Cpu,
    Memory,
    Disk,
    Network,
    Process,
    Thread,
    Handle,
    ResponseTime,
}

/// Comprehensive system resource metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub system_metrics: SystemMetrics,
    pub process_metrics: ProcessMetrics,
    pub performance_metrics: PerformanceMetrics,
    pub health_score: f64,
}

/// System-wide resource metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage_percent: f32,
    pub cpu_cores: usize,
    pub cpu_frequency_mhz: u64,
    pub memory_total_gb: f64,
    pub memory_used_gb: f64,
    pub memory_available_gb: f64,
    pub memory_usage_percent: f32,
    pub swap_total_gb: f64,
    pub swap_used_gb: f64,
    pub disk_metrics: Vec<DiskMetrics>,
    pub network_metrics: Vec<NetworkMetrics>,
    pub load_average: Vec<f64>,
    pub uptime_seconds: u64,
}

/// Process-specific resource metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessMetrics {
    pub pid: u32,
    pub cpu_usage_percent: f32,
    pub memory_usage_mb: u64,
    pub memory_usage_percent: f32,
    pub virtual_memory_mb: u64,
    pub thread_count: u32,
    pub handle_count: u32,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub status: String,
    pub parent_pid: Option<u32>,
}

/// Performance-related metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub response_time_ms: f64,
    pub throughput_ops_per_sec: f64,
    pub error_rate_percent: f64,
    pub cache_hit_ratio: f64,
    pub concurrent_connections: u32,
    pub queue_depth: u32,
}

/// Individual disk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskMetrics {
    pub name: String,
    pub mount_point: String,
    pub total_space_gb: f64,
    pub available_space_gb: f64,
    pub used_space_gb: f64,
    pub usage_percent: f32,
    pub read_bytes_per_sec: u64,
    pub write_bytes_per_sec: u64,
    pub read_ops_per_sec: u32,
    pub write_ops_per_sec: u32,
    pub avg_response_time_ms: f32,
}

/// Individual network interface metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub interface_name: String,
    pub bytes_received_per_sec: u64,
    pub bytes_sent_per_sec: u64,
    pub packets_received_per_sec: u32,
    pub packets_sent_per_sec: u32,
    pub errors_received: u64,
    pub errors_sent: u64,
    pub is_up: bool,
    pub speed_mbps: Option<u32>,
}

/// Resource optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub category: RecommendationCategory,
    pub priority: RecommendationPriority,
    pub title: String,
    pub description: String,
    pub estimated_impact: String,
    pub action_required: bool,
    pub auto_applicable: bool,
}

/// Recommendation categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Memory,
    Cpu,
    Disk,
    Network,
    Configuration,
    Cleanup,
}

/// Recommendation priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Historical resource usage data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceDataPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub cpu_percent: f32,
    pub memory_percent: f32,
    pub disk_usage_percent: f32,
    pub response_time_ms: f64,
}

/// Resource monitor managing system resource tracking
pub struct ResourceMonitor {
    config: Arc<RwLock<PerformanceConfig>>,
    system_info: Arc<RwLock<System>>,
    current_metrics: Arc<RwLock<ResourceMetrics>>,
    historical_data: Arc<RwLock<VecDeque<ResourceDataPoint>>>,
    active_alerts: Arc<RwLock<HashMap<String, ResourceAlert>>>,
    monitoring_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
    alert_callbacks: Arc<RwLock<Vec<Box<dyn Fn(&ResourceAlert) + Send + Sync>>>>,
    baseline_metrics: Arc<RwLock<Option<ResourceMetrics>>>,
    performance_trends: Arc<RwLock<HashMap<String, VecDeque<f64>>>>,
}

impl ResourceMonitor {
    pub fn new(config: Arc<RwLock<PerformanceConfig>>) -> Self {
        Self {
            config,
            system_info: Arc::new(RwLock::new(System::new_all())),
            current_metrics: Arc::new(RwLock::new(Self::default_metrics())),
            historical_data: Arc::new(RwLock::new(VecDeque::new())),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            monitoring_handle: Arc::new(RwLock::new(None)),
            alert_callbacks: Arc::new(RwLock::new(Vec::new())),
            baseline_metrics: Arc::new(RwLock::new(None)),
            performance_trends: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start resource monitoring
    pub async fn start_monitoring(&self) {
        info!("Starting resource monitoring");

        // Initialize system information
        {
            let mut system = self.system_info.write().await;
            system.refresh_all();
        }

        // Set baseline metrics
        self.set_baseline().await;

        // Start monitoring loop
        self.start_monitoring_loop().await;
    }

    /// Stop resource monitoring
    pub async fn stop_monitoring(&self) {
        info!("Stopping resource monitoring");

        if let Some(handle) = self.monitoring_handle.write().await.take() {
            handle.abort();
        }
    }

    /// Start the monitoring background loop
    async fn start_monitoring_loop(&self) {
        let config = self.config.read().await.monitoring.clone();
        let interval = Duration::from_secs(config.collection_interval_seconds);

        let monitor = self.clone();
        let monitoring_task = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);

            loop {
                ticker.tick().await;

                if let Err(e) = monitor.collect_and_analyze().await {
                    error!("Resource monitoring error: {}", e);
                }
            }
        });

        *self.monitoring_handle.write().await = Some(monitoring_task);
    }

    /// Collect current metrics and perform analysis
    async fn collect_and_analyze(&self) -> Result<(), String> {
        // Refresh system information
        {
            let mut system = self.system_info.write().await;
            system.refresh_all();
        }

        // Collect current metrics
        let metrics = self.collect_current_metrics().await?;

        // Update current metrics
        *self.current_metrics.write().await = metrics.clone();

        // Add to historical data
        self.add_to_history(&metrics).await;

        // Check for alerts
        self.check_alerts(&metrics).await;

        // Update performance trends
        self.update_trends(&metrics).await;

        // Clean old data
        self.cleanup_old_data().await;

        Ok(())
    }

    /// Collect comprehensive current resource metrics
    async fn collect_current_metrics(&self) -> Result<ResourceMetrics, String> {
        let system = self.system_info.read().await;
        
        // System metrics
        let system_metrics = self.collect_system_metrics(&system).await;
        
        // Process metrics
        let process_metrics = self.collect_process_metrics(&system).await;
        
        // Performance metrics (would integrate with actual performance data)
        let performance_metrics = PerformanceMetrics {
            response_time_ms: 0.0, // Would integrate with IPC optimizer
            throughput_ops_per_sec: 0.0,
            error_rate_percent: 0.0,
            cache_hit_ratio: 0.0,
            concurrent_connections: 0,
            queue_depth: 0,
        };

        // Calculate health score
        let health_score = self.calculate_health_score(&system_metrics, &process_metrics, &performance_metrics);

        Ok(ResourceMetrics {
            timestamp: chrono::Utc::now(),
            system_metrics,
            process_metrics,
            performance_metrics,
            health_score,
        })
    }

    /// Collect system-wide resource metrics
    async fn collect_system_metrics(&self, system: &System) -> SystemMetrics {
        // CPU metrics
        let cpus = system.cpus();
        let cpu_usage_percent = cpus.iter().map(|cpu| cpu.cpu_usage()).sum::<f32>() / cpus.len() as f32;
        let cpu_cores = cpus.len();
        let cpu_frequency_mhz = cpus.first().map(|cpu| cpu.frequency()).unwrap_or(0);

        // Memory metrics
        let total_memory = system.total_memory();
        let used_memory = system.used_memory();
        let available_memory = system.available_memory();
        let memory_total_gb = total_memory as f64 / 1_024_000_000.0;
        let memory_used_gb = used_memory as f64 / 1_024_000_000.0;
        let memory_available_gb = available_memory as f64 / 1_024_000_000.0;
        let memory_usage_percent = (used_memory as f64 / total_memory as f64 * 100.0) as f32;

        // Swap metrics
        let swap_total_gb = system.total_swap() as f64 / 1_024_000_000.0;
        let swap_used_gb = system.used_swap() as f64 / 1_024_000_000.0;

        // Disk metrics
        let disk_metrics = system.disks().iter().map(|disk| {
            let total_space = disk.total_space();
            let available_space = disk.available_space();
            let used_space = total_space - available_space;
            let usage_percent = if total_space > 0 {
                (used_space as f64 / total_space as f64 * 100.0) as f32
            } else {
                0.0
            };

            DiskMetrics {
                name: disk.name().to_string_lossy().to_string(),
                mount_point: disk.mount_point().to_string_lossy().to_string(),
                total_space_gb: total_space as f64 / 1_000_000_000.0,
                available_space_gb: available_space as f64 / 1_000_000_000.0,
                used_space_gb: used_space as f64 / 1_000_000_000.0,
                usage_percent,
                read_bytes_per_sec: 0,  // Would need delta calculation
                write_bytes_per_sec: 0, // Would need delta calculation
                read_ops_per_sec: 0,
                write_ops_per_sec: 0,
                avg_response_time_ms: 0.0,
            }
        }).collect();

        // Network metrics
        let network_metrics = system.networks().iter().map(|(interface_name, network)| {
            NetworkMetrics {
                interface_name: interface_name.clone(),
                bytes_received_per_sec: 0, // Would need delta calculation
                bytes_sent_per_sec: 0,     // Would need delta calculation
                packets_received_per_sec: 0,
                packets_sent_per_sec: 0,
                errors_received: network.total_errors_on_received(),
                errors_sent: network.total_errors_on_transmitted(),
                is_up: true, // Simplified assumption
                speed_mbps: None,
            }
        }).collect();

        // Load average (Unix-like systems)
        let load_average = system.load_average();
        let load_avg_vec = vec![load_average.one, load_average.five, load_average.fifteen];

        SystemMetrics {
            cpu_usage_percent,
            cpu_cores,
            cpu_frequency_mhz,
            memory_total_gb,
            memory_used_gb,
            memory_available_gb,
            memory_usage_percent,
            swap_total_gb,
            swap_used_gb,
            disk_metrics,
            network_metrics,
            load_average: load_avg_vec,
            uptime_seconds: system.uptime(),
        }
    }

    /// Collect process-specific metrics
    async fn collect_process_metrics(&self, system: &System) -> ProcessMetrics {
        if let Ok(current_pid) = sysinfo::get_current_pid() {
            if let Some(process) = system.process(current_pid) {
                return ProcessMetrics {
                    pid: current_pid.as_u32(),
                    cpu_usage_percent: process.cpu_usage(),
                    memory_usage_mb: process.memory() / 1_024_000,
                    memory_usage_percent: (process.memory() as f64 / system.total_memory() as f64 * 100.0) as f32,
                    virtual_memory_mb: process.virtual_memory() / 1_024_000,
                    thread_count: 0, // Would need platform-specific implementation
                    handle_count: 0, // Would need platform-specific implementation
                    start_time: chrono::Utc::now(), // Would use actual process start time
                    status: process.status().to_string(),
                    parent_pid: process.parent().map(|p| p.as_u32()),
                };
            }
        }

        // Fallback metrics if process not found
        ProcessMetrics {
            pid: 0,
            cpu_usage_percent: 0.0,
            memory_usage_mb: 0,
            memory_usage_percent: 0.0,
            virtual_memory_mb: 0,
            thread_count: 0,
            handle_count: 0,
            start_time: chrono::Utc::now(),
            status: "Unknown".to_string(),
            parent_pid: None,
        }
    }

    /// Calculate overall system health score (0.0 - 100.0)
    fn calculate_health_score(
        &self, 
        system: &SystemMetrics, 
        process: &ProcessMetrics,
        performance: &PerformanceMetrics
    ) -> f64 {
        // Simple scoring algorithm - can be made more sophisticated
        let cpu_score = ((100.0 - system.cpu_usage_percent as f64).max(0.0)).min(100.0);
        let memory_score = ((100.0 - system.memory_usage_percent as f64).max(0.0)).min(100.0);
        
        let disk_score = if system.disk_metrics.is_empty() {
            100.0
        } else {
            let avg_disk_usage = system.disk_metrics.iter()
                .map(|d| d.usage_percent as f64)
                .sum::<f64>() / system.disk_metrics.len() as f64;
            ((100.0 - avg_disk_usage).max(0.0)).min(100.0)
        };

        let response_time_score = if performance.response_time_ms > 0.0 {
            ((1000.0 - performance.response_time_ms).max(0.0) / 1000.0 * 100.0).min(100.0)
        } else {
            100.0
        };

        // Weighted average
        (cpu_score * 0.3 + memory_score * 0.3 + disk_score * 0.2 + response_time_score * 0.2).round()
    }

    /// Add metrics to historical data
    async fn add_to_history(&self, metrics: &ResourceMetrics) {
        let data_point = ResourceDataPoint {
            timestamp: metrics.timestamp,
            cpu_percent: metrics.system_metrics.cpu_usage_percent,
            memory_percent: metrics.system_metrics.memory_usage_percent,
            disk_usage_percent: metrics.system_metrics.disk_metrics.first()
                .map(|d| d.usage_percent)
                .unwrap_or(0.0),
            response_time_ms: metrics.performance_metrics.response_time_ms,
        };

        let mut history = self.historical_data.write().await;
        history.push_back(data_point);

        // Limit history size (keep last 24 hours at 30-second intervals = 2880 points)
        while history.len() > 2880 {
            history.pop_front();
        }
    }

    /// Check for resource alert conditions
    async fn check_alerts(&self, metrics: &ResourceMetrics) {
        let config = self.config.read().await.monitoring.alert_thresholds.clone();
        let mut new_alerts = Vec::new();

        // Check CPU usage
        if metrics.system_metrics.cpu_usage_percent > config.cpu_critical_percent {
            new_alerts.push(self.create_alert(
                ResourceType::Cpu,
                AlertLevel::Critical,
                format!("CPU usage is critically high: {:.1}%", metrics.system_metrics.cpu_usage_percent),
                metrics.system_metrics.cpu_usage_percent as f64,
                config.cpu_critical_percent as f64,
                vec![
                    "Consider closing unnecessary applications".to_string(),
                    "Check for runaway processes".to_string(),
                    "Enable CPU performance optimizations".to_string(),
                ],
            ));
        } else if metrics.system_metrics.cpu_usage_percent > config.cpu_warning_percent {
            new_alerts.push(self.create_alert(
                ResourceType::Cpu,
                AlertLevel::Warning,
                format!("CPU usage is high: {:.1}%", metrics.system_metrics.cpu_usage_percent),
                metrics.system_metrics.cpu_usage_percent as f64,
                config.cpu_warning_percent as f64,
                vec!["Monitor CPU usage trends".to_string()],
            ));
        }

        // Check Memory usage
        if metrics.system_metrics.memory_usage_percent > config.memory_critical_percent {
            new_alerts.push(self.create_alert(
                ResourceType::Memory,
                AlertLevel::Critical,
                format!("Memory usage is critically high: {:.1}%", metrics.system_metrics.memory_usage_percent),
                metrics.system_metrics.memory_usage_percent as f64,
                config.memory_critical_percent as f64,
                vec![
                    "Clear application caches".to_string(),
                    "Close unused applications".to_string(),
                    "Run memory optimization".to_string(),
                ],
            ));
        } else if metrics.system_metrics.memory_usage_percent > config.memory_warning_percent {
            new_alerts.push(self.create_alert(
                ResourceType::Memory,
                AlertLevel::Warning,
                format!("Memory usage is high: {:.1}%", metrics.system_metrics.memory_usage_percent),
                metrics.system_metrics.memory_usage_percent as f64,
                config.memory_warning_percent as f64,
                vec!["Monitor memory usage patterns".to_string()],
            ));
        }

        // Check Disk usage
        for disk in &metrics.system_metrics.disk_metrics {
            if disk.usage_percent > config.disk_critical_percent {
                new_alerts.push(self.create_alert(
                    ResourceType::Disk,
                    AlertLevel::Critical,
                    format!("Disk '{}' usage is critically high: {:.1}%", disk.name, disk.usage_percent),
                    disk.usage_percent as f64,
                    config.disk_critical_percent as f64,
                    vec![
                        "Delete unnecessary files".to_string(),
                        "Clear temporary files".to_string(),
                        "Move files to external storage".to_string(),
                    ],
                ));
            } else if disk.usage_percent > config.disk_warning_percent {
                new_alerts.push(self.create_alert(
                    ResourceType::Disk,
                    AlertLevel::Warning,
                    format!("Disk '{}' usage is high: {:.1}%", disk.name, disk.usage_percent),
                    disk.usage_percent as f64,
                    config.disk_warning_percent as f64,
                    vec!["Plan for disk cleanup".to_string()],
                ));
            }
        }

        // Check Response Time
        if metrics.performance_metrics.response_time_ms > config.response_time_critical_ms {
            new_alerts.push(self.create_alert(
                ResourceType::ResponseTime,
                AlertLevel::Critical,
                format!("Response time is critically high: {:.1}ms", metrics.performance_metrics.response_time_ms),
                metrics.performance_metrics.response_time_ms,
                config.response_time_critical_ms,
                vec![
                    "Check for network issues".to_string(),
                    "Restart application services".to_string(),
                    "Optimize database queries".to_string(),
                ],
            ));
        }

        // Process new alerts
        let mut active_alerts = self.active_alerts.write().await;
        for alert in new_alerts {
            let alert_id = alert.id.clone();
            
            // Check if alert already exists
            if !active_alerts.contains_key(&alert_id) {
                info!("New {} alert: {}", alert.level, alert.message);
                
                // Trigger alert callbacks
                let callbacks = self.alert_callbacks.read().await;
                for callback in callbacks.iter() {
                    callback(&alert);
                }
                
                active_alerts.insert(alert_id, alert);
            }
        }

        // Resolve alerts that are no longer active
        let thresholds = &config;
        active_alerts.retain(|_, alert| {
            let should_retain = match (&alert.resource_type, &alert.level) {
                (ResourceType::Cpu, AlertLevel::Critical) => 
                    metrics.system_metrics.cpu_usage_percent > thresholds.cpu_critical_percent,
                (ResourceType::Cpu, AlertLevel::Warning) => 
                    metrics.system_metrics.cpu_usage_percent > thresholds.cpu_warning_percent,
                (ResourceType::Memory, AlertLevel::Critical) => 
                    metrics.system_metrics.memory_usage_percent > thresholds.memory_critical_percent,
                (ResourceType::Memory, AlertLevel::Warning) => 
                    metrics.system_metrics.memory_usage_percent > thresholds.memory_warning_percent,
                _ => true, // Keep other alerts for now
            };
            
            if !should_retain {
                info!("Resolved {} alert: {}", alert.level, alert.message);
            }
            
            should_retain
        });
    }

    /// Create a resource alert
    fn create_alert(
        &self,
        resource_type: ResourceType,
        level: AlertLevel,
        message: String,
        current_value: f64,
        threshold_value: f64,
        recommendations: Vec<String>,
    ) -> ResourceAlert {
        ResourceAlert {
            id: format!("{:?}_{:?}_{}", resource_type, level, chrono::Utc::now().timestamp()),
            level,
            resource_type,
            message,
            current_value,
            threshold_value,
            timestamp: chrono::Utc::now(),
            resolved: false,
            recommendations,
        }
    }

    /// Update performance trends
    async fn update_trends(&self, metrics: &ResourceMetrics) {
        let mut trends = self.performance_trends.write().await;
        
        // Track CPU trend
        let cpu_trend = trends.entry("cpu_usage".to_string()).or_insert_with(VecDeque::new);
        cpu_trend.push_back(metrics.system_metrics.cpu_usage_percent as f64);
        if cpu_trend.len() > 100 { cpu_trend.pop_front(); }
        
        // Track Memory trend
        let memory_trend = trends.entry("memory_usage".to_string()).or_insert_with(VecDeque::new);
        memory_trend.push_back(metrics.system_metrics.memory_usage_percent as f64);
        if memory_trend.len() > 100 { memory_trend.pop_front(); }
        
        // Track Response Time trend
        let response_trend = trends.entry("response_time".to_string()).or_insert_with(VecDeque::new);
        response_trend.push_back(metrics.performance_metrics.response_time_ms);
        if response_trend.len() > 100 { response_trend.pop_front(); }
    }

    /// Clean up old data
    async fn cleanup_old_data(&self) {
        // Historical data cleanup is handled in add_to_history
        // Clean up resolved alerts older than 1 hour
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(1);
        
        let mut active_alerts = self.active_alerts.write().await;
        active_alerts.retain(|_, alert| {
            !alert.resolved || alert.timestamp > cutoff
        });
    }

    /// Set baseline metrics for comparison
    async fn set_baseline(&self) {
        if let Ok(metrics) = self.collect_current_metrics().await {
            *self.baseline_metrics.write().await = Some(metrics);
            info!("Baseline metrics established");
        }
    }

    /// Get current resource metrics
    pub async fn get_current_stats(&self) -> ResourceMetrics {
        self.current_metrics.read().await.clone()
    }

    /// Get historical resource data
    pub async fn get_historical_data(&self) -> Vec<ResourceDataPoint> {
        self.historical_data.read().await.iter().cloned().collect()
    }

    /// Get active alerts
    pub async fn get_active_alerts(&self) -> Vec<ResourceAlert> {
        self.active_alerts.read().await.values().cloned().collect()
    }

    /// Get performance trends
    pub async fn get_performance_trends(&self) -> HashMap<String, Vec<f64>> {
        self.performance_trends.read().await.iter()
            .map(|(k, v)| (k.clone(), v.iter().cloned().collect()))
            .collect()
    }

    /// Get optimization recommendations
    pub async fn get_optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let metrics = self.current_metrics.read().await;
        let mut recommendations = Vec::new();

        // Memory optimization recommendations
        if metrics.system_metrics.memory_usage_percent > 70.0 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::Memory,
                priority: if metrics.system_metrics.memory_usage_percent > 90.0 {
                    RecommendationPriority::Critical
                } else {
                    RecommendationPriority::Medium
                },
                title: "Optimize Memory Usage".to_string(),
                description: format!(
                    "Memory usage is at {:.1}%. Consider clearing caches and optimizing memory allocation.",
                    metrics.system_metrics.memory_usage_percent
                ),
                estimated_impact: "10-30% memory reduction".to_string(),
                action_required: true,
                auto_applicable: true,
            });
        }

        // CPU optimization recommendations
        if metrics.system_metrics.cpu_usage_percent > 80.0 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::Cpu,
                priority: RecommendationPriority::High,
                title: "Reduce CPU Usage".to_string(),
                description: format!(
                    "CPU usage is at {:.1}%. Consider optimizing background tasks and reducing processing load.",
                    metrics.system_metrics.cpu_usage_percent
                ),
                estimated_impact: "15-25% CPU reduction".to_string(),
                action_required: true,
                auto_applicable: false,
            });
        }

        // Disk optimization recommendations
        for disk in &metrics.system_metrics.disk_metrics {
            if disk.usage_percent > 80.0 {
                recommendations.push(OptimizationRecommendation {
                    category: RecommendationCategory::Disk,
                    priority: if disk.usage_percent > 95.0 {
                        RecommendationPriority::Critical
                    } else {
                        RecommendationPriority::Medium
                    },
                    title: format!("Clean Up Disk '{}'", disk.name),
                    description: format!(
                        "Disk usage is at {:.1}%. Clean up temporary files and unused data.",
                        disk.usage_percent
                    ),
                    estimated_impact: format!("5-20% of {:.1}GB space", disk.total_space_gb),
                    action_required: true,
                    auto_applicable: true,
                });
            }
        }

        recommendations
    }

    /// Register alert callback
    pub async fn register_alert_callback<F>(&self, callback: F)
    where
        F: Fn(&ResourceAlert) + Send + Sync + 'static,
    {
        let mut callbacks = self.alert_callbacks.write().await;
        callbacks.push(Box::new(callback));
    }

    /// Handle configuration updates
    pub async fn on_config_updated(&self) {
        debug!("Resource monitor configuration updated");
        
        // Restart monitoring with new configuration
        self.stop_monitoring().await;
        self.start_monitoring().await;
    }

    /// Create default metrics
    fn default_metrics() -> ResourceMetrics {
        ResourceMetrics {
            timestamp: chrono::Utc::now(),
            system_metrics: SystemMetrics {
                cpu_usage_percent: 0.0,
                cpu_cores: 1,
                cpu_frequency_mhz: 0,
                memory_total_gb: 0.0,
                memory_used_gb: 0.0,
                memory_available_gb: 0.0,
                memory_usage_percent: 0.0,
                swap_total_gb: 0.0,
                swap_used_gb: 0.0,
                disk_metrics: Vec::new(),
                network_metrics: Vec::new(),
                load_average: vec![0.0, 0.0, 0.0],
                uptime_seconds: 0,
            },
            process_metrics: ProcessMetrics {
                pid: 0,
                cpu_usage_percent: 0.0,
                memory_usage_mb: 0,
                memory_usage_percent: 0.0,
                virtual_memory_mb: 0,
                thread_count: 0,
                handle_count: 0,
                start_time: chrono::Utc::now(),
                status: "Unknown".to_string(),
                parent_pid: None,
            },
            performance_metrics: PerformanceMetrics {
                response_time_ms: 0.0,
                throughput_ops_per_sec: 0.0,
                error_rate_percent: 0.0,
                cache_hit_ratio: 0.0,
                concurrent_connections: 0,
                queue_depth: 0,
            },
            health_score: 100.0,
        }
    }
}

impl Clone for ResourceMonitor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            system_info: self.system_info.clone(),
            current_metrics: self.current_metrics.clone(),
            historical_data: self.historical_data.clone(),
            active_alerts: self.active_alerts.clone(),
            monitoring_handle: self.monitoring_handle.clone(),
            alert_callbacks: self.alert_callbacks.clone(),
            baseline_metrics: self.baseline_metrics.clone(),
            performance_trends: self.performance_trends.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resource_monitor_initialization() {
        let config = Arc::new(RwLock::new(super::super::PerformanceConfig::default()));
        let monitor = ResourceMonitor::new(config);
        
        // Test metric collection
        let metrics = monitor.collect_current_metrics().await.unwrap();
        assert!(metrics.health_score >= 0.0 && metrics.health_score <= 100.0);
        assert!(metrics.system_metrics.cpu_cores > 0);
    }

    #[tokio::test]
    async fn test_alert_generation() {
        let config = Arc::new(RwLock::new(super::super::PerformanceConfig::default()));
        let monitor = ResourceMonitor::new(config);
        
        // Create metrics that should trigger alerts
        let mut metrics = ResourceMonitor::default_metrics();
        metrics.system_metrics.cpu_usage_percent = 95.0; // Should trigger critical CPU alert
        metrics.system_metrics.memory_usage_percent = 85.0; // Should trigger memory warning
        
        monitor.check_alerts(&metrics).await;
        
        let alerts = monitor.get_active_alerts().await;
        assert!(!alerts.is_empty());
        
        // Check that CPU critical alert was generated
        let cpu_alerts: Vec<_> = alerts.iter()
            .filter(|a| a.resource_type == ResourceType::Cpu && a.level == AlertLevel::Critical)
            .collect();
        assert!(!cpu_alerts.is_empty());
    }

    #[tokio::test]
    async fn test_health_score_calculation() {
        let config = Arc::new(RwLock::new(super::super::PerformanceConfig::default()));
        let monitor = ResourceMonitor::new(config);
        
        let system_metrics = SystemMetrics {
            cpu_usage_percent: 50.0,
            memory_usage_percent: 60.0,
            disk_metrics: vec![DiskMetrics {
                name: "test".to_string(),
                mount_point: "/".to_string(),
                total_space_gb: 100.0,
                available_space_gb: 30.0,
                used_space_gb: 70.0,
                usage_percent: 70.0,
                read_bytes_per_sec: 0,
                write_bytes_per_sec: 0,
                read_ops_per_sec: 0,
                write_ops_per_sec: 0,
                avg_response_time_ms: 0.0,
            }],
            ..Default::default()
        };
        
        let process_metrics = ProcessMetrics {
            cpu_usage_percent: 10.0,
            memory_usage_mb: 100,
            memory_usage_percent: 5.0,
            ..Default::default()
        };
        
        let performance_metrics = PerformanceMetrics {
            response_time_ms: 25.0,
            ..Default::default()
        };
        
        let health_score = monitor.calculate_health_score(&system_metrics, &process_metrics, &performance_metrics);
        assert!(health_score >= 0.0 && health_score <= 100.0);
        assert!(health_score > 50.0); // Should be reasonably healthy with these metrics
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            cpu_usage_percent: 0.0,
            cpu_cores: 1,
            cpu_frequency_mhz: 0,
            memory_total_gb: 0.0,
            memory_used_gb: 0.0,
            memory_available_gb: 0.0,
            memory_usage_percent: 0.0,
            swap_total_gb: 0.0,
            swap_used_gb: 0.0,
            disk_metrics: Vec::new(),
            network_metrics: Vec::new(),
            load_average: vec![0.0, 0.0, 0.0],
            uptime_seconds: 0,
        }
    }
}

impl Default for ProcessMetrics {
    fn default() -> Self {
        Self {
            pid: 0,
            cpu_usage_percent: 0.0,
            memory_usage_mb: 0,
            memory_usage_percent: 0.0,
            virtual_memory_mb: 0,
            thread_count: 0,
            handle_count: 0,
            start_time: chrono::Utc::now(),
            status: "Unknown".to_string(),
            parent_pid: None,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            response_time_ms: 0.0,
            throughput_ops_per_sec: 0.0,
            error_rate_percent: 0.0,
            cache_hit_ratio: 0.0,
            concurrent_connections: 0,
            queue_depth: 0,
        }
    }
}

impl std::fmt::Display for AlertLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertLevel::Info => write!(f, "INFO"),
            AlertLevel::Warning => write!(f, "WARNING"),
            AlertLevel::Critical => write!(f, "CRITICAL"),
        }
    }
}