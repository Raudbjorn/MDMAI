use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use log::{info, error, debug, warn};
use tauri::Manager;
use sysinfo::{System, SystemExt, ProcessExt, Pid};

/// Process state enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessState {
    Stopped,
    Starting,
    Running,
    Stopping,
    Crashed,
    Restarting,
}

/// Process health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_percent: f32,
    pub memory_mb: f64,
    pub uptime_seconds: u64,
    pub timestamp: u64,
}

/// Process lifecycle event
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProcessEvent {
    Started {
        pid: u32,
        timestamp: u64,
    },
    Stopped {
        exit_code: Option<i32>,
        timestamp: u64,
    },
    Crashed {
        error: String,
        timestamp: u64,
    },
    Restarting {
        attempt: u32,
        max_attempts: u32,
        timestamp: u64,
    },
    HealthCheckFailed {
        reason: String,
        timestamp: u64,
    },
    HealthCheckPassed {
        timestamp: u64,
    },
    ResourceAlert {
        alert_type: String,
        value: f64,
        threshold: f64,
        timestamp: u64,
    },
}

/// Configuration for process management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessConfig {
    /// Maximum restart attempts before giving up
    pub max_restart_attempts: u32,
    /// Delay between restart attempts in milliseconds
    pub restart_delay_ms: u64,
    /// Health check interval in milliseconds
    pub health_check_interval_ms: u64,
    /// Health check timeout in milliseconds
    pub health_check_timeout_ms: u64,
    /// Maximum consecutive health check failures before restart
    pub max_health_check_failures: u32,
    /// Resource monitoring interval in milliseconds
    pub resource_monitor_interval_ms: u64,
    /// CPU usage threshold for alerts (percentage)
    pub cpu_alert_threshold: f32,
    /// Memory usage threshold for alerts (MB)
    pub memory_alert_threshold: f64,
    /// Enable automatic crash recovery
    pub auto_restart_on_crash: bool,
    /// Graceful shutdown timeout in milliseconds
    pub graceful_shutdown_timeout_ms: u64,
}

impl Default for ProcessConfig {
    fn default() -> Self {
        ProcessConfig {
            max_restart_attempts: 3,
            restart_delay_ms: 2000,
            health_check_interval_ms: 30000,
            health_check_timeout_ms: 5000,
            max_health_check_failures: 3,
            resource_monitor_interval_ms: 10000,
            cpu_alert_threshold: 80.0,
            memory_alert_threshold: 500.0,
            auto_restart_on_crash: true,
            graceful_shutdown_timeout_ms: 5000,
        }
    }
}

/// Process statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessStats {
    pub state: ProcessState,
    pub health: HealthStatus,
    pub pid: Option<u32>,
    pub start_time: Option<u64>,
    pub restart_count: u32,
    pub health_check_failures: u32,
    pub last_health_check: Option<u64>,
    pub resource_usage: Option<ResourceUsage>,
    pub events: Vec<ProcessEvent>,
}

/// Process manager for handling subprocess lifecycle
pub struct ProcessManager {
    config: Arc<RwLock<ProcessConfig>>,
    stats: Arc<RwLock<ProcessStats>>,
    health_check_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    resource_monitor_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    app_handle: Arc<Mutex<Option<tauri::AppHandle>>>,
    system_info: Arc<Mutex<System>>,
}

impl ProcessManager {
    /// Create a new process manager with default configuration
    pub fn new() -> Self {
        ProcessManager {
            config: Arc::new(RwLock::new(ProcessConfig::default())),
            stats: Arc::new(RwLock::new(ProcessStats {
                state: ProcessState::Stopped,
                health: HealthStatus::Unknown,
                pid: None,
                start_time: None,
                restart_count: 0,
                health_check_failures: 0,
                last_health_check: None,
                resource_usage: None,
                events: Vec::new(),
            })),
            health_check_handle: Arc::new(Mutex::new(None)),
            resource_monitor_handle: Arc::new(Mutex::new(None)),
            app_handle: Arc::new(Mutex::new(None)),
            system_info: Arc::new(Mutex::new(System::new_all())),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ProcessConfig) -> Self {
        ProcessManager {
            config: Arc::new(RwLock::new(config)),
            stats: Arc::new(RwLock::new(ProcessStats {
                state: ProcessState::Stopped,
                health: HealthStatus::Unknown,
                pid: None,
                start_time: None,
                restart_count: 0,
                health_check_failures: 0,
                last_health_check: None,
                resource_usage: None,
                events: Vec::new(),
            })),
            health_check_handle: Arc::new(Mutex::new(None)),
            resource_monitor_handle: Arc::new(Mutex::new(None)),
            app_handle: Arc::new(Mutex::new(None)),
            system_info: Arc::new(Mutex::new(System::new_all())),
        }
    }

    /// Set the Tauri app handle for event emission
    pub async fn set_app_handle(&self, handle: tauri::AppHandle) {
        *self.app_handle.lock().await = Some(handle);
    }

    /// Update configuration
    pub async fn update_config(&self, config: ProcessConfig) {
        *self.config.write().await = config;
        info!("Process configuration updated");
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> ProcessStats {
        self.stats.read().await.clone()
    }

    /// Register process start
    pub async fn on_process_started(&self, pid: u32) {
        let mut stats = self.stats.write().await;
        stats.state = ProcessState::Running;
        stats.health = HealthStatus::Unknown;
        stats.pid = Some(pid);
        stats.start_time = Some(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
        stats.health_check_failures = 0;
        
        let event = ProcessEvent::Started {
            pid,
            timestamp: stats.start_time.unwrap(),
        };
        stats.events.push(event.clone());
        
        // Emit event
        self.emit_event(event).await;
        
        info!("Process started with PID: {}", pid);
        
        // Start monitoring tasks
        self.start_health_monitoring().await;
        self.start_resource_monitoring(pid).await;
    }

    /// Register process stop
    pub async fn on_process_stopped(&self, exit_code: Option<i32>) {
        // Stop monitoring tasks
        self.stop_monitoring().await;
        
        let mut stats = self.stats.write().await;
        let was_running = stats.state == ProcessState::Running;
        stats.state = ProcessState::Stopped;
        stats.health = HealthStatus::Unknown;
        stats.pid = None;
        
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let event = ProcessEvent::Stopped {
            exit_code,
            timestamp,
        };
        stats.events.push(event.clone());
        
        // Emit event
        self.emit_event(event).await;
        
        // Check if this was a crash
        if was_running && exit_code != Some(0) {
            self.handle_crash(exit_code).await;
        }
        
        info!("Process stopped with exit code: {:?}", exit_code);
    }

    /// Handle process crash
    async fn handle_crash(&self, exit_code: Option<i32>) {
        let config = self.config.read().await;
        let mut stats = self.stats.write().await;
        
        stats.state = ProcessState::Crashed;
        
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let event = ProcessEvent::Crashed {
            error: format!("Process crashed with exit code: {:?}", exit_code),
            timestamp,
        };
        stats.events.push(event.clone());
        
        // Emit event
        let should_restart = config.auto_restart_on_crash && 
                           stats.restart_count < config.max_restart_attempts;
        
        drop(stats);
        drop(config);
        
        self.emit_event(event).await;
        
        if should_restart {
            self.schedule_restart().await;
        } else {
            warn!("Process crashed but auto-restart is disabled or max attempts reached");
        }
    }

    /// Schedule a restart
    async fn schedule_restart(&self) {
        let config = self.config.read().await;
        let mut stats = self.stats.write().await;
        
        stats.restart_count += 1;
        stats.state = ProcessState::Restarting;
        
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let event = ProcessEvent::Restarting {
            attempt: stats.restart_count,
            max_attempts: config.max_restart_attempts,
            timestamp,
        };
        stats.events.push(event.clone());
        
        let delay = config.restart_delay_ms;
        
        drop(stats);
        drop(config);
        
        self.emit_event(event).await;
        
        info!("Scheduling restart in {} ms", delay);
        
        // Note: Actual restart logic should be implemented by the caller
        // This just updates the state and emits events
    }

    /// Start health monitoring
    async fn start_health_monitoring(&self) {
        let config = self.config.read().await;
        let interval = config.health_check_interval_ms;
        drop(config);
        
        let stats = self.stats.clone();
        let config = self.config.clone();
        let app_handle = self.app_handle.clone();
        
        let handle = tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(Duration::from_millis(interval));
            
            loop {
                interval_timer.tick().await;
                
                let current_state = stats.read().await.state;
                if current_state != ProcessState::Running {
                    break;
                }
                
                // Perform health check (this should be implemented by the bridge)
                // For now, we'll just update the timestamp
                let mut stats_mut = stats.write().await;
                stats_mut.last_health_check = Some(
                    SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                );
                
                // Note: Actual health check logic should be implemented by the caller
                // This is just the monitoring framework
            }
        });
        
        *self.health_check_handle.lock().await = Some(handle);
    }

    /// Start resource monitoring
    async fn start_resource_monitoring(&self, pid: u32) {
        let config = self.config.read().await;
        let interval = config.resource_monitor_interval_ms;
        let cpu_threshold = config.cpu_alert_threshold;
        let memory_threshold = config.memory_alert_threshold;
        drop(config);
        
        let stats = self.stats.clone();
        let system_info = self.system_info.clone();
        let app_handle = self.app_handle.clone();
        
        let handle = tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(Duration::from_millis(interval));
            let pid_sysinfo = Pid::from(pid as usize);
            
            loop {
                interval_timer.tick().await;
                
                let current_state = stats.read().await.state;
                if current_state != ProcessState::Running {
                    break;
                }
                
                // Update system info
                let mut sys = system_info.lock().await;
                sys.refresh_process(pid_sysinfo);
                
                if let Some(process) = sys.process(pid_sysinfo) {
                    let cpu_usage = process.cpu_usage();
                    let memory_usage = process.memory() as f64 / 1024.0 / 1024.0; // Convert to MB
                    
                    let resource_usage = ResourceUsage {
                        cpu_percent: cpu_usage,
                        memory_mb: memory_usage,
                        uptime_seconds: process.run_time(),
                        timestamp: SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    };
                    
                    // Check thresholds and emit alerts
                    if cpu_usage > cpu_threshold {
                        let event = ProcessEvent::ResourceAlert {
                            alert_type: "cpu_high".to_string(),
                            value: cpu_usage as f64,
                            threshold: cpu_threshold as f64,
                            timestamp: resource_usage.timestamp,
                        };
                        
                        if let Some(handle) = app_handle.lock().await.as_ref() {
                            let _ = handle.emit_all("process-event", &event);
                        }
                    }
                    
                    if memory_usage > memory_threshold {
                        let event = ProcessEvent::ResourceAlert {
                            alert_type: "memory_high".to_string(),
                            value: memory_usage,
                            threshold: memory_threshold,
                            timestamp: resource_usage.timestamp,
                        };
                        
                        if let Some(handle) = app_handle.lock().await.as_ref() {
                            let _ = handle.emit_all("process-event", &event);
                        }
                    }
                    
                    // Update stats
                    let mut stats_mut = stats.write().await;
                    stats_mut.resource_usage = Some(resource_usage);
                }
            }
            
            debug!("Resource monitoring stopped");
        });
        
        *self.resource_monitor_handle.lock().await = Some(handle);
    }

    /// Stop all monitoring tasks
    async fn stop_monitoring(&self) {
        // Stop health check monitoring
        if let Some(handle) = self.health_check_handle.lock().await.take() {
            handle.abort();
        }
        
        // Stop resource monitoring
        if let Some(handle) = self.resource_monitor_handle.lock().await.take() {
            handle.abort();
        }
    }

    /// Handle health check result
    pub async fn on_health_check_result(&self, is_healthy: bool, error: Option<String>) {
        let config = self.config.read().await;
        let mut stats = self.stats.write().await;
        
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        stats.last_health_check = Some(timestamp);
        
        if is_healthy {
            stats.health_check_failures = 0;
            stats.health = HealthStatus::Healthy;
            
            let event = ProcessEvent::HealthCheckPassed { timestamp };
            stats.events.push(event.clone());
            drop(stats);
            drop(config);
            self.emit_event(event).await;
        } else {
            stats.health_check_failures += 1;
            
            if stats.health_check_failures >= config.max_health_check_failures {
                stats.health = HealthStatus::Unhealthy;
                
                let event = ProcessEvent::HealthCheckFailed {
                    reason: error.unwrap_or_else(|| "Unknown error".to_string()),
                    timestamp,
                };
                stats.events.push(event.clone());
                
                // Trigger restart if needed
                let should_restart = config.auto_restart_on_crash;
                drop(stats);
                drop(config);
                
                self.emit_event(event).await;
                
                if should_restart {
                    warn!("Max health check failures reached, triggering restart");
                    self.schedule_restart().await;
                }
            } else {
                stats.health = HealthStatus::Degraded;
                
                let event = ProcessEvent::HealthCheckFailed {
                    reason: error.unwrap_or_else(|| "Health check failed".to_string()),
                    timestamp,
                };
                stats.events.push(event.clone());
                drop(stats);
                drop(config);
                self.emit_event(event).await;
            }
        }
    }

    /// Reset restart counter
    pub async fn reset_restart_count(&self) {
        let mut stats = self.stats.write().await;
        stats.restart_count = 0;
        info!("Restart counter reset");
    }

    /// Clear event history
    pub async fn clear_events(&self) {
        let mut stats = self.stats.write().await;
        stats.events.clear();
        info!("Event history cleared");
    }

    /// Emit event through Tauri
    async fn emit_event(&self, event: ProcessEvent) {
        if let Some(handle) = self.app_handle.lock().await.as_ref() {
            if let Err(e) = handle.emit_all("process-event", &event) {
                error!("Failed to emit process event: {}", e);
            }
        }
    }

    /// Get recent events
    pub async fn get_recent_events(&self, limit: usize) -> Vec<ProcessEvent> {
        let stats = self.stats.read().await;
        let len = stats.events.len();
        let start = if len > limit { len - limit } else { 0 };
        stats.events[start..].to_vec()
    }

    /// Check if process should be restarted
    pub async fn should_restart(&self) -> bool {
        let config = self.config.read().await;
        let stats = self.stats.read().await;
        
        config.auto_restart_on_crash && 
        stats.restart_count < config.max_restart_attempts &&
        (stats.state == ProcessState::Crashed || stats.health == HealthStatus::Unhealthy)
    }
}

// Thread-safe wrapper for use in Tauri state
pub struct ProcessManagerState(pub Arc<ProcessManager>);

impl ProcessManagerState {
    pub fn new() -> Self {
        ProcessManagerState(Arc::new(ProcessManager::new()))
    }
    
    pub fn with_config(config: ProcessConfig) -> Self {
        ProcessManagerState(Arc::new(ProcessManager::with_config(config)))
    }
}

// Tauri commands for process management
#[tauri::command]
pub async fn get_process_stats(
    state: tauri::State<'_, ProcessManagerState>,
) -> Result<ProcessStats, String> {
    Ok(state.0.get_stats().await)
}

#[tauri::command]
pub async fn get_process_events(
    state: tauri::State<'_, ProcessManagerState>,
    limit: Option<usize>,
) -> Result<Vec<ProcessEvent>, String> {
    Ok(state.0.get_recent_events(limit.unwrap_or(100)).await)
}

#[tauri::command]
pub async fn update_process_config(
    state: tauri::State<'_, ProcessManagerState>,
    config: ProcessConfig,
) -> Result<(), String> {
    state.0.update_config(config).await;
    Ok(())
}

#[tauri::command]
pub async fn reset_process_restart_count(
    state: tauri::State<'_, ProcessManagerState>,
) -> Result<(), String> {
    state.0.reset_restart_count().await;
    Ok(())
}

#[tauri::command]
pub async fn clear_process_events(
    state: tauri::State<'_, ProcessManagerState>,
) -> Result<(), String> {
    state.0.clear_events().await;
    Ok(())
}