use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use parking_lot::RwLock;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Custom error types for better error handling
#[derive(Debug, Error)]
pub enum NativeError {
    #[error("System metric collection failed: {0}")]
    MetricCollection(String),
    #[error("File system operation failed: {0}")]
    FileSystem(#[from] std::io::Error),
    #[error("Notification failed: {0}")]
    Notification(String),
    #[error("Cache operation failed: {0}")]
    Cache(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub cpu_usage: f32,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub network_active: bool,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl SystemStatus {
    /// Check if the status is considered stale
    pub fn is_stale(&self, max_age: Duration) -> bool {
        let age = chrono::Utc::now() - self.last_updated;
        age.to_std().unwrap_or(Duration::MAX) > max_age
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSystemInfo {
    pub path: String,
    pub exists: bool,
    pub is_file: bool,
    pub is_dir: bool,
    pub size: Option<u64>,
    pub modified: Option<chrono::DateTime<chrono::Utc>>,
    pub permissions: FilePermissions,
}

impl FileSystemInfo {
    /// Create FileSystemInfo from a path
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, NativeError> {
        let path_ref = path.as_ref();
        let path_string = path_ref.to_string_lossy().into_owned();
        
        let metadata = std::fs::metadata(path_ref);
        let exists = metadata.is_ok();
        
        if let Ok(meta) = metadata {
            let modified = meta.modified()
                .ok()
                .and_then(|time| {
                    time.duration_since(SystemTime::UNIX_EPOCH)
                        .ok()
                        .and_then(|dur| chrono::DateTime::from_timestamp(dur.as_secs() as i64, 0))
                });

            let permissions = FilePermissions::from_metadata(&meta);

            Ok(Self {
                path: path_string,
                exists: true,
                is_file: meta.is_file(),
                is_dir: meta.is_dir(),
                size: Some(meta.len()),
                modified,
                permissions,
            })
        } else {
            Ok(Self {
                path: path_string,
                exists: false,
                is_file: false,
                is_dir: false,
                size: None,
                modified: None,
                permissions: FilePermissions::default(),
            })
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FilePermissions {
    pub readable: bool,
    pub writable: bool,
    pub executable: bool,
}

impl FilePermissions {
    fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        Self {
            readable: true, // If we can read metadata, it's readable
            writable: !metadata.permissions().readonly(),
            executable: {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    metadata.permissions().mode() & 0o111 != 0
                }
                #[cfg(not(unix))]
                {
                    false
                }
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    pub title: String,
    pub body: String,
    pub icon: Option<String>,
    pub urgency: NotificationUrgency,
}

impl NotificationConfig {
    pub fn new<T: Into<String>, B: Into<String>>(title: T, body: B) -> Self {
        Self {
            title: title.into(),
            body: body.into(),
            icon: None,
            urgency: NotificationUrgency::Normal,
        }
    }

    pub fn with_urgency(mut self, urgency: NotificationUrgency) -> Self {
        self.urgency = urgency;
        self
    }

    pub fn with_icon<I: Into<String>>(mut self, icon: I) -> Self {
        self.icon = Some(icon.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationUrgency {
    Low,
    Normal,
    Critical,
}

/// Configuration for the native features manager
#[derive(Debug, Clone)]
pub struct NativeFeaturesConfig {
    pub cache_max_age: Duration,
    pub max_notification_history: usize,
}

impl Default for NativeFeaturesConfig {
    fn default() -> Self {
        Self {
            cache_max_age: Duration::from_secs(30),
            max_notification_history: 100,
        }
    }
}

/// Trait for system metric collection - allows for dependency injection and testing
pub trait SystemMetricsProvider: Send + Sync {
    async fn get_cpu_usage(&self) -> Result<f32, NativeError>;
    async fn get_memory_usage(&self) -> Result<f64, NativeError>;
    async fn get_disk_usage(&self) -> Result<f64, NativeError>;
    async fn check_network_activity(&self) -> Result<bool, NativeError>;
}

/// Trait for notification sending - allows for different notification backends
pub trait NotificationSender: Send + Sync {
    async fn send_notification(&self, config: &NotificationConfig) -> Result<(), NativeError>;
}

/// Thread-safe native features manager with dependency injection for testability
pub struct NativeFeaturesManager<M = DefaultMetricsProvider, N = DefaultNotificationSender> 
where
    M: SystemMetricsProvider,
    N: NotificationSender,
{
    config: NativeFeaturesConfig,
    status_cache: Arc<RwLock<Option<SystemStatus>>>,
    notification_history: Arc<RwLock<Vec<NotificationConfig>>>,
    metrics_provider: Arc<M>,
    notification_sender: Arc<N>,
}

impl NativeFeaturesManager<DefaultMetricsProvider, DefaultNotificationSender> {
    pub fn new() -> Self {
        Self::with_config(NativeFeaturesConfig::default())
    }

    pub fn with_config(config: NativeFeaturesConfig) -> Self {
        Self {
            config,
            status_cache: Arc::new(RwLock::new(None)),
            notification_history: Arc::new(RwLock::new(Vec::new())),
            metrics_provider: Arc::new(DefaultMetricsProvider),
            notification_sender: Arc::new(DefaultNotificationSender),
        }
    }
}

impl<M: SystemMetricsProvider, N: NotificationSender> NativeFeaturesManager<M, N> {
    pub fn with_providers(
        config: NativeFeaturesConfig,
        metrics_provider: M,
        notification_sender: N,
    ) -> Self {
        Self {
            config,
            status_cache: Arc::new(RwLock::new(None)),
            notification_history: Arc::new(RwLock::new(Vec::new())),
            metrics_provider: Arc::new(metrics_provider),
            notification_sender: Arc::new(notification_sender),
        }
    }

    /// Get current system status with efficient caching
    pub async fn get_system_status(&self) -> Result<SystemStatus, NativeError> {
        // Check if cached status is still valid
        if let Some(status) = self.get_cached_status() {
            return Ok(status);
        }

        // Collect fresh metrics and update cache
        let status = self.collect_system_metrics().await?;
        self.update_cache(status.clone());
        Ok(status)
    }

    /// Get cached status if it's not stale
    fn get_cached_status(&self) -> Option<SystemStatus> {
        self.status_cache
            .read()
            .as_ref()
            .filter(|status| !status.is_stale(self.config.cache_max_age))
            .cloned()
    }

    /// Update the status cache
    fn update_cache(&self, status: SystemStatus) {
        *self.status_cache.write() = Some(status);
    }

    /// Collect system metrics using dependency injection
    async fn collect_system_metrics(&self) -> Result<SystemStatus, NativeError> {
        // Use ? operator with parallel collection for efficiency
        let (cpu_usage, memory_usage, disk_usage, network_active) = tokio::try_join!(
            self.metrics_provider.get_cpu_usage(),
            self.metrics_provider.get_memory_usage(),
            self.metrics_provider.get_disk_usage(),
            self.metrics_provider.check_network_activity(),
        )?;

        Ok(SystemStatus {
            cpu_usage,
            memory_usage,
            disk_usage,
            network_active,
            last_updated: chrono::Utc::now(),
        })
    }

    /// Get file system information using the improved method
    pub async fn get_file_info<P: AsRef<Path>>(&self, path: P) -> Result<FileSystemInfo, NativeError> {
        let path_buf = path.as_ref().to_path_buf();
        tokio::task::spawn_blocking(move || FileSystemInfo::from_path(path_buf))
            .await
            .map_err(|e| NativeError::MetricCollection(format!("Task join error: {}", e)))?
    }

    /// Send system notification with improved error handling
    pub async fn send_notification(&self, config: NotificationConfig) -> Result<(), NativeError> {
        // Store in history with size management
        self.add_to_notification_history(config.clone());
        
        // Send notification using injected sender
        self.notification_sender.send_notification(&config).await
    }

    /// Add notification to history with size limit
    fn add_to_notification_history(&self, config: NotificationConfig) {
        let mut history = self.notification_history.write();
        history.push(config);
        
        // Maintain size limit using efficient drain operation
        if history.len() > self.config.max_notification_history {
            let excess = history.len() - self.config.max_notification_history;
            history.drain(0..excess);
        }
    }

    /// Get notification history clone
    pub fn get_notification_history(&self) -> Vec<NotificationConfig> {
        self.notification_history.read().clone()
    }

    /// Clear notification history
    pub fn clear_notification_history(&self) {
        self.notification_history.write().clear();
    }

    /// Get system information summary with better error handling
    pub async fn get_system_info(&self) -> Result<HashMap<String, serde_json::Value>, NativeError> {
        let mut info = HashMap::with_capacity(8);
        
        // Add basic system information
        info.insert("platform".to_owned(), serde_json::Value::String(std::env::consts::OS.to_owned()));
        info.insert("architecture".to_owned(), serde_json::Value::String(std::env::consts::ARCH.to_owned()));
        
        // Add paths with proper error handling
        if let Ok(current_dir) = std::env::current_dir() {
            info.insert("current_dir".to_owned(), 
                       serde_json::Value::String(current_dir.to_string_lossy().into_owned()));
        }
        
        info.insert("temp_dir".to_owned(), 
                   serde_json::Value::String(std::env::temp_dir().to_string_lossy().into_owned()));
        
        // Add system status if available
        match self.get_system_status().await {
            Ok(status) => {
                if let Ok(status_json) = serde_json::to_value(status) {
                    info.insert("status".to_owned(), status_json);
                }
            },
            Err(e) => {
                log::warn!("Failed to get system status: {}", e);
            }
        }
        
        Ok(info)
    }
}

/// Default implementation of SystemMetricsProvider
#[derive(Debug, Default)]
pub struct DefaultMetricsProvider;

impl SystemMetricsProvider for DefaultMetricsProvider {
    async fn get_cpu_usage(&self) -> Result<f32, NativeError> {
        tokio::task::spawn_blocking(|| {
            #[cfg(target_os = "linux")]
            {
                std::fs::read_to_string("/proc/stat")
                    .map_err(|e| NativeError::MetricCollection(format!("Failed to read /proc/stat: {}", e)))
                    .and_then(|contents| {
                        let line = contents.lines().next()
                            .ok_or_else(|| NativeError::MetricCollection("No CPU line in /proc/stat".to_string()))?;
                        
                        if !line.starts_with("cpu ") {
                            return Err(NativeError::MetricCollection("Invalid CPU line format".to_string()));
                        }

                        let values: Result<Vec<u64>, _> = line
                            .split_whitespace()
                            .skip(1)
                            .take(7)
                            .map(str::parse)
                            .collect();

                        let values = values.map_err(|_| NativeError::MetricCollection("Failed to parse CPU values".to_string()))?;
                        
                        if values.len() >= 4 {
                            let idle = values[3];
                            let total: u64 = values.iter().sum();
                            
                            if total > 0 {
                                let usage = (100.0 * (total - idle) as f32) / total as f32;
                                Ok(usage.clamp(0.0, 100.0))
                            } else {
                                Ok(0.0)
                            }
                        } else {
                            Err(NativeError::MetricCollection("Insufficient CPU values".to_string()))
                        }
                    })
            }
            
            #[cfg(not(target_os = "linux"))]
            {
                // Mock data for non-Linux platforms
                Ok(15.5)
            }
        })
        .await
        .map_err(|e| NativeError::MetricCollection(format!("Task join error: {}", e)))?
    }

    async fn get_memory_usage(&self) -> Result<f64, NativeError> {
        tokio::task::spawn_blocking(|| {
            #[cfg(target_os = "linux")]
            {
                std::fs::read_to_string("/proc/meminfo")
                    .map_err(|e| NativeError::MetricCollection(format!("Failed to read /proc/meminfo: {}", e)))
                    .and_then(|contents| {
                        let mut total_mem = None;
                        let mut available_mem = None;
                        
                        for line in contents.lines() {
                            if let Some(value_str) = line.strip_prefix("MemTotal:").and_then(|s| s.trim().split_whitespace().next()) {
                                total_mem = value_str.parse().ok();
                            } else if let Some(value_str) = line.strip_prefix("MemAvailable:").and_then(|s| s.trim().split_whitespace().next()) {
                                available_mem = value_str.parse().ok();
                                break;
                            }
                        }
                        
                        match (total_mem, available_mem) {
                            (Some(total), Some(available)) if total > 0 => {
                                let used = total.saturating_sub(available);
                                let usage_percent = (used as f64 / total as f64) * 100.0;
                                Ok(usage_percent.clamp(0.0, 100.0))
                            }
                            _ => Err(NativeError::MetricCollection("Could not parse memory values".to_string()))
                        }
                    })
            }
            
            #[cfg(not(target_os = "linux"))]
            {
                Ok(45.2)
            }
        })
        .await
        .map_err(|e| NativeError::MetricCollection(format!("Task join error: {}", e)))?
    }

    async fn get_disk_usage(&self) -> Result<f64, NativeError> {
        tokio::task::spawn_blocking(|| {
            let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            
            #[cfg(unix)]
            {
                use std::ffi::CString;
                use std::mem::MaybeUninit;
                
                let path_cstr = CString::new(current_dir.to_string_lossy().as_bytes())
                    .map_err(|_| NativeError::MetricCollection("Invalid path for disk usage".to_string()))?;
                
                let mut statvfs = MaybeUninit::uninit();
                
                let result = unsafe {
                    libc::statvfs(path_cstr.as_ptr(), statvfs.as_mut_ptr())
                };
                
                if result == 0 {
                    let statvfs = unsafe { statvfs.assume_init() };
                    let total_blocks = statvfs.f_blocks;
                    let free_blocks = statvfs.f_bavail;
                    
                    if total_blocks > 0 {
                        let used_blocks = total_blocks.saturating_sub(free_blocks);
                        let usage_percent = (used_blocks as f64 / total_blocks as f64) * 100.0;
                        Ok(usage_percent.clamp(0.0, 100.0))
                    } else {
                        Ok(0.0)
                    }
                } else {
                    Err(NativeError::MetricCollection("statvfs call failed".to_string()))
                }
            }
            
            #[cfg(not(unix))]
            {
                Ok(25.7)
            }
        })
        .await
        .map_err(|e| NativeError::MetricCollection(format!("Task join error: {}", e)))?
    }

    async fn check_network_activity(&self) -> Result<bool, NativeError> {
        tokio::task::spawn_blocking(|| {
            #[cfg(target_os = "linux")]
            {
                std::fs::read_to_string("/proc/net/dev")
                    .map_err(|e| NativeError::MetricCollection(format!("Failed to read /proc/net/dev: {}", e)))
                    .map(|contents| {
                        contents
                            .lines()
                            .skip(2) // Skip header lines
                            .any(|line| {
                                let fields: Vec<&str> = line.split_whitespace().collect();
                                if fields.len() >= 10 {
                                    let rx_bytes: u64 = fields[1].parse().unwrap_or(0);
                                    let tx_bytes: u64 = fields[9].parse().unwrap_or(0);
                                    rx_bytes > 0 || tx_bytes > 0
                                } else {
                                    false
                                }
                            })
                    })
            }
            
            #[cfg(not(target_os = "linux"))]
            {
                Ok(true)
            }
        })
        .await
        .map_err(|e| NativeError::MetricCollection(format!("Task join error: {}", e)))?
    }
}

/// Default implementation of NotificationSender
#[derive(Debug, Default)]
pub struct DefaultNotificationSender;

impl NotificationSender for DefaultNotificationSender {
    async fn send_notification(&self, config: &NotificationConfig) -> Result<(), NativeError> {
        let config = config.clone();
        tokio::task::spawn_blocking(move || {
            #[cfg(target_os = "linux")]
            {
                use std::process::Command;
                
                let mut cmd = Command::new("notify-send");
                cmd.arg(&config.title).arg(&config.body);
                
                match config.urgency {
                    NotificationUrgency::Low => { cmd.arg("-u").arg("low"); },
                    NotificationUrgency::Normal => { cmd.arg("-u").arg("normal"); },
                    NotificationUrgency::Critical => { cmd.arg("-u").arg("critical"); },
                }
                
                if let Some(icon) = &config.icon {
                    cmd.arg("-i").arg(icon);
                }
                
                cmd.output()
                    .map_err(|e| NativeError::Notification(format!("notify-send failed: {}", e)))?;
            }
            
            #[cfg(target_os = "macos")]
            {
                use std::process::Command;
                
                let script = format!(
                    r#"display notification "{}" with title "{}""#,
                    config.body.replace('"', "\\\""),
                    config.title.replace('"', "\\\"")
                );
                
                Command::new("osascript")
                    .arg("-e")
                    .arg(&script)
                    .output()
                    .map_err(|e| NativeError::Notification(format!("osascript failed: {}", e)))?;
            }
            
            #[cfg(target_os = "windows")]
            {
                log::info!("Notification: {} - {}", config.title, config.body);
            }
            
            log::debug!("Sent notification: {}", config.title);
            Ok(())
        })
        .await
        .map_err(|e| NativeError::Notification(format!("Task join error: {}", e)))?
    }
}

impl<M: SystemMetricsProvider, N: NotificationSender> Default for NativeFeaturesManager<M, N> 
where 
    M: Default,
    N: Default,
{
    fn default() -> Self {
        Self::with_providers(
            NativeFeaturesConfig::default(),
            M::default(),
            N::default(),
        )
    }
}

impl Default for NativeFeaturesManager<DefaultMetricsProvider, DefaultNotificationSender> {
    fn default() -> Self {
        Self::new()
    }
}

// Safe Debug implementation that doesn't expose sensitive data
impl<M: SystemMetricsProvider, N: NotificationSender> std::fmt::Debug for NativeFeaturesManager<M, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeFeaturesManager")
            .field("config", &self.config)
            .field("cached_status", &self.status_cache.read().is_some())
            .field("notification_count", &self.notification_history.read().len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    // Mock implementations for testing
    struct MockMetricsProvider {
        cpu_usage: f32,
        memory_usage: f64,
        disk_usage: f64,
        network_active: bool,
    }

    impl MockMetricsProvider {
        fn new() -> Self {
            Self {
                cpu_usage: 25.0,
                memory_usage: 60.0,
                disk_usage: 80.0,
                network_active: true,
            }
        }
    }

    impl SystemMetricsProvider for MockMetricsProvider {
        async fn get_cpu_usage(&self) -> Result<f32, NativeError> {
            Ok(self.cpu_usage)
        }

        async fn get_memory_usage(&self) -> Result<f64, NativeError> {
            Ok(self.memory_usage)
        }

        async fn get_disk_usage(&self) -> Result<f64, NativeError> {
            Ok(self.disk_usage)
        }

        async fn check_network_activity(&self) -> Result<bool, NativeError> {
            Ok(self.network_active)
        }
    }

    struct MockNotificationSender {
        sent_count: Arc<AtomicU32>,
    }

    impl MockNotificationSender {
        fn new() -> Self {
            Self {
                sent_count: Arc::new(AtomicU32::new(0)),
            }
        }

        fn sent_count(&self) -> u32 {
            self.sent_count.load(Ordering::Relaxed)
        }
    }

    impl NotificationSender for MockNotificationSender {
        async fn send_notification(&self, _config: &NotificationConfig) -> Result<(), NativeError> {
            self.sent_count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_system_status_collection() {
        let manager = NativeFeaturesManager::with_providers(
            NativeFeaturesConfig::default(),
            MockMetricsProvider::new(),
            MockNotificationSender::new(),
        );

        let status = manager.get_system_status().await.unwrap();
        assert_eq!(status.cpu_usage, 25.0);
        assert_eq!(status.memory_usage, 60.0);
        assert_eq!(status.disk_usage, 80.0);
        assert!(status.network_active);
    }

    #[tokio::test]
    async fn test_notification_sending() {
        let mock_sender = MockNotificationSender::new();
        let sent_count = mock_sender.sent_count.clone();
        
        let manager = NativeFeaturesManager::with_providers(
            NativeFeaturesConfig::default(),
            MockMetricsProvider::new(),
            mock_sender,
        );

        let notification = NotificationConfig::new("Test", "Message");
        manager.send_notification(notification).await.unwrap();
        
        assert_eq!(sent_count.load(Ordering::Relaxed), 1);
        assert_eq!(manager.get_notification_history().len(), 1);
    }

    #[tokio::test]
    async fn test_notification_history_limit() {
        let config = NativeFeaturesConfig {
            cache_max_age: Duration::from_secs(30),
            max_notification_history: 2, // Small limit for testing
        };

        let manager = NativeFeaturesManager::with_providers(
            config,
            MockMetricsProvider::new(),
            MockNotificationSender::new(),
        );

        // Send 3 notifications
        for i in 0..3 {
            let notification = NotificationConfig::new(format!("Test {}", i), "Message");
            manager.send_notification(notification).await.unwrap();
        }

        // Should only keep the last 2
        let history = manager.get_notification_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].title, "Test 1");
        assert_eq!(history[1].title, "Test 2");
    }

    #[tokio::test]
    async fn test_cache_staleness() {
        let config = NativeFeaturesConfig {
            cache_max_age: Duration::from_millis(100),
            max_notification_history: 100,
        };

        let manager = NativeFeaturesManager::with_providers(
            config,
            MockMetricsProvider::new(),
            MockNotificationSender::new(),
        );

        // Get status to populate cache
        let status1 = manager.get_system_status().await.unwrap();
        
        // Should get cached version immediately
        let status2 = manager.get_system_status().await.unwrap();
        assert_eq!(status1.last_updated, status2.last_updated);

        // Wait for cache to become stale
        tokio::time::sleep(Duration::from_millis(150)).await;
        
        // Should get fresh status
        let status3 = manager.get_system_status().await.unwrap();
        assert!(status3.last_updated > status1.last_updated);
    }

    #[test]
    fn test_file_system_info_creation() {
        let temp_dir = std::env::temp_dir();
        let info = FileSystemInfo::from_path(&temp_dir).unwrap();
        
        assert!(info.exists);
        assert!(info.is_dir);
        assert!(!info.is_file);
        assert!(info.permissions.readable);
    }

    #[test]
    fn test_notification_config_builder() {
        let notification = NotificationConfig::new("Title", "Body")
            .with_urgency(NotificationUrgency::Critical)
            .with_icon("icon.png");
            
        assert_eq!(notification.title, "Title");
        assert_eq!(notification.body, "Body");
        assert_eq!(notification.icon, Some("icon.png".to_string()));
        assert!(matches!(notification.urgency, NotificationUrgency::Critical));
    }
}