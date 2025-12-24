/**
 * MDMAI Desktop Application - Core Library
 * 
 * This library provides the core functionality for the MDMAI desktop application,
 * including IPC communication, process management, and resource management.
 */

// Public module exports
pub mod ipc;
pub mod process_manager;
pub mod resource_manager;

// Re-export commonly used types
pub use ipc::{
    IpcManager, 
    JsonRpcRequest, 
    JsonRpcResponse, 
    JsonRpcError, 
    JsonRpcNotification,
    PerformanceMetrics,
    QueueConfig,
};

pub use process_manager::{
    ProcessManager,
    ProcessConfig,
    ProcessState,
    ProcessStats,
    ProcessEvent,
    HealthStatus,
    ResourceUsage,
};

pub use resource_manager::{
    ResourceManager,
    ResourceType,
    ResourceInfo,
    ResourceStats,
    ResourceLimits,
};

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Application configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AppConfig {
    pub ipc: QueueConfig,
    pub process: ProcessConfig,
    pub resources: ResourceLimits,
    pub log_level: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        AppConfig {
            ipc: QueueConfig::default(),
            process: ProcessConfig::default(),
            resources: ResourceLimits::default(),
            log_level: "info".to_string(),
        }
    }
}

/// Application result type using error-as-values pattern
pub type AppResult<T> = Result<T, AppError>;

/// Application error types
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AppError {
    IpcError { message: String, code: i32 },
    ProcessError { message: String },
    ResourceError { message: String },
    ConfigError { message: String },
    InternalError { message: String },
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::IpcError { message, code } => write!(f, "IPC Error ({}): {}", code, message),
            AppError::ProcessError { message } => write!(f, "Process Error: {}", message),
            AppError::ResourceError { message } => write!(f, "Resource Error: {}", message),
            AppError::ConfigError { message } => write!(f, "Config Error: {}", message),
            AppError::InternalError { message } => write!(f, "Internal Error: {}", message),
        }
    }
}

impl std::error::Error for AppError {}

impl From<String> for AppError {
    fn from(message: String) -> Self {
        AppError::InternalError { message }
    }
}

impl From<&str> for AppError {
    fn from(message: &str) -> Self {
        AppError::InternalError { message: message.to_string() }
    }
}

/// Utility functions for common operations
pub mod utils {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    
    /// Get current timestamp in seconds
    pub fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
    
    /// Format bytes into human readable string
    pub fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        
        if bytes == 0 {
            return "0 B".to_string();
        }
        
        let mut size = bytes as f64;
        let mut unit_index = 0;
        
        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }
        
        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.1} {}", size, UNITS[unit_index])
        }
    }
    
    /// Format duration into human readable string
    pub fn format_duration(duration: std::time::Duration) -> String {
        let total_seconds = duration.as_secs();
        
        if total_seconds < 60 {
            format!("{}s", total_seconds)
        } else if total_seconds < 3600 {
            let minutes = total_seconds / 60;
            let seconds = total_seconds % 60;
            format!("{}m {}s", minutes, seconds)
        } else if total_seconds < 86400 {
            let hours = total_seconds / 3600;
            let minutes = (total_seconds % 3600) / 60;
            format!("{}h {}m", hours, minutes)
        } else {
            let days = total_seconds / 86400;
            let hours = (total_seconds % 86400) / 3600;
            format!("{}d {}h", days, hours)
        }
    }
    
    /// Validate JSON-RPC method name
    pub fn is_valid_method_name(method: &str) -> bool {
        !method.is_empty() && 
        !method.starts_with("rpc.") && 
        method.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    }
    
    /// Create a JSON-RPC error response
    pub fn create_error_response(id: Option<serde_json::Value>, code: i32, message: String) -> JsonRpcResponse {
        use crate::ipc::RequestId;
        
        let request_id = id.and_then(|v| match v {
            serde_json::Value::Number(n) => {
                n.as_u64().map(RequestId::Number)
            }
            serde_json::Value::String(s) => {
                Some(RequestId::String(s))
            }
            _ => None
        });
        
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: request_id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message,
                data: None,
            }),
        }
    }
}

/// Constants used throughout the application
pub mod constants {
    // JSON-RPC constants
    pub const JSONRPC_VERSION: &str = "2.0";
    
    // Error codes
    pub const ERROR_PARSE: i32 = -32700;
    pub const ERROR_INVALID_REQUEST: i32 = -32600;
    pub const ERROR_METHOD_NOT_FOUND: i32 = -32601;
    pub const ERROR_INVALID_PARAMS: i32 = -32602;
    pub const ERROR_INTERNAL: i32 = -32603;
    pub const ERROR_SERVER: i32 = -32000;
    pub const ERROR_TIMEOUT: i32 = -32001;
    pub const ERROR_CANCELLED: i32 = -32002;
    
    // Default timeouts (milliseconds)
    pub const DEFAULT_REQUEST_TIMEOUT: u64 = 30000;
    pub const DEFAULT_HEALTH_CHECK_TIMEOUT: u64 = 5000;
    pub const DEFAULT_CLEANUP_TIMEOUT: u64 = 10000;
    pub const DEFAULT_SHUTDOWN_TIMEOUT: u64 = 30000;
    
    // Resource limits
    pub const MAX_CONCURRENT_REQUESTS: usize = 50;
    pub const MAX_QUEUE_SIZE: usize = 200;
    pub const MAX_MEMORY_USAGE_MB: u64 = 2048;
    pub const MAX_PROCESSES: u32 = 10;
    pub const MAX_FILE_HANDLES: u32 = 1000;
    
    // Monitoring intervals (milliseconds)
    pub const HEALTH_CHECK_INTERVAL: u64 = 30000;
    pub const METRICS_UPDATE_INTERVAL: u64 = 10000;
    pub const RESOURCE_CLEANUP_INTERVAL: u64 = 60000;
}

/// Logging utilities
pub mod logging {
    /// Initialize logging with the specified level
    pub fn init_logging(level: &str) -> Result<(), String> {
        let log_level = match level.to_lowercase().as_str() {
            "error" => log::LevelFilter::Error,
            "warn" => log::LevelFilter::Warn,
            "info" => log::LevelFilter::Info,
            "debug" => log::LevelFilter::Debug,
            "trace" => log::LevelFilter::Trace,
            _ => log::LevelFilter::Info,
        };
        
        env_logger::Builder::from_default_env()
            .filter_level(log_level)
            .format_timestamp_secs()
            .init();
        
        log::info!("Logging initialized at level: {}", level);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_format_bytes() {
        assert_eq!(utils::format_bytes(0), "0 B");
        assert_eq!(utils::format_bytes(512), "512 B");
        assert_eq!(utils::format_bytes(1024), "1.0 KB");
        assert_eq!(utils::format_bytes(1536), "1.5 KB");
        assert_eq!(utils::format_bytes(1048576), "1.0 MB");
        assert_eq!(utils::format_bytes(1073741824), "1.0 GB");
    }
    
    #[test]
    fn test_format_duration() {
        use std::time::Duration;
        
        assert_eq!(utils::format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(utils::format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(utils::format_duration(Duration::from_secs(3665)), "1h 1m");
        assert_eq!(utils::format_duration(Duration::from_secs(90061)), "1d 1h");
    }
    
    #[test]
    fn test_method_name_validation() {
        assert!(utils::is_valid_method_name("test_method"));
        assert!(utils::is_valid_method_name("test-method"));
        assert!(utils::is_valid_method_name("testMethod123"));
        assert!(!utils::is_valid_method_name(""));
        assert!(!utils::is_valid_method_name("rpc.internal"));
        assert!(!utils::is_valid_method_name("test method")); // space not allowed
    }
    
    #[test]
    fn test_error_response_creation() {
        let response = utils::create_error_response(
            Some(serde_json::json!(42)),
            constants::ERROR_INTERNAL,
            "Test error".to_string()
        );
        
        assert_eq!(response.jsonrpc, "2.0");
        assert!(response.id.is_some());
        assert!(response.result.is_none());
        assert!(response.error.is_some());
        
        let error = response.error.unwrap();
        assert_eq!(error.code, constants::ERROR_INTERNAL);
        assert_eq!(error.message, "Test error");
    }
}