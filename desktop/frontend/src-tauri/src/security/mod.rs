//! Comprehensive Security Management System
//! 
//! This module provides enterprise-grade security features including:
//! - Process sandboxing and isolation
//! - Security audit logging
//! - Permission management
//! - Input validation and sanitization
//! - Security monitoring and alerting
//! - Threat detection

pub mod audit;
pub mod keychain;
pub mod permissions;
pub mod sandbox;
pub mod validation;
pub mod monitoring;
pub mod crypto;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub use audit::*;
pub use keychain::*;
pub use permissions::*;
pub use sandbox::*;
pub use validation::*;
pub use monitoring::*;
pub use crypto::*;

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable security audit logging
    pub audit_logging_enabled: bool,
    /// Maximum log retention days
    pub audit_log_retention_days: u32,
    /// Enable process sandboxing
    pub process_sandboxing_enabled: bool,
    /// Enable input validation
    pub input_validation_enabled: bool,
    /// Enable security monitoring
    pub security_monitoring_enabled: bool,
    /// Monitor resource usage
    pub resource_monitoring_enabled: bool,
    /// Maximum memory usage per subprocess (MB)
    pub max_subprocess_memory_mb: u64,
    /// Maximum CPU usage percentage per subprocess
    pub max_subprocess_cpu_percent: f32,
    /// Network restrictions for subprocesses
    pub subprocess_network_restrictions: NetworkRestrictions,
    /// File system restrictions
    pub filesystem_restrictions: FilesystemRestrictions,
    /// Security alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Enable keychain integration
    pub keychain_integration_enabled: bool,
    /// Allowed commands for validation
    pub allowed_commands: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRestrictions {
    /// Block all network access
    pub block_all_network: bool,
    /// Allowed domains
    pub allowed_domains: Vec<String>,
    /// Allowed IP ranges
    pub allowed_ip_ranges: Vec<String>,
    /// Blocked ports
    pub blocked_ports: Vec<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemRestrictions {
    /// Read-only directories
    pub readonly_paths: Vec<String>,
    /// Completely blocked paths
    pub blocked_paths: Vec<String>,
    /// Allowed paths for write operations
    pub allowed_write_paths: Vec<String>,
    /// Maximum file size for operations (bytes)
    pub max_file_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Memory usage threshold (MB)
    pub memory_threshold_mb: u64,
    /// CPU usage threshold (%)
    pub cpu_threshold_percent: f32,
    /// File operations per second threshold
    pub file_ops_threshold: u32,
    /// Network requests per second threshold
    pub network_ops_threshold: u32,
    /// Failed authentication attempts threshold
    pub failed_auth_threshold: u32,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            audit_logging_enabled: true,
            audit_log_retention_days: 90,
            process_sandboxing_enabled: true,
            input_validation_enabled: true,
            security_monitoring_enabled: true,
            resource_monitoring_enabled: true,
            max_subprocess_memory_mb: 512,
            max_subprocess_cpu_percent: 50.0,
            subprocess_network_restrictions: NetworkRestrictions {
                block_all_network: false,
                allowed_domains: vec![],
                allowed_ip_ranges: vec!["127.0.0.0/8".to_string(), "::1/128".to_string()],
                blocked_ports: vec![22, 23, 25, 110, 143, 993, 995],
            },
            filesystem_restrictions: FilesystemRestrictions {
                readonly_paths: vec![
                    "/etc".to_string(),
                    "/usr".to_string(),
                    "/bin".to_string(),
                    "/sbin".to_string(),
                    "/boot".to_string(),
                ],
                blocked_paths: vec![
                    "/dev".to_string(),
                    "/proc".to_string(),
                    "/sys".to_string(),
                ],
                allowed_write_paths: vec![
                    "$APPDATA".to_string(),
                    "$APPLOCAL".to_string(),
                    "$APPCACHE".to_string(),
                    "$TEMP".to_string(),
                ],
                max_file_size: 100 * 1024 * 1024, // 100MB
            },
            alert_thresholds: AlertThresholds {
                memory_threshold_mb: 1024,
                cpu_threshold_percent: 80.0,
                file_ops_threshold: 100,
                network_ops_threshold: 50,
                failed_auth_threshold: 5,
            },
            keychain_integration_enabled: true,
            allowed_commands: vec![
                "python".to_string(),
                "python3".to_string(),
                "node".to_string(),
                "npm".to_string(),
                "cargo".to_string(),
                "rustc".to_string(),
                "git".to_string(),
                "cat".to_string(),
                "echo".to_string(),
                "ls".to_string(),
                "pwd".to_string(),
                "which".to_string(),
            ],
        }
    }
}

/// Security manager state
pub struct SecurityManager {
    config: Arc<RwLock<SecurityConfig>>,
    audit_logger: Arc<AuditLogger>,
    keychain_manager: Arc<KeychainManager>,
    permission_manager: Arc<PermissionManager>,
    sandbox_manager: Arc<SandboxManager>,
    input_validator: Arc<InputValidator>,
    security_monitor: Arc<SecurityMonitor>,
    crypto_manager: Arc<CryptoManager>,
    security_events: Arc<Mutex<Vec<SecurityEvent>>>,
    active_sessions: Arc<RwLock<HashMap<Uuid, SecuritySession>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    pub id: Uuid,
    pub event_type: SecurityEventType,
    pub severity: SecuritySeverity,
    pub message: String,
    pub details: serde_json::Value,
    pub timestamp: SystemTime,
    pub source_component: String,
    pub session_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityEventType {
    Authentication,
    Authorization,
    ProcessSandboxing,
    InputValidation,
    ResourceUsage,
    NetworkAccess,
    FileSystemAccess,
    CryptographicOperation,
    SecurityPolicyViolation,
    ThreatDetection,
    AuditLog,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecuritySeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecuritySession {
    pub id: Uuid,
    pub created_at: SystemTime,
    pub last_activity: SystemTime,
    pub permissions: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Result type for security operations
pub type SecurityResult<T> = Result<T, SecurityError>;

/// Security error types
#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
pub enum SecurityError {
    #[error("Authentication failed: {message}")]
    AuthenticationFailed { message: String },

    #[error("Authorization denied: {operation}")]
    AuthorizationDenied { operation: String },

    #[error("Input validation failed: {field}: {message}")]
    InputValidationFailed { field: String, message: String },

    #[error("Sandbox violation: {violation}")]
    SandboxViolation { violation: String },

    #[error("Resource limit exceeded: {resource}")]
    ResourceLimitExceeded { resource: String },

    #[error("Security policy violation: {policy}")]
    PolicyViolation { policy: String },

    #[error("Cryptographic error: {message}")]
    CryptographicError { message: String },

    #[error("Keychain error: {message}")]
    KeychainError { message: String },

    #[error("Audit logging error: {message}")]
    AuditError { message: String },

    #[error("Security configuration error: {message}")]
    ConfigurationError { message: String },

    #[error("Internal security error: {message}")]
    InternalError { message: String },
}

impl SecurityManager {
    /// Create new security manager with default configuration
    pub async fn new() -> SecurityResult<Self> {
        Self::with_config(SecurityConfig::default()).await
    }

    /// Create new security manager with custom configuration
    pub async fn with_config(config: SecurityConfig) -> SecurityResult<Self> {
        let config = Arc::new(RwLock::new(config));
        
        // Initialize all security components
        let audit_logger = Arc::new(AuditLogger::new(&config.read().await).await?);
        let keychain_manager = Arc::new(KeychainManager::new(&config.read().await).await?);
        let permission_manager = Arc::new(PermissionManager::new(&config.read().await)?);
        let sandbox_manager = Arc::new(SandboxManager::new(&config.read().await)?);
        let input_validator = Arc::new(InputValidator::new(&config.read().await)?);
        let security_monitor = Arc::new(SecurityMonitor::new(&config.read().await).await?);
        let crypto_manager = Arc::new(CryptoManager::new(&config.read().await)?);

        Ok(Self {
            config,
            audit_logger,
            keychain_manager,
            permission_manager,
            sandbox_manager,
            input_validator,
            security_monitor,
            crypto_manager,
            security_events: Arc::new(Mutex::new(Vec::new())),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Initialize the security manager
    pub async fn initialize(&self) -> SecurityResult<()> {
        log::info!("Initializing security manager");

        // Initialize all components
        self.audit_logger.initialize().await?;
        self.keychain_manager.initialize().await?;
        self.permission_manager.initialize().await?;
        self.sandbox_manager.initialize().await?;
        self.security_monitor.initialize().await?;

        // Log security initialization
        self.log_security_event(
            SecurityEventType::Authentication,
            SecuritySeverity::Medium,
            "Security manager initialized".to_string(),
            serde_json::json!({"component": "security_manager"}),
            None,
        ).await;

        log::info!("Security manager initialized successfully");
        Ok(())
    }

    /// Create a new security session
    pub async fn create_session(&self, permissions: Vec<String>) -> SecurityResult<Uuid> {
        let session_id = Uuid::new_v4();
        let session = SecuritySession {
            id: session_id,
            created_at: SystemTime::now(),
            last_activity: SystemTime::now(),
            permissions,
            metadata: HashMap::new(),
        };

        self.active_sessions.write().await.insert(session_id, session);

        self.log_security_event(
            SecurityEventType::Authentication,
            SecuritySeverity::Low,
            "Security session created".to_string(),
            serde_json::json!({"session_id": session_id}),
            Some(session_id),
        ).await;

        Ok(session_id)
    }

    /// Validate session and check permissions
    pub async fn validate_session_permission(&self, session_id: Uuid, required_permission: &str) -> SecurityResult<bool> {
        let mut sessions = self.active_sessions.write().await;
        
        if let Some(session) = sessions.get_mut(&session_id) {
            // Update last activity
            session.last_activity = SystemTime::now();
            
            // Check permission
            let has_permission = session.permissions.contains(&required_permission.to_string()) 
                || session.permissions.contains(&"*".to_string());

            if !has_permission {
                self.log_security_event(
                    SecurityEventType::Authorization,
                    SecuritySeverity::Medium,
                    format!("Permission denied: {}", required_permission),
                    serde_json::json!({
                        "session_id": session_id,
                        "required_permission": required_permission,
                        "available_permissions": session.permissions
                    }),
                    Some(session_id),
                ).await;

                return Err(SecurityError::AuthorizationDenied {
                    operation: required_permission.to_string(),
                });
            }

            Ok(true)
        } else {
            Err(SecurityError::AuthenticationFailed {
                message: "Invalid or expired session".to_string(),
            })
        }
    }

    /// Log a security event
    pub async fn log_security_event(
        &self,
        event_type: SecurityEventType,
        severity: SecuritySeverity,
        message: String,
        details: serde_json::Value,
        session_id: Option<Uuid>,
    ) {
        let event = SecurityEvent {
            id: Uuid::new_v4(),
            event_type: event_type.clone(),
            severity: severity.clone(),
            message: message.clone(),
            details: details.clone(),
            timestamp: SystemTime::now(),
            source_component: "security_manager".to_string(),
            session_id,
        };

        // Store in memory
        self.security_events.lock().await.push(event.clone());

        // Log to audit system
        if let Err(e) = self.audit_logger.log_security_event(&event).await {
            log::error!("Failed to log security event to audit system: {}", e);
        }

        // Trigger monitoring if critical
        if matches!(severity, SecuritySeverity::Critical | SecuritySeverity::High) {
            if let Err(e) = self.security_monitor.process_security_event(&event).await {
                log::error!("Failed to process security event in monitor: {}", e);
            }
        }

        log::info!(
            "[SECURITY] {} - {} - {} - {}",
            severity_to_string(&severity),
            event_type_to_string(&event_type),
            message,
            serde_json::to_string(&details).unwrap_or_default()
        );
    }

    /// Get security statistics
    pub async fn get_security_stats(&self) -> SecurityStats {
        let events = self.security_events.lock().await;
        let sessions = self.active_sessions.read().await;

        let event_counts = events.iter().fold(HashMap::new(), |mut acc, event| {
            *acc.entry(event_type_to_string(&event.event_type)).or_insert(0) += 1;
            acc
        });

        let severity_counts = events.iter().fold(HashMap::new(), |mut acc, event| {
            *acc.entry(severity_to_string(&event.severity)).or_insert(0) += 1;
            acc
        });

        SecurityStats {
            total_events: events.len(),
            active_sessions: sessions.len(),
            event_counts,
            severity_counts,
            recent_events: events.iter().rev().take(10).cloned().collect(),
        }
    }

    /// Update security configuration
    pub async fn update_config(&self, new_config: SecurityConfig) -> SecurityResult<()> {
        *self.config.write().await = new_config;
        
        self.log_security_event(
            SecurityEventType::SecurityPolicyViolation,
            SecuritySeverity::Medium,
            "Security configuration updated".to_string(),
            serde_json::json!({"component": "security_manager"}),
            None,
        ).await;

        Ok(())
    }

    /// Cleanup expired sessions
    pub async fn cleanup_expired_sessions(&self) -> SecurityResult<()> {
        let mut sessions = self.active_sessions.write().await;
        let now = SystemTime::now();
        const SESSION_TIMEOUT: Duration = Duration::from_secs(24 * 60 * 60); // 24 hours

        let expired_sessions: Vec<Uuid> = sessions
            .iter()
            .filter(|(_, session)| {
                now.duration_since(session.last_activity)
                    .unwrap_or(Duration::ZERO) > SESSION_TIMEOUT
            })
            .map(|(id, _)| *id)
            .collect();

        for session_id in expired_sessions {
            sessions.remove(&session_id);
            
            self.log_security_event(
                SecurityEventType::Authentication,
                SecuritySeverity::Low,
                "Security session expired".to_string(),
                serde_json::json!({"session_id": session_id}),
                Some(session_id),
            ).await;
        }

        Ok(())
    }

    /// Get reference to components for external use
    pub fn audit_logger(&self) -> &Arc<AuditLogger> { &self.audit_logger }
    pub fn keychain_manager(&self) -> &Arc<KeychainManager> { &self.keychain_manager }
    pub fn permission_manager(&self) -> &Arc<PermissionManager> { &self.permission_manager }
    pub fn sandbox_manager(&self) -> &Arc<SandboxManager> { &self.sandbox_manager }
    pub fn input_validator(&self) -> &Arc<InputValidator> { &self.input_validator }
    pub fn security_monitor(&self) -> &Arc<SecurityMonitor> { &self.security_monitor }
    pub fn crypto_manager(&self) -> &Arc<CryptoManager> { &self.crypto_manager }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityStats {
    pub total_events: usize,
    pub active_sessions: usize,
    pub event_counts: HashMap<String, u32>,
    pub severity_counts: HashMap<String, u32>,
    pub recent_events: Vec<SecurityEvent>,
}

// Helper functions
fn event_type_to_string(event_type: &SecurityEventType) -> String {
    match event_type {
        SecurityEventType::Authentication => "Authentication".to_string(),
        SecurityEventType::Authorization => "Authorization".to_string(),
        SecurityEventType::ProcessSandboxing => "ProcessSandboxing".to_string(),
        SecurityEventType::InputValidation => "InputValidation".to_string(),
        SecurityEventType::ResourceUsage => "ResourceUsage".to_string(),
        SecurityEventType::NetworkAccess => "NetworkAccess".to_string(),
        SecurityEventType::FileSystemAccess => "FileSystemAccess".to_string(),
        SecurityEventType::CryptographicOperation => "CryptographicOperation".to_string(),
        SecurityEventType::SecurityPolicyViolation => "SecurityPolicyViolation".to_string(),
        SecurityEventType::ThreatDetection => "ThreatDetection".to_string(),
        SecurityEventType::AuditLog => "AuditLog".to_string(),
    }
}

fn severity_to_string(severity: &SecuritySeverity) -> String {
    match severity {
        SecuritySeverity::Low => "Low".to_string(),
        SecuritySeverity::Medium => "Medium".to_string(),
        SecuritySeverity::High => "High".to_string(),
        SecuritySeverity::Critical => "Critical".to_string(),
    }
}

/// Constants for security operations
pub mod constants {
    pub const MAX_INPUT_LENGTH: usize = 1024 * 1024; // 1MB
    pub const MAX_COMMAND_ARGS: usize = 100;
    pub const MAX_ENVIRONMENT_VARS: usize = 100;
    pub const MAX_FILE_DESCRIPTORS: u32 = 1024;
    pub const MAX_OPEN_FILES: u32 = 100;
    pub const SECURITY_EVENT_BUFFER_SIZE: usize = 10000;
    pub const SESSION_CLEANUP_INTERVAL: u64 = 3600; // 1 hour
    pub const AUDIT_LOG_ROTATION_SIZE: u64 = 100 * 1024 * 1024; // 100MB
}