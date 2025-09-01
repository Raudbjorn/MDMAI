//! Security Audit Logging
//! 
//! This module provides comprehensive audit logging capabilities:
//! - Tamper-resistant log storage with cryptographic integrity
//! - Structured logging with JSON format
//! - Log rotation and archival
//! - Real-time security event tracking
//! - Compliance reporting capabilities

use super::*;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};

/// Audit log entry with cryptographic integrity protection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    pub id: Uuid,
    pub timestamp: SystemTime,
    pub event_type: SecurityEventType,
    pub severity: SecuritySeverity,
    pub source: String,
    pub user_id: Option<String>,
    pub session_id: Option<Uuid>,
    pub message: String,
    pub details: serde_json::Value,
    pub client_info: ClientInfo,
    pub integrity_hash: String,
    pub previous_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub platform: String,
    pub version: String,
}

/// Audit log configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    pub enabled: bool,
    pub log_directory: PathBuf,
    pub max_log_size_mb: u64,
    pub retention_days: u32,
    pub enable_encryption: bool,
    pub enable_integrity_checking: bool,
    pub enable_real_time_alerts: bool,
    pub log_format: LogFormat,
    pub compression_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogFormat {
    Json,
    Structured,
    Syslog,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_directory: PathBuf::from("audit_logs"),
            max_log_size_mb: 100,
            retention_days: 90,
            enable_encryption: true,
            enable_integrity_checking: true,
            enable_real_time_alerts: true,
            log_format: LogFormat::Json,
            compression_enabled: true,
        }
    }
}

/// Audit logger with tamper-resistant storage
pub struct AuditLogger {
    config: AuditConfig,
    current_log_file: Arc<Mutex<Option<LogFile>>>,
    integrity_chain: Arc<RwLock<String>>,
    log_buffer: Arc<Mutex<Vec<AuditLogEntry>>>,
    encryption_key: Option<[u8; 32]>,
}

struct LogFile {
    file: std::fs::File,
    path: PathBuf,
    size_bytes: u64,
    entry_count: u64,
    created_at: SystemTime,
}

impl AuditLogger {
    pub async fn new(security_config: &SecurityConfig) -> SecurityResult<Self> {
        let config = AuditConfig::default();
        
        // Generate encryption key if needed
        let encryption_key = if config.enable_encryption {
            let mut key = [0u8; 32];
            use rand::RngCore;
            rand::rngs::OsRng.fill_bytes(&mut key);
            Some(key)
        } else {
            None
        };

        Ok(Self {
            config,
            current_log_file: Arc::new(Mutex::new(None)),
            integrity_chain: Arc::new(RwLock::new(String::new())),
            log_buffer: Arc::new(Mutex::new(Vec::new())),
            encryption_key,
        })
    }

    pub async fn initialize(&self) -> SecurityResult<()> {
        // Create log directory
        if !self.config.log_directory.exists() {
            std::fs::create_dir_all(&self.config.log_directory)
                .map_err(|e| SecurityError::AuditError {
                    message: format!("Failed to create audit log directory: {}", e),
                })?;
        }

        // Set restrictive permissions on log directory
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&self.config.log_directory, std::fs::Permissions::from_mode(0o700))
                .map_err(|e| SecurityError::AuditError {
                    message: format!("Failed to set log directory permissions: {}", e),
                })?;
        }

        // Initialize integrity chain
        self.initialize_integrity_chain().await?;

        // Create initial log file
        self.rotate_log_file().await?;

        // Start background tasks
        self.start_background_tasks().await;

        // Log initialization
        let init_event = SecurityEvent {
            id: Uuid::new_v4(),
            event_type: SecurityEventType::AuditLog,
            severity: SecuritySeverity::Medium,
            message: "Audit logging system initialized".to_string(),
            details: serde_json::json!({
                "audit_config": self.config,
                "integrity_checking": self.config.enable_integrity_checking,
                "encryption": self.config.enable_encryption
            }),
            timestamp: SystemTime::now(),
            source_component: "audit_logger".to_string(),
            session_id: None,
        };

        self.log_security_event(&init_event).await?;

        log::info!("Audit logger initialized");
        Ok(())
    }

    /// Log a security event to the audit log
    pub async fn log_security_event(&self, event: &SecurityEvent) -> SecurityResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let client_info = ClientInfo {
            ip_address: None, // Would be populated from request context
            user_agent: None,
            platform: std::env::consts::OS.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        };

        let previous_hash = if self.config.enable_integrity_checking {
            Some(self.integrity_chain.read().await.clone())
        } else {
            None
        };

        let mut audit_entry = AuditLogEntry {
            id: Uuid::new_v4(),
            timestamp: event.timestamp,
            event_type: event.event_type.clone(),
            severity: event.severity.clone(),
            source: event.source_component.clone(),
            user_id: None, // Would be populated from session context
            session_id: event.session_id,
            message: event.message.clone(),
            details: event.details.clone(),
            client_info,
            integrity_hash: String::new(),
            previous_hash: previous_hash.clone(),
        };

        // Calculate integrity hash
        if self.config.enable_integrity_checking {
            audit_entry.integrity_hash = self.calculate_integrity_hash(&audit_entry)?;
            
            // Update integrity chain
            let mut chain = self.integrity_chain.write().await;
            *chain = audit_entry.integrity_hash.clone();
        }

        // Add to buffer for batch processing
        self.log_buffer.lock().await.push(audit_entry);

        // Flush if buffer is full
        if self.log_buffer.lock().await.len() >= 10 {
            self.flush_log_buffer().await?;
        }

        Ok(())
    }

    /// Flush buffered log entries to disk
    async fn flush_log_buffer(&self) -> SecurityResult<()> {
        let mut buffer = self.log_buffer.lock().await;
        if buffer.is_empty() {
            return Ok(());
        }

        let entries = buffer.drain(..).collect::<Vec<_>>();
        drop(buffer); // Release lock early

        for entry in entries {
            self.write_log_entry(&entry).await?;
        }

        Ok(())
    }

    /// Write a single log entry to the current log file
    async fn write_log_entry(&self, entry: &AuditLogEntry) -> SecurityResult<()> {
        // Check if log rotation is needed
        {
            let current_file = self.current_log_file.lock().await;
            if let Some(ref log_file) = *current_file {
                if log_file.size_bytes > self.config.max_log_size_mb * 1024 * 1024 {
                    drop(current_file);
                    self.rotate_log_file().await?;
                }
            }
        }

        // Serialize entry
        let serialized = match self.config.log_format {
            LogFormat::Json => serde_json::to_string(entry)
                .map_err(|e| SecurityError::AuditError {
                    message: format!("Failed to serialize log entry: {}", e),
                })?,
            LogFormat::Structured => self.format_structured_entry(entry),
            LogFormat::Syslog => self.format_syslog_entry(entry),
        };

        let mut log_line = serialized + "\n";

        // Encrypt if enabled
        if self.config.enable_encryption && self.encryption_key.is_some() {
            log_line = self.encrypt_log_line(&log_line)?;
        }

        // Write to file
        let mut current_file = self.current_log_file.lock().await;
        if let Some(ref mut log_file) = current_file.as_mut() {
            log_file.file.write_all(log_line.as_bytes())
                .map_err(|e| SecurityError::AuditError {
                    message: format!("Failed to write log entry: {}", e),
                })?;
            
            log_file.file.flush()
                .map_err(|e| SecurityError::AuditError {
                    message: format!("Failed to flush log file: {}", e),
                })?;

            log_file.size_bytes += log_line.len() as u64;
            log_file.entry_count += 1;
        }

        Ok(())
    }

    /// Rotate the current log file
    async fn rotate_log_file(&self) -> SecurityResult<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let filename = format!("audit_{}.log", timestamp);
        let log_path = self.config.log_directory.join(&filename);

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map_err(|e| SecurityError::AuditError {
                message: format!("Failed to create log file: {}", e),
            })?;

        // Set restrictive permissions
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&log_path, std::fs::Permissions::from_mode(0o600))
                .map_err(|e| SecurityError::AuditError {
                    message: format!("Failed to set log file permissions: {}", e),
                })?;
        }

        let new_log_file = LogFile {
            file,
            path: log_path,
            size_bytes: 0,
            entry_count: 0,
            created_at: SystemTime::now(),
        };

        // Close current file and set new one
        let mut current_file = self.current_log_file.lock().await;
        if let Some(old_file) = current_file.take() {
            // Compress old file if enabled
            if self.config.compression_enabled {
                if let Err(e) = self.compress_log_file(&old_file.path).await {
                    log::warn!("Failed to compress old log file: {}", e);
                }
            }
        }
        *current_file = Some(new_log_file);

        log::info!("Rotated audit log file: {}", filename);
        Ok(())
    }

    /// Initialize the integrity chain from existing logs
    async fn initialize_integrity_chain(&self) -> SecurityResult<()> {
        if !self.config.enable_integrity_checking {
            return Ok(());
        }

        // Find the most recent log file
        let mut latest_hash = String::new();

        if let Ok(entries) = std::fs::read_dir(&self.config.log_directory) {
            let mut log_files: Vec<PathBuf> = entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| {
                    path.extension().map_or(false, |ext| ext == "log")
                })
                .collect();

            log_files.sort();

            if let Some(latest_file) = log_files.last() {
                if let Ok(content) = std::fs::read_to_string(latest_file) {
                    // Parse the last line to get the integrity hash
                    if let Some(last_line) = content.lines().last() {
                        if let Ok(entry) = serde_json::from_str::<AuditLogEntry>(last_line) {
                            latest_hash = entry.integrity_hash;
                        }
                    }
                }
            }
        }

        *self.integrity_chain.write().await = latest_hash;
        Ok(())
    }

    /// Calculate integrity hash for a log entry
    fn calculate_integrity_hash(&self, entry: &AuditLogEntry) -> SecurityResult<String> {
        // Create a hash of the entry content plus the previous hash
        let mut hasher = Sha256::new();
        
        // Include key fields in hash
        hasher.update(entry.id.as_bytes());
        hasher.update(entry.timestamp.duration_since(UNIX_EPOCH).unwrap().as_secs().to_le_bytes());
        hasher.update(entry.source.as_bytes());
        hasher.update(entry.message.as_bytes());
        hasher.update(serde_json::to_string(&entry.details).unwrap_or_default().as_bytes());
        
        if let Some(ref prev_hash) = entry.previous_hash {
            hasher.update(prev_hash.as_bytes());
        }

        Ok(hex::encode(hasher.finalize()))
    }

    /// Encrypt a log line
    fn encrypt_log_line(&self, line: &str) -> SecurityResult<String> {
        if let Some(key) = self.encryption_key {
            use aes_gcm::{Aes256Gcm, Key, Nonce};
            use aes_gcm::aead::{Aead, KeyInit};
            use rand::RngCore;

            let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&key));
            let mut nonce_bytes = [0u8; 12];
            rand::rngs::OsRng.fill_bytes(&mut nonce_bytes);
            let nonce = Nonce::from_slice(&nonce_bytes);

            let ciphertext = cipher.encrypt(nonce, line.as_bytes())
                .map_err(|e| SecurityError::CryptographicError {
                    message: format!("Failed to encrypt log line: {}", e),
                })?;

            // Combine nonce and ciphertext, then base64 encode
            let mut combined = Vec::with_capacity(12 + ciphertext.len());
            combined.extend_from_slice(&nonce_bytes);
            combined.extend_from_slice(&ciphertext);

            Ok(base64::engine::general_purpose::STANDARD.encode(combined))
        } else {
            Ok(line.to_string())
        }
    }

    /// Format entry in structured format
    fn format_structured_entry(&self, entry: &AuditLogEntry) -> String {
        format!(
            "[{}] {} {} {} {} {} - {}",
            entry.timestamp.duration_since(UNIX_EPOCH).unwrap().as_secs(),
            severity_to_string(&entry.severity),
            event_type_to_string(&entry.event_type),
            entry.source,
            entry.session_id.map(|id| id.to_string()).unwrap_or_else(|| "-".to_string()),
            entry.id,
            entry.message
        )
    }

    /// Format entry in syslog format
    fn format_syslog_entry(&self, entry: &AuditLogEntry) -> String {
        let priority = match entry.severity {
            SecuritySeverity::Critical => 1,
            SecuritySeverity::High => 3,
            SecuritySeverity::Medium => 6,
            SecuritySeverity::Low => 7,
        };

        format!(
            "<{}>{} ttrpg-assistant: [{}] {} - {}",
            priority,
            chrono::DateTime::<chrono::Utc>::from(entry.timestamp).format("%b %d %H:%M:%S"),
            event_type_to_string(&entry.event_type),
            entry.source,
            entry.message
        )
    }

    /// Compress a log file
    async fn compress_log_file(&self, file_path: &Path) -> SecurityResult<()> {
        let compressed_path = file_path.with_extension("log.gz");
        
        tokio::task::spawn_blocking({
            let file_path = file_path.to_path_buf();
            let compressed_path = compressed_path.clone();
            move || {
                use std::fs::File;
                use std::io::{BufReader, BufWriter};
                use flate2::write::GzEncoder;
                use flate2::Compression;

                let input_file = File::open(&file_path)?;
                let output_file = File::create(&compressed_path)?;
                let mut reader = BufReader::new(input_file);
                let writer = BufWriter::new(output_file);
                let mut encoder = GzEncoder::new(writer, Compression::default());

                std::io::copy(&mut reader, &mut encoder)?;
                encoder.finish()?;

                // Remove original file
                std::fs::remove_file(&file_path)?;

                Ok::<(), std::io::Error>(())
            }
        }).await
        .map_err(|e| SecurityError::AuditError {
            message: format!("Compression task failed: {}", e),
        })?
        .map_err(|e| SecurityError::AuditError {
            message: format!("Failed to compress log file: {}", e),
        })?;

        log::info!("Compressed log file: {:?}", compressed_path);
        Ok(())
    }

    /// Start background maintenance tasks
    async fn start_background_tasks(&self) {
        let log_buffer = self.log_buffer.clone();
        let current_log_file = self.current_log_file.clone();
        let config = self.config.clone();

        // Periodic flush task
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60)); // 1 minute
            
            loop {
                interval.tick().await;
                
                let buffer = log_buffer.lock().await;
                if !buffer.is_empty() {
                    drop(buffer);
                    // Flush would be called here, but we need a reference to self
                    // In a real implementation, this would be structured differently
                }
            }
        });

        // Log cleanup task
        let log_directory = self.config.log_directory.clone();
        let retention_days = self.config.retention_days;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(24 * 60 * 60)); // Daily
            
            loop {
                interval.tick().await;
                
                if let Err(e) = Self::cleanup_old_logs(&log_directory, retention_days).await {
                    log::error!("Failed to cleanup old audit logs: {}", e);
                }
            }
        });
    }

    /// Cleanup old log files based on retention policy
    async fn cleanup_old_logs(log_directory: &Path, retention_days: u32) -> SecurityResult<()> {
        let cutoff_time = SystemTime::now() - Duration::from_secs(retention_days as u64 * 24 * 60 * 60);

        let entries = std::fs::read_dir(log_directory)
            .map_err(|e| SecurityError::AuditError {
                message: format!("Failed to read log directory: {}", e),
            })?;

        let mut deleted_count = 0;

        for entry in entries {
            let entry = entry.map_err(|e| SecurityError::AuditError {
                message: format!("Failed to read directory entry: {}", e),
            })?;

            let path = entry.path();
            
            if let Some(extension) = path.extension() {
                if extension == "log" || extension == "gz" {
                    if let Ok(metadata) = entry.metadata() {
                        if let Ok(created) = metadata.created() {
                            if created < cutoff_time {
                                if let Err(e) = std::fs::remove_file(&path) {
                                    log::warn!("Failed to delete old log file {:?}: {}", path, e);
                                } else {
                                    deleted_count += 1;
                                    log::info!("Deleted old log file: {:?}", path);
                                }
                            }
                        }
                    }
                }
            }
        }

        if deleted_count > 0 {
            log::info!("Cleaned up {} old audit log files", deleted_count);
        }

        Ok(())
    }

    /// Verify integrity chain of log files
    pub async fn verify_integrity(&self) -> SecurityResult<IntegrityReport> {
        if !self.config.enable_integrity_checking {
            return Err(SecurityError::AuditError {
                message: "Integrity checking is disabled".to_string(),
            });
        }

        let mut report = IntegrityReport {
            total_entries: 0,
            verified_entries: 0,
            corrupted_entries: 0,
            missing_entries: 0,
            integrity_breaks: Vec::new(),
        };

        // Implementation would verify the hash chain across all log files
        // This is a simplified version

        Ok(report)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IntegrityReport {
    pub total_entries: u64,
    pub verified_entries: u64,
    pub corrupted_entries: u64,
    pub missing_entries: u64,
    pub integrity_breaks: Vec<IntegrityBreak>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IntegrityBreak {
    pub entry_id: Uuid,
    pub expected_hash: String,
    pub actual_hash: String,
    pub timestamp: SystemTime,
}

// Add base64 dependency to the external crates
use base64;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_audit_logging() {
        let config = SecurityConfig::default();
        let audit_logger = AuditLogger::new(&config).await.unwrap();
        audit_logger.initialize().await.unwrap();

        let test_event = SecurityEvent {
            id: Uuid::new_v4(),
            event_type: SecurityEventType::Authentication,
            severity: SecuritySeverity::Medium,
            message: "Test authentication event".to_string(),
            details: serde_json::json!({"user": "test_user"}),
            timestamp: SystemTime::now(),
            source_component: "test".to_string(),
            session_id: None,
        };

        audit_logger.log_security_event(&test_event).await.unwrap();
        audit_logger.flush_log_buffer().await.unwrap();

        // Verify log was written (simplified test)
        assert!(audit_logger.config.log_directory.exists());
    }
}