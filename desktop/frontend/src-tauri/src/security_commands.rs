//! Security Command Handlers for Tauri
//! 
//! This module provides Tauri command handlers for security operations:
//! - Session management
//! - Permission checks
//! - Input validation
//! - Security monitoring
//! - Keychain operations

use crate::security::{
    SecurityManager, SecurityResult, SecurityError, SecurityEvent, SecurityEventType, 
    SecuritySeverity, ValidationRequest, PermissionRequest, CredentialData
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tauri::{State, Manager};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Security manager state wrapper for Tauri
pub struct SecurityManagerState {
    pub security_manager: Arc<RwLock<Option<SecurityManager>>>,
}

impl SecurityManagerState {
    pub fn new() -> Self {
        Self {
            security_manager: Arc::new(RwLock::new(None)),
        }
    }
}

/// Initialize the security manager
#[tauri::command]
pub async fn initialize_security_manager(
    state: State<'_, SecurityManagerState>,
) -> Result<(), String> {
    log::info!("Initializing security manager");
    
    let security_manager = SecurityManager::new().await
        .map_err(|e| format!("Failed to create security manager: {}", e))?;
    
    security_manager.initialize().await
        .map_err(|e| format!("Failed to initialize security manager: {}", e))?;
    
    *state.security_manager.write().await = Some(security_manager);
    
    log::info!("Security manager initialized successfully");
    Ok(())
}

/// Create a new security session
#[tauri::command]
pub async fn create_security_session(
    permissions: Vec<String>,
    state: State<'_, SecurityManagerState>,
) -> Result<String, String> {
    let manager_guard = state.security_manager.read().await;
    let manager = manager_guard.as_ref()
        .ok_or("Security manager not initialized")?;
    
    let session_id = manager.create_session(permissions).await
        .map_err(|e| format!("Failed to create session: {}", e))?;
    
    Ok(session_id.to_string())
}

/// Validate session permission
#[tauri::command]
pub async fn validate_session_permission(
    session_id: String,
    permission: String,
    state: State<'_, SecurityManagerState>,
) -> Result<bool, String> {
    let manager_guard = state.security_manager.read().await;
    let manager = manager_guard.as_ref()
        .ok_or("Security manager not initialized")?;
    
    let session_uuid = Uuid::parse_str(&session_id)
        .map_err(|e| format!("Invalid session ID: {}", e))?;
    
    match manager.validate_session_permission(session_uuid, &permission).await {
        Ok(valid) => Ok(valid),
        Err(SecurityError::AuthorizationDenied { operation }) => {
            log::warn!("Permission denied: {}", operation);
            Ok(false)
        },
        Err(e) => Err(format!("Permission validation failed: {}", e)),
    }
}

/// Validate user input
#[tauri::command]
pub async fn validate_input(
    field_name: String,
    value: serde_json::Value,
    context: HashMap<String, String>,
    state: State<'_, SecurityManagerState>,
) -> Result<serde_json::Value, String> {
    let manager_guard = state.security_manager.read().await;
    let manager = manager_guard.as_ref()
        .ok_or("Security manager not initialized")?;
    
    let request = ValidationRequest {
        field_name,
        value,
        context,
    };
    
    let result = manager.input_validator().validate_input(&request).await
        .map_err(|e| format!("Input validation failed: {}", e))?;
    
    if !result.valid {
        let error_messages: Vec<String> = result.errors.iter()
            .map(|e| e.message.clone())
            .collect();
        return Err(format!("Validation failed: {}", error_messages.join("; ")));
    }
    
    Ok(serde_json::json!({
        "valid": result.valid,
        "sanitized_value": result.sanitized_value,
        "warnings": result.warnings
    }))
}

/// Sanitize a string input
#[tauri::command]
pub async fn sanitize_string(
    input: String,
    state: State<'_, SecurityManagerState>,
) -> Result<String, String> {
    let manager_guard = state.security_manager.read().await;
    let manager = manager_guard.as_ref()
        .ok_or("Security manager not initialized")?;
    
    let sanitized = manager.input_validator().sanitize_string(&input).await;
    Ok(sanitized)
}

/// Validate file path
#[tauri::command]
pub async fn validate_file_path(
    path: String,
    state: State<'_, SecurityManagerState>,
) -> Result<serde_json::Value, String> {
    let manager_guard = state.security_manager.read().await;
    let manager = manager_guard.as_ref()
        .ok_or("Security manager not initialized")?;
    
    let result = manager.input_validator().validate_path(&path).await
        .map_err(|e| format!("Path validation failed: {}", e))?;
    
    Ok(serde_json::json!({
        "valid": result.valid,
        "sanitized_path": result.sanitized_value,
        "errors": result.errors,
        "warnings": result.warnings
    }))
}

/// Store credential in keychain
#[tauri::command]
pub async fn store_credential(
    service: String,
    account: String,
    secret: String,
    additional_data: HashMap<String, String>,
    expires_at: Option<u64>, // Unix timestamp
    description: Option<String>,
    state: State<'_, SecurityManagerState>,
) -> Result<String, String> {
    let manager_guard = state.security_manager.read().await;
    let manager = manager_guard.as_ref()
        .ok_or("Security manager not initialized")?;
    
    let credential = CredentialData {
        secret,
        additional_data,
        expires_at: expires_at.map(|ts| SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(ts)),
    };
    
    let entry_id = manager.keychain_manager()
        .store_credential(&service, &account, &credential, description.as_deref()).await
        .map_err(|e| format!("Failed to store credential: {}", e))?;
    
    Ok(entry_id)
}

/// Retrieve credential from keychain
#[tauri::command]
pub async fn retrieve_credential(
    service: String,
    account: String,
    state: State<'_, SecurityManagerState>,
) -> Result<serde_json::Value, String> {
    let manager_guard = state.security_manager.read().await;
    let manager = manager_guard.as_ref()
        .ok_or("Security manager not initialized")?;
    
    let credential = manager.keychain_manager()
        .retrieve_credential(&service, &account).await
        .map_err(|e| format!("Failed to retrieve credential: {}", e))?;
    
    Ok(serde_json::json!({
        "secret": credential.secret,
        "additional_data": credential.additional_data,
        "expires_at": credential.expires_at.map(|ts| 
            ts.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs())
    }))
}

/// Delete credential from keychain
#[tauri::command]
pub async fn delete_credential(
    service: String,
    account: String,
    state: State<'_, SecurityManagerState>,
) -> Result<(), String> {
    let manager_guard = state.security_manager.read().await;
    let manager = manager_guard.as_ref()
        .ok_or("Security manager not initialized")?;
    
    manager.keychain_manager()
        .delete_credential(&service, &account).await
        .map_err(|e| format!("Failed to delete credential: {}", e))
}

/// Check permission for a specific operation
#[tauri::command]
pub async fn check_permission(
    user_id: String,
    resource_id: String,
    action: String,
    context: HashMap<String, serde_json::Value>,
    state: State<'_, SecurityManagerState>,
) -> Result<serde_json::Value, String> {
    let manager_guard = state.security_manager.read().await;
    let manager = manager_guard.as_ref()
        .ok_or("Security manager not initialized")?;
    
    let request = PermissionRequest {
        user_id,
        resource_id,
        action,
        context,
    };
    
    let result = manager.permission_manager().check_permission(&request).await
        .map_err(|e| format!("Permission check failed: {}", e))?;
    
    Ok(serde_json::json!(result))
}

/// Get security statistics
#[tauri::command]
pub async fn get_security_stats(
    state: State<'_, SecurityManagerState>,
) -> Result<serde_json::Value, String> {
    let manager_guard = state.security_manager.read().await;
    let manager = manager_guard.as_ref()
        .ok_or("Security manager not initialized")?;
    
    let stats = manager.get_security_stats().await;
    Ok(serde_json::to_value(stats).unwrap_or_default())
}

/// Get recent security alerts
#[tauri::command]
pub async fn get_security_alerts(
    limit: Option<usize>,
    state: State<'_, SecurityManagerState>,
) -> Result<serde_json::Value, String> {
    let manager_guard = state.security_manager.read().await;
    let manager = manager_guard.as_ref()
        .ok_or("Security manager not initialized")?;
    
    let alerts = manager.security_monitor().get_recent_alerts(limit.unwrap_or(50)).await;
    Ok(serde_json::to_value(alerts).unwrap_or_default())
}

/// Generate secure random string
#[tauri::command]
pub async fn generate_secure_random(
    length: usize,
    state: State<'_, SecurityManagerState>,
) -> Result<String, String> {
    let manager_guard = state.security_manager.read().await;
    let manager = manager_guard.as_ref()
        .ok_or("Security manager not initialized")?;
    
    if length > 1024 {
        return Err("Length too large (max 1024 bytes)".to_string());
    }
    
    let random_string = manager.crypto_manager().generate_random_string(length);
    Ok(random_string)
}

/// Hash data using various algorithms
#[tauri::command]
pub async fn hash_data(
    data: String,
    algorithm: Option<String>,
    state: State<'_, SecurityManagerState>,
) -> Result<serde_json::Value, String> {
    let manager_guard = state.security_manager.read().await;
    let manager = manager_guard.as_ref()
        .ok_or("Security manager not initialized")?;
    
    let data_bytes = data.as_bytes();
    let algo = algorithm.as_deref().unwrap_or("blake3");
    
    let result = match algo.to_lowercase().as_str() {
        "sha256" => manager.crypto_manager().hash_sha256(data_bytes),
        "sha512" => manager.crypto_manager().hash_sha512(data_bytes),
        "blake3" => manager.crypto_manager().hash_blake3(data_bytes),
        _ => return Err("Unsupported hash algorithm".to_string()),
    };
    
    Ok(serde_json::to_value(result).unwrap_or_default())
}

/// Create sandboxed process
#[tauri::command]
pub async fn create_sandboxed_process(
    command: String,
    args: Vec<String>,
    working_dir: Option<String>,
    session_id: String,
    state: State<'_, SecurityManagerState>,
) -> Result<String, String> {
    let manager_guard = state.security_manager.read().await;
    let manager = manager_guard.as_ref()
        .ok_or("Security manager not initialized")?;
    
    // Validate session permission
    let session_uuid = Uuid::parse_str(&session_id)
        .map_err(|e| format!("Invalid session ID: {}", e))?;
    
    manager.validate_session_permission(session_uuid, "process.create").await
        .map_err(|e| format!("Permission denied: {}", e))?;
    
    // Validate command and arguments
    let validation_result = manager.input_validator()
        .validate_command(&command, &args).await
        .map_err(|e| format!("Command validation failed: {}", e))?;
    
    if !validation_result.valid {
        let error_messages: Vec<String> = validation_result.errors.iter()
            .map(|e| e.message.clone())
            .collect();
        return Err(format!("Command validation failed: {}", error_messages.join("; ")));
    }
    
    // Create sandboxed process
    let process_id = manager.sandbox_manager()
        .create_sandboxed_process(&command, &args, working_dir.as_deref(), None).await
        .map_err(|e| format!("Failed to create sandboxed process: {}", e))?;
    
    // Log security event
    manager.log_security_event(
        SecurityEventType::ProcessSandboxing,
        SecuritySeverity::Medium,
        format!("Created sandboxed process: {} with args: {:?}", command, args),
        serde_json::json!({
            "command": command,
            "args": args,
            "working_dir": working_dir,
            "process_id": process_id
        }),
        Some(session_uuid),
    ).await;
    
    Ok(process_id.to_string())
}

/// Terminate sandboxed process
#[tauri::command]
pub async fn terminate_sandboxed_process(
    process_id: String,
    session_id: String,
    state: State<'_, SecurityManagerState>,
) -> Result<(), String> {
    let manager_guard = state.security_manager.read().await;
    let manager = manager_guard.as_ref()
        .ok_or("Security manager not initialized")?;
    
    // Validate session permission
    let session_uuid = Uuid::parse_str(&session_id)
        .map_err(|e| format!("Invalid session ID: {}", e))?;
    
    manager.validate_session_permission(session_uuid, "process.terminate").await
        .map_err(|e| format!("Permission denied: {}", e))?;
    
    let process_uuid = Uuid::parse_str(&process_id)
        .map_err(|e| format!("Invalid process ID: {}", e))?;
    
    manager.sandbox_manager()
        .terminate_process(process_uuid).await
        .map_err(|e| format!("Failed to terminate process: {}", e))?;
    
    // Log security event
    manager.log_security_event(
        SecurityEventType::ProcessSandboxing,
        SecuritySeverity::Medium,
        format!("Terminated sandboxed process: {}", process_id),
        serde_json::json!({"process_id": process_id}),
        Some(session_uuid),
    ).await;
    
    Ok(())
}

/// Get sandbox process status
#[tauri::command]
pub async fn get_process_status(
    process_id: String,
    session_id: String,
    state: State<'_, SecurityManagerState>,
) -> Result<serde_json::Value, String> {
    let manager_guard = state.security_manager.read().await;
    let manager = manager_guard.as_ref()
        .ok_or("Security manager not initialized")?;
    
    // Validate session permission
    let session_uuid = Uuid::parse_str(&session_id)
        .map_err(|e| format!("Invalid session ID: {}", e))?;
    
    manager.validate_session_permission(session_uuid, "process.read").await
        .map_err(|e| format!("Permission denied: {}", e))?;
    
    let process_uuid = Uuid::parse_str(&process_id)
        .map_err(|e| format!("Invalid process ID: {}", e))?;
    
    let status = manager.sandbox_manager()
        .get_process_status(process_uuid).await
        .map_err(|e| format!("Failed to get process status: {}", e))?;
    
    Ok(serde_json::to_value(status).unwrap_or_default())
}

/// Log a custom security event
#[tauri::command]
pub async fn log_security_event(
    event_type: String,
    severity: String,
    message: String,
    details: serde_json::Value,
    session_id: Option<String>,
    state: State<'_, SecurityManagerState>,
) -> Result<(), String> {
    let manager_guard = state.security_manager.read().await;
    let manager = manager_guard.as_ref()
        .ok_or("Security manager not initialized")?;
    
    // Parse event type and severity
    let event_type = match event_type.as_str() {
        "Authentication" => SecurityEventType::Authentication,
        "Authorization" => SecurityEventType::Authorization,
        "InputValidation" => SecurityEventType::InputValidation,
        "ProcessSandboxing" => SecurityEventType::ProcessSandboxing,
        "ResourceUsage" => SecurityEventType::ResourceUsage,
        "NetworkAccess" => SecurityEventType::NetworkAccess,
        "FileSystemAccess" => SecurityEventType::FileSystemAccess,
        "CryptographicOperation" => SecurityEventType::CryptographicOperation,
        "SecurityPolicyViolation" => SecurityEventType::SecurityPolicyViolation,
        "ThreatDetection" => SecurityEventType::ThreatDetection,
        "AuditLog" => SecurityEventType::AuditLog,
        _ => return Err("Invalid event type".to_string()),
    };
    
    let security_severity = match severity.as_str() {
        "Low" => SecuritySeverity::Low,
        "Medium" => SecuritySeverity::Medium,
        "High" => SecuritySeverity::High,
        "Critical" => SecuritySeverity::Critical,
        _ => return Err("Invalid severity level".to_string()),
    };
    
    let session_uuid = if let Some(sid) = session_id {
        Some(Uuid::parse_str(&sid).map_err(|e| format!("Invalid session ID: {}", e))?)
    } else {
        None
    };
    
    manager.log_security_event(
        event_type,
        security_severity,
        message,
        details,
        session_uuid,
    ).await;
    
    Ok(())
}

/// Cleanup expired sessions and credentials
#[tauri::command]
pub async fn cleanup_expired_items(
    state: State<'_, SecurityManagerState>,
) -> Result<serde_json::Value, String> {
    let manager_guard = state.security_manager.read().await;
    let manager = manager_guard.as_ref()
        .ok_or("Security manager not initialized")?;
    
    // Cleanup expired sessions
    manager.cleanup_expired_sessions().await
        .map_err(|e| format!("Failed to cleanup sessions: {}", e))?;
    
    // Cleanup expired credentials
    let expired_credentials = manager.keychain_manager()
        .cleanup_expired_credentials().await
        .map_err(|e| format!("Failed to cleanup credentials: {}", e))?;
    
    Ok(serde_json::json!({
        "expired_credentials_removed": expired_credentials
    }))
}