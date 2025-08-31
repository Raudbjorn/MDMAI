//! Tauri commands for data management operations
//! 
//! This module provides the Tauri command interface for all data management
//! operations, bridging the Rust backend with the TypeScript frontend.

use crate::data_manager::*;
use std::sync::Arc;
use tauri::State;
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::Utc;

/// Data manager state wrapper for Tauri
pub struct DataManagerStateWrapper {
    inner: Arc<RwLock<Option<DataManagerState>>>,
}

impl DataManagerStateWrapper {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(None)),
        }
    }
    
    pub async fn initialize(&self, config: Option<DataManagerConfig>) -> DataResult<()> {
        let mut guard = self.inner.write().await;
        let manager = if let Some(config) = config {
            DataManagerState::with_config(config).await?
        } else {
            DataManagerState::new().await?
        };
        
        manager.initialize().await?;
        *guard = Some(manager);
        Ok(())
    }
    
    pub async fn get(&self) -> Option<DataManagerState> {
        let guard = self.inner.read().await;
        guard.clone()
    }
}

// Remove the problematic Clone implementation for now
// We'll use a different approach with proper state management

/// Initialize data manager
#[tauri::command]
pub async fn initialize_data_manager(
    state: State<'_, DataManagerStateWrapper>,
    config: Option<DataManagerConfig>,
) -> Result<(), String> {
    state.initialize(config).await.map_err(|e| e.to_string())
}

/// Initialize data manager with password for encryption
#[tauri::command]
pub async fn initialize_data_manager_with_password(
    state: State<'_, DataManagerStateWrapper>,
    password: String,
    config: Option<DataManagerConfig>,
) -> Result<(), String> {
    let mut guard = state.inner.write().await;
    let manager = if let Some(config) = config {
        DataManagerState::with_config_and_password(config, &password).await.map_err(|e| e.to_string())?
    } else {
        let default_config = DataManagerConfig::default();
        DataManagerState::with_config_and_password(default_config, &password).await.map_err(|e| e.to_string())?
    };
    
    manager.initialize().await.map_err(|e| e.to_string())?;
    *guard = Some(manager);
    Ok(())
}

/// Initialize encryption with password on existing data manager
/// This allows encryption to be set up after the data manager is already created
#[tauri::command]
pub async fn initialize_encryption_with_password(
    state: State<'_, DataManagerStateWrapper>,
    password: String,
) -> Result<(), String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    manager.encryption().initialize_with_password(&password).await.map_err(|e| e.to_string())?;
    Ok(())
}

/// Get data manager configuration
#[tauri::command]
pub async fn get_data_manager_config(
    state: State<'_, DataManagerStateWrapper>,
) -> Result<DataManagerConfig, String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    Ok(manager.config().clone())
}

/// Update data manager configuration
#[tauri::command]
pub async fn update_data_manager_config(
    state: State<'_, DataManagerStateWrapper>,
    new_config: DataManagerConfig,
) -> Result<(), String> {
    let guard = state.inner.read().await;
    if let Some(manager) = guard.as_ref() {
        let mut manager_clone = manager.clone();
        manager_clone.update_config(new_config).await.map_err(|e| e.to_string())?;
    } else {
        return Err("Data manager not initialized".to_string());
    }
    Ok(())
}

/// Campaign management commands
#[tauri::command]
pub async fn create_campaign(
    state: State<'_, DataManagerStateWrapper>,
    campaign: Campaign,
) -> Result<Campaign, String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    let storage = manager.storage().read().await;
    storage.create_campaign(&campaign).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_campaign(
    state: State<'_, DataManagerStateWrapper>,
    id: String,
) -> Result<Option<Campaign>, String> {
    let campaign_id = Uuid::parse_str(&id).map_err(|e| format!("Invalid UUID: {}", e))?;
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    let storage = manager.storage().read().await;
    storage.get_campaign(campaign_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn list_campaigns(
    state: State<'_, DataManagerStateWrapper>,
    params: Option<ListParams>,
) -> Result<ListResponse<Campaign>, String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    let storage = manager.storage().read().await;
    let list_params = params.unwrap_or_default();
    storage.list_campaigns(&list_params).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn update_campaign(
    state: State<'_, DataManagerStateWrapper>,
    id: String,
    campaign: Campaign,
) -> Result<Campaign, String> {
    let campaign_id = Uuid::parse_str(&id).map_err(|e| format!("Invalid UUID: {}", e))?;
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    let storage = manager.storage().read().await;
    storage.update_campaign(campaign_id, &campaign).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn delete_campaign(
    state: State<'_, DataManagerStateWrapper>,
    id: String,
) -> Result<(), String> {
    let campaign_id = Uuid::parse_str(&id).map_err(|e| format!("Invalid UUID: {}", e))?;
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    let storage = manager.storage().read().await;
    storage.delete_campaign(campaign_id).await.map_err(|e| e.to_string())
}

/// Character management commands
#[tauri::command]
pub async fn create_character(
    state: State<'_, DataManagerStateWrapper>,
    character: Character,
) -> Result<Character, String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    let storage = manager.storage().read().await;
    storage.create_character(&character).await.map_err(|e| e.to_string())
}

/// Backup management commands
#[tauri::command]
pub async fn create_backup(
    state: State<'_, DataManagerStateWrapper>,
    backup_type: String,
    description: Option<String>,
) -> Result<BackupMetadata, String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    
    match backup_type.as_str() {
        "full" => {
            manager.backup().create_full_backup(
                manager.storage(),
                manager.file_manager(),
                description
            ).await.map_err(|e| e.to_string())
        },
        "manual" => {
            let backup_id = Uuid::new_v4();
            let timestamp = Utc::now();
            let backup_name = format!("manual_backup_{}", timestamp.format("%Y%m%d_%H%M%S"));
            
            // Create a manual backup (similar to full backup)
            manager.backup().create_full_backup(
                manager.storage(),
                manager.file_manager(),
                description.or(Some("Manual backup".to_string()))
            ).await.map_err(|e| e.to_string())
        },
        _ => Err("Invalid backup type".to_string())
    }
}

#[tauri::command]
pub async fn list_backups(
    state: State<'_, DataManagerStateWrapper>,
) -> Result<Vec<BackupMetadata>, String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    manager.backup().list_backups(manager.storage()).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn restore_backup(
    state: State<'_, DataManagerStateWrapper>,
    backup_id: String,
    verify_integrity: Option<bool>,
) -> Result<RestoreResult, String> {
    let backup_uuid = Uuid::parse_str(&backup_id).map_err(|e| format!("Invalid UUID: {}", e))?;
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    
    manager.backup().restore_backup(
        backup_uuid,
        manager.storage(),
        manager.file_manager(),
        verify_integrity.unwrap_or(true)
    ).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn delete_backup(
    state: State<'_, DataManagerStateWrapper>,
    backup_id: String,
) -> Result<(), String> {
    let backup_uuid = Uuid::parse_str(&backup_id).map_err(|e| format!("Invalid UUID: {}", e))?;
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    
    manager.backup().delete_backup(backup_uuid, manager.storage()).await.map_err(|e| e.to_string())
}

/// Integrity checking commands
#[tauri::command]
pub async fn check_data_integrity(
    state: State<'_, DataManagerStateWrapper>,
) -> Result<IntegrityCheckResult, String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    manager.integrity().perform_initial_check(manager.storage()).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn repair_data_integrity(
    state: State<'_, DataManagerStateWrapper>,
    issues: Vec<IntegrityIssue>,
) -> Result<RepairResult, String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    manager.integrity().auto_repair(manager.storage(), &issues).await.map_err(|e| e.to_string())
}

/// File management commands
#[tauri::command]
pub async fn store_file(
    state: State<'_, DataManagerStateWrapper>,
    source_path: String,
    category: String,
    metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
) -> Result<StoredFile, String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    let path = std::path::Path::new(&source_path);
    
    let file_category = match category.as_str() {
        "rulebook" => FileCategory::RulebookPdf,
        "character_image" => FileCategory::CharacterImage,
        "npc_image" => FileCategory::NpcImage,
        "campaign_image" => FileCategory::CampaignImage,
        "map" => FileCategory::Map,
        "handout" => FileCategory::Handout,
        "audio" => FileCategory::Audio,
        "video" => FileCategory::Video,
        "document" => FileCategory::Document,
        _ => FileCategory::Other,
    };
    
    manager.file_manager().store_file(path, file_category, metadata).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn retrieve_file(
    state: State<'_, DataManagerStateWrapper>,
    file_id: String,
    stored_path: String,
) -> Result<Vec<u8>, String> {
    let uuid = Uuid::parse_str(&file_id).map_err(|e| format!("Invalid UUID: {}", e))?;
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    manager.file_manager().retrieve_file(uuid, &stored_path).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn export_file(
    state: State<'_, DataManagerStateWrapper>,
    file_id: String,
    stored_path: String,
    destination_path: String,
) -> Result<(), String> {
    let uuid = Uuid::parse_str(&file_id).map_err(|e| format!("Invalid UUID: {}", e))?;
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    let destination = std::path::Path::new(&destination_path);
    manager.file_manager().export_file(uuid, &stored_path, destination).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn delete_stored_file(
    state: State<'_, DataManagerStateWrapper>,
    stored_path: String,
) -> Result<(), String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    manager.file_manager().delete_file(&stored_path).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_storage_stats(
    state: State<'_, DataManagerStateWrapper>,
) -> Result<StorageStats, String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    manager.file_manager().get_storage_stats().await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn find_duplicate_files(
    state: State<'_, DataManagerStateWrapper>,
) -> Result<Vec<DuplicateGroup>, String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    manager.file_manager().find_duplicate_files().await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn optimize_storage(
    state: State<'_, DataManagerStateWrapper>,
    strategy: String,
) -> Result<OptimizationResult, String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    
    let dup_strategy = match strategy.as_str() {
        "keep_first" => DuplicateStrategy::KeepFirst,
        "keep_last" => DuplicateStrategy::KeepLast,
        "keep_oldest" => DuplicateStrategy::KeepOldest,
        "keep_newest" => DuplicateStrategy::KeepNewest,
        _ => return Err("Invalid strategy".to_string()),
    };
    
    manager.file_manager().optimize_storage(dup_strategy).await.map_err(|e| e.to_string())
}

/// Cache management commands
#[tauri::command]
pub async fn get_cache_stats(
    state: State<'_, DataManagerStateWrapper>,
) -> Result<CacheStats, String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    let cache = manager.cache().read().await;
    Ok(cache.get_stats())
}

#[tauri::command]
pub async fn clear_cache(
    state: State<'_, DataManagerStateWrapper>,
) -> Result<(), String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    let mut cache = manager.cache().write().await;
    cache.clear().await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cleanup_expired_cache(
    state: State<'_, DataManagerStateWrapper>,
) -> Result<(), String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    let mut cache = manager.cache().write().await;
    cache.cleanup_expired().await.map_err(|e| e.to_string())
}

/// Migration management commands
#[tauri::command]
pub async fn get_migration_status(
    state: State<'_, DataManagerStateWrapper>,
) -> Result<MigrationStatus, String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    manager.migration().get_migration_status(manager.storage()).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn run_pending_migrations(
    state: State<'_, DataManagerStateWrapper>,
) -> Result<Vec<Migration>, String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    manager.migration().run_pending_migrations(manager.storage()).await.map_err(|e| e.to_string())
}

/// Database statistics
#[tauri::command]
pub async fn get_database_stats(
    state: State<'_, DataManagerStateWrapper>,
) -> Result<serde_json::Value, String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    let storage = manager.storage().read().await;
    storage.get_database_stats().await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn perform_database_maintenance(
    state: State<'_, DataManagerStateWrapper>,
) -> Result<(), String> {
    let manager = state.get().await.ok_or("Data manager not initialized")?;
    let storage = manager.storage().read().await;
    storage.perform_maintenance().await.map_err(|e| e.to_string())
}

/// Data manager shutdown
#[tauri::command]
pub async fn shutdown_data_manager(
    state: State<'_, DataManagerStateWrapper>,
) -> Result<(), String> {
    let guard = state.inner.read().await;
    if let Some(manager) = guard.as_ref() {
        manager.shutdown().await.map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Create data manager state for Tauri
pub fn create_data_manager_state() -> DataManagerStateWrapper {
    DataManagerStateWrapper::new()
}