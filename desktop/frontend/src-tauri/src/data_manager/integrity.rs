//! Data integrity validation and corruption recovery
//! 
//! This module provides comprehensive data integrity checking including:
//! - Database constraint validation
//! - File hash verification
//! - Foreign key consistency checks
//! - Orphaned record detection
//! - Automatic corruption recovery
//! - Health monitoring and alerts

use super::*;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use sqlx::Row;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, BufReader};

/// Data integrity checker for comprehensive validation
pub struct IntegrityChecker {
    config: DataManagerConfig,
}

impl IntegrityChecker {
    /// Create new integrity checker
    pub fn new(config: &DataManagerConfig) -> DataResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
    
    /// Perform initial integrity check
    pub async fn perform_initial_check(&self, storage: &Arc<RwLock<DataStorage>>) -> DataResult<IntegrityCheckResult> {
        log::info!("Performing initial data integrity check");
        
        let check_id = Uuid::new_v4();
        let timestamp = Utc::now();
        
        let mut issues = Vec::new();
        let mut total_records = 0u64;
        let mut corrupted_records = 0u64;
        let mut missing_files = 0u64;
        let mut hash_mismatches = 0u64;
        let mut orphaned_records = 0u64;
        
        let tables_to_check = [
            "campaigns", "characters", "npcs", "sessions", "rulebooks",
            "personality_profiles", "locations", "items", "spells", "assets"
        ];
        
        // Check each table
        for table in &tables_to_check {
            log::debug!("Checking table: {}", table);
            
            let table_result = self.check_table(storage, table).await?;
            
            total_records += table_result.record_count;
            corrupted_records += table_result.corrupted_count;
            missing_files += table_result.missing_files;
            hash_mismatches += table_result.hash_mismatches;
            orphaned_records += table_result.orphaned_count;
            
            issues.extend(table_result.issues);
        }
        
        // Check foreign key constraints
        let fk_issues = self.check_foreign_key_constraints(storage).await?;
        orphaned_records += fk_issues.len() as u64;
        issues.extend(fk_issues);
        
        // Check file system integrity
        let file_issues = self.check_file_system_integrity().await?;
        missing_files += file_issues.len() as u64;
        issues.extend(file_issues);
        
        // Determine overall status
        let overall_status = self.determine_overall_status(&issues);
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&issues);
        
        let result = IntegrityCheckResult {
            check_id,
            timestamp,
            tables_checked: tables_to_check.iter().map(|s| s.to_string()).collect(),
            total_records,
            corrupted_records,
            missing_files,
            hash_mismatches,
            orphaned_records,
            issues,
            overall_status: overall_status.clone(),
            recommendations,
        };
        
        // Log summary
        log::info!(
            "Initial integrity check completed: {} total records, {} issues found, status: {:?}",
            total_records,
            result.issues.len(),
            overall_status
        );
        
        Ok(result)
    }
    
    /// Perform scheduled integrity check
    pub async fn perform_scheduled_check(&self, storage: &Arc<RwLock<DataStorage>>) -> DataResult<IntegrityCheckResult> {
        log::info!("Performing scheduled integrity check");
        self.perform_initial_check(storage).await
    }
    
    /// Check individual table for integrity issues
    async fn check_table(&self, storage: &Arc<RwLock<DataStorage>>, table_name: &str) -> DataResult<TableCheckResult> {
        let storage_guard = storage.read().await;
        
        let mut result = TableCheckResult {
            table_name: table_name.to_string(),
            record_count: 0,
            corrupted_count: 0,
            missing_files: 0,
            hash_mismatches: 0,
            orphaned_count: 0,
            issues: Vec::new(),
        };
        
        // Get record count
        let count_query = format!("SELECT COUNT(*) as count FROM {}", table_name);
        let count_result = sqlx::query(&count_query)
            .fetch_one(&storage_guard.pool)
            .await
            .map_err(|e| DataError::Database {
                message: format!("Failed to count records in table {}: {}", table_name, e),
            })?;
        
        result.record_count = count_result.try_get::<i64, _>("count").unwrap_or(0) as u64;
        
        // Table-specific checks
        match table_name {
            "campaigns" => {
                result.issues.extend(self.check_campaigns_table(&storage_guard.pool).await?);
            },
            "characters" => {
                result.issues.extend(self.check_characters_table(&storage_guard.pool).await?);
            },
            "npcs" => {
                result.issues.extend(self.check_npcs_table(&storage_guard.pool).await?);
            },
            "sessions" => {
                result.issues.extend(self.check_sessions_table(&storage_guard.pool).await?);
            },
            "rulebooks" => {
                let issues = self.check_rulebooks_table(&storage_guard.pool).await?;
                result.missing_files += issues.iter().filter(|i| matches!(i.issue_type, IntegrityIssueType::MissingFile)).count() as u64;
                result.hash_mismatches += issues.iter().filter(|i| matches!(i.issue_type, IntegrityIssueType::HashMismatch)).count() as u64;
                result.issues.extend(issues);
            },
            "assets" => {
                let issues = self.check_assets_table(&storage_guard.pool).await?;
                result.missing_files += issues.iter().filter(|i| matches!(i.issue_type, IntegrityIssueType::MissingFile)).count() as u64;
                result.hash_mismatches += issues.iter().filter(|i| matches!(i.issue_type, IntegrityIssueType::HashMismatch)).count() as u64;
                result.issues.extend(issues);
            },
            _ => {
                // Generic checks for other tables
                result.issues.extend(self.check_generic_table(&storage_guard.pool, table_name).await?);
            }
        }
        
        // Update counters based on issues found
        result.corrupted_count = result.issues.iter()
            .filter(|i| matches!(i.issue_type, IntegrityIssueType::DataCorruption | IntegrityIssueType::SchemaViolation))
            .count() as u64;
        
        Ok(result)
    }
    
    /// Check campaigns table
    async fn check_campaigns_table(&self, pool: &sqlx::SqlitePool) -> DataResult<Vec<IntegrityIssue>> {
        let mut issues = Vec::new();
        
        // Check for campaigns with invalid status
        let invalid_status_campaigns = sqlx::query(
            "SELECT id, name, status FROM campaigns WHERE status NOT IN ('planning', 'active', 'paused', 'completed', 'archived')"
        )
        .fetch_all(pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to check campaign status: {}", e),
        })?;
        
        for campaign in invalid_status_campaigns {
            let id: Uuid = campaign.try_get("id").unwrap_or_default();
            let name: String = campaign.try_get("name").unwrap_or_default();
            let status: String = campaign.try_get("status").unwrap_or_default();
            issues.push(IntegrityIssue {
                issue_type: IntegrityIssueType::SchemaViolation,
                table_name: "campaigns".to_string(),
                record_id: Some(id),
                field_name: Some("status".to_string()),
                description: format!("Campaign '{}' has invalid status: '{}'", name, status),
                severity: IntegritySeverity::Medium,
                auto_fixable: true,
            });
        }
        
        // Check for campaigns with invalid JSON in settings
        let campaigns_with_settings = sqlx::query(
            "SELECT id, name, settings FROM campaigns WHERE settings IS NOT NULL"
        )
        .fetch_all(pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to check campaign settings: {}", e),
        })?;
        
        for campaign in campaigns_with_settings {
            let id: Uuid = campaign.try_get("id").unwrap_or_default();
            let name: String = campaign.try_get("name").unwrap_or_default();
            let settings: Option<String> = campaign.try_get("settings").ok();
            // Try to parse JSON
            if let Some(settings_str) = settings {
                if let Err(_) = serde_json::from_str::<serde_json::Value>(&settings_str) {
                    issues.push(IntegrityIssue {
                        issue_type: IntegrityIssueType::DataCorruption,
                        table_name: "campaigns".to_string(),
                        record_id: Some(id),
                        field_name: Some("settings".to_string()),
                        description: format!("Campaign '{}' has corrupted settings JSON", name),
                    severity: IntegritySeverity::High,
                        auto_fixable: false,
                    });
                }
            }
        }
        
        Ok(issues)
    }
    
    /// Check characters table
    async fn check_characters_table(&self, pool: &sqlx::SqlitePool) -> DataResult<Vec<IntegrityIssue>> {
        let mut issues = Vec::new();
        
        // Check for characters with invalid level
        let invalid_level_characters = sqlx::query(
            "SELECT id, name, level FROM characters WHERE level < 1 OR level > 20"
        )
        .fetch_all(pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to check character levels: {}", e),
        })?;
        
        for character in invalid_level_characters {
            let id: Uuid = character.try_get("id").unwrap_or_default();
            let name: String = character.try_get("name").unwrap_or_default();
            let level: i32 = character.try_get("level").unwrap_or_default();
            issues.push(IntegrityIssue {
                issue_type: IntegrityIssueType::SchemaViolation,
                table_name: "characters".to_string(),
                record_id: Some(id),
                field_name: Some("level".to_string()),
                description: format!("Character '{}' has invalid level: {}", name, level),
                severity: IntegritySeverity::Medium,
                auto_fixable: true,
            });
        }
        
        // Check for characters with invalid status
        let invalid_status_characters = sqlx::query(
            "SELECT id, name, status FROM characters WHERE status NOT IN ('active', 'retired', 'dead', 'missing', 'archived')"
        )
        .fetch_all(pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to check character status: {}", e),
        })?;
        
        for character in invalid_status_characters {
            let id: Uuid = character.try_get("id").unwrap_or_default();
            let name: String = character.try_get("name").unwrap_or_default();
            let status: String = character.try_get("status").unwrap_or_default();
            issues.push(IntegrityIssue {
                issue_type: IntegrityIssueType::SchemaViolation,
                table_name: "characters".to_string(),
                record_id: Some(id),
                field_name: Some("status".to_string()),
                description: format!("Character '{}' has invalid status: '{}'", name, status),
                severity: IntegritySeverity::Medium,
                auto_fixable: true,
            });
        }
        
        Ok(issues)
    }
    
    /// Check NPCs table
    async fn check_npcs_table(&self, pool: &sqlx::SqlitePool) -> DataResult<Vec<IntegrityIssue>> {
        let mut issues = Vec::new();
        
        // Check for NPCs with invalid role
        let valid_roles = ["ally", "enemy", "neutral", "merchant", "quest_giver", "noble", "commoner", "authority", "other"];
        let role_list = valid_roles.iter().map(|r| format!("'{}'", r)).collect::<Vec<_>>().join(", ");
        
        let query = format!("SELECT id, name, role FROM npcs WHERE role NOT IN ({})", role_list);
        let invalid_role_npcs = sqlx::query(&query)
            .fetch_all(pool)
            .await
            .map_err(|e| DataError::Database {
                message: format!("Failed to check NPC roles: {}", e),
            })?;
        
        for npc in invalid_role_npcs {
            let id: Uuid = npc.try_get("id").unwrap_or_default();
            let name: String = npc.try_get("name").unwrap_or_default();
            let role: String = npc.try_get("role").unwrap_or_default();
            
            issues.push(IntegrityIssue {
                issue_type: IntegrityIssueType::SchemaViolation,
                table_name: "npcs".to_string(),
                record_id: Some(id),
                field_name: Some("role".to_string()),
                description: format!("NPC '{}' has invalid role: '{}'", name, role),
                severity: IntegritySeverity::Medium,
                auto_fixable: true,
            });
        }
        
        Ok(issues)
    }
    
    /// Check sessions table
    async fn check_sessions_table(&self, pool: &sqlx::SqlitePool) -> DataResult<Vec<IntegrityIssue>> {
        let mut issues = Vec::new();
        
        // Check for duplicate session numbers within campaigns
        let duplicate_sessions = sqlx::query(
            "SELECT campaign_id, session_number, COUNT(*) as count 
             FROM sessions 
             GROUP BY campaign_id, session_number 
             HAVING count > 1"
        )
        .fetch_all(pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to check session duplicates: {}", e),
        })?;
        
        for duplicate in duplicate_sessions {
            let campaign_id: Uuid = duplicate.try_get("campaign_id").unwrap_or_default();
            let session_number: i32 = duplicate.try_get("session_number").unwrap_or_default();
            let count: i64 = duplicate.try_get("count").unwrap_or_default();
            issues.push(IntegrityIssue {
                issue_type: IntegrityIssueType::SchemaViolation,
                table_name: "sessions".to_string(),
                record_id: None,
                field_name: Some("session_number".to_string()),
                description: format!(
                    "Campaign {} has {} sessions with session number {}",
                    campaign_id, count, session_number
                ),
                severity: IntegritySeverity::High,
                auto_fixable: false,
            });
        }
        
        // Check for sessions with negative duration
        let invalid_duration_sessions = sqlx::query(
            "SELECT id, title, duration_minutes FROM sessions WHERE duration_minutes < 0"
        )
        .fetch_all(pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to check session duration: {}", e),
        })?;
        
        for session in invalid_duration_sessions {
            let id: Uuid = session.try_get("id").unwrap_or_default();
            let title: String = session.try_get("title").unwrap_or_default();
            let duration_minutes: Option<i32> = session.try_get("duration_minutes").ok();
            issues.push(IntegrityIssue {
                issue_type: IntegrityIssueType::SchemaViolation,
                table_name: "sessions".to_string(),
                record_id: Some(id),
                field_name: Some("duration_minutes".to_string()),
                description: format!("Session '{}' has negative duration: {}", title, duration_minutes.unwrap_or(0)),
                severity: IntegritySeverity::Medium,
                auto_fixable: true,
            });
        }
        
        Ok(issues)
    }
    
    /// Check rulebooks table (includes file integrity)
    async fn check_rulebooks_table(&self, pool: &sqlx::SqlitePool) -> DataResult<Vec<IntegrityIssue>> {
        let mut issues = Vec::new();
        
        let rulebooks = sqlx::query(
            "SELECT id, title, file_path, file_hash FROM rulebooks WHERE file_path IS NOT NULL"
        )
        .fetch_all(pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to get rulebooks for checking: {}", e),
        })?;
        
        for rulebook in rulebooks {
            let id: Uuid = rulebook.try_get("id").unwrap_or_default();
            let title: String = rulebook.try_get("title").unwrap_or_default();
            let file_path: Option<String> = rulebook.try_get("file_path").ok();
            let file_hash: Option<String> = rulebook.try_get("file_hash").ok();
            
            if let Some(file_path_str) = &file_path {
                let path = Path::new(file_path_str);
                
                // Check if file exists
                if !path.exists() {
                    issues.push(IntegrityIssue {
                        issue_type: IntegrityIssueType::MissingFile,
                        table_name: "rulebooks".to_string(),
                        record_id: Some(id),
                        field_name: Some("file_path".to_string()),
                        description: format!("Rulebook '{}' references missing file: {}", title, file_path_str),
                        severity: IntegritySeverity::High,
                        auto_fixable: false,
                    });
                } else if let Some(expected_hash) = &file_hash {
                    // Verify file hash using streaming approach
                    match Self::calculate_file_hash_streaming(path).await {
                        Ok(actual_hash) => {
                            if actual_hash != *expected_hash {
                                issues.push(IntegrityIssue {
                                    issue_type: IntegrityIssueType::HashMismatch,
                                    table_name: "rulebooks".to_string(),
                                    record_id: Some(id),
                                    field_name: Some("file_hash".to_string()),
                                    description: format!("Rulebook '{}' file hash mismatch - file may be corrupted", title),
                                    severity: IntegritySeverity::High,
                                    auto_fixable: true,
                                });
                            }
                        },
                        Err(e) => {
                            issues.push(IntegrityIssue {
                                issue_type: IntegrityIssueType::MissingFile,
                                table_name: "rulebooks".to_string(),
                                record_id: Some(id),
                                field_name: Some("file_path".to_string()),
                                description: format!("Rulebook '{}' file cannot be read: {}", title, e),
                                severity: IntegritySeverity::High,
                                auto_fixable: false,
                            });
                        }
                    }
                }
            }
        }
        
        Ok(issues)
    }
    
    /// Check assets table (includes file integrity)
    async fn check_assets_table(&self, pool: &sqlx::SqlitePool) -> DataResult<Vec<IntegrityIssue>> {
        let mut issues = Vec::new();
        
        let assets = sqlx::query(
            "SELECT id, name, file_path, file_hash FROM assets"
        )
        .fetch_all(pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to get assets for checking: {}", e),
        })?;
        
        for asset in assets {
            let id: Uuid = asset.try_get("id").unwrap_or_default();
            let name: String = asset.try_get("name").unwrap_or_default();
            let file_path: String = asset.try_get("file_path").unwrap_or_default();
            let file_hash: String = asset.try_get("file_hash").unwrap_or_default();
            
            let path = Path::new(&file_path);
            
            // Check if file exists
            if !path.exists() {
                issues.push(IntegrityIssue {
                    issue_type: IntegrityIssueType::MissingFile,
                    table_name: "assets".to_string(),
                    record_id: Some(id),
                    field_name: Some("file_path".to_string()),
                    description: format!("Asset '{}' references missing file: {}", name, file_path),
                    severity: IntegritySeverity::High,
                    auto_fixable: false,
                });
            } else {
                // Verify file hash using streaming approach
                match Self::calculate_file_hash_streaming(path).await {
                    Ok(actual_hash) => {
                        if actual_hash != file_hash {
                            issues.push(IntegrityIssue {
                                issue_type: IntegrityIssueType::HashMismatch,
                                table_name: "assets".to_string(),
                                record_id: Some(id),
                                field_name: Some("file_hash".to_string()),
                                description: format!("Asset '{}' file hash mismatch - file may be corrupted", name),
                                severity: IntegritySeverity::High,
                                auto_fixable: true,
                            });
                        }
                    },
                    Err(e) => {
                        issues.push(IntegrityIssue {
                            issue_type: IntegrityIssueType::MissingFile,
                            table_name: "assets".to_string(),
                            record_id: Some(id),
                            field_name: Some("file_path".to_string()),
                            description: format!("Asset '{}' file cannot be read: {}", name, e),
                            severity: IntegritySeverity::High,
                            auto_fixable: false,
                        });
                    }
                }
            }
        }
        
        Ok(issues)
    }
    
    /// Generic table checks
    async fn check_generic_table(&self, pool: &sqlx::SqlitePool, table_name: &str) -> DataResult<Vec<IntegrityIssue>> {
        let mut issues = Vec::new();
        
        // Check for NULL values in required fields (id, created_at, updated_at)
        let query = format!(
            "SELECT COUNT(*) as count FROM {} WHERE id IS NULL OR created_at IS NULL OR updated_at IS NULL",
            table_name
        );
        
        let null_count = sqlx::query(&query)
            .fetch_one(pool)
            .await
            .map_err(|e| DataError::Database {
                message: format!("Failed to check NULL values in table {}: {}", table_name, e),
            })?
            .get::<i64, _>("count");
        
        if null_count > 0 {
            issues.push(IntegrityIssue {
                issue_type: IntegrityIssueType::SchemaViolation,
                table_name: table_name.to_string(),
                record_id: None,
                field_name: None,
                description: format!("Table {} has {} records with NULL required fields", table_name, null_count),
                severity: IntegritySeverity::Critical,
                auto_fixable: false,
            });
        }
        
        Ok(issues)
    }
    
    /// Check foreign key constraints
    async fn check_foreign_key_constraints(&self, storage: &Arc<RwLock<DataStorage>>) -> DataResult<Vec<IntegrityIssue>> {
        let storage_guard = storage.read().await;
        let mut issues = Vec::new();
        
        // Check character -> campaign references
        let orphaned_characters = sqlx::query(
            "SELECT id, name, campaign_id FROM characters 
             WHERE campaign_id NOT IN (SELECT id FROM campaigns)"
        )
        .fetch_all(&storage_guard.pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to check orphaned characters: {}", e),
        })?;
        
        for character in orphaned_characters {
            let id: Uuid = character.try_get("id").unwrap_or_default();
            let name: String = character.try_get("name").unwrap_or_default();
            let campaign_id: Uuid = character.try_get("campaign_id").unwrap_or_default();
            issues.push(IntegrityIssue {
                issue_type: IntegrityIssueType::OrphanedRecord,
                table_name: "characters".to_string(),
                record_id: Some(id),
                field_name: Some("campaign_id".to_string()),
                description: format!("Character '{}' references non-existent campaign: {}", name, campaign_id),
                severity: IntegritySeverity::High,
                auto_fixable: false,
            });
        }
        
        // Check NPC -> campaign references
        let orphaned_npcs = sqlx::query(
            "SELECT id, name, campaign_id FROM npcs 
             WHERE campaign_id NOT IN (SELECT id FROM campaigns)"
        )
        .fetch_all(&storage_guard.pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to check orphaned NPCs: {}", e),
        })?;
        
        for npc in orphaned_npcs {
            let id: Uuid = npc.try_get("id").unwrap_or_default();
            let name: String = npc.try_get("name").unwrap_or_default();
            let campaign_id: Uuid = npc.try_get("campaign_id").unwrap_or_default();
            issues.push(IntegrityIssue {
                issue_type: IntegrityIssueType::OrphanedRecord,
                table_name: "npcs".to_string(),
                record_id: Some(id),
                field_name: Some("campaign_id".to_string()),
                description: format!("NPC '{}' references non-existent campaign: {}", name, campaign_id),
                severity: IntegritySeverity::High,
                auto_fixable: false,
            });
        }
        
        // Check session -> campaign references
        let orphaned_sessions = sqlx::query(
            "SELECT id, title, campaign_id FROM sessions 
             WHERE campaign_id NOT IN (SELECT id FROM campaigns)"
        )
        .fetch_all(&storage_guard.pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to check orphaned sessions: {}", e),
        })?;
        
        for session in orphaned_sessions {
            let id: Uuid = session.try_get("id").unwrap_or_default();
            let title: String = session.try_get("title").unwrap_or_default();
            let campaign_id: Uuid = session.try_get("campaign_id").unwrap_or_default();
            issues.push(IntegrityIssue {
                issue_type: IntegrityIssueType::OrphanedRecord,
                table_name: "sessions".to_string(),
                record_id: Some(id),
                field_name: Some("campaign_id".to_string()),
                description: format!("Session '{}' references non-existent campaign: {}", title, campaign_id),
                severity: IntegritySeverity::High,
                auto_fixable: false,
            });
        }
        
        Ok(issues)
    }
    
    /// Check file system integrity
    async fn check_file_system_integrity(&self) -> DataResult<Vec<IntegrityIssue>> {
        let mut issues = Vec::new();
        
        // Check if required directories exist
        let required_dirs = [
            &self.config.data_dir,
            &self.config.files_dir,
            &self.config.backup_dir,
            &self.config.cache_dir,
        ];
        
        for dir in &required_dirs {
            if !dir.exists() {
                issues.push(IntegrityIssue {
                    issue_type: IntegrityIssueType::MissingFile,
                    table_name: "filesystem".to_string(),
                    record_id: None,
                    field_name: None,
                    description: format!("Required directory does not exist: {}", dir.display()),
                    severity: IntegritySeverity::Critical,
                    auto_fixable: true,
                });
            }
        }
        
        // Check database file
        if !self.config.database_path.exists() {
            issues.push(IntegrityIssue {
                issue_type: IntegrityIssueType::MissingFile,
                table_name: "filesystem".to_string(),
                record_id: None,
                field_name: None,
                description: format!("Database file does not exist: {}", self.config.database_path.display()),
                severity: IntegritySeverity::Critical,
                auto_fixable: false,
            });
        }
        
        Ok(issues)
    }
    
    /// Determine overall status based on issues
    fn determine_overall_status(&self, issues: &[IntegrityIssue]) -> IntegrityStatus {
        let critical_count = issues.iter().filter(|i| matches!(i.severity, IntegritySeverity::Critical)).count();
        let high_count = issues.iter().filter(|i| matches!(i.severity, IntegritySeverity::High)).count();
        let medium_count = issues.iter().filter(|i| matches!(i.severity, IntegritySeverity::Medium)).count();
        
        if critical_count > 0 {
            IntegrityStatus::Critical
        } else if high_count > 0 {
            IntegrityStatus::Error
        } else if medium_count > 0 {
            IntegrityStatus::Warning
        } else {
            IntegrityStatus::Healthy
        }
    }
    
    /// Generate recommendations based on issues
    fn generate_recommendations(&self, issues: &[IntegrityIssue]) -> Vec<String> {
        let mut recommendations = Vec::new();
        let auto_fixable_count = issues.iter().filter(|i| i.auto_fixable).count();
        let critical_count = issues.iter().filter(|i| matches!(i.severity, IntegritySeverity::Critical)).count();
        
        if auto_fixable_count > 0 {
            recommendations.push(format!(
                "Run automatic repair to fix {} auto-fixable issues",
                auto_fixable_count
            ));
        }
        
        if critical_count > 0 {
            recommendations.push("Critical issues detected - immediate attention required".to_string());
            recommendations.push("Consider restoring from a recent backup".to_string());
        }
        
        let missing_files = issues.iter().filter(|i| matches!(i.issue_type, IntegrityIssueType::MissingFile)).count();
        if missing_files > 0 {
            recommendations.push("Some files are missing - check file system permissions and storage".to_string());
        }
        
        let hash_mismatches = issues.iter().filter(|i| matches!(i.issue_type, IntegrityIssueType::HashMismatch)).count();
        if hash_mismatches > 0 {
            recommendations.push("File corruption detected - consider running file system check".to_string());
        }
        
        if issues.is_empty() {
            recommendations.push("Data integrity is healthy - no action required".to_string());
        }
        
        recommendations
    }
    
    /// Attempt automatic repair of fixable issues
    pub async fn auto_repair(&self, storage: &Arc<RwLock<DataStorage>>, issues: &[IntegrityIssue]) -> DataResult<RepairResult> {
        log::info!("Starting automatic repair");
        
        let storage_guard = storage.read().await;
        let mut repaired_count = 0;
        let mut failed_repairs = Vec::new();
        
        for issue in issues {
            if !issue.auto_fixable {
                continue;
            }
            
            match self.repair_issue(&storage_guard.pool, issue).await {
                Ok(()) => {
                    repaired_count += 1;
                    log::info!("Repaired issue: {}", issue.description);
                },
                Err(e) => {
                    failed_repairs.push(format!("Failed to repair '{}': {}", issue.description, e));
                    log::error!("Failed to repair issue '{}': {}", issue.description, e);
                }
            }
        }
        
        log::info!("Automatic repair completed: {} repaired, {} failed", repaired_count, failed_repairs.len());
        
        Ok(RepairResult {
            repaired_count,
            failed_repairs,
            timestamp: Utc::now(),
        })
    }
    
    /// Repair individual issue
    async fn repair_issue(&self, pool: &sqlx::SqlitePool, issue: &IntegrityIssue) -> DataResult<()> {
        match (&issue.issue_type, issue.table_name.as_str(), issue.field_name.as_deref()) {
            (IntegrityIssueType::SchemaViolation, "campaigns", Some("status")) => {
                // Fix invalid campaign status
                if let Some(record_id) = issue.record_id {
                    sqlx::query("UPDATE campaigns SET status = 'active' WHERE id = ?")
                        .bind(&record_id)
                        .execute(pool)
                        .await
                        .map_err(|e| DataError::Database {
                            message: format!("Failed to repair campaign status: {}", e),
                        })?;
                }
            },
            (IntegrityIssueType::SchemaViolation, "characters", Some("level")) => {
                // Fix invalid character level
                if let Some(record_id) = issue.record_id {
                    sqlx::query("UPDATE characters SET level = 1 WHERE id = ? AND (level < 1 OR level > 20)")
                        .bind(&record_id)
                        .execute(pool)
                        .await
                        .map_err(|e| DataError::Database {
                            message: format!("Failed to repair character level: {}", e),
                        })?;
                }
            },
            (IntegrityIssueType::SchemaViolation, "characters", Some("status")) => {
                // Fix invalid character status
                if let Some(record_id) = issue.record_id {
                    sqlx::query("UPDATE characters SET status = 'active' WHERE id = ?")
                        .bind(&record_id)
                        .execute(pool)
                        .await
                        .map_err(|e| DataError::Database {
                            message: format!("Failed to repair character status: {}", e),
                        })?;
                }
            },
            (IntegrityIssueType::HashMismatch, _, _) => {
                // Recalculate and update file hash
                if let Some(record_id) = issue.record_id {
                    self.recalculate_file_hash(pool, &issue.table_name, record_id).await?;
                }
            },
            (IntegrityIssueType::MissingFile, "filesystem", _) => {
                // Create missing directories
                if issue.description.contains("Required directory") {
                    let dir_path = self.extract_path_from_description(&issue.description);
                    if let Some(path) = dir_path {
                        std::fs::create_dir_all(path)
                            .map_err(|e| DataError::FileSystem {
                                message: format!("Failed to create directory: {}", e),
                            })?;
                    }
                }
            },
            _ => {
                return Err(DataError::Integrity {
                    message: format!("No repair method available for issue: {}", issue.description),
                });
            }
        }
        
        Ok(())
    }
    
    /// Recalculate file hash for a record
    async fn recalculate_file_hash(&self, pool: &sqlx::SqlitePool, table_name: &str, record_id: Uuid) -> DataResult<()> {
        let file_path = match table_name {
            "rulebooks" => {
                let row = sqlx::query("SELECT file_path FROM rulebooks WHERE id = ?")
                    .bind(&record_id)
                    .fetch_optional(pool)
                    .await
                    .map_err(|e| DataError::Database {
                        message: format!("Failed to get rulebook file path: {}", e),
                    })?;
                row.and_then(|r| r.try_get::<String, _>("file_path").ok())
            },
            "assets" => {
                let row = sqlx::query("SELECT file_path FROM assets WHERE id = ?")
                    .bind(&record_id)
                    .fetch_one(pool)
                    .await
                    .map_err(|e| DataError::Database {
                        message: format!("Failed to get asset file path: {}", e),
                    })?;
                Some(row.try_get("file_path").unwrap_or_default())
            },
            _ => None,
        };
        
        if let Some(path) = file_path {
            let new_hash = Self::calculate_file_hash_streaming(&path).await
                .map_err(|e| DataError::FileSystem {
                    message: format!("Failed to calculate file hash: {}", e),
                })?;
            
            match table_name {
                "rulebooks" => {
                    sqlx::query("UPDATE rulebooks SET file_hash = ? WHERE id = ?")
                        .bind(&new_hash)
                        .bind(&record_id)
                        .execute(pool)
                        .await
                        .map_err(|e| DataError::Database {
                            message: format!("Failed to update rulebook hash: {}", e),
                        })?;
                },
                "assets" => {
                    sqlx::query("UPDATE assets SET file_hash = ? WHERE id = ?")
                        .bind(&new_hash)
                        .bind(&record_id)
                        .execute(pool)
                        .await
                        .map_err(|e| DataError::Database {
                            message: format!("Failed to update asset hash: {}", e),
                        })?;
                },
                _ => {}
            }
        }
        
        Ok(())
    }
    
    /// Extract path from issue description
    fn extract_path_from_description<'a>(&self, description: &'a str) -> Option<&'a Path> {
        // Simple extraction - in real implementation, this would be more robust
        if let Some(start) = description.find(": ") {
            let path_str = &description[start + 2..];
            Some(Path::new(path_str))
        } else {
            None
        }
    }
    
    /// Calculate file hash using streaming I/O to avoid loading entire file into memory
    /// Uses 64KB chunks for optimal performance with large files
    async fn calculate_file_hash_streaming<P: AsRef<Path>>(path: P) -> Result<String, std::io::Error> {
        const CHUNK_SIZE: usize = 64 * 1024; // 64KB chunks
        
        let file = File::open(path.as_ref()).await?;
        let mut reader = BufReader::with_capacity(CHUNK_SIZE, file);
        let mut hasher = blake3::Hasher::new();
        let mut buffer = vec![0u8; CHUNK_SIZE];
        
        loop {
            let bytes_read = reader.read(&mut buffer).await?;
            if bytes_read == 0 {
                break; // End of file
            }
            
            // Only hash the actual bytes read, not the entire buffer
            hasher.update(&buffer[..bytes_read]);
        }
        
        Ok(hex::encode(hasher.finalize().as_bytes()))
    }
}

/// Result of table integrity check
#[derive(Debug)]
struct TableCheckResult {
    table_name: String,
    record_count: u64,
    corrupted_count: u64,
    missing_files: u64,
    hash_mismatches: u64,
    orphaned_count: u64,
    issues: Vec<IntegrityIssue>,
}

/// Result of repair operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairResult {
    pub repaired_count: usize,
    pub failed_repairs: Vec<String>,
    pub timestamp: DateTime<Utc>,
}