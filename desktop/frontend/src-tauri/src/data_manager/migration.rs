//! Database migration system
//! 
//! This module handles database schema migrations, version management,
//! and data transformation between versions.

use super::*;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use sqlx::Row;

/// Migration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Migration {
    pub version: u32,
    pub name: String,
    pub description: String,
    pub sql_file: String,
    pub checksum: String,
    pub applied_at: Option<DateTime<Utc>>,
    pub rollback_sql: Option<String>,
}

/// Migration manager for handling database schema changes
pub struct MigrationManager {
    config: DataManagerConfig,
    migrations: Vec<Migration>,
}

impl MigrationManager {
    /// Create new migration manager
    pub fn new(config: &DataManagerConfig) -> DataResult<Self> {
        let mut manager = Self {
            config: config.clone(),
            migrations: Vec::new(),
        };
        
        manager.discover_migrations()?;
        
        Ok(manager)
    }
    
    /// Discover available migrations
    fn discover_migrations(&mut self) -> DataResult<()> {
        let migrations_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/data_manager/migrations");
        
        if !migrations_dir.exists() {
            log::warn!("Migrations directory does not exist: {}", migrations_dir.display());
            return Ok(());
        }
        
        let mut migrations = Vec::new();
        
        // Read migration files
        let entries = fs::read_dir(&migrations_dir)
            .map_err(|e| DataError::Migration {
                message: format!("Failed to read migrations directory: {}", e),
            })?;
        
        for entry in entries {
            let entry = entry.map_err(|e| DataError::Migration {
                message: format!("Failed to read migration entry: {}", e),
            })?;
            
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("sql") {
                if let Some(migration) = self.parse_migration_file(&path)? {
                    migrations.push(migration);
                }
            }
        }
        
        // Sort migrations by version
        migrations.sort_by_key(|m| m.version);
        self.migrations = migrations;
        
        log::info!("Discovered {} migrations", self.migrations.len());
        Ok(())
    }
    
    /// Parse a migration file
    fn parse_migration_file(&self, path: &Path) -> DataResult<Option<Migration>> {
        let filename = path.file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| DataError::Migration {
                message: format!("Invalid migration filename: {}", path.display()),
            })?;
        
        // Parse filename format: 001_initial_schema.sql
        let parts: Vec<&str> = filename.splitn(2, '_').collect();
        if parts.len() != 2 {
            log::warn!("Ignoring migration file with invalid format: {}", filename);
            return Ok(None);
        }
        
        let version = parts[0].parse::<u32>()
            .map_err(|e| DataError::Migration {
                message: format!("Invalid version number in migration file {}: {}", filename, e),
            })?;
        
        let name = parts[1].trim_end_matches(".sql").replace('_', " ");
        
        // Read file content
        let content = fs::read_to_string(path)
            .map_err(|e| DataError::Migration {
                message: format!("Failed to read migration file {}: {}", path.display(), e),
            })?;
        
        // Generate checksum
        let mut hasher = blake3::Hasher::new();
        hasher.update(content.as_bytes());
        let checksum = hex::encode(hasher.finalize().as_bytes());
        
        // Extract description from SQL comments
        let description = self.extract_description_from_sql(&content);
        
        // Look for rollback SQL in the same file or separate rollback file
        let rollback_sql = self.find_rollback_sql(path, &content)?;
        
        Ok(Some(Migration {
            version,
            name,
            description,
            sql_file: path.to_string_lossy().to_string(),
            checksum,
            applied_at: None,
            rollback_sql,
        }))
    }
    
    /// Find rollback SQL for a migration
    fn find_rollback_sql(&self, migration_path: &Path, migration_content: &str) -> DataResult<Option<String>> {
        // Strategy 1: Look for embedded rollback SQL marked with comments
        let rollback_from_content = self.extract_rollback_from_content(migration_content);
        if rollback_from_content.is_some() {
            return Ok(rollback_from_content);
        }
        
        // Strategy 2: Look for separate .rollback.sql file
        let rollback_path = migration_path.with_extension("rollback.sql");
        if rollback_path.exists() {
            let rollback_content = fs::read_to_string(&rollback_path)
                .map_err(|e| DataError::Migration {
                    message: format!("Failed to read rollback file {}: {}", rollback_path.display(), e),
                })?;
            return Ok(Some(rollback_content.trim().to_string()));
        }
        
        // Strategy 3: Look for migration_name.down.sql file (Rails-style)
        let mut down_path = migration_path.to_path_buf();
        if let Some(stem) = down_path.file_stem() {
            if let Some(stem_str) = stem.to_str() {
                down_path.set_file_name(format!("{}.down.sql", stem_str));
                if down_path.exists() {
                    let rollback_content = fs::read_to_string(&down_path)
                        .map_err(|e| DataError::Migration {
                            message: format!("Failed to read down file {}: {}", down_path.display(), e),
                        })?;
                    return Ok(Some(rollback_content.trim().to_string()));
                }
            }
        }
        
        // No rollback SQL found
        Ok(None)
    }
    
    /// Extract rollback SQL from migration content (between special comment markers)
    fn extract_rollback_from_content(&self, content: &str) -> Option<String> {
        let lines: Vec<&str> = content.lines().collect();
        let mut rollback_lines = Vec::new();
        let mut in_rollback_section = false;
        
        for line in lines {
            let trimmed = line.trim();
            
            // Start of rollback section
            if trimmed.starts_with("-- ROLLBACK START") || trimmed.starts_with("-- BEGIN ROLLBACK") {
                in_rollback_section = true;
                continue;
            }
            
            // End of rollback section
            if trimmed.starts_with("-- ROLLBACK END") || trimmed.starts_with("-- END ROLLBACK") {
                break;
            }
            
            // Collect rollback lines
            if in_rollback_section {
                rollback_lines.push(line);
            }
        }
        
        if rollback_lines.is_empty() {
            None
        } else {
            Some(rollback_lines.join("\n").trim().to_string())
        }
    }

    /// Extract description from SQL comments
    fn extract_description_from_sql(&self, sql: &str) -> String {
        for line in sql.lines() {
            let line = line.trim();
            if line.starts_with("-- ") && !line.starts_with("-- Version:") {
                return line.trim_start_matches("-- ").to_string();
            }
        }
        "No description available".to_string()
    }
    
    /// Get current database schema version
    pub async fn get_current_version(&self, storage: &Arc<RwLock<DataStorage>>) -> DataResult<u32> {
        let storage_guard = storage.read().await;
        
        // Check if migrations table exists
        let table_exists = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
        )
        .fetch_one(&storage_guard.pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to check migrations table: {}", e),
        })?;
        
        if table_exists == 0 {
            // Create migrations table
            sqlx::query(
                r#"
                CREATE TABLE schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
                "#
            )
            .execute(&storage_guard.pool)
            .await
            .map_err(|e| DataError::Database {
                message: format!("Failed to create migrations table: {}", e),
            })?;
            
            return Ok(0);
        }
        
        // Get latest applied version
        let version = sqlx::query_scalar::<_, Option<i64>>(
            "SELECT MAX(version) FROM schema_migrations"
        )
        .fetch_one(&storage_guard.pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to get current version: {}", e),
        })?;
        
        Ok(version.unwrap_or(0) as u32)
    }
    
    /// Get applied migrations
    pub async fn get_applied_migrations(&self, storage: &Arc<RwLock<DataStorage>>) -> DataResult<Vec<Migration>> {
        let storage_guard = storage.read().await;
        
        let rows = sqlx::query(
            "SELECT version, name, checksum, applied_at FROM schema_migrations ORDER BY version"
        )
        .fetch_all(&storage_guard.pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to get applied migrations: {}", e),
        })?;
        
        let mut applied = Vec::new();
        for row in rows {
            let version: i64 = row.try_get("version").unwrap_or_default();
            let name: String = row.try_get("name").unwrap_or_default();
            let checksum: String = row.try_get("checksum").unwrap_or_default();
            let applied_at: String = row.try_get("applied_at").unwrap_or_default();
            
            applied.push(Migration {
                version: version as u32,
                name,
                description: "Applied migration".to_string(),
                sql_file: String::new(),
                checksum,
                applied_at: Some(DateTime::parse_from_rfc3339(&applied_at)
                    .map_err(|e| DataError::Database {
                        message: format!("Invalid timestamp in migrations table: {}", e),
                    })?
                    .with_timezone(&Utc)),
                rollback_sql: None,
            });
        }
        
        Ok(applied)
    }
    
    /// Run pending migrations
    pub async fn run_pending_migrations(&self, storage: &Arc<RwLock<DataStorage>>) -> DataResult<Vec<Migration>> {
        let current_version = self.get_current_version(storage).await?;
        let applied_migrations = self.get_applied_migrations(storage).await?;
        
        // Create lookup for applied migrations
        let applied_checksums: HashMap<u32, String> = applied_migrations
            .iter()
            .map(|m| (m.version, m.checksum.clone()))
            .collect();
        
        let mut executed = Vec::new();
        
        for migration in &self.migrations {
            if migration.version <= current_version {
                // Verify checksum for already applied migrations
                if let Some(applied_checksum) = applied_checksums.get(&migration.version) {
                    if applied_checksum != &migration.checksum {
                        return Err(DataError::Migration {
                            message: format!(
                                "Checksum mismatch for migration {} ({}). Database may be corrupted.",
                                migration.version, migration.name
                            ),
                        });
                    }
                }
                continue;
            }
            
            log::info!("Running migration {} ({})", migration.version, migration.name);
            self.execute_migration(migration, storage).await?;
            executed.push(migration.clone());
        }
        
        if !executed.is_empty() {
            log::info!("Successfully applied {} migrations", executed.len());
        }
        
        Ok(executed)
    }
    
    /// Execute a single migration
    async fn execute_migration(&self, migration: &Migration, storage: &Arc<RwLock<DataStorage>>) -> DataResult<()> {
        let storage_guard = storage.read().await;
        
        // Read SQL file
        let sql_content = fs::read_to_string(&migration.sql_file)
            .map_err(|e| DataError::Migration {
                message: format!("Failed to read migration file {}: {}", migration.sql_file, e),
            })?;
        
        // Begin transaction
        let mut tx = storage_guard.pool.begin().await
            .map_err(|e| DataError::Database {
                message: format!("Failed to begin migration transaction: {}", e),
            })?;
        
        // Execute migration SQL
        // Split on semicolon and execute each statement
        for statement in sql_content.split(';') {
            let statement = statement.trim();
            if !statement.is_empty() {
                sqlx::query(statement)
                    .execute(&mut *tx)
                    .await
                    .map_err(|e| DataError::Migration {
                        message: format!("Failed to execute migration statement: {}\nStatement: {}", e, statement),
                    })?;
            }
        }
        
        // Record migration as applied
        let now = Utc::now();
        sqlx::query(
            "INSERT INTO schema_migrations (version, name, checksum, applied_at) VALUES (?, ?, ?, ?)"
        )
        .bind(&migration.version)
        .bind(&migration.name)
        .bind(&migration.checksum)
        .bind(&now.to_rfc3339())
        .execute(&mut *tx)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to record migration: {}", e),
        })?;
        
        // Commit transaction
        tx.commit().await
            .map_err(|e| DataError::Database {
                message: format!("Failed to commit migration transaction: {}", e),
            })?;
        
        log::info!("Migration {} completed successfully", migration.version);
        Ok(())
    }
    
    /// Rollback to a specific version (if rollback SQL is available)
    pub async fn rollback_to_version(&self, target_version: u32, storage: &Arc<RwLock<DataStorage>>) -> DataResult<()> {
        let current_version = self.get_current_version(storage).await?;
        
        if target_version >= current_version {
            return Err(DataError::Migration {
                message: format!("Cannot rollback to version {} (current: {})", target_version, current_version),
            });
        }
        
        // Find migrations to rollback
        let mut rollback_migrations = Vec::new();
        for migration in self.migrations.iter().rev() {
            if migration.version > target_version && migration.version <= current_version {
                rollback_migrations.push(migration);
            }
        }
        
        if rollback_migrations.is_empty() {
            log::info!("No migrations to rollback");
            return Ok(());
        }
        
        let storage_guard = storage.read().await;
        
        // Execute rollbacks in reverse order
        for migration in rollback_migrations {
            if let Some(rollback_sql) = &migration.rollback_sql {
                log::info!("Rolling back migration {} ({})", migration.version, migration.name);
                
                let mut tx = storage_guard.pool.begin().await
                    .map_err(|e| DataError::Database {
                        message: format!("Failed to begin rollback transaction: {}", e),
                    })?;
                
                // Execute rollback SQL
                for statement in rollback_sql.split(';') {
                    let statement = statement.trim();
                    if !statement.is_empty() {
                        sqlx::query(statement)
                            .execute(&mut *tx)
                            .await
                            .map_err(|e| DataError::Migration {
                                message: format!("Failed to execute rollback statement: {}", e),
                            })?;
                    }
                }
                
                // Remove migration record
                sqlx::query("DELETE FROM schema_migrations WHERE version = ?")
                    .bind(&migration.version)
                    .execute(&mut *tx)
                    .await
                    .map_err(|e| DataError::Database {
                        message: format!("Failed to remove migration record: {}", e),
                    })?;
                
                tx.commit().await
                    .map_err(|e| DataError::Database {
                        message: format!("Failed to commit rollback transaction: {}", e),
                    })?;
                
                log::info!("Rollback of migration {} completed", migration.version);
            } else {
                return Err(DataError::Migration {
                    message: format!("No rollback available for migration {} ({})", migration.version, migration.name),
                });
            }
        }
        
        log::info!("Successfully rolled back to version {}", target_version);
        Ok(())
    }
    
    /// Validate all applied migrations
    pub async fn validate_migrations(&self, storage: &Arc<RwLock<DataStorage>>) -> DataResult<bool> {
        let applied_migrations = self.get_applied_migrations(storage).await?;
        let mut valid = true;
        
        for applied in &applied_migrations {
            if let Some(migration) = self.migrations.iter().find(|m| m.version == applied.version) {
                if migration.checksum != applied.checksum {
                    log::error!(
                        "Checksum mismatch for migration {}: expected {}, found {}",
                        applied.version,
                        migration.checksum,
                        applied.checksum
                    );
                    valid = false;
                }
            } else {
                log::error!("Applied migration {} not found in available migrations", applied.version);
                valid = false;
            }
        }
        
        Ok(valid)
    }
    
    /// Get migration status
    pub async fn get_migration_status(&self, storage: &Arc<RwLock<DataStorage>>) -> DataResult<MigrationStatus> {
        let current_version = self.get_current_version(storage).await?;
        let applied_migrations = self.get_applied_migrations(storage).await?;
        let latest_version = self.migrations.iter().map(|m| m.version).max().unwrap_or(0);
        
        let pending_migrations: Vec<&Migration> = self.migrations
            .iter()
            .filter(|m| m.version > current_version)
            .collect();
        
        let is_valid = self.validate_migrations(storage).await?;
        
        Ok(MigrationStatus {
            current_version,
            latest_version,
            applied_count: applied_migrations.len(),
            pending_count: pending_migrations.len(),
            is_valid,
            needs_migration: !pending_migrations.is_empty(),
            applied_migrations,
            pending_migrations: pending_migrations.into_iter().cloned().collect(),
        })
    }
    
    /// Create a new migration file template
    pub fn create_migration_template(&self, name: &str) -> DataResult<String> {
        let next_version = self.migrations.iter().map(|m| m.version).max().unwrap_or(0) + 1;
        let filename = format!("{:03}_{}.sql", next_version, name.replace(' ', "_"));
        
        let migrations_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/data_manager/migrations");
        
        let file_path = migrations_dir.join(&filename);
        
        let template = format!(
            r#"-- {}
-- Version: {}

-- Add your migration SQL here
-- This migration will be applied automatically when the application starts

-- Example:
-- CREATE TABLE example (
--     id TEXT PRIMARY KEY,
--     name TEXT NOT NULL,
--     created_at TEXT NOT NULL
-- );

-- Rollback SQL (optional, uncomment and modify if rollback is supported):
-- DROP TABLE example;
"#,
            name,
            next_version
        );
        
        fs::write(&file_path, template)
            .map_err(|e| DataError::Migration {
                message: format!("Failed to create migration template: {}", e),
            })?;
        
        Ok(file_path.to_string_lossy().to_string())
    }
}

/// Migration status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStatus {
    pub current_version: u32,
    pub latest_version: u32,
    pub applied_count: usize,
    pub pending_count: usize,
    pub is_valid: bool,
    pub needs_migration: bool,
    pub applied_migrations: Vec<Migration>,
    pub pending_migrations: Vec<Migration>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_migration_discovery() {
        let temp_dir = tempdir().unwrap();
        let config = DataManagerConfig {
            data_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        
        let manager = MigrationManager::new(&config).unwrap();
        
        // Should discover the initial schema migration
        assert!(!manager.migrations.is_empty());
        assert_eq!(manager.migrations[0].version, 1);
    }
}