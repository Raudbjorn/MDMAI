use anyhow::{Result, anyhow};
use rusqlite::{Connection, params, types::Value, Row};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::RwLock;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use tokio::fs;

/// Represents different SQL column types for proper data handling
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SqlColumnType {
    Integer,
    Real,
    Text,
    Blob,
    Null,
}

impl From<&Value> for SqlColumnType {
    fn from(value: &Value) -> Self {
        match value {
            Value::Null => SqlColumnType::Null,
            Value::Integer(_) => SqlColumnType::Integer,
            Value::Real(_) => SqlColumnType::Real,
            Value::Text(_) => SqlColumnType::Text,
            Value::Blob(_) => SqlColumnType::Blob,
        }
    }
}

/// Represents a properly typed database column value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedColumnValue {
    pub column_name: String,
    pub column_type: SqlColumnType,
    pub value: serde_json::Value,
}

impl TypedColumnValue {
    /// Create typed value from rusqlite Value, preserving original data types
    pub fn from_rusqlite_value(column_name: String, value: &Value) -> Self {
        let column_type = SqlColumnType::from(value);
        let json_value = match value {
            Value::Null => serde_json::Value::Null,
            Value::Integer(i) => serde_json::Value::Number((*i).into()),
            Value::Real(f) => serde_json::Value::Number(
                serde_json::Number::from_f64(*f).unwrap_or_else(|| 0.into())
            ),
            Value::Text(s) => serde_json::Value::String(s.clone()),
            Value::Blob(b) => {
                // Encode binary data as base64 to preserve it safely
                serde_json::Value::String(base64::encode(b))
            }
        };

        Self {
            column_name,
            column_type,
            value: json_value,
        }
    }

    /// Convert back to rusqlite Value for database operations
    pub fn to_rusqlite_value(&self) -> Result<Value> {
        match self.column_type {
            SqlColumnType::Null => Ok(Value::Null),
            SqlColumnType::Integer => {
                if let Some(i) = self.value.as_i64() {
                    Ok(Value::Integer(i))
                } else {
                    Err(anyhow!("Invalid integer value for column {}", self.column_name))
                }
            },
            SqlColumnType::Real => {
                if let Some(f) = self.value.as_f64() {
                    Ok(Value::Real(f))
                } else {
                    Err(anyhow!("Invalid real value for column {}", self.column_name))
                }
            },
            SqlColumnType::Text => {
                if let Some(s) = self.value.as_str() {
                    Ok(Value::Text(s.to_string()))
                } else {
                    Err(anyhow!("Invalid text value for column {}", self.column_name))
                }
            },
            SqlColumnType::Blob => {
                if let Some(s) = self.value.as_str() {
                    let blob_data = base64::decode(s)
                        .map_err(|e| anyhow!("Invalid base64 blob data for column {}: {}", self.column_name, e))?;
                    Ok(Value::Blob(blob_data))
                } else {
                    Err(anyhow!("Invalid blob value for column {}", self.column_name))
                }
            },
        }
    }
}

/// Database table schema information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSchema {
    pub table_name: String,
    pub columns: Vec<String>,
    pub column_types: HashMap<String, SqlColumnType>,
    pub primary_keys: Vec<String>,
}

/// Represents a database row with typed column values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedDatabaseRow {
    pub table_name: String,
    pub columns: Vec<TypedColumnValue>,
}

/// Backup metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    pub backup_id: String,
    pub backup_name: Option<String>,
    pub created_at: DateTime<Utc>,
    pub file_path: PathBuf,
    pub file_size_bytes: u64,
    pub table_count: usize,
    pub row_count: usize,
    pub checksum: String,
    pub format_version: String,
}

/// Database export format that preserves data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseExport {
    pub metadata: BackupMetadata,
    pub schemas: Vec<TableSchema>,
    pub data: HashMap<String, Vec<TypedDatabaseRow>>,
}

/// Thread-safe backup manager with proper SQL type handling
/// Fixes data corruption issues in database export/import operations
pub struct BackupManager {
    /// Base directory for backup storage
    backup_path: PathBuf,
    
    /// Registry of backup metadata
    backup_registry: Arc<RwLock<HashMap<String, BackupMetadata>>>,
    
    /// Backup retention period in days
    retention_days: u32,
    
    /// Database connection pool (in a real implementation, you'd use a proper pool)
    db_connection: Arc<RwLock<Option<Connection>>>,
}

impl BackupManager {
    /// Create a new backup manager
    pub fn new(retention_days: u32) -> Result<Self> {
        let backup_path = std::env::temp_dir().join("mdmai_backups");
        Self::with_backup_path(backup_path, retention_days)
    }

    /// Create backup manager with custom backup path
    pub fn with_backup_path<P: AsRef<Path>>(backup_path: P, retention_days: u32) -> Result<Self> {
        let backup_path = backup_path.as_ref().to_path_buf();
        
        // Ensure backup directory exists
        std::fs::create_dir_all(&backup_path)
            .map_err(|e| anyhow!("Failed to create backup directory: {}", e))?;

        Ok(Self {
            backup_path,
            backup_registry: Arc::new(RwLock::new(HashMap::new())),
            retention_days,
            db_connection: Arc::new(RwLock::new(None)),
        })
    }

    /// Initialize database connection
    pub async fn initialize_database<P: AsRef<Path>>(&self, db_path: P) -> Result<()> {
        let db_path = db_path.as_ref().to_path_buf();
        
        tokio::task::spawn_blocking(move || {
            let conn = Connection::open(&db_path)
                .map_err(|e| anyhow!("Failed to open database: {}", e))?;
            
            // Enable foreign key constraints and WAL mode for better performance
            conn.execute("PRAGMA foreign_keys = ON", [])
                .map_err(|e| anyhow!("Failed to enable foreign keys: {}", e))?;
            conn.execute("PRAGMA journal_mode = WAL", [])
                .map_err(|e| anyhow!("Failed to set WAL mode: {}", e))?;
            
            Ok(conn)
        }).await.map_err(|e| anyhow!("Database initialization task failed: {}", e))??;

        log::info!("Database initialized for backup operations");
        Ok(())
    }

    /// Create a comprehensive backup with proper type handling
    pub async fn create_backup(&self, backup_name: Option<String>) -> Result<String> {
        let backup_id = Uuid::new_v4().to_string();
        let backup_file_path = self.backup_path.join(format!("{}.backup.json", backup_id));
        
        log::info!("Creating backup: {}", backup_id);
        let start_time = std::time::Instant::now();

        // Export database with proper type handling
        let database_export = self.export_database_with_types().await?;
        
        // Calculate checksum for integrity verification
        let export_json = serde_json::to_string_pretty(&database_export)
            .map_err(|e| anyhow!("Failed to serialize backup data: {}", e))?;
        
        let checksum = self.calculate_checksum(&export_json);

        // Create backup metadata
        let metadata = BackupMetadata {
            backup_id: backup_id.clone(),
            backup_name,
            created_at: Utc::now(),
            file_path: backup_file_path.clone(),
            file_size_bytes: export_json.len() as u64,
            table_count: database_export.schemas.len(),
            row_count: database_export.data.values().map(|rows| rows.len()).sum(),
            checksum: checksum.clone(),
            format_version: "1.0.0".to_string(),
        };

        // Update the export with final metadata
        let final_export = DatabaseExport {
            metadata: metadata.clone(),
            schemas: database_export.schemas,
            data: database_export.data,
        };

        // Write backup to file
        let final_json = serde_json::to_string_pretty(&final_export)
            .map_err(|e| anyhow!("Failed to serialize final backup: {}", e))?;
        
        fs::write(&backup_file_path, final_json).await
            .map_err(|e| anyhow!("Failed to write backup file: {}", e))?;

        // Register backup
        self.backup_registry.write().insert(backup_id.clone(), metadata);

        let duration = start_time.elapsed();
        log::info!("Backup created successfully: {} ({} tables, {} rows) in {:?}", 
                  backup_id, final_export.schemas.len(), 
                  final_export.data.values().map(|rows| rows.len()).sum::<usize>(),
                  duration);

        // Clean up old backups
        self.cleanup_old_backups().await?;

        Ok(backup_id)
    }

    /// Restore database from backup with proper type handling
    pub async fn restore_backup(&self, backup_id: &str) -> Result<()> {
        let metadata = {
            let registry = self.backup_registry.read();
            registry.get(backup_id).cloned()
        };

        let metadata = metadata.ok_or_else(|| anyhow!("Backup not found: {}", backup_id))?;

        if !metadata.file_path.exists() {
            return Err(anyhow!("Backup file does not exist: {:?}", metadata.file_path));
        }

        log::info!("Restoring backup: {}", backup_id);
        let start_time = std::time::Instant::now();

        // Read and parse backup file
        let backup_content = fs::read_to_string(&metadata.file_path).await
            .map_err(|e| anyhow!("Failed to read backup file: {}", e))?;

        // Verify checksum
        let calculated_checksum = self.calculate_checksum(&backup_content);
        if calculated_checksum != metadata.checksum {
            return Err(anyhow!("Backup file corrupted: checksum mismatch"));
        }

        let database_export: DatabaseExport = serde_json::from_str(&backup_content)
            .map_err(|e| anyhow!("Failed to parse backup file: {}", e))?;

        // Restore database with proper type handling
        self.import_database_with_types(&database_export).await?;

        let duration = start_time.elapsed();
        log::info!("Backup restored successfully: {} in {:?}", backup_id, duration);

        Ok(())
    }

    /// List all available backups
    pub async fn list_backups(&self) -> Result<Vec<BackupMetadata>> {
        // Scan backup directory for backup files
        self.scan_backup_directory().await?;
        
        let mut backups: Vec<BackupMetadata> = self.backup_registry.read().values().cloned().collect();
        
        // Sort by creation date (newest first)
        backups.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        
        Ok(backups)
    }

    /// Delete a specific backup
    pub async fn delete_backup(&self, backup_id: &str) -> Result<()> {
        let metadata = {
            let mut registry = self.backup_registry.write();
            registry.remove(backup_id)
        };

        if let Some(metadata) = metadata {
            if metadata.file_path.exists() {
                fs::remove_file(&metadata.file_path).await
                    .map_err(|e| anyhow!("Failed to delete backup file: {}", e))?;
            }
            log::info!("Deleted backup: {}", backup_id);
        }

        Ok(())
    }

    /// Export database with proper type handling to prevent data corruption
    async fn export_database_with_types(&self) -> Result<DatabaseExport> {
        let db_conn = self.db_connection.read().clone()
            .ok_or_else(|| anyhow!("Database not initialized"))?;

        tokio::task::spawn_blocking(move || {
            let mut schemas = Vec::new();
            let mut data: HashMap<String, Vec<TypedDatabaseRow>> = HashMap::new();

            // Get all table names
            let table_names = Self::get_table_names(&db_conn)?;

            for table_name in table_names {
                // Get table schema with column types
                let schema = Self::get_table_schema(&db_conn, &table_name)?;
                schemas.push(schema.clone());

                // Export table data with proper type handling
                let rows = Self::export_table_data_with_types(&db_conn, &table_name, &schema)?;
                data.insert(table_name, rows);
            }

            Ok(DatabaseExport {
                metadata: BackupMetadata {
                    backup_id: String::new(), // Will be filled later
                    backup_name: None,
                    created_at: Utc::now(),
                    file_path: PathBuf::new(),
                    file_size_bytes: 0,
                    table_count: schemas.len(),
                    row_count: data.values().map(|rows| rows.len()).sum(),
                    checksum: String::new(),
                    format_version: "1.0.0".to_string(),
                },
                schemas,
                data,
            })
        }).await.map_err(|e| anyhow!("Database export task failed: {}", e))?
    }

    /// Import database with proper type handling
    async fn import_database_with_types(&self, export: &DatabaseExport) -> Result<()> {
        let db_conn = self.db_connection.read().clone()
            .ok_or_else(|| anyhow!("Database not initialized"))?;

        let export = export.clone();

        tokio::task::spawn_blocking(move || {
            // Start transaction for atomic restore
            let tx = db_conn.begin()
                .map_err(|e| anyhow!("Failed to start transaction: {}", e))?;

            // Clear existing data (be careful in production!)
            for schema in &export.schemas {
                tx.execute(&format!("DELETE FROM {}", schema.table_name), [])
                    .map_err(|e| anyhow!("Failed to clear table {}: {}", schema.table_name, e))?;
            }

            // Restore data with proper type handling
            for (table_name, rows) in &export.data {
                let schema = export.schemas.iter()
                    .find(|s| s.table_name == *table_name)
                    .ok_or_else(|| anyhow!("Schema not found for table: {}", table_name))?;

                for row in rows {
                    Self::insert_typed_row(&tx, schema, row)?;
                }
            }

            // Commit transaction
            tx.commit()
                .map_err(|e| anyhow!("Failed to commit restore transaction: {}", e))?;

            Ok(())
        }).await.map_err(|e| anyhow!("Database import task failed: {}", e))?
    }

    /// Get all table names from database
    fn get_table_names(conn: &Connection) -> Result<Vec<String>> {
        let mut stmt = conn.prepare(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).map_err(|e| anyhow!("Failed to prepare table query: {}", e))?;

        let table_names = stmt.query_map([], |row| {
            Ok(row.get::<_, String>(0)?)
        }).map_err(|e| anyhow!("Failed to execute table query: {}", e))?
        .collect::<Result<Vec<String>, _>>()
        .map_err(|e| anyhow!("Failed to collect table names: {}", e))?;

        Ok(table_names)
    }

    /// Get table schema with column type information
    fn get_table_schema(conn: &Connection, table_name: &str) -> Result<TableSchema> {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({})", table_name))
            .map_err(|e| anyhow!("Failed to prepare schema query: {}", e))?;

        let mut columns = Vec::new();
        let mut column_types = HashMap::new();
        let mut primary_keys = Vec::new();

        stmt.query_map([], |row| {
            let column_name: String = row.get(1)?;
            let type_name: String = row.get(2)?;
            let is_primary_key: bool = row.get(5)?;

            Ok((column_name, type_name, is_primary_key))
        }).map_err(|e| anyhow!("Failed to execute schema query: {}", e))?
        .for_each(|result| {
            if let Ok((column_name, type_name, is_primary_key)) = result {
                columns.push(column_name.clone());
                
                // Map SQLite type names to our enum
                let column_type = match type_name.to_uppercase().as_str() {
                    "INTEGER" | "INT" | "BIGINT" | "SMALLINT" => SqlColumnType::Integer,
                    "REAL" | "DOUBLE" | "FLOAT" | "NUMERIC" => SqlColumnType::Real,
                    "TEXT" | "VARCHAR" | "CHAR" | "STRING" => SqlColumnType::Text,
                    "BLOB" | "BINARY" => SqlColumnType::Blob,
                    _ => SqlColumnType::Text, // Default to text for unknown types
                };
                
                column_types.insert(column_name.clone(), column_type);
                
                if is_primary_key {
                    primary_keys.push(column_name);
                }
            }
        });

        Ok(TableSchema {
            table_name: table_name.to_string(),
            columns,
            column_types,
            primary_keys,
        })
    }

    /// Export table data with proper type handling
    fn export_table_data_with_types(
        conn: &Connection, 
        table_name: &str, 
        schema: &TableSchema
    ) -> Result<Vec<TypedDatabaseRow>> {
        let query = format!("SELECT * FROM {}", table_name);
        let mut stmt = conn.prepare(&query)
            .map_err(|e| anyhow!("Failed to prepare data query: {}", e))?;

        let rows = stmt.query_map([], |row| {
            let mut columns = Vec::new();
            
            for (idx, column_name) in schema.columns.iter().enumerate() {
                // Get raw value from database
                let value = row.get_ref(idx)
                    .map_err(|e| rusqlite::Error::InvalidColumnType(idx, column_name.clone(), e.to_string()))?;
                
                // Convert to Value enum
                let typed_value = match value {
                    rusqlite::types::ValueRef::Null => Value::Null,
                    rusqlite::types::ValueRef::Integer(i) => Value::Integer(i),
                    rusqlite::types::ValueRef::Real(f) => Value::Real(f),
                    rusqlite::types::ValueRef::Text(s) => Value::Text(String::from_utf8_lossy(s).to_string()),
                    rusqlite::types::ValueRef::Blob(b) => Value::Blob(b.to_vec()),
                };

                // Create typed column value
                let typed_column = TypedColumnValue::from_rusqlite_value(
                    column_name.clone(), 
                    &typed_value
                );
                columns.push(typed_column);
            }

            Ok(TypedDatabaseRow {
                table_name: table_name.to_string(),
                columns,
            })
        }).map_err(|e| anyhow!("Failed to execute data query: {}", e))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow!("Failed to collect table data: {}", e))?;

        Ok(rows)
    }

    /// Insert typed row back into database
    fn insert_typed_row(
        conn: &Connection, 
        schema: &TableSchema, 
        row: &TypedDatabaseRow
    ) -> Result<()> {
        let column_names = row.columns.iter()
            .map(|c| c.column_name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        
        let placeholders = (0..row.columns.len())
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(", ");
        
        let query = format!(
            "INSERT INTO {} ({}) VALUES ({})",
            schema.table_name, column_names, placeholders
        );

        let mut stmt = conn.prepare(&query)
            .map_err(|e| anyhow!("Failed to prepare insert query: {}", e))?;

        // Convert typed values back to rusqlite Values
        let values: Result<Vec<Value>> = row.columns.iter()
            .map(|col| col.to_rusqlite_value())
            .collect();

        let values = values?;
        let value_refs: Vec<&dyn rusqlite::ToSql> = values.iter()
            .map(|v| v as &dyn rusqlite::ToSql)
            .collect();

        stmt.execute(value_refs.as_slice())
            .map_err(|e| anyhow!("Failed to insert row: {}", e))?;

        Ok(())
    }

    /// Calculate checksum for backup integrity
    fn calculate_checksum(&self, data: &str) -> String {
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(data.as_bytes());
        hasher.finalize().to_hex().to_string()
    }

    /// Scan backup directory and update registry
    async fn scan_backup_directory(&self) -> Result<()> {
        let mut entries = fs::read_dir(&self.backup_path).await
            .map_err(|e| anyhow!("Failed to read backup directory: {}", e))?;

        while let Some(entry) = entries.next_entry().await
            .map_err(|e| anyhow!("Failed to read directory entry: {}", e))? {
            
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") &&
               path.file_name().and_then(|s| s.to_str()).map_or(false, |s| s.contains(".backup.")) {
                
                // Try to read backup metadata
                if let Ok(content) = fs::read_to_string(&path).await {
                    if let Ok(export) = serde_json::from_str::<DatabaseExport>(&content) {
                        let backup_id = export.metadata.backup_id.clone();
                        self.backup_registry.write().insert(backup_id, export.metadata);
                    }
                }
            }
        }

        Ok(())
    }

    /// Clean up old backups based on retention policy
    async fn cleanup_old_backups(&self) -> Result<()> {
        let cutoff_date = Utc::now() - chrono::Duration::days(self.retention_days as i64);
        let mut backups_to_delete = Vec::new();

        {
            let registry = self.backup_registry.read();
            for (backup_id, metadata) in registry.iter() {
                if metadata.created_at < cutoff_date {
                    backups_to_delete.push(backup_id.clone());
                }
            }
        }

        for backup_id in backups_to_delete {
            if let Err(e) = self.delete_backup(&backup_id).await {
                log::warn!("Failed to delete old backup {}: {}", backup_id, e);
            } else {
                log::info!("Cleaned up old backup: {}", backup_id);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_typed_column_value_conversion() {
        // Test different value types
        let int_value = Value::Integer(42);
        let typed_int = TypedColumnValue::from_rusqlite_value("test_int".to_string(), &int_value);
        assert_eq!(typed_int.column_type, SqlColumnType::Integer);
        assert_eq!(typed_int.value, serde_json::Value::Number(42.into()));

        let text_value = Value::Text("hello world".to_string());
        let typed_text = TypedColumnValue::from_rusqlite_value("test_text".to_string(), &text_value);
        assert_eq!(typed_text.column_type, SqlColumnType::Text);
        assert_eq!(typed_text.value, serde_json::Value::String("hello world".to_string()));

        let blob_value = Value::Blob(vec![1, 2, 3, 4]);
        let typed_blob = TypedColumnValue::from_rusqlite_value("test_blob".to_string(), &blob_value);
        assert_eq!(typed_blob.column_type, SqlColumnType::Blob);

        // Test round-trip conversion
        let recovered_int = typed_int.to_rusqlite_value().unwrap();
        assert_eq!(recovered_int, int_value);

        let recovered_text = typed_text.to_rusqlite_value().unwrap();
        assert_eq!(recovered_text, text_value);

        let recovered_blob = typed_blob.to_rusqlite_value().unwrap();
        assert_eq!(recovered_blob, blob_value);
    }

    #[tokio::test]
    async fn test_backup_manager_creation() {
        let temp_dir = tempdir().unwrap();
        let backup_manager = BackupManager::with_backup_path(temp_dir.path(), 30).unwrap();
        
        // Test that backup directory was created
        assert!(temp_dir.path().exists());
        
        // Test initial state
        let backups = backup_manager.list_backups().await.unwrap();
        assert_eq!(backups.len(), 0);
    }
}