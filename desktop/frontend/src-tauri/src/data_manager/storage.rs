//! SQLite-based encrypted storage implementation
//! 
//! This module provides encrypted local storage using SQLite with comprehensive
//! CRUD operations for all TTRPG data models.

use super::*;
use sqlx::{SqlitePool, Row};
use std::collections::HashMap;

/// SQLite storage manager with encryption support
pub struct DataStorage {
    pub pool: SqlitePool,
    encryption: Arc<EncryptionManager>,
    config: DataManagerConfig,
    prepared_statements: Arc<RwLock<HashMap<String, String>>>,
}

impl DataStorage {
    /// Create new storage instance
    pub async fn new(
        config: &DataManagerConfig, 
        encryption: &Arc<EncryptionManager>
    ) -> DataResult<Self> {
        // Ensure database directory exists
        if let Some(parent) = config.database_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| DataError::Database {
                    message: format!("Failed to create database directory: {}", e),
                })?;
        }

        // Create optimized connection pool with better settings
        let database_url = format!("sqlite:{}", config.database_path.to_string_lossy());
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(10) // Increase for better concurrency
            .min_connections(2)  // Keep minimum connections ready
            .acquire_timeout(std::time::Duration::from_secs(10))
            .idle_timeout(Some(std::time::Duration::from_secs(600)))
            .max_lifetime(Some(std::time::Duration::from_secs(1800)))
            .connect(&database_url)
            .await
            .map_err(|e| DataError::Database {
                message: format!("Failed to connect to database: {}", e),
            })?;

        // Enable foreign key constraints
        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(&pool)
            .await
            .map_err(|e| DataError::Database {
                message: format!("Failed to enable foreign keys: {}", e),
            })?;

        // Enable WAL mode and optimize SQLite settings for performance
        sqlx::query("PRAGMA journal_mode = WAL")
            .execute(&pool)
            .await
            .map_err(|e| DataError::Database {
                message: format!("Failed to enable WAL mode: {}", e),
            })?;
        
        // Optimize SQLite performance settings
        sqlx::query("PRAGMA synchronous = NORMAL") // Faster than FULL, still safe
            .execute(&pool)
            .await
            .map_err(|e| DataError::Database {
                message: format!("Failed to set synchronous mode: {}", e),
            })?;
        
        sqlx::query("PRAGMA cache_size = -64000") // 64MB cache
            .execute(&pool)
            .await
            .map_err(|e| DataError::Database {
                message: format!("Failed to set cache size: {}", e),
            })?;
        
        sqlx::query("PRAGMA temp_store = MEMORY") // Use memory for temp tables
            .execute(&pool)
            .await
            .map_err(|e| DataError::Database {
                message: format!("Failed to set temp store: {}", e),
            })?;
        
        sqlx::query("PRAGMA mmap_size = 268435456") // 256MB memory-mapped I/O
            .execute(&pool)
            .await
            .map_err(|e| DataError::Database {
                message: format!("Failed to set mmap size: {}", e),
            })?;

        let storage = Self {
            pool,
            encryption: encryption.clone(),
            config: config.clone(),
            prepared_statements: Arc::new(RwLock::new(HashMap::new())),
        };

        // Initialize database schema
        storage.initialize_schema().await?;

        Ok(storage)
    }

    /// Initialize database schema
    async fn initialize_schema(&self) -> DataResult<()> {
        // Run schema migrations
        let schema = include_str!("migrations/001_initial_schema.sql");
        
        // Split schema into individual statements and execute
        for statement in schema.split(';') {
            let statement = statement.trim();
            if !statement.is_empty() {
                sqlx::query(statement)
                    .execute(&self.pool)
                    .await
                    .map_err(|e| DataError::Database {
                        message: format!("Failed to execute schema statement: {}", e),
                    })?;
            }
        }

        log::info!("Database schema initialized successfully");
        Ok(())
    }

    /// Close the database connection
    pub async fn close(&self) -> DataResult<()> {
        self.pool.close().await;
        Ok(())
    }

    // Campaign operations
    pub async fn create_campaign(&self, campaign: &Campaign) -> DataResult<Campaign> {
        // Use async encryption for better performance
        let settings_encrypted = if self.config.encryption_enabled {
            tokio::task::spawn_blocking({
                let encryption = self.encryption.clone();
                let settings = campaign.settings.clone();
                move || encryption.encrypt_json(&settings)
            }).await
                .map_err(|e| DataError::Encryption {
                    message: format!("Encryption task failed: {}", e),
                })?
                .map_err(|e| DataError::Encryption {
                    message: format!("Failed to encrypt settings: {:?}", e),
                })?
        } else {
            campaign.settings.to_string()
        };

        let notes_encrypted = if let Some(notes) = &campaign.notes {
            if self.config.encryption_enabled {
                Some(self.encryption.encrypt_string(notes)?)
            } else {
                Some(notes.clone())
            }
        } else {
            None
        };

        let result = sqlx::query(
            r#"
            INSERT INTO campaigns (
                id, name, description, system, created_at, updated_at,
                is_active, settings, dm_id, image_path, notes, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#
        )
        .bind(&campaign.id)
        .bind(&campaign.name)
        .bind(&campaign.description)
        .bind(&campaign.system)
        .bind(&campaign.created_at)
        .bind(&campaign.updated_at)
        .bind(&campaign.is_active)
        .bind(&settings_encrypted)
        .bind(&campaign.dm_id)
        .bind(&campaign.image_path)
        .bind(&notes_encrypted)
        .bind(&campaign.status)
        .execute(&self.pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to create campaign: {}", e),
        })?;

        // Log audit entry
        self.log_audit_entry(
            "campaigns".to_string(),
            campaign.id,
            AuditOperation::Create,
            None,
            None,
            Some(serde_json::json!({"name": campaign.name})),
        ).await?;

        Ok(campaign.clone())
    }

    pub async fn get_campaign(&self, id: Uuid) -> DataResult<Option<Campaign>> {
        let row = sqlx::query(
            "SELECT * FROM campaigns WHERE id = ?"
        )
        .bind(&id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to get campaign: {}", e),
        })?;

        if let Some(row) = row {
            let settings_str: String = row.get("settings");
            let settings = if self.config.encryption_enabled {
                self.encryption.decrypt_json(&settings_str)?
            } else {
                serde_json::from_str(&settings_str).unwrap_or(serde_json::json!({}))
            };

            let notes = if let Some(encrypted_notes) = row.get::<Option<String>, _>("notes") {
                if self.config.encryption_enabled {
                    Some(self.encryption.decrypt_string(&encrypted_notes)?)
                } else {
                    Some(encrypted_notes)
                }
            } else {
                None
            };

            let status_str: String = row.get("status");
            let status: CampaignStatus = serde_json::from_str(&format!("\"{}\"", status_str))
                .map_err(|e| DataError::Database {
                    message: format!("Invalid campaign status: {}", e),
                })?;

            Ok(Some(Campaign {
                id: row.get("id"),
                name: row.get("name"),
                description: row.get("description"),
                system: row.get("system"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                is_active: row.get("is_active"),
                settings,
                dm_id: row.get("dm_id"),
                image_path: row.get("image_path"),
                notes,
                status,
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn list_campaigns(&self, params: &ListParams) -> DataResult<ListResponse<Campaign>> {
        let limit = params.limit.unwrap_or(100) as i32;
        let offset = params.offset.unwrap_or(0) as i32;

        // Build query with filters
        let mut query = "SELECT * FROM campaigns WHERE 1=1".to_string();
        let mut count_query = "SELECT COUNT(*) as count FROM campaigns WHERE 1=1".to_string();

        // Add filters
        for (key, value) in &params.filters {
            match key.as_str() {
                "is_active" => {
                    if let Some(is_active) = value.as_bool() {
                        query.push_str(&format!(" AND is_active = {}", is_active));
                        count_query.push_str(&format!(" AND is_active = {}", is_active));
                    }
                }
                "system" => {
                    if let Some(system) = value.as_str() {
                        query.push_str(&format!(" AND system = '{}'", system));
                        count_query.push_str(&format!(" AND system = '{}'", system));
                    }
                }
                "status" => {
                    if let Some(status) = value.as_str() {
                        query.push_str(&format!(" AND status = '{}'", status));
                        count_query.push_str(&format!(" AND status = '{}'", status));
                    }
                }
                _ => {} // Ignore unknown filters
            }
        }

        // Add sorting
        if let Some(sort_by) = &params.sort_by {
            let order = match params.sort_order {
                Some(SortOrder::Asc) => "ASC",
                _ => "DESC",
            };
            query.push_str(&format!(" ORDER BY {} {}", sort_by, order));
        } else {
            query.push_str(" ORDER BY updated_at DESC");
        }

        // Add pagination
        query.push_str(&format!(" LIMIT {} OFFSET {}", limit, offset));

        // Get total count
        let count_row = sqlx::query(&count_query)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| DataError::Database {
                message: format!("Failed to count campaigns: {}", e),
            })?;
        let total_count: i64 = count_row.get("count");

        // Get campaigns
        let rows = sqlx::query(&query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DataError::Database {
                message: format!("Failed to list campaigns: {}", e),
            })?;

        let mut campaigns = Vec::new();
        for row in rows {
            let settings_str: String = row.get("settings");
            let settings = if self.config.encryption_enabled {
                self.encryption.decrypt_json(&settings_str)?
            } else {
                serde_json::from_str(&settings_str).unwrap_or(serde_json::json!({}))
            };

            let notes = if let Some(encrypted_notes) = row.get::<Option<String>, _>("notes") {
                if self.config.encryption_enabled {
                    Some(self.encryption.decrypt_string(&encrypted_notes)?)
                } else {
                    Some(encrypted_notes)
                }
            } else {
                None
            };

            let status_str: String = row.get("status");
            let status: CampaignStatus = serde_json::from_str(&format!("\"{}\"", status_str))
                .map_err(|e| DataError::Database {
                    message: format!("Invalid campaign status: {}", e),
                })?;

            campaigns.push(Campaign {
                id: row.get("id"),
                name: row.get("name"),
                description: row.get("description"),
                system: row.get("system"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                is_active: row.get("is_active"),
                settings,
                dm_id: row.get("dm_id"),
                image_path: row.get("image_path"),
                notes,
                status,
            });
        }

        Ok(ListResponse {
            items: campaigns,
            total_count: total_count as u64,
            limit: limit as u32,
            offset: offset as u32,
            has_more: (offset + limit) < total_count as i32,
        })
    }

    pub async fn update_campaign(&self, id: Uuid, campaign: &Campaign) -> DataResult<Campaign> {
        let settings_encrypted = if self.config.encryption_enabled {
            self.encryption.encrypt_json(&campaign.settings)?
        } else {
            campaign.settings.to_string()
        };

        let notes_encrypted = if let Some(notes) = &campaign.notes {
            if self.config.encryption_enabled {
                Some(self.encryption.encrypt_string(notes)?)
            } else {
                Some(notes.clone())
            }
        } else {
            None
        };

        let result = sqlx::query(
            r#"
            UPDATE campaigns SET
                name = ?, description = ?, system = ?, updated_at = ?,
                is_active = ?, settings = ?, dm_id = ?, image_path = ?,
                notes = ?, status = ?
            WHERE id = ?
            "#
        )
        .bind(&campaign.name)
        .bind(&campaign.description)
        .bind(&campaign.system)
        .bind(&Utc::now())
        .bind(&campaign.is_active)
        .bind(&settings_encrypted)
        .bind(&campaign.dm_id)
        .bind(&campaign.image_path)
        .bind(&notes_encrypted)
        .bind(&campaign.status)
        .bind(&id)
        .execute(&self.pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to update campaign: {}", e),
        })?;

        if result.rows_affected() == 0 {
            return Err(DataError::NotFound {
                resource: format!("Campaign with id {}", id),
            });
        }

        // Log audit entry
        self.log_audit_entry(
            "campaigns".to_string(),
            id,
            AuditOperation::Update,
            None,
            None,
            Some(serde_json::json!({"name": campaign.name})),
        ).await?;

        Ok(campaign.clone())
    }

    pub async fn delete_campaign(&self, id: Uuid) -> DataResult<()> {
        let result = sqlx::query(
            "DELETE FROM campaigns WHERE id = ?"
        )
        .bind(&id)
        .execute(&self.pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to delete campaign: {}", e),
        })?;

        if result.rows_affected() == 0 {
            return Err(DataError::NotFound {
                resource: format!("Campaign with id {}", id),
            });
        }

        // Log audit entry
        self.log_audit_entry(
            "campaigns".to_string(),
            id,
            AuditOperation::Delete,
            None,
            None,
            None,
        ).await?;

        Ok(())
    }

    // Character operations
    pub async fn create_character(&self, character: &Character) -> DataResult<Character> {
        let stats_encrypted = if self.config.encryption_enabled {
            self.encryption.encrypt_json(&character.stats)?
        } else {
            character.stats.to_string()
        };

        let inventory_encrypted = if self.config.encryption_enabled {
            self.encryption.encrypt_json(&character.inventory)?
        } else {
            character.inventory.to_string()
        };

        let spells_encrypted = if self.config.encryption_enabled {
            self.encryption.encrypt_json(&character.spells)?
        } else {
            character.spells.to_string()
        };

        let features_encrypted = if self.config.encryption_enabled {
            self.encryption.encrypt_json(&character.features)?
        } else {
            character.features.to_string()
        };

        let personality_encrypted = if self.config.encryption_enabled {
            self.encryption.encrypt_json(&character.personality)?
        } else {
            character.personality.to_string()
        };

        let notes_encrypted = if let Some(notes) = &character.notes {
            if self.config.encryption_enabled {
                Some(self.encryption.encrypt_string(notes)?)
            } else {
                Some(notes.clone())
            }
        } else {
            None
        };

        let result = sqlx::query(
            r#"
            INSERT INTO characters (
                id, campaign_id, name, class, race, level, created_at, updated_at,
                is_player_character, owner_id, stats, inventory, spells, features,
                background, personality, image_path, notes, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#
        )
        .bind(&character.id)
        .bind(&character.campaign_id)
        .bind(&character.name)
        .bind(&character.class)
        .bind(&character.race)
        .bind(&character.level)
        .bind(&character.created_at)
        .bind(&character.updated_at)
        .bind(&character.is_player_character)
        .bind(&character.owner_id)
        .bind(&stats_encrypted)
        .bind(&inventory_encrypted)
        .bind(&spells_encrypted)
        .bind(&features_encrypted)
        .bind(&character.background)
        .bind(&personality_encrypted)
        .bind(&character.image_path)
        .bind(&notes_encrypted)
        .bind(&character.status)
        .execute(&self.pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to create character: {}", e),
        })?;

        // Log audit entry
        self.log_audit_entry(
            "characters".to_string(),
            character.id,
            AuditOperation::Create,
            None,
            None,
            Some(serde_json::json!({"name": character.name})),
        ).await?;

        Ok(character.clone())
    }

    // TODO: Implement remaining CRUD operations for all models
    // This is a foundational implementation showing the pattern

    /// Log an audit entry
    async fn log_audit_entry(
        &self,
        table_name: String,
        record_id: Uuid,
        operation: AuditOperation,
        user_id: Option<String>,
        old_values: Option<serde_json::Value>,
        new_values: Option<serde_json::Value>,
    ) -> DataResult<()> {
        let audit_id = Uuid::new_v4();
        let timestamp = Utc::now();
        let metadata = serde_json::json!({
            "source": "data_storage",
            "encrypted": self.config.encryption_enabled
        });

        sqlx::query(
            r#"
            INSERT INTO audit_logs (
                id, table_name, record_id, operation, user_id, timestamp,
                old_values, new_values, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#
        )
        .bind(&audit_id)
        .bind(&table_name)
        .bind(&record_id)
        .bind(&operation)
        .bind(&user_id)
        .bind(&timestamp)
        .bind(&old_values)
        .bind(&new_values)
        .bind(&metadata)
        .execute(&self.pool)
        .await
        .map_err(|e| DataError::Database {
            message: format!("Failed to log audit entry: {}", e),
        })?;

        Ok(())
    }

    /// Get database statistics
    pub async fn get_database_stats(&self) -> DataResult<serde_json::Value> {
        // Get table counts
        let tables = [
            "campaigns", "characters", "npcs", "sessions", "rulebooks",
            "personality_profiles", "locations", "items", "spells", "assets"
        ];

        let mut stats = serde_json::Map::new();
        
        for table in &tables {
            let query = format!("SELECT COUNT(*) as count FROM {}", table);
            let row = sqlx::query(&query)
                .fetch_one(&self.pool)
                .await
                .map_err(|e| DataError::Database {
                    message: format!("Failed to get count for table {}: {}", table, e),
                })?;
            
            let count: i64 = row.get("count");
            stats.insert(table.to_string(), serde_json::json!(count));
        }

        // Get database file size
        if let Ok(metadata) = std::fs::metadata(&self.config.database_path) {
            stats.insert("database_size_bytes".to_string(), serde_json::json!(metadata.len()));
        }

        Ok(serde_json::Value::Object(stats))
    }

    /// Perform database maintenance (vacuum, analyze, etc.)
    pub async fn perform_maintenance(&self) -> DataResult<()> {
        // Vacuum database to reclaim space
        sqlx::query("VACUUM")
            .execute(&self.pool)
            .await
            .map_err(|e| DataError::Database {
                message: format!("Failed to vacuum database: {}", e),
            })?;

        // Update statistics
        sqlx::query("ANALYZE")
            .execute(&self.pool)
            .await
            .map_err(|e| DataError::Database {
                message: format!("Failed to analyze database: {}", e),
            })?;

        log::info!("Database maintenance completed successfully");
        Ok(())
    }
}