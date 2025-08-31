//! Data models for TTRPG Assistant
//! 
//! This module defines all the data structures used in the TTRPG application,
//! including campaigns, characters, NPCs, rulebooks, sessions, and more.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use sqlx::FromRow;

/// Campaign information
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Campaign {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub system: String, // D&D 5e, Pathfinder, etc.
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub is_active: bool,
    pub settings: serde_json::Value, // Campaign-specific settings
    pub dm_id: Option<Uuid>, // Reference to DM character/user
    pub image_path: Option<String>,
    pub notes: Option<String>,
    pub status: CampaignStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "campaign_status", rename_all = "snake_case")]
pub enum CampaignStatus {
    Planning,
    Active,
    Paused,
    Completed,
    Archived,
}

/// Character sheet data
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Character {
    pub id: Uuid,
    pub campaign_id: Uuid,
    pub name: String,
    pub class: Option<String>,
    pub race: Option<String>,
    pub level: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub is_player_character: bool,
    pub owner_id: Option<String>, // Player identifier
    pub stats: serde_json::Value, // Ability scores, HP, AC, etc.
    pub inventory: serde_json::Value,
    pub spells: serde_json::Value,
    pub features: serde_json::Value,
    pub background: Option<String>,
    pub personality: serde_json::Value,
    pub image_path: Option<String>,
    pub notes: Option<String>,
    pub status: CharacterStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "character_status", rename_all = "snake_case")]
pub enum CharacterStatus {
    Active,
    Retired,
    Dead,
    Missing,
    Archived,
}

/// NPC (Non-Player Character) data
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Npc {
    pub id: Uuid,
    pub campaign_id: Uuid,
    pub name: String,
    pub race: Option<String>,
    pub class: Option<String>,
    pub role: NpcRole,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub stats: serde_json::Value,
    pub personality: serde_json::Value,
    pub relationships: serde_json::Value, // Relationships with PCs and other NPCs
    pub location: Option<String>,
    pub image_path: Option<String>,
    pub notes: Option<String>,
    pub is_alive: bool,
    pub importance: NpcImportance,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "npc_role", rename_all = "snake_case")]
pub enum NpcRole {
    Ally,
    Enemy,
    Neutral,
    Merchant,
    QuestGiver,
    Noble,
    Commoner,
    Authority,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "npc_importance", rename_all = "snake_case")]
pub enum NpcImportance {
    Major,
    Minor,
    Background,
}

/// Session tracking
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Session {
    pub id: Uuid,
    pub campaign_id: Uuid,
    pub session_number: i32,
    pub title: String,
    pub date_played: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub duration_minutes: Option<i32>,
    pub summary: Option<String>,
    pub notes: Option<String>,
    pub dm_notes: Option<String>, // Private DM notes
    pub participants: serde_json::Value, // List of participating characters
    pub events: serde_json::Value, // Major events that occurred
    pub loot: serde_json::Value,
    pub experience_awarded: Option<i32>,
    pub location: Option<String>,
    pub status: SessionStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "session_status", rename_all = "snake_case")]
pub enum SessionStatus {
    Planned,
    InProgress,
    Completed,
    Cancelled,
}

/// Rulebook and reference material
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Rulebook {
    pub id: Uuid,
    pub title: String,
    pub system: String,
    pub publisher: Option<String>,
    pub edition: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub file_path: Option<String>, // Path to PDF or other file
    pub file_size: Option<i64>,
    pub file_hash: Option<String>, // For integrity checking
    pub page_count: Option<i32>,
    pub description: Option<String>,
    pub tags: serde_json::Value,
    pub is_official: bool,
    pub copyright_info: Option<String>,
    pub access_level: AccessLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "access_level", rename_all = "snake_case")]
pub enum AccessLevel {
    Public,
    Private,
    Restricted,
}

/// Personality profiles for AI interactions
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct PersonalityProfile {
    pub id: Uuid,
    pub name: String,
    pub character_id: Option<Uuid>, // Link to character if applicable
    pub npc_id: Option<Uuid>, // Link to NPC if applicable
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub traits: serde_json::Value,
    pub speech_patterns: serde_json::Value,
    pub knowledge_areas: serde_json::Value,
    pub relationships: serde_json::Value,
    pub background_story: Option<String>,
    pub motivations: serde_json::Value,
    pub fears_and_flaws: serde_json::Value,
    pub voice_settings: serde_json::Value, // For TTS if available
    pub chat_history_summary: Option<String>,
    pub is_active: bool,
}

/// Locations and world-building
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Location {
    pub id: Uuid,
    pub campaign_id: Uuid,
    pub name: String,
    pub location_type: LocationType,
    pub parent_location_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub description: Option<String>,
    pub population: Option<i32>,
    pub government: Option<String>,
    pub notable_features: serde_json::Value,
    pub climate: Option<String>,
    pub economy: Option<String>,
    pub defenses: Option<String>,
    pub image_path: Option<String>,
    pub map_path: Option<String>,
    pub coordinates: Option<String>, // JSON with x,y or lat,lng
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "location_type", rename_all = "snake_case")]
pub enum LocationType {
    Continent,
    Country,
    Region,
    City,
    Town,
    Village,
    Building,
    Dungeon,
    Landmark,
    Other,
}

/// Items and equipment
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Item {
    pub id: Uuid,
    pub name: String,
    pub item_type: ItemType,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub description: Option<String>,
    pub rarity: ItemRarity,
    pub value_gp: Option<i32>,
    pub weight: Option<f64>,
    pub properties: serde_json::Value,
    pub stats: serde_json::Value, // Damage, AC bonus, etc.
    pub requirements: serde_json::Value,
    pub image_path: Option<String>,
    pub system: String,
    pub source_book: Option<String>,
    pub is_magical: bool,
    pub is_homebrew: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "item_type", rename_all = "snake_case")]
pub enum ItemType {
    Weapon,
    Armor,
    Shield,
    Tool,
    Consumable,
    Wondrous,
    Ring,
    Rod,
    Staff,
    Wand,
    Scroll,
    Potion,
    Gem,
    ArtObject,
    TradeGood,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "item_rarity", rename_all = "snake_case")]
pub enum ItemRarity {
    Common,
    Uncommon,
    Rare,
    VeryRare,
    Legendary,
    Artifact,
}

/// Spells and abilities
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Spell {
    pub id: Uuid,
    pub name: String,
    pub level: i32,
    pub school: SpellSchool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub casting_time: String,
    pub range: String,
    pub components: String,
    pub duration: String,
    pub description: String,
    pub at_higher_levels: Option<String>,
    pub classes: serde_json::Value, // Which classes can cast it
    pub ritual: bool,
    pub concentration: bool,
    pub system: String,
    pub source_book: Option<String>,
    pub is_homebrew: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "spell_school", rename_all = "snake_case")]
pub enum SpellSchool {
    Abjuration,
    Conjuration,
    Divination,
    Enchantment,
    Evocation,
    Illusion,
    Necromancy,
    Transmutation,
}

/// Game assets (images, maps, handouts)
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Asset {
    pub id: Uuid,
    pub campaign_id: Option<Uuid>,
    pub name: String,
    pub asset_type: AssetType,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub file_path: String,
    pub file_size: i64,
    pub file_hash: String,
    pub mime_type: String,
    pub description: Option<String>,
    pub tags: serde_json::Value,
    pub metadata: serde_json::Value, // Image dimensions, etc.
    pub is_public: bool,
    pub copyright_info: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "asset_type", rename_all = "snake_case")]
pub enum AssetType {
    Image,
    Map,
    Handout,
    Audio,
    Video,
    Document,
    Other,
}

/// Data operation audit log
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct AuditLog {
    pub id: Uuid,
    pub table_name: String,
    pub record_id: Uuid,
    pub operation: AuditOperation,
    pub user_id: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub old_values: Option<serde_json::Value>,
    pub new_values: Option<serde_json::Value>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "audit_operation", rename_all = "snake_case")]
pub enum AuditOperation {
    Create,
    Update,
    Delete,
    Import,
    Export,
    Backup,
    Restore,
}

/// Application settings and preferences
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Settings {
    pub id: Uuid,
    pub category: String,
    pub key: String,
    pub value: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub is_encrypted: bool,
    pub description: Option<String>,
}

/// Data synchronization metadata
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct SyncMetadata {
    pub table_name: String,
    pub record_id: Uuid,
    pub last_sync: DateTime<Utc>,
    pub sync_version: i64,
    pub sync_hash: String,
    pub conflict_resolution: Option<serde_json::Value>,
}

/// Data integrity check results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityCheckResult {
    pub check_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub tables_checked: Vec<String>,
    pub total_records: u64,
    pub corrupted_records: u64,
    pub missing_files: u64,
    pub hash_mismatches: u64,
    pub orphaned_records: u64,
    pub issues: Vec<IntegrityIssue>,
    pub overall_status: IntegrityStatus,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityIssue {
    pub issue_type: IntegrityIssueType,
    pub table_name: String,
    pub record_id: Option<Uuid>,
    pub field_name: Option<String>,
    pub description: String,
    pub severity: IntegritySeverity,
    pub auto_fixable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "integrity_issue_type", rename_all = "snake_case")]
pub enum IntegrityIssueType {
    MissingRecord,
    OrphanedRecord,
    InvalidForeignKey,
    MissingFile,
    HashMismatch,
    SchemaViolation,
    DataCorruption,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "integrity_severity", rename_all = "snake_case")]
pub enum IntegritySeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "integrity_status", rename_all = "snake_case")]
pub enum IntegrityStatus {
    Healthy,
    Warning,
    Error,
    Critical,
}

/// Backup metadata
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct BackupMetadata {
    pub id: Uuid,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub backup_type: BackupType,
    pub file_path: String,
    pub file_size: i64,
    pub compressed_size: i64,
    pub file_hash: String,
    pub description: Option<String>,
    pub database_version: String,
    pub app_version: String,
    pub compression_algorithm: String,
    pub encryption_enabled: bool,
    pub integrity_verified: bool,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "backup_type", rename_all = "snake_case")]
pub enum BackupType {
    Full,
    Incremental,
    Differential,
    Manual,
    Automatic,
    Shutdown,
}

/// Helper trait for model validation
pub trait Validate {
    fn validate(&self) -> Result<(), Vec<String>>;
}

/// Helper trait for model serialization
pub trait ModelSerialization {
    fn to_json(&self) -> Result<String, serde_json::Error>;
    fn from_json(json: &str) -> Result<Self, serde_json::Error>
    where
        Self: Sized;
}

// Implement ModelSerialization for all models
impl<T> ModelSerialization for T
where
    T: Serialize + for<'de> Deserialize<'de>,
{
    fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
    
    fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Common query parameters for listing operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListParams {
    pub limit: Option<u32>,
    pub offset: Option<u32>,
    pub sort_by: Option<String>,
    pub sort_order: Option<SortOrder>,
    pub filters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortOrder {
    Asc,
    Desc,
}

impl Default for ListParams {
    fn default() -> Self {
        Self {
            limit: Some(100),
            offset: Some(0),
            sort_by: None,
            sort_order: Some(SortOrder::Desc),
            filters: HashMap::new(),
        }
    }
}

/// Response wrapper for list operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResponse<T> {
    pub items: Vec<T>,
    pub total_count: u64,
    pub limit: u32,
    pub offset: u32,
    pub has_more: bool,
}