-- Initial schema for TTRPG Assistant Data Management System
-- Version: 1.0.0

-- Enable foreign key support
PRAGMA foreign_keys = ON;

-- Campaigns table
CREATE TABLE IF NOT EXISTS campaigns (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    system TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT 1,
    settings TEXT NOT NULL DEFAULT '{}',
    dm_id TEXT,
    image_path TEXT,
    notes TEXT,
    status TEXT NOT NULL DEFAULT 'planning' CHECK (status IN ('planning', 'active', 'paused', 'completed', 'archived'))
);

-- Characters table
CREATE TABLE IF NOT EXISTS characters (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    name TEXT NOT NULL,
    class TEXT,
    race TEXT,
    level INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    is_player_character BOOLEAN NOT NULL DEFAULT 1,
    owner_id TEXT,
    stats TEXT NOT NULL DEFAULT '{}',
    inventory TEXT NOT NULL DEFAULT '{}',
    spells TEXT NOT NULL DEFAULT '{}',
    features TEXT NOT NULL DEFAULT '{}',
    background TEXT,
    personality TEXT NOT NULL DEFAULT '{}',
    image_path TEXT,
    notes TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'retired', 'dead', 'missing', 'archived')),
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

-- NPCs table
CREATE TABLE IF NOT EXISTS npcs (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    name TEXT NOT NULL,
    race TEXT,
    class TEXT,
    role TEXT NOT NULL DEFAULT 'neutral' CHECK (role IN ('ally', 'enemy', 'neutral', 'merchant', 'quest_giver', 'noble', 'commoner', 'authority', 'other')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    stats TEXT NOT NULL DEFAULT '{}',
    personality TEXT NOT NULL DEFAULT '{}',
    relationships TEXT NOT NULL DEFAULT '{}',
    location TEXT,
    image_path TEXT,
    notes TEXT,
    is_alive BOOLEAN NOT NULL DEFAULT 1,
    importance TEXT NOT NULL DEFAULT 'minor' CHECK (importance IN ('major', 'minor', 'background')),
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    session_number INTEGER NOT NULL,
    title TEXT NOT NULL,
    date_played TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    duration_minutes INTEGER,
    summary TEXT,
    notes TEXT,
    dm_notes TEXT,
    participants TEXT NOT NULL DEFAULT '[]',
    events TEXT NOT NULL DEFAULT '[]',
    loot TEXT NOT NULL DEFAULT '[]',
    experience_awarded INTEGER,
    location TEXT,
    status TEXT NOT NULL DEFAULT 'planned' CHECK (status IN ('planned', 'in_progress', 'completed', 'cancelled')),
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE,
    UNIQUE(campaign_id, session_number)
);

-- Rulebooks table
CREATE TABLE IF NOT EXISTS rulebooks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    system TEXT NOT NULL,
    publisher TEXT,
    edition TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    file_path TEXT,
    file_size INTEGER,
    file_hash TEXT,
    page_count INTEGER,
    description TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    is_official BOOLEAN NOT NULL DEFAULT 1,
    copyright_info TEXT,
    access_level TEXT NOT NULL DEFAULT 'public' CHECK (access_level IN ('public', 'private', 'restricted'))
);

-- Personality profiles table
CREATE TABLE IF NOT EXISTS personality_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    character_id TEXT,
    npc_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    traits TEXT NOT NULL DEFAULT '{}',
    speech_patterns TEXT NOT NULL DEFAULT '{}',
    knowledge_areas TEXT NOT NULL DEFAULT '[]',
    relationships TEXT NOT NULL DEFAULT '{}',
    background_story TEXT,
    motivations TEXT NOT NULL DEFAULT '[]',
    fears_and_flaws TEXT NOT NULL DEFAULT '[]',
    voice_settings TEXT NOT NULL DEFAULT '{}',
    chat_history_summary TEXT,
    is_active BOOLEAN NOT NULL DEFAULT 1,
    FOREIGN KEY (character_id) REFERENCES characters(id) ON DELETE CASCADE,
    FOREIGN KEY (npc_id) REFERENCES npcs(id) ON DELETE CASCADE,
    CHECK (
        (character_id IS NOT NULL AND npc_id IS NULL) OR 
        (character_id IS NULL AND npc_id IS NOT NULL) OR
        (character_id IS NULL AND npc_id IS NULL)
    )
);

-- Locations table
CREATE TABLE IF NOT EXISTS locations (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    name TEXT NOT NULL,
    location_type TEXT NOT NULL DEFAULT 'other' CHECK (location_type IN ('continent', 'country', 'region', 'city', 'town', 'village', 'building', 'dungeon', 'landmark', 'other')),
    parent_location_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    description TEXT,
    population INTEGER,
    government TEXT,
    notable_features TEXT NOT NULL DEFAULT '[]',
    climate TEXT,
    economy TEXT,
    defenses TEXT,
    image_path TEXT,
    map_path TEXT,
    coordinates TEXT,
    notes TEXT,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_location_id) REFERENCES locations(id) ON DELETE SET NULL
);

-- Items table
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    item_type TEXT NOT NULL DEFAULT 'other' CHECK (item_type IN ('weapon', 'armor', 'shield', 'tool', 'consumable', 'wondrous', 'ring', 'rod', 'staff', 'wand', 'scroll', 'potion', 'gem', 'art_object', 'trade_good', 'other')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    description TEXT,
    rarity TEXT NOT NULL DEFAULT 'common' CHECK (rarity IN ('common', 'uncommon', 'rare', 'very_rare', 'legendary', 'artifact')),
    value_gp INTEGER,
    weight REAL,
    properties TEXT NOT NULL DEFAULT '{}',
    stats TEXT NOT NULL DEFAULT '{}',
    requirements TEXT NOT NULL DEFAULT '{}',
    image_path TEXT,
    system TEXT NOT NULL,
    source_book TEXT,
    is_magical BOOLEAN NOT NULL DEFAULT 0,
    is_homebrew BOOLEAN NOT NULL DEFAULT 0
);

-- Spells table
CREATE TABLE IF NOT EXISTS spells (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    level INTEGER NOT NULL,
    school TEXT NOT NULL CHECK (school IN ('abjuration', 'conjuration', 'divination', 'enchantment', 'evocation', 'illusion', 'necromancy', 'transmutation')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    casting_time TEXT NOT NULL,
    range TEXT NOT NULL,
    components TEXT NOT NULL,
    duration TEXT NOT NULL,
    description TEXT NOT NULL,
    at_higher_levels TEXT,
    classes TEXT NOT NULL DEFAULT '[]',
    ritual BOOLEAN NOT NULL DEFAULT 0,
    concentration BOOLEAN NOT NULL DEFAULT 0,
    system TEXT NOT NULL,
    source_book TEXT,
    is_homebrew BOOLEAN NOT NULL DEFAULT 0
);

-- Assets table (images, maps, handouts, etc.)
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    campaign_id TEXT,
    name TEXT NOT NULL,
    asset_type TEXT NOT NULL DEFAULT 'other' CHECK (asset_type IN ('image', 'map', 'handout', 'audio', 'video', 'document', 'other')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    file_hash TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    description TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    is_public BOOLEAN NOT NULL DEFAULT 0,
    copyright_info TEXT,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE SET NULL
);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id TEXT PRIMARY KEY,
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('create', 'update', 'delete', 'import', 'export', 'backup', 'restore')),
    user_id TEXT,
    timestamp TEXT NOT NULL,
    old_values TEXT,
    new_values TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);

-- Settings table
CREATE TABLE IF NOT EXISTS settings (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    is_encrypted BOOLEAN NOT NULL DEFAULT 0,
    description TEXT,
    UNIQUE(category, key)
);

-- Sync metadata table
CREATE TABLE IF NOT EXISTS sync_metadata (
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    last_sync TEXT NOT NULL,
    sync_version INTEGER NOT NULL DEFAULT 1,
    sync_hash TEXT NOT NULL,
    conflict_resolution TEXT,
    PRIMARY KEY (table_name, record_id)
);

-- Backup metadata table
CREATE TABLE IF NOT EXISTS backup_metadata (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    backup_type TEXT NOT NULL CHECK (backup_type IN ('full', 'incremental', 'differential', 'manual', 'automatic', 'shutdown')),
    file_path TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    compressed_size INTEGER NOT NULL,
    file_hash TEXT NOT NULL,
    description TEXT,
    database_version TEXT NOT NULL,
    app_version TEXT NOT NULL,
    compression_algorithm TEXT NOT NULL DEFAULT 'zstd',
    encryption_enabled BOOLEAN NOT NULL DEFAULT 0,
    integrity_verified BOOLEAN NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}'
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_characters_campaign_id ON characters(campaign_id);
CREATE INDEX IF NOT EXISTS idx_characters_owner_id ON characters(owner_id);
CREATE INDEX IF NOT EXISTS idx_characters_is_player_character ON characters(is_player_character);

CREATE INDEX IF NOT EXISTS idx_npcs_campaign_id ON npcs(campaign_id);
CREATE INDEX IF NOT EXISTS idx_npcs_role ON npcs(role);
CREATE INDEX IF NOT EXISTS idx_npcs_importance ON npcs(importance);

CREATE INDEX IF NOT EXISTS idx_sessions_campaign_id ON sessions(campaign_id);
CREATE INDEX IF NOT EXISTS idx_sessions_date_played ON sessions(date_played);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);

CREATE INDEX IF NOT EXISTS idx_rulebooks_system ON rulebooks(system);
CREATE INDEX IF NOT EXISTS idx_rulebooks_is_official ON rulebooks(is_official);

CREATE INDEX IF NOT EXISTS idx_personality_profiles_character_id ON personality_profiles(character_id);
CREATE INDEX IF NOT EXISTS idx_personality_profiles_npc_id ON personality_profiles(npc_id);
CREATE INDEX IF NOT EXISTS idx_personality_profiles_is_active ON personality_profiles(is_active);

CREATE INDEX IF NOT EXISTS idx_locations_campaign_id ON locations(campaign_id);
CREATE INDEX IF NOT EXISTS idx_locations_parent_location_id ON locations(parent_location_id);
CREATE INDEX IF NOT EXISTS idx_locations_type ON locations(location_type);

CREATE INDEX IF NOT EXISTS idx_items_type ON items(item_type);
CREATE INDEX IF NOT EXISTS idx_items_rarity ON items(rarity);
CREATE INDEX IF NOT EXISTS idx_items_system ON items(system);
CREATE INDEX IF NOT EXISTS idx_items_is_magical ON items(is_magical);

CREATE INDEX IF NOT EXISTS idx_spells_level ON spells(level);
CREATE INDEX IF NOT EXISTS idx_spells_school ON spells(school);
CREATE INDEX IF NOT EXISTS idx_spells_system ON spells(system);

CREATE INDEX IF NOT EXISTS idx_assets_campaign_id ON assets(campaign_id);
CREATE INDEX IF NOT EXISTS idx_assets_type ON assets(asset_type);
CREATE INDEX IF NOT EXISTS idx_assets_is_public ON assets(is_public);
CREATE INDEX IF NOT EXISTS idx_assets_file_hash ON assets(file_hash);

CREATE INDEX IF NOT EXISTS idx_audit_logs_table_name ON audit_logs(table_name);
CREATE INDEX IF NOT EXISTS idx_audit_logs_record_id ON audit_logs(record_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_operation ON audit_logs(operation);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);

CREATE INDEX IF NOT EXISTS idx_settings_category ON settings(category);
CREATE INDEX IF NOT EXISTS idx_settings_is_encrypted ON settings(is_encrypted);

-- Insert default settings
INSERT OR IGNORE INTO settings (id, category, key, value, created_at, updated_at, is_encrypted, description)
VALUES 
    ('default_app_version', 'app', 'version', '"1.0.0"', datetime('now'), datetime('now'), 0, 'Application version'),
    ('default_db_version', 'database', 'schema_version', '1', datetime('now'), datetime('now'), 0, 'Database schema version'),
    ('default_encryption_enabled', 'security', 'encryption_enabled', 'true', datetime('now'), datetime('now'), 0, 'Whether encryption is enabled'),
    ('default_backup_retention', 'backup', 'retention_days', '30', datetime('now'), datetime('now'), 0, 'How many days to keep backups'),
    ('default_cache_size', 'performance', 'cache_size_mb', '256', datetime('now'), datetime('now'), 0, 'Cache size in MB'),
    ('default_log_level', 'logging', 'level', '"info"', datetime('now'), datetime('now'), 0, 'Logging level');

-- Create triggers for updated_at timestamps
CREATE TRIGGER IF NOT EXISTS update_campaigns_timestamp 
    AFTER UPDATE ON campaigns
BEGIN
    UPDATE campaigns SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_characters_timestamp 
    AFTER UPDATE ON characters
BEGIN
    UPDATE characters SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_npcs_timestamp 
    AFTER UPDATE ON npcs
BEGIN
    UPDATE npcs SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_sessions_timestamp 
    AFTER UPDATE ON sessions
BEGIN
    UPDATE sessions SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_rulebooks_timestamp 
    AFTER UPDATE ON rulebooks
BEGIN
    UPDATE rulebooks SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_personality_profiles_timestamp 
    AFTER UPDATE ON personality_profiles
BEGIN
    UPDATE personality_profiles SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_locations_timestamp 
    AFTER UPDATE ON locations
BEGIN
    UPDATE locations SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_items_timestamp 
    AFTER UPDATE ON items
BEGIN
    UPDATE items SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_spells_timestamp 
    AFTER UPDATE ON spells
BEGIN
    UPDATE spells SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_assets_timestamp 
    AFTER UPDATE ON assets
BEGIN
    UPDATE assets SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_settings_timestamp 
    AFTER UPDATE ON settings
BEGIN
    UPDATE settings SET updated_at = datetime('now') WHERE id = NEW.id;
END;