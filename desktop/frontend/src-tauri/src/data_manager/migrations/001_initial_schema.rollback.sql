-- Rollback for 001_initial_schema.sql
-- This will drop all tables created in the initial schema
-- WARNING: This will permanently delete all data in these tables

-- Drop tables in reverse dependency order to avoid foreign key constraint issues
DROP TABLE IF EXISTS backup_metadata;
DROP TABLE IF EXISTS sync_metadata;
DROP TABLE IF EXISTS settings;
DROP TABLE IF EXISTS audit_logs;
DROP TABLE IF EXISTS assets;
DROP TABLE IF EXISTS spells;
DROP TABLE IF EXISTS items;
DROP TABLE IF EXISTS locations;
DROP TABLE IF EXISTS personality_profiles;
DROP TABLE IF EXISTS rulebooks;
DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS npcs;
DROP TABLE IF EXISTS characters;
DROP TABLE IF EXISTS campaigns;

-- Drop indexes if any were created (optional)
-- Note: SQLite automatically drops indexes when tables are dropped

-- Reset schema version (handled by migration system)