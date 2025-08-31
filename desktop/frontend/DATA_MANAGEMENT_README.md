# TTRPG Assistant Data Management System

This document describes the comprehensive data management system implemented for Phase 23.6 of the MDMAI project. The system provides robust local data persistence, backup/restore functionality, encryption, data integrity validation, and performance optimization for the TTRPG Assistant desktop application.

## Architecture Overview

The data management system is built with a layered architecture that provides:

### Core Components

1. **Data Storage Layer** (`storage.rs`)
   - SQLite database with comprehensive schema for TTRPG data
   - Encrypted storage for sensitive information
   - CRUD operations for campaigns, characters, NPCs, sessions, rulebooks, etc.
   - Foreign key integrity and transaction support

2. **Encryption Manager** (`encryption.rs`)
   - AES-256-GCM encryption for data at rest
   - Argon2 key derivation from user passwords
   - Secure key storage and rotation
   - File-level encryption for documents and assets

3. **Backup & Restore System** (`backup.rs`)
   - Full and incremental backups with compression (zstd)
   - Versioned backup storage with metadata
   - Cross-platform backup portability
   - Integrity verification and rollback capabilities

4. **Data Integrity Checker** (`integrity.rs`)
   - Comprehensive validation of data consistency
   - Hash verification for files and database records
   - Orphaned record detection and foreign key validation
   - Automatic repair for fixable issues

5. **File Manager** (`file_manager.rs`)
   - Secure storage of rulebooks, images, maps, and documents
   - Deduplication and storage optimization
   - Metadata extraction and thumbnail generation
   - Cross-platform file path handling

6. **Cache Manager** (`cache.rs`)
   - Multi-level caching (memory + disk)
   - LRU eviction policies with TTL support
   - Compression for large cached items
   - Performance metrics and monitoring

7. **Migration System** (`migration.rs`)
   - Database schema evolution management
   - Version tracking and rollback support
   - Automatic migration execution on startup
   - Migration integrity verification

## Data Models

The system supports comprehensive TTRPG data types:

### Primary Entities
- **Campaigns**: Game campaigns with system, settings, and status
- **Characters**: Player and NPC character sheets with stats, inventory, spells
- **Sessions**: Game session tracking with notes, events, and participants
- **Rulebooks**: PDF storage with indexing and search capabilities
- **Assets**: Images, maps, handouts, and media files
- **Locations**: World-building data with hierarchical relationships

### Supporting Data
- **Personality Profiles**: AI interaction data for NPCs and characters
- **Items & Spells**: Game mechanics data with searchable attributes
- **Audit Logs**: Complete operation history and change tracking
- **Settings**: Application configuration and user preferences

## Key Features

### 🔐 Security & Encryption
- **AES-256-GCM encryption** for sensitive data
- **Argon2 key derivation** from user passwords
- **Secure key rotation** and management
- **File-level encryption** for documents
- **Access control** and permission management

### 💾 Backup & Recovery
- **Automated backups** with configurable schedules
- **Incremental backups** to minimize storage usage
- **Compression** using zstd for space efficiency
- **Cross-platform portability** for data migration
- **Integrity verification** before and after restore
- **Safety backups** created before major operations

### 🔍 Data Integrity
- **Real-time validation** of data consistency
- **Hash verification** for file corruption detection
- **Foreign key integrity** checks
- **Orphaned record cleanup**
- **Automatic repair** for fixable issues
- **Comprehensive reporting** with severity levels

### 🚀 Performance Optimization
- **Multi-level caching** with memory and disk tiers
- **LRU eviction** with TTL expiration
- **Query optimization** with prepared statements
- **Index strategies** for fast lookups
- **Compression** for large data items
- **Background processing** for maintenance tasks

### 📊 Monitoring & Analytics
- **Performance metrics** for all operations
- **Cache hit rates** and memory usage
- **Storage utilization** tracking
- **Error reporting** and recovery suggestions
- **Database statistics** and health monitoring

## Usage Examples

### TypeScript Frontend Integration

```typescript
import { dataManager, type Campaign } from '$lib/data-manager-client';

// Initialize with encryption
await dataManager.initializeWithPassword('your-secure-password');

// Create a new campaign
const campaign: Campaign = {
    id: crypto.randomUUID(),
    name: "Lost Mines of Phandelver",
    system: "D&D 5e",
    description: "A classic D&D adventure",
    status: "active",
    is_active: true,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    settings: {
        variant_rules: ["feats", "multiclassing"],
        starting_level: 1
    }
};

await dataManager.createCampaign(campaign);

// List campaigns with filtering
const campaigns = await dataManager.listCampaigns({
    limit: 10,
    filters: { is_active: true, system: "D&D 5e" }
});

// Create a backup
await dataManager.createBackup('manual', 'Before major session');

// Check data integrity
const integrity = await dataManager.checkIntegrity();
if (integrity.issues.length > 0) {
    console.log('Issues found:', integrity.issues);
    await dataManager.repairIntegrity(
        integrity.issues.filter(i => i.auto_fixable)
    );
}
```

### Svelte Component Integration

```svelte
<script lang="ts">
    import { dataManager } from '$lib/data-manager-client';
    import DataManagerDashboard from '$lib/components/DataManagerDashboard.svelte';
    
    let initialized = $state(false);
    
    onMount(async () => {
        try {
            await dataManager.initialize();
            initialized = true;
        } catch (error) {
            console.error('Failed to initialize data manager:', error);
        }
    });
</script>

{#if initialized}
    <DataManagerDashboard />
{:else}
    <div>Initializing data management system...</div>
{/if}
```

## Configuration

### Data Manager Configuration

```typescript
const config: DataManagerConfig = {
    data_dir: "/path/to/app/data",
    database_path: "/path/to/app/data/app.db",
    files_dir: "/path/to/app/data/files",
    backup_dir: "/path/to/app/data/backups",
    cache_dir: "/path/to/app/data/cache",
    encryption_enabled: true,
    auto_backup_interval: 60, // minutes
    max_backup_count: 10,
    cache_size_limit_mb: 500,
    integrity_checking_enabled: true,
    integrity_check_interval_hours: 24
};
```

### Environment Setup

The system automatically creates necessary directories and database schema on first run. Required permissions:
- Read/write access to the data directory
- Sufficient disk space for databases and backups
- Network access for potential cloud sync features (future)

## Database Schema

The SQLite schema includes:

### Core Tables
- `campaigns` - Game campaigns with encrypted settings
- `characters` - Character sheets with stats, inventory, spells
- `npcs` - Non-player characters with personality data
- `sessions` - Game session records with notes and events
- `rulebooks` - PDF files with metadata and search indexing
- `assets` - Media files with hash verification

### System Tables
- `audit_logs` - Complete operation history
- `settings` - Application configuration
- `sync_metadata` - Data synchronization tracking
- `backup_metadata` - Backup file information
- `schema_migrations` - Database version tracking

### Indexes and Constraints
- Foreign key relationships maintained
- Composite indexes for common queries
- Unique constraints where appropriate
- Check constraints for data validation

## Performance Characteristics

### Target Metrics
- **Database operations**: < 50ms for CRUD operations
- **Cache hit rate**: > 85% for frequently accessed data
- **Backup creation**: < 2 minutes for typical datasets
- **Integrity checks**: < 30 seconds for full validation
- **Storage efficiency**: > 70% compression for backups

### Scalability Limits
- **Campaigns**: Unlimited (within disk space)
- **Characters per campaign**: 1000+ recommended maximum
- **Sessions per campaign**: 500+ recommended maximum
- **File storage**: Limited by available disk space
- **Cache size**: Configurable, default 500MB

## Security Considerations

### Data Protection
- **Encryption at rest** using industry-standard AES-256-GCM
- **Key derivation** with Argon2 and proper salt generation
- **Secure deletion** of sensitive data from memory
- **File permissions** restricted to application user
- **Audit logging** for all data operations

### Threat Model
- Protects against local file system access by unauthorized users
- Provides data integrity verification against corruption
- Enables secure backup and restore operations
- Does not protect against memory dumps or keystroke loggers
- Cloud sync security depends on transport layer implementation

## Maintenance and Operations

### Regular Maintenance
- **Automatic backups** run according to configured schedule
- **Cache cleanup** removes expired entries hourly
- **Integrity checks** run daily (configurable)
- **Database optimization** with VACUUM and ANALYZE
- **Log rotation** to prevent unbounded growth

### Troubleshooting
- **Integrity issues**: Use the built-in repair functionality
- **Performance problems**: Check cache hit rates and database statistics
- **Storage issues**: Run storage optimization and duplicate cleanup
- **Backup failures**: Verify disk space and permissions
- **Encryption problems**: Ensure password is correct and key file exists

## Future Enhancements

### Planned Features
- **Cloud synchronization** with conflict resolution
- **Multi-user support** with role-based permissions
- **Advanced search** with full-text indexing
- **API integrations** with online TTRPG tools
- **Plugin system** for custom data types
- **Advanced analytics** and campaign insights

### Extension Points
- **Custom data models** can be added via migration system
- **Additional file types** supported through file manager
- **Custom validation rules** can be implemented
- **External backup targets** can be integrated
- **Custom encryption providers** can be plugged in

## Installation Notes

The data management system is integrated into the Tauri application and requires no additional installation steps. Dependencies are automatically managed through Cargo.toml:

```toml
# Core data dependencies
sqlx = { version = "0.8", features = ["runtime-tokio-rustls", "sqlite", "chrono", "uuid", "json"] }
uuid = { version = "1.0", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }

# Encryption dependencies  
argon2 = "0.5"
aes-gcm = "0.10"
rand = "0.8"
sha2 = "0.10"
blake3 = "1.0"

# Compression and backup
zstd = "0.13"
tar = "0.4"
walkdir = "2.0"
```

This comprehensive data management system provides enterprise-grade local data persistence for the TTRPG Assistant, ensuring data integrity, security, and performance while maintaining ease of use for end users.