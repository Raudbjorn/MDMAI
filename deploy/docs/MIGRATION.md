# TTRPG Assistant MCP Server - Migration and Upgrade Guide

## Table of Contents
1. [Overview](#overview)
2. [Version Management](#version-management)
3. [Migration Process](#migration-process)
4. [Rollback Procedures](#rollback-procedures)
5. [Data Migration](#data-migration)
6. [Troubleshooting](#troubleshooting)

## Overview

This guide covers the migration and upgrade process for the TTRPG Assistant MCP Server, including version upgrades, data migration, and rollback procedures.

## Version Management

### Current Version

Check the current version:

```bash
python deploy/migration/version_manager.py --current

# Or using the Makefile
make deploy-status
```

### Version History

The application follows semantic versioning (MAJOR.MINOR.PATCH):

- **0.1.0** - Initial release with basic MCP functionality
- **0.2.0** - Added campaign management
- **0.3.0** - Added character generation
- **0.4.0** - Added personality system
- **0.5.0** - Added source management
- **0.6.0** - Added session tracking
- **0.7.0** - Added security features
- **0.8.0** - Performance optimizations
- **0.9.0** - Search enhancements
- **1.0.0** - First stable release

## Migration Process

### Pre-Migration Checklist

Before performing any migration:

1. **Check System Requirements**
   ```bash
   python deploy/scripts/check_requirements.py
   ```

2. **Create a Backup**
   ```bash
   python deploy/backup/backup_manager.py --create --type pre-update
   ```

3. **Stop the Service**
   ```bash
   sudo systemctl stop ttrpg-assistant
   ```

4. **Review Migration Plan**
   ```bash
   python deploy/migration/migrate.py 1.0.0 --dry-run
   ```

### Performing a Migration

#### Method 1: Automated Migration

```bash
# Run migration to specific version
python deploy/migration/migrate.py 1.0.0

# Or using the Makefile
make deploy-migrate VERSION=1.0.0
```

#### Method 2: Step-by-Step Migration

```bash
# 1. Create migration plan
python deploy/migration/migrate.py 1.0.0 --plan-only > migration_plan.json

# 2. Review the plan
cat migration_plan.json

# 3. Execute migration
python deploy/migration/migrate.py 1.0.0 --execute

# 4. Verify migration
python deploy/migration/migrate.py --verify
```

### Migration Options

```bash
python deploy/migration/migrate.py [target_version] [options]

Options:
  --data-dir PATH       Data directory (default: /var/lib/ttrpg-assistant)
  --config-dir PATH     Config directory (default: /etc/ttrpg-assistant)
  --backup-dir PATH     Backup directory (default: /var/lib/ttrpg-assistant/backup)
  --dry-run            Simulate migration without changes
  --skip-backup        Skip automatic backup (not recommended)
  --force              Force migration despite warnings
  --rollback ID        Rollback to specific backup
  --status             Show migration status
  --verify             Verify current installation
```

### Post-Migration Steps

After successful migration:

1. **Verify Installation**
   ```bash
   python deploy/migration/migrate.py --verify
   ```

2. **Start the Service**
   ```bash
   sudo systemctl start ttrpg-assistant
   ```

3. **Run Health Checks**
   ```bash
   curl http://localhost:8000/health
   ```

4. **Check Logs**
   ```bash
   tail -f /var/log/ttrpg-assistant/ttrpg-assistant.log
   ```

## Rollback Procedures

### Automatic Rollback

If a migration fails, automatic rollback will be attempted:

```bash
# The system will automatically:
1. Detect migration failure
2. Stop the migration process
3. Restore from pre-migration backup
4. Verify restoration
5. Report rollback status
```

### Manual Rollback

To manually rollback to a previous backup:

```bash
# List available backups
python deploy/backup/backup_manager.py --list

# Rollback to specific backup
python deploy/migration/migrate.py --rollback backup_20240315_143022

# Or restore using restore manager
python deploy/backup/restore_manager.py --restore backup_20240315_143022
```

### Emergency Recovery

If automatic rollback fails:

```bash
# 1. Stop all services
sudo systemctl stop ttrpg-assistant

# 2. Manually restore from backup
cd /var/lib/ttrpg-assistant
tar -xzf backup/backup_20240315_143022.tar.gz

# 3. Restore configuration
cd /etc/ttrpg-assistant
tar -xzf /var/lib/ttrpg-assistant/backup/config_backup.tar.gz

# 4. Reset version
echo '{"version": "0.9.0"}' > /etc/ttrpg-assistant/version.json

# 5. Restart services
sudo systemctl start ttrpg-assistant
```

## Data Migration

### Data Types

The following data types are migrated:

1. **Database Schema**
   - ChromaDB collections
   - Index structures
   - Metadata tables

2. **Application Data**
   - Campaigns
   - Characters
   - Sessions
   - Source documents

3. **Configuration**
   - Application settings
   - Security configurations
   - Feature flags

4. **Cache Data**
   - Search cache
   - Embedding cache
   - Result cache

### Custom Data Migration

For custom data migrations:

```python
# deploy/migration/custom_migrations/my_migration.py
from deploy.migration.data_migrator import DataMigrator

class MyCustomMigration(DataMigrator):
    def migrate_forward(self):
        """Migrate data to new format."""
        # Load old data
        old_data = self.load_data('old_format.json')
        
        # Transform data
        new_data = self.transform(old_data)
        
        # Save new data
        self.save_data('new_format.json', new_data)
        
    def migrate_backward(self):
        """Rollback to old format."""
        # Reverse the migration
        pass
```

### Data Export/Import

Export data before major upgrades:

```bash
# Export all data
python deploy/migration/data_migrator.py --export all_data.json

# Export specific data types
python deploy/migration/data_migrator.py --export campaigns.json --type campaigns

# Import data
python deploy/migration/data_migrator.py --import all_data.json
```

## Migration Scenarios

### Minor Version Update (e.g., 1.0.0 → 1.0.1)

```bash
# Simple update with minimal risk
git pull origin main
pip install -e . --upgrade
sudo systemctl restart ttrpg-assistant
```

### Minor Feature Update (e.g., 1.0.0 → 1.1.0)

```bash
# Standard migration process
make deploy-backup
make deploy-migrate VERSION=1.1.0
sudo systemctl restart ttrpg-assistant
```

### Major Version Update (e.g., 0.9.0 → 1.0.0)

```bash
# Full migration with verification
# 1. Create comprehensive backup
python deploy/backup/backup_manager.py --create --type pre-major-update

# 2. Stop services
sudo systemctl stop ttrpg-assistant

# 3. Run migration with verification
python deploy/migration/migrate.py 1.0.0 --verify

# 4. Test in staging
python -m src.main --test-mode

# 5. Start production
sudo systemctl start ttrpg-assistant
```

### Cross-Version Migration (e.g., 0.5.0 → 1.0.0)

```bash
# Multi-step migration
python deploy/migration/migrate.py 1.0.0 --plan-only

# Review migration path: 0.5.0 → 0.6.0 → 0.7.0 → 0.8.0 → 0.9.0 → 1.0.0

# Execute with checkpoints
python deploy/migration/migrate.py 1.0.0 --checkpoint-each-version
```

## Troubleshooting

### Common Migration Issues

#### 1. Insufficient Disk Space

```bash
# Check available space
df -h /var/lib/ttrpg-assistant

# Clean old backups
python deploy/backup/backup_manager.py --cleanup --older-than 30

# Retry migration
python deploy/migration/migrate.py 1.0.0
```

#### 2. Permission Errors

```bash
# Fix permissions
sudo chown -R ttrpg:ttrpg /var/lib/ttrpg-assistant
sudo chmod -R 750 /var/lib/ttrpg-assistant

# Retry with elevated privileges
sudo python deploy/migration/migrate.py 1.0.0
```

#### 3. Corrupted Data

```bash
# Verify data integrity
python deploy/migration/data_migrator.py --verify-integrity

# Restore from clean backup
python deploy/backup/restore_manager.py --restore last-known-good

# Skip corrupted data
python deploy/migration/migrate.py 1.0.0 --skip-corrupted
```

#### 4. Version Mismatch

```bash
# Reset version information
python deploy/migration/version_manager.py --reset --version 0.9.0

# Force migration
python deploy/migration/migrate.py 1.0.0 --force
```

### Migration Logs

Check migration logs for detailed information:

```bash
# Migration logs
tail -f /var/log/ttrpg-assistant/migration.log

# Backup logs
tail -f /var/lib/ttrpg-assistant/backup/backup.log

# System logs
journalctl -u ttrpg-assistant -f
```

### Recovery Options

If all else fails:

1. **Fresh Installation**
   ```bash
   # Backup user data
   python deploy/migration/data_migrator.py --export user_data.json
   
   # Reinstall
   sudo bash deploy/scripts/install.sh --clean
   
   # Import user data
   python deploy/migration/data_migrator.py --import user_data.json
   ```

2. **Manual Recovery**
   ```bash
   # Extract backup manually
   tar -xzf backup/backup_20240315_143022.tar.gz -C /tmp/recovery
   
   # Copy required files
   cp -r /tmp/recovery/data/* /var/lib/ttrpg-assistant/
   cp -r /tmp/recovery/config/* /etc/ttrpg-assistant/
   ```

3. **Support**
   - Check logs for specific error messages
   - Consult the troubleshooting guide
   - Open an issue on GitHub with migration logs

## Best Practices

1. **Always Create Backups**
   - Before any migration
   - After successful migration
   - Regular scheduled backups

2. **Test in Staging**
   - Use a staging environment
   - Test the migration process
   - Verify functionality

3. **Monitor After Migration**
   - Check application logs
   - Monitor performance metrics
   - Verify data integrity

4. **Document Custom Changes**
   - Record any manual modifications
   - Document custom migrations
   - Keep migration logs

5. **Plan Maintenance Windows**
   - Schedule during low-usage periods
   - Notify users in advance
   - Have rollback plan ready

## Support

For migration assistance:
- Documentation: https://github.com/Raudbjorn/MDMAI/wiki/Migration
- Issues: https://github.com/Raudbjorn/MDMAI/issues
- Migration logs: /var/log/ttrpg-assistant/migration.log