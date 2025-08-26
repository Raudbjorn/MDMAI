# TTRPG Assistant MCP Server - Backup and Restore Guide

## Table of Contents
1. [Overview](#overview)
2. [Backup Strategy](#backup-strategy)
3. [Creating Backups](#creating-backups)
4. [Restoring from Backups](#restoring-from-backups)
5. [Automated Backups](#automated-backups)
6. [Backup Management](#backup-management)
7. [Disaster Recovery](#disaster-recovery)

## Overview

This guide covers backup and restore procedures for the TTRPG Assistant MCP Server, ensuring data safety and quick recovery in case of failures.

## Backup Strategy

### What Gets Backed Up

1. **Application Data**
   - ChromaDB database
   - Campaign data
   - Character information
   - Session records
   - Source documents

2. **Configuration**
   - Application settings (.env)
   - System configuration (config.yaml)
   - Security settings
   - Feature flags

3. **Cache Data** (Optional)
   - Search cache
   - Embedding cache
   - Results cache

4. **Metadata**
   - Version information
   - Migration history
   - Audit logs

### Backup Types

| Type | Purpose | Frequency | Retention |
|------|---------|-----------|-----------|
| **Full Backup** | Complete system backup | Weekly | 4 weeks |
| **Incremental** | Changes since last backup | Daily | 7 days |
| **Pre-Update** | Before system updates | As needed | 2 versions |
| **Manual** | User-initiated | On demand | User defined |
| **Emergency** | Before critical operations | As needed | Until verified |

## Creating Backups

### Manual Backup

#### Using Backup Manager

```bash
# Create a manual backup
python deploy/backup/backup_manager.py --create

# With description
python deploy/backup/backup_manager.py --create \
  --description "Before major configuration change"

# Specify backup type
python deploy/backup/backup_manager.py --create \
  --type manual \
  --description "Weekly backup"
```

#### Using Makefile

```bash
# Quick backup
make deploy-backup

# With options
make deploy-backup TYPE=pre-update DESC="Before v1.0 upgrade"
```

### Backup Options

```bash
python deploy/backup/backup_manager.py [options]

Options:
  --create              Create new backup
  --type TYPE          Backup type: manual, scheduled, pre-update
  --description DESC   Backup description
  --data-dir PATH      Data directory to backup
  --backup-dir PATH    Backup storage location
  --config-dir PATH    Configuration directory
  --compress TYPE      Compression: gz, bz2, xz, none
  --encrypt            Encrypt backup (requires key)
  --verify             Verify after creation
```

### Selective Backup

Backup specific components:

```bash
# Backup only data
python deploy/backup/backup_manager.py --create \
  --include data \
  --exclude cache,logs

# Backup only configuration
python deploy/backup/backup_manager.py --create \
  --include config

# Backup campaigns only
python deploy/backup/backup_manager.py --create \
  --include data/campaigns
```

## Restoring from Backups

### Basic Restore

```bash
# List available backups
python deploy/backup/backup_manager.py --list

# Restore specific backup
python deploy/backup/restore_manager.py \
  --restore backup_20240315_143022
```

### Restore Options

```bash
python deploy/backup/restore_manager.py [options]

Options:
  --restore ID         Backup ID to restore
  --data-dir PATH     Target data directory
  --config-dir PATH   Target config directory
  --dry-run          Simulate without changes
  --no-verify        Skip verification
  --force            Force restore without confirmation
  --partial          Restore specific components only
```

### Selective Restore

Restore specific components:

```bash
# Restore only data
python deploy/backup/restore_manager.py \
  --restore backup_20240315_143022 \
  --partial data

# Restore only configuration
python deploy/backup/restore_manager.py \
  --restore backup_20240315_143022 \
  --partial config

# Restore specific directory
python deploy/backup/restore_manager.py \
  --restore backup_20240315_143022 \
  --partial data/campaigns
```

### Verification

Verify backup before restoration:

```bash
# Verify backup integrity
python deploy/backup/backup_manager.py \
  --verify backup_20240315_143022

# Get backup information
python deploy/backup/backup_manager.py \
  --info backup_20240315_143022
```

## Automated Backups

### Cron Schedule

Set up automated backups using cron:

```bash
# Edit crontab
crontab -e

# Add backup schedules
# Daily incremental backup at 2 AM
0 2 * * * /opt/ttrpg-assistant/deploy/backup/backup_manager.py --create --type scheduled

# Weekly full backup on Sunday at 3 AM
0 3 * * 0 /opt/ttrpg-assistant/deploy/backup/backup_manager.py --create --type scheduled --full

# Monthly archive on the 1st at 4 AM
0 4 1 * * /opt/ttrpg-assistant/deploy/backup/backup_manager.py --create --type archive
```

### Systemd Timer

Create systemd timer for automated backups:

```ini
# /etc/systemd/system/ttrpg-backup.service
[Unit]
Description=TTRPG Assistant Backup
After=network.target

[Service]
Type=oneshot
User=ttrpg
ExecStart=/opt/ttrpg-assistant/deploy/backup/backup_manager.py --create --type scheduled
StandardOutput=append:/var/log/ttrpg-assistant/backup.log
StandardError=append:/var/log/ttrpg-assistant/backup-error.log

# /etc/systemd/system/ttrpg-backup.timer
[Unit]
Description=Daily TTRPG Assistant Backup
Requires=ttrpg-backup.service

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

Enable the timer:

```bash
sudo systemctl enable ttrpg-backup.timer
sudo systemctl start ttrpg-backup.timer
```

### Backup Script

Create custom backup script:

```bash
#!/bin/bash
# /usr/local/bin/ttrpg-backup.sh

BACKUP_DIR="/var/lib/ttrpg-assistant/backup"
LOG_FILE="/var/log/ttrpg-assistant/backup.log"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> $LOG_FILE
}

# Create backup
log "Starting scheduled backup"
python /opt/ttrpg-assistant/deploy/backup/backup_manager.py \
  --create \
  --type scheduled \
  --description "Automated daily backup"

# Cleanup old backups
log "Cleaning up old backups"
python /opt/ttrpg-assistant/deploy/backup/backup_manager.py \
  --cleanup \
  --older-than 30

# Verify latest backup
LATEST=$(python /opt/ttrpg-assistant/deploy/backup/backup_manager.py --latest)
log "Verifying backup: $LATEST"
python /opt/ttrpg-assistant/deploy/backup/backup_manager.py --verify $LATEST

log "Backup complete"
```

## Backup Management

### Storage Management

```bash
# Check backup storage usage
du -sh /var/lib/ttrpg-assistant/backup

# List backups by size
python deploy/backup/backup_manager.py --list --sort-by size

# Remove old backups
python deploy/backup/backup_manager.py --cleanup --older-than 30

# Remove specific backup
python deploy/backup/backup_manager.py --delete backup_20240315_143022
```

### Backup Rotation

Configure automatic rotation:

```python
# deploy/backup/rotation_policy.json
{
  "policies": [
    {
      "type": "daily",
      "retain": 7,
      "compress_after": 1
    },
    {
      "type": "weekly",
      "retain": 4,
      "compress_after": 7
    },
    {
      "type": "monthly",
      "retain": 3,
      "archive": true
    }
  ],
  "max_total_size": "10GB",
  "max_backup_count": 50
}
```

Apply rotation policy:

```bash
python deploy/backup/backup_manager.py --apply-rotation
```

### Remote Backup

#### S3 Storage

```bash
# Configure S3 backup
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export S3_BUCKET="ttrpg-backups"

# Upload backup to S3
aws s3 cp backup_20240315_143022.tar.gz \
  s3://ttrpg-backups/backup_20240315_143022.tar.gz

# Sync all backups
aws s3 sync /var/lib/ttrpg-assistant/backup \
  s3://ttrpg-backups/ \
  --exclude "*.log"
```

#### SFTP/SCP

```bash
# Copy backup to remote server
scp backup_20240315_143022.tar.gz \
  user@backup-server:/backups/ttrpg/

# Automated remote backup
rsync -avz /var/lib/ttrpg-assistant/backup/ \
  user@backup-server:/backups/ttrpg/
```

## Disaster Recovery

### Recovery Plan

1. **Assess the Situation**
   ```bash
   # Check system status
   systemctl status ttrpg-assistant
   
   # Check data integrity
   python deploy/migration/data_migrator.py --verify-integrity
   
   # Review logs
   tail -n 100 /var/log/ttrpg-assistant/error.log
   ```

2. **Identify Recovery Point**
   ```bash
   # List available backups
   python deploy/backup/backup_manager.py --list
   
   # Find last good backup
   python deploy/backup/backup_manager.py --find-valid
   ```

3. **Perform Recovery**
   ```bash
   # Stop services
   sudo systemctl stop ttrpg-assistant
   
   # Restore from backup
   python deploy/backup/restore_manager.py \
     --restore backup_20240315_143022 \
     --verify
   
   # Start services
   sudo systemctl start ttrpg-assistant
   ```

4. **Verify Recovery**
   ```bash
   # Check service health
   curl http://localhost:8000/health
   
   # Verify data
   python deploy/migration/data_migrator.py --verify-integrity
   
   # Test functionality
   python -m src.main --test-mode
   ```

### Emergency Recovery

When standard recovery fails:

```bash
#!/bin/bash
# Emergency recovery script

# 1. Create emergency backup of current state
tar -czf /tmp/emergency_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
  /var/lib/ttrpg-assistant \
  /etc/ttrpg-assistant

# 2. Find and extract last known good backup
LAST_GOOD=$(find /var/lib/ttrpg-assistant/backup -name "*.tar.gz" \
  -exec tar -tzf {} \; 2>/dev/null | head -1)

# 3. Extract to recovery directory
mkdir -p /tmp/recovery
tar -xzf $LAST_GOOD -C /tmp/recovery

# 4. Restore data
rm -rf /var/lib/ttrpg-assistant/*
cp -r /tmp/recovery/data/* /var/lib/ttrpg-assistant/

# 5. Restore configuration
cp -r /tmp/recovery/config/* /etc/ttrpg-assistant/

# 6. Fix permissions
chown -R ttrpg:ttrpg /var/lib/ttrpg-assistant
chown -R ttrpg:ttrpg /etc/ttrpg-assistant

# 7. Restart service
systemctl restart ttrpg-assistant
```

### Data Recovery Tools

#### Partial Recovery

Recover specific data types:

```python
# recover_campaigns.py
import json
import tarfile
from pathlib import Path

def recover_campaigns(backup_file, output_dir):
    """Recover only campaign data from backup."""
    with tarfile.open(backup_file, 'r:gz') as tar:
        for member in tar.getmembers():
            if 'campaigns' in member.name:
                tar.extract(member, output_dir)
    print(f"Campaigns recovered to {output_dir}")

recover_campaigns('backup_20240315_143022.tar.gz', '/tmp/recovery')
```

#### Data Repair

Fix corrupted data:

```python
# repair_data.py
import json
from pathlib import Path

def repair_json_file(file_path):
    """Attempt to repair corrupted JSON file."""
    try:
        with open(file_path) as f:
            data = json.load(f)
        print(f"{file_path}: OK")
    except json.JSONDecodeError as e:
        print(f"{file_path}: Corrupted - {e}")
        # Attempt repair
        with open(file_path) as f:
            content = f.read()
        # Try to fix common issues
        content = content.replace('}\n{', '},\n{')
        content = content.strip()
        if not content.endswith(']') and not content.endswith('}'):
            content += '}'
        # Save repaired version
        with open(file_path + '.repaired', 'w') as f:
            f.write(content)
        print(f"Repair attempted: {file_path}.repaired")

# Repair all JSON files
for json_file in Path('/var/lib/ttrpg-assistant').rglob('*.json'):
    repair_json_file(json_file)
```

## Best Practices

### Backup Guidelines

1. **3-2-1 Rule**
   - 3 copies of important data
   - 2 different storage media
   - 1 offsite backup

2. **Regular Testing**
   - Test restore process monthly
   - Verify backup integrity weekly
   - Document recovery time

3. **Documentation**
   - Keep backup schedule documented
   - Record recovery procedures
   - Maintain contact information

4. **Security**
   - Encrypt sensitive backups
   - Restrict backup access
   - Audit backup operations

### Monitoring

Set up backup monitoring:

```bash
# Check backup status
python deploy/backup/backup_manager.py --status

# Monitor backup size trends
python deploy/backup/backup_manager.py --stats

# Alert on backup failures
python deploy/backup/backup_manager.py --check-health
```

### Backup Checklist

- [ ] Automated backups configured
- [ ] Retention policy defined
- [ ] Remote backup location set
- [ ] Recovery procedure tested
- [ ] Monitoring alerts configured
- [ ] Documentation updated
- [ ] Team trained on recovery
- [ ] Encryption enabled for sensitive data
- [ ] Backup verification scheduled
- [ ] Disaster recovery plan documented

## Troubleshooting

### Common Issues

1. **Backup Fails - Insufficient Space**
   ```bash
   # Check available space
   df -h /var/lib/ttrpg-assistant/backup
   
   # Clean up old backups
   python deploy/backup/backup_manager.py --cleanup --force
   
   # Use compression
   python deploy/backup/backup_manager.py --create --compress xz
   ```

2. **Restore Fails - Checksum Mismatch**
   ```bash
   # Skip verification
   python deploy/backup/restore_manager.py --restore backup_id --no-verify
   
   # Try alternative backup
   python deploy/backup/backup_manager.py --find-valid
   ```

3. **Slow Backup/Restore**
   ```bash
   # Use parallel compression
   export PIGZ_THREADS=4
   python deploy/backup/backup_manager.py --create --parallel
   
   # Exclude large cache files
   python deploy/backup/backup_manager.py --create --exclude cache
   ```

## Support

For backup assistance:
- Documentation: https://github.com/Raudbjorn/MDMAI/wiki/Backup
- Issues: https://github.com/Raudbjorn/MDMAI/issues
- Backup logs: /var/log/ttrpg-assistant/backup.log