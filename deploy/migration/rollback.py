#!/usr/bin/env python3
"""
Rollback management for TTRPG Assistant MCP Server.
Handles backup creation and restoration.
"""

import json
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RollbackManager:
    """Manages backup and rollback operations."""
    
    def __init__(self, backup_dir: Path):
        """Initialize rollback manager.
        
        Args:
            backup_dir: Backup directory path
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.backup_dir / 'backup_metadata.json'
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load backup metadata.
        
        Returns:
            Metadata dictionary
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading backup metadata: {e}")
                
        return {'backups': []}
        
    def _save_metadata(self):
        """Save backup metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving backup metadata: {e}")
            
    def create_backup(self, data_dir: Path, config_dir: Path, 
                     metadata: Optional[Dict] = None) -> str:
        """Create a backup of data and configuration.
        
        Args:
            data_dir: Data directory to backup
            config_dir: Configuration directory to backup
            metadata: Additional metadata to include
            
        Returns:
            Backup ID
        """
        backup_id = datetime.now().strftime('backup_%Y%m%d_%H%M%S')
        backup_file = self.backup_dir / f"{backup_id}.tar.gz"
        
        logger.info(f"Creating backup: {backup_id}")
        
        try:
            # Create tar archive
            with tarfile.open(backup_file, 'w:gz') as tar:
                # Add data directory
                tar.add(data_dir, arcname='data')
                
                # Add config directory
                tar.add(config_dir, arcname='config')
                
                # Add metadata
                if metadata:
                    metadata_json = json.dumps(metadata, indent=2)
                    metadata_info = tarfile.TarInfo('metadata.json')
                    metadata_info.size = len(metadata_json)
                    tar.addfile(metadata_info, 
                               fileobj=json.dumps(metadata, indent=2).encode())
                    
            # Update backup metadata
            backup_info = {
                'id': backup_id,
                'created_at': datetime.now().isoformat(),
                'file': str(backup_file),
                'size': backup_file.stat().st_size,
                'metadata': metadata or {}
            }
            
            self.metadata['backups'].append(backup_info)
            self._save_metadata()
            
            logger.info(f"Backup created successfully: {backup_file}")
            return backup_id
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            if backup_file.exists():
                backup_file.unlink()
            raise
            
    def rollback(self, backup_id: str) -> bool:
        """Rollback to a specific backup.
        
        Args:
            backup_id: Backup ID to restore
            
        Returns:
            True if successful
        """
        # Find backup
        backup_info = None
        for backup in self.metadata['backups']:
            if backup['id'] == backup_id:
                backup_info = backup
                break
                
        if not backup_info:
            logger.error(f"Backup not found: {backup_id}")
            return False
            
        backup_file = Path(backup_info['file'])
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False
            
        try:
            logger.info(f"Rolling back to backup: {backup_id}")
            
            # Extract to temporary directory
            temp_dir = self.backup_dir / f"temp_{backup_id}"
            temp_dir.mkdir(exist_ok=True)
            
            with tarfile.open(backup_file, 'r:gz') as tar:
                tar.extractall(temp_dir)
                
            # Restore data and config
            # Note: In production, you'd want to stop services first
            data_source = temp_dir / 'data'
            config_source = temp_dir / 'config'
            
            if data_source.exists():
                # Backup current data before overwriting
                current_backup = self.backup_dir / f"pre_rollback_{datetime.now():%Y%m%d_%H%M%S}"
                if Path('/var/lib/ttrpg-assistant').exists():
                    shutil.copytree('/var/lib/ttrpg-assistant', current_backup)
                    
                # Restore data
                shutil.rmtree('/var/lib/ttrpg-assistant', ignore_errors=True)
                shutil.copytree(data_source, '/var/lib/ttrpg-assistant')
                
            if config_source.exists():
                # Restore config
                shutil.rmtree('/etc/ttrpg-assistant', ignore_errors=True)
                shutil.copytree(config_source, '/etc/ttrpg-assistant')
                
            # Cleanup
            shutil.rmtree(temp_dir)
            
            logger.info(f"Rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
            
    def list_backups(self) -> List[Dict]:
        """List available backups.
        
        Returns:
            List of backup information
        """
        return sorted(
            self.metadata['backups'],
            key=lambda x: x['created_at'],
            reverse=True
        )
        
    def get_backup_metadata(self, backup_id: str) -> Optional[Dict]:
        """Get metadata for a specific backup.
        
        Args:
            backup_id: Backup ID
            
        Returns:
            Backup metadata or None
        """
        for backup in self.metadata['backups']:
            if backup['id'] == backup_id:
                return backup
        return None
        
    def delete_backup(self, backup_id: str) -> bool:
        """Delete a specific backup.
        
        Args:
            backup_id: Backup ID to delete
            
        Returns:
            True if successful
        """
        for i, backup in enumerate(self.metadata['backups']):
            if backup['id'] == backup_id:
                backup_file = Path(backup['file'])
                if backup_file.exists():
                    backup_file.unlink()
                    
                del self.metadata['backups'][i]
                self._save_metadata()
                
                logger.info(f"Backup deleted: {backup_id}")
                return True
                
        logger.error(f"Backup not found: {backup_id}")
        return False
        
    def cleanup_old_backups(self, retention_days: int = 30):
        """Clean up backups older than retention period.
        
        Args:
            retention_days: Number of days to retain backups
        """
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        backups_to_delete = []
        for backup in self.metadata['backups']:
            created_at = datetime.fromisoformat(backup['created_at'])
            if created_at < cutoff_date:
                backups_to_delete.append(backup['id'])
                
        for backup_id in backups_to_delete:
            self.delete_backup(backup_id)
            
        if backups_to_delete:
            logger.info(f"Cleaned up {len(backups_to_delete)} old backups")
            
    def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity.
        
        Args:
            backup_id: Backup ID to verify
            
        Returns:
            True if backup is valid
        """
        backup_info = self.get_backup_metadata(backup_id)
        if not backup_info:
            return False
            
        backup_file = Path(backup_info['file'])
        if not backup_file.exists():
            return False
            
        try:
            # Verify tar archive
            with tarfile.open(backup_file, 'r:gz') as tar:
                members = tar.getmembers()
                if not members:
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False