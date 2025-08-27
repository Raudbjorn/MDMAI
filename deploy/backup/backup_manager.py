#!/usr/bin/env python3
"""
Backup manager for TTRPG Assistant MCP Server.
Handles automated backups and scheduling.
"""

import os
import sys
import json
import shutil
import tarfile
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hashlib
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BackupManager:
    """Manages automated backup operations."""
    
    def __init__(self, data_dir: Path, backup_dir: Path, config_dir: Path):
        """Initialize backup manager.
        
        Args:
            data_dir: Data directory to backup
            backup_dir: Directory to store backups
            config_dir: Configuration directory
        """
        self.data_dir = Path(data_dir)
        self.backup_dir = Path(backup_dir)
        self.config_dir = Path(config_dir)
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup registry
        self.registry_file = self.backup_dir / 'backup_registry.json'
        self.registry = self._load_registry()
        
    def _load_registry(self) -> Dict:
        """Load backup registry.
        
        Returns:
            Registry dictionary
        """
        if self.registry_file.exists():
            try:
                with open(self.registry_file) as f:
                    return json.load(f)
            except (IOError, OSError) as e:
                logger.error(f"Error reading registry file: {e}")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing registry JSON: {e}")
                
        return {
            'backups': [],
            'settings': {
                'retention_days': 30,
                'max_backups': 10,
                'compression': 'gz'
            }
        }
        
    def _save_registry(self):
        """Save backup registry."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except (IOError, OSError) as e:
            logger.error(f"Error writing registry file: {e}")
        except TypeError as e:
            logger.error(f"Error serializing registry data: {e}")
            
    def create_backup(self, backup_type: str = 'manual', 
                     description: str = '') -> Optional[str]:
        """Create a new backup.
        
        Args:
            backup_type: Type of backup (manual, scheduled, pre-update)
            description: Optional description
            
        Returns:
            Backup ID if successful
        """
        backup_id = datetime.now().strftime('backup_%Y%m%d_%H%M%S')
        logger.info(f"Creating {backup_type} backup: {backup_id}")
        
        try:
            # Calculate data size
            data_size = self._calculate_directory_size(self.data_dir)
            
            # Check available space
            available_space = shutil.disk_usage(self.backup_dir).free
            if available_space < data_size * 1.5:  # Need 1.5x space for safety
                logger.error(f"Insufficient space for backup. Need {data_size * 1.5 / (1024**3):.1f} GB")
                return None
                
            # Create backup archive
            backup_file = self.backup_dir / f"{backup_id}.tar.gz"
            
            with tarfile.open(backup_file, 'w:gz') as tar:
                # Add data directory
                logger.info(f"Backing up data directory: {self.data_dir}")
                tar.add(self.data_dir, arcname='data', filter=self._filter_backup)
                
                # Add configuration
                logger.info(f"Backing up configuration: {self.config_dir}")
                tar.add(self.config_dir, arcname='config', filter=self._filter_backup)
                
                # Add backup metadata
                metadata = {
                    'backup_id': backup_id,
                    'type': backup_type,
                    'description': description,
                    'created_at': datetime.now().isoformat(),
                    'data_dir': str(self.data_dir),
                    'config_dir': str(self.config_dir),
                    'version': self._get_application_version()
                }
                
                metadata_json = json.dumps(metadata, indent=2).encode()
                metadata_info = tarfile.TarInfo('metadata.json')
                metadata_info.size = len(metadata_json)
                tar.addfile(metadata_info, fileobj=json.dumps(metadata, indent=2).encode())
                
            # Calculate checksum
            checksum = self._calculate_checksum(backup_file)
            
            # Update registry
            backup_entry = {
                'id': backup_id,
                'file': str(backup_file),
                'type': backup_type,
                'description': description,
                'created_at': datetime.now().isoformat(),
                'size': backup_file.stat().st_size,
                'checksum': checksum,
                'version': self._get_application_version()
            }
            
            self.registry['backups'].append(backup_entry)
            self._save_registry()
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            logger.info(f"Backup created successfully: {backup_id}")
            logger.info(f"Backup size: {backup_file.stat().st_size / (1024**2):.1f} MB")
            
            return backup_id
            
        except (IOError, OSError) as e:
            logger.error(f"File system error during backup creation: {e}")
            # Clean up partial backup
            if 'backup_file' in locals() and backup_file.exists():
                backup_file.unlink()
            return None
        except tarfile.TarError as e:
            logger.error(f"Archive error during backup creation: {e}")
            # Clean up partial backup
            if 'backup_file' in locals() and backup_file.exists():
                backup_file.unlink()
            return None
            
    def _filter_backup(self, tarinfo):
        """Filter function for tar to exclude certain files.
        
        Args:
            tarinfo: Tar info object
            
        Returns:
            tarinfo or None to exclude
        """
        # Exclude patterns
        exclude_patterns = [
            '__pycache__',
            '.pyc',
            '.pyo',
            '.git',
            '.DS_Store',
            'Thumbs.db',
            '*.log',
            '*.tmp',
            '*.cache'
        ]
        
        for pattern in exclude_patterns:
            if pattern in tarinfo.name:
                return None
                
        return tarinfo
        
    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory.
        
        Args:
            directory: Directory path
            
        Returns:
            Size in bytes
        """
        total_size = 0
        for path in directory.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size
        
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file.
        
        Args:
            file_path: File to checksum
            
        Returns:
            Checksum string
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
        
    def _get_application_version(self) -> str:
        """Get current application version.
        
        Returns:
            Version string
        """
        try:
            version_file = self.config_dir / 'version.json'
            if version_file.exists():
                with open(version_file) as f:
                    data = json.load(f)
                    return data.get('version', '0.1.0')
        except Exception:
            pass
        return '0.1.0'
        
    def _cleanup_old_backups(self):
        """Remove old backups based on retention policy."""
        retention_days = self.registry['settings']['retention_days']
        max_backups = self.registry['settings']['max_backups']
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # Sort backups by date
        backups = sorted(
            self.registry['backups'],
            key=lambda x: x['created_at']
        )
        
        backups_to_remove = []
        
        # Remove by age
        for backup in backups:
            created_at = datetime.fromisoformat(backup['created_at'])
            if created_at < cutoff_date:
                backups_to_remove.append(backup)
                
        # Remove by count (keep most recent)
        if len(backups) - len(backups_to_remove) > max_backups:
            excess_count = len(backups) - len(backups_to_remove) - max_backups
            for backup in backups[:excess_count]:
                if backup not in backups_to_remove:
                    backups_to_remove.append(backup)
                    
        # Remove backup files and registry entries
        for backup in backups_to_remove:
            backup_file = Path(backup['file'])
            if backup_file.exists():
                backup_file.unlink()
                logger.info(f"Removed old backup: {backup['id']}")
                
            self.registry['backups'].remove(backup)
            
        if backups_to_remove:
            self._save_registry()
            
    def list_backups(self) -> List[Dict]:
        """List all available backups.
        
        Returns:
            List of backup entries
        """
        return sorted(
            self.registry['backups'],
            key=lambda x: x['created_at'],
            reverse=True
        )
        
    def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity.
        
        Args:
            backup_id: Backup ID to verify
            
        Returns:
            True if backup is valid
        """
        # Find backup in registry
        backup = None
        for entry in self.registry['backups']:
            if entry['id'] == backup_id:
                backup = entry
                break
                
        if not backup:
            logger.error(f"Backup not found: {backup_id}")
            return False
            
        backup_file = Path(backup['file'])
        
        # Check file exists
        if not backup_file.exists():
            logger.error(f"Backup file missing: {backup_file}")
            return False
            
        # Verify checksum
        calculated_checksum = self._calculate_checksum(backup_file)
        if calculated_checksum != backup['checksum']:
            logger.error(f"Checksum mismatch for backup {backup_id}")
            return False
            
        # Verify tar integrity
        try:
            with tarfile.open(backup_file, 'r:gz') as tar:
                # Check for required members
                members = tar.getnames()
                if 'data' not in str(members) or 'config' not in str(members):
                    logger.error(f"Backup missing required components")
                    return False
                    
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False
            
        logger.info(f"Backup {backup_id} verified successfully")
        return True
        
    def get_backup_info(self, backup_id: str) -> Optional[Dict]:
        """Get detailed information about a backup.
        
        Args:
            backup_id: Backup ID
            
        Returns:
            Backup information or None
        """
        for backup in self.registry['backups']:
            if backup['id'] == backup_id:
                # Add additional info
                backup_file = Path(backup['file'])
                if backup_file.exists():
                    backup['exists'] = True
                    backup['size_mb'] = backup_file.stat().st_size / (1024**2)
                else:
                    backup['exists'] = False
                    
                return backup
                
        return None
        
    def delete_backup(self, backup_id: str, force: bool = False) -> bool:
        """Delete a specific backup.
        
        Args:
            backup_id: Backup ID to delete
            force: Skip confirmation
            
        Returns:
            True if successful
        """
        # Find backup in registry
        backup_to_delete = None
        backup_index = -1
        
        for i, backup in enumerate(self.registry['backups']):
            if backup['id'] == backup_id:
                backup_to_delete = backup
                backup_index = i
                break
        
        if not backup_to_delete:
            logger.error(f"Backup not found: {backup_id}")
            return False
        
        backup_file = Path(backup_to_delete['file'])
        
        # Confirm deletion unless forced
        if not force:
            logger.info(f"Backup to delete: {backup_id}")
            logger.info(f"  Date: {backup_to_delete['timestamp']}")
            logger.info(f"  Type: {backup_to_delete['type']}")
            logger.info(f"  Description: {backup_to_delete.get('description', 'N/A')}")
            if backup_file.exists():
                size_mb = backup_file.stat().st_size / (1024**2)
                logger.info(f"  Size: {size_mb:.1f} MB")
        
        try:
            # Delete backup file
            if backup_file.exists():
                backup_file.unlink()
                logger.info(f"Deleted backup file: {backup_file}")
            else:
                logger.warning(f"Backup file not found: {backup_file}")
            
            # Remove from registry
            del self.registry['backups'][backup_index]
            self._save_registry()
            
            logger.info(f"Backup {backup_id} deleted successfully")
            return True
            
        except (IOError, OSError) as e:
            logger.error(f"Error deleting backup: {e}")
            return False

def main():
    """Main entry point for backup manager."""
    parser = argparse.ArgumentParser(
        description="TTRPG Assistant Backup Manager"
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('/var/lib/ttrpg-assistant'),
        help='Data directory to backup'
    )
    
    parser.add_argument(
        '--backup-dir',
        type=Path,
        default=Path('/var/lib/ttrpg-assistant/backup'),
        help='Backup storage directory'
    )
    
    parser.add_argument(
        '--config-dir',
        type=Path,
        default=Path('/etc/ttrpg-assistant'),
        help='Configuration directory'
    )
    
    parser.add_argument(
        '--create',
        action='store_true',
        help='Create a new backup'
    )
    
    parser.add_argument(
        '--type',
        choices=['manual', 'scheduled', 'pre-update'],
        default='manual',
        help='Backup type'
    )
    
    parser.add_argument(
        '--description',
        type=str,
        default='',
        help='Backup description'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all backups'
    )
    
    parser.add_argument(
        '--verify',
        type=str,
        help='Verify backup integrity'
    )
    
    parser.add_argument(
        '--info',
        type=str,
        help='Get backup information'
    )
    
    parser.add_argument(
        '--delete',
        type=str,
        help='Delete a backup by ID'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force operation without confirmation'
    )
    
    args = parser.parse_args()
    
    # Initialize backup manager
    manager = BackupManager(
        data_dir=args.data_dir,
        backup_dir=args.backup_dir,
        config_dir=args.config_dir
    )
    
    # Execute requested operation
    if args.create:
        backup_id = manager.create_backup(
            backup_type=args.type,
            description=args.description
        )
        if backup_id:
            print(f"Backup created: {backup_id}")
        else:
            print("Backup creation failed")
            sys.exit(1)
            
    elif args.list:
        backups = manager.list_backups()
        if backups:
            print("\nAvailable Backups:")
            print("-" * 80)
            for backup in backups:
                print(f"ID: {backup['id']}")
                print(f"  Type: {backup['type']}")
                print(f"  Created: {backup['created_at']}")
                print(f"  Size: {backup['size'] / (1024**2):.1f} MB")
                print(f"  Description: {backup.get('description', 'N/A')}")
                print()
        else:
            print("No backups found")
            
    elif args.verify:
        if manager.verify_backup(args.verify):
            print(f"Backup {args.verify} is valid")
        else:
            print(f"Backup {args.verify} verification failed")
            sys.exit(1)
            
    elif args.info:
        info = manager.get_backup_info(args.info)
        if info:
            print(json.dumps(info, indent=2))
        else:
            print(f"Backup {args.info} not found")
            sys.exit(1)
    
    elif args.delete:
        # Confirm deletion if not forced
        if not args.force:
            response = input(f"Delete backup {args.delete}? This cannot be undone. [y/N]: ")
            if response.lower() != 'y':
                print("Deletion cancelled")
                sys.exit(0)
        
        if manager.delete_backup(args.delete, force=args.force):
            print(f"Backup {args.delete} deleted successfully")
        else:
            print(f"Failed to delete backup {args.delete}")
            sys.exit(1)
            
    else:
        parser.print_help()


if __name__ == '__main__':
    main()