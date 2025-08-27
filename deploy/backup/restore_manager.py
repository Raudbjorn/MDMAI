#!/usr/bin/env python3
"""
Restore manager for TTRPG Assistant MCP Server.
Handles restoration from backups.
"""

import os
import sys
import json
import shutil
import tarfile
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import hashlib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.platform_utils import (
    manage_service,
    set_file_permissions,
    set_owner,
    is_unix_like,
    is_windows
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RestoreManager:
    """Manages backup restoration operations."""
    
    def __init__(self, data_dir: Path, backup_dir: Path, config_dir: Path):
        """Initialize restore manager.
        
        Args:
            data_dir: Data directory to restore to
            backup_dir: Directory containing backups
            config_dir: Configuration directory
        """
        self.data_dir = Path(data_dir)
        self.backup_dir = Path(backup_dir)
        self.config_dir = Path(config_dir)
        
        # Load backup registry
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
                
        return {'backups': []}
        
    def restore_backup(self, backup_id: str, verify: bool = True,
                      dry_run: bool = False) -> bool:
        """Restore from a backup.
        
        Args:
            backup_id: Backup ID to restore
            verify: Verify backup before restoring
            dry_run: Simulate restoration without making changes
            
        Returns:
            True if successful
        """
        logger.info(f"Starting restoration from backup: {backup_id}")
        
        if dry_run:
            logger.info("DRY RUN MODE - No changes will be made")
            
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
        
        # Verify backup if requested
        if verify:
            if not self._verify_backup(backup, backup_file):
                logger.error("Backup verification failed")
                return False
                
        try:
            # Create pre-restore backup
            if not dry_run:
                pre_restore_backup = self._create_pre_restore_backup()
                if pre_restore_backup:
                    logger.info(f"Created pre-restore backup: {pre_restore_backup}")
                    
            # Extract backup to temporary directory
            temp_dir = self.backup_dir / f"restore_temp_{backup_id}"
            temp_dir.mkdir(exist_ok=True)
            
            logger.info(f"Extracting backup to temporary directory")
            with tarfile.open(backup_file, 'r:gz') as tar:
                tar.extractall(temp_dir)
                
            # Verify extracted contents
            data_source = temp_dir / 'data'
            config_source = temp_dir / 'config'
            metadata_file = temp_dir / 'metadata.json'
            
            if not data_source.exists() or not config_source.exists():
                logger.error("Backup missing required components")
                shutil.rmtree(temp_dir)
                return False
                
            # Load and display metadata
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    logger.info(f"Backup metadata: {json.dumps(metadata, indent=2)}")
                    
            if not dry_run:
                # Stop any running services
                self._stop_services()
                
                # Restore data directory
                logger.info(f"Restoring data directory: {self.data_dir}")
                if self.data_dir.exists():
                    shutil.rmtree(self.data_dir)
                shutil.copytree(data_source, self.data_dir)
                
                # Restore configuration
                logger.info(f"Restoring configuration: {self.config_dir}")
                if self.config_dir.exists():
                    # Backup current config
                    config_backup = self.config_dir.parent / f"config_backup_{datetime.now():%Y%m%d_%H%M%S}"
                    shutil.copytree(self.config_dir, config_backup)
                    shutil.rmtree(self.config_dir)
                shutil.copytree(config_source, self.config_dir)
                
                # Update permissions
                self._set_permissions()
                
                # Start services
                self._start_services()
                
            # Cleanup temporary directory
            shutil.rmtree(temp_dir)
            
            logger.info(f"Restoration completed successfully")
            
            if dry_run:
                logger.info("DRY RUN completed - no changes were made")
                
            return True
            
        except (IOError, OSError) as e:
            logger.error(f"File system error during restoration: {e}")
            
            # Cleanup on failure
            if 'temp_dir' in locals() and temp_dir.exists():
                shutil.rmtree(temp_dir)
        except tarfile.TarError as e:
            logger.error(f"Archive error during restoration: {e}")
            
            # Cleanup on failure
            if 'temp_dir' in locals() and temp_dir.exists():
                shutil.rmtree(temp_dir)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing backup metadata: {e}")
            
            # Cleanup on failure
            if 'temp_dir' in locals() and temp_dir.exists():
                shutil.rmtree(temp_dir)
                
            # Attempt to restore pre-restore backup
            if not dry_run and 'pre_restore_backup' in locals():
                logger.info("Attempting to restore pre-restore backup")
                self.restore_backup(pre_restore_backup, verify=False)
                
            return False
            
    def _verify_backup(self, backup: Dict, backup_file: Path) -> bool:
        """Verify backup integrity.
        
        Args:
            backup: Backup registry entry
            backup_file: Backup file path
            
        Returns:
            True if valid
        """
        logger.info("Verifying backup integrity")
        
        # Check file exists
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False
            
        # Verify checksum
        logger.info("Verifying checksum")
        calculated_checksum = self._calculate_checksum(backup_file)
        if calculated_checksum != backup.get('checksum'):
            logger.error("Checksum mismatch")
            return False
            
        # Verify tar integrity
        logger.info("Verifying archive integrity")
        try:
            with tarfile.open(backup_file, 'r:gz') as tar:
                tar.getmembers()
        except tarfile.TarError as e:
            logger.error(f"Archive verification failed: {e}")
            return False
        except (IOError, OSError) as e:
            logger.error(f"Error reading archive file: {e}")
            return False
            
        logger.info("Backup verification passed")
        return True
        
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum.
        
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
        
    def _create_pre_restore_backup(self) -> Optional[str]:
        """Create a backup before restoration.
        
        Returns:
            Backup ID if successful
        """
        try:
            from deploy.backup.backup_manager import BackupManager
            
            manager = BackupManager(
                data_dir=self.data_dir,
                backup_dir=self.backup_dir,
                config_dir=self.config_dir
            )
            
            return manager.create_backup(
                backup_type='pre-restore',
                description='Automatic backup before restoration'
            )
        except ImportError as e:
            logger.warning(f"Could not import backup manager module: {e}")
            return None
        except (IOError, OSError) as e:
            logger.warning(f"File system error creating pre-restore backup: {e}")
            return None
            
    def _stop_services(self):
        """Stop running services."""
        try:
            logger.info("Stopping ttrpg-assistant service")
            if manage_service('ttrpg-assistant', 'stop'):
                logger.info("Service stopped successfully")
            else:
                logger.warning("Service may not be running or not installed as a service")
        except (ImportError, OSError) as e:
            logger.warning(f"Could not stop services: {e}")
            
    def _start_services(self):
        """Start services after restoration."""
        try:
            logger.info("Starting ttrpg-assistant service")
            if manage_service('ttrpg-assistant', 'start'):
                logger.info("Service started successfully")
            else:
                logger.warning("Could not start service - may need manual start")
        except (ImportError, OSError) as e:
            logger.warning(f"Could not start services: {e}")
            
    def _set_permissions(self):
        """Set appropriate permissions on restored files."""
        try:
            # Only set ownership and permissions on Unix-like systems
            if is_unix_like():
                # Set ownership (only on Unix-like systems)
                for directory in [self.data_dir, self.config_dir]:
                    if directory.exists():
                        # Recursively set ownership
                        for path in directory.rglob('*'):
                            set_owner(path, user='ttrpg', group='ttrpg')
                        set_owner(directory, user='ttrpg', group='ttrpg')
                
                # Set permissions
                set_file_permissions(self.data_dir, 0o750)
                set_file_permissions(self.config_dir, 0o750)
                
                # Set file permissions recursively
                for directory in [self.data_dir, self.config_dir]:
                    for path in directory.rglob('*'):
                        if path.is_dir():
                            set_file_permissions(path, 0o750)
                        else:
                            set_file_permissions(path, 0o640)
                
                logger.info("Permissions set successfully")
            elif is_windows():
                logger.info("Running on Windows - skipping Unix-style permissions")
            else:
                logger.warning("Unknown platform - skipping permissions")
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not set permissions: {e}")
            

def main():
    """Main entry point for restore manager."""
    parser = argparse.ArgumentParser(
        description="TTRPG Assistant Restore Manager"
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('/var/lib/ttrpg-assistant'),
        help='Data directory to restore to'
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
        '--restore',
        type=str,
        help='Backup ID to restore'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate restoration without making changes'
    )
    
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip backup verification'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force restoration without confirmations'
    )
    
    args = parser.parse_args()
    
    if not args.restore:
        parser.print_help()
        sys.exit(1)
        
    # Initialize restore manager
    manager = RestoreManager(
        data_dir=args.data_dir,
        backup_dir=args.backup_dir,
        config_dir=args.config_dir
    )
    
    # Confirm restoration
    if not args.force and not args.dry_run:
        response = input(f"Restore from backup {args.restore}? This will overwrite current data. [y/N]: ")
        if response.lower() != 'y':
            print("Restoration cancelled")
            sys.exit(0)
            
    # Perform restoration
    success = manager.restore_backup(
        backup_id=args.restore,
        verify=not args.no_verify,
        dry_run=args.dry_run
    )
    
    if success:
        print(f"Restoration completed successfully")
    else:
        print(f"Restoration failed")
        sys.exit(1)


if __name__ == '__main__':
    main()