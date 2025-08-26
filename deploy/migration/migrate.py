#!/usr/bin/env python3
"""
Main migration orchestrator for TTRPG Assistant MCP Server.
Handles version upgrades, data migration, and rollback operations.
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from deploy.migration.version_manager import VersionManager
from deploy.migration.data_migrator import DataMigrator
from deploy.migration.schema_migrator import SchemaMigrator
from deploy.migration.rollback import RollbackManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MigrationStatus(Enum):
    """Migration status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class MigrationOrchestrator:
    """Orchestrates the migration process."""
    
    def __init__(self, data_dir: Path, config_dir: Path, backup_dir: Path):
        """Initialize migration orchestrator.
        
        Args:
            data_dir: Data directory path
            config_dir: Configuration directory path
            backup_dir: Backup directory path
        """
        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)
        self.backup_dir = Path(backup_dir)
        
        # Initialize managers
        self.version_manager = VersionManager(config_dir)
        self.data_migrator = DataMigrator(data_dir)
        self.schema_migrator = SchemaMigrator(data_dir)
        self.rollback_manager = RollbackManager(backup_dir)
        
        # Migration state
        self.migration_id = None
        self.status = MigrationStatus.PENDING
        self.errors: List[str] = []
        
    def create_migration_plan(self, target_version: str) -> Dict:
        """Create a migration plan.
        
        Args:
            target_version: Target version to migrate to
            
        Returns:
            Migration plan dictionary
        """
        current_version = self.version_manager.get_current_version()
        
        if not self.version_manager.is_valid_version(target_version):
            raise ValueError(f"Invalid target version: {target_version}")
            
        if self.version_manager.compare_versions(current_version, target_version) >= 0:
            logger.info(f"Already at version {current_version}, no migration needed")
            return {}
            
        # Get migration path
        migration_path = self.version_manager.get_migration_path(current_version, target_version)
        
        plan = {
            'migration_id': self._generate_migration_id(),
            'current_version': current_version,
            'target_version': target_version,
            'migration_path': migration_path,
            'steps': [],
            'estimated_duration': 0,
            'requires_backup': True,
            'requires_downtime': False
        }
        
        # Build migration steps
        for version in migration_path:
            step = {
                'version': version,
                'description': f"Migrate to version {version}",
                'migrations': []
            }
            
            # Check for schema migrations
            schema_migrations = self.schema_migrator.get_migrations(
                self.version_manager.get_previous_version(version),
                version
            )
            if schema_migrations:
                step['migrations'].extend([
                    {
                        'type': 'schema',
                        'name': m['name'],
                        'description': m.get('description', ''),
                        'reversible': m.get('reversible', True)
                    }
                    for m in schema_migrations
                ])
                step['requires_downtime'] = True
                plan['requires_downtime'] = True
                
            # Check for data migrations
            data_migrations = self.data_migrator.get_migrations(
                self.version_manager.get_previous_version(version),
                version
            )
            if data_migrations:
                step['migrations'].extend([
                    {
                        'type': 'data',
                        'name': m['name'],
                        'description': m.get('description', ''),
                        'reversible': m.get('reversible', True)
                    }
                    for m in data_migrations
                ])
                
            if step['migrations']:
                plan['steps'].append(step)
                plan['estimated_duration'] += len(step['migrations']) * 30  # 30 seconds per migration
                
        return plan
        
    def validate_prerequisites(self) -> Tuple[bool, List[str]]:
        """Validate migration prerequisites.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check disk space
        required_space = self._estimate_required_space()
        available_space = shutil.disk_usage(self.data_dir).free
        
        if available_space < required_space:
            errors.append(
                f"Insufficient disk space: {available_space / (1024**3):.1f} GB available, "
                f"{required_space / (1024**3):.1f} GB required"
            )
            
        # Check backup directory
        if not self.backup_dir.exists():
            try:
                self.backup_dir.mkdir(parents=True)
            except Exception as e:
                errors.append(f"Cannot create backup directory: {e}")
                
        # Check write permissions
        test_file = self.data_dir / '.migration_test'
        try:
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            errors.append(f"No write permission in data directory: {e}")
            
        # Check if application is running
        if self._is_application_running():
            errors.append("Application is currently running. Please stop it before migration.")
            
        return len(errors) == 0, errors
        
    def create_backup(self) -> str:
        """Create a backup before migration.
        
        Returns:
            Backup ID
        """
        logger.info("Creating backup before migration...")
        
        backup_id = self.rollback_manager.create_backup(
            data_dir=self.data_dir,
            config_dir=self.config_dir,
            metadata={
                'migration_id': self.migration_id,
                'current_version': self.version_manager.get_current_version(),
                'timestamp': datetime.now().isoformat()
            }
        )
        
        logger.info(f"Backup created: {backup_id}")
        return backup_id
        
    def execute_migration(self, plan: Dict, dry_run: bool = False) -> bool:
        """Execute the migration plan.
        
        Args:
            plan: Migration plan
            dry_run: If True, only simulate migration
            
        Returns:
            True if migration successful
        """
        self.migration_id = plan['migration_id']
        self.status = MigrationStatus.IN_PROGRESS
        
        logger.info(f"Starting migration {self.migration_id}")
        logger.info(f"Migrating from {plan['current_version']} to {plan['target_version']}")
        
        if dry_run:
            logger.info("DRY RUN MODE - No changes will be made")
            
        backup_id = None
        
        try:
            # Create backup if not dry run
            if not dry_run and plan['requires_backup']:
                backup_id = self.create_backup()
                
            # Execute migration steps
            for step in plan['steps']:
                logger.info(f"Executing step: {step['description']}")
                
                for migration in step['migrations']:
                    logger.info(f"  Running {migration['type']} migration: {migration['name']}")
                    
                    if not dry_run:
                        if migration['type'] == 'schema':
                            success = self.schema_migrator.execute_migration(
                                migration['name'],
                                step['version']
                            )
                        elif migration['type'] == 'data':
                            success = self.data_migrator.execute_migration(
                                migration['name'],
                                step['version']
                            )
                        else:
                            raise ValueError(f"Unknown migration type: {migration['type']}")
                            
                        if not success:
                            raise Exception(f"Migration failed: {migration['name']}")
                            
                # Update version after successful step
                if not dry_run:
                    self.version_manager.update_version(step['version'])
                    logger.info(f"Updated version to {step['version']}")
                    
            # Migration completed successfully
            self.status = MigrationStatus.COMPLETED
            logger.info(f"Migration {self.migration_id} completed successfully")
            
            # Clean up old backups if configured
            if not dry_run:
                self.rollback_manager.cleanup_old_backups()
                
            return True
            
        except Exception as e:
            self.status = MigrationStatus.FAILED
            self.errors.append(str(e))
            logger.error(f"Migration failed: {e}")
            
            if not dry_run and backup_id:
                logger.info("Attempting rollback...")
                if self.rollback(backup_id):
                    self.status = MigrationStatus.ROLLED_BACK
                    logger.info("Rollback successful")
                else:
                    logger.error("Rollback failed! Manual intervention required")
                    
            return False
            
    def rollback(self, backup_id: str) -> bool:
        """Rollback to a previous backup.
        
        Args:
            backup_id: Backup ID to rollback to
            
        Returns:
            True if rollback successful
        """
        try:
            logger.info(f"Rolling back to backup {backup_id}")
            
            # Execute rollback
            success = self.rollback_manager.rollback(backup_id)
            
            if success:
                # Update version from backup metadata
                metadata = self.rollback_manager.get_backup_metadata(backup_id)
                if metadata and 'current_version' in metadata:
                    self.version_manager.update_version(metadata['current_version'])
                    
                logger.info("Rollback completed successfully")
                return True
            else:
                logger.error("Rollback failed")
                return False
                
        except Exception as e:
            logger.error(f"Rollback error: {e}")
            return False
            
    def verify_migration(self, target_version: str) -> bool:
        """Verify that migration was successful.
        
        Args:
            target_version: Expected version after migration
            
        Returns:
            True if verification successful
        """
        logger.info("Verifying migration...")
        
        # Check version
        current_version = self.version_manager.get_current_version()
        if current_version != target_version:
            logger.error(f"Version mismatch: expected {target_version}, got {current_version}")
            return False
            
        # Verify schema
        if not self.schema_migrator.verify_schema(target_version):
            logger.error("Schema verification failed")
            return False
            
        # Verify data integrity
        if not self.data_migrator.verify_data_integrity():
            logger.error("Data integrity verification failed")
            return False
            
        logger.info("Migration verification successful")
        return True
        
    def get_migration_status(self) -> Dict:
        """Get current migration status.
        
        Returns:
            Status dictionary
        """
        return {
            'migration_id': self.migration_id,
            'status': self.status.value,
            'current_version': self.version_manager.get_current_version(),
            'errors': self.errors,
            'available_backups': self.rollback_manager.list_backups()
        }
        
    def _generate_migration_id(self) -> str:
        """Generate unique migration ID.
        
        Returns:
            Migration ID
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"migration_{timestamp}"
        
    def _estimate_required_space(self) -> int:
        """Estimate required disk space for migration.
        
        Returns:
            Required space in bytes
        """
        # Calculate size of data directory
        data_size = sum(
            f.stat().st_size
            for f in self.data_dir.rglob('*')
            if f.is_file()
        )
        
        # Need space for backup plus 20% buffer
        return int(data_size * 1.2)
        
    def _is_application_running(self) -> bool:
        """Check if application is currently running.
        
        Returns:
            True if application is running
        """
        pid_file = self.data_dir / '.pid'
        if pid_file.exists():
            try:
                with open(pid_file) as f:
                    pid = int(f.read().strip())
                    
                # Check if process is running
                import psutil
                return psutil.pid_exists(pid)
            except Exception:
                pass
                
        return False
        

def main():
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="TTRPG Assistant MCP Server Migration Tool"
    )
    
    parser.add_argument(
        'target_version',
        help='Target version to migrate to'
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('/var/lib/ttrpg-assistant'),
        help='Data directory path'
    )
    
    parser.add_argument(
        '--config-dir',
        type=Path,
        default=Path('/etc/ttrpg-assistant'),
        help='Configuration directory path'
    )
    
    parser.add_argument(
        '--backup-dir',
        type=Path,
        default=Path('/var/lib/ttrpg-assistant/backup'),
        help='Backup directory path'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate migration without making changes'
    )
    
    parser.add_argument(
        '--skip-backup',
        action='store_true',
        help='Skip backup creation (not recommended)'
    )
    
    parser.add_argument(
        '--rollback',
        type=str,
        help='Rollback to specified backup ID'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show migration status'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify current installation'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force migration even if prerequisites fail'
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = MigrationOrchestrator(
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        backup_dir=args.backup_dir
    )
    
    try:
        # Handle different operations
        if args.rollback:
            # Perform rollback
            success = orchestrator.rollback(args.rollback)
            sys.exit(0 if success else 1)
            
        elif args.status:
            # Show status
            status = orchestrator.get_migration_status()
            print(json.dumps(status, indent=2))
            sys.exit(0)
            
        elif args.verify:
            # Verify installation
            current_version = orchestrator.version_manager.get_current_version()
            success = orchestrator.verify_migration(current_version)
            sys.exit(0 if success else 1)
            
        else:
            # Perform migration
            # Validate prerequisites
            valid, errors = orchestrator.validate_prerequisites()
            if not valid and not args.force:
                logger.error("Prerequisites validation failed:")
                for error in errors:
                    logger.error(f"  - {error}")
                sys.exit(1)
                
            # Create migration plan
            plan = orchestrator.create_migration_plan(args.target_version)
            
            if not plan:
                logger.info("No migration needed")
                sys.exit(0)
                
            # Show migration plan
            print("\nMigration Plan:")
            print("="*60)
            print(f"Current Version: {plan['current_version']}")
            print(f"Target Version: {plan['target_version']}")
            print(f"Estimated Duration: {plan['estimated_duration']} seconds")
            print(f"Requires Downtime: {plan['requires_downtime']}")
            print(f"\nMigration Steps:")
            for step in plan['steps']:
                print(f"  - {step['description']}")
                for migration in step['migrations']:
                    print(f"    • {migration['type']}: {migration['name']}")
            print("="*60)
            
            # Confirm migration
            if not args.dry_run:
                response = input("\nProceed with migration? [y/N]: ")
                if response.lower() != 'y':
                    print("Migration cancelled")
                    sys.exit(0)
                    
            # Modify plan if skipping backup
            if args.skip_backup:
                plan['requires_backup'] = False
                
            # Execute migration
            success = orchestrator.execute_migration(plan, dry_run=args.dry_run)
            
            # Verify if successful and not dry run
            if success and not args.dry_run:
                success = orchestrator.verify_migration(args.target_version)
                
            sys.exit(0 if success else 1)
            
    except Exception as e:
        logger.error(f"Migration error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()