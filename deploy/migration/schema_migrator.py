#!/usr/bin/env python3
"""
Schema migration utilities for TTRPG Assistant MCP Server.
Handles database schema changes between versions.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SchemaMigrator:
    """Handles database schema migrations."""
    
    def __init__(self, data_dir: Path):
        """Initialize schema migrator.
        
        Args:
            data_dir: Data directory path
        """
        self.data_dir = Path(data_dir)
        self.chromadb_path = self.data_dir / 'chromadb'
        
    def get_migrations(self, from_version: str, to_version: str) -> List[Dict]:
        """Get list of schema migrations between versions.
        
        Args:
            from_version: Starting version
            to_version: Target version
            
        Returns:
            List of migration definitions
        """
        # Define schema migrations for each version
        migrations = {
            '0.2.0': [
                {
                    'name': 'create_campaigns_table',
                    'description': 'Create campaigns table',
                    'reversible': True
                }
            ],
            '0.3.0': [
                {
                    'name': 'create_characters_table',
                    'description': 'Create characters table',
                    'reversible': True
                }
            ],
            '0.5.0': [
                {
                    'name': 'add_source_metadata',
                    'description': 'Add metadata columns to sources',
                    'reversible': True
                }
            ],
            '1.0.0': [
                {
                    'name': 'add_indexes',
                    'description': 'Add performance indexes',
                    'reversible': True
                }
            ]
        }
        
        # Collect migrations for the version range
        result = []
        for version, version_migrations in migrations.items():
            if self._version_in_range(version, from_version, to_version):
                result.extend(version_migrations)
                
        return result
        
    def execute_migration(self, migration_name: str, target_version: str) -> bool:
        """Execute a specific schema migration.
        
        Args:
            migration_name: Name of the migration
            target_version: Target version
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Executing schema migration: {migration_name}")
            
            # Map migration names to SQL scripts
            migration_map = {
                'create_campaigns_table': self._create_campaigns_table,
                'create_characters_table': self._create_characters_table,
                'add_source_metadata': self._add_source_metadata,
                'add_indexes': self._add_indexes
            }
            
            if migration_name in migration_map:
                migration_map[migration_name]()
                logger.info(f"Schema migration {migration_name} completed")
                return True
            else:
                logger.error(f"Unknown migration: {migration_name}")
                return False
                
        except Exception as e:
            logger.error(f"Schema migration {migration_name} failed: {e}")
            return False
            
    def _create_campaigns_table(self):
        """Create campaigns table."""
        # This is a simplified example - actual implementation would depend on the database
        metadata_file = self.data_dir / 'schema_metadata.json'
        
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                
        metadata['campaigns_table'] = {
            'created': datetime.now().isoformat(),
            'version': '0.2.0'
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def _create_characters_table(self):
        """Create characters table."""
        metadata_file = self.data_dir / 'schema_metadata.json'
        
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                
        metadata['characters_table'] = {
            'created': datetime.now().isoformat(),
            'version': '0.3.0'
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def _add_source_metadata(self):
        """Add metadata columns to sources."""
        metadata_file = self.data_dir / 'schema_metadata.json'
        
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                
        metadata['source_metadata'] = {
            'added': datetime.now().isoformat(),
            'version': '0.5.0'
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def _add_indexes(self):
        """Add performance indexes."""
        metadata_file = self.data_dir / 'schema_metadata.json'
        
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                
        metadata['indexes'] = {
            'created': datetime.now().isoformat(),
            'version': '1.0.0',
            'indexes': [
                'idx_campaigns_id',
                'idx_characters_campaign',
                'idx_sources_type'
            ]
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def verify_schema(self, version: str) -> bool:
        """Verify schema matches expected version.
        
        Args:
            version: Version to verify against
            
        Returns:
            True if schema is valid
        """
        metadata_file = self.data_dir / 'schema_metadata.json'
        
        if not metadata_file.exists():
            logger.error("Schema metadata not found")
            return False
            
        with open(metadata_file) as f:
            metadata = json.load(f)
            
        # Check expected schema elements for version
        if version >= '0.2.0' and 'campaigns_table' not in metadata:
            logger.error("Campaigns table not found")
            return False
            
        if version >= '0.3.0' and 'characters_table' not in metadata:
            logger.error("Characters table not found")
            return False
            
        logger.info("Schema verification passed")
        return True
        
    def _version_in_range(self, version: str, from_version: str, to_version: str) -> bool:
        """Check if version is in migration range.
        
        Args:
            version: Version to check
            from_version: Starting version
            to_version: Ending version
            
        Returns:
            True if in range
        """
        from packaging import version as pkg_version
        
        v = pkg_version.parse(version)
        v_from = pkg_version.parse(from_version)
        v_to = pkg_version.parse(to_version)
        
        return v_from < v <= v_to