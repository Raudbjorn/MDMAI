#!/usr/bin/env python3
"""
Data migration utilities for TTRPG Assistant MCP Server.
Handles data transformation between versions.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)


class DataMigrator:
    """Handles data migration between versions."""
    
    def __init__(self, data_dir: Path):
        """Initialize data migrator.
        
        Args:
            data_dir: Data directory path
        """
        self.data_dir = Path(data_dir)
        self.migrations_dir = Path(__file__).parent / 'migrations'
        
    def get_migrations(self, from_version: str, to_version: str) -> List[Dict]:
        """Get list of data migrations between versions.
        
        Args:
            from_version: Starting version
            to_version: Target version
            
        Returns:
            List of migration definitions
        """
        # Define data migrations for each version
        migrations = {
            '0.2.0': [
                {
                    'name': 'add_campaign_schema',
                    'description': 'Add campaign management data structures',
                    'reversible': True
                }
            ],
            '0.3.0': [
                {
                    'name': 'add_character_tables',
                    'description': 'Add character generation tables',
                    'reversible': True
                }
            ],
            '0.4.0': [
                {
                    'name': 'add_personality_data',
                    'description': 'Add personality system data',
                    'reversible': True
                }
            ],
            '0.5.0': [
                {
                    'name': 'migrate_source_format',
                    'description': 'Update source management format',
                    'reversible': False
                }
            ],
            '0.6.0': [
                {
                    'name': 'add_session_tracking',
                    'description': 'Add session tracking data',
                    'reversible': True
                }
            ],
            '1.0.0': [
                {
                    'name': 'consolidate_indexes',
                    'description': 'Consolidate search indexes',
                    'reversible': False
                },
                {
                    'name': 'optimize_cache_structure',
                    'description': 'Optimize cache structure',
                    'reversible': False
                }
            ]
        }
        
        # Collect migrations for the version range
        result = []
        for version, version_migrations in migrations.items():
            # Check if this version is in our migration path
            if self._version_in_range(version, from_version, to_version):
                result.extend(version_migrations)
                
        return result
        
    def execute_migration(self, migration_name: str, target_version: str) -> bool:
        """Execute a specific data migration.
        
        Args:
            migration_name: Name of the migration
            target_version: Target version
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Executing data migration: {migration_name}")
            
            # Map migration names to methods
            migration_map = {
                'add_campaign_schema': self._migrate_campaign_schema,
                'add_character_tables': self._migrate_character_tables,
                'add_personality_data': self._migrate_personality_data,
                'migrate_source_format': self._migrate_source_format,
                'add_session_tracking': self._migrate_session_tracking,
                'consolidate_indexes': self._consolidate_indexes,
                'optimize_cache_structure': self._optimize_cache_structure
            }
            
            if migration_name in migration_map:
                migration_map[migration_name]()
                logger.info(f"Migration {migration_name} completed successfully")
                return True
            else:
                logger.error(f"Unknown migration: {migration_name}")
                return False
                
        except Exception as e:
            logger.error(f"Migration {migration_name} failed: {e}")
            return False
            
    def _migrate_campaign_schema(self):
        """Add campaign management data structures."""
        campaigns_dir = self.data_dir / 'campaigns'
        campaigns_dir.mkdir(exist_ok=True)
        
        # Create default campaign structure
        default_campaign = {
            'id': 'default',
            'name': 'Default Campaign',
            'created_at': datetime.now().isoformat(),
            'settings': {},
            'characters': [],
            'sessions': []
        }
        
        campaign_file = campaigns_dir / 'default.json'
        if not campaign_file.exists():
            with open(campaign_file, 'w') as f:
                json.dump(default_campaign, f, indent=2)
                
    def _migrate_character_tables(self):
        """Add character generation tables."""
        characters_dir = self.data_dir / 'characters'
        characters_dir.mkdir(exist_ok=True)
        
        # Create character templates
        templates_file = characters_dir / 'templates.json'
        if not templates_file.exists():
            templates = {
                'classes': ['Fighter', 'Wizard', 'Rogue', 'Cleric'],
                'races': ['Human', 'Elf', 'Dwarf', 'Halfling'],
                'backgrounds': ['Soldier', 'Scholar', 'Criminal', 'Noble']
            }
            with open(templates_file, 'w') as f:
                json.dump(templates, f, indent=2)
                
    def _migrate_personality_data(self):
        """Add personality system data."""
        personality_dir = self.data_dir / 'personality'
        personality_dir.mkdir(exist_ok=True)
        
        # Create personality profiles
        profiles_file = personality_dir / 'profiles.json'
        if not profiles_file.exists():
            profiles = {
                'default_traits': {
                    'helpfulness': 0.8,
                    'creativity': 0.7,
                    'detail_orientation': 0.6
                }
            }
            with open(profiles_file, 'w') as f:
                json.dump(profiles, f, indent=2)
                
    def _migrate_source_format(self):
        """Update source management format."""
        sources_dir = self.data_dir / 'sources'
        if sources_dir.exists():
            # Migrate old source format to new format
            for source_file in sources_dir.glob('*.json'):
                with open(source_file) as f:
                    data = json.load(f)
                    
                # Update format if needed
                if 'version' not in data:
                    data['version'] = '1.0'
                    data['migrated_at'] = datetime.now().isoformat()
                    
                    with open(source_file, 'w') as f:
                        json.dump(data, f, indent=2)
                        
    def _migrate_session_tracking(self):
        """Add session tracking data."""
        sessions_dir = self.data_dir / 'sessions'
        sessions_dir.mkdir(exist_ok=True)
        
        # Create session metadata
        metadata_file = sessions_dir / 'metadata.json'
        if not metadata_file.exists():
            metadata = {
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'session_count': 0
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
    def _consolidate_indexes(self):
        """Consolidate search indexes for better performance."""
        index_dir = self.data_dir / 'indexes'
        
        if index_dir.exists():
            # Consolidate multiple index files
            consolidated = {}
            
            for index_file in index_dir.glob('*.idx'):
                with open(index_file, 'rb') as f:
                    try:
                        data = pickle.load(f)
                        consolidated.update(data)
                    except Exception as e:
                        logger.warning(f"Could not read index {index_file}: {e}")
                        
            # Save consolidated index
            if consolidated:
                with open(index_dir / 'consolidated.idx', 'wb') as f:
                    pickle.dump(consolidated, f)
                    
                # Remove old index files
                for index_file in index_dir.glob('*.idx'):
                    if index_file.name != 'consolidated.idx':
                        index_file.unlink()
                        
    def _optimize_cache_structure(self):
        """Optimize cache directory structure."""
        cache_dir = self.data_dir / 'cache'
        
        if cache_dir.exists():
            # Create subdirectories for different cache types
            for subdir in ['search', 'embeddings', 'results', 'temp']:
                (cache_dir / subdir).mkdir(exist_ok=True)
                
            # Move existing cache files to appropriate subdirectories
            for cache_file in cache_dir.glob('*.cache'):
                if 'search' in cache_file.name:
                    dest = cache_dir / 'search' / cache_file.name
                elif 'embedding' in cache_file.name:
                    dest = cache_dir / 'embeddings' / cache_file.name
                else:
                    dest = cache_dir / 'results' / cache_file.name
                    
                if not dest.exists():
                    shutil.move(str(cache_file), str(dest))
                    
    def verify_data_integrity(self) -> bool:
        """Verify data integrity after migration.
        
        Returns:
            True if data is intact
        """
        required_dirs = [
            'chromadb',
            'cache',
            'campaigns',
            'characters',
            'personality',
            'sessions'
        ]
        
        for dir_name in required_dirs:
            dir_path = self.data_dir / dir_name
            if not dir_path.exists():
                logger.error(f"Required directory missing: {dir_path}")
                return False
                
        # Check for critical files
        critical_files = [
            'campaigns/default.json',
            'characters/templates.json',
            'personality/profiles.json'
        ]
        
        for file_path in critical_files:
            full_path = self.data_dir / file_path
            if not full_path.exists():
                logger.warning(f"Expected file missing: {full_path}")
                
        logger.info("Data integrity check passed")
        return True
        
    def export_data(self, output_file: Path, format: str = 'json'):
        """Export data for backup or transfer.
        
        Args:
            output_file: Output file path
            format: Export format (json, yaml)
        """
        export_data = {
            'version': '1.0.0',
            'exported_at': datetime.now().isoformat(),
            'data': {}
        }
        
        # Collect data from various sources
        for data_type in ['campaigns', 'characters', 'personality', 'sessions']:
            data_dir = self.data_dir / data_type
            if data_dir.exists():
                export_data['data'][data_type] = {}
                for data_file in data_dir.glob('*.json'):
                    with open(data_file) as f:
                        export_data['data'][data_type][data_file.stem] = json.load(f)
                        
        # Save export
        if format == 'json':
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format == 'yaml':
            import yaml
            with open(output_file, 'w') as f:
                yaml.dump(export_data, f, default_flow_style=False)
                
        logger.info(f"Data exported to {output_file}")
        
    def import_data(self, input_file: Path):
        """Import data from export file.
        
        Args:
            input_file: Input file path
        """
        # Load import data
        with open(input_file) as f:
            if input_file.suffix == '.json':
                import_data = json.load(f)
            elif input_file.suffix in ['.yaml', '.yml']:
                import yaml
                import_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported format: {input_file.suffix}")
                
        # Import data types
        for data_type, type_data in import_data.get('data', {}).items():
            data_dir = self.data_dir / data_type
            data_dir.mkdir(exist_ok=True)
            
            for name, content in type_data.items():
                with open(data_dir / f"{name}.json", 'w') as f:
                    json.dump(content, f, indent=2)
                    
        logger.info(f"Data imported from {input_file}")
        
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