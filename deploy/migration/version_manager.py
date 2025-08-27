#!/usr/bin/env python3
"""
Version management system for TTRPG Assistant MCP Server.
Handles version tracking, comparison, and migration paths.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from packaging import version
from packaging.version import Version, InvalidVersion

logger = logging.getLogger(__name__)


class VersionManager:
    """Manages application versions and migration paths."""
    
    # Version history and migration paths
    VERSION_HISTORY = [
        "0.1.0",  # Initial release
        "0.2.0",  # Added campaign management
        "0.3.0",  # Added character generation
        "0.4.0",  # Added personality system
        "0.5.0",  # Added source management
        "0.6.0",  # Added session tracking
        "0.7.0",  # Added security features
        "0.8.0",  # Added performance optimizations
        "0.9.0",  # Added search enhancements
        "1.0.0",  # First stable release
    ]
    
    def __init__(self, config_dir: Path):
        """Initialize version manager.
        
        Args:
            config_dir: Configuration directory path
        """
        self.config_dir = Path(config_dir)
        self.version_file = self.config_dir / 'version.json'
        self.version_info = self._load_version_info()
        
    def _load_version_info(self) -> Dict:
        """Load version information from file.
        
        Returns:
            Version information dictionary
        """
        if self.version_file.exists():
            try:
                with open(self.version_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading version info: {e}")
                
        # Default version info
        return {
            'version': '0.1.0',
            'install_date': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'migration_history': []
        }
        
    def _save_version_info(self):
        """Save version information to file."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.version_file, 'w') as f:
                json.dump(self.version_info, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving version info: {e}")
            raise
            
    def get_current_version(self) -> str:
        """Get current application version.
        
        Returns:
            Current version string
        """
        return self.version_info.get('version', '0.1.0')
        
    def update_version(self, new_version: str):
        """Update to a new version.
        
        Args:
            new_version: New version string
        """
        if not self.is_valid_version(new_version):
            raise ValueError(f"Invalid version: {new_version}")
            
        old_version = self.get_current_version()
        
        # Add to migration history
        migration_record = {
            'from_version': old_version,
            'to_version': new_version,
            'timestamp': datetime.now().isoformat()
        }
        
        self.version_info['version'] = new_version
        self.version_info['last_updated'] = datetime.now().isoformat()
        self.version_info['migration_history'].append(migration_record)
        
        self._save_version_info()
        logger.info(f"Version updated from {old_version} to {new_version}")
        
    def is_valid_version(self, version_str: str) -> bool:
        """Check if version string is valid.
        
        Args:
            version_str: Version string to validate
            
        Returns:
            True if valid
        """
        try:
            Version(version_str)
            return True
        except InvalidVersion:
            return False
        
    def parse_version(self, version_str: str) -> Tuple[int, int, int, str, str]:
        """Parse version string into components.
        
        Args:
            version_str: Version string
            
        Returns:
            Tuple of (major, minor, patch, prerelease, build)
        """
        try:
            v = Version(version_str)
        except InvalidVersion:
            raise ValueError(f"Invalid version: {version_str}")
            
        # Extract components from packaging.version.Version
        major = v.major
        minor = v.minor if v.minor is not None else 0
        patch = v.micro if v.micro is not None else 0
        
        # Handle prerelease and build metadata
        prerelease = ''
        if v.pre:
            # Format prerelease (alpha, beta, rc, etc.)
            phase, number = v.pre
            prerelease = f"{phase}{number}"
        elif v.dev is not None:
            prerelease = f"dev{v.dev}"
        
        # Local version identifier acts as build metadata
        build = v.local if v.local else ''
        
        return (major, minor, patch, prerelease, build)
            
    def compare_versions(self, version1: str, version2: str) -> int:
        """Compare two versions.
        
        Args:
            version1: First version
            version2: Second version
            
        Returns:
            -1 if version1 < version2, 0 if equal, 1 if version1 > version2
        """
        try:
            v1 = Version(version1)
            v2 = Version(version2)
        except InvalidVersion as e:
            raise ValueError(f"Invalid version for comparison: {e}")
        
        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1
        else:
            return 0
        
    def get_migration_path(self, from_version: str, to_version: str) -> List[str]:
        """Get migration path between versions.
        
        Args:
            from_version: Starting version
            to_version: Target version
            
        Returns:
            List of versions to migrate through
        """
        if self.compare_versions(from_version, to_version) >= 0:
            return []
            
        # Find positions in version history
        try:
            from_idx = self.VERSION_HISTORY.index(from_version)
            to_idx = self.VERSION_HISTORY.index(to_version)
        except ValueError:
            # Handle versions not in predefined history
            return self._calculate_custom_path(from_version, to_version)
            
        # Return intermediate versions
        return self.VERSION_HISTORY[from_idx + 1:to_idx + 1]
        
    def _calculate_custom_path(self, from_version: str, to_version: str) -> List[str]:
        """Calculate migration path for custom versions.
        
        Args:
            from_version: Starting version
            to_version: Target version
            
        Returns:
            List of versions to migrate through
        """
        path = []
        
        from_parts = self.parse_version(from_version)
        to_parts = self.parse_version(to_version)
        
        current = list(from_parts[:3])  # major, minor, patch
        target = list(to_parts[:3])
        
        # Increment major versions
        while current[0] < target[0]:
            current[0] += 1
            current[1] = 0
            current[2] = 0
            path.append(f"{current[0]}.{current[1]}.{current[2]}")
            
        # Increment minor versions
        while current[1] < target[1]:
            current[1] += 1
            current[2] = 0
            path.append(f"{current[0]}.{current[1]}.{current[2]}")
            
        # Increment patch versions
        while current[2] < target[2]:
            current[2] += 1
            path.append(f"{current[0]}.{current[1]}.{current[2]}")
            
        return path
        
    def get_previous_version(self, version: str) -> Optional[str]:
        """Get the previous version.
        
        Args:
            version: Current version
            
        Returns:
            Previous version or None
        """
        try:
            idx = self.VERSION_HISTORY.index(version)
            if idx > 0:
                return self.VERSION_HISTORY[idx - 1]
        except ValueError:
            # Calculate previous version
            parts = self.parse_version(version)
            if parts[2] > 0:  # patch
                return f"{parts[0]}.{parts[1]}.{parts[2] - 1}"
            elif parts[1] > 0:  # minor
                return f"{parts[0]}.{parts[1] - 1}.0"
            elif parts[0] > 0:  # major
                return f"{parts[0] - 1}.0.0"
                
        return None
        
    def get_next_version(self, version: str, bump_type: str = 'patch') -> str:
        """Get the next version based on bump type.
        
        Args:
            version: Current version
            bump_type: Type of version bump (major, minor, patch)
            
        Returns:
            Next version string
        """
        parts = self.parse_version(version)
        major, minor, patch = parts[:3]
        
        if bump_type == 'major':
            return f"{major + 1}.0.0"
        elif bump_type == 'minor':
            return f"{major}.{minor + 1}.0"
        elif bump_type == 'patch':
            return f"{major}.{minor}.{patch + 1}"
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")
            
    def get_migration_history(self) -> List[Dict]:
        """Get migration history.
        
        Returns:
            List of migration records
        """
        return self.version_info.get('migration_history', [])
        
    def get_version_info(self) -> Dict:
        """Get complete version information.
        
        Returns:
            Version information dictionary
        """
        return {
            'current_version': self.get_current_version(),
            'install_date': self.version_info.get('install_date'),
            'last_updated': self.version_info.get('last_updated'),
            'migration_count': len(self.get_migration_history()),
            'latest_available': self.VERSION_HISTORY[-1] if self.VERSION_HISTORY else None
        }
        
    def check_for_updates(self) -> Optional[str]:
        """Check if updates are available.
        
        Returns:
            Latest available version if update available, None otherwise
        """
        current = self.get_current_version()
        latest = self.VERSION_HISTORY[-1] if self.VERSION_HISTORY else current
        
        if self.compare_versions(current, latest) < 0:
            return latest
            
        return None
        
    def export_version_info(self, output_file: Path):
        """Export version information to file.
        
        Args:
            output_file: Output file path
        """
        info = self.get_version_info()
        info['migration_history'] = self.get_migration_history()
        info['version_history'] = self.VERSION_HISTORY
        
        with open(output_file, 'w') as f:
            json.dump(info, f, indent=2)
            
        logger.info(f"Version info exported to {output_file}")