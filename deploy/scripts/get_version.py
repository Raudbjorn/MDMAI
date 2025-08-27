#!/usr/bin/env python3
"""Simple script to extract version information for build processes."""

import sys
from pathlib import Path

# Add parent directories to path to import from project
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def get_version():
    """Extract version from setup.py or pyproject.toml."""
    try:
        # Try to get from setup.py first
        setup_py = project_root / "setup.py"
        if setup_py.exists():
            with open(setup_py) as f:
                for line in f:
                    if line.strip().startswith('version='):
                        # Extract version string
                        version = line.split('=')[1].strip().strip('",\'')
                        return version
        
        # Try pyproject.toml as fallback
        pyproject = project_root / "pyproject.toml"
        if pyproject.exists():
            with open(pyproject) as f:
                in_project = False
                for line in f:
                    if '[tool.poetry]' in line or '[project]' in line:
                        in_project = True
                    elif in_project and line.startswith('version'):
                        # Extract version string
                        version = line.split('=')[1].strip().strip('",\'')
                        return version
        
        # Default version if not found
        return "0.1.0"
        
    except Exception:
        return "0.1.0"

if __name__ == "__main__":
    print(get_version())