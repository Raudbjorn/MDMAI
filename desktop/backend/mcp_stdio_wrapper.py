#!/usr/bin/env python3
"""
MCP Server Stdio Wrapper for Tauri Desktop Application
This script serves as the entry point for the Python MCP server when run as a Tauri sidecar.
It ensures the server runs in stdio mode for direct communication with the Rust backend.
"""

import sys
import os
from pathlib import Path

# Add the project src directory to Python path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Set environment variable to indicate stdio mode
os.environ['MCP_STDIO_MODE'] = 'true'

# Import and run the main MCP server
try:
    from src.main import main
    
    if __name__ == "__main__":
        # Run the MCP server in stdio mode
        sys.exit(main())
        
except ImportError as e:
    print(f"Error: Failed to import MCP server: {e}", file=sys.stderr)
    print(f"Python path: {sys.path}", file=sys.stderr)
    print(f"Looking for MCP server in: {src_path}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error: Failed to start MCP server: {e}", file=sys.stderr)
    sys.exit(1)