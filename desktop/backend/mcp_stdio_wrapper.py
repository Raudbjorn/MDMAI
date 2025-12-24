#!/usr/bin/env python3
"""
MCP Server Stdio Wrapper for Tauri Desktop Application.

This module serves as a production-ready entry point for the Python MCP server
when executed as a Tauri sidecar process. It provides robust error handling,
logging, and environment configuration for stdio-based communication.

Features:
    - Automatic path resolution and Python path management
    - Environment configuration for stdio mode
    - Comprehensive error handling with detailed diagnostics
    - Graceful shutdown handling
    - Type-safe implementation with full type hints
"""

import logging
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import NoReturn, Optional

# Configure logging for sidecar execution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
    force=True
)
logger = logging.getLogger(__name__)


class MCPStdioWrapper:
    """Manages MCP server execution in stdio mode with comprehensive error handling."""
    
    def __init__(self) -> None:
        """Initialize the wrapper with path resolution and environment setup."""
        self.project_root = Path(__file__).parent.parent.parent
        self.src_path = self.project_root / "src"
        self._configure_environment()
        self._setup_signal_handlers()
    
    def _configure_environment(self) -> None:
        """Configure environment variables and Python path for MCP server execution."""
        # Add src directory to Python path if not already present
        src_str = str(self.src_path)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
            logger.info(f"Added {src_str} to Python path")
        
        # Set stdio mode for MCP communication
        os.environ['MCP_STDIO_MODE'] = 'true'
        
        # Ensure proper encoding for stdio communication
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['PYTHONUNBUFFERED'] = '1'
        
        logger.info("Environment configured for stdio mode")
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum: int, frame) -> None:
            logger.info(f"Received signal {signum}, shutting down gracefully")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _validate_dependencies(self) -> None:
        """Validate that required files and modules are available."""
        # Check if src directory exists
        if not self.src_path.exists():
            raise FileNotFoundError(
                f"Source directory not found: {self.src_path}\n"
                f"Project root: {self.project_root}\n"
                f"Current working directory: {Path.cwd()}"
            )
        
        # Check if main module exists
        main_module = self.src_path / "main.py"
        if not main_module.exists():
            raise FileNotFoundError(
                f"Main MCP server module not found: {main_module}"
            )
        
        logger.info("All dependencies validated successfully")
    
    def _import_and_run_server(self) -> int:
        """Import and execute the main MCP server with comprehensive error handling."""
        try:
            # Delayed import after environment setup
            from src.main import main as mcp_main
            logger.info("MCP server module imported successfully")
            
            # Execute the server
            logger.info("Starting MCP server in stdio mode")
            return mcp_main()
            
        except ImportError as e:
            logger.error(f"Failed to import MCP server: {e}")
            logger.error(f"Python path: {sys.path}")
            logger.error(f"Source path: {self.src_path}")
            
            # List available modules in src directory for debugging
            if self.src_path.exists():
                modules = [f.stem for f in self.src_path.glob("*.py")]
                logger.error(f"Available modules in src: {modules}")
            
            raise
        
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            raise
    
    def run(self) -> int:
        """Execute the MCP server with full error handling and logging.
        
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            logger.info("Starting MCP Stdio Wrapper")
            logger.info(f"Project root: {self.project_root}")
            logger.info(f"Python version: {sys.version}")
            
            # Validate environment and dependencies
            self._validate_dependencies()
            
            # Run the MCP server
            exit_code = self._import_and_run_server()
            
            logger.info(f"MCP server completed with exit code: {exit_code}")
            return exit_code
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down gracefully")
            return 0
            
        except Exception as e:
            logger.error(f"Fatal error in MCP wrapper: {e}")
            logger.error(traceback.format_exc())
            return 1


def main() -> int:
    """Main entry point for the MCP stdio wrapper.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    wrapper = MCPStdioWrapper()
    return wrapper.run()


if __name__ == "__main__":
    sys.exit(main())