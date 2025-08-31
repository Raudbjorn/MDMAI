#!/usr/bin/env python3
"""
PyOxidizer Entry Point for MDMAI MCP Server
Handles initialization, workarounds, and optimizations for the packaged executable.

This module serves as the main entry point when the MCP server is packaged 
with PyOxidizer. It includes necessary workarounds for ChromaDB compatibility
and optimizations for startup performance.
"""

import os
import sys
import warnings
from pathlib import Path

def setup_sqlite_compatibility():
    """Set up SQLite compatibility workaround for ChromaDB."""
    try:
        # Import pysqlite3 before any ChromaDB imports
        import pysqlite3
        
        # Replace the default sqlite3 module with pysqlite3
        sys.modules["sqlite3"] = pysqlite3
        
        print("INFO: SQLite3 compatibility workaround applied for ChromaDB", file=sys.stderr)
        return True
        
    except ImportError:
        # Fall back to system sqlite3
        print("WARNING: pysqlite3 not available, using system sqlite3", file=sys.stderr)
        
        # Check system sqlite3 version
        try:
            import sqlite3
            version = sqlite3.sqlite_version
            print(f"INFO: System SQLite version: {version}", file=sys.stderr)
            
            # ChromaDB requires SQLite >= 3.35.0
            major, minor, patch = map(int, version.split('.'))
            if major < 3 or (major == 3 and minor < 35):
                print(f"ERROR: ChromaDB requires SQLite >= 3.35.0, found {version}", file=sys.stderr)
                print("SOLUTION: Install pysqlite3-binary: pip install pysqlite3-binary", file=sys.stderr)
                return False
                
        except Exception as e:
            print(f"ERROR: Could not check SQLite version: {e}", file=sys.stderr)
            return False
            
        return True

def setup_environment():
    """Set up environment variables and paths for optimal performance."""
    
    # Set MCP stdio mode for Tauri communication
    os.environ['MCP_STDIO_MODE'] = 'true'
    
    # Optimize Python for embedded execution
    os.environ['PYTHONUNBUFFERED'] = '1'  # Disable buffering for stdio
    os.environ['PYTHONIOENCODING'] = 'utf-8'  # Ensure UTF-8 encoding
    
    # Torch optimizations for packaging
    os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(sys.executable), 'torch_cache')
    
    # Transformers cache directory (relative to executable)
    transformers_cache = os.path.join(os.path.dirname(sys.executable), 'transformers_cache')
    os.environ['TRANSFORMERS_CACHE'] = transformers_cache
    os.environ['HF_HOME'] = transformers_cache
    
    # ChromaDB specific settings
    os.environ['CHROMA_SERVER_AUTHN_PROVIDER'] = ''  # Disable auth for embedded mode
    
    # Create necessary directories
    Path(transformers_cache).mkdir(parents=True, exist_ok=True)
    Path(os.environ['TORCH_HOME']).mkdir(parents=True, exist_ok=True)
    
    # Suppress unnecessary warnings in packaged environment
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    
    print("INFO: Environment configured for PyOxidizer execution", file=sys.stderr)

def setup_logging():
    """Configure logging for packaged execution."""
    import logging
    
    # Configure root logger for stderr output (Tauri can capture this)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr,
        force=True,
    )
    
    # Reduce verbosity of third-party libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)

def validate_dependencies():
    """Validate that critical dependencies are available and functional."""
    
    print("INFO: Validating dependencies...", file=sys.stderr)
    
    # Test critical imports
    critical_modules = [
        ("mcp", "MCP framework"),
        ("fastmcp", "FastMCP framework"),
        ("chromadb", "ChromaDB vector database"),
        ("sentence_transformers", "Sentence Transformers"),
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("pydantic", "Pydantic data validation"),
    ]
    
    failed_imports = []
    
    for module_name, description in critical_modules:
        try:
            __import__(module_name)
            print(f"  ✓ {description}", file=sys.stderr)
        except ImportError as e:
            print(f"  ✗ {description}: {e}", file=sys.stderr)
            failed_imports.append((module_name, description, str(e)))
    
    if failed_imports:
        print("ERROR: Critical dependencies failed to load:", file=sys.stderr)
        for module_name, description, error in failed_imports:
            print(f"  - {description} ({module_name}): {error}", file=sys.stderr)
        return False
    
    print("INFO: All dependencies validated successfully", file=sys.stderr)
    return True

def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    import signal
    import atexit
    
    def signal_handler(signum, frame):
        print(f"INFO: Received signal {signum}, shutting down gracefully...", file=sys.stderr)
        sys.exit(0)
    
    def cleanup_handler():
        print("INFO: Performing cleanup on exit...", file=sys.stderr)
        # Any cleanup code would go here
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup handler
    atexit.register(cleanup_handler)

def main():
    """Main entry point for PyOxidizer-packaged MDMAI MCP Server."""
    
    print("INFO: Starting MDMAI MCP Server (PyOxidizer)", file=sys.stderr)
    
    try:
        # Step 1: Set up SQLite compatibility
        if not setup_sqlite_compatibility():
            print("FATAL: SQLite compatibility check failed", file=sys.stderr)
            sys.exit(1)
        
        # Step 2: Configure environment
        setup_environment()
        
        # Step 3: Set up logging
        setup_logging()
        
        # Step 4: Set up signal handlers
        setup_signal_handlers()
        
        # Step 5: Validate dependencies
        if not validate_dependencies():
            print("FATAL: Dependency validation failed", file=sys.stderr)
            sys.exit(1)
        
        # Step 6: Import and run the main MCP server
        print("INFO: Starting MCP server...", file=sys.stderr)
        
        # Import the main module (this must be done after all setup)
        from src.main import main as mcp_main
        
        # Run the MCP server
        return mcp_main()
        
    except KeyboardInterrupt:
        print("INFO: Received keyboard interrupt, shutting down...", file=sys.stderr)
        return 0
        
    except Exception as e:
        print(f"FATAL: Unhandled exception in main: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())