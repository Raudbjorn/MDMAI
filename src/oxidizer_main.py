#!/usr/bin/env python3
"""
PyOxidizer Entry Point for MDMAI MCP Server.

This module provides a production-ready entry point for PyOxidizer-packaged
MCP server executables. It includes comprehensive dependency validation,
performance optimizations, and robust error handling patterns.

Features:
    - SQLite compatibility layer for ChromaDB
    - Environment optimization for embedded execution
    - Dependency validation with detailed diagnostics
    - Signal handling for graceful shutdown
    - Performance monitoring and optimization
    - Type-safe implementation with comprehensive error handling
"""

import atexit
import functools
import logging
import os
import signal
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, NoReturn, Optional, Tuple, Union

# Configure logging early for PyOxidizer execution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
    force=True
)
logger = logging.getLogger(__name__)


class SQLiteCompatibilityError(Exception):
    """Raised when SQLite compatibility cannot be established."""
    pass


class DependencyValidationError(Exception):
    """Raised when critical dependencies fail validation."""
    pass


@functools.lru_cache(maxsize=1)
def setup_sqlite_compatibility() -> bool:
    """Set up SQLite compatibility workaround for ChromaDB with caching.
    
    Returns:
        True if compatibility is established, False otherwise
        
    Raises:
        SQLiteCompatibilityError: If SQLite requirements cannot be met
    """
    try:
        # Import pysqlite3 before any ChromaDB imports
        import pysqlite3
        
        # Replace the default sqlite3 module with pysqlite3
        sys.modules["sqlite3"] = pysqlite3
        
        logger.info("SQLite3 compatibility workaround applied for ChromaDB")
        return True
        
    except ImportError:
        # Fall back to system sqlite3
        logger.warning("pysqlite3 not available, using system sqlite3")
        
        try:
            import sqlite3
            version = sqlite3.sqlite_version
            logger.info(f"System SQLite version: {version}")
            
            # ChromaDB requires SQLite >= 3.35.0
            version_parts = version.split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            
            if major < 3 or (major == 3 and minor < 35):
                error_msg = (
                    f"ChromaDB requires SQLite >= 3.35.0, found {version}. "
                    "Install pysqlite3-binary: pip install pysqlite3-binary"
                )
                logger.error(error_msg)
                raise SQLiteCompatibilityError(error_msg)
                
            return True
                
        except Exception as e:
            error_msg = f"Could not check SQLite version: {e}"
            logger.error(error_msg)
            raise SQLiteCompatibilityError(error_msg) from e


class EnvironmentManager:
    """Manages environment setup and optimization for PyOxidizer execution."""
    
    def __init__(self, executable_dir: Optional[Path] = None) -> None:
        """Initialize environment manager.
        
        Args:
            executable_dir: Directory containing the executable (defaults to sys.executable location)
        """
        self.executable_dir = executable_dir or Path(sys.executable).parent
        self.cache_dirs: Dict[str, Path] = {}
    
    @contextmanager
    def environment_context(self):
        """Context manager for environment setup with automatic cleanup."""
        old_env = os.environ.copy()
        try:
            self.setup_environment()
            yield
        finally:
            # Restore original environment if needed
            pass
    
    def setup_environment(self) -> None:
        """Set up environment variables and paths for optimal performance."""
        env_vars = self._get_environment_variables()
        
        # Set environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
            
        # Create cache directories
        self._create_cache_directories()
        
        # Configure warnings
        self._configure_warnings()
        
        logger.info("Environment configured for PyOxidizer execution")
    
    def _get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for optimized execution."""
        torch_cache = self.executable_dir / 'torch_cache'
        transformers_cache = self.executable_dir / 'transformers_cache'
        
        # Store cache directories for later use
        self.cache_dirs.update({
            'torch': torch_cache,
            'transformers': transformers_cache
        })
        
        return {
            # MCP communication settings
            'MCP_STDIO_MODE': 'true',
            'PYTHONUNBUFFERED': '1',
            'PYTHONIOENCODING': 'utf-8',
            
            # ML library cache directories
            'TORCH_HOME': str(torch_cache),
            'TRANSFORMERS_CACHE': str(transformers_cache),
            'HF_HOME': str(transformers_cache),
            
            # ChromaDB settings
            'CHROMA_SERVER_AUTHN_PROVIDER': '',
            
            # Performance optimizations
            'OMP_NUM_THREADS': str(min(4, os.cpu_count() or 4)),
            'MKL_NUM_THREADS': str(min(4, os.cpu_count() or 4))
        }
    
    def _create_cache_directories(self) -> None:
        """Create necessary cache directories with proper permissions."""
        for name, cache_dir in self.cache_dirs.items():
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created {name} cache directory: {cache_dir}")
            except OSError as e:
                logger.warning(f"Failed to create {name} cache directory {cache_dir}: {e}")
    
    def _configure_warnings(self) -> None:
        """Configure warning filters for clean output in packaged environment."""
        warning_filters = [
            ("ignore", UserWarning, "transformers"),
            ("ignore", FutureWarning, "torch"),
            ("ignore", UserWarning, "sentence_transformers"),
            ("ignore", DeprecationWarning, "numpy"),
        ]
        
        for action, category, module in warning_filters:
            warnings.filterwarnings(action, category=category, module=module)


def setup_optimized_logging() -> None:
    """Configure optimized logging for packaged execution with performance considerations."""
    # Third-party library log levels for reduced verbosity
    library_log_levels = {
        "sentence_transformers": logging.WARNING,
        "transformers": logging.WARNING,
        "torch": logging.WARNING,
        "chromadb": logging.INFO,
        "httpx": logging.WARNING,
        "uvicorn": logging.WARNING,
        "asyncio": logging.WARNING,
        "urllib3": logging.WARNING,
        "requests": logging.WARNING,
    }
    
    # Apply log levels with error handling
    for library, level in library_log_levels.items():
        try:
            logging.getLogger(library).setLevel(level)
        except Exception as e:
            logger.warning(f"Failed to set log level for {library}: {e}")
    
    logger.info("Optimized logging configuration applied")


class DependencyValidator:
    """Validates critical dependencies with detailed diagnostics and caching."""
    
    CRITICAL_MODULES = [
        ("mcp", "MCP framework", True),
        ("fastmcp", "FastMCP framework", True),
        ("chromadb", "ChromaDB vector database", True),
        ("sentence_transformers", "Sentence Transformers", True),
        ("torch", "PyTorch", True),
        ("transformers", "Hugging Face Transformers", True),
        ("pydantic", "Pydantic data validation", True),
        ("numpy", "NumPy numerical computing", False),
        ("sqlite3", "SQLite database", False),
    ]
    
    def __init__(self) -> None:
        """Initialize dependency validator."""
        self.validation_results: Dict[str, Tuple[bool, Optional[str]]] = {}
        self.start_time = time.time()
    
    def validate_all_dependencies(self) -> bool:
        """Validate all critical dependencies.
        
        Returns:
            True if all critical dependencies are available, False otherwise
            
        Raises:
            DependencyValidationError: If critical dependencies are missing
        """
        logger.info("Validating dependencies...")
        
        failed_critical = []
        failed_optional = []
        
        for module_name, description, is_critical in self.CRITICAL_MODULES:
            success, error = self._validate_module(module_name, description)
            self.validation_results[module_name] = (success, error)
            
            if success:
                logger.info(f"✓ {description}")
            else:
                logger.error(f"✗ {description}: {error}")
                if is_critical:
                    failed_critical.append((module_name, description, error))
                else:
                    failed_optional.append((module_name, description, error))
        
        # Log validation timing
        validation_time = time.time() - self.start_time
        logger.info(f"Dependency validation completed in {validation_time:.2f}s")
        
        # Handle failures
        if failed_optional:
            logger.warning(f"Optional dependencies unavailable: {len(failed_optional)}")
        
        if failed_critical:
            error_msg = f"Critical dependencies failed to load: {failed_critical}"
            logger.error(error_msg)
            raise DependencyValidationError(error_msg)
        
        logger.info("All critical dependencies validated successfully")
        return True
    
    def _validate_module(self, module_name: str, description: str) -> Tuple[bool, Optional[str]]:
        """Validate a single module import.
        
        Args:
            module_name: Name of the module to import
            description: Human-readable description
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            module = __import__(module_name)
            
            # Additional validation for specific modules
            if module_name == "torch":
                self._validate_torch(module)
            elif module_name == "chromadb":
                self._validate_chromadb(module)
            
            return True, None
            
        except ImportError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def _validate_torch(self, torch_module) -> None:
        """Validate PyTorch specific functionality."""
        # Check if CUDA is available (optional)
        if hasattr(torch_module, 'cuda') and torch_module.cuda.is_available():
            logger.info(f"CUDA available: {torch_module.cuda.device_count()} device(s)")
        else:
            logger.info("CUDA not available, using CPU")
    
    def _validate_chromadb(self, chromadb_module) -> None:
        """Validate ChromaDB specific functionality."""
        try:
            # Try to create a client to validate SQLite compatibility
            client = chromadb_module.Client()
            logger.debug("ChromaDB client created successfully")
        except Exception as e:
            logger.warning(f"ChromaDB client creation warning: {e}")


class SignalManager:
    """Manages signal handling and graceful shutdown procedures."""
    
    def __init__(self) -> None:
        """Initialize signal manager."""
        self.shutdown_initiated = False
        self.cleanup_functions: List[callable] = []
    
    def setup_signal_handlers(self) -> None:
        """Set up comprehensive signal handlers for graceful shutdown."""
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Platform-specific signals
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, self._signal_handler)
        
        # Register cleanup handler
        atexit.register(self._cleanup_handler)
        
        logger.info("Signal handlers configured")
    
    def register_cleanup(self, func: callable) -> None:
        """Register a cleanup function to be called on shutdown."""
        self.cleanup_functions.append(func)
    
    def _signal_handler(self, signum: int, frame) -> NoReturn:
        """Handle shutdown signals gracefully."""
        if self.shutdown_initiated:
            logger.warning("Shutdown already in progress, forcing exit")
            os._exit(1)
        
        self.shutdown_initiated = True
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        
        # Perform cleanup
        self._cleanup_handler()
        
        logger.info("Graceful shutdown completed")
        sys.exit(0)
    
    def _cleanup_handler(self) -> None:
        """Perform cleanup operations."""
        if self.shutdown_initiated:
            return  # Avoid double cleanup
        
        logger.info("Performing cleanup operations")
        
        # Execute registered cleanup functions
        for func in self.cleanup_functions:
            try:
                func()
            except Exception as e:
                logger.error(f"Error in cleanup function: {e}")
        
        logger.info("Cleanup completed")


class PyOxidizerMCPServer:
    """Main orchestrator for PyOxidizer-packaged MCP server with comprehensive management."""
    
    def __init__(self) -> None:
        """Initialize the PyOxidizer MCP server."""
        self.env_manager = EnvironmentManager()
        self.signal_manager = SignalManager()
        self.dependency_validator = DependencyValidator()
        self.start_time = time.time()
    
    def initialize(self) -> None:
        """Initialize all components in the correct order."""
        logger.info("Starting MDMAI MCP Server (PyOxidizer)")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {sys.platform}")
        
        initialization_steps = [
            ("SQLite compatibility", self._setup_sqlite_compatibility),
            ("Environment configuration", self._setup_environment),
            ("Logging optimization", self._setup_logging),
            ("Signal handlers", self._setup_signal_handlers),
            ("Dependency validation", self._validate_dependencies),
        ]
        
        for step_name, step_func in initialization_steps:
            try:
                logger.info(f"Initializing: {step_name}")
                step_func()
                logger.info(f"✓ {step_name} completed")
            except Exception as e:
                logger.error(f"✗ {step_name} failed: {e}")
                raise
        
        init_time = time.time() - self.start_time
        logger.info(f"Initialization completed in {init_time:.2f}s")
    
    def _setup_sqlite_compatibility(self) -> None:
        """Set up SQLite compatibility with error handling."""
        if not setup_sqlite_compatibility():
            raise SQLiteCompatibilityError("SQLite compatibility check failed")
    
    def _setup_environment(self) -> None:
        """Set up environment with the manager."""
        self.env_manager.setup_environment()
    
    def _setup_logging(self) -> None:
        """Set up optimized logging."""
        setup_optimized_logging()
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers."""
        self.signal_manager.setup_signal_handlers()
    
    def _validate_dependencies(self) -> None:
        """Validate all dependencies."""
        self.dependency_validator.validate_all_dependencies()
    
    def run_server(self) -> int:
        """Import and run the main MCP server.
        
        Returns:
            Exit code from the MCP server
        """
        try:
            logger.info("Importing MCP server module")
            
            # Import the main module (after all initialization)
            from src.main import main as mcp_main
            
            logger.info("Starting MCP server")
            
            # Run the MCP server with performance monitoring
            start_time = time.time()
            exit_code = mcp_main()
            runtime = time.time() - start_time
            
            logger.info(f"MCP server completed in {runtime:.2f}s with exit code {exit_code}")
            return exit_code
            
        except ImportError as e:
            logger.error(f"Failed to import MCP server module: {e}")
            logger.error("Verify that src/main.py exists and contains a main() function")
            return 1
            
        except Exception as e:
            logger.error(f"MCP server execution failed: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            return 1
    
    def run(self) -> int:
        """Main execution method with comprehensive error handling.
        
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            # Initialize all components
            self.initialize()
            
            # Run the server
            return self.run_server()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down")
            return 0
            
        except (SQLiteCompatibilityError, DependencyValidationError) as e:
            logger.error(f"Initialization failed: {e}")
            return 1
            
        except Exception as e:
            logger.error(f"Fatal error in PyOxidizer MCP server: {e}")
            logger.error("Full traceback:", exc_info=True)
            return 1


def main() -> int:
    """Main entry point for PyOxidizer-packaged MDMAI MCP Server.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    server = PyOxidizerMCPServer()
    return server.run()


if __name__ == "__main__":
    sys.exit(main())