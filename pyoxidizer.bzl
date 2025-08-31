# -*- mode: python ; coding: utf-8 -*-
"""
PyOxidizer Configuration for MDMAI MCP Server
Creates a standalone executable with embedded Python and all dependencies
for the TTRPG Assistant MCP server.

This configuration handles complex dependencies including:
- ChromaDB vector database with SQLite requirements
- FastMCP framework
- PyTorch and sentence-transformers
- PDF processing libraries
- Security and authentication modules

Author: MDMAI Project
PyOxidizer Version: 0.24.0+
"""

# Configuration variables
PYOXIDIZER_VERSION = "0.24.0"
AUTHOR = "MDMAI Project"
APP_NAME = "mdmai-mcp-server"

def get_app_version():
    """Read the version from pyproject.toml."""
    import re
    try:
        with open("pyproject.toml", "r", encoding="utf-8") as f:
            content = f.read()
        # Look for a line like: version = "0.1.0"
        match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
        if match:
            return match.group(1)
    except Exception as e:
        print(f"Warning: Could not read version from pyproject.toml: {e}")
    return "0.1.0"  # fallback default

APP_VERSION = get_app_version()
# Build configuration
DEBUG_BUILD = VARS.get("debug", False)
OPTIMIZE_LEVEL = 2 if not DEBUG_BUILD else 0
ENABLE_TELEMETRY = False

# Platform-specific settings
def get_platform_suffix():
    """Get platform-specific suffix for build artifacts."""
    if BUILD_TARGET_TRIPLE.contains("windows"):
        return ".exe"
    return ""

def resolve_targets():
    """Resolve build targets for the MDMAI MCP Server."""
    return [
        "install",
        "windows-exe",
        "linux-exe", 
        "macos-exe",
    ]

def make_python_distribution():
    """Configure the Python distribution for packaging."""
    # Use Python 3.11 for better SQLite compatibility with ChromaDB
    return default_python_distribution(flavor="standalone_dynamic")

def configure_python_executable():
    """Configure the main Python executable with all dependencies."""
    
    # Get Python distribution
    dist = make_python_distribution()
    
    # Create Python executable configuration
    config = dist.make_python_interpreter_config()
    
    # Configure Python interpreter settings for MCP server
    config.run_command = "from src.oxidizer_main import main; main()"
    
    # Enable optimizations
    config.optimization_level = OPTIMIZE_LEVEL
    config.write_modules_directory_env = None  # Don't write modules to disk
    
    # Configure module search paths
    config.module_search_paths = [
        "$ORIGIN",
        "$ORIGIN/lib",
        "$ORIGIN/packages",
    ]
    
    # Memory management for large ML models
    config.sys_frozen = True
    config.sys_meipass = None
    
    # Disable buffering for stdio communication with Tauri
    config.buffered_stdio = False
    config.stdio_encoding_errors = "strict"
    config.stdio_encoding_name = "utf-8"
    
    return dist.to_python_executable(
        name=APP_NAME + get_platform_suffix(),
        config=config,
    )

def install_application_dependencies(exe):
    """Install all application dependencies into the executable."""
    
    # Core MCP and FastMCP dependencies
    core_packages = [
        "mcp==1.0.0",
        "fastmcp==0.1.5",
        "fastapi==0.109.0",
        "uvicorn[standard]==0.27.0",
        "pydantic==2.5.0",
        "pydantic-settings==2.1.0",
    ]
    
    # Vector database and ChromaDB with SQLite workaround
    vector_db_packages = [
        "chromadb==0.4.22",
        "pysqlite3-binary==0.5.2.post4",  # SQLite 3.35+ compatibility
        "sqlalchemy==2.0.25",
    ]
    
    # AI/ML dependencies with optimized versions
    ml_packages = [
        "sentence-transformers==2.3.0",
        "torch==2.8.0",
        "transformers==4.53.0",
        "tiktoken==0.5.0",
        "numpy==1.26.0",
        "rank-bm25==0.2.2",
    ]
    
    # PDF processing libraries
    pdf_packages = [
        "pypdf==6.0.0",
        "pdfplumber==0.10.3",
        "pikepdf==8.10.0",
        "python-magic==0.4.27",
    ]
    
    # HTTP and async libraries
    async_packages = [
        "httpx==0.26.0",
        "aiohttp==3.12.16",
        "aiofiles==23.2.0",
        "tenacity==8.2.0",
    ]
    
    # Security and authentication
    security_packages = [
        "python-jose[cryptography]==3.3.0",
        "passlib[bcrypt]==1.7.4",
        "cryptography==44.0.1",
        "authlib==1.3.1",
    ]
    
    # Utilities and logging
    utility_packages = [
        "structlog==24.1.0",
        "python-dotenv==1.0.0",
        "rich==13.0.0",
        "orjson==3.9.15",
        "click==8.1.0",
    ]
    
    # Error handling patterns
    error_packages = [
        "returns==0.22.0",
    ]
    
    # Combine all packages
    all_packages = (
        core_packages + 
        vector_db_packages + 
        ml_packages + 
        pdf_packages + 
        async_packages + 
        security_packages + 
        utility_packages + 
        error_packages
    )
    
    # Install packages with error handling
    for package in all_packages:
        try:
            for resource in exe.pip_install([package]):
                # Place critical libraries in memory for faster startup
                if any(critical in package for critical in ["mcp", "fastmcp", "chromadb", "torch"]):
                    resource.add_location = "in-memory"
                else:
                    # Less critical packages can be on filesystem
                    resource.add_location = "filesystem-relative:lib"
                
                exe.add_python_resource(resource)
        except:
            # Log package installation failures but continue
            print(f"Warning: Failed to install package {package}")
    
    return exe

def add_application_resources(exe):
    """Add application-specific resources and configurations."""
    
    # Add the main application source code
    for resource in exe.read_package_root(
        path="src",
        packages=["src"],
    ):
        resource.add_location = "in-memory"
        exe.add_python_resource(resource)
    
    # Add configuration files
    config_files = [
        "config/logging_config.py",
        "config/settings.py", 
        "config/__init__.py",
    ]
    
    for config_file in config_files:
        try:
            exe.add_python_resource(exe.read_virtual_file(
                path=config_file,
                is_package_data=True,
            ))
        except:
            print(f"Warning: Could not add config file {config_file}")
    
    # Add SQLite workaround module for ChromaDB compatibility
    sqlite_workaround = """
# SQLite version workaround for ChromaDB compatibility
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass  # Fall back to system sqlite3
"""
    
    exe.add_python_resource(exe.read_virtual_file(
        path="sqlite_workaround.py",
        content=sqlite_workaround.encode("utf-8"),
    ))
    
    return exe

def configure_windows_executable(exe):
    """Configure Windows-specific executable settings."""
    exe.windows_subsystem = "console"  # Console app for stdio
    exe.windows_icon_path = "desktop/frontend/src-tauri/icons/icon.ico"
    
    # Version information
    exe.windows_file_version = APP_VERSION
    exe.windows_product_version = APP_VERSION
    exe.windows_file_description = "MDMAI TTRPG Assistant MCP Server"
    exe.windows_product_name = "MDMAI MCP Server"
    exe.windows_company_name = "MDMAI Project"
    exe.windows_copyright = "Copyright (c) 2024 MDMAI Project"
    
    return exe

def make_install():
    """Create installation target with the configured executable."""
    exe = configure_python_executable()
    exe = install_application_dependencies(exe)
    exe = add_application_resources(exe)
    
    # Configure platform-specific settings
    if BUILD_TARGET_TRIPLE.contains("windows"):
        exe = configure_windows_executable(exe)
    
    files = FileManifest()
    files.add_python_resource(".", exe)
    
    return files

def make_windows_exe():
    """Create Windows executable target."""
    return make_install()

def make_linux_exe():
    """Create Linux executable target."""
    return make_install()

def make_macos_exe():
    """Create macOS executable target.""" 
    return make_install()

def make_msi():
    """Create Windows MSI installer."""
    if not BUILD_TARGET_TRIPLE.contains("windows"):
        return None
        
    exe = make_windows_exe()
    
    # Create MSI installer
    msi = WiXMSIBuilder(
        id_prefix="mdmai.mcp.server",
        product_name="MDMAI MCP Server",
        product_version=APP_VERSION,
        product_manufacturer="MDMAI Project",
    )
    
    msi.help_url = "https://github.com/mdmai-project/MDMAI"
    msi.upgrade_code = "e2b7c1a2-4f3b-4c8e-9b2a-7e6d2f1c5a9b"  # Unique GUID for MSI upgrade code
    
    msi.add_program_files_manifest(exe)
    
    return msi

def make_macos_app_bundle():
    """Create macOS application bundle."""
    if not BUILD_TARGET_TRIPLE.contains("apple"):
        return None
        
    exe = make_macos_exe()
    
    bundle = MacOsApplicationBundleBuilder("MDMAI MCP Server")
    bundle.set_info_plist_key("CFBundleVersion", APP_VERSION)
    bundle.set_info_plist_key("CFBundleShortVersionString", APP_VERSION)
    bundle.set_info_plist_key("CFBundleIdentifier", "com.mdmai.mcp-server")
    bundle.set_info_plist_key("CFBundleName", "MDMAI MCP Server")
    bundle.set_info_plist_key("CFBundleDisplayName", "MDMAI MCP Server")
    
    bundle.add_macos_file(exe, "Contents/MacOS/")
    
    return bundle

# Register all build targets
register_target("install", make_install)
register_target("windows-exe", make_windows_exe, depends=["install"])  
register_target("linux-exe", make_linux_exe, depends=["install"])
register_target("macos-exe", make_macos_exe, depends=["install"])

# Platform-specific installers
register_target("msi", make_msi, depends=["windows-exe"])
register_target("macos-app-bundle", make_macos_app_bundle, depends=["macos-exe"])

# Resolve which targets to build
resolve_targets_impl(resolve_targets)