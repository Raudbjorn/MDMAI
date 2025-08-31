# PyOxidizer Packaging Guide for MDMAI MCP Server

This guide provides comprehensive instructions for packaging the MDMAI TTRPG Assistant MCP Server using PyOxidizer to create standalone, self-contained executables that work without requiring Python to be installed on the target system.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Building Process](#building-process)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## Overview

PyOxidizer is a utility for producing binaries that embed Python. It creates standalone executables with:

- **Self-contained**: All Python dependencies embedded
- **No Python Installation Required**: Target systems don't need Python installed
- **Optimized Performance**: Faster startup than traditional Python applications
- **Cross-platform**: Build for Windows, Linux, and macOS

### Key Features of Our Configuration

- **ChromaDB Compatibility**: Handles SQLite version requirements
- **FastMCP Integration**: Includes all MCP framework dependencies
- **AI/ML Libraries**: Bundles PyTorch, Transformers, and sentence-transformers
- **Security Libraries**: Includes authentication and encryption modules
- **Optimized Startup**: Critical libraries loaded in memory for fast startup

## Prerequisites

### System Requirements

- **Operating System**: Linux, Windows, or macOS
- **Architecture**: x86_64 (recommended), aarch64 (ARM64)
- **RAM**: At least 8GB (16GB recommended for building)
- **Disk Space**: At least 10GB free space for build artifacts

### Required Software

1. **Rust Toolchain** (latest stable)
2. **PyOxidizer** (version 0.24.0+)
3. **Python** (3.10 or 3.11 recommended)
4. **Git** (for cloning dependencies)

#### Platform-Specific Requirements

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install build-essential pkg-config libssl-dev

# RHEL/CentOS/Fedora
sudo yum groupinstall "Development Tools"
sudo yum install openssl-devel
```

**Windows:**
- Microsoft C++ Build Tools or Visual Studio with C++ workload
- Windows 10 SDK (latest)

**macOS:**
- Xcode Command Line Tools

## Quick Start

### 1. Install PyOxidizer

Choose one of the installation methods:

**Option A: Automated Installation (Recommended)**
```bash
# Linux/macOS
./scripts/install_pyoxidizer.sh

# Windows (PowerShell as Administrator)
.\scripts\install_pyoxidizer.ps1
```

**Option B: Manual Installation**
```bash
# Install Rust first
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install PyOxidizer
cargo install pyoxidizer
```

### 2. Build the Executable

```bash
# Build for current platform
python scripts/build_pyoxidizer.py

# Build for all platforms
python scripts/build_pyoxidizer.py --all

# Build for specific platforms
python scripts/build_pyoxidizer.py --platform linux windows
```

### 3. Test the Build

```bash
# Simple startup test
python scripts/test_pyoxidizer_stdio.py --simple

# Comprehensive MCP protocol test
python scripts/test_pyoxidizer_stdio.py
```

## Detailed Setup

### 1. Install Rust

PyOxidizer is written in Rust, so you need the Rust toolchain:

```bash
# Install rustup (Rust installer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Source the cargo environment
source ~/.cargo/env

# Verify installation
rustc --version
cargo --version
```

### 2. Install PyOxidizer

```bash
# Install PyOxidizer
cargo install pyoxidizer

# Verify installation
pyoxidizer --version
```

### 3. Install Project Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install additional SQLite compatibility package
pip install pysqlite3-binary
```

## Building Process

### Configuration File

The main configuration is in `pyoxidizer.bzl`. Key sections:

```python
# Entry point configuration
config.run_command = "from src.oxidizer_main import main; main()"

# Dependency management
core_packages = ["mcp==1.0.0", "fastmcp==0.1.5", ...]
vector_db_packages = ["chromadb==0.4.22", "pysqlite3-binary==0.5.2.post4", ...]
```

### Build Targets

- **install**: Basic executable build
- **windows-exe**: Windows executable
- **linux-exe**: Linux executable
- **macos-exe**: macOS executable
- **msi**: Windows MSI installer
- **macos-app-bundle**: macOS application bundle

### Build Commands

```bash
# Build using PyOxidizer directly
pyoxidizer build install

# Build specific target
pyoxidizer build windows-exe

# Build with custom target triple
pyoxidizer build --target-triple x86_64-pc-windows-msvc windows-exe
```

### Using Build Script

The build script provides additional features:

```bash
# Show help
python scripts/build_pyoxidizer.py --help

# Clean build artifacts
python scripts/build_pyoxidizer.py --clean

# Build with cleanup
python scripts/build_pyoxidizer.py --clean --platform linux
```

## Testing

### Automated Testing

```bash
# Test executable startup
python scripts/test_pyoxidizer_stdio.py --simple

# Test MCP protocol communication
python scripts/test_pyoxidizer_stdio.py

# Test specific executable
python scripts/test_pyoxidizer_stdio.py --executable ./dist/pyoxidizer/mdmai-mcp-server-linux-x86_64/mdmai-mcp-server
```

### Manual Testing

1. **Startup Test**:
```bash
./dist/pyoxidizer/mdmai-mcp-server-linux-x86_64/mdmai-mcp-server
```

2. **MCP Protocol Test**:
```bash
# Send initialize message via stdin
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}}}' | ./mdmai-mcp-server
```

3. **Integration with Tauri**:
   - Copy executable to `desktop/backend/`
   - Update Tauri configuration to use the executable
   - Test with the desktop application

## Troubleshooting

### Common Issues

#### 1. SQLite Version Error
```
ERROR: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0
```

**Solution**: The configuration includes `pysqlite3-binary` and automatic module substitution.

#### 2. Large Executable Size
```
WARNING: Executable size is very large (>500MB)
```

**Solutions**:
- Review included packages in `pyoxidizer.bzl`
- Move non-critical packages to filesystem location
- Enable compression: `upx=True` (if UPX is installed)

#### 3. Missing Dependencies
```
ERROR: Failed to import module 'xyz'
```

**Solutions**:
- Add missing package to `pyoxidizer.bzl`
- Check `hiddenimports` section
- Verify package is listed in `requirements.txt`

#### 4. Slow Startup
```
INFO: Server taking >10 seconds to start
```

**Solutions**:
- Move critical packages to `in-memory` location
- Review PyTorch and Transformers cache settings
- Consider using model quantization

#### 5. Build Failures

**Linux Build Issues**:
```bash
# Missing build tools
sudo apt-get install build-essential pkg-config libssl-dev

# Python version mismatch
pyenv install 3.11.0
pyenv global 3.11.0
```

**Windows Build Issues**:
```powershell
# Missing Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# PowerShell execution policy
Set-ExecutionPolicy Bypass -Scope Process -Force
```

**macOS Build Issues**:
```bash
# Missing Xcode tools
xcode-select --install

# ARM64 build on Intel Mac
rustup target add aarch64-apple-darwin
```

### Debugging Tips

1. **Enable Verbose Logging**:
```python
# In src/oxidizer_main.py
logging.basicConfig(level=logging.DEBUG)
```

2. **Check Process Environment**:
```bash
# List environment variables
env | grep -E "(PYTHON|PATH|TORCH|TRANSFORMERS|CHROMA)"
```

3. **Inspect Dependencies**:
```bash
# List imported modules
python -c "import sys; print('\n'.join(sys.modules.keys()))"
```

4. **Monitor Resource Usage**:
```bash
# Monitor memory usage
top -p $(pidof mdmai-mcp-server)

# Monitor file access
strace -e trace=openat ./mdmai-mcp-server 2>&1 | grep -v ENOENT
```

## Advanced Configuration

### Custom Dependency Management

```python
# In pyoxidizer.bzl
def install_custom_packages(exe):
    """Install packages with custom configurations."""
    
    # Install specific PyTorch version
    for resource in exe.pip_install(["torch==2.8.0+cpu"], ["--index-url", "https://download.pytorch.org/whl/cpu"]):
        resource.add_location = "filesystem-relative:lib"
        exe.add_python_resource(resource)
    
    # Install from wheel files
    for resource in exe.pip_install(["/path/to/custom-package.whl"]):
        resource.add_location = "in-memory"
        exe.add_python_resource(resource)
```

### Performance Optimization

```python
# Memory optimization
config.sys_frozen = True
config.sys_dont_write_bytecode = True

# Startup optimization
config.module_search_paths = ["$ORIGIN", "$ORIGIN/lib"]
config.optimize_level = 2
```

### Platform-Specific Builds

```bash
# Cross-compilation for different targets
pyoxidizer build --target-triple x86_64-pc-windows-msvc windows-exe
pyoxidizer build --target-triple aarch64-apple-darwin macos-exe
pyoxidizer build --target-triple x86_64-unknown-linux-gnu linux-exe
```

### Custom Resource Handling

```python
# Add data files
exe.add_python_resource(exe.read_package_root(
    path="data",
    packages=["data"],
))

# Add configuration files
exe.add_python_resource(exe.read_virtual_file(
    path="config/default.json",
    content=json.dumps(default_config).encode("utf-8"),
))
```

## Distribution

### Packaging for Distribution

```bash
# Create distribution packages
python scripts/build_pyoxidizer.py --all

# Packages are created in dist/pyoxidizer/
ls -la dist/pyoxidizer/
```

### Archive Formats

- **Linux**: `.tar.gz` archives
- **Windows**: `.zip` archives
- **macOS**: `.tar.gz` archives or `.app` bundles

### Integration with Desktop Application

1. Copy the appropriate executable to `desktop/backend/`
2. Update `desktop/frontend/src-tauri/tauri.conf.json`:

```json
{
  "tauri": {
    "bundle": {
      "externalBin": [
        "mdmai-mcp-server"
      ]
    }
  }
}
```

3. Update the sidecar configuration to use the packaged executable

## Performance Characteristics

### Startup Time
- **Cold Start**: ~5-10 seconds (first run)
- **Warm Start**: ~2-3 seconds (subsequent runs)

### Memory Usage
- **Base Usage**: ~200-300 MB
- **With Models Loaded**: ~1-2 GB
- **Peak Usage**: ~3-4 GB (during large operations)

### Executable Size
- **Minimal Build**: ~300-400 MB
- **Full Build**: ~800MB-1.2GB
- **With ML Models**: ~1.5-2GB

## Security Considerations

### Code Signing

```bash
# Windows code signing
signtool sign /f certificate.pfx /p password /t http://timestamp.server.com mdmai-mcp-server.exe

# macOS code signing
codesign --force --verify --verbose --sign "Developer ID" mdmai-mcp-server
```

### Antivirus Considerations

Some antivirus software may flag PyOxidizer executables as potentially unwanted programs (PUPs) due to their self-extracting nature. Consider:

1. Code signing the executable
2. Submitting to antivirus vendors for whitelisting
3. Providing installation instructions for users

## Support and Maintenance

### Updating Dependencies

1. Update `requirements.txt`
2. Update package versions in `pyoxidizer.bzl`
3. Rebuild and test
4. Update documentation

### Version Management

```python
# In pyoxidizer.bzl
APP_VERSION = "0.1.0"
```

### Monitoring and Logging

The packaged executable includes structured logging that outputs to stderr, which can be captured by the Tauri parent process.

## Conclusion

PyOxidizer provides a robust solution for packaging the MDMAI MCP Server as a standalone executable. While the initial setup requires some configuration, the result is a self-contained application that can be distributed without Python installation requirements.

For additional support or questions about the packaging process, refer to the project documentation or raise an issue in the project repository.