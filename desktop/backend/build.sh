#!/usr/bin/env bash
#
# Build Script for TTRPG Assistant MCP Server Backend
# Supports cross-platform architecture detection and building
#
# Supported platforms:
#   - macOS: x86_64 (Intel), aarch64 (Apple Silicon M1/M2/M3)
#   - Linux: x86_64, aarch64
#   - Windows: x86_64 (via WSL or Git Bash)
#

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Function to detect system architecture
detect_architecture() {
    local arch=$(uname -m)
    local normalized_arch=""
    
    case "$arch" in
        x86_64|amd64)
            normalized_arch="x86_64"
            ;;
        arm64|aarch64)
            # ARM64 is reported as 'arm64' on macOS but needs 'aarch64' for Rust
            normalized_arch="aarch64"
            ;;
        armv7l|armv7)
            normalized_arch="armv7"
            ;;
        i386|i686)
            normalized_arch="i686"
            ;;
        *)
            log_error "Unsupported architecture: $arch"
            exit 1
            ;;
    esac
    
    echo "$normalized_arch"
}

# Function to detect operating system
detect_os() {
    local os=""
    
    case "$(uname -s)" in
        Darwin)
            os="darwin"
            ;;
        Linux)
            os="linux"
            ;;
        MINGW*|MSYS*|CYGWIN*|Windows_NT)
            os="windows"
            ;;
        *)
            log_error "Unsupported operating system: $(uname -s)"
            exit 1
            ;;
    esac
    
    echo "$os"
}

# Function to get the full target triple for Rust
get_rust_target() {
    local arch="$1"
    local os="$2"
    local target=""
    
    case "$os" in
        darwin)
            # macOS uses apple-darwin
            target="${arch}-apple-darwin"
            ;;
        linux)
            # Detect libc variant (glibc vs musl)
            if ldd --version 2>&1 | grep -q musl; then
                target="${arch}-unknown-linux-musl"
            else
                target="${arch}-unknown-linux-gnu"
            fi
            ;;
        windows)
            # Windows typically uses MSVC
            target="${arch}-pc-windows-msvc"
            ;;
        *)
            log_error "Cannot determine Rust target for OS: $os"
            exit 1
            ;;
    esac
    
    echo "$target"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check build dependencies
check_dependencies() {
    local missing_deps=()
    
    # Check for Python
    if ! command_exists python3 && ! command_exists python; then
        missing_deps+=("python3")
    fi
    
    # Check for pip
    if ! command_exists pip3 && ! command_exists pip; then
        missing_deps+=("pip")
    fi
    
    # Check for Rust (if using PyOxidizer)
    if [[ "${USE_PYOXIDIZER:-false}" == "true" ]]; then
        if ! command_exists cargo; then
            missing_deps+=("rust/cargo")
        fi
    fi
    
    # Check for PyInstaller (if using PyInstaller)
    if [[ "${USE_PYINSTALLER:-true}" == "true" ]]; then
        if ! python3 -c "import PyInstaller" 2>/dev/null; then
            log_warning "PyInstaller not found, will attempt to install"
        fi
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install the missing dependencies and try again"
        exit 1
    fi
}

# Function to setup Python virtual environment
setup_venv() {
    local venv_path="${1:-venv}"
    
    if [[ ! -d "$venv_path" ]]; then
        log_info "Creating Python virtual environment at $venv_path"
        python3 -m venv "$venv_path"
    fi
    
    # Activate virtual environment
    if [[ -f "$venv_path/bin/activate" ]]; then
        source "$venv_path/bin/activate"
    elif [[ -f "$venv_path/Scripts/activate" ]]; then
        source "$venv_path/Scripts/activate"
    else
        log_error "Cannot find virtual environment activation script"
        exit 1
    fi
    
    log_success "Virtual environment activated"
}

# Function to install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies"
    
    # Upgrade pip
    pip install --upgrade pip --quiet
    
    # Install project dependencies
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt --quiet
    elif [[ -f "pyproject.toml" ]]; then
        pip install -e . --quiet
    else
        log_warning "No requirements.txt or pyproject.toml found"
    fi
    
    # Install build tools
    if [[ "${USE_PYINSTALLER:-true}" == "true" ]]; then
        pip install pyinstaller --quiet
        log_success "PyInstaller installed"
    fi
    
    if [[ "${USE_PYOXIDIZER:-false}" == "true" ]]; then
        pip install pyoxidizer --quiet
        log_success "PyOxidizer installed"
    fi
}

# Function to build with PyInstaller
build_with_pyinstaller() {
    local arch="$1"
    local os="$2"
    local target="$3"
    
    log_info "Building with PyInstaller for $target"
    
    # Create PyInstaller spec if it doesn't exist
    if [[ ! -f "pyinstaller.spec" ]]; then
        log_info "Creating PyInstaller spec file"
        cat > pyinstaller.spec << 'EOF'
# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

# Get the architecture and OS
arch = sys.argv[-2] if len(sys.argv) > 2 else 'x86_64'
os_name = sys.argv[-1] if len(sys.argv) > 1 else 'linux'

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('*.json', '.'),
        ('*.yaml', '.'),
        ('*.yml', '.'),
    ],
    hiddenimports=[
        'encodings',
        'asyncio',
        'concurrent.futures',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=f'mcp-server-{arch}-{os_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
EOF
    fi
    
    # Build with PyInstaller
    pyinstaller \
        --clean \
        --noconfirm \
        --onefile \
        --name "mcp-server" \
        pyinstaller.spec \
        -- "$arch" "$os"
    
    # Move output to correct location
    local output_dir="dist/${target}"
    mkdir -p "$output_dir"
    
    if [[ -f "dist/mcp-server-${arch}-${os}" ]]; then
        mv "dist/mcp-server-${arch}-${os}" "$output_dir/mcp-server"
    elif [[ -f "dist/mcp-server-${arch}-${os}.exe" ]]; then
        mv "dist/mcp-server-${arch}-${os}.exe" "$output_dir/mcp-server.exe"
    fi
    
    log_success "Build complete: $output_dir/mcp-server"
}

# Function to build with PyOxidizer
build_with_pyoxidizer() {
    local arch="$1"
    local os="$2"
    local target="$3"
    
    log_info "Building with PyOxidizer for $target"
    
    # Create PyOxidizer config if it doesn't exist
    if [[ ! -f "pyoxidizer.toml" ]]; then
        log_info "Creating PyOxidizer configuration"
        cat > pyoxidizer.toml << EOF
[build]
target = "$target"

[[bin]]
name = "mcp-server"
path = "main.py"

[python]
version = "3.10"

[packaging]
include_source = false
include_resources = true
EOF
    fi
    
    # Build with PyOxidizer
    pyoxidizer build --release --target "$target"
    
    # Move output to correct location
    local output_dir="dist/${target}"
    mkdir -p "$output_dir"
    
    local oxidizer_output="build/${target}/release/mcp-server"
    if [[ "$os" == "windows" ]]; then
        oxidizer_output="${oxidizer_output}.exe"
    fi
    
    if [[ -f "$oxidizer_output" ]]; then
        cp "$oxidizer_output" "$output_dir/"
        log_success "Build complete: $output_dir/mcp-server"
    else
        log_error "PyOxidizer build output not found at: $oxidizer_output"
        exit 1
    fi
}

# Function to create a development bundle
create_dev_bundle() {
    local arch="$1"
    local os="$2"
    local target="$3"
    
    log_info "Creating development bundle for $target"
    
    local output_dir="dist/${target}"
    mkdir -p "$output_dir"
    
    # Create wrapper script
    cat > "$output_dir/mcp-server" << 'EOF'
#!/usr/bin/env python3
import sys
import os

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Set environment for MCP stdio mode
os.environ['MCP_STDIO_MODE'] = 'true'

# Import and run main
from main import main

if __name__ == "__main__":
    main()
EOF
    
    chmod +x "$output_dir/mcp-server"
    
    # Copy Python files
    cp -r *.py "$output_dir/" 2>/dev/null || true
    cp -r src "$output_dir/" 2>/dev/null || true
    
    log_success "Development bundle created: $output_dir/mcp-server"
}

# Main build function
main() {
    log_info "Starting build process"
    
    # Parse command line arguments
    BUILD_MODE="${1:-release}"
    USE_PYINSTALLER="${USE_PYINSTALLER:-true}"
    USE_PYOXIDIZER="${USE_PYOXIDIZER:-false}"
    SKIP_DEPS="${SKIP_DEPS:-false}"
    
    # Detect system information
    ARCH=$(detect_architecture)
    OS=$(detect_os)
    TARGET=$(get_rust_target "$ARCH" "$OS")
    
    log_info "System Information:"
    log_info "  Architecture: $ARCH"
    log_info "  Operating System: $OS"
    log_info "  Rust Target: $TARGET"
    log_info "  Build Mode: $BUILD_MODE"
    
    # Check dependencies
    check_dependencies
    
    # Setup virtual environment
    if [[ "$SKIP_DEPS" != "true" ]]; then
        setup_venv
        install_dependencies
    fi
    
    # Clean previous builds
    log_info "Cleaning previous builds"
    rm -rf dist build __pycache__ *.spec
    
    # Build based on selected method
    if [[ "$BUILD_MODE" == "dev" ]]; then
        create_dev_bundle "$ARCH" "$OS" "$TARGET"
    elif [[ "$USE_PYOXIDIZER" == "true" ]] && command_exists cargo; then
        build_with_pyoxidizer "$ARCH" "$OS" "$TARGET"
    elif [[ "$USE_PYINSTALLER" == "true" ]]; then
        build_with_pyinstaller "$ARCH" "$OS" "$TARGET"
    else
        log_error "No build method available"
        exit 1
    fi
    
    # Verify build output
    local output_file="dist/${TARGET}/mcp-server"
    if [[ "$OS" == "windows" ]]; then
        output_file="${output_file}.exe"
    fi
    
    if [[ -f "$output_file" ]]; then
        local size=$(du -h "$output_file" | cut -f1)
        log_success "Build completed successfully!"
        log_info "Output: $output_file ($size)"
        
        # Make executable on Unix systems
        if [[ "$OS" != "windows" ]]; then
            chmod +x "$output_file"
        fi
    else
        log_error "Build output not found: $output_file"
        exit 1
    fi
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi