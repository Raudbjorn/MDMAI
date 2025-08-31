#!/bin/sh
#
# Build Script for TTRPG Assistant MCP Server Backend
# POSIX-compliant version with comprehensive error handling and platform support
#
# Supported platforms:
#   - macOS: x86_64 (Intel), aarch64 (Apple Silicon M1/M2/M3)
#   - Linux: x86_64, aarch64  
#   - Windows: x86_64 (via WSL or Git Bash)
#

set -eu

# Script metadata
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly LOG_FILE="${SCRIPT_DIR}/build.log"

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m' # No Color

# Global variables
BUILD_START_TIME=""
TEMP_FILES_LIST=""

# Add temporary file to cleanup list
add_temp_file() {
    TEMP_FILES_LIST="$TEMP_FILES_LIST $1"
}

# Cleanup function for trap
cleanup() {
    exit_code=$?
    
    if [ -n "$TEMP_FILES_LIST" ]; then
        log_debug "Cleaning up temporary files..."
        for temp_file in $TEMP_FILES_LIST; do
            [ -f "$temp_file" ] && rm -f "$temp_file"
        done
    fi
    
    if [ $exit_code -ne 0 ]; then
        log_error "Build failed with exit code $exit_code"
        log_info "Check $LOG_FILE for detailed logs"
    fi
    
    exit $exit_code
}

# Set up trap for cleanup
trap cleanup EXIT INT TERM

# Enhanced logging functions with timestamps and file logging
log_with_level() {
    local level="$1"
    local color="$2"
    local message="$3"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Console output with colors
    echo -e "${color}[${level}]${NC} ${message}"
    
    # File output without colors
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

log_info() {
    log_with_level "INFO" "$BLUE" "$1"
}

log_success() {
    log_with_level "SUCCESS" "$GREEN" "$1"
}

log_warning() {
    log_with_level "WARNING" "$YELLOW" "$1"
}

log_error() {
    log_with_level "ERROR" "$RED" "$1" >&2
}

log_debug() {
    if [ "${DEBUG:-false}" = "true" ]; then
        log_with_level "DEBUG" "$CYAN" "$1"
    fi
}

log_step() {
    log_with_level "STEP" "$BOLD$BLUE" "$1"
}

# Progress tracking
show_progress() {
    local current=$1
    local total=$2
    local desc="$3"
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((current * width / total))
    local remaining=$((width - completed))
    
    printf "\r${CYAN}[%-${width}s] %d%% %s${NC}" \
           "$(printf '%*s' "$completed" | tr ' ' '=')$(printf '%*s' "$remaining")" \
           "$percentage" "$desc"
    
    if [ $current -eq $total ]; then
        echo
    fi
}

# Enhanced architecture detection with validation
detect_architecture() {
    arch=$(uname -m)
    log_debug "Raw architecture: $arch"
    
    case "$arch" in
        x86_64|amd64) echo "x86_64" ;;
        arm64|aarch64) echo "aarch64" ;;  # ARM64 is reported as 'arm64' on macOS but needs 'aarch64' for Rust
        armv7l|armv7) echo "armv7" ;;
        i386|i686) echo "i686" ;;
        *) log_error "Unsupported architecture: $arch"
           log_info "Supported architectures: x86_64, aarch64, armv7, i686"
           return 1 ;;
    esac
}

# Enhanced OS detection with version information
detect_os() {
    kernel_name=$(uname -s)
    log_debug "Kernel name: $kernel_name"
    
    case "$kernel_name" in
        Darwin)
            macos_version=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
            log_debug "macOS version: $macos_version"
            echo "darwin" ;;
        Linux)
            if [ -f /etc/os-release ]; then
                distro=$(. /etc/os-release; echo "$NAME $VERSION_ID")
                log_debug "Linux distribution: $distro"
            fi
            echo "linux" ;;
        MINGW*|MSYS*|CYGWIN*|Windows_NT)
            log_debug "Windows environment detected: $kernel_name"
            echo "windows" ;;
        *) log_error "Unsupported operating system: $kernel_name"
           log_info "Supported systems: macOS (Darwin), Linux, Windows"
           return 1 ;;
    esac
}

# Validate system compatibility
validate_system() {
    log_step "Validating system compatibility..."
    
    arch=$(detect_architecture) || return 1
    os=$(detect_os) || return 1
    
    log_info "Detected system: $os/$arch"
    
    # Additional OS-specific checks
    case "$os" in
        darwin) command -v sw_vers >/dev/null 2>&1 || log_warning "Cannot determine macOS version" ;;
        linux) [ -f /etc/os-release ] || log_warning "Cannot determine Linux distribution" ;;
        windows) log_info "Windows environment detected - ensure dependencies are available" ;;
    esac
}

# Enhanced Rust target detection
get_rust_target() {
    arch="$1"
    os="$2"
    
    log_debug "Determining Rust target for $os/$arch"
    
    case "$os" in
        darwin) echo "${arch}-apple-darwin" ;;
        linux)
            # Detect libc variant
            libc_variant="gnu"
            if command -v ldd >/dev/null 2>&1 && ldd --version 2>&1 | grep -qi musl; then
                libc_variant="musl"
                log_debug "Detected musl libc"
            else
                log_debug "Detected glibc (or ldd not found)"
            fi
            echo "${arch}-unknown-linux-${libc_variant}" ;;
        windows) echo "${arch}-pc-windows-msvc" ;;  # Prefer MSVC over GNU
        *) log_error "Cannot determine Rust target for OS: $os"; return 1 ;;
    esac
}

# Enhanced command existence check
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get command version with fallback
get_command_version() {
    cmd="$1"
    version_flag="${2:---version}"
    
    if command_exists "$cmd"; then
        "$cmd" "$version_flag" 2>/dev/null | head -n1 || echo "unknown"
    else
        echo "not found"
    fi
}

# Check version compatibility (major.minor format)
check_version_compat() {
    version="$1"
    min_major="$2"
    min_minor="${3:-0}"
    
    major=$(echo "$version" | cut -d. -f1)
    minor=$(echo "$version" | cut -d. -f2)
    
    [ "$major" -gt "$min_major" ] || [ "$major" -eq "$min_major" -a "$minor" -ge "$min_minor" ]
}

# Check Python and pip availability
check_python_deps() {
    # Find Python command
    for cmd in python3 python; do
        if command_exists "$cmd"; then
            PYTHON_CMD="$cmd"
            break
        fi
    done
    
    [ -z "$PYTHON_CMD" ] && return 1
    
    python_version=$(get_command_version "$PYTHON_CMD" --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    log_info "Python version: $python_version ($PYTHON_CMD)"
    
    # Check version compatibility
    if [ -n "$python_version" ] && ! check_version_compat "$python_version" 3 8; then
        log_warning "Python $python_version may be too old (recommend 3.8+)"
    fi
    
    # Find pip command
    for cmd in pip3 pip; do
        if command_exists "$cmd"; then
            PIP_CMD="$cmd"
            break
        fi
    done
    
    [ -z "$PIP_CMD" ] && return 1
    
    pip_version=$(get_command_version "$PIP_CMD" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    log_info "Pip version: $pip_version ($PIP_CMD)"
    return 0
}

# Check Rust dependencies for PyOxidizer
check_rust_deps() {
    [ "${USE_PYOXIDIZER:-false}" != "true" ] && return 0
    
    if command_exists cargo; then
        cargo_version=$(get_command_version cargo | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        rust_version=$(get_command_version rustc | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        log_info "Cargo version: $cargo_version"
        log_info "Rust version: $rust_version"
    else
        log_error "rust/cargo required for PyOxidizer"
        return 1
    fi
}

# Enhanced dependency checking with modular approach
check_dependencies() {
    log_step "Checking build dependencies..."
    
    PYTHON_CMD=""
    PIP_CMD=""
    
    # Check core dependencies
    if ! check_python_deps; then
        log_error "Missing critical dependencies: python3 and pip"
        log_info "Please install Python 3.8+ and pip, then try again"
        return 1
    fi
    
    # Check optional dependencies
    check_rust_deps || return 1
    
    # Check PyInstaller if needed
    if [ "${USE_PYINSTALLER:-true}" = "true" ] && [ -n "$PYTHON_CMD" ]; then
        if ! "$PYTHON_CMD" -c "import PyInstaller" 2>/dev/null; then
            log_debug "PyInstaller not found, will install during setup"
        else
            pyinstaller_version=$("$PYTHON_CMD" -c "import PyInstaller; print(PyInstaller.__version__)" 2>/dev/null || echo "unknown")
            log_info "PyInstaller version: $pyinstaller_version"
        fi
    fi
    
    log_success "All required dependencies are available"
}

# Enhanced virtual environment setup
setup_venv() {
    venv_path="${1:-venv}"
    log_step "Setting up Python virtual environment..."
    
    if [ ! -d "$venv_path" ]; then
        log_info "Creating virtual environment at: $venv_path"
        if ! "$PYTHON_CMD" -m venv "$venv_path"; then
            log_error "Failed to create virtual environment"
            return 1
        fi
        log_success "Virtual environment created"
    else
        log_info "Using existing virtual environment: $venv_path"
    fi
    
    # Find and activate virtual environment
    if [ -f "$venv_path/bin/activate" ]; then
        activate_script="$venv_path/bin/activate"
    elif [ -f "$venv_path/Scripts/activate" ]; then
        activate_script="$venv_path/Scripts/activate"
    else
        log_error "Cannot find virtual environment activation script in $venv_path"
        return 1
    fi
    
    # shellcheck source=/dev/null
    . "$activate_script"
    
    log_success "Virtual environment activated: $(which python)"
    
    # Verify activation
    case "${VIRTUAL_ENV:-}" in
        *"$venv_path"*) ;;
        *) log_warning "Virtual environment may not be properly activated" ;;
    esac
}

# Count installation steps
count_install_steps() {
    steps=0
    [ "${SKIP_PIP_UPGRADE:-false}" != "true" ] && steps=$((steps + 1))
    [ -f "requirements.txt" ] || [ -f "pyproject.toml" ] && steps=$((steps + 1))
    [ "${USE_PYINSTALLER:-true}" = "true" ] && steps=$((steps + 1))
    [ "${USE_PYOXIDIZER:-false}" = "true" ] && steps=$((steps + 1))
    echo $steps
}

# Install project dependencies
install_project_deps() {
    if [ -f "requirements.txt" ]; then
        log_info "Installing from requirements.txt"
        pip install -r requirements.txt --quiet
    elif [ -f "pyproject.toml" ]; then
        log_info "Installing from pyproject.toml"
        pip install -e . --quiet
    else
        log_warning "No requirements.txt or pyproject.toml found"
    fi
}

# Enhanced dependency installation with modular approach
install_dependencies() {
    log_step "Installing Python dependencies..."
    
    steps=$(count_install_steps)
    current_step=0
    
    # Upgrade pip
    if [ "${SKIP_PIP_UPGRADE:-false}" != "true" ]; then
        current_step=$((current_step + 1))
        show_progress $current_step $steps "Upgrading pip..."
        if pip install --upgrade pip --quiet; then
            log_success "Pip upgraded"
        else
            log_warning "Failed to upgrade pip, continuing..."
        fi
    fi
    
    # Install project dependencies
    if [ -f "requirements.txt" ] || [ -f "pyproject.toml" ]; then
        current_step=$((current_step + 1))
        show_progress $current_step $steps "Installing project dependencies..."
        install_project_deps || {
            log_error "Failed to install project dependencies"
            return 1
        }
    fi
    
    # Install build tools
    if [ "${USE_PYINSTALLER:-true}" = "true" ]; then
        current_step=$((current_step + 1))
        show_progress $current_step $steps "Installing PyInstaller..."
        pip install pyinstaller --quiet || {
            log_error "Failed to install PyInstaller"
            return 1
        }
        log_success "PyInstaller installed"
    fi
    
    if [ "${USE_PYOXIDIZER:-false}" = "true" ]; then
        current_step=$((current_step + 1))
        show_progress $current_step $steps "Installing PyOxidizer..."
        pip install pyoxidizer --quiet || {
            log_error "Failed to install PyOxidizer"
            return 1
        }
        log_success "PyOxidizer installed"
    fi
    
    log_success "All dependencies installed successfully"
}

# Function to build with PyInstaller
build_with_pyinstaller() {
    local arch="$1"
    local os="$2"
    local target="$3"
    
    log_info "Building with PyInstaller for $target"
    
    # Create PyInstaller spec if it doesn't exist
    if [ ! -f "pyinstaller.spec" ]; then
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
    
    if [ -f "dist/mcp-server-${arch}-${os}" ]; then
        mv "dist/mcp-server-${arch}-${os}" "$output_dir/mcp-server"
    elif [ -f "dist/mcp-server-${arch}-${os}.exe" ]; then
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
    if [ ! -f "pyoxidizer.toml" ]; then
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
    if [ "$os" = "windows" ]; then
        oxidizer_output="${oxidizer_output}.exe"
    fi
    
    if [ -f "$oxidizer_output" ]; then
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

# Display usage information
show_usage() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS] [BUILD_MODE]

Enhanced build script for TTRPG Assistant MCP Server Backend

BUILD_MODE:
    release      Build optimized production binary (default)
    dev          Create development bundle with source files
    debug        Build with debug information

OPTIONS:
    -h, --help              Show this help message
    -v, --verbose           Enable verbose logging
    --debug                 Enable debug output
    --skip-deps            Skip dependency installation
    --skip-venv            Skip virtual environment setup
    --use-pyoxidizer       Use PyOxidizer for building (requires Rust)
    --use-pyinstaller      Use PyInstaller for building (default)
    --clean-all            Clean all build artifacts before building

ENVIRONMENT VARIABLES:
    USE_PYINSTALLER=true|false    Choose PyInstaller (default: true)
    USE_PYOXIDIZER=true|false     Choose PyOxidizer (default: false)
    SKIP_DEPS=true|false          Skip dependency installation
    SKIP_PIP_UPGRADE=true|false   Skip pip upgrade
    DEBUG=true|false              Enable debug output

Examples:
    $SCRIPT_NAME                    # Build with default settings
    $SCRIPT_NAME dev                # Create development bundle
    $SCRIPT_NAME --debug release    # Build with debug output
    USE_PYOXIDIZER=true $SCRIPT_NAME # Build with PyOxidizer

EOF
}

# POSIX-compliant argument parsing
parse_arguments() {
    while [ $# -gt 0 ]; do
        case $1 in
            -h|--help) show_usage; exit 0 ;;
            -v|--verbose) VERBOSE=true ;;
            --debug) DEBUG=true ;;
            --skip-deps) SKIP_DEPS_ARG=true ;;
            --skip-venv) SKIP_VENV_ARG=true ;;
            --use-pyoxidizer) USE_PYOXIDIZER=true; USE_PYINSTALLER=false ;;
            --use-pyinstaller) USE_PYINSTALLER=true; USE_PYOXIDIZER=false ;;
            --clean-all) CLEAN_ALL=true ;;
            release|dev|debug) BUILD_MODE="$1" ;;
            *) log_error "Unknown argument: $1"; show_usage; exit 1 ;;
        esac
        shift
    done
}

# Clean build artifacts
clean_artifacts() {
    if [ "${CLEAN_ALL:-false}" = "true" ]; then
        log_step "Cleaning all build artifacts..."
        rm -rf dist build __pycache__ *.spec venv .pytest_cache
        log_success "Build artifacts cleaned"
    else
        log_step "Cleaning previous builds..."
        rm -rf dist build __pycache__ *.spec
    fi
}

# Enhanced main function with better structure
main() {
    BUILD_START_TIME=$(date '+%s')
    
    # Initialize defaults
    BUILD_MODE="release"
    CLEAN_ALL=false
    SKIP_DEPS_ARG=false
    SKIP_VENV_ARG=false
    
    # Initialize log file
    echo "=== Build started at $(date) ===" > "$LOG_FILE"
    
    # Parse arguments
    parse_arguments "$@"
    
    # Set defaults and exports
    USE_PYINSTALLER="${USE_PYINSTALLER:-true}"
    USE_PYOXIDIZER="${USE_PYOXIDIZER:-false}"
    SKIP_DEPS="${SKIP_DEPS_ARG:-false}"
    
    export USE_PYINSTALLER USE_PYOXIDIZER DEBUG VERBOSE
    
    log_info "${BOLD}Starting build process${NC}"
    log_info "Build mode: $BUILD_MODE"
    log_info "Log file: $LOG_FILE"
    
    # System validation and detection
    validate_system || exit 1
    
    arch=$(detect_architecture) || exit 1
    os=$(detect_os) || exit 1
    target=$(get_rust_target "$arch" "$os") || exit 1
    
    log_info "${BOLD}System Information:${NC}"
    log_info "  Architecture: $arch"
    log_info "  Operating System: $os"
    log_info "  Rust Target: $target"
    
    # Dependency checking and setup
    check_dependencies || exit 1
    
    if [ "${SKIP_VENV_ARG:-false}" != "true" ] && [ "$SKIP_DEPS" != "true" ]; then
        setup_venv || exit 1
        install_dependencies || exit 1
    fi
    
    # Clean artifacts
    clean_artifacts
    
    # Build execution
    output_file="dist/${target}/mcp-server"
    [ "$os" = "windows" ] && output_file="${output_file}.exe"
    
    case "$BUILD_MODE" in
        dev) create_dev_bundle "$arch" "$os" "$target" || exit 1 ;;
        release|debug)
            if [ "$USE_PYOXIDIZER" = "true" ] && command_exists cargo; then
                build_with_pyoxidizer "$arch" "$os" "$target" || exit 1
            elif [ "$USE_PYINSTALLER" = "true" ]; then
                build_with_pyinstaller "$arch" "$os" "$target" || exit 1
            else
                log_error "No suitable build method available"
                exit 1
            fi ;;
        *) log_error "Unknown build mode: $BUILD_MODE"; exit 1 ;;
    esac
    
    # Result verification and reporting
    if [ -f "$output_file" ]; then
        size=$(du -h "$output_file" | cut -f1)
        elapsed=$(($(date '+%s') - BUILD_START_TIME))
        
        # Make executable on Unix systems
        [ "$os" != "windows" ] && chmod +x "$output_file"
        
        log_success "${BOLD}Build completed successfully!${NC}"
        log_info "Output: $output_file ($size)"
        log_info "Build time: ${elapsed}s"
        
        echo "=== Build completed successfully at $(date) ===" >> "$LOG_FILE"
    else
        log_error "Build output not found: $output_file"
        exit 1
    fi
}

# Run main function if script is executed directly
if [ "$(basename "$0")" = "build.sh" ]; then
    main "$@"
fi