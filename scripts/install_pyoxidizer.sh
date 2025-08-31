#!/bin/bash
"""
PyOxidizer Installation Script for MDMAI Project
Installs PyOxidizer on Linux, macOS, and Windows (WSL/Git Bash)

This script automates the installation of PyOxidizer, which is required
to build standalone executables of the MDMAI MCP Server.
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}INFO:${NC} $1"
}

print_success() {
    echo -e "${GREEN}SUCCESS:${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

print_error() {
    echo -e "${RED}ERROR:${NC} $1"
}

# Function to detect platform
detect_platform() {
    case "$(uname -s)" in
        Linux*)     PLATFORM=linux;;
        Darwin*)    PLATFORM=macos;;
        CYGWIN*|MINGW*|MSYS*) PLATFORM=windows;;
        *)          PLATFORM=unknown;;
    esac
    echo $PLATFORM
}

# Function to detect architecture
detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64)   ARCH=x86_64;;
        aarch64|arm64)  ARCH=aarch64;;
        i386|i686)      ARCH=i686;;
        *)              ARCH=unknown;;
    esac
    echo $ARCH
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Rust (required for PyOxidizer)
install_rust() {
    print_info "Checking for Rust installation..."
    
    if command_exists rustc && command_exists cargo; then
        RUST_VERSION=$(rustc --version)
        print_success "Rust is already installed: $RUST_VERSION"
        return 0
    fi
    
    print_info "Rust not found. Installing Rust..."
    
    if [ "$PLATFORM" = "windows" ]; then
        print_info "Please install Rust manually from https://rustup.rs/"
        print_info "Then run this script again."
        exit 1
    else
        # Install rustup
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        
        # Source the cargo environment
        source ~/.cargo/env
        
        print_success "Rust installed successfully"
    fi
}

# Function to install PyOxidizer from source
install_pyoxidizer_from_source() {
    print_info "Installing PyOxidizer from source..."
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    # Clone PyOxidizer repository
    print_info "Cloning PyOxidizer repository..."
    git clone https://github.com/indygreg/PyOxidizer.git
    cd PyOxidizer
    
    # Build and install
    print_info "Building PyOxidizer (this may take a while)..."
    cargo install --path pyoxidizer
    
    # Clean up
    cd "$HOME"
    rm -rf "$TEMP_DIR"
    
    print_success "PyOxidizer installed from source"
}

# Function to install PyOxidizer via cargo
install_pyoxidizer_cargo() {
    print_info "Installing PyOxidizer via cargo..."
    
    # Install PyOxidizer
    cargo install pyoxidizer
    
    print_success "PyOxidizer installed via cargo"
}

# Function to install PyOxidizer via pre-built binaries
install_pyoxidizer_binary() {
    print_info "Installing PyOxidizer from pre-built binaries..."
    
    # Determine download URL
    VERSION="0.24.0"  # Latest stable version
    
    case "$PLATFORM" in
        linux)
            if [ "$ARCH" = "x86_64" ]; then
                BINARY_URL="https://github.com/indygreg/PyOxidizer/releases/download/pyoxidizer%2F${VERSION}/pyoxidizer-${VERSION}-x86_64-unknown-linux-musl.tar.gz"
            else
                print_warning "No pre-built binary for $PLATFORM-$ARCH, falling back to cargo install"
                install_pyoxidizer_cargo
                return
            fi
            ;;
        macos)
            if [ "$ARCH" = "x86_64" ]; then
                BINARY_URL="https://github.com/indygreg/PyOxidizer/releases/download/pyoxidizer%2F${VERSION}/pyoxidizer-${VERSION}-x86_64-apple-darwin.tar.gz"
            elif [ "$ARCH" = "aarch64" ]; then
                BINARY_URL="https://github.com/indygreg/PyOxidizer/releases/download/pyoxidizer%2F${VERSION}/pyoxidizer-${VERSION}-aarch64-apple-darwin.tar.gz"
            else
                print_warning "No pre-built binary for $PLATFORM-$ARCH, falling back to cargo install"
                install_pyoxidizer_cargo
                return
            fi
            ;;
        windows)
            if [ "$ARCH" = "x86_64" ]; then
                BINARY_URL="https://github.com/indygreg/PyOxidizer/releases/download/pyoxidizer%2F${VERSION}/pyoxidizer-${VERSION}-x86_64-pc-windows-msvc.zip"
            else
                print_warning "No pre-built binary for $PLATFORM-$ARCH, falling back to cargo install"
                install_pyoxidizer_cargo
                return
            fi
            ;;
        *)
            print_error "Unsupported platform: $PLATFORM"
            exit 1
            ;;
    esac
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    # Download binary
    print_info "Downloading PyOxidizer binary from $BINARY_URL"
    
    if command_exists curl; then
        curl -L -o pyoxidizer.tar.gz "$BINARY_URL"
    elif command_exists wget; then
        wget -O pyoxidizer.tar.gz "$BINARY_URL"
    else
        print_error "Neither curl nor wget found. Please install one of them."
        exit 1
    fi
    
    # Extract binary
    if [ "$PLATFORM" = "windows" ]; then
        unzip pyoxidizer.zip
        BINARY_PATH="pyoxidizer.exe"
    else
        tar -xzf pyoxidizer.tar.gz
        BINARY_PATH="pyoxidizer"
    fi
    
    # Install binary to ~/.local/bin
    INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$INSTALL_DIR"
    cp "$BINARY_PATH" "$INSTALL_DIR/"
    chmod +x "$INSTALL_DIR/$BINARY_PATH"
    
    # Clean up
    cd "$HOME"
    rm -rf "$TEMP_DIR"
    
    print_success "PyOxidizer binary installed to $INSTALL_DIR"
    
    # Check if ~/.local/bin is in PATH
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        print_warning "~/.local/bin is not in your PATH"
        print_info "Add the following to your shell configuration file (~/.bashrc, ~/.zshrc, etc.):"
        echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
}

# Function to verify PyOxidizer installation
verify_installation() {
    print_info "Verifying PyOxidizer installation..."
    
    if command_exists pyoxidizer; then
        PYOXIDIZER_VERSION=$(pyoxidizer --version)
        print_success "PyOxidizer is installed and available: $PYOXIDIZER_VERSION"
        return 0
    else
        print_error "PyOxidizer is not available in PATH"
        return 1
    fi
}

# Function to install additional dependencies
install_dependencies() {
    print_info "Installing additional dependencies..."
    
    case "$PLATFORM" in
        linux)
            # Check if we can use apt, yum, or other package managers
            if command_exists apt-get; then
                print_info "Installing build dependencies via apt..."
                sudo apt-get update
                sudo apt-get install -y build-essential pkg-config libssl-dev
            elif command_exists yum; then
                print_info "Installing build dependencies via yum..."
                sudo yum groupinstall -y "Development Tools"
                sudo yum install -y openssl-devel
            elif command_exists dnf; then
                print_info "Installing build dependencies via dnf..."
                sudo dnf groupinstall -y "Development Tools"
                sudo dnf install -y openssl-devel
            else
                print_warning "Could not detect package manager. Please install build-essential and openssl development headers manually."
            fi
            ;;
        macos)
            # Check if Xcode command line tools are installed
            if ! command_exists cc; then
                print_info "Installing Xcode command line tools..."
                xcode-select --install
            fi
            ;;
        windows)
            print_info "On Windows, ensure you have Microsoft C++ Build Tools installed"
            print_info "Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
            ;;
    esac
}

# Main installation function
main() {
    echo "=========================================="
    echo "PyOxidizer Installation Script for MDMAI"
    echo "=========================================="
    
    # Detect platform and architecture
    PLATFORM=$(detect_platform)
    ARCH=$(detect_arch)
    
    print_info "Detected platform: $PLATFORM"
    print_info "Detected architecture: $ARCH"
    
    if [ "$PLATFORM" = "unknown" ] || [ "$ARCH" = "unknown" ]; then
        print_error "Unsupported platform or architecture: $PLATFORM-$ARCH"
        exit 1
    fi
    
    # Check if PyOxidizer is already installed
    if command_exists pyoxidizer; then
        CURRENT_VERSION=$(pyoxidizer --version)
        print_info "PyOxidizer is already installed: $CURRENT_VERSION"
        read -p "Do you want to reinstall? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Installation cancelled by user"
            exit 0
        fi
    fi
    
    # Install dependencies
    install_dependencies
    
    # Install Rust (required for PyOxidizer)
    install_rust
    
    # Install PyOxidizer
    print_info "Choose installation method:"
    echo "1) Pre-built binary (recommended, faster)"
    echo "2) Cargo install (requires compilation)"
    echo "3) Build from source (latest development version)"
    
    read -p "Enter choice (1-3) [1]: " -n 1 -r
    echo
    
    case "$REPLY" in
        2)
            install_pyoxidizer_cargo
            ;;
        3)
            install_pyoxidizer_from_source
            ;;
        *)
            install_pyoxidizer_binary
            ;;
    esac
    
    # Verify installation
    if verify_installation; then
        print_success "PyOxidizer installation completed successfully!"
        echo
        print_info "Next steps:"
        echo "1. Navigate to your MDMAI project directory"
        echo "2. Run: python scripts/build_pyoxidizer.py --platform linux"
        echo "3. Or run: python scripts/build_pyoxidizer.py --all (for all platforms)"
    else
        print_error "Installation verification failed"
        exit 1
    fi
}

# Run main function
main "$@"