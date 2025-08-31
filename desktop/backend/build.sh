#!/bin/bash

# Build script for MCP Server backend
# This creates a standalone Python executable for use with Tauri

set -e

echo "Building MCP Server backend..."

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Create virtual environment if it doesn't exist
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv"
fi

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Install requirements
echo "Installing requirements..."
pip install -q --upgrade pip
pip install -q pyinstaller

# Install project requirements
pip install -q -r "$PROJECT_ROOT/requirements.txt"

# Build with PyInstaller
echo "Building executable with PyInstaller..."
cd "$SCRIPT_DIR"
pyinstaller --clean --noconfirm pyinstaller.spec

# Copy to Tauri binaries directory
TAURI_BIN_DIR="$PROJECT_ROOT/desktop/frontend/src-tauri/binaries"
mkdir -p "$TAURI_BIN_DIR"

# Determine target triple
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    TARGET_TRIPLE="x86_64-unknown-linux-gnu"
    EXT=""
elif [[ "$OSTYPE" == "darwin"* ]]; then
    TARGET_TRIPLE="x86_64-apple-darwin"
    EXT=""
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    TARGET_TRIPLE="x86_64-pc-windows-msvc"
    EXT=".exe"
else
    echo "Unknown OS: $OSTYPE"
    exit 1
fi

# Copy the executable
if [ -f "$SCRIPT_DIR/dist/mcp-server$EXT" ]; then
    cp "$SCRIPT_DIR/dist/mcp-server$EXT" "$TAURI_BIN_DIR/mcp-server-$TARGET_TRIPLE$EXT"
    echo "✅ Built and copied mcp-server to: $TAURI_BIN_DIR/mcp-server-$TARGET_TRIPLE$EXT"
else
    echo "❌ Build failed: executable not found"
    exit 1
fi

echo "Build complete!"